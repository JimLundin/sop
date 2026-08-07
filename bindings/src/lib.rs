//! Python bindings for the sop core.
//!
//! The implementation lives entirely in the `sop` crate; this module is
//! marshalling. Parsing builds Python objects directly from the tape, and
//! writing feeds Python objects into the core's [`sop::Builder`], so the text
//! form -- escaping, layout, all of it -- is produced by the one writer.
//!
//! Value mapping:
//!
//!     sop object   <->  dict[str, Value]
//!     sop array    <->  list[Value]
//!     sop string   <->  str
//!     sop number   <->  int | float
//!     sop symbol   <->  True | False | None | Symbol(name)
//!     sop tagged   <->  Tagged(tag, value)

use std::collections::HashMap;

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

create_exception!(_core, SopError, PyValueError);

fn sop_error(message: String, line: usize, column: usize) -> PyErr {
    Python::attach(|py| {
        let text = if line == 0 {
            message.clone()
        } else {
            format!("{line}:{column}: {message}")
        };
        let err = SopError::new_err(text);
        let value = err.value(py);
        let _ = value.setattr("message", message);
        let _ = value.setattr("line", line);
        let _ = value.setattr("column", column);
        err
    })
}

fn raise(e: sop::Error) -> PyErr {
    sop_error(e.message, e.line, e.column)
}

fn ser_error(message: String) -> PyErr {
    sop_error(message, 0, 0)
}

/// CPython's recursion guard, entered once per `emit` frame and left on drop.
struct Recursion;

impl Recursion {
    fn enter(py: Python<'_>, task: &'static std::ffi::CStr) -> PyResult<Recursion> {
        match unsafe { pyo3::ffi::Py_EnterRecursiveCall(task.as_ptr()) } {
            0 => Ok(Recursion),
            _ => Err(PyErr::fetch(py)),
        }
    }
}

impl Drop for Recursion {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::Py_LeaveRecursiveCall() }
    }
}

// ---------------------------------------------------------------------------
// Value types
// ---------------------------------------------------------------------------

/// A bare identifier used as a scalar. A symbol is never equal to a string,
/// which is why this is a distinct type and not a `str` subclass.
#[pyclass(frozen, eq, hash, skip_from_py_object, module = "sop._core")]
#[derive(PartialEq, Eq, Hash, Clone)]
pub struct Symbol {
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl Symbol {
    #[new]
    fn new(name: String) -> PyResult<Self> {
        // The SDK spells these three as Python values, so a Symbol holding
        // one would be a second spelling of a value that already has one.
        if matches!(name.as_str(), "true" | "false" | "null") {
            return Err(PyValueError::new_err(format!(
                "`{name}` is spelled with the Python value, not Symbol({name:?})"
            )));
        }
        if !sop::is_identifier(&name) {
            return Err(PyValueError::new_err(format!(
                "{name:?} is not an identifier, so it cannot be a symbol"
            )));
        }
        Ok(Symbol { name })
    }

    fn __repr__(&self) -> String {
        format!("Symbol({:?})", self.name)
    }

    fn __str__(&self) -> &str {
        &self.name
    }

    /// Makes `copy`, `deepcopy` and `pickle` work on a native type.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String,))> {
        Ok((py.get_type::<Symbol>().into_any().unbind(), (self.name.clone(),)))
    }
}

/// A tag applied to a payload. Tags are constructive: a tagged value is never
/// equal to the value it wraps.
#[pyclass(module = "sop._core")]
pub struct Tagged {
    #[pyo3(get)]
    tag: String,
    #[pyo3(get)]
    value: Py<PyAny>,
}

#[pymethods]
impl Tagged {
    #[new]
    fn new(tag: String, value: Py<PyAny>) -> PyResult<Self> {
        if !sop::is_identifier(&tag) {
            return Err(PyValueError::new_err(format!(
                "{tag:?} is not an identifier, so it cannot be a tag"
            )));
        }
        Ok(Tagged { tag, value })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Tagged({:?}, {})", self.tag, self.value.bind(py).repr()?))
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.cast::<Tagged>() else {
            // A tagged value is never equal to its payload, or to anything
            // else. Beyond that, comparison is Python's business.
            return Ok(false);
        };
        let other = other.borrow();
        Ok(self.tag == other.tag && self.value.bind(py).eq(other.value.bind(py))?)
    }

    /// Makes `copy`, `deepcopy` and `pickle` work on a native type.
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (String, Py<PyAny>))> {
        Ok((
            py.get_type::<Tagged>().into_any().unbind(),
            (self.tag.clone(), self.value.clone_ref(py)),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tape -> Python
// ---------------------------------------------------------------------------

/// The core treats `true`, `false` and `null` as ordinary symbols. Choosing
/// host types for them is the SDK's call, and this is where it is made.
/// `keys` shares one Python string per distinct object key, as the stdlib
/// json scanner's memo does: a key repeated across ten thousand records
/// costs one allocation and one hash, not ten thousand.
fn build<'a>(
    py: Python<'_>,
    node: sop::Ref<'a>,
    keys: &mut HashMap<&'a str, Py<PyString>>,
) -> PyResult<Py<PyAny>> {
    // The core reads any depth; building the Python value is recursive, so
    // past the interpreter's own limit this raises RecursionError -- the
    // interpreter's bound, not one of ours.
    let _guard = Recursion::enter(py, c" while reading sop")?;
    Ok(match node.kind() {
        sop::Kind::Number => match node.as_i64() {
            Some(i) => i.into_pyobject(py)?.into_any().unbind(),
            None => PyFloat::new(py, node.as_f64().unwrap()).into_any().unbind(),
        },
        sop::Kind::String => PyString::new(py, node.as_str().unwrap()).into_any().unbind(),
        sop::Kind::Symbol => match node.symbol().unwrap() {
            "true" => PyBool::new(py, true).to_owned().into_any().unbind(),
            "false" => PyBool::new(py, false).to_owned().into_any().unbind(),
            "null" => py.None(),
            name => Py::new(py, Symbol { name: name.to_owned() })?.into_any(),
        },
        sop::Kind::Tagged => Py::new(
            py,
            Tagged {
                tag: node.tag().unwrap().to_owned(),
                value: build(py, node.payload().unwrap(), keys)?,
            },
        )?
        .into_any(),
        sop::Kind::Array => {
            let items = PyList::empty(py);
            for item in node.array() {
                items.append(build(py, item, keys)?)?;
            }
            items.into_any().unbind()
        }
        sop::Kind::Object => {
            // Duplicates are accepted and the last wins, which is what
            // assigning into a dict does, keeping the first position.
            let map = PyDict::new(py);
            for (key, item) in node.object() {
                let shared = match keys.get(key) {
                    Some(k) => k.clone_ref(py),
                    None => {
                        let new = PyString::new(py, key).unbind();
                        keys.insert(key, new.clone_ref(py));
                        new
                    }
                };
                map.set_item(shared, build(py, item, keys)?)?;
            }
            map.into_any().unbind()
        }
    })
}

// ---------------------------------------------------------------------------
// Python -> text
// ---------------------------------------------------------------------------

enum Kind<'py> {
    Null,
    Bool(bool),
    Int(Bound<'py, PyInt>),
    Float(f64),
    Str(String),
    Symbol(String),
    Tagged(String, Bound<'py, PyAny>),
    Array(Bound<'py, PyList>),
    Object(Bound<'py, PyDict>),
}

/// Classify a Python object as a sop value.
///
/// The builtin cases are matched on the *exact* type. A subclass may carry a
/// tag — `class Iban(str)` is written `iban "DE89"` — and treating it as a
/// plain string here would drop the tag silently. Anything this refuses goes
/// to the shape layer, which knows about tags.
///
/// `bool` is tested before `int`: in Python `bool` subclasses `int`, so
/// `isinstance(True, int)` holds and the wrong order would spell `true` as `1`.
fn classify<'py>(value: &Bound<'py, PyAny>) -> PyResult<Kind<'py>> {
    if value.is_none() {
        return Ok(Kind::Null);
    }
    if let Ok(b) = value.cast_exact::<PyBool>() {
        return Ok(Kind::Bool(b.is_true()));
    }
    if let Ok(symbol) = value.cast::<Symbol>() {
        return Ok(Kind::Symbol(symbol.get().name.clone()));
    }
    if let Ok(tagged) = value.cast::<Tagged>() {
        let borrowed = tagged.borrow();
        return Ok(Kind::Tagged(
            borrowed.tag.clone(),
            borrowed.value.bind(value.py()).clone(),
        ));
    }
    if let Ok(s) = value.cast_exact::<PyString>() {
        // A Python `str` may hold a lone surrogate; sop text cannot, and the
        // parser rejects the escape for the same reason. Refuse it as a
        // SopError rather than leaking the codec's UnicodeEncodeError.
        let text = s.to_cow().map_err(|_| {
            ser_error("string contains a lone surrogate, which sop text cannot hold".into())
        })?;
        return Ok(Kind::Str(text.into_owned()));
    }
    if let Ok(i) = value.cast_exact::<PyInt>() {
        return Ok(Kind::Int(i.clone()));
    }
    if let Ok(f) = value.cast_exact::<PyFloat>() {
        return Ok(Kind::Float(f.value()));
    }
    if let Ok(items) = value.cast_exact::<PyList>() {
        return Ok(Kind::Array(items.clone()));
    }
    if let Ok(map) = value.cast_exact::<PyDict>() {
        return Ok(Kind::Object(map.clone()));
    }
    if let Ok(items) = value.cast_exact::<PyTuple>() {
        return Ok(Kind::Array(PyList::new(value.py(), items.iter())?));
    }
    Err(ser_error(format!("not a sop value: {}", value.repr()?)))
}

/// Feed a Python value into the core's builder; the text is then the core
/// writer's business.
fn emit(value: &Bound<'_, PyAny>, doc: &mut sop::Builder) -> PyResult<()> {
    // A Python value can nest arbitrarily, or cyclically, and overflowing the
    // stack here is an abort the caller cannot catch. Borrowing CPython's own
    // recursion guard -- as the stdlib json encoder does -- turns that into a
    // RecursionError at the interpreter's limit, with no limit of our own.
    let _guard = Recursion::enter(value.py(), c" while writing sop")?;
    match classify(value)? {
        Kind::Null => doc.symbol("null"),
        Kind::Bool(true) => doc.symbol("true"),
        Kind::Bool(false) => doc.symbol("false"),
        // Symbol and Tagged validate their names on construction, so the
        // builder cannot be handed one that does not round-trip.
        Kind::Symbol(name) => doc.symbol(&name),
        Kind::Str(s) => doc.string(&s),
        // Python integers are unbounded; the core's numeric domain is i64 or
        // f64. A larger one would not read back to an equal value, so it is
        // refused for the same reason the parser refuses a literal that
        // overflows to infinity. The remedy is the same either way: carry
        // exact values as a tagged string.
        Kind::Int(i) => match i.extract::<i64>() {
            Ok(v) => doc.int(v),
            Err(_) => {
                return Err(ser_error(format!(
                    "{} is out of range for sop's numeric domain; use a tagged string",
                    i.str()?.to_cow()?
                )));
            }
        },
        Kind::Float(f) if f.is_finite() => doc.float(f),
        Kind::Float(f) => return Err(ser_error(format!("{f} has no sop representation"))),
        Kind::Tagged(tag, payload) => {
            doc.tag(&tag);
            emit(&payload, doc)?;
        }
        Kind::Array(items) => {
            doc.begin_array();
            for item in items.iter() {
                emit(&item, doc)?;
            }
            doc.end_array();
        }
        Kind::Object(map) => {
            doc.begin_object();
            for (key, item) in map.iter() {
                let key = key
                    .cast::<PyString>()
                    .map_err(|_| ser_error("object keys must be strings".to_string()))?
                    .to_cow()
                    .map_err(|_| {
                        ser_error("object key contains a lone surrogate".to_string())
                    })?;
                doc.key(&key);
                emit(&item, doc)?;
            }
            doc.end_object();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pyfunction]
fn loads(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    let document = sop::Document::parse(text).map_err(raise)?;
    build(py, document.root(), &mut HashMap::new())
}

#[pyfunction]
#[pyo3(signature = (value, indent = None))]
fn dumps(value: &Bound<'_, PyAny>, indent: Option<usize>) -> PyResult<String> {
    let mut doc = sop::Builder::new();
    emit(value, &mut doc)?;
    let document = doc.finish();
    Ok(match indent.unwrap_or(0) {
        0 => document.to_string(),
        indent => document.to_string_pretty(indent),
    })
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("SopError", module.py().get_type::<SopError>())?;
    module.add_class::<Symbol>()?;
    module.add_class::<Tagged>()?;
    module.add_function(wrap_pyfunction!(loads, module)?)?;
    module.add_function(wrap_pyfunction!(dumps, module)?)?;
    Ok(())
}
