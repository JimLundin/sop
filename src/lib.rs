//! sop — the sop interchange format, as the extension module behind the
//! `sop` Python package.
//!
//! One implementation, one pass each way. `loads` builds Python objects
//! directly as it parses ([`parse`]); `dumps` spells Python objects directly
//! as text ([`write`]); [`text`] holds the lexical facts they share. There is
//! no intermediate document tree.
//!
//! Value mapping. Reading produces immutable values only; writing also
//! accepts their mutable counterparts:
//!
//! ```text
//! sop object   <->  frozendict[str, Value]   (dict is written too)
//! sop array    <->  tuple[Value, ...]        (list is written too)
//! sop string   <->  str
//! sop number   <->  int | float
//! sop symbol   <->  True | False | None | Symbol(name)
//! sop tagged   <->  Tagged(tag, value)
//! ```
//!
//! Anything else -- a dataclass, an enum, a set -- is spelled by the
//! `convert` hook the SDK passes to `dumps`: one Python call per object the
//! writer cannot classify, and the traversal continues in place, so the
//! graph is walked exactly once.

mod parse;
mod text;
mod write;

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::OnceLock;

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

create_exception!(_core, SopError, PyValueError);

/// The `builtins.frozendict` type, taken once at module import.
pub(crate) static FROZENDICT: OnceLock<Py<PyAny>> = OnceLock::new();

pub(crate) fn sop_error(message: String, line: usize, column: usize) -> PyErr {
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

pub(crate) fn ser_error(message: String) -> PyErr {
    sop_error(message, 0, 0)
}

/// CPython's recursion guard, entered once per parse or emit frame and left
/// on drop.
pub(crate) struct Recursion;

impl Recursion {
    pub(crate) fn enter(py: Python<'_>, task: &'static std::ffi::CStr) -> PyResult<Recursion> {
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
    pub(crate) name: String,
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
        if !text::is_identifier(&name) {
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
///
/// Frozen, like `Symbol`: the native types are immutable. Equality and
/// hashing behave as a frozen dataclass's would — equal by `(tag, value)`,
/// hashable exactly when the payload is.
///
/// The payload is never `None`, a bool or a `Symbol`: those are bare symbols
/// on the wire, and a tag cannot be applied to a bare symbol, so such a
/// value could never be written or read back.
#[pyclass(frozen, module = "sop._core")]
pub struct Tagged {
    #[pyo3(get)]
    pub(crate) tag: String,
    #[pyo3(get)]
    pub(crate) value: Py<PyAny>,
}

#[pymethods]
impl Tagged {
    #[new]
    fn new(tag: String, value: Bound<'_, PyAny>) -> PyResult<Self> {
        if !text::is_identifier(&tag) {
            return Err(PyValueError::new_err(format!(
                "{tag:?} is not an identifier, so it cannot be a tag"
            )));
        }
        let symbolish = if value.is_none() {
            Some("None, which is spelled `null`")
        } else if value.is_instance_of::<PyBool>() {
            Some("a bool, which is spelled `true` or `false`")
        } else if value.is_instance_of::<Symbol>() {
            Some("a Symbol")
        } else {
            None
        };
        if let Some(what) = symbolish {
            return Err(PyValueError::new_err(format!(
                "a tag cannot be applied to a bare symbol, so Tagged cannot hold {what}"
            )));
        }
        Ok(Tagged { tag, value: value.unbind() })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Tagged({:?}, {})", self.tag, self.value.bind(py).repr()?))
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Tagged>() else {
            // A tagged value is never equal to its payload. `NotImplemented`
            // rather than `False`, so the other operand gets its say;
            // comparison beyond these types is Python's business.
            return Ok(py.NotImplemented());
        };
        let other = other.get();
        let equal = self.tag == other.tag && self.value.bind(py).eq(other.value.bind(py))?;
        Ok(PyBool::new(py, equal).to_owned().into_any().unbind())
    }

    /// A frozen dataclass's hash, over `(tag, value)`. Delegating to the
    /// payload's own hash keeps the contract with `__eq__`: equal values hash
    /// equal, and an unhashable payload makes the whole value unhashable
    /// rather than quietly hashable by identity.
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.tag.hash(&mut hasher);
        self.value.bind(py).hash()?.hash(&mut hasher);
        Ok(hasher.finish() as isize)
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
// Module
// ---------------------------------------------------------------------------

/// Read sop text into the untyped value mapping: objects become frozendicts,
/// arrays tuples, symbols `True`/`False`/`None`/`Symbol`, tagged values
/// `Tagged`. Raises `SopError`, carrying `message`, `line` and `column`.
#[pyfunction]
fn loads(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    parse::Parser::new(py, text).parse_document()
}

/// Write a value as sop text. `indent` selects the pretty writer; `convert`
/// is called once per object the writer cannot classify and must return a
/// sop value to spell in its place. Raises `SopError` for a value with no
/// sop representation.
#[pyfunction]
#[pyo3(signature = (value, indent = None, convert = None))]
fn dumps(
    value: &Bound<'_, PyAny>,
    indent: Option<usize>,
    convert: Option<&Bound<'_, PyAny>>,
) -> PyResult<String> {
    let mut out = String::new();
    write::emit(value, &mut out, indent.unwrap_or(0), 0, convert, false, true)?;
    Ok(out)
}

// Declared as an inline `mod`, not a function: stub generation only sees
// exports declared this way. The doc comment is the module's `__doc__`.
/// The Rust implementation of sop as an extension module. An implementation
/// detail of `sop`: import the package, not this module.
#[pymodule]
mod _core {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::{SopError, Symbol, Tagged, dumps, loads};

    #[pymodule_init]
    fn init(module: &Bound<'_, PyModule>) -> PyResult<()> {
        let frozendict = module.py().import("builtins")?.getattr("frozendict")?;
        let _ = super::FROZENDICT.set(frozendict.unbind());
        Ok(())
    }
}
