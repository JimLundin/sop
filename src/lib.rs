//! sop — the sop interchange format, as the extension module behind the
//! `sop` Python package.
//!
//! One implementation, one pass each way. `loads` builds Python objects
//! directly as it parses ([`parse`]); `dumps` spells Python objects directly
//! as text ([`write`]); [`text`] holds the lexical facts they share. There is
//! no intermediate document tree.
//!
//! Value mapping. One set of values, immutable, read and written alike:
//!
//! ```text
//! sop object   <->  frozendict[str, Value]
//! sop array    <->  tuple[Value, ...]
//! sop string   <->  str
//! sop number   <->  int | float
//! sop symbol   <->  True | False | None | Symbol(name)
//! sop tagged   <->  Tagged(tag, value)
//! ```
//!
//! Anything else -- a dataclass, an enum, a set, and the mutable
//! counterparts `list` and `dict` -- is spelled by the `convert` hook the SDK
//! passes to `dumps`: one Python call per object the writer cannot classify,
//! and the traversal continues in place, so the graph is walked exactly once.

mod parse;
mod text;
mod write;

use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyGenericAlias, PyType};

#[pyclass(frozen, extends = PyValueError, subclass, module = "sop._core")]
pub(crate) struct SopError {
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl SopError {
    #[new]
    fn new(message: String) -> Self {
        Self { message }
    }
    fn __str__(&self) -> String {
        self.message.clone()
    }
}

#[pyclass(frozen, extends = SopError, module = "sop._core")]
pub(crate) struct ParseError {
    #[pyo3(get)]
    line: usize,
    #[pyo3(get)]
    column: usize,
}

#[pymethods]
impl ParseError {
    // Where first and what second, in the order `__str__` renders them and
    // the order `ShapeError` takes its own two: both subclasses are the base
    // plus a location, so both spell the location first.
    #[new]
    fn new(line: usize, column: usize, message: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(SopError { message }).add_subclass(Self { line, column })
    }
    // A `Bound` receiver, so both halves are read without a borrow flag:
    // `get` for this class's own fields, `as_super` for the message, which
    // lives on the base.
    fn __str__(slf: &Bound<'_, Self>) -> String {
        let this = slf.get();
        format!(
            "{}:{}: {}",
            this.line,
            this.column,
            slf.as_super().get().message
        )
    }
}

#[pyclass(frozen, extends = SopError, module = "sop._core")]
pub(crate) struct ShapeError {
    #[pyo3(get)]
    path: String,
}

#[pymethods]
impl ShapeError {
    #[new]
    fn new(path: String, message: String) -> PyClassInitializer<Self> {
        PyClassInitializer::from(SopError { message }).add_subclass(Self { path })
    }
    fn __str__(slf: &Bound<'_, Self>) -> String {
        format!("{}: {}", slf.get().path, slf.as_super().get().message)
    }
}

/// A read error, at the position in the text where reading stopped.
pub(crate) fn sop_error(line: usize, column: usize, message: String) -> PyErr {
    PyErr::new::<ParseError, _>((line, column, message))
}

/// A write error, at the path into the value where writing stopped. The
/// same spelling the reader uses, so a value that fails to write and a
/// document that fails to read name the place the same way.
pub(crate) fn ser_error(path: String, message: String) -> PyErr {
    PyErr::new::<ShapeError, _>((path, message))
}

/// CPython's recursion guard, entered once per parse or emit frame and left
/// on drop.
pub(crate) struct Recursion;

impl Recursion {
    pub(crate) fn enter(py: Python<'_>, task: &'static std::ffi::CStr) -> PyResult<Self> {
        match unsafe { pyo3::ffi::Py_EnterRecursiveCall(task.as_ptr()) } {
            0 => Ok(Self),
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
#[pyclass(frozen, eq, hash, module = "sop._core")]
#[derive(PartialEq, Eq, Hash, Clone)]
pub(crate) struct Symbol {
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
        Ok(Self { name })
    }

    fn __repr__(&self) -> String {
        format!("Symbol({:?})", self.name)
    }

    /// Rebuilt by calling the class with the argument it was built from, so
    /// `copy`, `deepcopy` and `pickle` all work. Every other value the
    /// reader produces already does, and one that is handed to a worker
    /// process and cannot come back is one the caller never gets to read.
    /// Going back through the constructor is also what re-checks the name,
    /// so a hostile pickle cannot mint a `Symbol("null")`.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> (Bound<'py, PyType>, (String,)) {
        (slf.get_type(), (slf.get().name.clone(),))
    }
}

/// A tag applied to a payload, generic in what it carries: `Tagged[V]`. Tags
/// are constructive: a tagged value is never equal to the value it wraps.
///
/// Frozen, like `Symbol`: the native types are immutable. Equality and
/// hashing behave as a frozen dataclass's would — equal by `(tag, value)`,
/// hashable exactly when the payload is.
///
/// The payload is never `None`, a bool or a `Symbol`: those are bare symbols
/// on the wire, and a tag cannot be applied to a bare symbol, so such a
/// value could never be written or read back.
#[pyclass(frozen, module = "sop._core")]
pub(crate) struct Tagged {
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
        Ok(Self {
            tag,
            value: value.unbind(),
        })
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Tagged({:?}, {})",
            self.tag,
            self.value.bind(py).repr()?
        ))
    }

    /// As `Symbol`, and for the same reasons. `deepcopy` copies the payload
    /// because it is one of the arguments the value is rebuilt from.
    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> (Bound<'py, PyType>, (String, Bound<'py, PyAny>)) {
        let this = slf.get();
        (
            slf.get_type(),
            (this.tag.clone(), this.value.bind(slf.py()).clone()),
        )
    }

    /// `Tagged[V]`, so a payload's type can be named in a shape as well as in
    /// the stubs -- `loads[Tagged[str]]` reads a tag over a string.
    #[classmethod]
    fn __class_getitem__<'py>(
        cls: &Bound<'py, PyType>,
        item: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyGenericAlias>> {
        PyGenericAlias::new(cls.py(), cls.as_any(), item)
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let Ok(other) = other.cast::<Self>() else {
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
    // Truncating on a 32-bit target and wrapping into the sign bit are both
    // fine: a hash only has to be deterministic and agree with `__eq__`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    fn __hash__(&self, py: Python<'_>) -> PyResult<isize> {
        let mut hasher = DefaultHasher::new();
        self.tag.hash(&mut hasher);
        self.value.bind(py).hash()?.hash(&mut hasher);
        Ok(hasher.finish() as isize)
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

/// Write a value as sop text. `convert` is called once per object the
/// writer cannot classify -- every mutable container included -- and must
/// return a sop value to spell in its place. Raises `SopError` for a value
/// with no sop representation, and passes through whatever `convert` itself
/// raises: `Symbol` and `Tagged` refuse a bad name with a plain `ValueError`,
/// and the SDK does not restate the rule by re-wrapping it.
#[pyfunction]
fn dumps(value: &Bound<'_, PyAny>, convert: &Bound<'_, PyAny>) -> PyResult<String> {
    let mut writer = write::Writer::new(convert);
    writer.dump(value)?;
    Ok(writer.out)
}

// Declared as an inline `mod`, not a function: stub generation only sees
// exports declared this way. The doc comment is the module's `__doc__`.
/// The Rust implementation of sop as an extension module. An implementation
/// detail of `sop`: import the package, not this module.
#[pymodule]
mod _core {
    #[pymodule_export]
    use super::{ParseError, ShapeError, SopError, Symbol, Tagged, dumps, loads};
}
