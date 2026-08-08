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
        Ok(Symbol { name })
    }

    fn __repr__(&self) -> String {
        format!("Symbol({:?})", self.name)
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
    writer.emit(value, false, true)?;
    Ok(writer.out)
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
