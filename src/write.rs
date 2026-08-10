//! Writing Python values out as sop text.
//!
//! One traversal and one writer: [`Writer`] walks the value and spells it
//! directly, so escaping and number spelling live here and nowhere else.
//!
//! The writer knows exactly the values reading produces — the immutable set.
//! Everything else, the mutable counterparts included, is spelled by the
//! `convert` hook the SDK passes to `dumps`: one Python call per object the
//! writer cannot classify, its head only with its children untouched, inside
//! this same traversal, so the graph is walked exactly once.

use std::borrow::Cow;
use std::fmt::Write as _;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyFrozenDict, PyFrozenDictMethods, PyInt, PyString, PyTuple};

use crate::text::{escape_string, is_identifier, write_f64};
use crate::{Recursion, ShapeError, SopError, Symbol, Tagged, ser_error};

/// What kind of sop value an object is, and the object itself.
///
/// Each variant carries the value rather than a reading of it: cloning a
/// `Bound` is a reference count, where pulling out the `str`, the symbol's
/// name or the tag would copy a string on every value written. `bool` and
/// `f64` are the exceptions, being copies already.
enum Classified<'py> {
    Null,
    Bool(bool),
    Int(Bound<'py, PyInt>),
    Float(f64),
    Str(Bound<'py, PyString>),
    Symbol(Bound<'py, Symbol>),
    Tagged(Bound<'py, Tagged>),
    Array(Bound<'py, PyTuple>),
    Object(Bound<'py, PyFrozenDict>),
}

/// Classify a Python object as a sop value.
///
/// The builtin cases are matched on the *exact* type. A subclass may carry a
/// tag — `class Iban(str)` is written `iban "DE89"` — and treating it as a
/// plain string here would drop the tag silently. `None` is a type the writer
/// does not know, which the SDK's `convert` hook may spell.
///
/// Only the kind is decided here, so this cannot fail. A value of a known
/// type that still cannot be spelled — an oversized integer, a non-finite
/// float, a lone surrogate — is refused by [`Writer::emit`], where the
/// spelling is: it has been classified, so no conversion is consulted for it.
///
/// `bool` is tested before `int`: in Python `bool` subclasses `int`, so
/// `isinstance(True, int)` holds and the wrong order would spell `true` as `1`.
fn classify<'py>(value: &Bound<'py, PyAny>) -> Option<Classified<'py>> {
    if value.is_none() {
        return Some(Classified::Null);
    }
    if let Ok(b) = value.cast_exact::<PyBool>() {
        return Some(Classified::Bool(b.is_true()));
    }
    if let Ok(symbol) = value.cast::<Symbol>() {
        return Some(Classified::Symbol(symbol.clone()));
    }
    if let Ok(tagged) = value.cast::<Tagged>() {
        return Some(Classified::Tagged(tagged.clone()));
    }
    if let Ok(s) = value.cast_exact::<PyString>() {
        return Some(Classified::Str(s.clone()));
    }
    if let Ok(i) = value.cast_exact::<PyInt>() {
        return Some(Classified::Int(i.clone()));
    }
    if let Ok(f) = value.cast_exact::<PyFloat>() {
        return Some(Classified::Float(f.value()));
    }
    if let Ok(items) = value.cast_exact::<PyTuple>() {
        return Some(Classified::Array(items.clone()));
    }
    if let Ok(map) = value.cast_exact::<PyFrozenDict>() {
        return Some(Classified::Object(map.clone()));
    }
    None
}

/// Name one step of the place an error happened.
///
/// The path is assembled on the way out rather than carried on the way in:
/// each frame the error passes through prepends the step it took, so only a
/// write that actually fails pays for knowing where. Carrying a path down
/// the descent instead would charge every value written for the few that
/// fail -- and it is the cheapest values, the elements of a flat array, that
/// it would charge the most.
///
/// Anything that is not one of the writer's own errors passes through
/// untouched: a `ValueError` from `Symbol`, a `RecursionError` from the
/// guard, whatever else the hook raised.
fn at(py: Python<'_>, err: PyErr, segment: &str) -> PyErr {
    if !err.is_instance_of::<ShapeError>(py) {
        return err;
    }
    let value = err.value(py);
    match (
        value.getattr("path").and_then(|p| p.extract::<String>()),
        value.getattr("message").and_then(|m| m.extract::<String>()),
    ) {
        (Ok(path), Ok(message)) => ser_error(format!("{segment}{path}"), message),
        _ => err,
    }
}

pub(crate) struct Writer<'a, 'py> {
    pub(crate) out: String,
    convert: &'a Bound<'py, PyAny>,
}

impl<'a, 'py> Writer<'a, 'py> {
    pub(crate) const fn new(convert: &'a Bound<'py, PyAny>) -> Self {
        Writer {
            out: String::new(),
            convert,
        }
    }

    /// Spell the whole value. The root is the last step named, so a failure
    /// at the top level reads `$` and one below it reads `$.orders[2]`.
    pub(crate) fn dump(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.emit(value, false, true)
            .map_err(|err| at(value.py(), err, "$"))
    }

    /// A failure at the value being written now, which names no place yet:
    /// the frames it unwinds through are what know where that is.
    fn err<T>(&self, message: String) -> PyResult<T> {
        Err(ser_error(String::new(), message))
    }

    /// A hook failure becomes the writer's own, so that it too collects a
    /// path on the way out. `convert` is handed a single object and cannot
    /// know where in the graph it sits.
    fn locate(&self, py: Python<'_>, err: PyErr) -> PyErr {
        if !err.get_type(py).is(py.get_type::<SopError>()) {
            return err;
        }
        match err.value(py).getattr("message").and_then(|m| m.extract()) {
            Ok(message) => ser_error(String::new(), message),
            Err(_) => err,
        }
    }

    /// Spell a Python value.
    ///
    /// `may_convert` is false exactly when the value at hand *is* a hook
    /// result, so a hook that returns another unknown is an error rather than
    /// a loop. `under_tag` is true for a tag's payload, where a bare symbol
    /// has no legal spelling.
    pub(crate) fn emit(
        &mut self,
        value: &Bound<'_, PyAny>,
        under_tag: bool,
        may_convert: bool,
    ) -> PyResult<()> {
        // A Python value can nest arbitrarily, or cyclically, and overflowing
        // the stack here is an abort the caller cannot catch. Borrowing
        // CPython's own recursion guard -- as the stdlib json encoder does --
        // turns that into a RecursionError at the interpreter's limit, with no
        // limit of our own.
        let py = value.py();
        let _guard = Recursion::enter(py, c" while writing sop")?;
        let Some(kind) = classify(value) else {
            if !may_convert {
                return self.err(format!("not a sop value: {}", value.repr()?));
            }
            let converted = self
                .convert
                .call1((value,))
                .map_err(|err| self.locate(py, err))?;
            return self.emit(&converted, under_tag, false);
        };
        match kind {
            Classified::Null => {
                self.refuse_tagged_symbol(under_tag, "null")?;
                self.out.push_str("null");
            }
            Classified::Bool(b) => {
                let name = if b { "true" } else { "false" };
                self.refuse_tagged_symbol(under_tag, name)?;
                self.out.push_str(name);
            }
            // Symbol and Tagged validate their names on construction, so the
            // writer cannot be handed one that does not round-trip.
            Classified::Symbol(symbol) => {
                let name = &symbol.get().name;
                self.refuse_tagged_symbol(under_tag, name)?;
                self.out.push_str(name);
            }
            Classified::Str(s) => {
                // A Python `str` may hold a lone surrogate; sop text cannot,
                // and the parser rejects the escape for the same reason.
                // Refuse it as a SopError rather than leaking the codec's
                // UnicodeEncodeError.
                let Ok(text) = s.to_cow() else {
                    return self.err(
                        "string contains a lone surrogate, which sop text cannot hold".into(),
                    );
                };
                escape_string(&text, &mut self.out);
            }
            // Python integers are unbounded; the format's numeric domain is
            // i64 or f64. A larger one would not read back to an equal value,
            // so it is refused for the same reason the parser refuses a
            // literal that overflows to infinity. The remedy is the same
            // either way: carry exact values as a tagged string.
            Classified::Int(i) => match i.extract::<i64>() {
                Ok(v) => {
                    let _ = write!(self.out, "{v}");
                }
                Err(_) => {
                    return self.err(format!(
                        "{} is out of range for sop's numeric domain; use a tagged string",
                        i.str()?.to_cow()?
                    ));
                }
            },
            Classified::Float(f) if f.is_finite() => write_f64(f, &mut self.out),
            Classified::Float(f) => {
                return self.err(format!("{f} has no sop representation"));
            }
            Classified::Tagged(tagged) => {
                let tagged = tagged.get();
                self.out.push_str(&tagged.tag);
                self.out.push(' ');
                self.emit(tagged.value.bind(py), true, true)?;
            }
            Classified::Array(items) => {
                self.out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        self.out.push(',');
                    }
                    self.emit(&item, false, true)
                        .map_err(|err| at(py, err, &format!("[{i}]")))?;
                }
                self.out.push(']');
            }
            Classified::Object(map) => {
                self.out.push('{');
                for (i, (key, item)) in map.iter().enumerate() {
                    if i > 0 {
                        self.out.push(',');
                    }
                    let text = self.key_text(&key)?;
                    if is_identifier(&text) {
                        self.out.push_str(&text);
                    } else {
                        escape_string(&text, &mut self.out);
                    }
                    self.out.push(':');
                    self.emit(&item, false, true)
                        .map_err(|err| at(py, err, &format!(".{text}")))?;
                }
                self.out.push('}');
            }
        }
        Ok(())
    }

    fn refuse_tagged_symbol(&self, under_tag: bool, name: &str) -> PyResult<()> {
        if under_tag {
            return self.err(format!(
                "a tag cannot be applied to a bare symbol (`{name}`)"
            ));
        }
        Ok(())
    }

    /// An object key's spelling, which the caller writes and then extends
    /// the path with. Exact `str` only: under the protocol a str subclass
    /// carries a tag, and a key has nowhere to put one, so accepting it here
    /// would drop the tag silently.
    fn key_text<'k>(&self, key: &'k Bound<'_, PyAny>) -> PyResult<Cow<'k, str>> {
        match key.cast_exact::<PyString>() {
            Ok(key) => match key.to_cow() {
                Ok(text) => Ok(text),
                Err(_) => self.err("object key contains a lone surrogate".to_string()),
            },
            Err(_) if key.is_instance_of::<PyString>() => self.err(
                "a subclass of str carries a tag, which an object key cannot hold".to_string(),
            ),
            Err(_) => self.err("object keys must be strings".to_string()),
        }
    }
}
