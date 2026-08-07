//! Writing Python values out as sop text.
//!
//! One traversal and one writer: `emit` walks the value and spells it
//! directly, so escaping and number spelling live here and nowhere else.
//! The `convert` hook the SDK passes to `dumps` spells one
//! unclassifiable object as a plain value — its head only, children
//! untouched — inside the same traversal, so the graph is walked exactly
//! once.

use std::fmt::Write as _;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyMapping, PyString, PyTuple};

use crate::text::{escape_string, is_identifier, write_f64};
use crate::{FROZENDICT, Recursion, Symbol, Tagged, ser_error};

enum Classified<'py> {
    Null,
    Bool(bool),
    Int(Bound<'py, PyInt>),
    Float(f64),
    Str(String),
    Symbol(String),
    Tagged(String, Bound<'py, PyAny>),
    Array(Bound<'py, PyList>),
    Tuple(Bound<'py, PyTuple>),
    Object(Bound<'py, PyDict>),
    /// A frozendict's `(key, value)` pairs, via `items()`.
    FrozenObject(Bound<'py, PyList>),
}

/// Classify a Python object as a sop value.
///
/// The builtin cases are matched on the *exact* type. A subclass may carry a
/// tag — `class Iban(str)` is written `iban "DE89"` — and treating it as a
/// plain string here would drop the tag silently. `Ok(None)` is a type the
/// writer does not know, which the SDK's `convert` hook may spell; `Err` is
/// a value of a known type that cannot be spelled at all, which no
/// conversion can fix.
///
/// `bool` is tested before `int`: in Python `bool` subclasses `int`, so
/// `isinstance(True, int)` holds and the wrong order would spell `true` as `1`.
fn classify<'py>(value: &Bound<'py, PyAny>) -> PyResult<Option<Classified<'py>>> {
    if value.is_none() {
        return Ok(Some(Classified::Null));
    }
    if let Ok(b) = value.cast_exact::<PyBool>() {
        return Ok(Some(Classified::Bool(b.is_true())));
    }
    if let Ok(symbol) = value.cast::<Symbol>() {
        return Ok(Some(Classified::Symbol(symbol.get().name.clone())));
    }
    if let Ok(tagged) = value.cast::<Tagged>() {
        let tagged = tagged.get();
        return Ok(Some(Classified::Tagged(
            tagged.tag.clone(),
            tagged.value.bind(value.py()).clone(),
        )));
    }
    if let Ok(s) = value.cast_exact::<PyString>() {
        // A Python `str` may hold a lone surrogate; sop text cannot, and the
        // parser rejects the escape for the same reason. Refuse it as a
        // SopError rather than leaking the codec's UnicodeEncodeError.
        let text = s.to_cow().map_err(|_| {
            ser_error("string contains a lone surrogate, which sop text cannot hold".into())
        })?;
        return Ok(Some(Classified::Str(text.into_owned())));
    }
    if let Ok(i) = value.cast_exact::<PyInt>() {
        return Ok(Some(Classified::Int(i.clone())));
    }
    if let Ok(f) = value.cast_exact::<PyFloat>() {
        return Ok(Some(Classified::Float(f.value())));
    }
    if let Ok(items) = value.cast_exact::<PyList>() {
        return Ok(Some(Classified::Array(items.clone())));
    }
    if let Ok(items) = value.cast_exact::<PyTuple>() {
        return Ok(Some(Classified::Tuple(items.clone())));
    }
    if let Ok(map) = value.cast_exact::<PyDict>() {
        return Ok(Some(Classified::Object(map.clone())));
    }
    if let Some(t) = FROZENDICT.get()
        && value.get_type().as_ptr() == t.as_ptr()
    {
        let items = value.cast::<PyMapping>()?.items()?;
        return Ok(Some(Classified::FrozenObject(items)));
    }
    Ok(None)
}

/// An object key. Exact `str` only: under the protocol a str subclass
/// carries a tag, and a key has nowhere to put one, so accepting it here
/// would drop the tag silently. Written bare when its spelling is an
/// identifier, quoted otherwise.
fn write_key(out: &mut String, key: &Bound<'_, PyAny>) -> PyResult<()> {
    match key.cast_exact::<PyString>() {
        Ok(key) => {
            let key = key
                .to_cow()
                .map_err(|_| ser_error("object key contains a lone surrogate".to_string()))?;
            if is_identifier(&key) {
                out.push_str(&key);
            } else {
                escape_string(&key, out);
            }
            Ok(())
        }
        Err(_) if key.is_instance_of::<PyString>() => Err(ser_error(
            "a subclass of str carries a tag, which an object key cannot hold".to_string(),
        )),
        Err(_) => Err(ser_error("object keys must be strings".to_string())),
    }
}

fn refuse_tagged_symbol(under_tag: bool, name: &str) -> PyResult<()> {
    if under_tag {
        return Err(ser_error(format!("a tag cannot be applied to a bare symbol (`{name}`)")));
    }
    Ok(())
}

/// One `key: value` member.
fn write_member(
    out: &mut String,
    key: &Bound<'_, PyAny>,
    item: &Bound<'_, PyAny>,
    convert: &Bound<'_, PyAny>,
    first: bool,
) -> PyResult<()> {
    if !first {
        out.push(',');
    }
    write_key(out, key)?;
    out.push(':');
    emit(item, out, convert, false, true)
}

/// Spell a Python value into `out`.
///
/// `convert` spells one unclassifiable object as a plain value, and
/// `may_convert` is false exactly when the value at hand *is* a hook result,
/// so a hook that returns another unknown is an error rather than a loop.
/// `under_tag` is true for a tag's payload, where a bare symbol has no legal
/// spelling.
pub(crate) fn emit(
    value: &Bound<'_, PyAny>,
    out: &mut String,
    convert: &Bound<'_, PyAny>,
    under_tag: bool,
    may_convert: bool,
) -> PyResult<()> {
    // A Python value can nest arbitrarily, or cyclically, and overflowing the
    // stack here is an abort the caller cannot catch. Borrowing CPython's own
    // recursion guard -- as the stdlib json encoder does -- turns that into a
    // RecursionError at the interpreter's limit, with no limit of our own.
    let _guard = Recursion::enter(value.py(), c" while writing sop")?;
    let kind = match classify(value)? {
        Some(kind) => kind,
        None => {
            if !may_convert {
                return Err(ser_error(format!("not a sop value: {}", value.repr()?)));
            }
            let converted = convert.call1((value,))?;
            return emit(&converted, out, convert, under_tag, false);
        }
    };
    match kind {
        Classified::Null => {
            refuse_tagged_symbol(under_tag, "null")?;
            out.push_str("null");
        }
        Classified::Bool(b) => {
            let name = if b { "true" } else { "false" };
            refuse_tagged_symbol(under_tag, name)?;
            out.push_str(name);
        }
        // Symbol and Tagged validate their names on construction, so the
        // writer cannot be handed one that does not round-trip.
        Classified::Symbol(name) => {
            refuse_tagged_symbol(under_tag, &name)?;
            out.push_str(&name);
        }
        Classified::Str(s) => escape_string(&s, out),
        // Python integers are unbounded; the format's numeric domain is i64
        // or f64. A larger one would not read back to an equal value, so it
        // is refused for the same reason the parser refuses a literal that
        // overflows to infinity. The remedy is the same either way: carry
        // exact values as a tagged string.
        Classified::Int(i) => match i.extract::<i64>() {
            Ok(v) => {
                let _ = write!(out, "{v}");
            }
            Err(_) => {
                return Err(ser_error(format!(
                    "{} is out of range for sop's numeric domain; use a tagged string",
                    i.str()?.to_cow()?
                )));
            }
        },
        Classified::Float(f) if f.is_finite() => write_f64(f, out),
        Classified::Float(f) => return Err(ser_error(format!("{f} has no sop representation"))),
        Classified::Tagged(tag, payload) => {
            out.push_str(&tag);
            out.push(' ');
            emit(&payload, out, convert, true, true)?;
        }
        Classified::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit(&item, out, convert, false, true)?;
            }
            out.push(']');
        }
        Classified::Tuple(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                emit(&item, out, convert, false, true)?;
            }
            out.push(']');
        }
        Classified::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                write_member(out, &key, &item, convert, i == 0)?;
            }
            out.push('}');
        }
        Classified::FrozenObject(items) => {
            out.push('{');
            for (i, pair) in items.iter().enumerate() {
                let (key, item) = pair.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()?;
                write_member(out, &key, &item, convert, i == 0)?;
            }
            out.push('}');
        }
    }
    Ok(())
}
