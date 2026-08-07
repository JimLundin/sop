//! Lexical facts shared by the reader and the writer: what an identifier is,
//! how a string is escaped, how a number is spelled. No pyo3 in here.

use std::fmt::Write as _;

/// U+000B and U+000C are whitespace but not line terminators, so they do not
/// end a line comment.
#[inline]
pub(crate) fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// ECMAScript `IdentifierName`, via `unicode-ident`'s XID_Start/XID_Continue.
#[inline]
pub(crate) fn is_id_start(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphabetic() || c == '_' || c == '$';
    }
    unicode_ident::is_xid_start(c)
}

#[inline]
pub(crate) fn is_id_part(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_alphanumeric() || c == '_' || c == '$';
    }
    c == '\u{200C}' || c == '\u{200D}' || unicode_ident::is_xid_continue(c)
}

/// Whether a string can be written as a bare identifier: an unquoted object
/// key, a symbol, or a tag name.
pub(crate) fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if is_id_start(c) => chars.all(is_id_part),
        _ => false,
    }
}

/// Escape a string as a sop string literal, quotes included.
pub(crate) fn escape_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Append the spelling of a finite float, always with a point or an exponent:
/// number kind is spelling-determined, so a float never reads back as an
/// integer, and `-0.0` keeps its sign. The caller has already refused a
/// non-finite value, which has no spelling.
pub(crate) fn write_f64(f: f64, out: &mut String) {
    debug_assert!(f.is_finite(), "a non-finite float has no sop spelling");
    let start = out.len();
    let _ = write!(out, "{f}");
    if !out[start..].contains(['.', 'e', 'E']) {
        out.push_str(".0");
    }
}
