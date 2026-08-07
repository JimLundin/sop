//! Writing a document back out as sop text.
//!
//! Serialises straight from the tape. A `Document` comes from the parser or
//! the [`Builder`](crate::Builder), and both uphold every invariant the writer
//! needs — symbol and tag names are identifiers, numbers are finite — so
//! serialisation is infallible.

use std::collections::HashMap;
use std::fmt::Write as _;

use crate::document::{ArrayIter, Document, Kind, Ref};
use crate::parser::is_identifier;

/// Escape a string as a sop string literal, quotes included.
fn escape_string(s: &str, out: &mut String) {
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

/// Append the spelling of a float, always with a point or an exponent:
/// number kind is spelling-determined, so a float never reads back as an
/// integer, and `-0.0` keeps its sign. `false` if the value has no spelling,
/// which the tape cannot hold: the parser rejects literals that overflow to
/// infinity and the builder refuses a non-finite float.
fn write_f64(f: f64, out: &mut String) -> bool {
    if !f.is_finite() {
        return false;
    }
    let start = out.len();
    let _ = write!(out, "{f}");
    if !out[start..].contains(['.', 'e', 'E']) {
        out.push_str(".0");
    }
    true
}

fn pad(out: &mut String, indent: usize, level: usize) {
    if indent > 0 {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', indent * level));
    }
}

fn write(root: Ref<'_>, out: &mut String, indent: usize) {
    /// What is open above the value being written. Only a container pushes a
    /// frame -- a flat array costs one frame however long it is -- so nesting
    /// costs heap rather than stack and any parsed document can be written.
    /// The bool is whether the next child is the first.
    enum Frame<'a> {
        Array(ArrayIter<'a>, bool),
        Object(std::vec::IntoIter<(&'a str, Ref<'a>)>, bool),
    }
    let mut frames: Vec<Frame<'_>> = Vec::new();
    let mut next = Some(root);
    loop {
        // Spell the head of the pending value, chaining through tags; a
        // non-empty container opens here and closes in the advance step.
        while let Some(node) = next.take() {
            match node.kind() {
                Kind::Symbol => out.push_str(node.symbol().unwrap()),
                Kind::String => escape_string(node.as_str().unwrap(), out),
                Kind::Number => match node.as_i64() {
                    Some(i) => {
                        let _ = write!(out, "{i}");
                    }
                    None => {
                        let wrote = write_f64(node.as_f64().unwrap(), out);
                        assert!(wrote, "a parsed number is finite");
                    }
                },
                Kind::Tagged => {
                    out.push_str(node.tag().unwrap());
                    out.push(' ');
                    next = Some(node.payload().unwrap());
                }
                Kind::Array => {
                    if node.is_empty() {
                        out.push_str("[]");
                    } else {
                        out.push('[');
                        frames.push(Frame::Array(node.array(), true));
                    }
                }
                Kind::Object => {
                    if node.is_empty() {
                        out.push_str("{}");
                        continue;
                    }
                    // The tape keeps duplicate keys as written; the last
                    // value survives, in the first occurrence's position.
                    // Collecting is last-wins, and removal on emission skips
                    // later duplicates, so one map resolves in linear time.
                    let mut last: HashMap<&str, Ref<'_>> = node.object().collect();
                    let mut members: Vec<(&str, Ref<'_>)> = Vec::with_capacity(last.len());
                    for (key, _) in node.object() {
                        if let Some(value) = last.remove(key) {
                            members.push((key, value));
                        }
                    }
                    out.push('{');
                    frames.push(Frame::Object(members.into_iter(), true));
                }
            }
        }

        // Advance the innermost open container: next child, or close it.
        let level = frames.len();
        let close = match frames.last_mut() {
            None => return,
            Some(Frame::Array(iter, first)) => match iter.next() {
                Some(item) => {
                    if !*first {
                        out.push(',');
                    }
                    *first = false;
                    pad(out, indent, level);
                    next = Some(item);
                    None
                }
                None => Some(']'),
            },
            Some(Frame::Object(iter, first)) => match iter.next() {
                Some((key, value)) => {
                    if !*first {
                        out.push(',');
                    }
                    *first = false;
                    pad(out, indent, level);
                    if is_identifier(key) {
                        out.push_str(key);
                    } else {
                        escape_string(key, out);
                    }
                    out.push(':');
                    if indent > 0 {
                        out.push(' ');
                    }
                    next = Some(value);
                    None
                }
                None => Some('}'),
            },
        };
        if let Some(bracket) = close {
            frames.pop();
            pad(out, indent, frames.len());
            out.push(bracket);
        }
    }
}

impl Document {
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        write(self.root(), &mut out, 0);
        out
    }

    pub fn to_string_pretty(&self, indent: usize) -> String {
        let mut out = String::new();
        write(self.root(), &mut out, indent.max(1));
        out
    }
}
