//! Reading sop text straight into Python objects.
//!
//! One pass: the scanner lexes and the Python value is built as it goes —
//! there is no intermediate document. Byte-oriented rather than
//! character-oriented: ASCII takes a fast path and only the cases that can be
//! non-ASCII — whitespace, identifiers, string contents — decode a `char`.
//! Line and column are tracked in *characters*, so diagnostics agree with
//! anything that measures the source in code points.
//!
//! Descent is recursive, guarded by CPython's own recursion check: past the
//! interpreter's limit `loads` raises RecursionError — the interpreter's
//! bound, not one of ours — exactly as `dumps` is bounded when writing.

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyFrozenDict, PyString, PyTuple};

use crate::text::{is_id_part, is_id_start, is_line_terminator};
use crate::{Recursion, Symbol, Tagged, sop_error};

const BOM: &str = "\u{FEFF}";

/// The tokens a value can start with.
fn starts_value(c: char) -> bool {
    matches!(c, '{' | '[' | '"' | '+' | '-') || c.is_ascii_digit() || is_id_start(c)
}

pub(crate) struct Parser<'a, 'py> {
    py: Python<'py>,
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
    /// Decoded string contents, one string at a time, reused across the
    /// document so decoding allocates only when a string outgrows it.
    scratch: String,
    /// One Python string per distinct object key, as the stdlib json
    /// scanner's memo does: a key repeated across ten thousand records costs
    /// one allocation and one hash, not ten thousand.
    keys: HashMap<String, Py<PyString>>,
}

impl<'a, 'py> Parser<'a, 'py> {
    pub(crate) fn new(py: Python<'py>, src: &'a str) -> Self {
        // A BOM is skipped in leading position, and only there.
        let pos = if src.starts_with(BOM) { BOM.len() } else { 0 };
        Parser {
            py,
            src,
            bytes: src.as_bytes(),
            pos,
            line: 1,
            col: 1,
            scratch: String::new(),
            keys: HashMap::new(),
        }
    }

    // -- cursor -------------------------------------------------------------

    #[inline]
    fn eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    #[inline]
    fn byte(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    #[inline]
    fn ch(&self) -> Option<char> {
        if self.eof() {
            return None;
        }
        let b = self.bytes[self.pos];
        if b < 0x80 {
            Some(b as char)
        } else {
            self.src[self.pos..].chars().next()
        }
    }

    fn bump(&mut self) -> char {
        let b = self.bytes[self.pos];
        if b < 0x80 {
            self.pos += 1;
            if b == b'\n' || b == b'\r' {
                // CRLF counts once, on the LF.
                if b == b'\r' && self.byte(0) == Some(b'\n') {
                    self.col += 1;
                } else {
                    self.line += 1;
                    self.col = 1;
                }
            } else {
                self.col += 1;
            }
            return b as char;
        }
        let c = self.src[self.pos..].chars().next().expect("on a char boundary");
        self.pos += c.len_utf8();
        if is_line_terminator(c) {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        c
    }

    fn err<T>(&self, message: impl Into<String>) -> PyResult<T> {
        self.err_at(message, self.line, self.col)
    }

    fn err_at<T>(&self, message: impl Into<String>, line: usize, column: usize) -> PyResult<T> {
        Err(sop_error(message.into(), line, column))
    }

    // -- whitespace and comments --------------------------------------------

    fn skip_ws(&mut self) -> PyResult<()> {
        loop {
            let Some(b) = self.byte(0) else { return Ok(()) };
            if b < 0x80 && (b as char).is_whitespace() {
                self.bump();
            } else if b == b'/' && self.byte(1) == Some(b'/') {
                self.bump();
                self.bump();
                while let Some(c) = self.ch() {
                    if is_line_terminator(c) {
                        break;
                    }
                    self.bump();
                }
            } else if b == b'/' && self.byte(1) == Some(b'*') {
                let (line, column) = (self.line, self.col);
                self.bump();
                self.bump();
                loop {
                    if self.eof() {
                        return self.err_at("unterminated block comment", line, column);
                    }
                    if self.byte(0) == Some(b'*') && self.byte(1) == Some(b'/') {
                        self.bump();
                        self.bump();
                        break;
                    }
                    self.bump(); // block comments do not nest
                }
            } else if b < 0x80 {
                return Ok(());
            } else {
                let c = self.ch().expect("not at eof");
                if c == '\u{FEFF}' {
                    return self.err("U+FEFF is only permitted at the start of a document");
                }
                // `char::is_whitespace` is the Unicode `White_Space`
                // property, and correctly excludes U+FEFF.
                if c.is_whitespace() {
                    self.bump();
                } else {
                    return Ok(());
                }
            }
        }
    }

    // -- entry point --------------------------------------------------------

    pub(crate) fn parse_document(mut self) -> PyResult<Py<PyAny>> {
        self.skip_ws()?;
        if self.eof() {
            return self.err("unexpected end of input: a document must contain one value");
        }
        let value = self.parse_value(false)?;
        self.skip_ws()?;
        if !self.eof() {
            return self.err("trailing content after the top-level value");
        }
        Ok(value)
    }

    // -- values -------------------------------------------------------------

    /// One value, head to completion. `under_tag` is true for a tag's
    /// payload, where a bare symbol has no legal spelling.
    fn parse_value(&mut self, under_tag: bool) -> PyResult<Py<PyAny>> {
        let _guard = Recursion::enter(self.py, c" while reading sop")?;
        let c = self.ch().expect("every caller checks for eof");
        match c {
            '{' => self.parse_object(),
            '[' => self.parse_array(),
            '"' => {
                self.parse_string()?;
                Ok(PyString::new(self.py, &self.scratch).into_any().unbind())
            }
            '+' | '-' => self.parse_number(),
            c if c.is_ascii_digit() => self.parse_number(),
            c if is_id_start(c) => {
                let (line, column) = (self.line, self.col);
                let name = self.read_identifier();
                self.skip_ws()?;
                // The argument is taken greedily. The tokens that can start a
                // value and those that can follow one are disjoint, so one
                // token decides.
                if let Some(c) = self.ch()
                    && starts_value(c)
                {
                    let value = self.parse_value(true)?;
                    let tagged = Tagged { tag: name.to_owned(), value };
                    return Ok(Py::new(self.py, tagged)?.into_any());
                }
                // A bare symbol cannot be a tag's payload. Without this,
                // a missing comma between two identifiers would silently
                // denote one tagged value instead of being an error.
                if under_tag {
                    return self.err_at("a tag cannot be applied to a bare symbol", line, column);
                }
                // On the wire `true`, `false` and `null` are ordinary
                // symbols. Choosing host types for them is the SDK's call,
                // and this is where it is made.
                Ok(match name {
                    "true" => PyBool::new(self.py, true).to_owned().into_any().unbind(),
                    "false" => PyBool::new(self.py, false).to_owned().into_any().unbind(),
                    "null" => self.py.None(),
                    name => Py::new(self.py, Symbol { name: name.to_owned() })?.into_any(),
                })
            }
            c => self.err(format!("unexpected character {c:?}")),
        }
    }

    fn parse_array(&mut self) -> PyResult<Py<PyAny>> {
        let (line, column) = (self.line, self.col);
        self.bump();
        let mut items: Vec<Py<PyAny>> = Vec::new();
        loop {
            // Checked at the head of each element, so the empty array and a
            // trailing comma are the same case, as in `parse_object`.
            self.skip_ws()?;
            match self.byte(0) {
                None => return self.err_at("unterminated array", line, column),
                Some(b']') => {
                    self.bump();
                    break;
                }
                Some(_) => {}
            }
            items.push(self.parse_value(false)?);
            self.skip_ws()?;
            match self.byte(0) {
                Some(b',') => self.bump(),
                Some(b']') => {
                    self.bump();
                    break;
                }
                None => return self.err_at("unterminated array", line, column),
                Some(_) => {
                    return self.err(
                        "expected ',' or ']': only an identifier may be followed by a value",
                    );
                }
            };
        }
        Ok(PyTuple::new(self.py, items)?.into_any().unbind())
    }

    fn parse_object(&mut self) -> PyResult<Py<PyAny>> {
        let (line, column) = (self.line, self.col);
        self.bump();
        // Duplicates are accepted and the last wins, which is what assigning
        // into a dict does, keeping the first position. The dict is
        // scaffolding; what leaves is a frozendict.
        let map = PyDict::new(self.py);
        loop {
            self.skip_ws()?;
            if self.eof() {
                return self.err_at("unterminated object", line, column);
            }
            if self.byte(0) == Some(b'}') {
                self.bump();
                break;
            }
            let key = self.parse_key()?;
            self.skip_ws()?;
            if self.byte(0) != Some(b':') {
                return self.err("expected ':' after an object key (tags on keys are not permitted)");
            }
            self.bump();
            self.skip_ws()?;
            if self.eof() {
                return self.err("expected a value after ':'");
            }
            let value = self.parse_value(false)?;
            map.set_item(key, value)?;
            self.skip_ws()?;
            match self.byte(0) {
                Some(b',') => {
                    self.bump(); // a following '}' is a legal trailing comma
                }
                Some(b'}') => {
                    self.bump();
                    break;
                }
                None => return self.err_at("unterminated object", line, column),
                Some(_) => {
                    return self.err(
                        "expected ',' or '}': only an identifier may be followed by a value",
                    );
                }
            }
        }
        Ok(PyFrozenDict::from_sequence(map.as_any())?.into_any().unbind())
    }

    // -- identifiers and keys -----------------------------------------------

    fn read_identifier(&mut self) -> &'a str {
        let start = self.pos;
        self.bump();
        // ASCII fast path; identifiers are overwhelmingly ASCII.
        while let Some(b) = self.byte(0) {
            if b < 0x80 {
                if b.is_ascii_alphanumeric() || b == b'_' || b == b'$' {
                    self.pos += 1;
                    self.col += 1;
                    continue;
                }
                break;
            }
            match self.ch() {
                Some(c) if is_id_part(c) => {
                    self.bump();
                }
                _ => break,
            }
        }
        &self.src[start..self.pos]
    }

    fn parse_key(&mut self) -> PyResult<Py<PyString>> {
        let c = self.ch().expect("caller checked for eof");
        if c == '"' {
            self.parse_string()?;
            return Ok(key_string(self.py, &mut self.keys, &self.scratch));
        }
        if is_id_start(c) {
            // An identifier key denotes the string of its spelling.
            let name = self.read_identifier();
            return Ok(key_string(self.py, &mut self.keys, name));
        }
        self.err(format!("expected an object key, found {c:?}"))
    }

    // -- strings ------------------------------------------------------------

    /// Decode a string literal into `self.scratch`, both quotes consumed.
    fn parse_string(&mut self) -> PyResult<()> {
        let (line, column) = (self.line, self.col);
        self.bump();
        self.scratch.clear();

        loop {
            // Fast path: copy the longest run of plain ASCII in one go.
            let run_start = self.pos;
            while let Some(b) = self.byte(0) {
                if b >= 0x20 && b < 0x80 && b != b'"' && b != b'\\' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.pos > run_start {
                self.col += self.pos - run_start; // the run holds no line terminator
                let chunk = &self.src[run_start..self.pos];
                self.scratch.push_str(chunk);
            }

            let Some(b) = self.byte(0) else {
                return self.err_at("unterminated string", line, column);
            };
            match b {
                b'"' => {
                    self.bump();
                    return Ok(());
                }
                b'\\' => {
                    self.bump();
                    let c = self.read_escape()?;
                    self.scratch.push(c);
                }
                b if b < 0x20 => {
                    return self.err(format!("unescaped control character U+{b:04X} in string"));
                }
                _ => {
                    let c = self.bump();
                    self.scratch.push(c);
                }
            }
        }
    }

    fn read_escape(&mut self) -> PyResult<char> {
        if self.eof() {
            return self.err("unterminated escape sequence");
        }
        match self.bump() {
            c @ ('"' | '\\' | '/') => Ok(c),
            'b' => Ok('\u{8}'),
            'f' => Ok('\u{c}'),
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            'u' => self.read_unicode_escape(),
            c => self.err(format!("invalid escape sequence '\\{c}'")),
        }
    }

    /// The `\uXXXX` body, a following low surrogate consumed with it.
    fn read_unicode_escape(&mut self) -> PyResult<char> {
        let high = self.read_hex4()?;
        if (0xD800..=0xDBFF).contains(&high)
            && self.byte(0) == Some(b'\\')
            && self.byte(1) == Some(b'u')
        {
            let saved = (self.pos, self.line, self.col);
            self.bump();
            self.bump();
            let low = self.read_hex4()?;
            if (0xDC00..=0xDFFF).contains(&low) {
                let cp = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
                return Ok(char::from_u32(cp).expect("valid surrogate pair"));
            }
            (self.pos, self.line, self.col) = saved;
        }

        // A `str` is well-formed UTF-8 and cannot hold an unpaired surrogate.
        // The alternative to rejecting one is substituting U+FFFD, which loses
        // data silently. Rejecting matches what serde_json does.
        match char::from_u32(high) {
            Some(c) => Ok(c),
            None => self.err(format!("unpaired surrogate escape U+{high:04X}")),
        }
    }

    fn read_hex4(&mut self) -> PyResult<u32> {
        let mut value = 0u32;
        for _ in 0..4 {
            match self.byte(0) {
                Some(b) if b.is_ascii_hexdigit() => {
                    value = value * 16 + (b as char).to_digit(16).expect("hex digit");
                    self.pos += 1;
                    self.col += 1;
                }
                _ => return self.err("'\\u' must be followed by four hexadecimal digits"),
            }
        }
        Ok(value)
    }

    // -- numbers ------------------------------------------------------------

    fn parse_number(&mut self) -> PyResult<Py<PyAny>> {
        let start = self.pos;
        if matches!(self.byte(0), Some(b'+') | Some(b'-')) {
            self.bump();
        }

        match self.byte(0) {
            Some(b) if b.is_ascii_digit() => {
                if b == b'0' {
                    self.bump();
                    if self.byte(0).is_some_and(|b| b.is_ascii_digit()) {
                        return self.err("leading zeros are not permitted");
                    }
                } else {
                    self.eat_digits();
                }
            }
            _ => return self.err("expected a digit"),
        }

        let mut exact = true;
        if self.byte(0) == Some(b'.') {
            self.bump();
            exact = false;
            if !self.byte(0).is_some_and(|b| b.is_ascii_digit()) {
                return self.err("expected a digit after '.'");
            }
            self.eat_digits();
        }

        if matches!(self.byte(0), Some(b'e') | Some(b'E')) {
            self.bump();
            exact = false;
            if matches!(self.byte(0), Some(b'+') | Some(b'-')) {
                self.bump();
            }
            if !self.byte(0).is_some_and(|b| b.is_ascii_digit()) {
                return self.err("expected a digit in the exponent");
            }
            self.eat_digits();
        }

        // Number kind is spelling-determined: digits alone denote an integer,
        // a point or an exponent a float. An integer beyond i64 falls back to
        // f64, matching how it would read on any peer without big integers.
        let text = &self.src[start..self.pos];
        if exact && let Ok(i) = text.parse::<i64>() {
            return Ok(i.into_pyobject(self.py)?.into_any().unbind());
        }
        match text.parse::<f64>() {
            // An overflow to infinity would produce a value with no spelling:
            // `Infinity` lexes as a symbol, not a number, so a parsed infinity
            // could never be written back out.
            Ok(f) if f.is_finite() => Ok(PyFloat::new(self.py, f).into_any().unbind()),
            Ok(_) => self.err(format!("`{text}` is out of range for a 64-bit float")),
            Err(_) => self.err(format!("`{text}` is not a representable number")),
        }
    }

    #[inline]
    fn eat_digits(&mut self) {
        while self.byte(0).is_some_and(|b| b.is_ascii_digit()) {
            self.pos += 1;
            self.col += 1;
        }
    }
}

/// The key memo. A free function over the fields rather than a method, so the
/// borrow of `self.scratch` at a call site does not conflict with it.
fn key_string(
    py: Python<'_>,
    keys: &mut HashMap<String, Py<PyString>>,
    text: &str,
) -> Py<PyString> {
    if let Some(k) = keys.get(text) {
        return k.clone_ref(py);
    }
    let new = PyString::new(py, text).unbind();
    keys.insert(text.to_owned(), new.clone_ref(py));
    new
}
