//! The scanner. Reads UTF-8 bytes and appends to a [`Document`] tape.
//!
//! Byte-oriented rather than character-oriented: ASCII takes a fast path and
//! only the cases that can be non-ASCII — whitespace, identifiers, string
//! contents — decode a `char`. Line and column are tracked in *characters*, so
//! diagnostics agree with anything that measures the source in code points.


use crate::Span;
use crate::document::{Document, Node};

/// Every error comes from the parser, so it always has a position.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{line}:{column}: {message}")]
pub struct Error {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

// ---------------------------------------------------------------------------
// Character classes
// ---------------------------------------------------------------------------

/// U+000B and U+000C are whitespace but not line terminators, so they do not
/// end a line comment.
#[inline]
fn is_line_terminator(c: char) -> bool {
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
pub fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if is_id_start(c) => chars.all(is_id_part),
        _ => false,
    }
}

const BOM: &str = "\u{FEFF}";

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// What is open above the value being parsed. Nesting pushes a `Frame` onto a
/// heap vector rather than a native stack frame, which is why depth is
/// unbounded: the grammar places no limit, and the parser need not invent one.
enum Frame {
    /// A tag whose payload is being parsed; `end` is finalised at `at`.
    Tag { at: usize },
    /// `line` and `column` locate the opening bracket, for unterminated errors.
    Array { at: usize, len: u32, line: usize, column: usize },
    Object { at: usize, len: u32, line: usize, column: usize },
}

pub(crate) struct Parser<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
    nodes: Vec<Node>,
    strings: String,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(src: &'a str) -> Self {
        // A BOM is skipped in leading position, and only there.
        let pos = if src.starts_with(BOM) { BOM.len() } else { 0 };
        Parser {
            src,
            bytes: src.as_bytes(),
            pos,
            line: 1,
            col: 1,
            // One node per value; most documents are mostly scalars, so this
            // over-reserves a little and reallocates rarely.
            nodes: Vec::with_capacity(src.len() / 16 + 8),
            strings: String::with_capacity(src.len() / 4 + 8),
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

    fn err<T>(&self, message: impl Into<String>) -> Result<T, Error> {
        self.err_at(message, self.line, self.col)
    }

    fn err_at<T>(&self, message: impl Into<String>, line: usize, column: usize) -> Result<T, Error> {
        Err(Error { message: message.into(), line, column })
    }


    // -- whitespace and comments --------------------------------------------

    fn skip_ws(&mut self) -> Result<(), Error> {
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

    pub(crate) fn parse_document(mut self) -> Result<Document, Error> {
        self.skip_ws()?;
        if self.eof() {
            return self.err("unexpected end of input: a document must contain one value");
        }
        let mut frames: Vec<Frame> = Vec::new();

        // Each pass around `'value` parses the head of one value. A composite
        // pushes a frame and comes back around for its first child; a
        // completed value falls through to the reduction loop, which closes
        // every frame it finishes and decides what comes next.
        'value: loop {
            let c = self.ch().expect("every entry to 'value checks for eof");
            match c {
                '{' => {
                    let (line, column) = (self.line, self.col);
                    self.bump();
                    let at = self.nodes.len();
                    self.nodes.push(Node::Object { len: 0, end: 0 });
                    if self.object_member(line, column)? {
                        self.nodes[at] = Node::Object { len: 0, end: self.nodes.len() as u32 };
                    } else {
                        frames.push(Frame::Object { at, len: 0, line, column });
                        continue 'value;
                    }
                }
                '[' => {
                    let (line, column) = (self.line, self.col);
                    self.bump();
                    let at = self.nodes.len();
                    self.nodes.push(Node::Array { len: 0, end: 0 });
                    self.skip_ws()?;
                    match self.byte(0) {
                        None => return self.err_at("unterminated array", line, column),
                        Some(b']') => {
                            self.bump();
                            self.nodes[at] = Node::Array { len: 0, end: self.nodes.len() as u32 };
                        }
                        Some(_) => {
                            frames.push(Frame::Array { at, len: 0, line, column });
                            continue 'value;
                        }
                    }
                }
                '"' => {
                    let span = self.parse_string()?;
                    self.nodes.push(Node::Str(span));
                }
                '+' | '-' => self.parse_number()?,
                c if c.is_ascii_digit() => self.parse_number()?,
                c if is_id_start(c) => {
                    let (line, column) = (self.line, self.col);
                    let name = self.read_identifier();
                    self.skip_ws()?;
                    // The argument is taken greedily. The tokens that can
                    // start a value and those that can follow one are
                    // disjoint, so one token decides.
                    if let Some(c) = self.ch()
                        && Self::starts_value(c)
                    {
                        let span = self.intern_from_source(name);
                        frames.push(Frame::Tag { at: self.nodes.len() });
                        self.nodes.push(Node::Tag { name: span, end: 0 });
                        continue 'value;
                    }
                    // A bare symbol cannot be a tag's payload. Without this,
                    // a missing comma between two identifiers would silently
                    // denote one tagged value instead of being an error.
                    if matches!(frames.last(), Some(Frame::Tag { .. })) {
                        return self.err_at(
                            "a tag cannot be applied to a bare symbol",
                            line,
                            column,
                        );
                    }
                    let span = self.intern_from_source(name);
                    self.nodes.push(Node::Symbol(span));
                }
                c => return self.err(format!("unexpected character {c:?}")),
            }

            // A value just completed; close what it finishes.
            loop {
                match frames.last_mut() {
                    None => {
                        self.skip_ws()?;
                        if !self.eof() {
                            return self.err("trailing content after the top-level value");
                        }
                        return Ok(Document { nodes: self.nodes, strings: self.strings });
                    }
                    Some(Frame::Tag { at }) => {
                        let at = *at;
                        frames.pop();
                        let Node::Tag { name, .. } = self.nodes[at] else {
                            unreachable!("a tag frame points at a tag node")
                        };
                        self.nodes[at] = Node::Tag { name, end: self.nodes.len() as u32 };
                    }
                    Some(Frame::Array { at, len, line, column }) => {
                        *len += 1;
                        self.skip_ws()?;
                        match self.byte(0) {
                            Some(b',') => {
                                self.bump(); // a following ']' is a legal trailing comma
                                self.skip_ws()?;
                                match self.byte(0) {
                                    None => return self.err_at("unterminated array", *line, *column),
                                    Some(b']') => {}
                                    Some(_) => continue 'value,
                                }
                            }
                            Some(b']') => {}
                            None => return self.err_at("unterminated array", *line, *column),
                            Some(_) => {
                                return self.err(
                                    "expected ',' or ']': only an identifier may be followed by a value",
                                );
                            }
                        }
                        self.bump(); // the ']'
                        let (at, len) = (*at, *len);
                        frames.pop();
                        self.nodes[at] = Node::Array { len, end: self.nodes.len() as u32 };
                    }
                    Some(Frame::Object { at, len, line, column }) => {
                        *len += 1;
                        self.skip_ws()?;
                        let closed = match self.byte(0) {
                            Some(b',') => {
                                self.bump(); // a following '}' is a legal trailing comma
                                self.object_member(*line, *column)?
                            }
                            Some(b'}') => {
                                self.bump();
                                true
                            }
                            None => return self.err_at("unterminated object", *line, *column),
                            Some(_) => {
                                return self.err(
                                    "expected ',' or '}': only an identifier may be followed by a value",
                                );
                            }
                        };
                        if !closed {
                            continue 'value;
                        }
                        let (at, len) = (*at, *len);
                        frames.pop();
                        self.nodes[at] = Node::Object { len, end: self.nodes.len() as u32 };
                    }
                }
            }
        }
    }

    // -- values -------------------------------------------------

    /// The tokens a value can start with.
    fn starts_value(c: char) -> bool {
        matches!(c, '{' | '[' | '"' | '+' | '-') || c.is_ascii_digit() || is_id_start(c)
    }

    /// Copy a slice of the source into the arena. Split out so the borrow of
    /// `self.src` ends before `self.strings` is touched.
    fn intern_from_source(&mut self, text: &str) -> Span {
        let start = self.strings.len() as u32;
        let len = text.len() as u32;
        // SAFETY-free version: `text` always points into `self.src`, which is
        // not the buffer being appended to.
        self.strings.push_str(text);
        Span { start, len }
    }

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

    fn parse_key(&mut self) -> Result<Span, Error> {
        let c = self.ch().expect("caller checked for eof");
        if c == '"' {
            return self.parse_string();
        }
        if is_id_start(c) {
            // An identifier key denotes the string of its spelling.
            let name = self.read_identifier();
            return Ok(self.intern_from_source(name));
        }
        self.err(format!("expected an object key, found {c:?}"))
    }

    /// After `{` or a member's `,`: either the object closes here, or the key
    /// and its `:` are consumed, the key is on the tape, and a value follows.
    /// Duplicates are accepted and the last one wins; the tape keeps both, and
    /// resolution happens in whatever view is built on top.
    fn object_member(&mut self, line: usize, column: usize) -> Result<bool, Error> {
        self.skip_ws()?;
        if self.eof() {
            return self.err_at("unterminated object", line, column);
        }
        if self.byte(0) == Some(b'}') {
            self.bump();
            return Ok(true);
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
        self.nodes.push(Node::Key(key));
        Ok(false)
    }


    // -- strings ------------------------------------------------------------

    fn parse_string(&mut self) -> Result<Span, Error> {
        let (line, column) = (self.line, self.col);
        self.bump();
        let start = self.strings.len() as u32;

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
                let at = self.strings.len();
                self.strings.push_str(chunk);
                debug_assert!(self.strings.len() - at == chunk.len());
            }

            let Some(b) = self.byte(0) else {
                return self.err_at("unterminated string", line, column);
            };
            match b {
                b'"' => {
                    self.bump();
                    let len = self.strings.len() as u32 - start;
                    return Ok(Span { start, len });
                }
                b'\\' => {
                    self.bump();
                    let c = self.read_escape()?;
                    self.strings.push(c);
                }
                b if b < 0x20 => {
                    return self.err(format!("unescaped control character U+{b:04X} in string"));
                }
                _ => {
                    let c = self.bump();
                    self.strings.push(c);
                }
            }
        }
    }

    fn read_escape(&mut self) -> Result<char, Error> {
        if self.eof() {
            return self.err("unterminated escape sequence");
        }
        let c = self.bump();
        let simple = match c {
            '"' => Some('"'),
            '\\' => Some('\\'),
            '/' => Some('/'),
            'b' => Some('\u{8}'),
            'f' => Some('\u{c}'),
            'n' => Some('\n'),
            'r' => Some('\r'),
            't' => Some('\t'),
            _ => None,
        };
        if let Some(c) = simple {
            return Ok(c);
        }
        if c != 'u' {
            return self.err(format!("invalid escape sequence '\\{c}'"));
        }

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
        char::from_u32(high)
            .map(Ok)
            .unwrap_or_else(|| self.err(format!("unpaired surrogate escape U+{high:04X}")))
    }

    fn read_hex4(&mut self) -> Result<u32, Error> {
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

    fn parse_number(&mut self) -> Result<(), Error> {
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

        let text = &self.src[start..self.pos];
        if exact && let Ok(i) = text.parse::<i64>() {
            self.nodes.push(Node::Int(i));
            return Ok(());
        }
        match text.parse::<f64>() {
            // An overflow to infinity would produce a value with no spelling:
            // `Infinity` lexes as a symbol, not a number, so a parsed infinity
            // could never be written back out.
            Ok(f) if f.is_finite() => {
                self.nodes.push(Node::Float(f));
                Ok(())
            }
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
