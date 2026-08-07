//! The parsed form: a flat tape, not a tree.
//!
//! A `Document` is two allocations — a `Vec<Node>` in depth-first order and one
//! `String` holding every decoded string, symbol, tag and key back to back.
//! Nothing is boxed and nothing is per-node allocated, so parsing is close to a
//! memcpy plus a scan, and handing the result across an FFI boundary is a
//! matter of exposing two slices.
//!
//! Composite nodes store `end`, the index one past the last node of their
//! subtree, which makes sibling traversal O(1) without a parent stack.

use crate::Span;
use crate::parser::is_identifier;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Number,
    String,
    Symbol,
    Tagged,
    Array,
    Object,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Node {
    Int(i64),
    Float(f64),
    Str(Span),
    Symbol(Span),
    /// An object key. Always immediately followed by its value's subtree.
    Key(Span),
    /// The payload subtree starts at this node's index + 1.
    Tag { name: Span, end: u32 },
    Array { len: u32, end: u32 },
    Object { len: u32, end: u32 },
}

impl Node {
    /// Index one past this node's subtree.
    fn end(&self, at: usize) -> usize {
        match self {
            Node::Tag { end, .. } | Node::Array { end, .. } | Node::Object { end, .. } => {
                *end as usize
            }
            _ => at + 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Document {
    pub(crate) nodes: Vec<Node>,
    pub(crate) strings: String,
}

impl Document {
    pub fn root(&self) -> Ref<'_> {
        Ref { doc: self, at: 0 }
    }

    pub(crate) fn span(&self, span: Span) -> &str {
        &self.strings[span.start as usize..span.start as usize + span.len as usize]
    }
}

/// A cursor into a `Document`. Copyable and the size of a pointer plus a
/// `usize`; walking a document allocates nothing.
#[derive(Debug, Clone, Copy)]
pub struct Ref<'a> {
    doc: &'a Document,
    at: usize,
}

impl<'a> Ref<'a> {
    fn node(&self) -> Node {
        self.doc.nodes[self.at]
    }

    pub fn kind(&self) -> Kind {
        match self.node() {
            Node::Int(_) | Node::Float(_) => Kind::Number,
            Node::Str(_) => Kind::String,
            Node::Symbol(_) => Kind::Symbol,
            Node::Tag { .. } => Kind::Tagged,
            Node::Array { .. } => Kind::Array,
            Node::Object { .. } => Kind::Object,
            Node::Key(_) => unreachable!("a key is never addressed as a value"),
        }
    }

    fn next_sibling(&self) -> usize {
        self.node().end(self.at)
    }

    /// The exact integer, when the literal was integer-shaped and fits.
    pub fn as_i64(&self) -> Option<i64> {
        match self.node() {
            Node::Int(i) => Some(i),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self.node() {
            Node::Int(i) => Some(i as f64),
            Node::Float(f) => Some(f),
            _ => None,
        }
    }

    /// The contents of a string. Borrowed from the document's arena.
    pub fn as_str(&self) -> Option<&'a str> {
        match self.node() {
            Node::Str(span) => Some(self.doc.span(span)),
            _ => None,
        }
    }

    /// The spelling of a symbol.
    ///
    /// `true`, `false` and `null` are ordinary symbols here. What they mean is
    /// a decision for whoever is consuming the document — the core does not
    /// have host types to map them onto, and inventing three is not its job.
    pub fn symbol(&self) -> Option<&'a str> {
        match self.node() {
            Node::Symbol(span) => Some(self.doc.span(span)),
            _ => None,
        }
    }

    /// The tag name, for a tagged value.
    pub fn tag(&self) -> Option<&'a str> {
        match self.node() {
            Node::Tag { name, .. } => Some(self.doc.span(name)),
            _ => None,
        }
    }

    /// The payload of a tagged value.
    pub fn payload(&self) -> Option<Ref<'a>> {
        match self.node() {
            Node::Tag { .. } => Some(Ref { doc: self.doc, at: self.at + 1 }),
            _ => None,
        }
    }

    /// Element or member count, for arrays and objects.
    pub fn len(&self) -> usize {
        match self.node() {
            Node::Array { len, .. } | Node::Object { len, .. } => len as usize,
            _ => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn array(&self) -> ArrayIter<'a> {
        let remaining = match self.node() {
            Node::Array { len, .. } => len as usize,
            _ => 0,
        };
        ArrayIter { doc: self.doc, at: self.at + 1, remaining }
    }

    /// Members in the order written, **including duplicate keys**. Duplicates
    /// are legal, and the tape keeps the document as written rather than
    /// resolving during the parse — resolution costs a keyed lookup per member
    /// on the hot path, for a case that is almost always pathological. Folding
    /// this into a map gives the last-wins answer for free, as Python's `dict`
    /// does; a consumer that needs one value should use [`Ref::get`].
    pub fn object(&self) -> ObjectIter<'a> {
        let remaining = match self.node() {
            Node::Object { len, .. } => len as usize,
            _ => 0,
        };
        ObjectIter { doc: self.doc, at: self.at + 1, remaining }
    }

    /// The effective value for a key: the **last** occurrence.
    /// Linear scan; build a map if the object is large or queried often.
    pub fn get(&self, key: &str) -> Option<Ref<'a>> {
        self.object().filter(|(k, _)| *k == key).map(|(_, v)| v).last()
    }
}

pub struct ArrayIter<'a> {
    doc: &'a Document,
    at: usize,
    remaining: usize,
}

impl<'a> Iterator for ArrayIter<'a> {
    type Item = Ref<'a>;

    fn next(&mut self) -> Option<Ref<'a>> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let item = Ref { doc: self.doc, at: self.at };
        self.at = item.next_sibling();
        Some(item)
    }

}


pub struct ObjectIter<'a> {
    doc: &'a Document,
    at: usize,
    remaining: usize,
}

impl<'a> Iterator for ObjectIter<'a> {
    type Item = (&'a str, Ref<'a>);

    fn next(&mut self) -> Option<(&'a str, Ref<'a>)> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let key = match self.doc.nodes[self.at] {
            Node::Key(span) => self.doc.span(span),
            _ => unreachable!("object members alternate key and value"),
        };
        let value = Ref { doc: self.doc, at: self.at + 1 };
        self.at = value.next_sibling();
        Some((key, value))
    }

}


// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

/// Assembles a document that was never text — the way in for host values,
/// such as the Python objects the bindings serialise. The tape built here is
/// indistinguishable from a parsed one, so the writer, and every formatting
/// rule in it, exists once.
///
/// Structure is asserted, not returned: a key outside an object, a mismatched
/// `end_`, or a second root is a programmer error in the caller, not bad data.
pub struct Builder {
    nodes: Vec<Node>,
    strings: String,
    frames: Vec<Frame>,
}

enum Frame {
    /// `end` is finalised at this index when the payload completes.
    Tag(usize),
    Array(usize, u32),
    /// The bool: a key has been given and its value is pending.
    Object(usize, u32, bool),
}

impl Builder {
    pub fn new() -> Builder {
        Builder { nodes: Vec::new(), strings: String::new(), frames: Vec::new() }
    }

    fn intern(&mut self, text: &str) -> Span {
        let start = self.strings.len() as u32;
        self.strings.push_str(text);
        Span { start, len: text.len() as u32 }
    }

    /// A value is legal at the root, in an array, under a tag, or after a key.
    fn slot(&self) {
        if let Some(Frame::Object(_, _, pending)) = self.frames.last() {
            assert!(*pending, "an object member is a key, then its value");
        }
    }

    /// A value just completed: close the tags it finishes and count it.
    fn settle(&mut self) {
        loop {
            match self.frames.last_mut() {
                Some(Frame::Tag(at)) => {
                    let at = *at;
                    self.frames.pop();
                    let Node::Tag { name, .. } = self.nodes[at] else {
                        unreachable!("a tag frame points at a tag node")
                    };
                    self.nodes[at] = Node::Tag { name, end: self.nodes.len() as u32 };
                }
                Some(Frame::Array(_, len)) => {
                    *len += 1;
                    return;
                }
                Some(Frame::Object(_, len, pending)) => {
                    *len += 1;
                    *pending = false;
                    return;
                }
                None => return,
            }
        }
    }

    pub fn int(&mut self, value: i64) {
        self.slot();
        self.nodes.push(Node::Int(value));
        self.settle();
    }

    /// Finite only: a non-finite float has no spelling in the format.
    pub fn float(&mut self, value: f64) {
        assert!(value.is_finite(), "a non-finite float has no sop spelling");
        self.slot();
        self.nodes.push(Node::Float(value));
        self.settle();
    }

    pub fn string(&mut self, text: &str) {
        self.slot();
        let span = self.intern(text);
        self.nodes.push(Node::Str(span));
        self.settle();
    }

    /// `name` must be an identifier. `true`, `false` and `null` are written
    /// this way — to the core they are ordinary symbols.
    pub fn symbol(&mut self, name: &str) {
        assert!(is_identifier(name), "a symbol is an identifier");
        self.slot();
        let span = self.intern(name);
        self.nodes.push(Node::Symbol(span));
        self.settle();
    }

    /// Applies to the next value; tags stack.
    pub fn tag(&mut self, name: &str) {
        assert!(is_identifier(name), "a tag is an identifier");
        self.slot();
        self.frames.push(Frame::Tag(self.nodes.len()));
        let span = self.intern(name);
        self.nodes.push(Node::Tag { name: span, end: 0 });
    }

    pub fn begin_array(&mut self) {
        self.slot();
        self.frames.push(Frame::Array(self.nodes.len(), 0));
        self.nodes.push(Node::Array { len: 0, end: 0 });
    }

    pub fn end_array(&mut self) {
        let Some(Frame::Array(at, len)) = self.frames.pop() else {
            panic!("`end_array` closes the innermost `begin_array`");
        };
        self.nodes[at] = Node::Array { len, end: self.nodes.len() as u32 };
        self.settle();
    }

    pub fn begin_object(&mut self) {
        self.slot();
        self.frames.push(Frame::Object(self.nodes.len(), 0, false));
        self.nodes.push(Node::Object { len: 0, end: 0 });
    }

    pub fn end_object(&mut self) {
        let Some(Frame::Object(at, len, pending)) = self.frames.pop() else {
            panic!("`end_object` closes the innermost `begin_object`");
        };
        assert!(!pending, "the last key has no value");
        self.nodes[at] = Node::Object { len, end: self.nodes.len() as u32 };
        self.settle();
    }

    /// Any string; the writer quotes what is not an identifier.
    pub fn key(&mut self, name: &str) {
        match self.frames.last_mut() {
            Some(Frame::Object(_, _, pending)) if !*pending => *pending = true,
            _ => panic!("a key belongs inside an object, before its value"),
        }
        let span = self.intern(name);
        self.nodes.push(Node::Key(span));
    }

    /// The finished document. Every `begin_` and `tag` is closed by now, and
    /// a document holds exactly one value.
    pub fn finish(self) -> Document {
        assert!(self.frames.is_empty(), "every `begin_` and `tag` is closed before `finish`");
        let end = match self.nodes.first() {
            Some(node) => node.end(0),
            None => 0,
        };
        assert!(end != 0 && end == self.nodes.len(), "a document holds exactly one value");
        Document { nodes: self.nodes, strings: self.strings }
    }
}
