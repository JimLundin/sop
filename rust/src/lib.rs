//! sop — the sop interchange format.
//!
//! The core parses text into a [`Document`]: a flat tape, two allocations for
//! the whole document, nothing boxed per node. Read it through [`Ref`], which
//! borrows and allocates nothing.
//!
//! ```
//! let doc = sop::Document::parse(r#"{ id: uuid "9f1c", status: Active }"#)?;
//! let root = doc.root();
//! assert_eq!(root.get("id").unwrap().tag(), Some("uuid"));
//! assert_eq!(root.get("status").unwrap().symbol(), Some("Active"));
//! # Ok::<(), sop::Error>(())
//! ```

mod document;
mod parser;
mod ser;

pub use document::{ArrayIter, Builder, Document, Kind, ObjectIter, Ref};
pub use parser::{Error, is_identifier};

/// A byte range into a document's string arena.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    pub(crate) start: u32,
    pub(crate) len: u32,
}

impl Document {
    /// The grammar sets no bound on nesting and neither does the parser: it
    /// is iterative, so depth costs heap rather than stack, and any input
    /// either parses or returns an [`Error`].
    pub fn parse(src: &str) -> Result<Document, Error> {
        parser::Parser::new(src).parse_document()
    }
}
