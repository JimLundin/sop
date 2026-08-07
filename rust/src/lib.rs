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
        // The tape indexes nodes and text with u32, so a larger document has
        // no representation; refusing it up front keeps the contract that any
        // input either parses or returns an `Error`. A parse within the limit
        // cannot overflow: every node and every decoded byte consumes at
        // least one source byte.
        if src.len() > u32::MAX as usize {
            return Err(Error { message: "document exceeds 4 GiB".to_string(), line: 1, column: 1 });
        }
        parser::Parser::new(src).parse_document()
    }
}
