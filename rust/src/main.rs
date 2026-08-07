//! Runs corpus.json against the core.
//!
//! stdout is a deterministic report, one line per case, so that it can be
//! diffed against the Python SDK's runner: both entry points must produce
//! byte-identical output. Mismatches against the corpus's own expectations go
//! to stderr and set a non-zero exit status.

use std::process::ExitCode;

use sop::{Document, Ref};

fn field<'a>(node: Ref<'a>, key: &str) -> Ref<'a> {
    node.get(key).unwrap_or_else(|| panic!("missing key {key}"))
}

fn text<'a>(node: Ref<'a>) -> &'a str {
    node.as_str().expect("a string")
}

fn main() -> ExitCode {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../corpus.json");
    let source = std::fs::read_to_string(path).expect("read corpus");
    // Every JSON text is a valid sop document, so the corpus reads
    // itself with the parser under test.
    let corpus = Document::parse(&source).expect("corpus parses as sop");
    let root = corpus.root();

    let mut report: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in field(root, "valid").array() {
        let name = text(field(case, "name"));
        let src = text(field(case, "src"));
        let expected = text(field(case, "out"));
        match Document::parse(src) {
            Err(e) => {
                report.push(format!("valid {name} err {}:{}: {}", e.line, e.column, e.message));
                failures.push(format!("valid {name}: unexpected error: {e}"));
            }
            Ok(doc) => {
                let actual = doc.to_string();
                report.push(format!("valid {name} ok {actual}"));
                if actual != expected {
                    failures.push(format!("valid {name}: expected {expected:?}, got {actual:?}"));
                } else if Document::parse(&actual).map(|d| d.to_string()).as_deref() != Ok(&actual) {
                    failures.push(format!("valid {name}: output is not a fixed point"));
                }
            }
        }
    }

    for case in field(root, "invalid").array() {
        let name = text(field(case, "name"));
        match Document::parse(text(field(case, "src"))) {
            Err(e) => {
                report.push(format!("invalid {name} err {}:{}: {}", e.line, e.column, e.message));
            }
            Ok(doc) => {
                report.push(format!("invalid {name} ok {}", doc.to_string()));
                failures.push(format!("invalid {name}: accepted, should have been rejected"));
            }
        }
    }

    println!("{}", report.join("\n"));
    for failure in &failures {
        eprintln!("FAIL {failure}");
    }
    eprintln!("core: {} cases, {} failures", report.len(), failures.len());

    if failures.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
