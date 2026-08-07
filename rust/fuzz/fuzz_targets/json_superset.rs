//! Section 5 as a fuzz target: every JSON text must be a valid sop document
//! denoting the same value.
//!
//! `serde_json` is the oracle. If it accepts an input and sop rejects it, the
//! superset claim is broken. If both accept, the values must agree.

#![no_main]

use libfuzzer_sys::fuzz_target;

/// Compare a serde_json value against a parsed sop document structurally.
fn agrees(j: &serde_json::Value, s: sop::Ref<'_>) -> bool {
    use serde_json::Value as J;
    match (j, s.kind()) {
        // The core has no bool or null: they are symbols like any other, and
        // giving them meaning is the consumer's job.
        (J::Null, sop::Kind::Symbol) => s.symbol() == Some("null"),
        (J::Bool(a), sop::Kind::Symbol) => {
            s.symbol() == Some(if *a { "true" } else { "false" })
        }
        // `arbitrary_precision` keeps the literal as written, which matters:
        // serde_json's own float conversion is lossy for integer literals too
        // large for u64 (it disagreed with std by ~15 ulp on an 86-digit
        // number, and std was right). Compare the literals through std instead,
        // so the oracle is structural and not numeric.
        (J::Number(a), sop::Kind::Number) => {
            a.as_str().parse::<f64>().ok() == s.as_f64()
        }
        (J::String(a), sop::Kind::String) => Some(a.as_str()) == s.as_str(),
        (J::Array(a), sop::Kind::Array) => {
            a.len() == s.len() && a.iter().zip(s.array()).all(|(x, y)| agrees(x, y))
        }
        (J::Object(a), sop::Kind::Object) => {
            // serde_json drops earlier duplicates; sop keeps them on the tape
            // and resolves on the way out (4.4), so compare on lookup.
            a.iter().all(|(k, v)| s.get(k).map(|w| agrees(v, w)).unwrap_or(false))
        }
        _ => false,
    }
}

fuzz_target!(|data: &[u8]| {
    let Ok(src) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(src) else {
        return; // not JSON; section 5 says nothing about it
    };

    // Section 5 has exactly one permitted exception, and the target asserts
    // that it is the only one: a literal outside the binary64 range. RFC 8259
    // section 6 allows a JSON parser to set limits on range and precision, so
    // this stays inside the JSON spec even though it leaves sop's section 2.6
    // ("no restriction is placed on range or precision") behind.
    let doc = match sop::Document::parse(src) {
        Ok(d) => d,
        Err(e) if e.message.contains("out of range") => return,
        Err(e) => panic!("section 5 violated: serde_json accepted {src:?}, sop said {e}"),
    };
    assert!(agrees(&json, doc.root()), "value mismatch on {src:?}");
});
