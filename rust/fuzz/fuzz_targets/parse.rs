//! Coverage-guided fuzzing of the parser.
//!
//! The contract under test: for *any* input, `parse` returns `Ok` or `Err` and
//! never panics, hangs, or overflows the stack. Anything that parses must then
//! survive a serialise/re-parse round trip unchanged, which turns the
//! serialiser into an oracle for the parser and vice versa.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Ok(src) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(doc) = sop::Document::parse(src) else {
        return;
    };

    // A parsed document always has a spelling, that spelling parses, and
    // serialising it again is a fixed point.
    let text = doc.to_string();
    let again = sop::Document::parse(&text).expect("serialised output re-parses");
    assert_eq!(again.to_string(), text, "round trip changed the value: {text:?}");

    // Indentation is presentation only.
    let pretty = doc.to_string_pretty(2);
    let from_pretty = sop::Document::parse(&pretty).expect("pretty re-parses");
    assert_eq!(from_pretty.to_string(), text);
});
