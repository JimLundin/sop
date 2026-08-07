//! Tests that need no Python side.

/// xorshift64*, so the tests are reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[test]
fn serialised_output_is_a_fixed_point() {
    const ALPHABET: [&str; 30] = [
        "{", "}", "[", "]", ",", ":", "\"", "'", "/", "*", "\\", "+", "-", ".", "e", "0", "1",
        " ", "\t", "\n", "\u{2028}", "\u{feff}", "\u{200c}", "\u{0}", "é", "😀", "\\u", "\\ud800",
        "a", "true",
    ];
    let mut rng = Rng(0xC0FFEE);
    let mut accepted = 0;
    for _ in 0..200_000 {
        let n = rng.next() % 24;
        let src: String = (0..n)
            .map(|_| ALPHABET[(rng.next() % ALPHABET.len() as u64) as usize])
            .collect();
        // The contract is Ok or Err, never a panic and never a hang.
        if let Ok(doc) = sop::Document::parse(&src) {
            accepted += 1;
            let text = doc.to_string();
            let again = sop::Document::parse(&text).expect("output re-parses");
            assert_eq!(again.to_string(), text, "not a fixed point: {src:?}");
        }
    }
    assert!(accepted > 1000, "the alphabet stopped producing valid documents");
}

#[test]
fn nesting_is_unbounded() {
    // The parser and the writer are iterative, so a million levels cost heap,
    // not stack, and the depth limit the recursive versions needed is gone.
    let n = 1_000_000;
    let src = format!("{}1{}", "[".repeat(n), "]".repeat(n));
    let doc = sop::Document::parse(&src).expect("parse");
    assert_eq!(doc.to_string(), src);
    // Stacked tags nest too, and look lexical rather than structural, so they
    // are the easy case to leave recursive by accident.
    let tags = "a ".repeat(n) + "1";
    let doc = sop::Document::parse(&tags).expect("parse");
    let text = doc.to_string();
    assert_eq!(sop::Document::parse(&text).expect("round trip").to_string(), text);
}

#[test]
fn large_objects_stay_linear() {
    // Duplicate-key handling is quadratic if done by scanning. The tape stores
    // members as written; parse and serialise must both resolve in linear time.
    let src: String = std::iter::once("{".to_string())
        .chain((0..64_000).map(|i| format!("k{i}:1,")))
        .chain(std::iter::once("}".to_string()))
        .collect();
    let start = std::time::Instant::now();
    let doc = sop::Document::parse(&src).expect("parse");
    assert_eq!(doc.root().len(), 64_000);
    assert!(start.elapsed().as_secs_f64() < 1.0, "parsing went quadratic");
    assert!(doc.to_string().len() > 500_000);
    assert!(start.elapsed().as_secs_f64() < 3.0, "serialising went quadratic");
}

#[test]
fn duplicate_keys_resolve_to_the_last() {
    // The tape keeps both members; `get` and the serialiser are where the
    // resolution happens, so both are checked here.
    let doc = sop::Document::parse("{a: 1, b: 2, a: 3}").expect("parse");
    assert_eq!(doc.root().len(), 3, "the tape keeps the document as written");
    assert_eq!(doc.root().get("a").and_then(|v| v.as_i64()), Some(3));
    assert_eq!(doc.to_string(), "{a:3,b:2}");
}

#[test]
fn accessors_return_none_for_the_wrong_kind() {
    // `Ref`'s accessors are `Option` because a caller can ask a string for its
    // tag. Nothing inside the crate does that, so without this the API's whole
    // misuse surface would be untested.
    let doc = sop::Document::parse(r#"[1, "s", Sym, tag 1, [], {}, 1.5]"#).unwrap();
    let items: Vec<sop::Ref<'_>> = doc.root().array().collect();
    let (number, string, symbol, tagged, array, object, float) = (
        items[0], items[1], items[2], items[3], items[4], items[5], items[6],
    );

    assert_eq!(number.as_i64(), Some(1));
    assert_eq!(float.as_i64(), None); // a float is not an integer literal
    assert_eq!(float.as_f64(), Some(1.5));
    assert_eq!(number.as_f64(), Some(1.0)); // an integer literal is still a number
    assert_eq!(string.as_f64(), None);

    assert_eq!(string.as_str(), Some("s"));
    assert_eq!(symbol.as_str(), None); // a symbol is not a string
    assert_eq!(symbol.symbol(), Some("Sym"));
    assert_eq!(string.symbol(), None);

    assert_eq!(tagged.tag(), Some("tag"));
    assert_eq!(tagged.payload().unwrap().as_i64(), Some(1));
    assert_eq!(number.tag(), None);
    assert!(number.payload().is_none());

    assert_eq!(number.len(), 0);
    assert!(array.is_empty() && object.is_empty());
    assert_eq!(number.array().count(), 0);
    assert_eq!(number.object().count(), 0);
    assert!(number.get("x").is_none());

    assert_eq!(doc.root().kind(), sop::Kind::Array);
}

#[test]
fn pretty_printing_is_presentation_only() {
    let src = r#"{a: [1, {b: tag "x"}], c: {}, d: []}"#;
    let doc = sop::Document::parse(src).unwrap();
    let pretty = doc.to_string_pretty(2);
    assert!(pretty.contains('\n'));
    assert_eq!(
        sop::Document::parse(&pretty).unwrap().to_string(),
        doc.to_string()
    );
    // An indent of zero would produce compact output, which `to_string` is for.
    assert!(doc.to_string_pretty(0).contains('\n'));
}

#[test]
fn is_identifier_matches_the_grammar() {
    for name in ["a", "_", "$", "a1", "été", "_$x", "if", "true"] {
        assert!(sop::is_identifier(name), "{name}");
    }
    for name in ["", "1a", "a b", "a,b", "a.b", " a", "\u{200C}a"] {
        assert!(!sop::is_identifier(name), "{name}");
    }
}

#[test]
fn numbers_the_domain_cannot_hold_are_rejected() {
    for src in ["1e400", "-1e400", &"9".repeat(400)] {
        let err = sop::Document::parse(src).unwrap_err();
        assert!(err.message.contains("out of range"), "{src}: {}", err.message);
    }
}

#[test]
fn built_documents_are_indistinguishable_from_parsed_ones() {
    // Everything the writer knows -- escaping, identifier keys, padding, tag
    // spacing -- applies to a built tape exactly as to a parsed one.
    let mut doc = sop::Builder::new();
    doc.tag("secret");
    doc.begin_object();
    doc.key("id");
    doc.begin_array();
    doc.int(1);
    doc.float(2.5);
    doc.string("x \"y\"");
    doc.symbol("true");
    doc.end_array();
    doc.key("weird key");
    doc.symbol("Active");
    doc.end_object();
    let text = doc.finish().to_string();
    assert_eq!(text, r#"secret {id:[1,2.5,"x \"y\"",true],"weird key":Active}"#);
    assert_eq!(sop::Document::parse(&text).expect("round trip").to_string(), text);
}
