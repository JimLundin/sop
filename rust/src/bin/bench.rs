//! Times parsing, walking and serialising. Generates its own input.
use std::time::Instant;

fn best<T>(mut f: impl FnMut() -> T) -> f64 {
    (0..5)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(f());
            t0.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

/// A tag-heavy document, in the shape of a typical API response.
fn generate(records: usize) -> String {
    let mut out = String::from("// generated benchmark corpus\n[\n");
    for i in 0..records {
        out.push_str(&format!(
            "  Account {{\n    \
             id: uuid \"9f1c2e7a-3b44-4f80-9c1d-{i:012}\",\n    \
             created_at: instant \"2026-08-05T14:23:11Z\",\n    \
             balance: decimal \"{}.{:02}\",\n    \
             location: geo {{ lat: {:.4}, lng: {:.4} }},\n    \
             roles: set [\"admin\", \"beta\", \"reader\"],\n    \
             status: Active,\n    \
             limits: [ +7, -2.5e10, 0.125, 9007199254740993 ],\n    \
             note: \"line \\u00e9 with \\\"escapes\\\" and \\\\ backslash\",\n    \
             secret: secret env \"TOKEN_{i}\",  // stacked tags\n    \
             deleted_at: null,\n  }},\n",
            i % 100_000,
            i % 100,
            (i % 180) as f64 - 90.0,
            (i % 360) as f64 - 180.0,
        ));
    }
    out.push_str("]\n");
    out
}

fn main() {
    let text = match std::env::args().nth(1) {
        Some(path) => std::fs::read_to_string(path).expect("read input"),
        None => generate(24_000),
    };
    let mb = text.len() as f64 / 1e6;

    let parse = best(|| sop::Document::parse(&text).expect("parse"));
    let doc = sop::Document::parse(&text).unwrap();
    let write = best(|| doc.to_string());
    let walk = best(|| {
        fn sum(node: sop::Ref<'_>) -> f64 {
            match node.kind() {
                sop::Kind::Number => node.as_f64().unwrap_or(0.0),
                sop::Kind::Array => node.array().map(sum).sum(),
                sop::Kind::Object => node.object().map(|(_, v)| sum(v)).sum(),
                sop::Kind::Tagged => sum(node.payload().unwrap()),
                _ => 0.0,
            }
        }
        sum(doc.root())
    });

    println!("parse      {mb:6.2} MB in {parse:6.3}s  ({:7.1} MB/s)", mb / parse);
    println!("walk       {mb:6.2} MB in {walk:6.3}s  ({:7.1} MB/s)", mb / walk);
    println!("serialise  {mb:6.2} MB in {write:6.3}s  ({:7.1} MB/s)", mb / write);
}
