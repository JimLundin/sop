# sop

An implementation of the sop interchange format: a Rust core that parses, and a
Python SDK that calls into it.

There is no separate specification. **The format is what this implementation
accepts**, and the tests are where that is written down:

- `corpus.json` — 83 conformance cases, each an input and either its exact
  serialised output or the fact that it is rejected. Run through both the core
  and the SDK; the two reports must be byte-identical.
- `python/test_shapes.py` — how Python types map onto sop values, in both
  directions, including every rejection.
- `python/test_python.py` — the semantics that exist only on the Python side.
- `python/test_properties.py` — Hypothesis properties: round trips, robustness
  against arbitrary input, typed round trips per shape.
- `rust/tests/parse.rs` — unbounded nesting, duplicate keys, accessor
  behaviour.

A quick tour of the syntax is in `corpus.json`'s valid cases; every construct
the format has appears there with its canonical spelling.

## The core

`rust/` parses text into a `Document`: a flat tape, two allocations for the
whole document — a `Vec<Node>` in depth-first order and one string arena —
with nothing boxed per node. 200 MB/s to parse, and walking a parsed document
runs at 1.4 GB/s because reading it allocates nothing.

```rust
let doc = sop::Document::parse(text)?;
let root = doc.root();
for (key, value) in root.object() {
    match value.kind() {
        sop::Kind::Tagged => println!("{key}: {} …", value.tag().unwrap()),
        sop::Kind::Symbol => println!("{key}: {}", value.symbol().unwrap()),
        _ => {}
    }
}
```

The whole API is two ways in, a cursor, and a writer:

```rust
Document::parse(&str)                     -> Result<Document, Error>
Builder::{new, int, float, string, symbol, tag, key,
          begin_array/end_array, begin_object/end_object, finish}
Document::to_string() / to_string_pretty(indent)
Ref::{kind, as_i64, as_f64, as_str, symbol, tag, payload,
      len, array, object, get}
```

`Builder` assembles a document that was never text — the bindings feed Python
values through it — and the tape it produces is indistinguishable from a
parsed one, so every formatting rule lives in the one writer.

The parser and the writer are iterative, so nesting depth costs heap rather
than stack and there is no depth limit: any input either parses or returns an
`Error`.

There is no owned value tree, no equality relation, no JSON projection and no
linter in the core. It parses.

## The SDK

`python/` is `sop._core`, the Rust core as an extension module, plus a shape
layer. No pure-Python parser and no fallback. **Python 3.15 only** — the
extension is built against the interpreter it runs on, not the limited API.

Two functions.

```python
order  = sop.loads[Order](text)          # Order
events = sop.loads[list[Event]](text)    # list[Event]
raw    = sop.loads[Any](text)            # whatever is there

text = sop.dumps(order)
```

Reading takes a shape because text carries no type; writing does not, because
the object does. There is no shapeless `loads` — `loads[Any]` is the escape
hatch and it has to be written down.

The whole public API is six names: `loads`, `dumps`, `Symbol`, `Tagged`,
`SopError`, and `ShapeError` (a subclass of `SopError`, so one `except` covers
both).

The subscript is annotated `TypeForm[T]` (PEP 747), so shapes that are not
classes work too — `sop.loads[Order | None](text)` reveals as `Order | None`,
which the older `type[T]` spelling could never express.

A class is carried under its own name unless it says otherwise:

```python
@dataclass
class Deposit:                # carried as `Deposit { ... }`
    amount: Money
    from_: str                # reads the key `from`

@dataclass
class Account:
    __sop_tag__ = None        # carried as a bare `{ ... }`

class Iban(str):
    __sop_tag__ = "iban"      # carried as `iban "DE89…"`

Event = Deposit | Withdraw | Reversal
events = sop.loads[list[Event]](text)
```

Builtins are exempt from the default: `str` and `list` are how the format's own
types are spelled, not user classes waiting for a tag.

A tagged dataclass is a tagged object; anything else with a tag is a tagged
string, built with `cls(text)` and spelled with `str(obj)`. A type that cannot
be built from its own spelling supplies `__sop_parse__`:

```python
class Money(Decimal):    __sop_tag__ = "decimal"
class Id(UUID):          __sop_tag__ = "uuid"

class Instant(datetime):
    __sop_tag__ = "instant"
    @classmethod
    def __sop_parse__(cls, text): return cls.fromisoformat(text)
```

The SDK has no built-in opinion about what `Decimal` or `UUID` are called on
the wire. That is a schema decision and it belongs to the schema.

| Shape | Reads |
|---|---|
| `@dataclass class X` | an object, or `X { ... }` if the class is tagged |
| `list[T]` | an array |
| `set[T]` | `set [ ... ]` — a *tagged* array, not a bare one |
| `dict[str, V]` | an object with uniform values |
| `X \| None` | the value, or the symbol `null` |
| `A \| B \| C` | a discriminated union, keyed on the tag |
| `Enum` | a symbol, matched on its spelling |
| a class with `__sop_tag__` | `tag "…"`, built with `cls(text)`, spelled with `str(obj)` |
| `str` | a string, never a symbol |
| `sop.Tagged` | any tagged value, tag preserved |
| `Any` | whatever was there |

Errors name the path:

```
$.location.lat: expected a number, found a string
$[0]: unknown tag `Unknown`; expected one of `Deposit`, `Reversal`, `Withdraw`
```

The core reads and writes any depth; building or walking the Python value is
recursive, so a pathologically deep or cyclic value raises `RecursionError` at
the interpreter's own limit rather than a limit of the SDK's.

The core has no booleans and no null. `true`, `false` and `null` are ordinary
symbols to it; mapping them onto `True`, `False` and `None` is the SDK's
decision, made in the bindings. `Symbol` and `Tagged` are native types and
validate on construction: a symbol must be an identifier, and `Symbol("null")`
is refused because `null` already has a spelling on this side. That makes them unable to hold a value the
parser could never produce, so the writer is total and Python `==` is the only
comparison you need. Comparing Python objects is Python's business.

Throughput is machine-dependent; on one box, the core parses at 200 MB/s and
the SDK at 38 MB/s, with the gap being the Python objects the SDK has to build.
Writing goes through the core directly for values that came from `loads`.

## Layout

```
build.sh             builds the core and the extension, installs the extension
build_corpus.py      generates corpus.json
corpus.json          83 conformance cases

rust/src/document.rs the tape: Document, Node, Ref, iterators
rust/src/parser.rs   the scanner
rust/src/ser.rs      writing a document back out
rust/src/main.rs     the conformance runner
rust/src/bin/bench.rs a benchmark, which generates its own input
rust/tests/parse.rs  nesting, linearity, duplicate keys, accessors
rust/fuzz/           two cargo-fuzz targets

bindings/src/lib.rs  PyO3 bindings — marshalling only, no format logic

python/sop/__init__.py the public API: loads and dumps
python/sop/_shape.py  how Python types map onto sop values
python/sop/_core.pyi  type stubs for the extension
python/run_corpus.py  the conformance runner, SDK side
python/test_shapes.py     the shape language, both directions, and its errors
python/test_python.py     Python semantics: host types, native types, equality,
                          numeric limits, the write fast path
python/test_properties.py Hypothesis property tests: round trips, robustness,
                          typed round trips per shape
```

## Running things

```sh
./build.sh                                     # needs Rust and Python 3.15
cd rust && cargo test --release
cd python && python3.15 -m pytest
python3.15 -m mypy --python-version 3.15 python/sop
python3.15 -m coverage run --branch --source=sop -m pytest && python3.15 -m coverage report
cd rust && cargo llvm-cov --ignore-filename-regex 'bin/|main.rs' 
cd rust/fuzz && cargo +nightly fuzz run parse -- -max_total_time=300
```

Coverage: 100% branch on the Python package (174 tests), and near-total line
coverage on the Rust core (8 tests plus the corpus run); the uncovered core
lines are `unreachable!()` arms.

Conformance is `python3 python/run_corpus.py` against
`cargo run --release --bin sop`: both write the same report format and their
output must be byte-identical across all 83 cases.
