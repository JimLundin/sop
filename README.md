# sop

An implementation of the sop interchange format: a Rust extension module that
reads and writes, and a Python shape layer on top.

There is no separate specification. **The format is what this implementation
accepts**, and the tests are where that is written down:

- `tests/test_corpus.py` — 85 conformance cases, each an input and either its
  exact serialised output or the fact that it is rejected. The expected
  outputs are written by hand; the implementation is never used to produce
  them.
- `tests/test_shapes.py` — how Python types map onto sop values, in both
  directions, including every rejection.
- `tests/test_python.py` — the semantics that exist only on the Python side.
- `tests/test_properties.py` — Hypothesis properties: round trips, robustness
  against arbitrary input, typed round trips per shape.

A quick tour of the syntax is in `tests/test_corpus.py`'s valid cases; every
construct the format has appears there with its canonical spelling.

## The implementation

One Rust crate, one pass each way. `loads` is a recursive-descent parser that
builds Python objects directly as it lexes (`src/parse.rs`); `dumps` walks
Python objects and spells them directly as text (`src/write.rs`); the lexical
facts they share — what an identifier is, how a string is escaped, how a float
is spelled — live once, in `src/text.rs`. There is no intermediate document
tree, no owned value tree, and no Rust-facing API: the crate's library is
named `_core`, after the extension module it becomes.

Reader and writer know exactly one set of values — the immutable one the
reader produces. Everything else, a dataclass and a `dict` alike, reaches the
writer through the single `convert` hook, so there is one way into the writer
rather than two.

## The SDK

`python/` is `sop._core`, that extension module, plus a shape layer. No
pure-Python parser and no fallback. **Python 3.15 only** — the
extension is built against the interpreter it runs on, not the limited API.

Two functions.

```python
order = sop.loads[Order](text)  # Order
events = sop.loads[list[Event]](text)  # list[Event]
raw = sop.loads[Any](text)  # whatever is there

text = sop.dumps(order)
```

Reading takes a shape because text carries no type; writing does not, because
the object does. There is no shapeless `loads` — `loads[Any]` is the escape
hatch and it has to be written down. `Value` names the closed set untyped
reading can produce — `None | bool | int | float | str | Symbol | Tagged |
tuple[Value, ...] | frozendict[str, Value]` — so `loads[Value]` is the same
escape hatch with its result typed precisely.

Everything read is immutable. Mutation is opt-in: a shape such as `list[T]`
or `dict[str, V]` declares it, and the decoded result is freshly built.
Writing accepts the mutable counterparts — `dict`, `list`, `set` — and spells
them identically; the core knows only the immutable set, so those are frozen
by the shape layer on the way out, and a value that came from `loads` is
written without touching Python at all.

The whole public API is seven names: `loads`, `dumps`, `Value`, `Symbol`,
`Tagged`, `SopError`, and `ShapeError` (a subclass of `SopError`, so one
`except` covers both).

The subscript is annotated `TypeForm[T]` (PEP 747), so shapes that are not
classes work too — `sop.loads[Order | None](text)` reveals as `Order | None`,
which the older `type[T]` spelling could never express.

A class is carried under its own name unless it says otherwise:

```python
@dataclass
class Deposit:  # carried as `Deposit { ... }`
    amount: Money
    from_: str = field(metadata={"sop": "from"})  # reads the key `from`


@dataclass
class Account:
    __sop_tag__ = None  # carried as a bare `{ ... }`


class Iban(str):
    __sop_tag__ = "iban"  # carried as `iban "DE89…"`


Event = Deposit | Withdraw | Reversal
events = sop.loads[list[Event]](text)
```

Builtins are exempt from the default: `str` and `list` are how the format's own
types are spelled, not user classes waiting for a tag. The name default
carries *dataclasses*, whose fields are declared; any other class is carried
only if it names a tag explicitly — there is no declared way to spell an
arbitrary object, so one that never opted in fails loudly instead of being
written through `str`.

A tagged dataclass is a tagged object; anything else with a tag is a tagged
string, built with `cls(text)` and spelled with `str(obj)`. A type that cannot
be built from its own spelling supplies `__sop_parse__`:

```python
class Money(Decimal):
    __sop_tag__ = "decimal"


class Id(UUID):
    __sop_tag__ = "uuid"


class Instant(datetime):
    __sop_tag__ = "instant"

    @classmethod
    def __sop_parse__(cls, text):
        return cls.fromisoformat(text)
```

The SDK has no built-in opinion about what `Decimal` or `UUID` are called on
the wire. That is a schema decision and it belongs to the schema.

| Shape | Reads |
|---|---|
| `@dataclass class X` | an object, or `X { ... }` if the class is tagged |
| `list[T]` | an array |
| `tuple[T, ...]` | an array, like `list[T]`, read back immutable |
| `set[T]` | `set [ ... ]` — a *tagged* array, not a bare one |
| `frozenset[T]` | `set [ ... ]`, like `set[T]`, read back immutable |
| `dict[str, V]` | an object with uniform values |
| `frozendict[str, V]` | an object, like `dict[str, V]`, read back immutable |
| `X \| None` | the value, or the symbol `null` |
| `A \| B \| C` | a discriminated union, keyed on the tag |
| `Enum` | a symbol, matched on its spelling |
| a class with `__sop_tag__` | `tag "…"`, built with `cls(text)`, spelled with `str(obj)` |
| `str` | a string, never a symbol |
| `sop.Tagged` | any tagged value, tag preserved |
| `Any` | whatever was there |

Number kind is spelling-determined — digits alone denote an integer, a point
or an exponent a float — and writing preserves it, the sign of `-0.0`
included.

Errors name the path:

```
$.location.lat: expected a number, found a string
$[0]: unknown tag `Unknown`; expected one of `Deposit`, `Reversal`, `Withdraw`
```

Reading and writing are both recursive, guarded by CPython's own recursion
check, so a pathologically deep or cyclic value raises `RecursionError` at
the interpreter's own limit rather than a limit of the SDK's.

The format has no booleans and no null. `true`, `false` and `null` are
ordinary symbols on the wire; mapping them onto `True`, `False` and `None` is
the SDK's decision, made in the parser. `Symbol` and `Tagged` are native types and
validate on construction: a symbol must be an identifier, `Symbol("null")` is
refused because `null` already has a spelling on this side, and a `Tagged`
cannot hold `None`, a bool or a `Symbol` — a tag cannot be applied to a bare
symbol, so such a value could never be written or read back. Those two say so
themselves, with a plain `ValueError`; the SDK does not restate the rule or
re-wrap the error. That makes them
unable to hold a value the parser could never produce, so the writer is total
and Python `==` is the only comparison you need. Comparing Python objects is
Python's business.

Throughput is machine-dependent; on one box, `loads` runs at ~35 MB/s and
`dumps` of what it produced at ~40 MB/s, the cost dominated by the Python
objects involved either way. Writing a plain `dict` or `list` is slower —
each one costs a hop through `convert` and the frozen copy it returns — which
is the price of the writer having one entrance rather than two.

## Layout

```
Cargo.toml           one crate, whose cdylib is the extension module
pyproject.toml       the Python package, built by maturin
.cargo/config.toml   defaults PYO3_PYTHON to python3.15

src/lib.rs           the module: Symbol, Tagged, SopError, loads, dumps
src/parse.rs         reading text into Python objects, one pass
src/write.rs         writing Python objects out as text, one traversal
src/text.rs          shared lexical facts: identifiers, escapes, number spelling

python/sop/__init__.py the public API: loads and dumps
python/sop/_shape.py  how Python types map onto sop values
python/sop/_core.pyi  type stubs for the extension, kept by hand
python/sop/py.typed   PEP 561 marker, so checkers read the stubs

tests/test_corpus.py      the 85 conformance cases
tests/test_shapes.py      the shape language, both directions, and its errors
tests/test_python.py      Python semantics: host types, native types, equality,
                          numeric limits, the single-traversal write path
tests/test_properties.py  Hypothesis property tests: round trips, robustness,
                          typed round trips per shape
```

## Running things

```sh
uv venv --python 3.15 && . .venv/bin/activate  # needs Rust and Python 3.15
uv pip install --group dev                     # the toolchain below
maturin develop --release                      # build into the active venv
uv pip install .                               # or: the wheel, via maturin build
pytest                                         # tests + branch coverage, 100% enforced
ruff check --fix . && ruff format .            # lint and format
mypy && pyright                                # both strict; configured in pyproject.toml
```

Coverage: 100% branch on the Python package, enforced by pytest itself
(`--cov-fail-under=100`); the Rust implementation is exercised end to end by
the pytest suite, the corpus and the Hypothesis properties — there is no
Rust-only surface left to test separately.
