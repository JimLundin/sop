# sop

An implementation of the sop interchange format: a Rust extension module that
reads and writes, and a Python shape layer on top.

There is no separate specification. **The format is what this implementation
accepts**, and the tests are where that is written down:

- `tests/test_corpus.py` — 91 conformance cases, each an input and either its
  exact serialised output or the fact that it is rejected. The expected
  outputs are written by hand; the implementation is never used to produce
  them.
- `tests/test_shapes.py` — how Python types map onto sop values, in both
  directions, including every rejection.
- `tests/test_python.py` — the semantics that exist only on the Python side.
- `tests/test_properties.py` — Hypothesis properties: round trips, robustness
  against arbitrary input, typed round trips per shape.
- `tests/test_soundness.py` — the shape language as a claim rather than a list
  of cases: a second, independent reading of what each shape means, Hypothesis
  driving arbitrary documents at arbitrary shapes to look for one that decodes
  into something the shape did not say, and the boundary of the language —
  what is a shape and what is refused — written out row by row.
- `tests/typing_corpus.py` — the same claim as a type checker sees it: every
  shape a user can write, what it reveals, and which checkers report what on
  it. `tests/test_typing.py` runs each of them over it and holds it to that.

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

Reading a shape happens in three stages, each with one job:

```
TypeForm  ──analyse──▶  Shape  ──build──▶  Decoder
what was written        the IR             what runs
```

`_ir.py` is the shape language written down as a type: a closed sum of frozen
dataclasses, so a `match` over it is exhaustive and `assert_never` proves it.
A shape cannot be added without the checkers naming every place that has to
answer for it, which is what keeps the two directions and the tests in step.

`_analyse.py` decides which `Shape` a `TypeForm` denotes, and is the only
thing that looks at Python's type language. `_decode.py` compiles a `Shape`
into the function that reads it — one closure per node, with every dispatch
decision already made — so decoding is calling a function per node rather than
matching a shape per value. The compiled decoder for a dataclass is kept on
the class, which is both the expensive part to work out and the part that has
to stay collectable.

The IR is a Python object graph and not a serialised one. Both halves of the
split live in Python — the core owns the value domain, not the shape language
— so there is no boundary for it to cross, and nothing to give up strict types
for.

Two functions.

```python
order = sop.loads[Order](text)  # Order
events = sop.loads[list[Event]](text)  # list[Event]
raw = sop.loads[Any](text)  # whatever is there

text = sop.dumps(order)
```

Reading takes a shape because text carries no type; writing does not, because
the object does. `loads` is one class carrying the shape it reads at, and the
shape it carries by default is `Value` — so `loads(text)` is not a shapeless
read, it is the read at the value domain, typed precisely. Turning checking
*off* is a separate decision and still has to be written down: `loads[Any]`
is the escape hatch.

The two are not inverses and are not meant to be: `dumps` takes nearly any
Python object, while untyped `loads` answers `Value` and nothing else.

Every sequence is written in array notation, and carries a tag saying what it
was — `Set [ ... ]`, `Roles [ ... ]` — so the shape of the data stays legible
even where nothing can read it back. `list` and `tuple` carry none: they
*are* the format's array.

`Value` is the core's own name for the closed set untyped reading can
produce — `None | bool | int | float | str | Symbol | Tagged | tuple[Value,
...] | frozendict[str, Value]` — so `loads[Value]` is the same escape hatch
with its result typed precisely. It is defined by the extension module, which
is what produces and accepts it, not by the layer above.

Everything read is immutable — not so that reading and writing mirror each
other, but so that what you are handed cannot change under you. Mutation is
opt-in: a shape such as `list[T]` or `dict[str, V]` declares it, and the
decoded result is freshly built.
Writing accepts the mutable counterparts — `dict`, `list`, `set` — and spells
them identically; the core knows only the immutable set, so those are frozen
by the shape layer on the way out, and a value that came from `loads` is
written without touching Python at all.

The whole public API is eight names: `loads`, `dumps`, `Value`, `Symbol`,
`Tagged`, and three errors. `SopError` is the base, so one `except` catches
everything; the two below it each carry the one location they have, and
neither invents the other's. `ParseError` has a `line` and `column` in the
text. `ShapeError` has a `path` into the value — `$.orders[1].customer` —
and is raised in both directions, whether the value read back does not fit
the shape or the object written has no sop spelling. The path is assembled
as the error leaves the traversal, so nothing pays for it until something
fails.

The subscript is annotated `TypeForm[T]` (PEP 747), so shapes that are not
classes work too — `sop.loads[Order | None](text)` reveals as `Order | None`,
which the older `type[T]` spelling could never express.

A carried class is tagged with its own name. That is the whole rule, and
nothing declares anything:

```python
@dataclass
class Deposit:  # carried as `Deposit { ... }`
    amount: Money
    from_: str = field(metadata={"sop": "from"})  # reads the key `from`


Event = Deposit | Withdraw | Reversal
events = sop.loads[list[Event]](text)
```

Builtins are exempt: `str` and `list` are how the format's own types are
spelled, not user classes waiting for a tag.

Three kinds of class are carried and none of them declare anything — a
dataclass, an enum, and five privileged classes, which are carried as tagged
strings:

| Class | Carried as |
|---|---|
| `Decimal` | `Decimal "19.99"` |
| `UUID` | `UUID "9f1c2e7a-…"` |
| `datetime` | `datetime "2026-08-05T14:23:11"` |
| `date` | `date "2026-08-05"` |
| `time` | `time "14:23:11"` |

Those five are a list of what the SDK knows how to build and spell, not a
list of names — the name is the class's own, here as everywhere else. They
are matched *exactly*: a subclass of one of them is not carried, and is
refused like any other class the SDK does not know. Subtypes are a thing to
add later.

The convention that a tag is a constructor spelled in PascalCase is a
convention: `datetime`, `date` and `time` are not, and are carried under
those names anyway. The format asks only that a tag be an identifier.

Everything else has no sop spelling and is refused in both directions. There
is no way for a class to opt in: no dunder, no registry, no decorator, no
global state. A class the SDK does not already know fails loudly rather than
being written through `str`, which would carry an object that declared
nothing as `Name "<object at 0x…>"`.

What that costs is the choice of *which* classes can be spelled at all,
which the SDK now makes for everyone. That is the price of the protocol
being a list one can read rather than a contract one has to know, and it is
meant to be paid back: user classes and subtypes are things to add later.

| Shape | Reads |
|---|---|
| `@dataclass class X` | `X { ... }` — an object under the class's own name |
| `list[T]` | an array |
| `tuple[T, ...]` | an array, like `list[T]`, read back immutable |
| `set[T]` | `Set [ ... ]` — an array tagged with what it was |
| `frozenset[T]` | `Set [ ... ]`, like `set[T]`, read back immutable |
| `dict[str, V]` | an object with uniform values |
| `frozendict[str, V]` | an object, like `dict[str, V]`, read back immutable |
| `X \| None` | the value, or the symbol `null` |
| `A \| B \| C` | a union, keyed on what the document says it is |
| `Enum` | a symbol, matched on its spelling |
| a privileged class | `Name "…"` — the five above, matched exactly |
| `str` | a string, never a symbol |
| `int` | a number spelled with digits alone |
| `float` | a number, read as a float however it was spelled |
| `bool` | the symbol `true` or the symbol `false` |
| `None` | the symbol `null` |
| `sop.Symbol` | a symbol, as itself |
| `sop.Tagged[V]` | a tagged value whose payload has shape `V`, tag preserved |
| `sop.Tagged` | the same with `V` defaulted — any payload |
| `sop.Value` | whatever was there, typed as the domain reading produces |
| `Any` | whatever was there |
| `type X = ...` | whatever the alias denotes |

Everything Python's type language has and this table does not — `Literal`,
`Annotated`, a `TypedDict`, a bare `list`, `tuple[int, str]`, a `Protocol` — is
refused the same way a document that does not fit is, with a `ShapeError`
naming the shape at the path it was asked for. So is a shape whose values
Python could not build: `set[list[int]]` is a set of lists, which a set cannot
hold. `tests/test_soundness.py` has the enumeration.

An object's keys are strings, so `dict[str, V]` is the shape that reads one;
`dict[Any, V]` is accepted and means the same thing. A key type that is
neither — `dict[int, V]` — is refused rather than silently coerced. That is a
limitation of the format as it stands and not a statement about what keys
ought to be; carrying other key types is a thing to add later, and the shape
language is where it would be lifted.

A union is read by what the document says it is, not by the order the
alternatives were written in. Every value on the wire carries a discriminant —
a tag, a symbol's spelling, or simply its kind — and that is what chooses:

```python
sop.loads[int | float]("1")  # 1,   an int
sop.loads[float | int]("1")  # 1,   the same int
sop.loads[int | float]("1.5")  # 1.5, a float
```

Number kind is spelling-determined everywhere, unions included, so `int |
float` and `float | int` are the same shape and read the same way. A shape
that *names* a value beats one that merely admits it: `float` reads an
integer-spelled number, but only where no `int` member is there to take it
first. The wildcards work the same way — `Symbol` takes any symbol and
`Tagged` any tag, so an enum or a dataclass that names one wins over them.

Two members that cannot be told apart are a schema error, raised when the
shape is first read rather than on whichever document happens to reach the
collision:

```
$: union members Deposit and Deposit cannot be told apart: both read a value tagged `Deposit`
$: union members list[int] and tuple[int, ...] cannot be told apart: both read an array
```

That is the whole rule, and it is why there is no order to learn: if a union
is ambiguous the format says so, and if it is not, the document decides.

A shape may contain itself. A dataclass whose field names it, a pair that name
each other, and a recursive `type` alias all read:

```python
@dataclass
class Node:
    name: str
    children: list["Node"]


type Tree = int | list[Tree]
```

The one thing refused is an alias that is directly one of its own
alternatives — `type Loop = Loop | int` — which has nothing to read before
reading itself again.

Unknown keys are ignored. A document may carry keys a dataclass does not
declare, and they are dropped rather than refused, so a reader built against
an older schema keeps working against a newer writer. That is the one place
the SDK is deliberately lax: everything else it does not recognise is an
error. A field the *reader* requires and the document lacks is still a
`ShapeError`, so this buys forward compatibility without giving up the
missing-key check.

Number kind is spelling-determined — digits alone denote an integer, a point
or an exponent a float — and writing preserves it, the sign of `-0.0`
included. A float is written plainly while its decimal exponent is in
`(-6, 21]` and with an exponent outside that, so `1e300` is three characters
of mantissa rather than three hundred of padding. Those are ECMAScript's
bounds, which RFC 8785 adopts as the canonical spelling of a number in a JSON
document; only the bounds are borrowed, and a positive exponent is written
without its `+`.

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
the SDK's decision, made in the parser. `Tagged` is generic in what it carries — `Tagged[int]` reads
`Retries 3` — and defaults to `Tagged[Value]`, which is what reading
produces. There is deliberately no bound on the parameter: the writer builds
a `Tagged[object]` whose payload is still unconverted. `Symbol` and `Tagged`
are native types and validate on construction: a symbol must be an identifier, `Symbol("null")` is
refused because `null` already has a spelling on this side, and a `Tagged`
cannot hold `None`, a bool or a `Symbol` — a tag cannot be applied to a bare
symbol, so such a value could never be written or read back. Those two say so
themselves, with a plain `ValueError`; the SDK does not restate the rule or
re-wrap the error. That makes them
unable to hold a value the parser could never produce, so the writer is total
and Python `==` is the only comparison you need. Comparing Python objects is
Python's business.

Throughput is machine-dependent and dominated by the Python objects involved
either way; on one box `loads` runs in the tens of MB/s. The one ratio worth
knowing is stable: a `dict` or `list` costs about **3x** what the same data
costs as a `frozendict` or `tuple`, because each container takes a hop
through `convert` and the frozen copy it returns. That is the price of the
writer having one entrance instead of two, and it is only paid by values the
writer did not produce.

Reading mirrors that. `loads[Any]` and `loads[Value]` cost nothing above the
parse: `Value` is exactly what the parser produces, so reading through it is a
check that cannot fail, and it answers the parsed value itself rather than
rebuilding an equal copy. Every other shape is compiled once and then run, so
what a read costs is the walk and not the shape.

`loads[Shape]` holds the compiled decoder, so `read = sop.loads[Order]` pays
for the shape once however many documents go through it. Writing
`sop.loads[Order](text)` inline compiles each time; the expensive part — a
dataclass's fields and their decoders — is kept on the class, so what is
rebuilt is the wrappers around it, and a one-off read is still cheaper than it
was before compiling.

A union no longer costs several times what its members cost. It is one lookup
in a table built when the shape was compiled, so where a value sits among the
alternatives does not matter: on a list of 2000 records the last of three
members reads as fast as the first.

## Layout

```
Cargo.toml           one crate, whose cdylib is the extension module
pyproject.toml       the Python package, built by maturin
.cargo/config.toml   defaults PYO3_PYTHON to python3.15
.github/workflows/   CI: every check under "Running things", on one runner

src/lib.rs           the module: Symbol, Tagged, the errors, loads, dumps
src/parse.rs         reading text into Python objects, one pass
src/write.rs         writing Python objects out as text, one traversal
src/text.rs          shared lexical facts: identifiers, escapes, number spelling

python/sop/__init__.py the public API: loads and dumps
python/sop/_ir.py      the shape language as a closed sum, and the value domain
python/sop/_analyse.py which shape a Python type denotes
python/sop/_decode.py  compiling a shape into the function that reads it
python/sop/_encode.py  how Python objects are spelled as sop values
python/sop/_core.pyi   type stubs for the extension, kept by hand
python/sop/py.typed    PEP 561 marker, so checkers read the stubs

tests/test_corpus.py      the 91 conformance cases
tests/test_shapes.py      the shape language, both directions, and its errors
tests/test_python.py      Python semantics: host types, native types, equality,
                          numeric limits, the single-traversal write path
tests/test_properties.py  Hypothesis property tests: round trips, robustness,
                          typed round trips per shape
tests/test_soundness.py   soundness of `loads[S] -> S`, and what the shape
                          language covers of Python's type language
tests/typing_corpus.py    every shape as a type checker sees it: what it
                          reveals, and where the checkers differ
tests/test_typing.py      runs mypy, pyright and pyrefly over that corpus
tests/conftest.py         the Hypothesis profiles the suite is run under
```

## Running things

```sh
uv venv --python 3.15 && . .venv/bin/activate  # needs Rust and Python 3.15
uv pip install --group dev                     # the toolchain below
maturin develop --release                      # build into the active venv
uv pip install .                               # or: the wheel, via maturin build

cargo fmt --check                              # the crate's own two gates,
cargo clippy --all-targets -- -D warnings      # pedantic, and warnings are errors
pytest                                         # tests + branch coverage, 100% enforced
pytest tests/test_soundness.py \
  --hypothesis-profile=deep --no-cov           # the soundness properties, deeper
ruff check --fix . && ruff format .            # lint and format
mypy                                           # strict; the gate, configured in pyproject.toml
stubtest sop._core --ignore-missing-stub       # the hand-written stubs against the module
pyright                                        # strict too, but advisory (see below)
```

Every one of those is a gate in CI (`.github/workflows/ci.yml`), pyright
excepted. The Rust checks run before the release build, so a formatting slip
is reported without waiting on a compile.

`stubtest` is there because `python/sop/_core.pyi` is written by hand and
nothing else compares it to the module it describes -- a stub that claims
less than the runtime enforces is a check that passes on code which crashes.

### Type checkers

`loads[S]` is a subscript whose argument is a type expression, and PEP 747 is
what makes that legal: the checker converts the expression to a `TypeForm[S]`
and solves `S` from it. That conversion is new and the checkers are not in the
same place with it, so the table below is measured rather than remembered —
`tests/typing_corpus.py` is one file of every shape a user can write and
`tests/test_typing.py` runs each checker over it on every test run.

| the shape as written | mypy | pyright | pyright + experimental | pyrefly |
|---|---|---|---|---|
| a class, `list[int]`, `sop.Tagged[int]` | yes | yes | yes | yes |
| `None` | yes | no | no | yes |
| `Order \| None`, `A \| B` | yes | no | no | yes |
| `type X = ...`, `sop.Value` | yes | no | no | no |
| any of those passed to a function | yes | no | yes | yes |

pyright applies the conversion to the argument of a *call* and not to the
argument of a *subscript*, in either configuration;
`enableExperimentalFeatures` turns on the call case and `TypeForm(...)` with
it, which is why only the last row moves. So somebody running pyright over
`sop.loads[Order | None](text)` is told `reportArgumentType` and handed an
`Unknown`. Nothing on this side fixes that — the same shape handed to a
function is accepted, so it is the syntax pyright has not reached yet and not
anything about the signature. The corpus pins it either way: the day pyright
reads a subscript, the expectation stops matching and this paragraph goes.

`ty` is not in the table because `typing.TypeForm` is not in its stdlib stubs
yet, so it cannot read the signature at all.

mypy is the gate, and the corpus is checked by it as strictly as the package
is. pyright is worth running and stays advisory: it reports five things about
`python/sop`, of which three are its bundled typeshed not yet having 3.15's
`frozendict` constructor — `pyright --typeshedpath` at a newer copy clears
them — and two are strict-mode lints, one about a `match` that is exhaustive
by a `raise` after it and one about `shape is None`, which pyright believes a
`TypeForm[Any]` cannot be and which the format's own `None` shape needs.

Coverage: 100% branch on the Python package, enforced by pytest itself
(`--cov-fail-under=100`); the Rust implementation is exercised end to end by
the pytest suite, the corpus and the Hypothesis properties — there is no
Rust-only surface left to test separately.

That is coverage of the code. Coverage of the *language* it implements is a
separate question, and `tests/test_soundness.py` is where it is answered: the
two tables above are held equal to the rows in that file by a test, so a shape
cannot be documented without being covered, nor a class join the privileged
five without appearing in all three places. The claim those tables make —
`loads[S]` answers an `S` — is checked there as a property rather than a list,
against a reading of each shape written independently of the SDK's, and CI runs
it a second time under a deeper Hypothesis profile.
