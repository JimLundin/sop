"""Whether the shape language is sound, and how much of Python's type language
it covers.

`loads[T]` claims to answer a `T`.  The other modules check that claim case by
case; this one checks it *as* a claim, two ways.

Soundness: `conforms` below is a second, independent reading of what each
shape means, written here and calling into nothing the SDK provides -- the
SDK's own checker is the thing under test.  Hypothesis then drives arbitrary
documents at arbitrary shapes looking for the one thing that must never
happen: a document that decodes without raising, into a value that is not what
the shape said it would be.  Anything other than a `SopError` leaving `loads`
or `dumps` fails these tests too, since none of them catch anything else.

Coverage: the shape language is a subset of Python's type language, and a
subset only means something if its boundary is written down.  `SUPPORTED` and
`REFUSED` are that boundary, one row per construct, and the gates at the
bottom hold the README's two tables, the implementation's carrier table and
these rows to each other -- so a shape cannot be documented without being
covered here, nor a carrier added to the SDK without appearing in both.
"""

import dataclasses
import enum
import pathlib
import re
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    Protocol,
    TypedDict,
    TypeForm,
    cast,
)
from uuid import UUID

import pytest
import sop
from hypothesis import given
from hypothesis import strategies as st

# The value domain's own generator lives with the properties it was written
# for; the shapes below need it wherever a shape stops asking questions --
# `Any`, `sop.Value`, and a bare `sop.Tagged`'s payload.
from test_properties import identifiers, integers, values

# ---------------------------------------------------------------------------
# Schemas used below -- fixed classes, not built per example, so that the
# answers the SDK caches on a class are exercised rather than freshly computed
# every time.  Frozen, so that a set of them is a shape at all.
# ---------------------------------------------------------------------------


class Colour(enum.Enum):
    Red = "Red"
    Blue = "azul"  # value and name differ


class Priority(enum.Enum):
    Low = 1  # a non-string value, so the member's name is its spelling
    High = 2


@dataclass(frozen=True)
class Point:
    x: int
    y: float


@dataclass(frozen=True)
class Label:
    text: str
    note: str | None = None


@dataclass(frozen=True)
class Wrapper:  # a dataclass whose own fields are shapes
    items: tuple[int, ...]
    seen: frozenset[str]
    who: Point | Label


type Numbers = list[int]  # a `type` statement, which denotes its right-hand side

_CARRIED = {
    Decimal: st.decimals(allow_nan=False, allow_infinity=False),
    UUID: st.uuids(),
    datetime: st.datetimes(),
    date: st.dates(),
    time: st.times(),
}
"""The privileged classes and how to generate one of each.  Named here rather
than imported from the SDK: the gate at the bottom is what checks that this
list and the SDK's are the same list."""


# ---------------------------------------------------------------------------
# The oracle: what a shape means, read a second time
# ---------------------------------------------------------------------------


def conforms(value: object, shape: TypeForm[Any]) -> bool:
    """Whether `value` really is a `shape`, structurally and to the leaves.

    Deliberately a re-implementation.  Asking the SDK whether its own answer
    fits would prove nothing, so this walks the shape itself and knows only
    what the README says each one reads.

    Exact types, not `isinstance`, wherever the format draws a line Python
    does not: a bool is an `int` to Python and a symbol to sop, and a
    `datetime` is a `date` to Python and a different carrier to sop.  A
    subclass would satisfy the annotation and still be a value the format
    cannot mean, so the stricter reading is the one that has teeth."""
    if isinstance(shape, typing.TypeAliasType):
        if shape is sop.Value:
            return _in_the_value_domain(value)
        return conforms(value, shape.__value__)
    if shape is Any:
        return True
    if shape is None or shape is types.NoneType:
        return value is None

    origin, args = typing.get_origin(shape), typing.get_args(shape)
    if origin is types.UnionType:
        return any(conforms(value, member) for member in args)
    if origin is list or origin is tuple or origin is set or origin is frozenset:
        return type(value) is origin and all(
            conforms(item, args[0]) for item in typing.cast(Iterable[object], value)
        )
    if origin is dict or origin is frozendict:
        return type(value) is origin and all(
            type(key) is str and conforms(item, args[1])
            for key, item in typing.cast(Mapping[object, object], value).items()
        )
    if origin is sop.Tagged:
        return type(value) is sop.Tagged and conforms(value.value, args[0])
    if origin is not None:
        raise AssertionError(f"the oracle does not model {shape!r}")

    if shape is bool or shape is int or shape is float or shape is str:
        return type(value) is shape
    if shape is sop.Symbol:
        return type(value) is sop.Symbol
    if shape is sop.Tagged:
        # Bare, so the payload is the parameter's default: the value domain.
        return type(value) is sop.Tagged and _in_the_value_domain(value.value)
    if isinstance(shape, type) and issubclass(shape, enum.Enum):
        return isinstance(value, shape)
    if shape in _CARRIED:
        return type(value) is shape
    if dataclasses.is_dataclass(shape):
        hints = typing.get_type_hints(shape)
        return type(value) is shape and all(
            conforms(getattr(value, field.name), hints[field.name])
            for field in dataclasses.fields(shape)
            if field.init  # the rest are the class's own business, not decode's
        )
    raise AssertionError(f"the oracle does not model {shape!r}")


def _in_the_value_domain(value: object) -> bool:
    """Whether a value is one the parser could have produced -- which is what
    `sop.Value` names, and what a bare `sop.Tagged` carries."""
    if value is None or type(value) in (bool, int, float, str, sop.Symbol):
        return True
    if type(value) is sop.Tagged:
        return _in_the_value_domain(value.value)
    if type(value) is tuple:
        return all(_in_the_value_domain(item) for item in value)
    if type(value) is frozendict:
        return all(
            type(key) is str and _in_the_value_domain(item)
            for key, item in value.items()
        )
    return False


# ---------------------------------------------------------------------------
# Generating shapes, and values of a shape
# ---------------------------------------------------------------------------


def _can_carry_a_tag(value: object) -> bool:
    """A tag cannot be applied to a bare symbol, so `Tagged` refuses `None`, a
    bool and a `Symbol` as its payload -- and an enum member with them, which
    `Tagged` cannot see coming but the writer does, since a member is written
    as a symbol."""
    return not (value is None or isinstance(value, (bool, sop.Symbol, enum.Enum)))


def _taggable(shape: TypeForm[Any]) -> bool:
    """Whether any value of this shape could carry a tag.  `Tagged[bool]` and
    `Tagged[Colour]` name values that cannot exist -- an enum is carried as a
    symbol, and a symbol is what a tag cannot be applied to -- so they are
    shapes with nothing to generate, not shapes worth generating."""
    if shape is None or shape is types.NoneType or shape is bool:
        return False
    if shape is sop.Symbol:
        return False
    if typing.get_origin(shape) is types.UnionType:
        return any(map(_taggable, typing.get_args(shape)))
    return not (isinstance(shape, type) and issubclass(shape, enum.Enum))


_LEAVES: list[TypeForm[Any]] = [
    None,
    bool,
    int,
    float,
    str,
    sop.Symbol,
    sop.Tagged,
    Colour,
    Priority,
    Point,
    Label,
    Wrapper,
    Numbers,
    Any,
    sop.Value,
    *_CARRIED,
]

# Every one of these decodes to something a set can hold.  `set[list[int]]`
# does not -- neither directly nor through an alias, and neither does a set of
# a shape that stopped asking questions -- so those are refused shapes below
# rather than generated ones here.
_HASHABLE_LEAVES: list[TypeForm[Any]] = [
    shape for shape in _LEAVES if shape not in (Any, sop.Value, Numbers)
]

# Written out rather than generated from pairs of leaves: two members that
# share a tag are a schema error the SDK refuses by design, and a union of one
# arbitrary shape with another mostly draws that.
_UNIONS: list[TypeForm[Any]] = [
    Point | Label,
    Point | Label | Wrapper,
    Point | str,
    Colour | Point,
]


_KEYS = st.text(max_size=6)  # an object's keys, which are strings


def _at(origin: Any, *args: Any) -> TypeForm[Any]:
    """`origin[args]` -- `list[s]`, `dict[str, s]`, `tuple[s, ...]`.

    Written out as a subscript it would read as a type expression with a
    variable in it, which is not one: the shapes being combined here are
    values, drawn by Hypothesis, and a call is where both sides agree on
    that.  What it builds is the same object either spelling produces."""
    return cast(TypeForm[Any], origin[args[0] if len(args) == 1 else args])


def _optional(shape: object) -> bool:
    """Whether `shape | None` is a shape.  One that already reads `null` --
    `None` itself, and the two that read anything -- would be the other
    member worded twice, which the shape language refuses."""
    return shape is not None and shape is not Any and shape is not sop.Value


def _extend(
    inner: st.SearchStrategy[Any], hashable: st.SearchStrategy[Any]
) -> st.SearchStrategy[Any]:
    """Every way the shape language builds a shape out of shapes."""
    return (
        inner.map(lambda s: _at(list, s))
        | inner.map(lambda s: _at(tuple, s, ...))
        | inner.map(lambda s: _at(dict, str, s))
        | inner.map(lambda s: _at(frozendict, str, s))
        | hashable.map(lambda s: _at(set, s))
        | hashable.map(lambda s: _at(frozenset, s))
        # `None | None` is not a union and not a shape; nothing else is barred.
        | inner.filter(_optional).map(lambda s: s | None)
        | inner.filter(_taggable).map(lambda s: _at(sop.Tagged, s))
    )


def _hashable_extend(inner: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    """The subset of the above whose values a set can hold."""
    return (
        inner.map(lambda s: _at(tuple, s, ...))
        | inner.map(lambda s: _at(frozenset, s))
        | inner.map(lambda s: _at(frozendict, str, s))
        | inner.filter(_optional).map(lambda s: s | None)
        | inner.filter(_taggable).map(lambda s: _at(sop.Tagged, s))
    )


hashable_shapes = st.recursive(
    st.sampled_from(_HASHABLE_LEAVES), _hashable_extend, max_leaves=3
)

shapes = st.recursive(
    st.sampled_from(_LEAVES) | st.sampled_from(_UNIONS),
    lambda inner: _extend(inner, hashable_shapes),
    max_leaves=4,
)


def values_for(shape: TypeForm[Any]) -> st.SearchStrategy[Any]:
    """A strategy for values of `shape`: the generator side of `conforms`, and
    the same walk over the same language."""
    if isinstance(shape, typing.TypeAliasType):
        return values if shape is sop.Value else values_for(shape.__value__)
    if shape is Any:
        return values
    if shape is None or shape is types.NoneType:
        return st.none()

    origin, args = typing.get_origin(shape), typing.get_args(shape)
    if origin is types.UnionType:
        return st.one_of([values_for(member) for member in args])
    if origin is list:
        return st.lists(values_for(args[0]), max_size=3)
    if origin is tuple:
        return st.lists(values_for(args[0]), max_size=3).map(tuple)
    if origin is set:
        return st.sets(values_for(args[0]), max_size=3)
    if origin is frozenset:
        return st.frozensets(values_for(args[0]), max_size=3)
    if origin is dict:
        return st.dictionaries(_KEYS, values_for(args[1]), max_size=3)
    if origin is frozendict:
        return st.dictionaries(_KEYS, values_for(args[1]), max_size=3).map(frozendict)
    if origin is sop.Tagged:
        payloads = values_for(args[0]).filter(_can_carry_a_tag)
        return st.builds(sop.Tagged, identifiers, payloads)

    if shape is bool:
        return st.booleans()
    if shape is int:
        return integers
    if shape is float:
        return st.floats(allow_nan=False, allow_infinity=False)
    if shape is str:
        return st.text(max_size=8)
    if shape is sop.Symbol:
        return st.builds(sop.Symbol, identifiers)
    if shape is sop.Tagged:
        return st.builds(sop.Tagged, identifiers, values.filter(_can_carry_a_tag))
    if shape in _CARRIED:
        return _CARRIED[shape]
    if isinstance(shape, type) and issubclass(shape, enum.Enum):
        return st.sampled_from(list(shape))
    if dataclasses.is_dataclass(shape):
        hints = typing.get_type_hints(shape)
        return st.builds(
            shape,
            **{
                field.name: values_for(hints[field.name])
                for field in dataclasses.fields(shape)
                if field.init
            },
        )
    raise AssertionError(f"nothing generates a {shape!r}")


shaped_values = shapes.flatmap(lambda s: st.tuples(st.just(s), values_for(s)))

# Documents to read at a shape that was chosen without reference to them: sop
# text of every kind there is, well-typed for some other shape or not text of a
# value at all.
noise = st.text(
    alphabet=st.sampled_from(list("{}[](),:\"'/*\\+-.0123456789eE_$ \t\nSetPointRed")),
    max_size=40,
)
documents = (
    values.map(sop.dumps)
    | shapes.flatmap(values_for).map(sop.dumps)
    | noise
    | st.text(max_size=30)
)


# ---------------------------------------------------------------------------
# Soundness: what comes back is what the shape said
# ---------------------------------------------------------------------------


@given(shaped_values)
def test_a_value_of_a_shape_reads_back_as_that_shape(
    shaped: tuple[TypeForm[Any], Any],
) -> None:
    shape, value = shaped
    read = sop.loads[shape](sop.dumps(value))
    assert conforms(read, shape), f"{shape}: {read!r} is not a {shape}"
    assert read == value


@given(shapes, documents)
def test_decoding_is_sound(shape: TypeForm[Any], text: str) -> None:
    """The claim, stated as a property: whatever `loads[S]` answers is an `S`.

    A document that does not fit is a `ShapeError`, and text that is not a
    document is a `ParseError`; the only outcome ruled out is a value that
    decoded and is not what was asked for.  Nothing else is caught, so an
    exception the SDK does not name fails here as loudly as an unsound read."""
    try:
        read = sop.loads[shape](text)
    except sop.SopError:
        return
    assert conforms(read, shape), f"{shape}: {read!r} is not a {shape}"


@given(shapes, documents)
def test_reading_a_document_twice_is_the_same_read(
    shape: TypeForm[Any], text: str
) -> None:
    """Decoding has no state to carry between calls -- the answers the SDK
    caches on a class are about the class, not about the document."""
    try:
        once = sop.loads[shape](text)
    except sop.SopError as first:
        with pytest.raises(sop.SopError) as again:
            sop.loads[shape](text)
        # The same failure, at the same path, said the same way: the message a
        # union builds out of its members is sorted, not enumerated.
        assert str(again.value) == str(first)
        return
    assert sop.loads[shape](text) == once


@given(shaped_values)
def test_writing_is_total_over_the_shape_language(
    shaped: tuple[TypeForm[Any], Any],
) -> None:
    """Every value a shape can name has a spelling, and the spelling is the
    same every time it is asked for."""
    _, value = shaped
    assert sop.dumps(value) == sop.dumps(value)


KINDS = [
    "null",
    "true",
    "false",
    "3",
    "-3.5",
    '"s"',
    "Active",
    "[]",
    "[1]",
    "{}",
    "{ a: 1 }",
    "Set [1]",
    "Tag 1",
    "Point { x: 1, y: 2.5 }",
    'Decimal "1.5"',
]
"""One document of every kind the format has, canonical rather than drawn."""

MATRIX: list[TypeForm[Any]] = [
    *_LEAVES,
    *_UNIONS,
    list[int],
    tuple[int, ...],
    set[int],
    frozenset[str],
    dict[str, int],
    frozendict[str, int],
    int | None,
    sop.Tagged[int],
    list[Point | Label],
    dict[str, list[int]],
]


@pytest.mark.parametrize("text", KINDS)
@pytest.mark.parametrize("shape", MATRIX, ids=[str(shape) for shape in MATRIX])
def test_every_shape_reads_every_kind_of_document_soundly(
    shape: TypeForm[Any], text: str
) -> None:
    """The same claim as above, over a cross product rather than a search.

    The property drives both sides at once, which reaches the deep and the
    nested but pairs a shape with the document that would embarrass it only by
    luck; a bool read at `int`, or a symbol read at `str`, is one draw in
    thousands.  There are few enough kinds of document to try them all against
    every shape, and what is small enough to be exhaustive should not be left
    to a search."""
    try:
        read = sop.loads[shape](text)
    except sop.SopError:
        return
    assert conforms(read, shape), f"{shape}: {read!r} is not a {shape}"


@given(st.text(max_size=60) | noise)
def test_the_value_domain_is_total_over_the_parser(text: str) -> None:
    """`sop.Value` is the core's own account of what reading can produce, and
    `loads[Value]` is documented as a check that cannot fail.  This is that
    claim from the outside: whatever the parser produces is in the domain."""
    try:
        read = sop.loads[sop.Value](text)
    except sop.SopError:
        return
    assert _in_the_value_domain(read)


# ---------------------------------------------------------------------------
# Coverage: what is a shape
# ---------------------------------------------------------------------------

SUPPORTED: list[tuple[str, TypeForm[Any], str, object]] = [
    # label as the README spells it, shape, a document, what it reads as
    ("int", int, "3", 3),
    ("float", float, "3.5", 3.5),
    ("bool", bool, "true", True),
    ("str", str, '"a"', "a"),
    ("None", None, "null", None),
    ("sop.Symbol", sop.Symbol, "Active", sop.Symbol("Active")),
    ("@dataclass class X", Point, "Point { x: 1, y: 2.5 }", Point(1, 2.5)),
    ("list[T]", list[int], "[1, 2]", [1, 2]),
    ("tuple[T, ...]", tuple[int, ...], "[1, 2]", (1, 2)),
    ("set[T]", set[int], "Set [1, 2]", {1, 2}),
    ("frozenset[T]", frozenset[int], "Set [1]", frozenset({1})),
    ("dict[str, V]", dict[str, int], "{ a: 1 }", {"a": 1}),
    ("frozendict[str, V]", frozendict[str, int], "{ a: 1 }", frozendict({"a": 1})),
    ("X | None", int | None, "null", None),
    ("A | B | C", Point | Label, 'Label { text: "x" }', Label("x")),
    ("Enum", Colour, "azul", Colour.Blue),
    ("a privileged class", Decimal, 'Decimal "19.99"', Decimal("19.99")),
    ("a privileged class", UUID, f'UUID "{UUID(int=1)}"', UUID(int=1)),
    (
        "a privileged class",
        datetime,
        'datetime "2026-08-05T14:23:11"',
        datetime(2026, 8, 5, 14, 23, 11),
    ),
    ("a privileged class", date, 'date "2026-08-05"', date(2026, 8, 5)),
    ("a privileged class", time, 'time "14:23:11"', time(14, 23, 11)),
    ("sop.Tagged[V]", sop.Tagged[int], "Retries 3", sop.Tagged("Retries", 3)),
    ("sop.Tagged", sop.Tagged, 'Retries "x"', sop.Tagged("Retries", "x")),
    ("Any", Any, "[1]", (1,)),
    ("sop.Value", sop.Value, "[1]", (1,)),
    ("type X = ...", Numbers, "[1]", [1]),
]


@pytest.mark.parametrize(
    "shape, text, expected",
    [row[1:] for row in SUPPORTED],
    ids=[f"{row[0]}: {row[2]}" for row in SUPPORTED],
)
def test_a_supported_shape_reads_and_writes(
    shape: TypeForm[Any], text: str, expected: object
) -> None:
    read = sop.loads[shape](text)
    assert read == expected
    assert conforms(read, shape), f"{read!r} is not a {shape}"
    # And what it read writes back to something that reads the same way, which
    # is the only sense in which the two directions are inverses.
    assert sop.loads[shape](sop.dumps(read)) == expected


class _TypedDict(TypedDict):
    a: int


class _NamedTuple(NamedTuple):
    a: int


class _Protocol(Protocol):
    def f(self) -> None: ...


class _Plain:
    """A class the SDK does not know, which is every class not listed above."""


class _Money(Decimal):
    """A subclass of a privileged class, which is not itself privileged."""


REFUSED: list[tuple[str, Any, str]] = [
    # Python's type language reaches well past the format's, and every
    # construct that lands outside it lands the same way: a `ShapeError`
    # naming the shape, at the path where it was asked for.
    ("Annotated[T, ...]", Annotated[int, "meta"], "3"),
    ("Literal[...]", Literal[1, 2], "1"),
    ("tuple[T, U]", tuple[int, str], '[1, "a"]'),
    ("dict[K, V] with non-string keys", dict[int, str], "{}"),
    ("list", list, "[]"),
    ("dict", dict, "{}"),
    ("tuple", tuple, "[]"),
    ("set", set, "Set []"),
    ("frozenset", frozenset, "Set []"),
    ("set[T] a set cannot hold", set[list[int]], "Set [[1]]"),
    ("frozenset[T] a set cannot hold", frozenset[dict[str, int]], "Set [{}]"),
    ("Sequence[T]", Sequence[int], "[1]"),
    ("Iterable[T]", Iterable[int], "[1]"),
    ("Mapping[K, V]", Mapping[str, int], "{}"),
    ("Callable[..., T]", Callable[[int], int], "3"),
    ("TypedDict", _TypedDict, "{ a: 1 }"),
    ("NamedTuple", _NamedTuple, "[1]"),
    ("Protocol", _Protocol, "3"),
    ("type[T]", type[int], "3"),
    ("object", object, "3"),
    ("bytes", bytes, '"x"'),
    ("complex", complex, "3"),
    ("Never", typing.Never, "3"),
    ("LiteralString", typing.LiteralString, '"x"'),
    ("a class the SDK does not know", _Plain, "_Plain {}"),
    ("a subclass of a privileged class", _Money, '_Money "1"'),
    ("a string annotation", "int", "3"),
    ("something that is not a type at all", 42, "3"),
]


@pytest.mark.parametrize(
    "shape, text", [row[1:] for row in REFUSED], ids=[row[0] for row in REFUSED]
)
def test_a_refused_shape_is_refused_and_nothing_else(shape: Any, text: str) -> None:
    # A `ShapeError`, and not a `TypeError` or an `AttributeError` out of the
    # SDK's own introspection: a shape it does not understand is a document
    # error like any other, and says so at a path.
    with pytest.raises(sop.ShapeError):
        sop.loads[shape](text)


def _generator() -> Iterable[int]:
    yield 1


UNWRITABLE: list[tuple[str, object]] = [
    ("object", object()),
    ("bytes", b"x"),
    ("complex", 1j),
    ("range", range(3)),
    ("a generator", _generator()),
    ("a function", _generator),
    ("a class", Point),
    ("a class the SDK does not know", _Plain()),
    ("a subclass of a privileged class", _Money(1)),
    ("an object with non-string keys", {1: 2}),
]


@pytest.mark.parametrize(
    "value", [row[1] for row in UNWRITABLE], ids=[row[0] for row in UNWRITABLE]
)
def test_what_has_no_spelling_is_refused_in_the_other_direction_too(
    value: object,
) -> None:
    with pytest.raises(sop.SopError):
        sop.dumps(value)


# ---------------------------------------------------------------------------
# The gates: the README, the SDK and the rows above say the same thing
# ---------------------------------------------------------------------------

_README = pathlib.Path(__file__).parent.parent / "README.md"


def _documented(header: str) -> set[str]:
    """The first column of the README table under `header`, which is where
    each of these tables is written down for a reader."""
    rows: set[str] = set()
    inside = False
    for line in _README.read_text(encoding="utf-8").splitlines():
        if line.strip() == header:
            inside = True
            continue
        if not inside:
            continue
        if not line.startswith("|"):
            break
        # An escaped pipe is a cell's content -- `X \| None` -- not its edge.
        cell = re.split(r"(?<!\\)\|", line)[1].strip()
        if cell and set(cell) != {"-"}:  # the rule under the header row
            rows.add(cell.replace("`", "").replace("\\|", "|"))
    return rows


def test_every_documented_shape_is_covered_and_nothing_else_is_documented() -> None:
    # The README's shape table is the claim; `SUPPORTED` is what stands behind
    # it.  Equality, so that a shape cannot be added to one without the other.
    assert _documented("| Shape | Reads |") == {label for label, *_ in SUPPORTED}


def test_the_privileged_classes_are_one_list_in_three_places() -> None:
    # The one place a test reaches into the SDK, because the point of this gate
    # is that the implementation's table cannot grow a class quietly.
    from sop._ir import CARRIERS as _CARRIERS

    implemented = {cls.__name__ for cls in _CARRIERS}
    covered = {shape.__name__ for _label, shape, *_ in SUPPORTED if shape in _CARRIED}
    assert implemented == covered == _documented("| Class | Carried as |")


@given(
    st.sampled_from([*_LEAVES, *_UNIONS]).flatmap(
        lambda shape: st.tuples(st.just(shape), values_for(shape))
    )
)
def test_the_oracle_and_the_generators_know_every_leaf(
    shaped: tuple[TypeForm[Any], Any],
) -> None:
    # Both walks dispatch over the language themselves, and both raise rather
    # than shrug at a shape they have not heard of -- so a shape added to the
    # SDK and to `_LEAVES` but to neither walk fails here rather than passing
    # every property above by never being generated.
    shape, value = shaped
    assert conforms(value, shape)
