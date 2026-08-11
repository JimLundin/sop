"""Every shape a user can write, as a type checker sees it.

`loads[S]` takes a `TypeForm[S]` (PEP 747) and answers an `S`. That is a claim
about static types, so it is checked by type checkers rather than by running
anything -- but each line here is run as well, against a document that really
does read that way, so the two halves cannot drift apart: they are the same
expression.

The SDK cannot fix a type checker, but it can know exactly what its users will
see, which is what the markers are for. A line tagged

    # expect: subscript

reports, under the checkers that tag names and no others, exactly the
diagnostics it names -- the tags are defined a few lines down, with what each
one is about. A marker sits at the end of the line it is about, or on its own
line directly above one. A line with no marker must be clean under every
checker in the table. `tests/test_typing.py` runs them and holds them to it.

The one thing no marker can record is the other asymmetry: a shape the type
system accepts and the SDK refuses at run time. Those are gathered under
`refused_at_run_time` below, each with the error it really raises, because a
user who writes one gets no warning until it runs.
"""

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import (
    Annotated,
    Any,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TypeAliasType,
    TypedDict,
    TypeForm,
    Union,
    assert_type,
)
from uuid import UUID

import pytest
import sop

# ---------------------------------------------------------------------------
# What the checkers do not agree about, named once here and referred to by the
# lines it happens on
# ---------------------------------------------------------------------------
#
# expect-set: subscript
#     pyright[reportArgumentType,reportAssertTypeFailure]
#     pyright-default[reportArgumentType,reportAssertTypeFailure]
#
# PEP 747's implicit conversion is what lets a bare type expression -- `None`,
# `X | Y`, an alias -- stand where a `TypeForm` is asked for.  pyright applies
# it to the argument of a *call* and not to the argument of a *subscript*, in
# both of its configurations; `loads[S]` is a subscript, so a pyright user
# reading a shape that is not a class sees an error and an `Unknown` in place
# of the value.  Nothing on this side can fix that: the same shape passed to a
# function is accepted (see `forwarding`), so it is the syntax and not the
# signature that pyright is not yet reading.
#
# expect-set: alias
#     subscript
#     pyrefly[assert-type,bad-index]
#
# pyrefly does convert in a subscript -- `None` and every union below are
# clean for it -- but not a `TypeAliasType`, which is what `type X = ...`, an
# explicit `TypeAliasType(...)` and `sop.Value` all are.
#
# expect-set: display
#     pyright[reportAssertTypeFailure]
#     pyright-default[reportAssertTypeFailure]
#
# Two shapes pyright reads correctly and then compares by how it prints them:
# `Any` reached through a `TypeForm` prints as `typing.Any`, and a parameter
# defaulted to `Value` prints with the alias left unexpanded on one side and
# expanded on the other.  The shape is right; only `assert_type` disagrees.
#
# expect-set: call-default
#     pyright-default[reportArgumentType,reportAssertTypeFailure]
#
# The call form, which pyright reads once experimental features are on and
# not before.  This is the one line that tells the two pyright runs apart, and
# so the one that would change if the flag stopped being needed.
#
# expect-set: typeform-call
#     pyright-default[reportAssertTypeFailure]
#
# `TypeForm(...)` itself is PEP 747, so a default pyright does not read it
# either.
#
# expect-set: str-in-a-form
#     mypy[index,maybe-unrecognized-str-typeform]
#
# A type expression with a string inside it is ambiguous -- `Literal["a"]` or
# a forward reference? -- and mypy asks for `TypeForm(...)` around it rather
# than guessing.  See `string_annotations`.


# ---------------------------------------------------------------------------
# Schemas, spelled short so that every case below fits on the one line its
# marker belongs to
# ---------------------------------------------------------------------------


class Colour(enum.Enum):
    Red = "Red"


@dataclass
class P:
    x: int


@dataclass
class L:
    text: str


@dataclass
class Node:  # PEP 649: the annotation names the class it is inside
    kid: "Node | None" = None


@dataclass
class Box[T]:  # a generic dataclass, which is a shape the SDK does not read
    item: T


class Rows(TypedDict):
    a: int


class Row(NamedTuple):
    a: int


class Spelled(Protocol):
    def __str__(self) -> str: ...


type Numbers = list[int]  # PEP 695
type Tree = int | list[Tree]  # and recursive
type Pair[T] = tuple[T, T]  # and generic
Legacy = list[int]  # the implicit alias, which predates all of it
Explicit = TypeAliasType("Explicit", list[int])  # and the explicit spelling


# ---------------------------------------------------------------------------
# The scalars, which are the format's own types
# ---------------------------------------------------------------------------


def scalars() -> None:
    assert_type(sop.loads[int]("1"), int)
    assert_type(sop.loads[float]("1.5"), float)
    assert_type(sop.loads[str]('"a"'), str)
    assert_type(sop.loads[bool]("true"), bool)
    assert_type(sop.loads[None]("null"), None)  # expect: subscript
    assert_type(sop.loads[sop.Symbol]("Active"), sop.Symbol)


# ---------------------------------------------------------------------------
# The containers, parameterised by shapes
# ---------------------------------------------------------------------------


def containers() -> None:
    assert_type(sop.loads[list[int]]("[1]"), list[int])
    assert_type(sop.loads[tuple[int, ...]]("[1]"), tuple[int, ...])
    assert_type(sop.loads[set[str]]('Set ["a"]'), set[str])
    assert_type(sop.loads[frozenset[str]]('Set ["a"]'), frozenset[str])
    assert_type(sop.loads[dict[str, int]]("{a: 1}"), dict[str, int])
    assert_type(sop.loads[frozendict[str, int]]("{}"), frozendict[str, int])
    assert_type(sop.loads[list[list[int]]]("[[1]]"), list[list[int]])
    assert_type(sop.loads[dict[str, list[P]]]("{}"), dict[str, list[P]])


# ---------------------------------------------------------------------------
# The unions, in all three spellings
# ---------------------------------------------------------------------------


def unions() -> None:
    assert_type(sop.loads[P | None]("null"), P | None)  # expect: subscript
    assert_type(sop.loads[P | L]("P {x: 1}"), P | L)  # expect: subscript
    assert_type(sop.loads[P | L | None]("null"), P | L | None)  # expect: subscript
    assert_type(sop.loads[Optional[int]]("null"), Optional[int])  # expect: subscript
    assert_type(sop.loads[Union[int, str]]("1"), Union[int, str])  # expect: subscript
    assert_type(sop.loads[list[P | L]]("[]"), list[P | L])


# ---------------------------------------------------------------------------
# The aliases, which are the constructs a checker is likeliest to differ on:
# an alias is a value at run time and a type expression to a checker, and
# reconciling the two is exactly what PEP 747's implicit conversion does
# ---------------------------------------------------------------------------


def aliases() -> None:
    assert_type(sop.loads[Numbers]("[1]"), list[int])  # expect: alias
    assert_type(sop.loads[Tree]("[1]"), int | list[Tree])  # expect: alias
    assert_type(sop.loads[Legacy]("[1]"), list[int])
    assert_type(sop.loads[Explicit]("[1]"), list[int])  # expect: alias
    assert_type(sop.loads[list[Numbers]]("[]"), list[list[int]])


# ---------------------------------------------------------------------------
# The classes the SDK carries
# ---------------------------------------------------------------------------


def carried_classes() -> None:
    assert_type(sop.loads[P]("P {x: 1}"), P)
    assert_type(sop.loads[Node]("Node {}"), Node)
    assert_type(sop.loads[Colour]("Red"), Colour)
    assert_type(sop.loads[Decimal]('Decimal "1.5"'), Decimal)
    assert_type(sop.loads[UUID](f'UUID "{UUID(int=1)}"'), UUID)
    assert_type(sop.loads[datetime]('datetime "2026-08-05T00:00:00"'), datetime)
    assert_type(sop.loads[date]('date "2026-08-05"'), date)
    assert_type(sop.loads[time]('time "14:23:11"'), time)


# ---------------------------------------------------------------------------
# The SDK's own types, and the two escape hatches
# ---------------------------------------------------------------------------


def natives() -> None:
    assert_type(sop.loads[Any]("1"), Any)  # expect: display
    assert_type(sop.loads[sop.Value]("1"), sop.Value)  # expect: alias
    assert_type(sop.loads[sop.Tagged[int]]("A 1"), sop.Tagged[int])
    # PEP 696: the parameter's default is `Value`, so a bare `Tagged` is one.
    assert_type(sop.loads[sop.Tagged]("A 1"), sop.Tagged[sop.Value])  # expect: display
    assert_type(sop.loads[list[sop.Tagged[int]]]("[]"), list[sop.Tagged[int]])


# ---------------------------------------------------------------------------
# A shape held in a variable, which is how a library forwards one
# ---------------------------------------------------------------------------


def pair[T](shape: TypeForm[T], text: str) -> tuple[T, T]:
    """A `TypeForm` passed through rather than written out, which is what any
    wrapper around `loads` has to be able to do."""
    return sop.loads[shape](text), sop.loads[shape](text)


def forwarding() -> None:
    assert_type(pair(int, "1"), tuple[int, int])
    assert_type(pair(list[P], "[]"), tuple[list[P], list[P]])
    # expect: call-default
    assert_type(pair(P | None, "null"), tuple[P | None, P | None])


# ---------------------------------------------------------------------------
# Writing, which takes an object rather than a shape
# ---------------------------------------------------------------------------


def writing() -> None:
    assert_type(sop.dumps(P(1)), str)
    assert_type(sop.dumps({"a": [1, 2]}), str)
    assert_type(sop.dumps(Colour.Red), str)
    assert_type(sop.dumps({1, 2}), str)


# ---------------------------------------------------------------------------
# The errors, which carry the one location each of them has
# ---------------------------------------------------------------------------


def errors() -> None:
    try:
        sop.loads[int]('"a"')
    except sop.ShapeError as exc:
        assert_type(exc.path, str)
        assert_type(exc.message, str)
    try:
        sop.loads[int]("{")
    except sop.ParseError as exc:
        assert_type(exc.line, int)
        assert_type(exc.column, int)


# ---------------------------------------------------------------------------
# Shapes a checker accepts and the SDK refuses.  The type system can say `S`
# where the SDK has no reading for `S`, and nothing warns until it runs, so
# what each one raises is written down here.
# ---------------------------------------------------------------------------


def refused_at_run_time() -> None:
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        # expect: typeform-call
        assert_type(sop.loads[TypeForm(Literal["a"])]('"a"'), Literal["a"])
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        # expect: typeform-call pyright-default[reportArgumentType]
        assert_type(sop.loads[TypeForm(Annotated[int, "m"])]("1"), int)
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[Rows]("{a: 1}"), Rows)
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[Row]("[1]"), Row)
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[Spelled]('"a"'), Spelled)
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[Sequence[int]]("[1]"), Sequence[int])
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[tuple[int, str]]('[1, "a"]'), tuple[int, str])
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[type[int]]("1"), type[int])
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[object]("1"), object)
    # A generic dataclass, and a generic alias applied to an argument: both are
    # a class or an alias with a parameter still attached, which the SDK reads
    # neither of.  User generics are a thing to add later, like user classes.
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        assert_type(sop.loads[Box[int]]("Box {item: 1}"), Box[int])
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        # expect: alias
        assert_type(sop.loads[Pair[int]]("[1, 2]"), tuple[int, int])


# ---------------------------------------------------------------------------
# A type expression that contains a string, which is the one place PEP 747
# asks the writer for help
# ---------------------------------------------------------------------------


def string_annotations() -> None:
    """`Literal["a"]` holds a string, and a checker cannot tell a string
    inside a type expression from a forward reference to be resolved later --
    so PEP 747 has `TypeForm(...)`, which says which one is meant and is the
    identity at run time.  Both spellings reach the SDK as the same object;
    only the checkers tell them apart."""
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        # expect: str-in-a-form
        sop.loads[Literal["a"]]('"a"')
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[TypeForm(Literal["a"])]('"a"')


# ---------------------------------------------------------------------------
# And the other direction: what a checker refuses, which is what makes the
# claim above worth anything.  Strict mypy reports an ignore it did not need,
# so these fail the gate if the shape layer ever degrades to `Any`.
# ---------------------------------------------------------------------------


def rejected_statically() -> None:
    wrong_scalar: str = sop.loads[int]("1")  # type: ignore[assignment]
    wrong_element: list[str] = sop.loads[list[int]]("[1]")  # type: ignore[assignment]
    wrong_class: L = sop.loads[P]("P {x: 1}")  # type: ignore[assignment]
    # And the run-time values are the shapes', not the annotations': the
    # ignores above are lies, which is what makes them errors to begin with.
    assert isinstance(wrong_scalar, int)
    assert wrong_element == [1]
    assert isinstance(wrong_class, P)


CASES = [
    scalars,
    containers,
    unions,
    aliases,
    carried_classes,
    natives,
    forwarding,
    writing,
    errors,
    string_annotations,
    refused_at_run_time,
    rejected_statically,
]
"""Every case above, for the run-time half to call."""
