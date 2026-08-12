"""The shape language, written down as a type.

This module is the specification: what shapes there are, and what each one
claims of a document.  It holds no logic beyond that -- `_analyse` decides
which shape a `TypeForm` denotes and `_decode` decides what each one does, so
neither of them is where the language is defined.

`Shape` is closed, so a `match` over it is exhaustive and `assert_never`
proves it.  Adding a shape means adding a variant, and the checkers then name
every place that has to answer for it; a shape cannot be added quietly, nor
handled in one direction and forgotten in the other.

The variants carry what the shape was built of, already analysed -- `ArrayOf`
holds the shape of its items, not the `TypeForm` they came from -- so the IR
is a finished object graph and nothing below it looks at Python's type
language again.
"""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Literal, assert_never
from uuid import UUID

from ._core import Symbol, Tagged

# ---------------------------------------------------------------------------
# The value domain
# ---------------------------------------------------------------------------

type Value = (
    None
    | bool
    | int
    | float
    | str
    | Symbol
    | Tagged[Value]
    | tuple[Value, ...]
    | frozendict[str, Value]
)
"""The value domain, which belongs to the core -- `loads` is what produces it.

It is spelled here rather than in `_core.pyi` because this is the only place
the object can come from.  A stub does not exist at run time and `sop.Value`
does; only a `type` statement can build an alias that refers to itself, so the
extension has no way to emit one.  The stub imports this name back for its own
signatures, so there is one definition and nothing to keep in step.

`Tagged[Value]`, not a bare `Tagged`: bare would lean on the parameter's
default, which is this alias, and resolving that default across the import
means finishing this definition first.  Naming the payload asks nothing of the
default, so the cycle has nothing left to resolve."""


# ---------------------------------------------------------------------------
# What a document says about itself
# ---------------------------------------------------------------------------


class Kind(enum.StrEnum):
    """The kinds of value the format has, which is what a document
    discriminates on before anything else is asked of it.

    The value spells the kind for an error message, so `expected` and `found`
    are drawn from one vocabulary."""

    NULL = "`null`"
    BOOL = "`true` or `false`"
    INT = "an integer"
    FLOAT = "a float"
    STR = "a string"
    SYMBOL = "a symbol"
    TAGGED = "a tagged value"
    ARRAY = "an array"
    OBJECT = "an object"


type Slot = tuple[Literal["tag", "sym"], str] | Kind
"""What a shape can claim exclusively in a union.

A tagged value and a symbol name themselves, so they are discriminated by
that name rather than merely by their kind: two dataclasses are both
`Kind.TAGGED` and are still told apart.  Everything else is claimed by kind,
because nothing finer is on the wire."""


# ---------------------------------------------------------------------------
# The shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Dynamic:
    """`Any` and `Value`: whatever is there, answered as it stands."""


@dataclass(frozen=True, slots=True)
class Null:
    """`None`: the symbol `null`, and nothing else."""


@dataclass(frozen=True, slots=True)
class Bool:
    """`bool`: the symbol `true` or the symbol `false`.  Never a number,
    though Python's `bool` is an `int`."""


@dataclass(frozen=True, slots=True)
class Int:
    """`int`: a number spelled with digits alone."""


@dataclass(frozen=True, slots=True)
class Float:
    """`float`: a number, however it was spelled -- so an integer-spelled
    one reads as a float too."""


@dataclass(frozen=True, slots=True)
class Str:
    """`str`: a string, never a symbol."""


@dataclass(frozen=True, slots=True)
class SymbolOf:
    """`sop.Symbol`: any symbol at all, answered as itself."""


@dataclass(frozen=True, slots=True)
class EnumOf:
    """An enum, carried as a symbol and matched on its spelling.  The
    spellings are resolved here so a union can discriminate on them."""

    cls: type[enum.Enum]
    spellings: frozenset[str]


@dataclass(frozen=True, slots=True)
class ArrayOf:
    """`list[T]` and `tuple[T, ...]`, which read the same array; the shape
    declares whether what comes back mutates."""

    item: Shape
    mutable: bool


@dataclass(frozen=True, slots=True)
class SetOf:
    """`set[T]` and `frozenset[T]`, carried as an array tagged `Set`."""

    item: Shape
    mutable: bool


@dataclass(frozen=True, slots=True)
class MappingOf:
    """`dict[str, V]` and `frozendict[str, V]`."""

    value: Shape
    mutable: bool


@dataclass(frozen=True, slots=True)
class TaggedOf:
    """`sop.Tagged` and `sop.Tagged[V]`: a tag this shape does not name, over
    a payload it does."""

    payload: Shape


@dataclass(frozen=True, slots=True)
class CarrierOf:
    """One of the privileged classes, carried as a tagged string."""

    cls: type
    build: Callable[[str], Any]


@dataclass(frozen=True, slots=True)
class Record:
    """A dataclass, carried as an object tagged with the class's own name."""

    cls: type


@dataclass(frozen=True, slots=True)
class Deferred:
    """A reference back to an alias still being analysed.

    `type Tree = int | list[Tree]` denotes a shape that contains itself, so
    analysing it eagerly would never finish.  The inner reference becomes one
    of these instead: a cell that the alias fills once it is analysed, and
    which everything holding the reference then reads through.

    The cell is a list because the shapes are frozen and this is the one edge
    that cannot be built bottom-up.  It holds exactly one shape once filled,
    and is empty only while the alias it names is still being read."""

    cell: list[Shape]
    name: str


@dataclass(frozen=True, slots=True)
class UnionOf:
    """`A | B | C`, discriminated on what the document says it is.  Flattened
    and with no duplicate claim among the members: `_analyse` refuses a union
    whose members cannot be told apart, so the members here are distinct on
    the wire."""

    members: tuple[Shape, ...]


type Shape = (
    Dynamic
    | Null
    | Bool
    | Int
    | Float
    | Str
    | SymbolOf
    | EnumOf
    | ArrayOf
    | SetOf
    | MappingOf
    | TaggedOf
    | CarrierOf
    | Record
    | Deferred
    | UnionOf
)


# ---------------------------------------------------------------------------
# The privileged classes
# ---------------------------------------------------------------------------

type Carriers[T] = frozendict[type[T], tuple[Callable[[T], str], Callable[[str], T]]]

CARRIERS: Carriers[Any] = frozendict(
    {
        Decimal: (str, Decimal),
        UUID: (str, UUID),
        datetime: (datetime.isoformat, datetime.fromisoformat),
        date: (date.isoformat, date.fromisoformat),
        time: (time.isoformat, time.fromisoformat),
    }
)
"""The classes carried as tagged strings: how each is spelled and built, and
nothing about what it is called -- a carried class is tagged with its own
name, this one included.

Matched exactly.  A subclass of one of these is not carried, and is refused
like any other class the SDK does not know.

A `frozendict`, so the one table the SDK dispatches on cannot be reached into
and changed at run time.  There is deliberately no way for a class to opt in,
and a mutable table is exactly the way in that would be.

Read from both directions -- `_analyse` takes the builder and `_shape.convert`
takes the speller -- which is why it lives here with the language rather than
with either side of it."""


# ---------------------------------------------------------------------------
# What each shape claims
# ---------------------------------------------------------------------------


def claims(shape: Shape) -> tuple[frozenset[Slot], frozenset[Kind]]:
    """What a shape reads: the slots it claims exclusively, and the kinds it
    will take when nothing has claimed them.

    Two tiers, because a shape that names a value exactly should win over one
    that merely admits it: `int | float` reads `1` as an integer and `1.0` as
    a float, whichever order it was written in, because `Float` claims
    `FLOAT` outright and `INT` only if no `Int` is there to take it.  The
    wildcards are the same idea -- `Symbol` takes any symbol, `Tagged` any
    tag, `Any` anything -- so a shape that names one beats them.

    Two members claiming the same slot are a union that cannot be told apart,
    which `_analyse` refuses.  Wildcards are not claims and do not collide;
    the first member offering one is the one that gets it, and since the
    wildcards are distinguishable only by what they read, that is not a
    choice between different answers."""
    match shape:
        case Dynamic():
            return frozenset(), frozenset(Kind)
        case Null():
            return frozenset({Kind.NULL}), frozenset()
        case Bool():
            return frozenset({Kind.BOOL}), frozenset()
        case Int():
            return frozenset({Kind.INT}), frozenset()
        case Float():
            # An integer-spelled number reads as a float, but only where
            # nothing spelled `int` is on offer to take it first.
            return frozenset({Kind.FLOAT}), frozenset({Kind.INT})
        case Str():
            return frozenset({Kind.STR}), frozenset()
        case SymbolOf():
            return frozenset(), frozenset({Kind.SYMBOL})
        case EnumOf(_, spellings):
            return frozenset(("sym", s) for s in spellings), frozenset()
        case ArrayOf(_, _):
            return frozenset({Kind.ARRAY}), frozenset()
        case MappingOf(_, _):
            return frozenset({Kind.OBJECT}), frozenset()
        case SetOf(_, _):
            return frozenset({("tag", "Set")}), frozenset()
        case CarrierOf(cls, _) | Record(cls):
            return frozenset({("tag", cls.__name__)}), frozenset()
        case TaggedOf(_):
            return frozenset(), frozenset({Kind.TAGGED})
        case Deferred(cell, _):  # pragma: no cover
            # Unreachable: this is asked of a union's members, and a deferred
            # shape is never one.  An alias reaches back into itself through a
            # container -- `list[Tree]` -- and an alias that is directly one of
            # its own alternatives is refused when it is analysed.
            return claims(cell[0])
        case UnionOf(members):
            exact: set[Slot] = set()
            wider: set[Kind] = set()
            for member in members:
                member_exact, member_wider = claims(member)
                exact |= member_exact
                wider |= member_wider
            return frozenset(exact), frozenset(wider)
        case _:  # pragma: no cover -- exhaustive; the checkers prove it
            assert_never(shape)


def names(shape: Shape) -> str:
    """What a shape reads, for an error message."""
    match shape:
        case Dynamic():  # pragma: no cover
            # Unreachable: naming a shape is for saying what was expected, and
            # `Any` takes everything, so it never fails to be met.
            return "anything"
        case Null():
            return "`null`"
        case Bool():
            return "`true` or `false`"
        case Int():
            return "an integer"
        case Float():
            return "a number"
        case Str():
            return "a string"
        case SymbolOf():
            return "a symbol"
        case EnumOf(_, spellings):
            return " or ".join(f"`{s}`" for s in sorted(spellings))
        case ArrayOf(_, _):
            return "an array"
        case MappingOf(_, _):
            return "an object"
        case SetOf(_, _):
            return "an array tagged `Set`"
        case CarrierOf(cls, _) | Record(cls):
            return f"a value tagged `{cls.__name__}`"
        case TaggedOf(_):
            return "a tagged value"
        case Deferred(_, name):  # pragma: no cover
            # Unreachable, as in `claims`: a deferred shape forwards to what
            # the alias denotes, so a failure names that and not this.  The
            # alias's own name is what it would say, and the only spelling of
            # it that does not recurse.
            return f"`{name}`"
        case UnionOf(members):
            # The tagged members are collapsed under one `tagged`, because a
            # union of five dataclasses saying it five times reads as noise.
            tags = sorted(
                m.cls.__name__ for m in members if isinstance(m, (Record, CarrierOf))
            )
            rest = sorted(
                {names(m) for m in members if not isinstance(m, (Record, CarrierOf))}
            )
            if tags:
                spelled = " or ".join(f"`{tag}`" for tag in tags)
                rest.insert(0, f"a value tagged {spelled}")
            return " or ".join(rest)
        case _:  # pragma: no cover -- exhaustive; the checkers prove it
            assert_never(shape)
