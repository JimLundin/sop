"""Reading Python's type language into sop's.

One direction, once per shape: a `TypeForm` the user wrote becomes the `Shape`
that denotes it, or a `ShapeError` saying it denotes nothing.  This is the only
place that looks at `get_origin`, `get_args` or a `TypeAliasType`; past it the
IR is a finished object graph.

Analysis is a chain of questions rather than a match, because what it is
matching on is Python's type language and not sop's -- a `TypeForm` is not a
closed set and there is no sum to be exhaustive over.  The sum is what comes
out.
"""

import builtins
import dataclasses
import enum
import itertools
import types
import typing
from typing import Any, TypeForm, TypeIs, get_args, get_origin

from ._core import ShapeError, Symbol, Tagged
from ._ir import (
    CARRIERS,
    ArrayOf,
    CarrierOf,
    Deferred,
    Dynamic,
    EnumOf,
    Float,
    Int,
    MappingOf,
    Null,
    Record,
    SetOf,
    Shape,
    Slot,
    Str,
    SymbolOf,
    TaggedOf,
    UnionOf,
    Value,
    claims,
    meeting,
)
from ._ir import Bool as BoolShape


def _origin(shape: TypeForm[Any]) -> type | None:
    """The class a parameterised shape is built from -- `list[int]` gives
    `list`, `int | None` gives `UnionType` -- or `None` if it is not
    parameterised by a class.

    `typing.get_origin` and `get_args` answer typeshed's `AnnotationForm`,
    which is spelled `Any` until PEP 747 reaches it; these two wrappers are
    where that `Any` stops and the shapes below stay shapes.

    `type`, not `type[T]`: that would read better for `list[int]`, whose
    origin really is a `type[list[int]]`, but it is false for a union, whose
    origin is `UnionType` while `type[int | None]` means `type[int] |
    type[None]`.  A union is one of the shapes dispatched on below, so the
    weaker answer is the true one -- as it is in typeshed, for the same
    reason."""
    origin = get_origin(shape)
    return origin if isinstance(origin, type) else None


def _args(shape: TypeForm[Any]) -> tuple[TypeForm[Any], ...]:
    """What a parameterised shape is parameterised by, which are shapes."""
    args: tuple[TypeForm[Any], ...] = get_args(shape)
    return args


def _is_class(shape: object) -> TypeIs[type]:
    """Whether a shape is a class, and so is asked what kind of class it is.
    `type` rather than `type[object]`: what the class constructs is exactly
    what is not known yet, and the callers narrow it with `issubclass`."""
    return isinstance(shape, type)


def _name_of(shape: TypeForm[Any]) -> str:
    """A shape as the user wrote it, for an error message.

    A parameterised shape is spelled out in full: `list[int].__name__` is
    `list`, which is not what was written and not enough to tell it from
    `list[str]`.  Anything else answers its own name, and a shape that is not
    a class has neither."""
    if get_origin(shape) is not None:
        return str(shape)
    return getattr(shape, "__name__", None) or str(shape)


def enum_spelling(member: enum.Enum) -> str:
    """How a member is written: its value when that is a string, otherwise
    its name.  Reading matches exactly this spelling and nothing else.

    Here rather than with either direction, because both of them spell an
    enum and they have to agree."""
    return member.value if isinstance(member.value, str) else member.name


type _Aliases = dict[typing.TypeAliasType, list[Shape]]


def analyse(
    shape: TypeForm[Any], path: str = "$", aliases: _Aliases | None = None
) -> Shape:
    """The `Shape` a `TypeForm` denotes.

    A shape is an alias for another shape, a parameterised type, or a plain
    class; each is read by the helper below that knows about that kind, so
    none of them dispatches on more than one.

    `aliases` holds the aliases still being read, so one that denotes a shape
    containing itself resolves to a `Deferred` rather than recursing."""
    if aliases is None:
        aliases = {}
    if isinstance(shape, typing.TypeAliasType):
        if shape is Value:
            # `Value` is the set the parser produces, so reading through it is
            # a check that cannot fail and walking the value would do nothing
            # but rebuild an equal copy.  It answers what was there, as `Any`
            # does; what separates the two is static, and stays static.
            return Dynamic()
        # A `type X = ...` alias denotes its right-hand side.
        if (cell := aliases.get(shape)) is not None:
            return Deferred(cell, shape.__name__)
        cell = []
        aliases[shape] = cell
        denoted = analyse(shape.__value__, path, aliases)
        cell.append(denoted)
        del aliases[shape]
        return denoted
    if shape is Any:
        return Dynamic()
    if shape is None:
        # `deleted_at: None` and `deleted_at: NoneType` are one shape; from
        # here on it is spelled as the class, so only classes are left.
        shape = types.NoneType
    if (origin := _origin(shape)) is not None:
        return _parameterised(shape, origin, path, aliases)
    if _is_class(shape):
        return _plain(shape, path)
    raise ShapeError(path, f"unsupported shape {shape!r}")


def _parameterised(
    shape: TypeForm[Any], origin: type, path: str, aliases: _Aliases
) -> Shape:
    """A shape built from another -- `list[T]`, `A | B` -- read by what it is
    built from and what it is built of."""
    match origin, _args(shape):
        case types.UnionType, members:
            return _union(members, path, aliases)
        case builtins.list, (item,):
            return ArrayOf(analyse(item, path, aliases), mutable=True)
        case builtins.tuple, (item, builtins.Ellipsis):
            return ArrayOf(analyse(item, path, aliases), mutable=False)
        case builtins.tuple, _:
            # `tuple[int, str]` is a row type, which a sop array does not model.
            raise ShapeError(
                path,
                f"unsupported shape {shape!r}: only `tuple[T, ...]` reads an array",
            )
        case builtins.set, (item,):
            return SetOf(analyse(item, path, aliases), mutable=True)
        case builtins.frozenset, (item,):
            return SetOf(analyse(item, path, aliases), mutable=False)
        case _, (item,) if origin is Tagged:
            return TaggedOf(analyse(item, path, aliases))
        case builtins.dict, (builtins.str | typing.Any, item):
            return MappingOf(analyse(item, path, aliases), mutable=True)
        case builtins.frozendict, (builtins.str | typing.Any, item):
            return MappingOf(analyse(item, path, aliases), mutable=False)
        case ((builtins.dict | builtins.frozendict), _):
            # An object's keys are strings; a shape claiming otherwise would
            # decode to something its own annotation contradicts.
            raise ShapeError(
                path, f"unsupported shape {shape!r}: object keys are strings"
            )
    raise ShapeError(path, f"unsupported shape {shape!r}")


def _plain(cls: type, path: str) -> Shape:
    """A shape that is just a class, read by what kind of class it is."""
    match cls:
        case types.NoneType:
            return Null()
        case builtins.bool:
            return BoolShape()
        case builtins.int:
            return Int()
        case builtins.float:
            return Float()
        case builtins.str:
            return Str()
        case _:
            pass  # not one the format spells itself; the rest is below

    if issubclass(cls, enum.Enum):
        return EnumOf(cls, frozenset(enum_spelling(m) for m in cls))
    if cls is Symbol:
        return SymbolOf()
    if cls is Tagged:
        # A bare `Tagged` names no payload shape, so its payload is whatever
        # was there -- which is what `Tagged[Value]`, its default, means.
        return TaggedOf(Dynamic())
    if dataclasses.is_dataclass(cls):
        return Record(cls)
    # One of the privileged few is a tagged string; the builtins, the enums
    # and the dataclasses are ruled out by now.
    if (carrier := CARRIERS.get(cls)) is not None:
        _, build = carrier
        return CarrierOf(cls, build)
    raise ShapeError(path, f"unsupported shape {cls!r}")


def _union(members: tuple[TypeForm[Any], ...], path: str, aliases: _Aliases) -> Shape:
    """`A | B`, with the members analysed and checked against each other.

    A union is read by what the document says it is, so two members that say
    the same thing cannot be told apart.  That is a fact about the schema
    rather than about any one document, so it is settled here, once, when the
    shape is first read -- not on whichever document happens to reach the
    collision first.

    A member that is itself a union -- which an alias can be -- is left whole
    and answers for its own members: `claims` aggregates them, so a collision
    across the two levels is found just the same, and dispatch reaches the
    inner union by way of the outer one."""
    # Each analysed member paired with the member it was written as, so a
    # collision can name what the user wrote rather than what it became.
    pairs = [(member, analyse(member, path, aliases)) for member in members]

    for _, inner in pairs:
        if isinstance(inner, Deferred) and not inner.cell:
            # `type T = T | int`: the alias is one of its own alternatives, so
            # there is nothing to read before reading it again.
            raise ShapeError(
                path, f"alias `{inner.name}` is one of its own union members"
            )

    # Each member's claims worked out once, not again for every other member
    # it is held against.
    claimed = [(member, claims(inner)) for member, inner in pairs]
    for (one_form, one), (other_form, other) in itertools.combinations(claimed, 2):
        if (slot := meeting(one, other)) is not None:
            raise ShapeError(
                path,
                f"union members {_name_of(one_form)} and {_name_of(other_form)} "
                f"cannot be told apart: both read {_spell(slot)}",
            )

    return UnionOf(tuple(inner for _, inner in pairs))


def _spell(slot: Slot) -> str:
    """A claimed slot, for the error that says two members share it."""
    match slot:
        case ("tag", name):
            return f"a value tagged `{name}`"
        case ("sym", name):
            return f"the symbol `{name}`"
        case kind:
            return str(kind)
