"""Compiling a shape into the function that reads it.

    TypeForm  --analyse-->  Shape  --build-->  Decoder
    what was written        the IR             what runs

`build` matches over the shape sum once per shape and returns a closure with
every dispatch decision already made: what a `list[T]` does to its items is
resolved when the shape is compiled, not when a value arrives.  Decoding a
document is then calling a function per node, with no shape left to look at.

The compiled decoder is kept, so the cost is paid once per shape rather than
once per value.  A shape is compiled the first time it is read and not before:
nothing here runs at import.
"""

import dataclasses
import enum
import typing
from collections.abc import Callable
from typing import Any, TypeForm, assert_never

from ._analyse import analyse, enum_spelling
from ._core import ShapeError, Symbol, Tagged
from ._ir import (
    ArrayOf,
    CarrierOf,
    Deferred,
    Dynamic,
    EnumOf,
    Float,
    Int,
    Kind,
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
    names,
)
from ._ir import Bool as BoolShape

type Decoder = Callable[[Value, str], Any]
"""A value and the path it sits at, answered as whatever its shape said."""


# ---------------------------------------------------------------------------
# What a document says about itself
# ---------------------------------------------------------------------------


def kind_of(value: Value) -> Kind:
    """Which kind of value this is.  Total over `Value`: an unhandled member
    would fall out of the match and return `None`, which the declared `Kind`
    refuses, so the checkers prove there is no default arm to write -- and so
    the match never falls through."""
    match value:
        case None:
            return Kind.NULL
        case True | False:
            return Kind.BOOL
        case Symbol():
            return Kind.SYMBOL
        case Tagged():
            return Kind.TAGGED
        case str():
            return Kind.STR
        case int():
            return Kind.INT
        case float():
            return Kind.FLOAT
        case tuple():
            return Kind.ARRAY
        # The last arm cannot fail to match, which coverage cannot see.
        case frozendict():  # pragma: no branch
            return Kind.OBJECT


def describe(value: Value) -> str:
    """What was actually there, for an error message.  The kinds that name
    themselves say the name, because that is what the reader has to change."""
    match value:
        case None:
            return "the symbol `null`"
        case Symbol():
            return f"the symbol `{value.name}`"
        case Tagged():
            return f"a value tagged `{value.tag}`"
        case True | False:
            return f"the symbol `{'true' if value else 'false'}`"
        case _:
            return str(kind_of(value))


def _slot_of(value: Value, kind: Kind) -> Slot:
    """The slot a value fills, which is how a union finds it.  A tagged value
    and a symbol name themselves; everything else is claimed by kind."""
    if kind is Kind.TAGGED:
        assert isinstance(value, Tagged)
        return ("tag", value.tag)
    if kind is Kind.SYMBOL:
        assert isinstance(value, Symbol)
        return ("sym", value.name)
    return kind


def _wrong(shape: Shape, value: Value, path: str) -> ShapeError:
    return ShapeError(path, f"expected {names(shape)}, found {describe(value)}")


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

type _Memo = dict[object, list[Decoder]]
"""What is already being compiled: a dataclass by its class, an alias that
contains itself by the identity of its cell.  Both are the same knot."""


def build(shape: Shape, memo: _Memo | None = None) -> Decoder:
    """Compile a shape into the function that reads it.

    `memo` holds the records already under construction.  A shape that refers
    to itself -- `Node.children: list[Node]` -- would otherwise build forever,
    because building the field means building the record again.  The cell is
    published before the fields are built, so a recursive reference finds it
    and forwards to whatever lands there.  One indirection is enough because
    the IR is a Python object graph; a representation that had to be
    serialised would need the reference to be a name instead."""
    if memo is None:
        memo = {}
    match shape:
        case Dynamic():
            return lambda value, path: value

        case Null():

            def null(value: Value, path: str) -> None:
                if value is not None:
                    raise _wrong(shape, value, path)
                return None

            return null

        case BoolShape():

            def boolean(value: Value, path: str) -> bool:
                if not isinstance(value, bool):
                    raise _wrong(shape, value, path)
                return value

            return boolean

        case Int():

            def integer(value: Value, path: str) -> int:
                # bool subclasses int in Python; a sop boolean is a symbol.
                if isinstance(value, bool) or not isinstance(value, int):
                    raise _wrong(shape, value, path)
                return value

            return integer

        case Float():

            def number(value: Value, path: str) -> float:
                # A float-spelled number only.  `1` is an integer by the
                # format's own rule, and reading it here would be a guess.
                if not isinstance(value, float):
                    raise _wrong(shape, value, path)
                return value

            return number

        case Str():

            def text(value: Value, path: str) -> str:
                # A symbol is never a string, so `Active` is not "Active".
                if not isinstance(value, str):
                    raise _wrong(shape, value, path)
                return value

            return text

        case SymbolOf():

            def symbol(value: Value, path: str) -> Symbol:
                if not isinstance(value, Symbol):
                    raise _wrong(shape, value, path)
                return value

            return symbol

        case EnumOf(cls, _):
            by_spelling = {enum_spelling(m): m for m in cls}

            def member(value: Value, path: str) -> enum.Enum:
                if not isinstance(value, Symbol):
                    raise _wrong(shape, value, path)
                found = by_spelling.get(value.name)
                if found is None:
                    raise _wrong(shape, value, path)
                return found

            return member

        case ArrayOf(item, mutable):
            each = build(item, memo)

            def array(value: Value, path: str) -> Any:
                if not isinstance(value, tuple):
                    raise _wrong(shape, value, path)
                read = [each(v, f"{path}[{i}]") for i, v in enumerate(value)]
                return read if mutable else tuple(read)

            return array

        case SetOf(item, mutable):
            each_element = build(item, memo)

            def as_set(value: Value, path: str) -> Any:
                if not isinstance(value, Tagged) or value.tag != "Set":
                    raise _wrong(shape, value, path)
                body = value.value
                if not isinstance(body, tuple):
                    raise ShapeError(
                        path, f"`Set` carries an array, found {describe(body)}"
                    )
                read = [each_element(v, f"{path}[]") for v in body]
                try:
                    return set(read) if mutable else frozenset(read)
                except TypeError as exc:
                    # A shape whose elements a set cannot hold -- `set[list[int]]`,
                    # or a set of a dataclass that is not frozen.  Decoded first
                    # and built after, so an element's own failure stays its own
                    # error and only the building is caught here.
                    raise ShapeError(path, f"unsupported shape: {exc}") from None

            return as_set

        case MappingOf(item, mutable):
            each_value = build(item, memo)

            def mapping(value: Value, path: str) -> Any:
                if not isinstance(value, frozendict):
                    raise _wrong(shape, value, path)
                read = {k: each_value(v, f"{path}.{k}") for k, v in value.items()}
                return read if mutable else frozendict(read)

            return mapping

        case TaggedOf(payload):
            each_payload = build(payload, memo)

            def tagged(value: Value, path: str) -> Tagged[Any]:
                if not isinstance(value, Tagged):
                    raise _wrong(shape, value, path)
                return Tagged(value.tag, each_payload(value.value, path))

            return tagged

        case CarrierOf(carried, make):
            carrier_tag = carried.__name__

            def carrier(value: Value, path: str) -> Any:
                if not isinstance(value, Tagged) or value.tag != carrier_tag:
                    raise _wrong(shape, value, path)
                body = value.value
                if not isinstance(body, str):
                    raise ShapeError(
                        path,
                        f"`{carrier_tag}` carries a string, found {describe(body)}",
                    )
                try:
                    built: Any = make(body)
                    return built
                except (ValueError, ArithmeticError) as exc:
                    raise ShapeError(
                        path, f'`{carrier_tag} "{body}"` is not valid: {exc}'
                    ) from None

            return carrier

        case Record(cls):
            return _build_record(shape, cls, memo)

        case Deferred(cell, _):
            # The alias this refers back to may still be compiling, so the
            # cell is read through rather than compiled again.
            key = id(cell)
            if (compiling := memo.get(key)) is not None:
                return lambda value, path: compiling[0](value, path)
            landed: list[Decoder] = []
            memo[key] = landed
            denoted = build(cell[0], memo)
            landed.append(denoted)
            return denoted

        case UnionOf(members):
            return _build_union(shape, members, memo)

        case _:  # pragma: no cover -- exhaustive; the checkers prove it
            assert_never(shape)


_SLOT = "__sopdecoder_cache__"
"""Where a dataclass's compiled decoder is kept: on the class itself.

A table here would be a GC root, and a decoder names the class it builds --
a self-referential dataclass names it twice over.  Weak keys do not help,
because it is the *value* that refers back.  On the class the same reference
is an ordinary cycle, which the collector takes apart once nothing else
refers to the class, so a class built at run time stays collectable.

Read out of `cls.__dict__` and not with `getattr`: an inherited decoder would
build the base, and a subclass declares its own fields."""


def _build_record(shape: Shape, cls: type, memo: _Memo) -> Decoder:
    """A dataclass: its fields resolved to decoders once, then read by name.

    This is where compiling costs anything -- resolving annotations walks the
    MRO -- so it is what is worth keeping, and the wrappers around it are
    cheap enough to rebuild."""
    if (pending := memo.get(cls)) is not None:
        # Already being built, so this is a reference back into a shape that
        # contains itself.  Forward to the decoder that will land in the cell
        # once the outer build finishes.
        return lambda value, path: pending[0](value, path)
    if (done := cls.__dict__.get(_SLOT)) is not None:
        decoder: Decoder = done
        return decoder
    cell: list[Decoder] = []
    memo[cls] = cell

    tag = cls.__name__
    hints = typing.get_type_hints(cls, include_extras=True)
    # Each field's wire key, its decoder and whether the document must carry
    # it -- resolved here, so reading an instance is a walk over this and no
    # questions about the class.  The key is what `metadata={"sop": "from"}`
    # names, or the field's own name verbatim.
    plan = tuple(
        (
            field.name,
            str(field.metadata.get("sop", field.name)),
            build(analyse(hints[field.name]), memo),
            field.default is dataclasses.MISSING
            and field.default_factory is dataclasses.MISSING,
        )
        for field in dataclasses.fields(cls)
        if field.init
    )

    def record(value: Value, path: str) -> Any:
        # Always tagged, and always with the class's own name -- which is what
        # a union keys on.
        if not isinstance(value, Tagged) or value.tag != tag:
            raise _wrong(shape, value, path)
        body = value.value
        if not isinstance(body, frozendict):
            raise ShapeError(path, f"expected an object, found {describe(body)}")
        kwargs: dict[str, Any] = {}
        for name, key, decode_field, required in plan:
            if key in body:
                kwargs[name] = decode_field(body[key], f"{path}.{key}")
            elif required:
                raise ShapeError(path, f"missing key `{key}`")
        try:
            built: Any = cls(**kwargs)
            return built
        except (ValueError, ArithmeticError) as exc:
            # A validating `__post_init__` fails here; it surfaces as a shape
            # error with a path, like every other way a document can miss.
            raise ShapeError(path, f"not a valid {tag}: {exc}") from None

    cell.append(record)
    try:
        setattr(cls, _SLOT, record)
    except TypeError:  # pragma: no cover -- a dataclass is never a static type
        pass
    return record


def _build_union(shape: Shape, members: tuple[Shape, ...], memo: _Memo) -> Decoder:
    """A union's dispatch table, built once.

    The document's own discriminant chooses, so the order the members were
    written in does not come into it.  `_analyse` has already refused a union
    whose members' claims meet, so filling this table cannot collide."""
    table: dict[Slot, Decoder] = {}
    for member in members:
        decoder = build(member, memo)
        for slot in claims(member):
            table[slot] = decoder

    def union(value: Value, path: str) -> Any:
        # The slot the document fills, then the kind it belongs to.  At most
        # one of the two is claimed -- a member claiming the kind and one
        # claiming a slot within it is the collision `_analyse` refuses -- so
        # there is nothing to choose between.
        kind = kind_of(value)
        chosen = table.get(_slot_of(value, kind))
        if chosen is None:
            chosen = table.get(kind)
        if chosen is None:
            raise _wrong(shape, value, path)
        return chosen(value, path)

    return union


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def decoder_for[T](shape: TypeForm[T]) -> Callable[[Value, str], T]:
    """Compile a shape, which is what `loads[Shape]` holds on to.

    There is no table of these here.  A shape names classes, and a table
    keyed on shapes would hold every class ever read for the life of the
    process; what is worth keeping is kept on the classes themselves (see
    `_SLOT`), which is both the expensive part and the part that can be
    collected."""
    compiled: Callable[[Value, str], T] = build(analyse(shape))
    return compiled
