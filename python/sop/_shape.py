"""How Python types map onto sop values.

A class is carried under its own name unless it says otherwise:

    class Deposit:                                         ->  Deposit { ... }
    class Geo:      __sop_tag__ = "geo"                    ->  geo { ... }
    class Account:  __sop_tag__ = None                     ->  { ... }
    class Iban(str):__sop_tag__ = "iban"                   ->  iban "DE89…"

A tagged dataclass is a tagged object; anything else with a tag is a tagged
string, built with `cls(text)` and spelled with `str(obj)`.  A type that cannot
be built from its own spelling supplies `__sop_parse__`:

    class Instant(datetime):
        __sop_tag__ = "instant"
        @classmethod
        def __sop_parse__(cls, text): return cls.fromisoformat(text)

The name default carries dataclasses, whose fields are declared; any other
class is carried only when it names a tag itself, because there is no
declared way to spell an arbitrary object.

That is the whole protocol.  No registry, no decorator, no global state, and
no opinion about what `Decimal` or `UUID` should be called -- those are schema
decisions and they belong to the schema.
"""

import builtins
import dataclasses
import enum
import functools
import types
import typing
from typing import Any, TypeForm, get_args, get_origin

from ._core import SopError, Symbol, Tagged

type Value = (
    None | bool | int | float | str | Symbol | Tagged | tuple[Value, ...] | frozendict[str, Value]
)
"""The closed set untyped reading produces -- immutable throughout.  Mutation
is opt-in: a shape such as `list[T]` or `dict[str, V]` declares it, and the
decoded result is freshly built.  `loads[Value]` is the escape hatch with its
result typed precisely; `loads[Any]` reads the same way and types the result
as `Any`.  Kept in step with the alias in `_core.pyi`."""

type Shape = TypeForm[Any] | None
"""A type used as a schema: anything `TypeForm` accepts, and the bare `None`
an annotation such as `deleted_at: None` evaluates to."""


class ShapeError(SopError):
    """The document parsed, but does not have the requested shape.

    A subclass of `SopError`, so one `except` covers both failure modes."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.message = message
        # A parse error carries a position; a shape error has none, but the
        # attributes exist on every SopError, as the stubs promise.
        self.line = 0
        self.column = 0


# A set is written `set ["admin", "beta"]` -- a tag applied to an array, and a
# tagged value is never equal to its payload, so `list[str]` and `set[str]` are
# different shapes rather than two spellings of one.
_SET_TAG = "set"


def _freeze(mapping: dict[str, Any]) -> frozendict[str, Any]:
    """The immutable counterpart of a mapping just built.  Spelled as a merge
    onto an empty frozendict because neither checker's bundled stubs can yet
    resolve `frozendict(mapping)`."""
    return frozendict() | mapping


@functools.cache
def _tag_of(shape: object) -> str | None:
    """A class's wire tag: `__sop_tag__` if set, otherwise its own name.
    Setting it to `None` is how a class asks to be carried as a bare object.

    `None` for everything the format spells some other way, so that what this
    names is exactly what a union can discriminate on: the builtins, because
    `str` and `list` are how the format's own types are spelled rather than
    user classes waiting for a tag; enums, which are carried as symbols; and
    `Symbol`, `Tagged` and `Any`, which match on kind rather than on a tag."""
    if not isinstance(shape, type) or shape.__module__ in ("builtins", "typing"):
        return None
    tag: str | None = getattr(shape, "__sop_tag__", shape.__name__)
    # Asked after the tag is in hand: an enum is carried as a symbol, and
    # `Symbol` and `Tagged` match on kind, so none of the three names a tag.
    return None if issubclass(shape, (enum.Enum, Symbol, Tagged)) else tag


def _class_of(obj: object) -> type[object]:
    """`type(obj)`, asked of an `object` rather than of an `Any`, so that what
    comes back is a class the checkers know."""
    return type(obj)


def _as_class(shape: object) -> type[object] | None:
    """The shape as a class, or `None` if it is not one.  Asked of an `object`
    for the same reason as `_class_of`."""
    return shape if isinstance(shape, type) else None


def _scalar_tag(cls: type) -> str | None:
    """The tag for the tagged-*string* carrier: explicit `__sop_tag__` only.

    The name default carries dataclasses, whose fields are declared and read
    back.  An arbitrary class declares nothing, and defaulting it through
    `str(obj)` would spell a class that never opted in as
    `Name "<object at 0x…>"` -- so without a named tag it is not carried."""
    tag = getattr(cls, "__sop_tag__", None)
    return tag if isinstance(tag, str) else None


@functools.cache
def _hints(cls: type) -> dict[str, Any]:
    """Field types, with PEP 649 annotations evaluated.  Cached: resolution
    walks the MRO, and decoding asks once per instance."""
    return typing.get_type_hints(cls, include_extras=True)


@functools.cache
def _fields(shape: type) -> tuple[tuple[dataclasses.Field[Any], str], ...]:
    """Each field with its sop key, resolved once per class.  The key is what
    `metadata={"sop": "from"}` names, or the field's own name verbatim."""
    return tuple((f, str(f.metadata.get("sop", f.name))) for f in dataclasses.fields(shape))


def _enum_spelling(member: enum.Enum) -> str:
    """How a member is written: its value when that is a string, otherwise its
    name.  Reading matches exactly this spelling and nothing else."""
    return member.value if isinstance(member.value, str) else member.name


def _describe(value: Value) -> str:
    """What was actually there, for an error message.  Total over `Value`: an
    unhandled member would fall out of the match and return `None`, which the
    declared `str` refuses, so the checkers prove there is no default arm to
    write -- and so the match never falls through."""
    match value:
        case None:
            return "the symbol `null`"
        case True | False:
            return f"the symbol `{'true' if value else 'false'}`"
        case Symbol():
            return f"the symbol `{value.name}`"
        case Tagged():
            return f"a value tagged `{value.tag}`"
        case str():
            return "a string"
        case int() | float():
            return "a number"
        case tuple():
            return "an array"
        # The last arm cannot fail to match, which coverage cannot see.
        case frozendict():  # pragma: no branch
            return "an object"


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(value: Value, shape: Shape, path: str = "$") -> Any:
    """Read a parsed value back as `shape`, raising `ShapeError` -- naming the
    path -- for anything that does not fit."""
    if isinstance(shape, typing.TypeAliasType):
        # A `type X = ...` alias -- the SDK's own `Value` included -- denotes
        # its right-hand side.
        return decode(value, shape.__value__, path)

    # The shapes with a fixed spelling in the format, matched by identity.
    match shape:
        case typing.Any:
            return value
        case None | types.NoneType:
            if value is not None:
                raise ShapeError(path, f"expected `null`, found {_describe(value)}")
            return None
        case builtins.bool:
            if not isinstance(value, bool):
                raise ShapeError(path, f"expected `true` or `false`, found {_describe(value)}")
            return value
        case builtins.int:
            # bool subclasses int in Python; a sop boolean is a symbol.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ShapeError(path, f"expected an integer, found {_describe(value)}")
            return value
        case builtins.float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ShapeError(path, f"expected a number, found {_describe(value)}")
            return float(value)
        case builtins.str:
            # A symbol is never a string, so `Active` is not "Active".
            if not isinstance(value, str):
                raise ShapeError(path, f"expected a string, found {_describe(value)}")
            return value
        case _:
            pass  # not spelled by the format itself; the rest is below

    # The parameterised shapes, matched on what they are made of.  Untyped
    # values are immutable, and the shape declares whether the decoded result
    # mutates: `list[T]` and `tuple[T, ...]` read the same array, `dict` and
    # `frozendict` the same object.
    match get_origin(shape), get_args(shape):
        case types.UnionType, members:
            return _union(value, members, path)
        case builtins.list, (item,):
            return [decode(v, item, f"{path}[{i}]") for i, v in enumerate(_array(value, path))]
        case builtins.tuple, (item, builtins.Ellipsis):
            return tuple(decode(v, item, f"{path}[{i}]") for i, v in enumerate(_array(value, path)))
        case builtins.tuple, _:
            # `tuple[int, str]` is a row type, which a sop array does not model.
            raise ShapeError(
                path, f"unsupported shape {shape!r}: only `tuple[T, ...]` reads an array"
            )
        case (builtins.set | builtins.frozenset) as origin, (item,):
            if not isinstance(value, Tagged) or value.tag != _SET_TAG:
                raise ShapeError(
                    path, f"expected an array tagged `{_SET_TAG}`, found {_describe(value)}"
                )
            decoded = (decode(v, item, f"{path}[]") for v in _array(value.value, path))
            return set(decoded) if origin is set else frozenset(decoded)
        case (builtins.dict | builtins.frozendict) as origin, (builtins.str | typing.Any, item):
            if not isinstance(value, frozendict):
                raise ShapeError(path, f"expected an object, found {_describe(value)}")
            mapped = {k: decode(v, item, f"{path}.{k}") for k, v in value.items()}
            return mapped if origin is dict else _freeze(mapped)
        case ((builtins.dict | builtins.frozendict), _):
            # An object's keys are strings; a shape claiming otherwise would
            # decode to something its own annotation contradicts.
            raise ShapeError(path, f"unsupported shape {shape!r}: object keys are strings")
        case _:
            pass  # not parameterised; the classes are below

    # The classes, matched on what they are rather than on a spelling.
    if (cls := _as_class(shape)) is not None:
        if issubclass(cls, enum.Enum):
            return _decode_enum(value, cls, path)
        if cls is Symbol or cls is Tagged:
            if not isinstance(value, cls):
                raise ShapeError(path, f"expected {cls.__name__}, found {_describe(value)}")
            return value
        if dataclasses.is_dataclass(cls):
            return _decode_dataclass(value, cls, path)
        # Any other class with a *declared* tag is carried as a tagged string;
        # the builtins, the enums and the dataclasses are ruled out by now.
        if tag := _scalar_tag(cls):
            return _decode_scalar(value, cls, tag, path)

    raise ShapeError(path, f"unsupported shape {shape!r}")


def _array(value: Value, path: str) -> tuple[Value, ...]:
    if not isinstance(value, tuple):
        raise ShapeError(path, f"expected an array, found {_describe(value)}")
    return value


def _union(value: Value, members: tuple[Any, ...], path: str) -> Any:
    if types.NoneType in members and value is None:
        return None
    members = tuple(m for m in members if m is not types.NoneType)
    if len(members) == 1:
        return decode(value, members[0], path)

    # A member's wire tag is the union's discriminant, and `_tag_of` names
    # exactly the members that are carried tagged.
    tagged: dict[str, Any] = {}
    for member in members:
        if name := _tag_of(member):
            if name in tagged:
                # A shared discriminant is a schema bug; falling back to
                # trying members in order would hide it until it bit.
                raise ShapeError(
                    path,
                    f"union members {tagged[name].__name__} and {member.__name__} "
                    f"share the tag `{name}`",
                )
            tagged[name] = member

    if len(tagged) == len(members):
        # Every member is tagged, so the tag decides and a mismatch is a real
        # error rather than one failed attempt out of several.
        expected = ", ".join(f"`{t}`" for t in sorted(tagged))
        if not isinstance(value, Tagged):
            raise ShapeError(path, f"expected a value tagged {expected}, found {_describe(value)}")
        if member := tagged.get(value.tag):
            return decode(value, member, path)
        raise ShapeError(path, f"unknown tag `{value.tag}`; expected one of {expected}")

    # In a mixed union a recognised tag still decides; anything else tries
    # each member in order.
    if isinstance(value, Tagged) and (member := tagged.get(value.tag)):
        return decode(value, member, path)

    reasons: list[str] = []
    for candidate in members:
        try:
            return decode(value, candidate, path)
        except ShapeError as exc:
            reasons.append(exc.message)
    raise ShapeError(path, "no union member matched: " + "; ".join(reasons))


def _decode_scalar(value: Value, shape: type, tag: str, path: str) -> Any:
    if not isinstance(value, Tagged) or value.tag != tag:
        raise ShapeError(path, f"expected a value tagged `{tag}`, found {_describe(value)}")
    if not isinstance(value.value, str):
        raise ShapeError(path, f"`{tag}` carries a string, found {_describe(value.value)}")
    try:
        return getattr(shape, "__sop_parse__", shape)(value.value)
    except (ValueError, ArithmeticError) as exc:
        raise ShapeError(path, f'`{tag} "{value.value}"` is not valid: {exc}') from None


def _decode_enum(value: Value, shape: type[enum.Enum], path: str) -> Any:
    if not isinstance(value, Symbol):
        raise ShapeError(path, f"expected a symbol, found {_describe(value)}")
    for member in shape:
        if _enum_spelling(member) == value.name:
            return member
    spelled = ", ".join(f"`{_enum_spelling(m)}`" for m in shape)
    raise ShapeError(path, f"`{value.name}` is not one of {spelled}")


def _decode_dataclass(value: Value, shape: type, path: str) -> Any:
    if tag := _tag_of(shape):
        if not isinstance(value, Tagged):
            raise ShapeError(path, f"expected a value tagged `{tag}`, found {_describe(value)}")
        if value.tag != tag:
            raise ShapeError(path, f"expected tag `{tag}`, found `{value.tag}`")
        value = value.value

    if not isinstance(value, frozendict):
        raise ShapeError(path, f"expected an object, found {_describe(value)}")

    hints = _hints(shape)
    kwargs: dict[str, Any] = {}
    for field, key in _fields(shape):
        if not field.init:
            continue
        if key in value:
            kwargs[field.name] = decode(value[key], hints[field.name], f"{path}.{key}")
        elif field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
            raise ShapeError(path, f"missing key `{key}`")
    try:
        return shape(**kwargs)
    except (ValueError, ArithmeticError) as exc:
        # A validating `__post_init__` fails here; it surfaces as a shape
        # error with a path, like every other way a document can miss.
        raise ShapeError(path, f"not a valid {shape.__name__}: {exc}") from None


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def convert(obj: Any) -> Any:
    """One step of writing: spell the head of an object the core cannot
    classify, leaving its children untouched.  The core calls this from inside
    its single traversal and continues in place, so the graph is walked
    exactly once.

    The core knows only the immutable values reading produces, so the mutable
    counterparts are frozen here and spell identically.  Past that, precedence
    mirrors reading: an enum is a symbol, a dataclass is an object, and any
    other class with a tag is a tagged string spelled with `str(obj)`."""
    # Every question is asked of the class.  A dataclass *class* handed to
    # `dumps` is an instance of `type`, which is not a dataclass, so it falls
    # through to the error rather than being spelled as its own fields.
    cls = _class_of(obj)
    if issubclass(cls, enum.Enum):
        return Symbol(_enum_spelling(obj))
    if dataclasses.is_dataclass(cls):
        body = _freeze({key: getattr(obj, field.name) for field, key in _fields(cls)})
        tag = _tag_of(cls)
        return Tagged(tag, body) if tag else body
    if tag := _scalar_tag(cls):
        return Tagged(tag, str(obj))
    # A subclass that declared no tag is still carried as what it is, which is
    # how a subclass of `set` already behaved.  `tuple` is here for the same
    # reason, so a named tuple is an array rather than an odd one out.
    if issubclass(cls, (list, tuple)):
        return tuple(obj)
    if issubclass(cls, dict):
        return _freeze(obj)
    if issubclass(cls, (set, frozenset)):
        # Sorted so output is stable; sop arrays are ordered and sets are not.
        return Tagged(_SET_TAG, tuple(sorted(obj, key=repr)))
    raise ShapeError("$", f"cannot encode {cls.__name__}")
