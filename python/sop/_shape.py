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

import dataclasses
import enum
import functools
import types
import typing
from annotationlib import Format, get_annotations
from collections.abc import Callable
from typing import Any, get_args, get_origin

from ._core import SopError, Symbol, Tagged

type Shape = Any
"""A type used as a schema.  Anything `TypeForm` accepts."""

_NONE = type(None)


class ShapeError(SopError):
    """The document parsed, but does not have the requested shape.

    A subclass of `SopError`, so one `except` covers both failure modes."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message
        # A parse error carries a position; a shape error has none, but the
        # attributes exist on every SopError, as the stubs promise.
        self.line = 0
        self.column = 0


# A set is written `set ["admin", "beta"]` -- a tag applied to an array, and a
# tagged value is never equal to its payload, so `list[str]` and `set[str]` are
# different shapes rather than two spellings of one.
_SET_TAG = "set"


_UNTAGGED = "__sop_untagged__"


def _cache[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """`functools.cache`, typed so the decorated function keeps its signature.
    Typeshed types the wrapper's arguments as `Hashable`, which `type[...]`
    does not satisfy under current mypy."""
    return typing.cast("Callable[P, T]", functools.cache(fn))


@_cache
def _tag_of(shape: Any) -> str | None:
    """A class's tag: `__sop_tag__` if set, otherwise its name.  Setting it to
    `None` is how a class asks to be carried as a bare object.

    Builtins are exempt from the default, because `str` and `list` are how the
    format's own types are spelled, not user classes waiting for a tag."""
    if not isinstance(shape, type) or shape.__module__ == "builtins":
        return None
    tag = getattr(shape, "__sop_tag__", _UNTAGGED)
    return shape.__name__ if tag is _UNTAGGED else tag


def _scalar_tag(cls: type) -> str | None:
    """The tag for the tagged-*string* carrier: explicit `__sop_tag__` only.

    The name default carries dataclasses, whose fields are declared and read
    back.  An arbitrary class declares nothing, and defaulting it through
    `str(obj)` would spell a class that never opted in as
    `Name "<object at 0x…>"` -- so without a named tag it is not carried."""
    tag = getattr(cls, "__sop_tag__", None)
    return tag if isinstance(tag, str) else None


@_cache
def _hints(cls: type) -> dict[str, Any]:
    """Field types, with PEP 649 annotations evaluated. Cached: the merge
    walks the MRO, and decoding asks once per instance."""
    merged: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        merged.update(get_annotations(base, format=Format.VALUE))
    return merged


@_cache
def _fields(shape: type) -> tuple[tuple[dataclasses.Field, str], ...]:
    """Each field with its sop key, resolved once per class."""
    return tuple((f, _key(f)) for f in dataclasses.fields(shape))


def _key(field: dataclasses.Field) -> str:
    """The sop key for a field: `metadata={"sop": "from"}` names the wire key;
    otherwise the field's own name is used verbatim."""
    return str(field.metadata.get("sop", field.name))


def _enum_spelling(member: enum.Enum) -> str:
    """How a member is written: its value when that is a string, otherwise its
    name.  Reading matches exactly this spelling and nothing else."""
    return member.value if isinstance(member.value, str) else member.name


def _describe(value: Any) -> str:
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
        case frozendict():
            return "an object"
        case _:  # pragma: no cover - defensive; every parsed value is above
            return type(value).__name__


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def decode(value: Any, shape: Shape, path: str = "$") -> Any:
    if shape is Any:
        return value
    if isinstance(shape, typing.TypeAliasType):
        # A `type X = ...` alias -- the SDK's own `Value` included -- denotes
        # its right-hand side.
        return decode(value, shape.__value__, path)
    if shape is None or shape is _NONE:
        if value is not None:
            raise ShapeError(path, f"expected `null`, found {_describe(value)}")
        return None

    # Untyped values are immutable -- tuples and frozendicts -- and the shape
    # declares whether the decoded result mutates: `list[T]` and `tuple[T,
    # ...]` read the same array, `dict` and `frozendict` the same object.
    origin = get_origin(shape)
    if origin in (typing.Union, types.UnionType):
        return _union(value, get_args(shape), path)
    if origin is list:
        item = (get_args(shape) or (Any,))[0]
        return [decode(v, item, f"{path}[{i}]") for i, v in enumerate(_array(value, path))]
    if origin is tuple:
        args = get_args(shape)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise ShapeError(
                path, f"unsupported shape {shape!r}: only `tuple[T, ...]` reads an array"
            )
        return tuple(
            decode(v, args[0], f"{path}[{i}]") for i, v in enumerate(_array(value, path))
        )
    if origin in (set, frozenset):
        if not isinstance(value, Tagged) or value.tag != _SET_TAG:
            raise ShapeError(
                path, f"expected an array tagged `{_SET_TAG}`, found {_describe(value)}"
            )
        item = (get_args(shape) or (Any,))[0]
        members = (decode(v, item, f"{path}[]") for v in _array(value.value, path))
        return set(members) if origin is set else frozenset(members)
    if origin in (dict, frozendict):
        if not isinstance(value, frozendict):
            raise ShapeError(path, f"expected an object, found {_describe(value)}")
        args = get_args(shape) or (Any, Any)
        if args[0] not in (str, Any):
            raise ShapeError(path, f"unsupported shape {shape!r}: object keys are strings")
        mapped = {k: decode(v, args[-1], f"{path}.{k}") for k, v in value.items()}
        return mapped if origin is dict else frozendict(mapped)

    if isinstance(shape, type):
        if issubclass(shape, enum.Enum):
            return _decode_enum(value, shape, path)
        if shape is bool:
            if value is not True and value is not False:
                raise ShapeError(path, f"expected `true` or `false`, found {_describe(value)}")
            return value
        if shape is int:
            # bool subclasses int in Python; a sop boolean is a symbol.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ShapeError(path, f"expected an integer, found {_describe(value)}")
            return value
        if shape is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ShapeError(path, f"expected a number, found {_describe(value)}")
            return float(value)
        if shape is str:
            # A symbol is never a string, so `Active` is not "Active".
            if not isinstance(value, str):
                raise ShapeError(path, f"expected a string, found {_describe(value)}")
            return value
        if shape is Symbol or shape is Tagged:
            if not isinstance(value, shape):
                raise ShapeError(path, f"expected {shape.__name__}, found {_describe(value)}")
            return value
        if dataclasses.is_dataclass(shape):
            return _decode_dataclass(value, shape, path)
        # Any other class with a *declared* tag is carried as a tagged
        # string; decode has ruled out the builtins, the enums and the
        # dataclasses by now.
        if tag := _scalar_tag(shape):
            return _decode_scalar(value, shape, tag, path)

    raise ShapeError(path, f"unsupported shape {shape!r}")


def _array(value: Any, path: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ShapeError(path, f"expected an array, found {_describe(value)}")
    return value


def _union(value: Any, members: tuple[Any, ...], path: str) -> Any:
    if _NONE in members and value is None:
        return None
    members = tuple(m for m in members if m is not _NONE)
    if len(members) == 1:
        return decode(value, members[0], path)

    # A member's wire tag is the union's discriminant.  Not every class
    # `_tag_of` names is *carried* tagged: enums are symbols, and `Symbol`,
    # `Tagged` and `Any` match on kind, not on a tag.
    tagged: dict[str, Any] = {}
    for member in members:
        if member is Any or member is Symbol or member is Tagged:
            continue
        if isinstance(member, type) and issubclass(member, enum.Enum):
            continue
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

    reasons = []
    for candidate in members:
        try:
            return decode(value, candidate, path)
        except ShapeError as exc:
            reasons.append(exc.message)
    raise ShapeError(path, "no union member matched: " + "; ".join(reasons))


def _decode_scalar(value: Any, shape: type, tag: str, path: str) -> Any:
    if not isinstance(value, Tagged) or value.tag != tag:
        raise ShapeError(path, f"expected a value tagged `{tag}`, found {_describe(value)}")
    if not isinstance(value.value, str):
        raise ShapeError(path, f"`{tag}` carries a string, found {_describe(value.value)}")
    try:
        return getattr(shape, "__sop_parse__", shape)(value.value)
    except (ValueError, ArithmeticError) as exc:
        raise ShapeError(path, f'`{tag} "{value.value}"` is not valid: {exc}') from None


def _decode_enum(value: Any, shape: type[enum.Enum], path: str) -> Any:
    if not isinstance(value, Symbol):
        raise ShapeError(path, f"expected a symbol, found {_describe(value)}")
    for member in shape:
        if _enum_spelling(member) == value.name:
            return member
    spelled = ", ".join(f"`{_enum_spelling(m)}`" for m in shape)
    raise ShapeError(path, f"`{value.name}` is not one of {spelled}")


def _decode_dataclass(value: Any, shape: type, path: str) -> Any:
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
    classify, leaving its children untouched.  The bindings call this from
    inside their single traversal and continue in place, so the graph is
    walked exactly once.

    Precedence mirrors reading: an enum is a symbol, a dataclass is an
    object, and any other class with a tag is a tagged string spelled with
    `str(obj)`."""
    if isinstance(obj, enum.Enum):
        spelling = _enum_spelling(obj)
        try:
            return Symbol(spelling)
        except ValueError as exc:
            # Symbol's constructor speaks ValueError; the shape layer's
            # contract is that only SopError escapes.
            raise ShapeError(
                "$", f"{type(obj).__name__}.{obj.name} has no symbol spelling: {exc}"
            ) from None
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # The body is a mutable intermediate; the writer accepts the mutable
        # counterparts, it just never produces them, hence the casts.
        body = {key: getattr(obj, field.name) for field, key in _fields(type(obj))}
        tag = _tag_of(type(obj))
        return Tagged(tag, typing.cast("Any", body)) if tag else body
    if tag := _scalar_tag(type(obj)):
        return Tagged(tag, str(obj))
    if isinstance(obj, (set, frozenset)):
        # Sorted so output is stable; sop arrays are ordered and sets are not.
        return Tagged(_SET_TAG, typing.cast("Any", sorted(obj, key=repr)))
    raise ShapeError("$", f"cannot encode {type(obj).__name__}")
