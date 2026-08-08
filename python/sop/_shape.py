"""How Python types map onto sop values.

A tag is a constructor, and constructors are spelled in PascalCase -- which a
class's own name already is, so the default needs no declaration.  A class is
carried under its own name unless it says otherwise:

    class Deposit:                                         ->  Deposit { ... }
    class Account:  __sop_tag__ = None                     ->  { ... }
    class Ledger:   __sop_tag__ = "Book"                   ->  Book { ... }
    class Iban(str):__sop_tag__ = "Iban"                   ->  Iban "DE89…"

A tagged dataclass is a tagged object; anything else with a tag is a tagged
string, built with `cls(text)` and spelled with `str(obj)`.  A type that cannot
be built from its own spelling supplies `__sop_parse__`:

    class Instant(datetime):
        __sop_tag__ = "Instant"
        @classmethod
        def __sop_parse__(cls, text): return cls.fromisoformat(text)

The name default carries dataclasses, whose fields are declared; any other
class is carried only when it names a tag itself, because there is no
declared way to spell an arbitrary object.

That is the whole protocol.  No registry, no decorator, no global state, and
no opinion about what `Decimal` or `UUID` should be called -- those are schema
decisions and they belong to the schema.

On typing: `object` means a value whose type is not known, which is a fact
about the value; `Any` means checking is off, which is a decision.  The two
are not interchangeable and `Any` spreads, so it is spelled only where this
module is genuinely dynamic -- `_decode` and `_union` answer whatever the
shape they were handed says, and `decode` in front of them is generic, which
is where that becomes a static type again.
"""

import builtins
import dataclasses
import enum
import functools
import types
import typing
from collections.abc import Callable, Iterable, Set
from typing import Any, TypeForm, TypeIs, get_args, get_origin

from ._core import SopError, Symbol, Tagged

type Value = (
    None | bool | int | float | str | Symbol | Tagged | tuple[Value, ...] | frozendict[str, Value]
)
"""The value domain, which belongs to the core -- `_core.pyi` declares it, and
`loads` is what produces it.

It is written out a second time here, which is a wart.  A stub does not exist
at run time, and `sop.Value` does; only a `type` statement can build an alias
that refers to itself, so the extension has no way to emit one and this is the
only place the object can come from.  The two spellings have to agree."""


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


def _is_object(obj: object) -> TypeIs[dict[object, object]]:
    """Whether a value is written as an object.  A `dict` exactly: a mapping
    that is not one has no declared spelling, and `frozendict` is the writer's
    own business."""
    return isinstance(obj, dict)


# Both of these are written as arrays; they are asked apart only because a
# value with no order of its own has to be given one first.
def _is_ordered(obj: object) -> TypeIs[Iterable[object]]:
    """Whether a value is a run of values in a meaningful order."""
    return isinstance(obj, (list, tuple))


def _is_unordered(obj: object) -> TypeIs[Set[object]]:
    """Whether a value is a run of values whose order carries no meaning."""
    return isinstance(obj, (set, frozenset))


@functools.cache
def _tag_of(cls: type) -> str | None:
    """A class's wire tag: `__sop_tag__` if set, otherwise its own name.
    Setting it to `None` is how a class asks to be carried as a bare object.

    `None` for everything the format spells some other way, so that what this
    names is exactly what a union can discriminate on: the builtins, because
    `str` and `list` are how the format's own types are spelled rather than
    user classes waiting for a tag; enums, which are carried as symbols; and
    `Symbol`, `Tagged` and `Any`, which match on kind rather than on a tag.

    A class, not a shape: only a class can name a tag, and only a class is
    the hashable key this is cached on."""
    if cls.__module__ in ("builtins", "typing"):
        return None
    if issubclass(cls, (enum.Enum, Symbol, Tagged)):
        return None
    return getattr(cls, "__sop_tag__", cls.__name__)


def _array_tag(cls: type) -> str | None:
    """The tag an array-shaped value is carried under, so that what the data
    was stays visible even though every one of them spells as an array.

    `list` and `tuple` carry none: they *are* the format's array.  A set has
    no native spelling, so it is carried under `Set`.  Anything else -- a
    named tuple, a subclass of any of these -- is carried under its own tag,
    like every other class.  Read and written from this one place."""
    if cls is list or cls is tuple:
        return None
    if cls is set or cls is frozenset:
        return "Set"
    return _tag_of(cls)


def _scalar_tag(cls: type) -> str | None:
    """The tag for the tagged-*string* carrier: explicit `__sop_tag__` only.

    The name default carries dataclasses, whose fields are declared and read
    back.  An arbitrary class declares nothing, and defaulting it through
    `str(obj)` would spell a class that never opted in as
    `Name "<object at 0x…>"` -- so without a named tag it is not carried."""
    tag = getattr(cls, "__sop_tag__", None)
    return tag if isinstance(tag, str) else None


@functools.cache
def _hints(cls: type) -> dict[str, TypeForm[Any]]:
    """Each field's declared type -- which is to say its shape -- with PEP 649
    annotations evaluated.  Cached: resolution walks the MRO, and decoding
    asks once per instance."""
    hints: dict[str, TypeForm[Any]] = typing.get_type_hints(cls, include_extras=True)
    return hints


@functools.cache
def _fields(shape: type) -> tuple[tuple[dataclasses.Field[object], str], ...]:
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


def decode[T](value: Value, shape: TypeForm[T], path: str = "$") -> T:
    """Read a parsed value back as `shape`, raising `ShapeError` -- naming the
    path -- for anything that does not fit.

    The shape is the type: what comes back is a `T` because `shape` said so.
    This is the one place the dynamic answer below is read as a static one."""
    decoded: T = _decode(value, shape, path)
    return decoded


def _decode(value: Value, shape: TypeForm[Any], path: str = "$") -> Any:
    """One step of `decode`.  A shape is an alias for another shape, a
    parameterised type, or a plain class; each is read by the helper below
    that knows about that kind, so neither of them dispatches on both."""
    if isinstance(shape, typing.TypeAliasType):
        # A `type X = ...` alias -- the SDK's own `Value` included -- denotes
        # its right-hand side.
        return _decode(value, shape.__value__, path)
    if shape is Any:
        return value
    if shape is None:
        # `deleted_at: None` and `deleted_at: NoneType` are one shape; from
        # here on it is spelled as the class, so only classes are left.
        shape = types.NoneType
    if (origin := _origin(shape)) is not None:
        return _decode_parameterised(value, shape, origin, path)
    if _is_class(shape):
        return _decode_plain(value, shape, path)
    raise ShapeError(path, f"unsupported shape {shape!r}")


def _decode_parameterised(value: Value, shape: TypeForm[Any], origin: type, path: str) -> Any:
    """A shape built from another -- `list[T]`, `A | B` -- read by what it is
    built from and what it is built of.

    Untyped values are immutable, and the shape declares whether the decoded
    result mutates: `list[T]` and `tuple[T, ...]` read the same array, `dict`
    and `frozendict` the same object."""
    match origin, _args(shape):
        case types.UnionType, members:
            return _union(value, members, path)
        case builtins.list, (item,):
            return [_decode(v, item, f"{path}[{i}]") for i, v in enumerate(_array(value, path))]
        case builtins.tuple, (item, builtins.Ellipsis):
            return tuple(
                _decode(v, item, f"{path}[{i}]") for i, v in enumerate(_array(value, path))
            )
        case builtins.tuple, _:
            # `tuple[int, str]` is a row type, which a sop array does not model.
            raise ShapeError(
                path,
                f"unsupported shape {shape!r}: only `tuple[T, ...]` reads an array",
            )
        case ((builtins.set | builtins.frozenset), (item,)):
            tag = _array_tag(origin)
            if not isinstance(value, Tagged) or value.tag != tag:
                raise ShapeError(
                    path, f"expected an array tagged `{tag}`, found {_describe(value)}"
                )
            decoded = (_decode(v, item, f"{path}[]") for v in _array(value.value, path))
            return set(decoded) if origin is set else frozenset(decoded)
        case _, (item,) if origin is Tagged:
            if not isinstance(value, Tagged):
                raise ShapeError(path, f"expected a tagged value, found {_describe(value)}")
            return Tagged(value.tag, _decode(value.value, item, path))
        case ((builtins.dict | builtins.frozendict), (builtins.str | typing.Any, item)):
            if not isinstance(value, frozendict):
                raise ShapeError(path, f"expected an object, found {_describe(value)}")
            mapped = {k: _decode(v, item, f"{path}.{k}") for k, v in value.items()}
            return mapped if origin is dict else frozendict(mapped)
        case ((builtins.dict | builtins.frozendict), _):
            # An object's keys are strings; a shape claiming otherwise would
            # decode to something its own annotation contradicts.
            raise ShapeError(path, f"unsupported shape {shape!r}: object keys are strings")
    raise ShapeError(path, f"unsupported shape {shape!r}")


def _decode_plain(value: Value, cls: type, path: str) -> Any:
    """A shape that is just a class, read by what kind of class it is."""
    match cls:
        case types.NoneType:
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
            pass  # not one the format spells itself; the rest is below

    if issubclass(cls, enum.Enum):
        return _decode_enum(value, cls, path)
    if cls is Symbol or cls is Tagged:
        if not isinstance(value, cls):
            raise ShapeError(path, f"expected {cls.__name__}, found {_describe(value)}")
        return value
    if dataclasses.is_dataclass(cls):
        return _decode_dataclass(value, cls, path)
    # Any other class with a *declared* tag is carried as a tagged string; the
    # builtins, the enums and the dataclasses are ruled out by now.
    if tag := _scalar_tag(cls):
        return _decode_scalar(value, cls, tag, path)
    raise ShapeError(path, f"unsupported shape {cls!r}")


def _array(value: Value, path: str) -> tuple[Value, ...]:
    if not isinstance(value, tuple):
        raise ShapeError(path, f"expected an array, found {_describe(value)}")
    return value


def _name_of(shape: TypeForm[Any]) -> str:
    """A shape's name for an error message.  Not every shape is a class, and
    only a class is guaranteed to have `__name__`."""
    return getattr(shape, "__name__", None) or str(shape)


def _union(value: Value, members: tuple[TypeForm[Any], ...], path: str) -> Any:
    if types.NoneType in members and value is None:
        return None
    members = tuple(m for m in members if m is not types.NoneType)
    if len(members) == 1:
        return _decode(value, members[0], path)

    # A member's wire tag is the union's discriminant, and `_tag_of` names
    # exactly the members that are carried tagged.
    tagged: dict[str, TypeForm[Any]] = {}
    for member in members:
        # A member that is not a class names no tag, so it cannot be what the
        # union keys on -- `list[int]` and `Any` among them.
        if not _is_class(member):
            continue
        cls: type = member  # hashable, as the cached lookup requires
        if name := _tag_of(cls):
            if name in tagged:
                # A shared discriminant is a schema bug; falling back to
                # trying members in order would hide it until it bit.
                raise ShapeError(
                    path,
                    f"union members {_name_of(tagged[name])} and {_name_of(member)} "
                    f"share the tag `{name}`",
                )
            tagged[name] = member

    if len(tagged) == len(members):
        # Every member is tagged, so the tag decides and a mismatch is a real
        # error rather than one failed attempt out of several.
        expected = ", ".join(f"`{t}`" for t in sorted(tagged))
        if not isinstance(value, Tagged):
            raise ShapeError(path, f"expected a value tagged {expected}, found {_describe(value)}")
        if (chosen := tagged.get(value.tag)) is not None:
            return _decode(value, chosen, path)
        raise ShapeError(path, f"unknown tag `{value.tag}`; expected one of {expected}")

    # In a mixed union a recognised tag still decides; anything else tries
    # each member in order.
    if isinstance(value, Tagged) and (chosen := tagged.get(value.tag)) is not None:
        return _decode(value, chosen, path)

    reasons: list[str] = []
    for candidate in members:
        try:
            return _decode(value, candidate, path)
        except ShapeError as exc:
            reasons.append(exc.message)
    raise ShapeError(path, "no union member matched: " + "; ".join(reasons))


def _decode_scalar[T](value: Value, shape: type[T], tag: str, path: str) -> T:
    if not isinstance(value, Tagged) or value.tag != tag:
        raise ShapeError(path, f"expected a value tagged `{tag}`, found {_describe(value)}")
    if not isinstance(value.value, str):
        raise ShapeError(path, f"`{tag}` carries a string, found {_describe(value.value)}")
    # The class builds itself from its own spelling unless it says otherwise.
    parse: Callable[[str], T] = getattr(shape, "__sop_parse__", shape)
    try:
        return parse(value.value)
    except (ValueError, ArithmeticError) as exc:
        raise ShapeError(path, f'`{tag} "{value.value}"` is not valid: {exc}') from None


def _decode_enum[E: enum.Enum](value: Value, shape: type[E], path: str) -> E:
    if not isinstance(value, Symbol):
        raise ShapeError(path, f"expected a symbol, found {_describe(value)}")
    for member in shape:
        if _enum_spelling(member) == value.name:
            return member
    spelled = ", ".join(f"`{_enum_spelling(m)}`" for m in shape)
    raise ShapeError(path, f"`{value.name}` is not one of {spelled}")


def _decode_dataclass[T](value: Value, shape: type[T], path: str) -> T:
    # The lookups below are cached on the class itself, and a cache key is
    # asked to be hashable -- which a bare `type` is known to be and a
    # `type[T]` is not, T being free.
    cls: type = shape
    if tag := _tag_of(cls):
        if not isinstance(value, Tagged):
            raise ShapeError(path, f"expected a value tagged `{tag}`, found {_describe(value)}")
        if value.tag != tag:
            raise ShapeError(path, f"expected tag `{tag}`, found `{value.tag}`")
        value = value.value

    if not isinstance(value, frozendict):
        raise ShapeError(path, f"expected an object, found {_describe(value)}")

    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for field, key in _fields(cls):
        if not field.init:
            continue
        if key in value:
            kwargs[field.name] = _decode(value[key], hints[field.name], f"{path}.{key}")
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


type Spelled = Symbol | Tagged[object] | frozendict[object, object] | tuple[object, ...]
"""What `convert` answers: a sop value whose *head* is spelled, its children
left as they were for the traversal to reach in turn.  `object`, not `Any`,
because those children are values of a type not yet known -- which is a fact
about them, where `Any` would be a decision to stop checking."""


def convert(obj: object) -> Spelled:
    """One step of writing: spell the head of an object the core cannot
    classify, leaving its children untouched.  The core calls this from inside
    its single traversal and continues in place, so the graph is walked
    exactly once.

    The core knows only the immutable values reading produces, so the mutable
    counterparts are frozen here and spell identically.  Past that, precedence
    mirrors reading: an enum is a symbol, a dataclass is an object, and any
    other class with a tag is a tagged string spelled with `str(obj)`."""
    if isinstance(obj, enum.Enum):
        return Symbol(_enum_spelling(obj))
    # Every remaining question is asked of the class.  A dataclass *class*
    # handed to `dumps` is an instance of `type`, which is not a dataclass, so
    # it falls through to the error rather than spelling its own fields.
    cls = type(obj)
    if dataclasses.is_dataclass(cls):
        body: dict[object, object] = {key: getattr(obj, f.name) for f, key in _fields(cls)}
        tag = _tag_of(cls)
        return Tagged(tag, frozendict(body)) if tag else frozendict(body)
    if tag := _scalar_tag(cls):
        return Tagged(tag, str(obj))
    if _is_object(obj):
        return frozendict(obj)
    if _is_unordered(obj):
        # Sorted because an array is ordered and a set is not: one order is
        # chosen and kept to, or the same value would not write the same twice.
        return _spell_array(sorted(obj, key=repr), cls)
    if _is_ordered(obj):
        return _spell_array(obj, cls)
    raise ShapeError("$", f"cannot encode {cls.__name__}")


def _spell_array[T](items: Iterable[T], cls: type) -> Tagged[tuple[T, ...]] | tuple[T, ...]:
    """A run of values as an array, tagged with what it was unless it is the
    format's own array already."""
    spelled = tuple(items)
    tag = _array_tag(cls)
    return Tagged(tag, spelled) if tag else spelled
