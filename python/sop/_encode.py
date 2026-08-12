"""How Python objects are spelled as sop values.

The writing direction.  Reading declares a shape and is compiled from it
(`_analyse`, `_decode`); writing declares nothing, because the object already
knows what it is -- so this is a dispatch on the object in front of it and
there is no shape to compile.

A carried class is tagged with its own name -- a dataclass, whose fields are
declared, and the classes in `_ir.CARRIERS`, which says how to build and spell
them.  An enum is the exception, carried as a symbol.

    @dataclass class Deposit: ...     ->  Deposit { ... }
    Decimal("19.99")                  ->  Decimal "19.99"

Everything else is refused in both directions, with no way to opt in --
a subclass of a carried class included.  User classes and subtypes are things
to add later, and `CARRIERS` is where a user table would go in front of this
one.
"""

import dataclasses
import enum
import functools
from collections.abc import Callable, Iterable, Set
from typing import TypeIs

from ._analyse import enum_spelling
from ._core import SopError, Symbol, Tagged
from ._core import dumps as _dumps
from ._ir import CARRIERS, Value

__all__ = ["Value", "convert"]


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


def _per_class[V](compute: Callable[[type], V]) -> Callable[[type], V]:
    """`functools.cache` for a question asked of a class, with the answer kept
    on the class rather than here.

    Here would be a GC root, and these answers name the class they are about.
    A cache in this module would pin such a class for the life of the process
    -- weak keys do not help, because it is the *value* that refers back.  On
    the class the same reference is an ordinary cycle, which the collector
    takes apart once nothing else refers to the class.

    The slot is named after the function, so two of these cannot collide.
    Read out of `cls.__dict__` and not with `getattr`: an inherited answer
    would be the base's, and a subclass declares its own fields.

    Only ever asked about a dataclass, which is always a Python class, so
    setting the answer cannot be refused the way a static type would refuse
    it."""
    slot = f"__sop{compute.__name__}_cache__"

    @functools.wraps(compute)
    def cached(cls: type) -> V:
        try:
            answer: V = cls.__dict__[slot]
        except KeyError:
            answer = compute(cls)
            setattr(cls, slot, answer)
        return answer

    return cached


def _array_tag(cls: type) -> str | None:
    """The tag an array-shaped value is carried under, so that what the data
    was stays visible even though every one of them spells as an array.

    `list` and `tuple` carry none: they *are* the format's array.  A set has
    no native spelling, so it is carried under `Set`.  Anything else -- a
    named tuple, a subclass of any of these -- is carried under its own tag,
    like every other class.

    Its own name, with nothing else consulted: `convert` has already answered
    for an enum, and the core has already answered for `Symbol` and `Tagged`,
    so what reaches here is a sequence and the only question left is what it
    was called."""
    if cls is list or cls is tuple:
        return None
    if cls is set or cls is frozenset:
        return "Set"
    return cls.__name__


@_per_class
def _fields(shape: type) -> tuple[tuple[dataclasses.Field[object], str], ...]:
    """Each field with its sop key, resolved once per class.  The key is what
    `metadata={"sop": "from"}` names, or the field's own name verbatim."""
    return tuple(
        (f, str(f.metadata.get("sop", f.name))) for f in dataclasses.fields(shape)
    )


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
    mirrors reading: an enum is a symbol, a dataclass is a tagged object, and
    one of the privileged classes is a tagged string.

    A failure here is a `SopError` and not a `ShapeError`, because this is
    handed one object and has no idea where in the graph it sits.  The core
    is the one walking, so the core is what turns it into a `ShapeError` at
    the path it had reached."""
    if isinstance(obj, enum.Enum):
        return Symbol(enum_spelling(obj))
    # Every remaining question is asked of the class.  A dataclass *class*
    # handed to `dumps` is an instance of `type`, which is not a dataclass, so
    # it falls through to the error rather than spelling its own fields.
    cls = type(obj)
    if dataclasses.is_dataclass(cls):
        body: dict[object, object] = {
            key: getattr(obj, f.name) for f, key in _fields(cls)
        }
        return Tagged(cls.__name__, frozendict(body))
    if (carrier := CARRIERS.get(cls)) is not None:
        spell, _ = carrier
        return Tagged(cls.__name__, spell(obj))
    if _is_object(obj):
        return frozendict(obj)
    if _is_unordered(obj):
        # An array is ordered and a set is not, so an order has to be chosen:
        # what the elements themselves spell as.
        return _spell_array(sorted(obj, key=_spelling), cls)
    if _is_ordered(obj):
        return _spell_array(obj, cls)
    raise SopError(f"cannot encode {cls.__name__}")


def _spelling(value: object) -> str:
    """A value's own sop text, which is the order a set is written in.

    The order has to be a property of the values, or the same set would not
    write the same twice -- and `repr` is not one.  A class that does not
    define `__repr__` gets the default, which spells the object's address, so
    sorting on it puts the same four values in a different order in the next
    process.  What a value spells as is the one key that is both total over
    everything writable and the same every time.

    It costs a second pass over the set's elements, which is what buying an
    order that is theirs and not their allocator's costs."""
    try:
        return _dumps(value, convert)
    except SopError:
        # No spelling at all.  The traversal is about to fail on this element
        # anyway, and it is the one that knows where in the document it sits,
        # so the failure is left to it.  Ordering these together is immaterial
        # when the write cannot finish.
        return ""


def _spell_array[T](
    items: Iterable[T], cls: type
) -> Tagged[tuple[T, ...]] | tuple[T, ...]:
    """A run of values as an array, tagged with what it was unless it is the
    format's own array already."""
    spelled = tuple(items)
    tag = _array_tag(cls)
    return Tagged(tag, spelled) if tag else spelled
