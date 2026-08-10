"""sop — the sop interchange format, for Python.

Two functions.

    order  = sop.loads[Order](text)          # Order
    events = sop.loads[list[Event]](text)    # list[Event]
    raw    = sop.loads[Any](text)            # whatever is there

    text = sop.dumps(order)

Reading takes a shape because text carries no type; writing does not, because
the object does.  There is no shapeless `loads`: `loads[Any]` is the escape
hatch and it has to be written down.

One set of values, immutable, is what reading produces and what the core
writes; the mutable counterparts are frozen on the way out and spell
identically:

    sop object   <->  frozendict[str, Value]   (dict is written too)
    sop array    <->  tuple[Value, ...]        (list is written too)
    sop string   <->  str
    sop number   <->  int | float
    sop symbol   <->  True | False | None | Symbol(name)
    sop tagged   <->  Tagged(tag, value)

The core has no booleans and no null -- `true`, `false` and `null` are ordinary
symbols to it.  Mapping them onto `True`, `False` and `None` is this SDK's
decision, which is why `Symbol("null")` is refused: `null` already has a
spelling here.

`Symbol` and `Tagged` are distinct types, so a symbol is never equal to a
string and a tagged value is never equal to its payload.  Past that, comparing
Python objects is Python's business.

The format itself is implemented once, in Rust (`sop._core`).  There is no
pure-Python parser and no fallback.
"""

from typing import TypeForm

from . import _shape
from ._core import ParseError, ShapeError, SopError, Symbol, Tagged
from ._core import dumps as _dumps
from ._core import loads as _loads
from ._shape import Value

__all__ = [
    "ParseError",
    "ShapeError",
    "SopError",
    "Symbol",
    "Tagged",
    "Value",
    "dumps",
    "loads",
]


class _TypedLoads[T]:
    """`loads[Shape]` — reads a document and returns `Shape`."""

    __slots__ = ("_shape",)

    def __init__(self, shape: TypeForm[T]) -> None:
        self._shape = shape

    def __call__(self, text: str) -> T:
        return _shape.decode(_loads(text), self._shape)


class _Loads:
    __slots__ = ()

    def __getitem__[T](self, shape: TypeForm[T]) -> _TypedLoads[T]:
        return _TypedLoads(shape)


def dumps(value: object) -> str:
    """Write a value as sop text.

    The core writer already understands every value `loads` produces, so one
    that came from `loads` is written without touching Python.  Anything else
    -- a dataclass, an enum, a set, a `dict` or a `list` -- is spelled by the
    shape layer's `convert`, one call per unrecognised object, inside the same
    single traversal.
    """
    return _dumps(value, _shape.convert)


loads = _Loads()
