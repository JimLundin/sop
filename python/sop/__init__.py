"""sop — the sop interchange format, for Python.

Two functions.

    order  = sop.loads[Order](text)          # Order
    events = sop.loads[list[Event]](text)    # list[Event]
    raw    = sop.loads[Any](text)            # whatever is there

    text = sop.dumps(order)

Reading takes a shape because text carries no type; writing does not, because
the object does.  There is no shapeless `loads`: `loads[Any]` is the escape
hatch and it has to be written down.

Untyped reading produces immutable values only; writing also accepts their
mutable counterparts:

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

from typing import Any as _Any, TypeForm as _TypeForm

from . import _shape
from ._core import SopError, Symbol, Tagged
from ._core import dumps as _dumps, loads as _loads
from ._shape import ShapeError

__all__ = ["ShapeError", "SopError", "Symbol", "Tagged", "Value", "dumps", "loads"]

type Value = (
    None | bool | int | float | str | Symbol | Tagged | tuple[Value, ...] | frozendict[str, Value]
)
"""The closed set untyped reading produces — immutable throughout.  Mutation
is opt-in: a shape such as `list[T]` or `dict[str, V]` declares it, and the
decoded result is freshly built.  `loads[Value]` is the escape hatch with its
result typed precisely; `loads[Any]` reads the same way and types the result
as `Any`.  Kept in step with the alias in `_core.pyi`."""


class _TypedLoads[T]:
    """`loads[Shape]` — reads a document and returns `Shape`."""

    __slots__ = ("_shape",)

    def __init__(self, shape: _shape.Shape) -> None:
        self._shape = shape

    def __call__(self, text: str) -> T:
        return _shape.decode(_loads(text), self._shape)

    def __repr__(self) -> str:
        # `list[int].__name__` is "list", which loses the parameter; only a
        # plain class is better described by its name than by str().
        shape = self._shape
        name = shape.__name__ if isinstance(shape, type) else shape
        return f"sop.loads[{name!s}]"


class _Loads:
    __slots__ = ()

    def __getitem__[T](self, shape: _TypeForm[T]) -> _TypedLoads[T]:
        return _TypedLoads(shape)

    def __repr__(self) -> str:
        return "sop.loads"


def dumps(value: _Any) -> str:
    """Write a value as sop text.

    The core writer already understands every sop value, so a value that came
    from `loads` is written without touching Python.  Anything else -- a
    dataclass, an enum, a set -- is spelled by the shape layer's `convert`,
    one call per unrecognised object, inside the same single traversal.
    """
    return _dumps(value, _shape.convert)


loads = _Loads()
