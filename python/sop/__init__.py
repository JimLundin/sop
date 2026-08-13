"""sop — the sop interchange format, for Python.

Two functions.

    order  = sop.loads[Order](text)          # Order
    events = sop.loads[list[Event]](text)    # list[Event]
    raw    = sop.loads[Any](text)            # whatever is there

    text = sop.dumps(order)

Reading takes a shape because text carries no type; writing does not, because
the object does.  `loads` itself is the read at `Value` -- the domain the
parser produces -- so `loads(text)` is typed precisely rather than untyped.
Checking off is a separate decision and still has to be written down:
`loads[Any]` is the escape hatch.

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

from collections.abc import Callable
from typing import TypeForm

from . import _encode
from ._core import ParseError, ShapeError, SopError, Symbol, Tagged
from ._core import dumps as _dumps
from ._core import loads as _loads
from ._decode import decoder_for as _decoder_for
from ._ir import Value

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


class _Loads[T]:
    """`loads[Shape]` — reads a document and returns `Shape`.

    One class, carrying the shape it reads at.  `loads` is this at `Value`,
    which is what a document says when nothing else is asked of it, so
    `loads(text)` is not a shapeless read: it is a read at the value domain,
    spelled by being the default rather than by being written out.  Every
    other shape is the same object under a different parameter, which is why
    `read = loads[Order]` is a value like any other."""

    __slots__ = ("_read", "_shape")

    def __init__(self, shape: TypeForm[T]) -> None:
        self._shape = shape
        self._read: Callable[[Value], T] | None = None

    def __call__(self, text: str) -> T:
        # Compiled on the first read and kept, so `read = loads[Order]` pays
        # for the shape once however many documents go through it.  Lazily,
        # because subscripting is not reading: `loads[Order]` written and not
        # called should cost nothing.
        if (read := self._read) is None:
            read = self._read = _decoder_for(self._shape)
        try:
            return read(_loads(text))
        except ShapeError as exc:
            # The root, spelled once, where the failure leaves the traversal.
            raise ShapeError("$" + exc.path, exc.message) from None

    def __getitem__[S](self, shape: TypeForm[S]) -> _Loads[S]:
        return _Loads(shape)


def dumps(value: object) -> str:
    """Write a value as sop text.

    The core writer already understands every value `loads` produces, so one
    that came from `loads` is written without touching Python.  Anything else
    -- a dataclass, an enum, a set, a `dict` or a `list` -- is spelled by the
    writing side's `convert`, one call per unrecognised object, inside the same
    single traversal.
    """
    return _dumps(value, _encode.convert)


loads: _Loads[Value] = _Loads(Value)
