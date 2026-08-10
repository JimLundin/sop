"""Type stubs for the Rust core extension module, kept by hand: `Tagged`'s
parameter and its default are beyond what stub generation could emit."""

from collections.abc import Callable
from typing import final

# The value domain the core produces.  It is spelled where it can also exist
# at run time -- see `_shape.Value` -- and re-exported here under its own name,
# so the signatures below mean by `Value` exactly what `sop.Value` is.  The
# cycle this makes is a checker's to resolve; a stub never runs.
from ._shape import Value as Value

class SopError(ValueError):
    """A value has no sop spelling.  Raised on its own while writing, where
    there is no position and no path; the two subclasses below add the one
    each reading direction has."""

    @property
    def message(self) -> str: ...
    def __init__(self, message: str) -> None: ...

class ParseError(SopError):
    """The text is not sop, at a position in it."""

    @property
    def line(self) -> int: ...
    @property
    def column(self) -> int: ...
    def __init__(self, message: str, line: int, column: int) -> None: ...

class ShapeError(SopError):
    """The document parsed, but does not have the requested shape, at a path
    in it."""

    @property
    def path(self) -> str: ...
    def __init__(self, path: str, message: str) -> None: ...

@final
class Symbol:
    # Read-only: the native types are frozen, and saying so is what lets
    # `Tagged` be covariant in its payload.
    @property
    def name(self) -> str: ...
    def __init__(self, name: str) -> None: ...
    def __hash__(self) -> int: ...

@final
class Tagged[V = Value]:
    """A tag over a payload, generic in the payload.

    The default is `Value`, which is what reading produces and so what a bare
    `Tagged` means.  There is deliberately no bound: the writer builds a
    `Tagged[object]` whose payload is still an unconverted Python object,
    spelled later in the same traversal.  What the payload may *not* be --
    `None`, a bool, a `Symbol` -- is not a type the parameter could exclude,
    so the constructor refuses those at runtime instead.
    """

    @property
    def tag(self) -> str: ...
    @property
    def value(self) -> V: ...
    def __init__(self, tag: str, value: V) -> None: ...
    def __hash__(self) -> int: ...

def loads(text: str) -> Value: ...
def dumps(value: object, convert: Callable[[object], object]) -> str: ...
