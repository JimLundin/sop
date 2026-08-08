"""Type stubs for the Rust core extension module, kept by hand: the recursive
`Value` alias and `Tagged`'s parameter are beyond what stub generation could
emit."""

from collections.abc import Callable
from typing import final

# The closed set the core can decode.  Kept in step with `sop.Value`, which
# cannot be imported here without a cycle through the package.
type Value = (
    None
    | bool
    | int
    | float
    | str
    | Symbol
    | Tagged
    | tuple[Value, ...]
    | frozendict[str, Value]
)

class SopError(ValueError):
    message: str
    line: int
    column: int

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
