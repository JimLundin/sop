"""Type stubs for the Rust core extension module, kept by hand: the recursive
`Value` alias and the typed `convert` callable are beyond what stub
generation could emit."""

from collections.abc import Callable
from typing import Any, final

# The closed set the core can decode.  Kept in step with `sop.Value`, which
# cannot be imported here without a cycle through the package.
type Value = (
    None | bool | int | float | str | Symbol | Tagged | tuple[Value, ...] | frozendict[str, Value]
)

class SopError(ValueError):
    message: str
    line: int
    column: int

@final
class Symbol:
    name: str
    def __init__(self, name: str) -> None: ...
    def __hash__(self) -> int: ...

@final
class Tagged:
    tag: str
    value: Value
    # Any payload but a bare symbol, checked at construction.  Wider than
    # `Value` because the writer's `convert` hook builds a `Tagged` whose
    # payload still holds unconverted objects, spelled later in the traversal.
    def __init__(self, tag: str, value: Any) -> None: ...
    def __hash__(self) -> int: ...

def loads(text: str) -> Value: ...
def dumps(value: Any, convert: Callable[[Any], Any]) -> str: ...
