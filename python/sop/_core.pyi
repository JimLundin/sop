"""Type stubs for the Rust core extension module.

Richer than `maturin generate-stubs -F stubs` can currently emit — the
recursive `Value` alias and the typed `convert` callable are beyond the
introspection output — so this file is kept by hand and reviewed against the
generator's output on each PyO3 upgrade.
"""

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
    def __init__(self, tag: str, value: Value) -> None: ...
    def __hash__(self) -> int: ...

def loads(text: str) -> Value: ...
def dumps(
    value: Any, indent: int | None = ..., convert: Callable[[Any], Any] | None = ...
) -> str: ...
