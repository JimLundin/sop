"""Type stubs for the Rust core extension module."""

from typing import Any, final

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
    value: Any
    def __init__(self, tag: str, value: Any) -> None: ...

def loads(text: str) -> Any: ...
def dumps(value: Any, indent: int | None = ...) -> str: ...
