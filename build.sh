#!/bin/sh
# Builds the Rust core and the Python extension, and installs the extension.
#
# Requires a Rust toolchain and Python 3.15.  The extension is built against
# the interpreter it will run on -- not the limited API -- because the SDK
# supports one Python version.
set -e

: "${PYO3_PYTHON:=$(command -v python3.15 || command -v python3)}"
export PYO3_PYTHON
echo "building against $PYO3_PYTHON"

cargo build --release --manifest-path rust/Cargo.toml
cargo build --release --manifest-path bindings/Cargo.toml

case "$(uname -s)" in
    Darwin) built=bindings/target/release/lib_core.dylib ;;
    *) built=bindings/target/release/lib_core.so ;;
esac

cp "$built" python/sop/_core.so
"$PYO3_PYTHON" - <<'PY'
import sys; sys.path.insert(0, "python")
from typing import Any
import sop
print("sop ok:", sop.loads[Any]("ok"))
PY
