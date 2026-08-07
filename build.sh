#!/bin/sh
# Builds the extension and installs it into the active virtual environment.
#
# Requires a Rust toolchain, Python 3.15 and maturin:
#
#     uv venv --python 3.15 && . .venv/bin/activate
#     uv pip install maturin
#
# The extension is built against the interpreter it will run on -- not the
# limited API -- because the SDK supports one Python version.
set -e

maturin develop --release
python - <<'PY'
from typing import Any
import sop
print("sop ok:", sop.loads[Any]("ok"))
PY
