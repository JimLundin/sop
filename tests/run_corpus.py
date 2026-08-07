#!/usr/bin/env python3
"""Runs corpus.json against the SDK.

stdout is a deterministic report, one line per case.  Mismatches against the
corpus's expectations go to stderr and set a non-zero exit status.
"""

import pathlib
import sys
from typing import Any

import sop

CORPUS = pathlib.Path(__file__).resolve().parent.parent / "corpus.json"


def main() -> int:
    # Every JSON text is a valid sop document, so the corpus is read
    # by the implementation under test rather than by the json module.
    corpus = sop.loads[Any](CORPUS.read_text(encoding="utf-8"))

    report: list[str] = []
    failures: list[str] = []

    for case in corpus["valid"]:
        name, src, expected = case["name"], case["src"], case["out"]
        try:
            value = sop.loads[Any](src)
        except sop.SopError as exc:
            report.append(f"valid {name} err {exc.line}:{exc.column}: {exc.message}")
            failures.append(f"valid {name}: unexpected error: {exc}")
            continue
        actual = sop.dumps(value)
        report.append(f"valid {name} ok {actual}")
        if actual != expected:
            failures.append(f"valid {name}: expected {expected!r}, got {actual!r}")
        elif sop.loads[Any](actual) != value:
            failures.append(f"valid {name}: output does not re-parse to an equal value")

    for case in corpus["invalid"]:
        name, src = case["name"], case["src"]
        try:
            value = sop.loads[Any](src)
        except sop.SopError as exc:
            report.append(f"invalid {name} err {exc.line}:{exc.column}: {exc.message}")
            continue
        report.append(f"invalid {name} ok {sop.dumps(value)}")
        failures.append(f"invalid {name}: accepted, should have been rejected")

    print("\n".join(report))
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    print(f"sdk: {len(report)} cases, {len(failures)} failures", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
