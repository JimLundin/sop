"""What each type checker makes of `tests/typing_corpus.py`.

The corpus is the claim; this is where every checker is asked whether it
agrees, and held to the markers the corpus carries. A line with no marker has
to be clean under all of them, so a shape that reads fine here and errors for
somebody running a different checker is a failure rather than a surprise
report from a user.

The markers are not a wish list. Where a checker's PEP 747 support is not
finished, the marker says so and this file pins it: if the checker fixes it,
the expectation stops matching and the caveat can be dropped, which is the
only way a caveat ever gets dropped.

Checkers that are not installed are skipped rather than assumed clean --
`pytest -rs` says which. CI installs mypy and pyright.
"""

import io
import json
import re
import shutil
import subprocess
import sys
import tokenize
from collections.abc import Callable
from pathlib import Path

import pytest
import typing_corpus

CORPUS = Path(__file__).parent / "typing_corpus.py"
ROOT = CORPUS.parent.parent

type Diagnostics = set[tuple[int, str]]
"""A checker's reading of the corpus: the line, and the rule it names there."""


def _run(command: list[str], cwd: Path = ROOT) -> str:
    """A checker's stdout.  A non-zero exit means it found something, which is
    the ordinary case here, so the status is not what is looked at -- but
    nothing on stdout means it never ran, and that is worth saying plainly
    rather than failing later on unreadable output."""
    finished = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if not finished.stdout.strip():
        raise AssertionError(
            f"{' '.join(command)} said nothing on stdout:\n{finished.stderr}"
        )
    return finished.stdout


def _mypy(tmp: Path) -> Diagnostics:
    # `-O json` rather than the human format: the code is a field rather than
    # something to pull back out of a message with a regex.
    out = _run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "-O",
            "json",
            "--cache-dir",
            str(tmp / "mypy"),
            str(CORPUS),
        ]
    )
    found: Diagnostics = set()
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        report = json.loads(line)
        if report["severity"] == "error" and Path(report["file"]).name == CORPUS.name:
            found.add((report["line"], report["code"]))
    return found


def _pyright(experimental: bool) -> Callable[[Path], Diagnostics]:
    """pyright twice: once as this project configures it, and once as somebody
    else's default would.  PEP 747's implicit conversion -- what lets a bare
    type expression stand where a `TypeForm` is asked for -- is behind
    `enableExperimentalFeatures`, so the two readings differ, and the second
    is the one a user of this library gets."""

    def check(tmp: Path) -> Diagnostics:
        # A copy, because pyright reads its settings for the files under the
        # config's own directory and the corpus is not under this one.  The
        # copy keeps its name and its line numbers; the interpreter it is
        # checked against is still this project's, named absolutely.
        shutil.copy(CORPUS, tmp / CORPUS.name)
        (tmp / "pyrightconfig.json").write_text(
            json.dumps(
                {
                    "typeCheckingMode": "strict",
                    "pythonVersion": "3.15",
                    "venvPath": str(ROOT),
                    "venv": ".venv",
                    "enableExperimentalFeatures": experimental,
                }
            )
        )
        out = _run(["pyright", "--outputjson", "--project", str(tmp)], cwd=tmp)
        found: Diagnostics = set()
        for report in json.loads(out)["generalDiagnostics"]:
            if report["severity"] != "error":
                continue
            # A general error carries no rule; name it after its severity so
            # that it still has something to be recorded under.
            found.add(
                (report["range"]["start"]["line"] + 1, report.get("rule", "error"))
            )
        return found

    return check


def _pyrefly(tmp: Path) -> Diagnostics:
    out = _run(
        [
            "pyrefly",
            "check",
            "--output-format",
            "json",
            "--python-interpreter-path",
            sys.executable,
            str(CORPUS),
        ]
    )
    found: Diagnostics = set()
    for report in json.loads(out)["errors"]:
        if report["severity"] == "error":
            found.add((report["line"], report["name"]))
    return found


CHECKERS: dict[str, tuple[str, Callable[[Path], Diagnostics]]] = {
    # name as a marker spells it: executable, how to run it
    "mypy": ("mypy", _mypy),
    "pyright": ("pyright", _pyright(experimental=True)),
    "pyright-default": ("pyright", _pyright(experimental=False)),
    "pyrefly": ("pyrefly", _pyrefly),
}
"""The checkers this file knows how to ask.

`ty` is missing on purpose: its stdlib stubs have no `typing.TypeForm` yet, so
it cannot read the API's signature at all and every line of the corpus would
be recorded as an error about the same one missing name.  It belongs here the
day that lands."""

_DEFINITION = re.compile(r"#\s*expect-set:\s*([\w-]+)\s*$")
_MARKER = re.compile(r"#\s*expect:\s*(.*)$")
_CLAIM = re.compile(r"([\w-]+)(?:\[([^\]]*)\])?")


def _comments(source: str) -> list[tuple[int, bool, str]]:
    """Every real comment: its line, whether it has the line to itself, and
    what it says.  Tokenised rather than searched for, so that the example in
    this file's own docstring stays a string."""
    found: list[tuple[int, bool, str]] = []
    lines = source.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            line, column = token.start
            alone = not lines[line - 1][:column].strip()
            found.append((line, alone, token.string))
    return found


def _sets(comments: list[tuple[int, bool, str]]) -> dict[str, dict[str, set[str]]]:
    """The names a marker can use, each one a checker-to-rules mapping.

    A definition is `# expect-set: name` and the `#` lines under it, which are
    claims -- `pyright[rule,rule]` -- or the name of a set defined before it,
    so that one gap can be described as another plus a checker.  A blank
    comment line ends it."""
    defined: dict[str, dict[str, set[str]]] = {}
    name: str | None = None
    for _, _, text in comments:
        if (definition := _DEFINITION.search(text)) is not None:
            name = definition.group(1)
            defined[name] = {}
            continue
        body = text.lstrip("#").strip()
        if name is None:
            continue
        if not body:
            name = None
            continue
        for checker, rules in _CLAIM.findall(body):
            if not rules and checker in defined:
                for other, inherited in defined[checker].items():
                    defined[name].setdefault(other, set()).update(inherited)
            elif rules:
                defined[name].setdefault(checker, set()).update(
                    rule.strip() for rule in rules.split(",")
                )
    return defined


def _expected() -> dict[str, Diagnostics]:
    """What the corpus says each checker reports, read out of its markers.

    A marker sits at the end of the line it is about, or on a line of its own
    directly above one -- the second spelling is for the lines that have no
    room left."""
    source = CORPUS.read_text()
    lines = source.splitlines()
    comments = _comments(source)
    defined = _sets(comments)
    expected: dict[str, Diagnostics] = {name: set() for name in CHECKERS}
    for line, alone, text in comments:
        if (marker := _MARKER.search(text)) is None:
            continue
        # A marker on a line of its own is about the next line that has code.
        while alone and line < len(lines) and not lines[line - 1].split("#")[0].strip():
            line += 1
        for checker, rules in _CLAIM.findall(marker.group(1)):
            named = (
                defined.get(checker, {})
                if not rules
                else {checker: {rule.strip() for rule in rules.split(",")}}
            )
            for reported, rule_names in named.items():
                expected.setdefault(reported, set()).update(
                    (line, rule) for rule in rule_names
                )
    return expected


def _difference(found: Diagnostics, expected: Diagnostics) -> str:
    """A failure worth reading: what was reported and not written down, what
    was written down and not reported, and the marker that would fix it."""
    lines = []
    for label, diagnostics in (
        ("reported but not expected", found - expected),
        ("expected but not reported", expected - found),
    ):
        for number, rule in sorted(diagnostics):
            source = CORPUS.read_text().splitlines()[number - 1].strip()
            lines.append(f"  {label}: line {number} [{rule}]  {source}")
    return "\n".join(lines)


@pytest.mark.parametrize("name", list(CHECKERS))
def test_a_checker_reads_the_corpus_the_way_the_corpus_says(
    name: str, tmp_path: Path
) -> None:
    executable, check = CHECKERS[name]
    if shutil.which(executable) is None:
        pytest.skip(f"{executable} is not installed")
    found = check(tmp_path)
    expected = _expected()[name]
    assert found == expected, f"{name} disagrees with the corpus:\n" + _difference(
        found, expected
    )


def test_every_marker_names_a_checker_that_is_asked() -> None:
    # A marker for a checker nobody runs is a claim nothing checks.
    assert set(_expected()) == set(CHECKERS)


@pytest.mark.parametrize(
    "case", typing_corpus.CASES, ids=[case.__name__ for case in typing_corpus.CASES]
)
def test_the_corpus_runs(case: Callable[[], None]) -> None:
    """The run-time half: every line the checkers read is a line that works."""
    case()
