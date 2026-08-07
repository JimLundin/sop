"""Property tests, driven by Hypothesis.

These replace a hand-written generator and random-value harness.  Hypothesis
shrinks failures to a minimal case, remembers them between runs, and biases
towards the edges (empty containers, surrogate-adjacent text, subnormal floats)
that a uniform generator reaches only by luck.
"""

import enum
import math
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import sop

settings.register_profile(
    "sop",
    max_examples=400,
    deadline=None,  # the first call pays for the extension's import
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("sop")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Identifiers, minus the three names the SDK spells with Python values.  Kept
# ASCII: the Unicode classes are the core's business and the corpus covers them.
identifiers = st.from_regex(r"\A[A-Za-z_$][A-Za-z0-9_$]{0,7}\Z").filter(
    lambda name: name not in ("true", "false", "null")
)

# The core's numeric domain, not Python's.
integers = st.integers(min_value=-(2**63), max_value=2**63 - 1)
floats = st.floats(allow_nan=False, allow_infinity=False)

scalars = (
    st.none()
    | st.booleans()
    | integers
    | floats
    | st.text(max_size=12)
    | st.builds(sop.Symbol, identifiers)
)


def _taggable(value: Any) -> bool:
    """A tag cannot be applied to a bare symbol, so `Tagged` refuses `None`,
    booleans and `Symbol` payloads."""
    return not (value is None or isinstance(value, (bool, sop.Symbol)))


def containers(children: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    # Untyped reading produces immutable values, so the round-trip strategies
    # generate those; the mutable counterparts are covered by the
    # write-equivalence property below.
    return (
        st.lists(children, max_size=4).map(tuple)
        | st.dictionaries(st.text(max_size=8), children, max_size=4).map(frozendict)
        | st.builds(sop.Tagged, identifiers, children.filter(_taggable))
    )


values = st.recursive(scalars, containers, max_leaves=12)


def _thaw(value: Any) -> Any:
    """The mutable counterpart: tuples as lists, frozendicts as dicts."""
    if isinstance(value, tuple):
        return [_thaw(v) for v in value]
    if isinstance(value, frozendict):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, sop.Tagged):
        return sop.Tagged(value.tag, _thaw(value.value))
    return value


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


@given(values)
def test_a_value_survives_a_round_trip(value):
    assert sop.loads[Any](sop.dumps(value)) == value


@given(values)
def test_value_is_total_over_untyped_reading(value):
    # Everything untyped reading can produce fits the `Value` shape.
    assert sop.loads[sop.Value](sop.dumps(value)) == value


@given(values)
def test_writing_is_a_fixed_point(value):
    once = sop.dumps(value)
    assert sop.dumps(sop.loads[Any](once)) == once


@given(values)
def test_mutable_counterparts_write_identically(value):
    # dict, list and their nested mixtures spell exactly as their immutable
    # counterparts; only reading is immutable-only.
    assert sop.dumps(_thaw(value)) == sop.dumps(value)


@given(values.filter(_taggable))
def test_a_tag_is_not_transparent(value):
    # A tagged value is never equal to its payload.
    assert sop.Tagged("t", value) != value


# ---------------------------------------------------------------------------
# Robustness: anything at all goes in, only SopError comes out
# ---------------------------------------------------------------------------


@given(st.text(max_size=60))
def test_arbitrary_text_never_crashes(text):
    try:
        sop.loads[Any](text)
    except sop.SopError:
        pass


@given(
    st.text(
        alphabet=st.sampled_from(list('{}[](),:"\'/*\\+-.0123456789eE_$ \t\n\r\u2028')),
        max_size=40,
    )
)
def test_parser_shaped_noise_never_crashes(text):
    # Drawn from the characters that carry meaning, so far more of these reach
    # deep into the parser than uniform text would.
    try:
        parsed = sop.loads[Any](text)
    except sop.SopError:
        return
    once = sop.dumps(parsed)
    assert sop.dumps(sop.loads[Any](once)) == once


# ---------------------------------------------------------------------------
# Typed round trips, one per shape
# ---------------------------------------------------------------------------


class Colour(enum.Enum):
    Red = "Red"
    Blue = "azul"


@dataclass
class Point:
    __sop_tag__ = None
    x: int
    y: float


@dataclass
class Named:
    label: str  # tagged with its own name


class Iban(str):
    __sop_tag__ = "iban"


@pytest.mark.parametrize(
    "shape, strategy",
    [
        (int, integers),
        (float, floats),
        (str, st.text(max_size=12)),
        (bool, st.booleans()),
        (list[int], st.lists(integers, max_size=5)),
        (tuple[int, ...], st.lists(integers, max_size=5).map(tuple)),
        (set[str], st.sets(st.text(max_size=6), max_size=5)),
        (frozenset[str], st.frozensets(st.text(max_size=6), max_size=5)),
        (dict[str, int], st.dictionaries(st.text(max_size=6), integers, max_size=5)),
        (dict[str, list[int]], st.dictionaries(
            st.text(max_size=6), st.lists(integers, max_size=3), max_size=3)),
        (int | None, st.none() | integers),
        (Colour, st.sampled_from(Colour)),
        (Iban, st.builds(Iban, st.text(max_size=8))),
        (Point, st.builds(Point, integers, floats)),
        (Named, st.builds(Named, st.text(max_size=8))),
        (list[Point | Named], st.lists(
            st.builds(Point, integers, floats) | st.builds(Named, st.text(max_size=6)),
            max_size=4)),
        (sop.Tagged, st.builds(sop.Tagged, identifiers, st.text(max_size=8))),
    ],
)
def test_typed_round_trip(shape, strategy):
    @given(strategy)
    @settings(max_examples=150, deadline=None)
    def check(value):
        assert sop.loads[shape](sop.dumps(value)) == value

    check()


# ---------------------------------------------------------------------------
# Numbers, where Python and the format disagree about what exists
# ---------------------------------------------------------------------------


@given(integers)
def test_integers_are_exact_in_range(value):
    assert sop.loads[int](sop.dumps(value)) == value


@given(st.integers(min_value=2**63) | st.integers(max_value=-(2**63) - 1))
def test_integers_out_of_range_are_refused(value):
    with pytest.raises(sop.SopError, match="out of range"):
        sop.dumps(value)


@given(floats)
def test_floats_keep_their_value(value):
    # Including the sign of zero: a float's spelling always carries a point
    # or an exponent, so nothing is folded into an integer on the way out.
    read = sop.loads[float](sop.dumps(value))
    assert read == value and math.copysign(1, read) == math.copysign(1, value)


@given(floats)
def test_a_float_reads_back_as_a_float(value):
    # Number kind is spelling-determined and writing preserves it.
    read = sop.loads[Any](sop.dumps(value))
    assert isinstance(read, float) and read == value


@given(st.sampled_from([math.inf, -math.inf, math.nan]))
def test_non_finite_floats_have_no_spelling(value):
    with pytest.raises(sop.SopError, match="no sop representation"):
        sop.dumps(value)


# ---------------------------------------------------------------------------
# The native types refuse what the parser could not produce
# ---------------------------------------------------------------------------


@given(identifiers | st.text(max_size=10))
def test_symbol_accepts_exactly_the_identifiers(name):
    try:
        symbol = sop.Symbol(name)
    except ValueError:
        return
    # Anything Symbol accepted must survive a round trip as a symbol.
    parsed_back = sop.loads[sop.Symbol](sop.dumps(symbol))
    assert parsed_back == symbol


@given(identifiers | st.text(max_size=10), integers)
def test_tagged_accepts_exactly_the_identifier_tags(tag, payload):
    try:
        tagged = sop.Tagged(tag, payload)
    except ValueError:
        return
    assert sop.loads[Any](sop.dumps(tagged)) == tagged
