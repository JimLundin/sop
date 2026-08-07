"""Python-specific expectations.

The corpus checks that the SDK and the core agree about the *format*.  This
file checks the things that are true only on this side of the boundary: how
the format's values become Python ones, what the native types refuse, and
where Python's own semantics need guarding against.
"""

import copy
import pickle
import re
from dataclasses import dataclass
from typing import Any

import pytest

import sop

# ---------------------------------------------------------------------------
# Host types: the SDK's decision, not the format's
# ---------------------------------------------------------------------------


def test_distinguished_symbols_become_python_values():
    # The core has no bool and no null: these arrive as symbols and the SDK
    # chooses what they mean.
    assert sop.loads[Any]("{a: true, b: false, c: null}") == {
        "a": True,
        "b": False,
        "c": None,
    }


def test_and_they_go_back():
    assert sop.dumps({"a": True, "b": False, "c": None}) == "{a:true,b:false,c:null}"


def test_any_other_symbol_stays_a_symbol():
    assert sop.loads[Any]("Active") == sop.Symbol("Active")


def test_bool_is_not_written_as_a_number():
    # bool subclasses int in Python, so the writer has to test it first.
    assert sop.dumps(True) == "true"
    assert sop.dumps(1) == "1"
    assert sop.dumps([True, 1]) == "[true,1]"


# ---------------------------------------------------------------------------
# The native value types refuse what the parser could not produce
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["true", "false", "null"])
def test_symbol_refuses_the_names_python_already_spells(name):
    with pytest.raises(ValueError, match="spelled with the Python value"):
        sop.Symbol(name)


@pytest.mark.parametrize("name", ["a b", "", "1x", "a,b", "a.b"])
def test_symbol_requires_an_identifier(name):
    with pytest.raises(ValueError, match="not an identifier"):
        sop.Symbol(name)


@pytest.mark.parametrize("tag", ["a b", "", "1x"])
def test_tagged_requires_an_identifier_tag(tag):
    with pytest.raises(ValueError, match="not an identifier"):
        sop.Tagged(tag, 1)


def test_unicode_identifiers_are_fine():
    assert sop.Symbol("été").name == "été"
    assert sop.dumps({"été": sop.Symbol("café")}) == "{été:café}"


# ---------------------------------------------------------------------------
# Equality is Python's, and the types make it come out right
# ---------------------------------------------------------------------------


def test_a_symbol_is_not_a_string():
    assert sop.Symbol("x") != "x"
    assert sop.loads[Any]("Active") != "Active"


def test_a_tagged_value_is_not_its_payload():
    assert sop.Tagged("a", 1) != 1
    assert sop.Tagged("a", 1) != sop.Tagged("b", 1)
    assert sop.Tagged("a", 1) == sop.Tagged("a", 1)


def test_symbols_are_hashable_and_usable_as_keys():
    assert {sop.Symbol("a"): 1}[sop.Symbol("a")] == 1
    assert len({sop.Symbol("a"), sop.Symbol("a")}) == 1


@pytest.mark.parametrize(
    "value", [sop.Symbol("x"), sop.Tagged("t", {"a": [1, sop.Symbol("y")]})]
)
def test_native_values_copy_and_pickle(value):
    assert copy.copy(value) == value
    assert copy.deepcopy(value) == value
    assert pickle.loads(pickle.dumps(value)) == value


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_shape_error_is_a_sop_error():
    # One `except` covers both failure modes.
    assert issubclass(sop.ShapeError, sop.SopError)
    with pytest.raises(sop.SopError):
        sop.loads[int]('"x"')


def test_parse_errors_carry_a_position():
    with pytest.raises(sop.SopError) as caught:
        sop.loads[Any]("[\n  1,\n  2 3\n]")
    assert (caught.value.line, caught.value.column) == (3, 5)
    assert "," in caught.value.message


def test_shape_errors_carry_a_path():
    with pytest.raises(sop.ShapeError) as caught:
        sop.loads[dict[str, int]]("{a: Active}")
    assert caught.value.path == "$.a"


def test_loads_needs_a_shape():
    # `loads[Any]` is the escape hatch and it has to be written down.
    with pytest.raises(TypeError):
        sop.loads("1")  # type: ignore[operator]


def test_unencodable_object():
    # The core rejects it first, then the shape layer does, and the shape
    # layer's message is the one that reaches the caller because it has a path.
    with pytest.raises(sop.ShapeError, match="cannot encode object"):
        sop.dumps(object())


# ---------------------------------------------------------------------------
# Objects
# ---------------------------------------------------------------------------


def test_insertion_order_is_preserved():
    assert list(sop.loads[Any]("{b: 1, a: 2, c: 3}")) == ["b", "a", "c"]


def test_duplicate_keys_take_the_last_value_in_the_first_position():
    assert sop.loads[Any]("{a: 1, b: 2, a: 3}") == {"a": 3, "b": 2}
    assert list(sop.loads[Any]("{a: 1, b: 2, a: 3}")) == ["a", "b"]


def test_object_keys_must_be_strings():
    # Not coerced with str(): `{1: "a"}` and `{"1": "a"}` are different Python
    # values and quietly conflating them is the SDK inventing a mapping.
    with pytest.raises(sop.ShapeError, match="keys must be strings"):
        sop.dumps({1: "a"})


# ---------------------------------------------------------------------------
# Numbers: Python's are unbounded, the format's are not
# ---------------------------------------------------------------------------


def test_integers_are_exact_within_the_range():
    for value in (0, -1, 2**53 + 1, 2**63 - 1, -(2**63)):
        assert sop.loads[int](sop.dumps(value)) == value


def test_integers_outside_the_range_are_refused_on_write():
    # Writing one would produce a document that cannot be read back to an
    # equal value, so it is refused rather than quietly rounded.
    with pytest.raises(sop.SopError, match="out of range"):
        sop.dumps(2**63)


def test_out_of_range_literals_are_refused_on_read():
    with pytest.raises(sop.SopError, match="out of range"):
        sop.loads[Any]("1e400")


def test_non_finite_floats_have_no_spelling():
    for value in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(sop.SopError, match="no sop representation"):
            sop.dumps(value)


def test_a_large_float_stays_a_float():
    # Beyond 2^53 an integral float prints as a run of digits, which would read
    # back as an int; the writer keeps the point.
    assert sop.dumps(1e16) == "10000000000000000.0"
    assert isinstance(sop.loads[Any](sop.dumps(1e16)), float)


def test_a_float_that_is_a_whole_number_is_written_plainly():
    assert sop.dumps(2.0) == "2"
    assert sop.loads[float]("2") == 2.0


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def test_depth_is_bounded_by_the_interpreter_not_the_sdk():
    # The core is iterative and reads any depth. Building the Python value is
    # recursive, so past the interpreter's own stack limit the SDK raises
    # RecursionError rather than imposing a limit -- or crashing.
    value = sop.loads[Any]("[" * 2_000 + "1" + "]" * 2_000)
    for _ in range(2_000):
        value = value[0]
    assert value == 1
    with pytest.raises(RecursionError):
        sop.loads[Any]("[" * 1_000_000 + "1" + "]" * 1_000_000)


# ---------------------------------------------------------------------------
# Writing: the fast path and the fallback must agree
# ---------------------------------------------------------------------------


@dataclass
class Point:
    __sop_tag__ = None
    x: int


class Iban(str):
    __sop_tag__ = "iban"


def test_plain_values_take_the_fast_path():
    value = sop.loads[Any]('{a: [1, "x", Active], b: uuid "9f1c"}')
    assert sop.dumps(value) == '{a:[1,"x",Active],b:uuid "9f1c"}'


def test_a_typed_object_falls_back_to_the_shape_layer():
    assert sop.dumps(Point(1)) == "{x:1}"


def test_a_mixed_graph_falls_back_correctly():
    # The fast path gets part-way through and fails; the retry must produce the
    # whole document, not a fragment.
    assert sop.dumps({"a": 1, "b": Point(2), "c": [3]}) == "{a:1,b:{x:2},c:[3]}"


def test_a_tagged_subclass_of_a_builtin_keeps_its_tag():
    # `Iban` is a `str`, so a writer matching builtins loosely would drop the
    # tag and emit a bare string.
    assert sop.dumps(Iban("DE89")) == 'iban "DE89"'
    assert sop.dumps({"iban": Iban("DE89")}) == '{iban:iban "DE89"}'
    assert sop.dumps([Iban("DE89")]) == '[iban "DE89"]'


def test_indent():
    assert sop.dumps({"a": 1}, indent=2) == "{\n  a: 1\n}"
    assert sop.loads[Any](sop.dumps({"a": [1, 2]}, indent=2)) == {"a": [1, 2]}


def test_a_runaway_value_is_an_error_not_a_crash():
    # The writer walks a value the caller built, which can nest arbitrarily or
    # cyclically. CPython's recursion guard turns that into RecursionError at
    # the interpreter's limit; the stack must never actually overflow.
    cycle = []
    cycle.append(cycle)
    with pytest.raises(RecursionError):
        sop.dumps(cycle)


def test_a_lone_surrogate_has_no_spelling():
    # The parser rejects the escape; the writer refuses the value, as SopError.
    with pytest.raises(sop.SopError, match="lone surrogate"):
        sop.dumps("a\ud800b")
    with pytest.raises(sop.SopError, match="lone surrogate"):
        sop.dumps({"a\ud800b": 1})


# ---------------------------------------------------------------------------
# The last uncovered corners
# ---------------------------------------------------------------------------


def test_repr_of_the_loaders():
    assert repr(sop.loads) == "sop.loads"
    assert repr(sop.loads[int]) == "sop.loads[int]"
    assert repr(sop.loads[list[int]]) == "sop.loads[list[int]]"
    assert repr(sop.loads[Point]) == "sop.loads[Point]"
    assert repr(sop.loads[int | None]) == "sop.loads[int | None]"


@pytest.mark.parametrize(
    "text, description",
    [
        ("null", "the symbol `null`"),
        ("true", "the symbol `true`"),
        ("false", "the symbol `false`"),
        ("Active", "the symbol `Active`"),
        ('uuid "x"', "a value tagged `uuid`"),
        ('"x"', "a string"),
        ("1", "a number"),
        ("[]", "an array"),
    ],
)
def test_errors_name_what_was_actually_there(text, description):
    with pytest.raises(sop.ShapeError, match=re.escape(description)):
        sop.loads[Point](text)


def test_an_object_of_the_wrong_shape_reports_the_missing_key():
    with pytest.raises(sop.ShapeError, match="missing key"):
        sop.loads[Point]("{}")


def test_an_unknown_python_object_is_described_by_its_type():
    with pytest.raises(sop.ShapeError, match="cannot encode complex"):
        sop.dumps(1 + 2j)


def test_a_tag_wrapping_a_typed_object_is_encoded_through():
    # The payload of a Tagged is not assumed to be a plain value already.
    assert sop.dumps(sop.Tagged("wrapper", Point(1))) == "wrapper {x:1}"


def test_a_symbol_inside_a_typed_graph_is_encoded():
    # Symbols only reach the shape layer when something else in the graph sent
    # the whole value down the fallback path.
    assert sop.dumps({"a": Point(1), "b": sop.Symbol("x")}) == "{a:{x:1},b:x}"


def test_a_class_that_opts_out_of_a_tag_is_not_a_scalar():
    # `__sop_tag__ = None` on a non-dataclass leaves nothing to carry it as.
    class Opaque:
        __sop_tag__ = None

    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Opaque]('"x"')
