"""Python-specific expectations.

The corpus checks that the SDK and the core agree about the *format*.  This
file checks the things that are true only on this side of the boundary: how
the format's values become Python ones, what the native types refuse, and
where Python's own semantics need guarding against.
"""

import enum
import math
import re
from dataclasses import dataclass
from typing import Any

import pytest
import sop

# ---------------------------------------------------------------------------
# Schemas used below
# ---------------------------------------------------------------------------


@dataclass
class Point:
    __sop_tag__ = None
    x: int


class Iban(str):
    __sop_tag__ = "Iban"


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


def test_untyped_reading_produces_immutable_values():
    # The whole untyped result is immutable; mutation is something a shape
    # such as `list[T]` or `dict[str, V]` has to declare.
    value = sop.loads[Any]("{a: [1, 2], b: {c: 3}}")
    assert isinstance(value, frozendict)
    assert isinstance(value["a"], tuple)
    assert isinstance(value["b"], frozendict)
    with pytest.raises(TypeError):
        value["d"] = 4  # type: ignore[index]


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


@pytest.mark.parametrize("payload", [None, True, False])
def test_tagged_refuses_what_spells_as_a_symbol(payload):
    # A tag cannot be applied to a bare symbol, so such a value could never
    # be written or read back; it is refused at construction.
    with pytest.raises(ValueError, match="bare symbol"):
        sop.Tagged("t", payload)
    with pytest.raises(ValueError, match="bare symbol"):
        sop.Tagged("t", sop.Symbol("x"))


def test_a_missing_comma_is_an_error_not_a_tag():
    # `[Red Green]` used to denote one doubly-named value; a bare symbol
    # cannot be a tag's payload, so the typo is caught where it happens.
    with pytest.raises(sop.SopError, match="bare symbol"):
        sop.loads[Any]("[Red Green]")


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


def test_tagged_equality_defers_to_the_other_operand():
    # For anything that is not a Tagged, __eq__ answers NotImplemented rather
    # than False, so Python asks the other side before settling the question.
    class Agreeable:
        def __eq__(self, other: object) -> bool:
            return True

    assert sop.Tagged("a", 1) == Agreeable()
    assert Agreeable() == sop.Tagged("a", 1)


def test_symbols_are_hashable_and_usable_as_keys():
    assert {sop.Symbol("a"): 1}[sop.Symbol("a")] == 1
    assert len({sop.Symbol("a"), sop.Symbol("a")}) == 1


def test_tagged_hashes_like_a_frozen_dataclass():
    # Equal values must hash equal, or sets and dicts misbehave.
    assert hash(sop.Tagged("a", 1)) == hash(sop.Tagged("a", 1))
    assert len({sop.Tagged("a", 1), sop.Tagged("a", 1)}) == 1
    assert {sop.Tagged("a", 1): "x"}[sop.Tagged("a", 1)] == "x"
    # An unhashable payload makes the whole value unhashable, as a frozen
    # dataclass's field would.
    with pytest.raises(TypeError):
        hash(sop.Tagged("a", [1]))


def test_native_values_are_immutable():
    with pytest.raises(AttributeError):
        sop.Symbol("x").name = "y"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        sop.Tagged("a", 1).value = 2  # type: ignore[misc]


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
    assert str(caught.value).startswith("$.a: ")
    # No position -- the document parsed -- but the attributes every SopError
    # promises are present.
    assert (caught.value.line, caught.value.column) == (0, 0)


def test_loads_needs_a_shape():
    # `loads[Any]` is the escape hatch and it has to be written down.
    with pytest.raises(TypeError):
        sop.loads("1")  # type: ignore[operator]


def test_value_names_the_untyped_result():
    # `Value` is the closed set untyped reading produces, so reading through
    # it changes nothing about the result.
    text = '{a: [1, "x", Active], b: Uuid "9f1c", c: null, d: [true, 1.5], e: Bag [2]}'
    assert sop.loads[sop.Value](text) == sop.loads[Any](text)


def test_a_type_alias_is_its_right_hand_side():
    type Port = int
    assert sop.loads[Port]("8080") == 8080


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
    with pytest.raises(sop.SopError, match="keys must be strings"):
        sop.dumps({1: "a"})


def test_a_tagged_string_cannot_be_an_object_key():
    # A key has nowhere to carry a tag, and writing the bare string would
    # drop it silently.
    with pytest.raises(sop.SopError, match="an object key cannot hold"):
        sop.dumps({Iban("DE89"): 1})


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


def test_a_float_keeps_its_kind():
    # Number kind is spelling-determined: digits alone denote an integer, so
    # a float always keeps a point or an exponent.
    assert sop.dumps(2.0) == "2.0"
    assert isinstance(sop.loads[Any]("2.0"), float)
    assert sop.dumps(1e16) == "10000000000000000.0"
    assert isinstance(sop.loads[Any](sop.dumps(1e16)), float)
    assert sop.loads[float]("2") == 2.0  # an integer literal is still a number


def test_negative_zero_keeps_its_sign():
    assert sop.dumps(-0.0) == "-0.0"
    assert math.copysign(1, sop.loads[Any]("-0.0")) == -1.0


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
# Writing: one traversal; the shape layer converts unknown objects in place
# ---------------------------------------------------------------------------


def test_plain_values_take_the_fast_path():
    value = sop.loads[Any]('{a: [1, "x", Active], b: Uuid "9f1c"}')
    assert sop.dumps(value) == '{a:[1,"x",Active],b:Uuid "9f1c"}'


def test_a_typed_object_is_converted():
    assert sop.dumps(Point(1)) == "{x:1}"


def test_a_mixed_graph_converts_in_place():
    # The traversal happens once; the convert hook spells each unknown object
    # where it is met, and the plain values around it never touch Python.
    assert sop.dumps({"a": 1, "b": Point(2), "c": [3]}) == "{a:1,b:{x:2},c:[3]}"


def test_a_tag_cannot_wrap_a_value_that_spells_as_a_symbol():
    # Tagged(t, enum_member) is constructible -- the payload is an object --
    # but the member spells as a symbol, which a tag cannot apply to.
    class Status(enum.Enum):
        Up = "up"

    with pytest.raises(sop.SopError, match="bare symbol"):
        sop.dumps(sop.Tagged("t", Status.Up))


def test_a_tagged_subclass_of_a_builtin_keeps_its_tag():
    # `Iban` is a `str`, so a writer matching builtins loosely would drop the
    # tag and emit a bare string.
    assert sop.dumps(Iban("DE89")) == 'Iban "DE89"'
    assert sop.dumps({"iban": Iban("DE89")}) == '{iban:Iban "DE89"}'
    assert sop.dumps([Iban("DE89")]) == '[Iban "DE89"]'


def test_an_untagged_subclass_is_carried_as_what_it_is():
    # A subclass that declared no tag is still a list or a dict, and is
    # written as one -- which is how a subclass of `set` already behaved.
    class Roles(list): ...

    class Ordered(dict): ...

    class Pair(tuple): ...

    assert sop.dumps(Roles([1, 2])) == "[1,2]"
    assert sop.dumps(Ordered(a=1)) == "{a:1}"
    assert sop.dumps(Pair((1, 2))) == "[1,2]"
    # A declared tag still wins: it is a tagged string, spelled with `str`.
    assert sop.dumps(Iban("DE89")) == 'Iban "DE89"'


def test_a_runaway_value_is_an_error_not_a_crash():
    # The writer walks a value the caller built, which can nest arbitrarily or
    # cyclically. CPython's recursion guard turns that into RecursionError at
    # the interpreter's limit; the stack must never actually overflow.
    cycle = []
    cycle.append(cycle)
    with pytest.raises(RecursionError):
        sop.dumps(cycle)
    # A dict is frozen on the way out, so it takes a different route through
    # the writer than a list and has to be guarded just as well.
    loop: dict[str, object] = {}
    loop["self"] = loop
    with pytest.raises(RecursionError):
        sop.dumps(loop)


def test_a_lone_surrogate_has_no_spelling():
    # The parser rejects the escape; the writer refuses the value, as SopError.
    with pytest.raises(sop.SopError, match="lone surrogate"):
        sop.dumps("a\ud800b")
    with pytest.raises(sop.SopError, match="lone surrogate"):
        sop.dumps({"a\ud800b": 1})


# ---------------------------------------------------------------------------
# The last uncovered corners
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, description",
    [
        ("null", "the symbol `null`"),
        ("true", "the symbol `true`"),
        ("false", "the symbol `false`"),
        ("Active", "the symbol `Active`"),
        ('Uuid "x"', "a value tagged `Uuid`"),
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
