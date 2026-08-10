"""Python-specific expectations.

The corpus checks that the SDK and the core agree about the *format*.  This
file checks the things that are true only on this side of the boundary: how
the format's values become Python ones, what the native types refuse, and
where Python's own semantics need guarding against.
"""

import copy
import enum
import gc
import math
import os
import pickle
import re
import subprocess
import sys
import types
import weakref
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


def test_tagged_hashes_as_the_same_number_a_frozen_dataclass_would():
    # Not merely "agrees with `__eq__`" -- the same number, because it *is*
    # the hash of `(tag, value)`.  A hash mixed together in the extension
    # would satisfy `__eq__` just as well and still be wrong here.
    @dataclass(frozen=True)
    class Fields:
        tag: str
        value: object

    for tag, payload in (("a", 1), ("T", "x"), ("Deposit", (1, 2))):
        assert hash(sop.Tagged(tag, payload)) == hash(Fields(tag, payload))
        # And this is only ever equal if the tag went through Python's own
        # `str` hash, which is seeded per process -- see below.
        assert hash(sop.Tagged(tag, payload)) == hash((tag, payload))


def test_a_tag_does_not_hash_the_same_in_every_process():
    # Hash randomisation is what stops keys taken from a document being made
    # to collide, and tags are taken from documents.  A hash of the SDK's own
    # would answer the same number in every process, so an attacker could
    # work the colliding tags out in advance.
    reading = 'from sop import Tagged; print(hash(Tagged("a", 1)))'
    seen = {
        subprocess.run(
            [sys.executable, "-c", reading],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(seen) == 3


def test_every_value_survives_copying_and_pickling():
    # A value handed to a worker process is pickled there and back, so one
    # that cannot be rebuilt is one the caller never gets to read.  Every
    # other member of `Value` already round-trips; `Symbol` and `Tagged` are
    # the two that had to say how.
    parsed = sop.loads[sop.Value]("{a: Sym, b: T [1,2], c: {d: null}, e: 1.5}")
    for value in (
        None,
        True,
        1,
        1.5,
        "s",
        (1, 2),
        sop.Symbol("s"),
        sop.Tagged("T", 1),
        parsed,
    ):
        assert pickle.loads(pickle.dumps(value)) == value
        assert copy.copy(value) == value
        assert copy.deepcopy(value) == value
    # And a whole document still spells the same after the round trip.
    assert sop.dumps(pickle.loads(pickle.dumps(parsed))) == sop.dumps(parsed)


def test_a_deep_copy_copies_the_payload_and_a_shallow_one_does_not():
    # The payload is one of the arguments the value is rebuilt from, so the
    # two kinds of copy differ in the way they do everywhere else.
    payload = [1, 2]
    tagged = sop.Tagged("T", payload)
    assert copy.copy(tagged).value is payload
    assert copy.deepcopy(tagged).value == payload
    assert copy.deepcopy(tagged).value is not payload


def test_a_value_is_rebuilt_by_calling_its_class():
    # Rebuilding goes back through the constructor rather than restoring the
    # fields behind its back...
    assert sop.Symbol("s").__reduce__() == (sop.Symbol, ("s",))
    assert sop.Tagged("T", 1).__reduce__() == (sop.Tagged, ("T", 1))

    # ...which is what makes this hold: a pickle from anywhere else cannot
    # mint a value the constructor would have refused.
    for arguments, refused in (
        ((sop.Symbol, ("null",)), "spelled with the Python value"),
        ((sop.Tagged, ("not an identifier", 1)), "is not an identifier"),
        ((sop.Tagged, ("T", None)), "bare symbol"),
    ):

        class Forged:
            def __reduce__(self, arguments=arguments):
                return arguments

        with pytest.raises(ValueError, match=refused):
            pickle.loads(pickle.dumps(Forged()))


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
    # The path is where it failed, and it is the whole location: a document
    # that parsed has no position left to report.
    assert caught.value.path == "$.a"
    assert not hasattr(caught.value, "line")


def test_every_error_survives_a_round_trip():
    # An exception raised in a worker process is pickled back to its parent,
    # so an error that cannot be rebuilt from its own `args` is one the
    # caller never gets to read. Each class is reconstructed by calling it
    # with the arguments it was raised with, which is what makes this hold.
    raised: list[sop.SopError] = []
    for act in (
        lambda: sop.loads[Any]("[1 2]"),
        lambda: sop.loads[dict[str, int]]("{a: Active}"),
        lambda: sop.dumps(object()),
    ):
        with pytest.raises(sop.SopError) as caught:
            act()
        raised.append(caught.value)

    assert [type(e) for e in raised] == [
        sop.ParseError,
        sop.ShapeError,
        sop.ShapeError,
    ]
    for error in raised:
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is type(error)
        assert str(copy) == str(error)
        assert copy.message == error.message


def test_the_error_subclasses_are_final():
    # `SopError` is the one open class, so a caller can extend the `except`
    # that already catches everything; the two below it each carry the one
    # location they have and are closed.  The stubs say `@final` of exactly
    # these two, and a stub that claims less than the runtime enforces is a
    # check that passes on code which crashes.
    class Extended(sop.SopError): ...

    assert issubclass(Extended, sop.SopError)
    for closed in (sop.ParseError, sop.ShapeError):
        with pytest.raises(TypeError, match="not an acceptable base type"):
            type("Nope", (closed,), {})


def test_an_error_is_built_location_first_and_cannot_be_edited_after():
    # Both subclasses are the base plus a location, so both take the location
    # first, in the order `__str__` renders it.  And the native types are
    # frozen, errors included: what an error says about where it happened is
    # fixed when it is raised.
    parsed = sop.ParseError(3, 7, "boom")
    shaped = sop.ShapeError("$.a", "boom")
    assert str(parsed) == "3:7: boom"
    assert str(shaped) == "$.a: boom"
    assert (parsed.line, parsed.column, parsed.message) == (3, 7, "boom")
    assert (shaped.path, shaped.message) == ("$.a", "boom")

    for error, attribute in (
        (parsed, "line"),
        (parsed, "column"),
        (parsed, "message"),
        (shaped, "path"),
        (shaped, "message"),
    ):
        with pytest.raises(AttributeError, match="not writable"):
            setattr(error, attribute, "edited")


def test_a_write_error_names_where_in_the_value_it_failed():
    # The path is assembled as the error leaves the traversal, so it reads
    # exactly as the reader's does -- and costs nothing when nothing fails.
    @dataclass
    class Customer:
        ref: object

    @dataclass
    class Order:
        customer: Customer

    with pytest.raises(sop.ShapeError) as caught:
        sop.dumps({"orders": [Order(Customer("fine")), Order(Customer(object()))]})
    assert caught.value.path == "$.orders[1].customer.ref"
    assert str(caught.value).startswith("$.orders[1].customer.ref: cannot encode")

    # A failure at the top level is still named.
    with pytest.raises(sop.ShapeError) as bare:
        sop.dumps(object())
    assert bare.value.path == "$"


def test_a_non_sop_error_from_the_hook_passes_through_unlocated():
    # Only the writer's own errors collect a path; `Tagged` refusing a bare
    # symbol is a plain ValueError and the writer does not restate it.
    with pytest.raises(ValueError, match="bare symbol") as caught:
        sop.dumps({"a": [sop.Tagged("T", None)]})
    assert not isinstance(caught.value, sop.SopError)


def test_each_error_carries_only_the_location_it_has():
    # Reading text has a position, reading a value has a path, and writing
    # has neither; none of the three invents one of the others.
    with pytest.raises(sop.ParseError) as parsed:
        sop.loads[Any]("[1 2]")
    assert (parsed.value.line, parsed.value.column) == (1, 4)
    assert not hasattr(parsed.value, "path")

    with pytest.raises(sop.ShapeError) as shaped:
        sop.loads[int]('"x"')
    assert shaped.value.path == "$"
    assert not hasattr(shaped.value, "line")


def test_loads_needs_a_shape():
    # `loads[Any]` is the escape hatch and it has to be written down.
    with pytest.raises(TypeError):
        sop.loads("1")  # type: ignore[operator]


def test_value_names_the_untyped_result():
    # `Value` is the closed set untyped reading produces, so reading through
    # it changes nothing about the result.
    text = '{a: [1, "x", Active], b: Uuid "9f1c", c: null, d: [true, 1.5], e: Set [2]}'
    assert sop.loads[sop.Value](text) == sop.loads[Any](text)


def test_the_value_shape_does_not_re_walk_the_parsed_value():
    # `Value` is a check that cannot fail, so it answers what was parsed
    # rather than walking it to build an equal copy.  Walking it would exhaust
    # the stack on a document the parser itself handles comfortably.
    deep = "[" * 2_000 + "1" + "]" * 2_000
    assert sop.loads[sop.Value](deep) == sop.loads[Any](deep)


def test_a_type_alias_is_its_right_hand_side():
    type Port = int
    assert sop.loads[Port]("8080") == 8080


def test_unencodable_object():
    # The core rejects it first and asks the shape layer, which rejects it
    # too; the core is what was walking, so the core is what says where.
    with pytest.raises(sop.ShapeError, match=r"^\$: cannot encode object"):
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


def test_a_subclass_is_carried_as_what_it_is():
    # A subclass is still a list, a dict or a tuple and is written as one --
    # under its own name, so what the data was stays visible.  Only the
    # builtins the format spells natively go untagged.
    class Roles(list): ...

    class Ordered(dict): ...

    class Pair(tuple): ...

    assert sop.dumps(Roles([1, 2])) == "Roles [1,2]"
    assert sop.dumps(Pair((1, 2))) == "Pair [1,2]"
    assert sop.dumps(Ordered(a=1)) == "{a:1}"
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


# ---------------------------------------------------------------------------
# What the SDK works out about a class
# ---------------------------------------------------------------------------


def test_a_subclass_does_not_inherit_what_was_worked_out_for_its_base():
    # A class's fields and annotations are resolved once and kept on the class
    # itself, so they are read off that class and never off a base.  The base
    # is written first on purpose: that is what fills its own answers in, and
    # an inherited lookup would then hand them to the subclass.
    @dataclass
    class Base:
        a: int

    @dataclass
    class Sub(Base):
        b: str = "x"

    assert sop.dumps(Base(1)) == "Base {a:1}"
    assert sop.dumps(Sub(1, "y")) == 'Sub {a:1,b:"y"}'
    assert sop.loads[Sub]('Sub {a:1,b:"y"}') == Sub(1, "y")


def test_what_is_worked_out_about_a_class_dies_with_the_class():
    # Those answers are kept on the class rather than in a table here, so a
    # class built at run time is collectable once nothing else refers to it.
    # `Node` is self-referential on purpose: its own annotation names it, so
    # a table here would hold it up through the *value* it stored, which is
    # something weak keys cannot prevent.
    module = types.ModuleType("sop_throwaway")
    sys.modules[module.__name__] = module
    exec(
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Node:\n"
        "    name: str\n"
        '    next: "Node | None" = None\n',
        module.__dict__,
    )
    node = module.Node

    assert (
        sop.dumps(node("a", node("b")))
        == 'Node {name:"a",next:Node {name:"b",next:null}}'
    )
    assert sop.loads[node]('Node {name:"a"}') == node("a")

    seen = weakref.ref(node)
    del node, module
    del sys.modules["sop_throwaway"]
    gc.collect()
    assert seen() is None
