"""The shape language: every shape, in both directions, and its errors.

Run with `pytest` from the repository root, against an installed build
(`maturin develop`) — the in-tree package has no extension module.
"""

import enum
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from typing import Annotated, Any, TypeForm, cast
from uuid import UUID

import pytest
import sop

# ---------------------------------------------------------------------------
# Schemas used below
# ---------------------------------------------------------------------------


class Colour(enum.Enum):
    Red = "Red"
    Blue = "azul"  # value and name differ, so both lookups get exercised


@dataclass
class Plain:
    x: int  # tagged with its own name


@dataclass
class Geo:  # the convention is the default: carried as `Geo`
    lat: float
    lng: float


@dataclass
class Deposit:
    amount: Decimal


@dataclass
class Withdraw:
    atm: str


@dataclass
class Branch:
    name: str
    children: list[Branch] = field(default_factory=list)


class Priority(enum.Enum):
    Low = 1  # a non-string value, so the member's name is its spelling
    High = 2


def roundtrip[T](shape: TypeForm[T], value: object) -> T:
    """Write a value and read it back through the same shape."""
    return sop.loads[shape](sop.dumps(value))


# ---------------------------------------------------------------------------
# Scalars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, text, expected",
    [
        (str, '"x"', "x"),
        (int, "1", 1),
        (int, "-0", 0),
        (float, "1.5", 1.5),
        (float, "2", 2.0),  # an integer literal is still a number
        (bool, "true", True),
        (bool, "false", False),
        (type(None), "null", None),
        (Any, "Active", sop.Symbol("Active")),
    ],
)
def test_scalar_shapes(shape: TypeForm[Any], text: str, expected: object) -> None:
    assert sop.loads[shape](text) == expected


@pytest.mark.parametrize(
    "shape, text, message",
    [
        # A symbol is never a string.
        (str, "Active", "expected a string"),
        (str, "1", "expected a string"),
        (int, "true", "expected an integer"),
        (int, "1.5", "expected an integer"),
        (int, '"1"', "expected an integer"),
        (bool, "1", "expected `true` or `false`"),
        (bool, "Active", "expected `true` or `false`"),
        (float, "true", "expected a number"),
        (type(None), "1", "expected `null`"),
        (sop.Symbol, '"x"', "expected a symbol"),
        (sop.Tagged, "1", "expected a tagged value"),
    ],
)
def test_scalar_mismatch(shape: TypeForm[Any], text: str, message: str) -> None:
    with pytest.raises(sop.ShapeError, match=message):
        sop.loads[shape](text)


def test_unsupported_shape() -> None:
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[object]("1")


# ---------------------------------------------------------------------------
# set: an array that says what it was
# ---------------------------------------------------------------------------


def test_set_reads_a_tagged_array() -> None:
    assert sop.loads[set[str]]('Set ["a", "b"]') == {"a", "b"}


def test_set_rejects_a_bare_array() -> None:
    # Every sequence spells as an array, so the tag is what tells them apart.
    with pytest.raises(sop.ShapeError, match="tagged `Set`"):
        sop.loads[set[str]]('["a"]')


def test_set_rejects_another_tag() -> None:
    with pytest.raises(sop.ShapeError, match="tagged `Set`"):
        sop.loads[set[str]]('Bag ["a"]')


def test_list_rejects_a_tagged_array() -> None:
    with pytest.raises(sop.ShapeError, match="expected an array"):
        sop.loads[list[str]]('Set ["a"]')


def test_set_of_the_wrong_element_type() -> None:
    with pytest.raises(sop.ShapeError, match="expected an integer"):
        sop.loads[set[int]]('Set ["a"]')


def test_empty_set() -> None:
    assert sop.loads[set[int]]("Set []") == set()
    assert sop.dumps(set()) == "Set []"


def test_set_writes_a_tagged_array() -> None:
    assert sop.dumps({"a"}) == 'Set ["a"]'


def test_set_output_is_deterministic() -> None:
    # sop arrays are ordered and Python sets are not, so the writer has to pick
    # an order and stick to it, or output stops being reproducible.
    values = {3, 1, 2, 10}
    assert sop.dumps(values) == sop.dumps(set(reversed(list(values))))


def test_set_output_does_not_depend_on_how_the_elements_repr() -> None:
    # The order has to come from what the values *spell as*.  `repr` is not
    # that: a class which does not define one gets the default, which spells
    # the object's address, so the same set writes in a different order in
    # the next process.  This `__repr__` stands in for that -- it disagrees
    # with the spelling in a way an address does by accident.
    @dataclass(frozen=True, repr=False)
    class Thing:
        n: int

        def __repr__(self) -> str:
            # Sorts opposite to the spelling, as an address does whenever the
            # allocator happens to hand these out in a different order.
            return "cba"[self.n - 1]

    assert sop.dumps({Thing(2), Thing(1), Thing(3)}) == (
        "Set [Thing {n:1},Thing {n:2},Thing {n:3}]"
    )


def test_a_set_element_with_no_spelling_is_still_named_where_it_sits() -> None:
    # Ordering the elements asks each of them how it spells, and this one has
    # no answer.  The failure is left to the traversal, which is the thing
    # that knows where in the document the set sits.
    with pytest.raises(sop.ShapeError) as caught:
        sop.dumps({"roles": {object()}})
    assert caught.value.path == "$.roles[0]"
    assert "cannot encode object" in caught.value.message


def test_frozenset_writes_the_same_way() -> None:
    # The wire says unordered; whether the Python value can be mutated is the
    # shape's business, exactly as with `list[T]` and `tuple[T, ...]`.
    assert sop.dumps(frozenset({"a"})) == 'Set ["a"]'


@pytest.mark.parametrize("value", [set(), {1}, {1, 2, 3}, {"a", "b"}])
def test_set_roundtrip(value: set[Any]) -> None:
    assert roundtrip(set[Any], value) == value


def test_nested_sets() -> None:
    value = {"tags": {"a", "b"}, "none": set()}
    assert roundtrip(dict[str, set[str]], value) == value


def test_set_inside_a_dataclass() -> None:
    @dataclass
    class Holder:
        roles: set[str]

    assert sop.loads[Holder]('Holder {roles: Set ["admin"]}') == Holder({"admin"})
    assert sop.dumps(Holder({"admin"})) == 'Holder {roles:Set ["admin"]}'


def test_a_sequence_carries_the_tag_of_what_it_was() -> None:
    # Every sequence spells as an array; the tag is how you can still see what
    # the data was.  `list` and `tuple` are the format's own array and carry
    # none, which is why a plain array stays plain.
    class Roles(list[int]): ...

    class Tags(set[int]): ...

    assert sop.dumps([1, 2]) == "[1,2]"
    assert sop.dumps((1, 2)) == "[1,2]"
    assert sop.dumps(Roles([1, 2])) == "Roles [1,2]"
    assert sop.dumps(Tags({1})) == "Tags [1]"


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def test_list_of_lists() -> None:
    assert sop.loads[list[list[int]]]("[[1],[2,3],[]]") == [[1], [2, 3], []]


def test_dict_of_lists() -> None:
    assert sop.loads[dict[str, list[int]]]("{a:[1],b:[]}") == {"a": [1], "b": []}


def test_dict_rejects_an_array() -> None:
    with pytest.raises(sop.ShapeError, match="expected an object"):
        sop.loads[dict[str, int]]("[1]")


def test_list_rejects_an_object() -> None:
    with pytest.raises(sop.ShapeError, match="expected an array"):
        sop.loads[list[int]]("{a:1}")


def test_tuple_reads_an_array_immutably() -> None:
    # The wire value is the one `list[T]` reads; the shape declares that the
    # result does not mutate.
    assert sop.loads[tuple[int, ...]]("[1,2]") == (1, 2)
    assert sop.loads[tuple[int, ...]]("[]") == ()
    assert roundtrip(tuple[int, ...], (1, 2, 3)) == (1, 2, 3)


def test_only_the_homogeneous_tuple_is_a_shape() -> None:
    # `tuple[int, str]` is a row type, which a sop array does not model.
    with pytest.raises(sop.ShapeError, match=r"only `tuple\[T, \.\.\.\]`"):
        sop.loads[tuple[int, str]]("[1]")


def test_tuples_are_written_as_arrays() -> None:
    assert sop.dumps((1, 2)) == "[1,2]"


def test_frozenset_reads_a_tagged_array_immutably() -> None:
    value = sop.loads[frozenset[str]]('Set ["a", "b"]')
    assert value == frozenset({"a", "b"}) and isinstance(value, frozenset)
    assert roundtrip(frozenset[str], frozenset({"a", "b"})) == {"a", "b"}


def test_dict_keys_must_be_str_shaped() -> None:
    # An object's keys are strings; a shape claiming otherwise would decode to
    # something its own annotation contradicts.
    with pytest.raises(sop.ShapeError, match="object keys are strings"):
        sop.loads[dict[int, int]]("{a: 1}")
    assert sop.loads[dict[Any, int]]("{a: 1}") == {"a": 1}


def test_frozendict_shape_reads_an_object_immutably() -> None:
    value = sop.loads[frozendict[str, int]]("{a: 1}")
    assert isinstance(value, frozendict) and value == {"a": 1}
    assert sop.loads[dict[str, int]]("{a: 1}") == {"a": 1}  # dict opts into mutation


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_enum_by_value() -> None:
    assert sop.loads[Colour]("azul") is Colour.Blue


def test_enum_reads_only_its_spelling() -> None:
    # `Blue` is written `azul`; the member's Python name is not a second
    # accepted spelling.
    with pytest.raises(
        sop.ShapeError, match="expected `Red` or `azul`, found the symbol `Blue`"
    ):
        sop.loads[Colour]("Blue")


def test_enum_writes_its_value() -> None:
    assert sop.dumps(Colour.Blue) == "azul"


def test_enum_roundtrip() -> None:
    assert roundtrip(Colour, Colour.Red) is Colour.Red


def test_enum_rejects_a_string() -> None:
    with pytest.raises(
        sop.ShapeError, match="expected `Red` or `azul`, found a string"
    ):
        sop.loads[Colour]('"azul"')


def test_enum_rejects_an_unknown_symbol() -> None:
    with pytest.raises(
        sop.ShapeError, match="expected `Red` or `azul`, found the symbol `Green`"
    ):
        sop.loads[Colour]("Green")


def test_an_enum_value_with_no_symbol_spelling_is_refused() -> None:
    # `Symbol` decides what a symbol may be, and says so itself: the shape
    # layer does not restate the rule or re-wrap the error.
    class Weird(enum.Enum):
        Spaced = "not an identifier"
        Reserved = "true"

    with pytest.raises(ValueError, match="not an identifier"):
        sop.dumps(Weird.Spaced)
    with pytest.raises(ValueError, match="spelled with the Python value"):
        sop.dumps(Weird.Reserved)


# ---------------------------------------------------------------------------
# Tags on classes
# ---------------------------------------------------------------------------


def test_a_class_is_tagged_with_its_own_name() -> None:
    assert sop.loads[Plain]("Plain { x: 1 }") == Plain(1)
    assert sop.dumps(Plain(1)) == "Plain {x:1}"


def test_a_dataclass_is_always_tagged() -> None:
    # There is no way to ask for a bare object or for a different name: the
    # class's own name is the tag, and that is the whole rule.
    assert sop.dumps(Geo(1.0, 2.0)) == "Geo {lat:1.0,lng:2.0}"


def test_a_tagged_class_rejects_a_bare_object() -> None:
    with pytest.raises(sop.ShapeError, match="tagged `Plain`"):
        sop.loads[Plain]("{ x: 1 }")


def test_a_tagged_class_rejects_the_wrong_tag() -> None:
    with pytest.raises(
        sop.ShapeError,
        match="expected a value tagged `Plain`, found a value tagged `Other`",
    ):
        sop.loads[Plain]("Other { x: 1 }")


def test_a_tag_over_something_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(sop.ShapeError, match="expected an object"):
        sop.loads[Plain]("Plain 1")


def test_builtins_do_not_get_a_default_tag() -> None:
    # `str` and `list` are how the format's own types are spelled. Defaulting
    # them to a tag would make `dumps("x")` produce `str "x"`.
    assert sop.dumps("x") == '"x"'
    assert sop.dumps([1]) == "[1]"
    assert sop.dumps({"a": 1.5}) == "{a:1.5}"
    assert sop.loads[dict[str, float]]("{a: 1.5}") == {"a": 1.5}


# ---------------------------------------------------------------------------
# The privileged classes, carried as tagged strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, text",
    [
        (Decimal("19.99"), 'Decimal "19.99"'),
        (UUID(int=7), 'UUID "00000000-0000-0000-0000-000000000007"'),
        (datetime(2026, 8, 5, 14, 23, 11), 'datetime "2026-08-05T14:23:11"'),
        (date(2026, 8, 5), 'date "2026-08-05"'),
        (time(14, 23, 11), 'time "14:23:11"'),
    ],
)
def test_a_privileged_class_is_carried_under_its_own_tag(
    value: object, text: str
) -> None:
    assert sop.dumps(value) == text
    assert roundtrip(type(value), value) == value


def test_a_datetime_is_not_carried_as_the_date_it_subclasses() -> None:
    # `datetime` is a subclass of `date`, so the two are told apart by which
    # comes first along the MRO and not by an `isinstance` test.
    assert sop.dumps(datetime(2026, 8, 5)) == 'datetime "2026-08-05T00:00:00"'


def test_a_privileged_class_is_written_canonically() -> None:
    # `str(datetime(...))` spells a space where ISO 8601 has a `T`, so the
    # table spells with `isoformat` rather than leaning on `str`.
    written = sop.dumps(datetime(2026, 8, 5, 14, 23, 11))
    assert written == 'datetime "2026-08-05T14:23:11"'


def test_a_subclass_of_a_privileged_class_is_not_carried() -> None:
    # Matched exactly.  Subtypes are a later thing; for now a subclass is
    # refused like any other class the SDK does not know.
    class Money(Decimal):
        pass

    with pytest.raises(sop.ShapeError, match="cannot encode Money"):
        sop.dumps(Money("19.99"))
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Money]('Money "19.99"')


def test_a_carrier_rejects_the_wrong_tag() -> None:
    with pytest.raises(
        sop.ShapeError,
        match="expected a value tagged `Decimal`, found a value tagged `Swift`",
    ):
        sop.loads[Decimal]('Swift "DE89"')


def test_a_carrier_rejects_an_untagged_value() -> None:
    with pytest.raises(sop.ShapeError, match="expected a value tagged `Decimal`"):
        sop.loads[Decimal]('"19.99"')


def test_a_tag_mismatch_reads_the_same_for_either_carried_kind() -> None:
    # Being untagged and being tagged wrong are one distinction, drawn in one
    # place -- so a dataclass and a privileged class do not describe the same
    # mistake two different ways.
    for shape in (Plain, Decimal):
        with pytest.raises(sop.ShapeError, match="expected a value tagged `"):
            sop.loads[shape]('Other "x"')
        with pytest.raises(sop.ShapeError, match="expected a value tagged `"):
            sop.loads[shape]("1")


def test_a_carrier_rejects_a_non_string_payload() -> None:
    with pytest.raises(sop.ShapeError, match="carries a string"):
        sop.loads[Decimal]("Decimal 1")


def test_a_carrier_reports_a_bad_value() -> None:
    with pytest.raises(sop.ShapeError, match="is not valid"):
        sop.loads[Decimal]('Decimal "not a number"')


def test_a_carrier_reports_a_bad_value_from_isoformat() -> None:
    with pytest.raises(sop.ShapeError, match="is not valid"):
        sop.loads[datetime]('datetime "not a date"')


def test_a_tagged_shape_reads_its_payload() -> None:
    # `Tagged` is generic in what it carries, so the payload has a shape too.
    value = sop.loads[sop.Tagged[int]]("Retries 3")
    assert (value.tag, value.value) == ("Retries", 3)
    assert roundtrip(sop.Tagged[int], value) == value


def test_a_tagged_shape_checks_its_payload() -> None:
    with pytest.raises(sop.ShapeError, match="expected an integer"):
        sop.loads[sop.Tagged[int]]('Retries "3"')


def test_a_tagged_shape_rejects_an_untagged_value() -> None:
    with pytest.raises(sop.ShapeError, match="expected a tagged value"):
        sop.loads[sop.Tagged[int]]("3")


def test_a_bare_tagged_shape_takes_any_payload() -> None:
    # Undecorated, `Tagged` means `Tagged[Value]` -- whatever was there.
    assert sop.loads[sop.Tagged]('Duration "PT15M"').value == "PT15M"


def test_unknown_tags_survive_as_Tagged() -> None:
    # Nothing in the schema models `Duration`, and it still round-trips.
    value = sop.loads[sop.Tagged]('Duration "PT15M"')
    assert (value.tag, value.value) == ("Duration", "PT15M")
    assert sop.dumps(value) == 'Duration "PT15M"'


def test_a_class_the_sdk_does_not_know_is_not_carried() -> None:
    # A dataclass declares its fields, so its name can carry it as an object,
    # and the privileged classes are carried because the table says how.  An
    # arbitrary class is neither, and there is no way for it to opt in -- so
    # it is refused in both directions rather than being spelled
    # `Anon "<object at 0x…>"` through `str`.
    class Anon:
        pass

    with pytest.raises(sop.ShapeError, match="cannot encode Anon"):
        sop.dumps(Anon())
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Anon]('"x"')


def test_a_str_subclass_is_not_carried_either() -> None:
    # There is no user opt-in, so a `str` subclass is not a tagged string; it
    # is refused, rather than quietly losing the distinction by spelling as
    # the `str` it subclasses.
    class Iban(str):
        pass

    with pytest.raises(sop.ShapeError, match="cannot encode Iban"):
        sop.dumps(Iban("DE89"))


# ---------------------------------------------------------------------------
# Unions
# ---------------------------------------------------------------------------


def test_optional() -> None:
    assert sop.loads[int | None]("null") is None
    assert sop.loads[int | None]("1") == 1


def test_tagged_union_discriminates_on_the_tag() -> None:
    events = sop.loads[list[Deposit | Withdraw]](
        '[Deposit { amount: Decimal "5.00" }, Withdraw { atm: "A1" }]'
    )
    assert isinstance(events[0], Deposit) and isinstance(events[1], Withdraw)


def test_tagged_union_names_the_alternatives() -> None:
    # Spelled out in full -- and only here.  The list of alternatives is
    # built in the `raise`, so a value whose tag matches never pays to say
    # what it could have been instead.
    with pytest.raises(sop.ShapeError) as caught:
        sop.loads[Deposit | Withdraw]("Other { atm: 1 }")
    assert str(caught.value) == (
        "$: expected a value tagged `Deposit` or `Withdraw`, "
        "found a value tagged `Other`"
    )


def test_tagged_union_rejects_an_untagged_value() -> None:
    with pytest.raises(sop.ShapeError) as caught:
        sop.loads[Deposit | Withdraw]("{ atm: 1 }")
    assert str(caught.value) == (
        "$: expected a value tagged `Deposit` or `Withdraw`, found an object"
    )


def test_untagged_union_tries_each_member() -> None:
    assert sop.loads[int | str]("1") == 1
    assert sop.loads[int | str]('"a"') == "a"


def test_untagged_union_reports_every_reason() -> None:
    with pytest.raises(sop.ShapeError, match="expected a string or an integer"):
        sop.loads[int | str]("Active")


def test_mixed_union() -> None:
    assert sop.loads[Deposit | int]("7") == 7
    assert isinstance(
        sop.loads[Deposit | int]('Deposit { amount: Decimal "1" }'), Deposit
    )


def test_a_union_member_that_is_not_a_class_names_no_tag() -> None:
    # `list[int]` is a shape but not a class, so it can name no tag and cannot
    # be what the union keys on; it is tried in turn like any other member.
    assert sop.loads[list[int] | int]("[1, 2]") == [1, 2]
    assert sop.loads[list[int] | int]("7") == 7


def test_a_shared_tag_in_a_union_is_an_error() -> None:
    # A shared discriminant is a schema bug; silently degrading to trying
    # members in order would hide it until the members diverged.  Two
    # dataclasses share a tag exactly when they share a name, which is what
    # these two scopes arrange.
    def one() -> type:
        @dataclass
        class Dup:
            x: int

        return Dup

    def two() -> type:
        @dataclass
        class Dup:
            y: int

        return Dup

    shape = cast(TypeForm[Any], one() | two())
    with pytest.raises(
        sop.ShapeError, match="cannot be told apart: both read a value tagged `Dup`"
    ):
        sop.loads[shape]("Dup {x: 1}")


def test_a_union_keys_a_privileged_class_on_its_own_name() -> None:
    assert sop.loads[Decimal | Plain]('Decimal "19.99"') == Decimal("19.99")
    assert sop.loads[Decimal | Plain]("Plain { x: 1 }") == Plain(1)
    with pytest.raises(
        sop.ShapeError, match="expected a value tagged `Decimal` or `Plain`"
    ):
        sop.loads[Decimal | Plain]("Nope { x: 1 }")


def test_a_non_string_enum_matches_by_name() -> None:
    assert sop.loads[Priority]("High") is Priority.High
    assert sop.dumps(Priority.High) == "High"


def test_null_in_a_wider_union() -> None:
    assert sop.loads[int | str | None]("null") is None
    assert sop.loads[int | str | None]('"x"') == "x"


def test_a_recursive_shape_resolves() -> None:
    tree = sop.loads[Branch](
        'Branch { name: "root", children: [Branch { name: "leaf" }] }'
    )
    assert tree.children[0].name == "leaf"
    assert sop.loads[Branch](sop.dumps(tree)) == tree


def test_a_bare_none_is_the_same_shape_as_nonetype() -> None:
    # An annotation may evaluate to either; they name one shape.
    assert sop.loads[None]("null") is None
    assert sop.loads[type(None)]("null") is None


def test_a_parameterised_shape_with_no_reading_is_refused() -> None:
    # `Sequence[int]` is a real type but no row of the shape table reads it.
    from collections.abc import Sequence

    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Sequence[int]]("[1]")


def test_an_exotic_alias_is_not_a_shape() -> None:
    # Legal to write, but no row of the shape table matches it.
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Annotated[int, []]]("1")


def test_kind_matched_members_do_not_discriminate() -> None:
    # Enums are carried as symbols, and `Tagged` and `Any` match on kind, so
    # none of them is a wire tag the union can key on.
    assert sop.loads[Colour | Plain]("azul") is Colour.Blue
    unmodelled = sop.loads[Plain | sop.Tagged]('Duration "PT15M"')
    assert isinstance(unmodelled, sop.Tagged) and unmodelled.tag == "Duration"
    assert sop.loads[Plain | Any]("7") == 7


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------


@dataclass
class Fields:
    from_: str = field(metadata={"sop": "from"})  # metadata names the wire key
    cls_: str = field(default="c", metadata={"sop": "class"})
    n: int = 3  # default
    computed: int = field(init=False, default=0)  # not read back


def test_metadata_names_the_wire_key() -> None:
    assert sop.loads[Fields]('Fields {from: "x"}').from_ == "x"
    assert sop.loads[Fields]('Fields {from: "x", class: "y"}').cls_ == "y"


def test_a_field_name_is_otherwise_verbatim() -> None:
    # One aliasing mechanism: without metadata, the field's own name is the
    # wire key, underscore and all.
    @dataclass
    class Verbatim:
        key_: str

    assert sop.loads[Verbatim]('Verbatim {key_: "x"}').key_ == "x"
    assert sop.dumps(Verbatim("x")) == 'Verbatim {key_:"x"}'


def test_defaults_are_used_when_a_key_is_absent() -> None:
    assert sop.loads[Fields]('Fields {from: "x"}').n == 3


def test_missing_required_key() -> None:
    with pytest.raises(sop.ShapeError, match="missing key `from`"):
        sop.loads[Fields]("Fields {}")


def test_unknown_keys_are_ignored() -> None:
    assert sop.loads[Fields]('Fields {from: "x", surplus: 1}').from_ == "x"


def test_init_false_fields_are_not_read() -> None:
    assert sop.loads[Fields]('Fields {from: "x", computed: 9}').computed == 0


def test_a_validating_dataclass_reports_a_shape_error() -> None:
    # A `__post_init__` that raises ValueError is the document missing the
    # shape, and it surfaces with a path like every other mismatch.
    @dataclass
    class Positive:
        n: int

        def __post_init__(self) -> None:
            if self.n < 0:
                raise ValueError("n must be positive")

    with pytest.raises(sop.ShapeError, match="not a valid Positive") as caught:
        sop.loads[list[Positive]]("[Positive {n: -1}]")
    assert str(caught.value).startswith("$[0]: ")


def test_field_names_are_written_with_their_alias() -> None:
    assert sop.dumps(Fields("x")) == 'Fields {from:"x",class:"c",n:3,computed:0}'


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, text, path",
    [
        (dict[str, list[int]], '{a: [1, "x"]}', "$.a[1]"),
        (list[Geo], "[Geo {lat: 1, lng: Active}]", "$[0].lng"),
        (dict[str, dict[str, int]], "{a: {b: Active}}", "$.a.b"),
        (list[int], '["x"]', "$[0]"),
        (Fields, "{}", "$"),
    ],
)
def test_error_paths(shape: TypeForm[Any], text: str, path: str) -> None:
    with pytest.raises(sop.ShapeError) as caught:
        sop.loads[shape](text)
    assert str(caught.value).startswith(f"{path}: ")


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------


class Status(enum.Enum):
    Active = "Active"
    Suspended = "Suspended"


@dataclass
class Account:
    id: UUID
    created_at: datetime
    balance: Decimal
    location: Geo
    roles: set[str]
    status: Status
    deleted_at: datetime | None
    session_ttl: sop.Tagged | None = None


TYPED_RESPONSE = """Account {
  id: UUID "9f1c2e7a-3b44-4f80-9c1d-2a5e7b0f1234",
  created_at: datetime "2026-08-05T14:23:11Z",
  balance: Decimal "19.99",
  session_ttl: Duration "PT15M",
  location: Geo { lat: 47.6062, lng: -122.3321 },
  roles: Set ["admin", "beta"],
  status: Active,
  deleted_at: null,
}"""


def test_a_typed_api_response() -> None:
    account = sop.loads[Account](TYPED_RESPONSE)
    assert account.id == UUID("9f1c2e7a-3b44-4f80-9c1d-2a5e7b0f1234")
    assert account.balance == Decimal("19.99")
    assert account.location.lat == 47.6062
    assert account.roles == {"admin", "beta"}
    assert account.status is Status.Active
    assert account.deleted_at is None
    assert account.session_ttl is not None
    assert account.session_ttl.tag == "Duration"  # an unmodelled tag survives
    assert roundtrip(Account, account) == account


@dataclass
class Reversal:
    reason: Status


DISCRIMINATED_UNION = """[
  Deposit  { amount: Decimal "500.00" },
  Withdraw { atm: "ATM-4417" },
  Reversal { reason: Suspended },
]"""


def test_a_discriminated_union() -> None:
    events = sop.loads[list[Deposit | Withdraw | Reversal]](DISCRIMINATED_UNION)
    assert [type(e).__name__ for e in events] == ["Deposit", "Withdraw", "Reversal"]
    assert roundtrip(list[Deposit | Withdraw | Reversal], events) == events


@dataclass
class Tcp:
    host: str
    port: int


@dataclass
class Unix:
    path: str


@dataclass
class Service:
    name: str
    listen: list[Tcp | Unix]
    replicas: int
    canary: bool
    labels: dict[str, str] = field(default_factory=dict)


CONFIGURATION = """Service {
  name: "billing-api",
  listen: [
    Tcp { host: "0.0.0.0", port: 8080 },
    Unix { path: "/var/run/billing.sock" },
  ],
  replicas: 3,
  canary: false,
  labels: { tier: "gold" },
}"""


def test_a_configuration_document() -> None:
    service = sop.loads[Service](CONFIGURATION)
    assert service.name == "billing-api"
    assert isinstance(service.listen[0], Tcp) and isinstance(service.listen[1], Unix)
    assert service.canary is False
    assert service.labels == {"tier": "gold"}
    assert roundtrip(Service, service) == service


# ---------------------------------------------------------------------------
# A union is read by what the document says it is
# ---------------------------------------------------------------------------


@dataclass
class Amount:
    n: int


class Bearing(enum.Enum):
    North = "norte"


def test_a_union_reads_the_same_whichever_order_it_is_written_in() -> None:
    # The document carries the discriminant, so the order the alternatives
    # were written in is not part of the shape.  `int | float` is the case
    # that used to differ: a float shape takes an integer-spelled number, so
    # whichever member was tried first decided what came back.
    assert sop.loads[int | float]("1") == 1
    assert sop.loads[float | int]("1") == 1
    assert isinstance(sop.loads[float | int]("1"), int)
    assert isinstance(sop.loads[int | float]("1.5"), float)
    assert isinstance(sop.loads[float | int]("1.5"), float)


def test_number_kind_decides_within_a_union_as_it_does_outside_one() -> None:
    # Spelling determines kind everywhere, unions included.
    assert isinstance(sop.loads[int | float]("1e3"), float)
    assert isinstance(sop.loads[int | float]("-0"), int)


def test_a_float_shape_still_widens_when_nothing_claims_the_integer() -> None:
    # `Float` takes an integer-spelled number, but only where no `int` member
    # is there to take it first.
    assert sop.loads[float | str]("1") == 1.0
    assert isinstance(sop.loads[float | str]("1"), float)


def test_members_that_cannot_be_told_apart_are_refused_when_the_shape_is_read() -> None:
    # Ambiguity is a fact about the schema, not about any one document, so it
    # is an error the first time the shape is read rather than on whichever
    # document happens to reach the collision.
    with pytest.raises(
        sop.ShapeError, match="cannot be told apart: both read an array"
    ):
        sop.loads[list[int] | tuple[int, ...]]("[]")
    with pytest.raises(
        sop.ShapeError, match="cannot be told apart: both read an object"
    ):
        sop.loads[dict[str, int] | frozendict[str, int]]("{}")


def test_two_enums_sharing_a_spelling_cannot_be_told_apart() -> None:
    class Compass(enum.Enum):
        North = "norte"  # the same spelling as `Bearing.North`

    with pytest.raises(
        sop.ShapeError, match="cannot be told apart: both read the symbol `norte`"
    ):
        sop.loads[Bearing | Compass]("norte")


def test_enums_with_disjoint_spellings_discriminate_on_them() -> None:
    class Season(enum.Enum):
        Summer = "verano"

    assert sop.loads[Bearing | Season]("norte") is Bearing.North
    assert sop.loads[Season | Bearing]("verano") is Season.Summer


def test_a_wildcard_does_not_collide_with_a_member_that_names_the_value() -> None:
    # `Symbol` takes any symbol and `Tagged` any tag, so a shape that names
    # one beats them rather than being ambiguous with them.
    assert sop.loads[Bearing | sop.Symbol]("norte") is Bearing.North
    assert sop.loads[Bearing | sop.Symbol]("otro") == sop.Symbol("otro")
    assert sop.loads[Amount | sop.Tagged]("Amount {n: 1}") == Amount(1)
    assert sop.loads[Amount | sop.Tagged]("Other 1") == sop.Tagged("Other", 1)


# ---------------------------------------------------------------------------
# Shapes that contain themselves
# ---------------------------------------------------------------------------

type Tree = int | list[Tree]
type Nest = int | dict[str, Nest]


def test_a_recursive_alias_reads() -> None:
    assert sop.loads[Tree]("[1, [2, [3]], 4]") == [1, [2, [3]], 4]


def test_a_recursive_alias_through_an_object_reads() -> None:
    assert sop.loads[Nest]("{a: {b: 1}, c: 2}") == {"a": {"b": 1}, "c": 2}


def test_an_alias_that_is_one_of_its_own_alternatives_is_refused() -> None:
    # `type T = T | int` has nothing to read before reading itself again.
    # mypy refuses it for the same reason, wherever it is written, so the
    # ignore below is the two agreeing rather than a complaint being
    # silenced -- which is also why this alias stays here and the two
    # readable ones moved up to module scope.
    type Loop = Loop | int  # type: ignore[misc]

    with pytest.raises(sop.ShapeError, match="`Loop` is one of its own union members"):
        sop.loads[Loop]("1")


@dataclass
class Chain:
    name: str
    next: Chain | None = None


def test_a_self_referential_dataclass_reads() -> None:
    assert sop.loads[Chain]('Chain {name: "a", next: Chain {name: "b"}}') == Chain(
        "a", Chain("b")
    )


@dataclass
class Ping:
    pong: Pong | None = None


@dataclass
class Pong:
    ping: Ping | None = None


def test_mutually_recursive_dataclasses_read() -> None:
    assert sop.loads[Ping]("Ping {pong: Pong {ping: null}}") == Ping(Pong(None))


def test_a_set_shape_needs_the_tag_to_carry_an_array() -> None:
    with pytest.raises(sop.ShapeError, match="`Set` carries an array, found a number"):
        sop.loads[set[int]]("Set 1")


type Scalar = int | str


def test_a_union_member_that_is_itself_a_union_answers_for_its_own_members() -> None:
    # An alias can denote a union, so a union's member can be one.  It is left
    # whole and asked what it reads, rather than being taken apart.
    assert sop.loads[Scalar | None]("1") == 1
    assert sop.loads[Scalar | None]('"x"') == "x"
    assert sop.loads[Scalar | None]("null") is None
    with pytest.raises(sop.ShapeError, match="found the symbol `Active`"):
        sop.loads[Scalar | None]("Active")


def test_a_nested_union_still_collides_across_the_two_levels() -> None:
    with pytest.raises(sop.ShapeError, match="cannot be told apart"):
        sop.loads[Scalar | int]("1")
