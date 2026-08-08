"""The shape language: every shape, in both directions, and its errors.

Run with `pytest` from the repository root, against an installed build
(`maturin develop`) — the in-tree package has no extension module.
"""

import enum
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

import pytest
import sop

# ---------------------------------------------------------------------------
# Schemas used below
# ---------------------------------------------------------------------------


class Colour(enum.Enum):
    Red = "Red"
    Blue = "azul"  # value and name differ, so both lookups get exercised


class Iban(str):
    __sop_tag__ = "iban"


class Money(Decimal):
    __sop_tag__ = "decimal"


class Id(UUID):
    __sop_tag__ = "uuid"


class Instant(datetime):
    __sop_tag__ = "instant"

    @classmethod
    def __sop_parse__(cls, text: str) -> Instant:
        return cls.fromisoformat(text)


@dataclass
class Plain:
    x: int  # tagged with its own name


@dataclass
class Bare:
    __sop_tag__ = None  # carried as a bare object
    x: int


@dataclass
class Geo:
    __sop_tag__ = "geo"
    lat: float
    lng: float


@dataclass
class Deposit:
    amount: Money


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


def roundtrip(shape, value):
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
def test_scalar_shapes(shape, text, expected):
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
        (sop.Symbol, '"x"', "expected Symbol"),
        (sop.Tagged, "1", "expected Tagged"),
    ],
)
def test_scalar_mismatch(shape, text, message):
    with pytest.raises(sop.ShapeError, match=message):
        sop.loads[shape](text)


def test_unsupported_shape():
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[object]("1")


# ---------------------------------------------------------------------------
# set: a tagged array, which is a different value from a bare array
# ---------------------------------------------------------------------------


def test_set_reads_a_tagged_array():
    assert sop.loads[set[str]]('set ["a", "b"]') == {"a", "b"}


def test_set_rejects_a_bare_array():
    with pytest.raises(sop.ShapeError, match="tagged `set`"):
        sop.loads[set[str]]('["a"]')


def test_set_rejects_another_tag():
    with pytest.raises(sop.ShapeError, match="tagged `set`"):
        sop.loads[set[str]]('bag ["a"]')


def test_list_rejects_a_tagged_array():
    with pytest.raises(sop.ShapeError, match="expected an array"):
        sop.loads[list[str]]('set ["a"]')


def test_set_of_the_wrong_element_type():
    with pytest.raises(sop.ShapeError, match="expected an integer"):
        sop.loads[set[int]]('set ["a"]')


def test_empty_set():
    assert sop.loads[set[int]]("set []") == set()
    assert sop.dumps(set()) == "set []"


def test_set_writes_a_tagged_array():
    assert sop.dumps({"a"}) == 'set ["a"]'


def test_set_output_is_deterministic():
    # sop arrays are ordered and Python sets are not, so the writer has to pick
    # an order and stick to it, or output stops being reproducible.
    values = {3, 1, 2, 10}
    assert sop.dumps(values) == sop.dumps(set(reversed(list(values))))


def test_frozenset_writes_the_same_way():
    assert sop.dumps(frozenset({"a"})) == 'set ["a"]'


@pytest.mark.parametrize("value", [set(), {1}, {1, 2, 3}, {"a", "b"}])
def test_set_roundtrip(value):
    assert roundtrip(set[Any], value) == value


def test_nested_sets():
    value = {"tags": {"a", "b"}, "none": set()}
    assert roundtrip(dict[str, set[str]], value) == value


def test_set_inside_a_dataclass():
    @dataclass
    class Holder:
        __sop_tag__ = None
        roles: set[str]

    assert sop.loads[Holder]('{roles: set ["admin"]}') == Holder({"admin"})
    assert sop.dumps(Holder({"admin"})) == '{roles:set ["admin"]}'


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def test_list_of_lists():
    assert sop.loads[list[list[int]]]("[[1],[2,3],[]]") == [[1], [2, 3], []]


def test_dict_of_lists():
    assert sop.loads[dict[str, list[int]]]("{a:[1],b:[]}") == {"a": [1], "b": []}


def test_dict_rejects_an_array():
    with pytest.raises(sop.ShapeError, match="expected an object"):
        sop.loads[dict[str, int]]("[1]")


def test_list_rejects_an_object():
    with pytest.raises(sop.ShapeError, match="expected an array"):
        sop.loads[list[int]]("{a:1}")


def test_tuple_reads_an_array_immutably():
    # The wire value is the one `list[T]` reads; the shape declares that the
    # result does not mutate.
    assert sop.loads[tuple[int, ...]]("[1,2]") == (1, 2)
    assert sop.loads[tuple[int, ...]]("[]") == ()
    assert roundtrip(tuple[int, ...], (1, 2, 3)) == (1, 2, 3)


def test_only_the_homogeneous_tuple_is_a_shape():
    # `tuple[int, str]` is a row type, which a sop array does not model.
    with pytest.raises(sop.ShapeError, match=r"only `tuple\[T, \.\.\.\]`"):
        sop.loads[tuple[int, str]]("[1]")


def test_tuples_are_written_as_arrays():
    assert sop.dumps((1, 2)) == "[1,2]"


def test_frozenset_reads_a_tagged_array_immutably():
    value = sop.loads[frozenset[str]]('set ["a", "b"]')
    assert value == frozenset({"a", "b"}) and isinstance(value, frozenset)
    assert roundtrip(frozenset[str], frozenset({"a", "b"})) == {"a", "b"}


def test_dict_keys_must_be_str_shaped():
    # An object's keys are strings; a shape claiming otherwise would decode to
    # something its own annotation contradicts.
    with pytest.raises(sop.ShapeError, match="object keys are strings"):
        sop.loads[dict[int, int]]("{a: 1}")
    assert sop.loads[dict[Any, int]]("{a: 1}") == {"a": 1}


def test_frozendict_shape_reads_an_object_immutably():
    value = sop.loads[frozendict[str, int]]("{a: 1}")
    assert isinstance(value, frozendict) and value == {"a": 1}
    assert sop.loads[dict[str, int]]("{a: 1}") == {"a": 1}  # dict opts into mutation


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


def test_enum_by_value():
    assert sop.loads[Colour]("azul") is Colour.Blue


def test_enum_reads_only_its_spelling():
    # `Blue` is written `azul`; the member's Python name is not a second
    # accepted spelling.
    with pytest.raises(sop.ShapeError, match="`Blue` is not one of `Red`, `azul`"):
        sop.loads[Colour]("Blue")


def test_enum_writes_its_value():
    assert sop.dumps(Colour.Blue) == "azul"


def test_enum_roundtrip():
    assert roundtrip(Colour, Colour.Red) is Colour.Red


def test_enum_rejects_a_string():
    with pytest.raises(sop.ShapeError, match="expected a symbol"):
        sop.loads[Colour]('"azul"')


def test_enum_rejects_an_unknown_symbol():
    with pytest.raises(sop.ShapeError, match="is not one of"):
        sop.loads[Colour]("Green")


def test_an_enum_value_with_no_symbol_spelling_is_refused():
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


def test_a_class_is_tagged_with_its_own_name():
    assert sop.loads[Plain]("Plain { x: 1 }") == Plain(1)
    assert sop.dumps(Plain(1)) == "Plain {x:1}"


def test_none_opts_out_of_the_tag():
    assert sop.loads[Bare]("{ x: 1 }") == Bare(1)
    assert sop.dumps(Bare(1)) == "{x:1}"


def test_an_explicit_tag_overrides_the_name():
    assert sop.loads[Geo]("geo { lat: 1, lng: 2 }") == Geo(1.0, 2.0)
    assert sop.dumps(Geo(1.0, 2.0)) == "geo {lat:1.0,lng:2.0}"


def test_a_tagged_class_rejects_a_bare_object():
    with pytest.raises(sop.ShapeError, match="tagged `Plain`"):
        sop.loads[Plain]("{ x: 1 }")


def test_a_tagged_class_rejects_the_wrong_tag():
    with pytest.raises(sop.ShapeError, match="expected tag `Plain`"):
        sop.loads[Plain]("Other { x: 1 }")


def test_an_untagged_class_rejects_a_tagged_object():
    with pytest.raises(sop.ShapeError, match="expected an object"):
        sop.loads[Bare]("Bare { x: 1 }")


def test_builtins_do_not_get_a_default_tag():
    # `str` and `list` are how the format's own types are spelled. Defaulting
    # them to a tag would make `dumps("x")` produce `str "x"`.
    assert sop.dumps("x") == '"x"'
    assert sop.dumps([1]) == "[1]"
    assert sop.dumps({"a": 1.5}) == "{a:1.5}"
    assert sop.loads[dict[str, float]]("{a: 1.5}") == {"a": 1.5}


# ---------------------------------------------------------------------------
# Tagged scalars
# ---------------------------------------------------------------------------


def test_tagged_scalar_is_built_from_its_text():
    assert sop.loads[Iban]('iban "DE89"') == "DE89"
    assert isinstance(sop.loads[Iban]('iban "DE89"'), Iban)


def test_tagged_scalar_is_spelled_with_str():
    assert sop.dumps(Iban("DE89")) == 'iban "DE89"'
    assert sop.dumps(Money("19.99")) == 'decimal "19.99"'


def test_sop_parse_hook():
    value = sop.loads[Instant]('instant "2026-08-05T14:23:11"')
    assert value.year == 2026 and isinstance(value, Instant)


@pytest.mark.parametrize(
    "shape, value",
    [
        (Iban, Iban("DE89")),
        (Money, Money("19.99")),
        (Id, Id(int=7)),
        (Instant, Instant(2026, 8, 5, 14, 23, 11)),
    ],
)
def test_tagged_scalar_roundtrip(shape, value):
    assert roundtrip(shape, value) == value


def test_tagged_scalar_rejects_the_wrong_tag():
    with pytest.raises(sop.ShapeError, match="tagged `iban`"):
        sop.loads[Iban]('swift "DE89"')


def test_tagged_scalar_rejects_a_non_string_payload():
    with pytest.raises(sop.ShapeError, match="carries a string"):
        sop.loads[Iban]("iban 1")


def test_tagged_scalar_reports_a_bad_value():
    with pytest.raises(sop.ShapeError, match="is not valid"):
        sop.loads[Money]('decimal "not a number"')


def test_unknown_tags_survive_as_Tagged():
    # Nothing in the schema models `duration`, and it still round-trips.
    value = sop.loads[sop.Tagged]('duration "PT15M"')
    assert (value.tag, value.value) == ("duration", "PT15M")
    assert sop.dumps(value) == 'duration "PT15M"'


def test_a_class_without_a_declared_tag_is_not_a_scalar():
    # A dataclass declares its fields, so its name can carry it as an object;
    # an arbitrary class declares nothing, so without an explicit
    # `__sop_tag__` it is not carried at all -- rather than being spelled
    # `Anon "<object at 0x…>"` through `str`.
    class Anon:
        pass

    with pytest.raises(sop.ShapeError, match="cannot encode Anon"):
        sop.dumps(Anon())
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Anon]('"x"')


# ---------------------------------------------------------------------------
# Unions
# ---------------------------------------------------------------------------


def test_optional():
    assert sop.loads[int | None]("null") is None
    assert sop.loads[int | None]("1") == 1


def test_tagged_union_discriminates_on_the_tag():
    events = sop.loads[list[Deposit | Withdraw]](
        '[Deposit { amount: decimal "5.00" }, Withdraw { atm: "A1" }]'
    )
    assert isinstance(events[0], Deposit) and isinstance(events[1], Withdraw)


def test_tagged_union_names_the_alternatives():
    with pytest.raises(sop.ShapeError, match="`Deposit`, `Withdraw`"):
        sop.loads[Deposit | Withdraw]("Other { atm: 1 }")


def test_tagged_union_rejects_an_untagged_value():
    with pytest.raises(sop.ShapeError, match="expected a value tagged"):
        sop.loads[Deposit | Withdraw]("{ atm: 1 }")


def test_untagged_union_tries_each_member():
    assert sop.loads[int | str]("1") == 1
    assert sop.loads[int | str]('"a"') == "a"


def test_untagged_union_reports_every_reason():
    with pytest.raises(sop.ShapeError, match="no union member matched"):
        sop.loads[int | str]("Active")


def test_mixed_union():
    assert sop.loads[Deposit | int]("7") == 7
    assert isinstance(sop.loads[Deposit | int]('Deposit { amount: decimal "1" }'), Deposit)


def test_a_shared_tag_in_a_union_is_an_error():
    # A shared discriminant is a schema bug; silently degrading to trying
    # members in order would hide it until the members diverged.
    @dataclass
    class One:
        __sop_tag__ = "dup"
        x: int

    @dataclass
    class Two:
        __sop_tag__ = "dup"
        y: int

    with pytest.raises(sop.ShapeError, match="share the tag `dup`"):
        sop.loads[One | Two]("dup {x: 1}")


def test_a_non_string_enum_matches_by_name():
    assert sop.loads[Priority]("High") is Priority.High
    assert sop.dumps(Priority.High) == "High"


def test_null_in_a_wider_union():
    assert sop.loads[int | str | None]("null") is None
    assert sop.loads[int | str | None]('"x"') == "x"


def test_a_recursive_shape_resolves():
    tree = sop.loads[Branch]('Branch { name: "root", children: [Branch { name: "leaf" }] }')
    assert tree.children[0].name == "leaf"
    assert sop.loads[Branch](sop.dumps(tree)) == tree


def test_an_exotic_alias_is_not_a_shape():
    # Legal to write, but no row of the shape table matches it.
    with pytest.raises(sop.ShapeError, match="unsupported shape"):
        sop.loads[Annotated[int, []]]("1")


def test_kind_matched_members_do_not_discriminate():
    # Enums are carried as symbols, and `Tagged` and `Any` match on kind, so
    # none of them is a wire tag the union can key on.
    assert sop.loads[Colour | Plain]("azul") is Colour.Blue
    assert sop.loads[Plain | sop.Tagged]('duration "PT15M"').tag == "duration"
    assert sop.loads[Plain | Any]("7") == 7


# ---------------------------------------------------------------------------
# Dataclass fields
# ---------------------------------------------------------------------------


@dataclass
class Fields:
    __sop_tag__ = None
    from_: str = field(metadata={"sop": "from"})  # metadata names the wire key
    cls_: str = field(default="c", metadata={"sop": "class"})
    n: int = 3  # default
    computed: int = field(init=False, default=0)  # not read back


def test_metadata_names_the_wire_key():
    assert sop.loads[Fields]('{from: "x"}').from_ == "x"
    assert sop.loads[Fields]('{from: "x", class: "y"}').cls_ == "y"


def test_a_field_name_is_otherwise_verbatim():
    # One aliasing mechanism: without metadata, the field's own name is the
    # wire key, underscore and all.
    @dataclass
    class Verbatim:
        __sop_tag__ = None
        key_: str

    assert sop.loads[Verbatim]('{key_: "x"}').key_ == "x"
    assert sop.dumps(Verbatim("x")) == '{key_:"x"}'


def test_defaults_are_used_when_a_key_is_absent():
    assert sop.loads[Fields]('{from: "x"}').n == 3


def test_missing_required_key():
    with pytest.raises(sop.ShapeError, match="missing key `from`"):
        sop.loads[Fields]("{}")


def test_unknown_keys_are_ignored():
    assert sop.loads[Fields]('{from: "x", surplus: 1}').from_ == "x"


def test_init_false_fields_are_not_read():
    assert sop.loads[Fields]('{from: "x", computed: 9}').computed == 0


def test_a_validating_dataclass_reports_a_shape_error():
    # A `__post_init__` that raises ValueError is the document missing the
    # shape, and it surfaces with a path like every other mismatch.
    @dataclass
    class Positive:
        __sop_tag__ = None
        n: int

        def __post_init__(self) -> None:
            if self.n < 0:
                raise ValueError("n must be positive")

    with pytest.raises(sop.ShapeError, match="not a valid Positive") as caught:
        sop.loads[list[Positive]]("[{n: -1}]")
    assert str(caught.value).startswith("$[0]: ")


def test_field_names_are_written_with_their_alias():
    assert sop.dumps(Fields("x")) == '{from:"x",class:"c",n:3,computed:0}'


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, text, path",
    [
        (dict[str, list[int]], '{a: [1, "x"]}', "$.a[1]"),
        (list[Geo], "[geo {lat: 1, lng: Active}]", "$[0].lng"),
        (dict[str, dict[str, int]], "{a: {b: Active}}", "$.a.b"),
        (list[int], '["x"]', "$[0]"),
        (Fields, "{}", "$"),
    ],
)
def test_error_paths(shape, text, path):
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
    __sop_tag__ = None
    id: Id
    created_at: Instant
    balance: Money
    location: Geo
    roles: set[str]
    status: Status
    deleted_at: Instant | None
    session_ttl: sop.Tagged | None = None


TYPED_RESPONSE = """{
  id: uuid "9f1c2e7a-3b44-4f80-9c1d-2a5e7b0f1234",
  created_at: instant "2026-08-05T14:23:11Z",
  balance: decimal "19.99",
  session_ttl: duration "PT15M",
  location: geo { lat: 47.6062, lng: -122.3321 },
  roles: set ["admin", "beta"],
  status: Active,
  deleted_at: null,
}"""


def test_a_typed_api_response():
    account = sop.loads[Account](TYPED_RESPONSE)
    assert account.id == Id("9f1c2e7a-3b44-4f80-9c1d-2a5e7b0f1234")
    assert account.balance == Money("19.99")
    assert account.location.lat == 47.6062
    assert account.roles == {"admin", "beta"}
    assert account.status is Status.Active
    assert account.deleted_at is None
    assert account.session_ttl.tag == "duration"  # an unmodelled tag survives
    assert roundtrip(Account, account) == account


@dataclass
class Reversal:
    reason: Status


DISCRIMINATED_UNION = """[
  Deposit  { amount: decimal "500.00" },
  Withdraw { atm: "ATM-4417" },
  Reversal { reason: Suspended },
]"""


def test_a_discriminated_union():
    events = sop.loads[list[Deposit | Withdraw | Reversal]](DISCRIMINATED_UNION)
    assert [type(e).__name__ for e in events] == ["Deposit", "Withdraw", "Reversal"]
    assert roundtrip(list[Deposit | Withdraw | Reversal], events) == events


@dataclass
class Tcp:
    __sop_tag__ = "tcp"
    host: str
    port: int


@dataclass
class Unix:
    __sop_tag__ = "unix"
    path: str


@dataclass
class Service:
    __sop_tag__ = "service"
    name: str
    listen: list[Tcp | Unix]
    replicas: int
    canary: bool
    labels: dict[str, str] = field(default_factory=dict)


CONFIGURATION = """service {
  name: "billing-api",
  listen: [
    tcp { host: "0.0.0.0", port: 8080 },
    unix { path: "/var/run/billing.sock" },
  ],
  replicas: 3,
  canary: false,
  labels: { tier: "gold" },
}"""


def test_a_configuration_document():
    service = sop.loads[Service](CONFIGURATION)
    assert service.name == "billing-api"
    assert isinstance(service.listen[0], Tcp) and isinstance(service.listen[1], Unix)
    assert service.canary is False
    assert service.labels == {"tier": "gold"}
    assert roundtrip(Service, service) == service
