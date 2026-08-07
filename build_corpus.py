#!/usr/bin/env python3
"""Builds corpus.json, the shared test corpus for both implementations.

Written as a generator rather than by hand so that the escaping in the source
strings stays readable.  Expected outputs are written out by hand; neither
implementation is used to produce them.
"""

import json
import pathlib

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

TYPED_RESPONSE_OUT = (
    '{id:uuid "9f1c2e7a-3b44-4f80-9c1d-2a5e7b0f1234",'
    'created_at:instant "2026-08-05T14:23:11Z",'
    'balance:decimal "19.99",'
    'session_ttl:duration "PT15M",'
    "location:geo {lat:47.6062,lng:-122.3321},"
    'roles:set ["admin","beta"],'
    "status:Active,deleted_at:null}"
)

DISCRIMINATED_UNION = """[
  Deposit  { amount: decimal "500.00", from: iban "DE89370400440532013000" },
  Withdraw { amount: decimal "20.00",  atm: "ATM-4417" },
  Reversal { of: uuid "8c1d0a3e-...", reason: InsufficientFunds },
]"""

DISCRIMINATED_UNION_OUT = (
    '[Deposit {amount:decimal "500.00",from:iban "DE89370400440532013000"},'
    'Withdraw {amount:decimal "20.00",atm:"ATM-4417"},'
    'Reversal {of:uuid "8c1d0a3e-...",reason:InsufficientFunds}]'
)

CONFIGURATION = """service {
  name: "billing-api",
  listen: [
    tcp { host: "0.0.0.0", port: 8080 },
    unix "/var/run/billing.sock",
  ],

  // secret(env("..."))
  db_password: secret env "BILLING_DB_PASSWORD",

  replicas: 3,
  canary: false,
}"""

CONFIGURATION_OUT = (
    'service {name:"billing-api",'
    'listen:[tcp {host:"0.0.0.0",port:8080},unix "/var/run/billing.sock"],'
    'db_password:secret env "BILLING_DB_PASSWORD",'
    "replicas:3,canary:false}"
)

# (name, source, expected compact serialisation)
VALID = [
    ("json_object", '{"a": 1, "b": [true, false, null]}', "{a:1,b:[true,false,null]}"),
    ("empty_object", "{}", "{}"),
    ("empty_array", "[]", "[]"),
    ("trailing_comma_array", "[1,2,3,]", "[1,2,3]"),
    ("trailing_comma_object", "{a:1,}", "{a:1}"),
    ("line_comment", "[ // note\n 1 ]", "[1]"),
    ("line_comment_at_eof", "1 // note", "1"),
    ("block_comment", "[/* note */1]", "[1]"),
    ("block_comment_no_nest", "/* /* */ 1", "1"),
    ("vt_not_line_terminator", "1 // c\v2", "1"),
    ("crlf_line_comment", "[1, // c\r\n2]", "[1,2]"),
    ("unquoted_keys", "{a:1,$b:2,_c:3}", "{a:1,$b:2,_c:3}"),
    ("key_needing_quotes", '{"a b": 1}', '{"a b":1}'),
    ("no_reserved_words", "{null: 1, if: 2, true: 3}", "{null:1,if:2,true:3}"),
    ("symbol", "foo", "foo"),
    ("tag_named_true", 'true "x"', 'true "x"'),
    ("right_associative", "a b c 1", "a b c 1"),
    ("adjacent_identifiers", "[a b]", "[a b]"),
    ("two_symbols", "[a, b]", "[a,b]"),
    ("tag_on_array", 'set ["admin", "beta"]', 'set ["admin","beta"]'),
    ("nested_tag_object", "geo { lat: 47.6062, lng: -122.3321 }", "geo {lat:47.6062,lng:-122.3321}"),
    ("leading_plus", "[+7]", "[7]"),
    ("negative_zero", "[-0]", "[0]"),
    ("exponent", "[-2.5e10]", "[-25000000000]"),
    ("fraction", "[0.5,1.25]", "[0.5,1.25]"),
    ("tenth", "[0.1]", "[0.1]"),
    ("beyond_double_precision", "[9007199254740993]", "[9007199254740993]"),
    # These used to sit in a "divergent" section because the two
    # implementations spelled them differently. There is one implementation now.
    ("big_integer", "123456789012345678901234567890", "123456789012345680000000000000.0"),
    ("beyond_i64", "9223372036854775808", "9223372036854776000.0"),
    ("large_float", "[1e30]", "[1000000000000000000000000000000.0]"),
    ("small_float", "[1.5e-5]", "[0.000015]"),
    ("float_repr", "[1e16]", "[10000000000000000.0]"),
    ("float_underflow", "[1e-400]", "[0]"),
    ("duplicate_keys", "{a:1,a:2}", "{a:2}"),
    ("bom", "\ufeff{}", "{}"),
    ("unicode_whitespace_2028", "[\u20281\u2028]", "[1]"),
    ("unicode_whitespace_nbsp", "[\u00a01\u00a0]", "[1]"),
    ("unicode_whitespace_ideographic", "[\u30001\u3000]", "[1]"),
    ("unicode_identifier", "{\u00e9t\u00e9: caf\u00e9}", "{\u00e9t\u00e9:caf\u00e9}"),
    # The source uses \u00e9 and \/; the serialiser emits e-acute raw and drops
    # the needless escape of the solidus.
    ("escapes", r'["\u00e9\n\\\"\/\b\f\r\t"]', '["\u00e9' + r'\n\\\"/\b\f\r\t' + '"]'),
    ("surrogate_pair", r'["\ud83d\ude00"]', '["\U0001f600"]'),
    ("astral_raw", '["\U0001f600"]', '["\U0001f600"]'),
    ("del_is_unescaped", '["\u007f"]', '["\u007f"]'),
    # U+2028 and U+0085 end a line comment, but inside a string they are just
    # characters: only U+0000..U+001F have to be escaped.  JSON accepts them
    # too, which matters because every JSON text has to parse.
    ("raw_u2028_in_string", '["a\u2028b"]', '["a\u2028b"]'),
    ("raw_u0085_in_string", '["a\u0085b"]', '["a\u0085b"]'),
    ("deep_nesting", "[[[[1]]]]", "[[[[1]]]]"),
    # The grammar sets no depth bound and neither does the implementation: the
    # parser and the writer are iterative, so nesting costs heap, not stack.
    # 512 levels sits past every conventional recursion limit.
    ("deep_nesting_512", "[" * 512 + "1" + "]" * 512, "[" * 512 + "1" + "]" * 512),
    ("typed_response", TYPED_RESPONSE, TYPED_RESPONSE_OUT),
    ("discriminated_union", DISCRIMINATED_UNION, DISCRIMINATED_UNION_OUT),
    ("configuration", CONFIGURATION, CONFIGURATION_OUT),
]

INVALID = [
    ("tag_on_key", "{ foo bar: 1 }"),
    ("number_as_tag", "[1 2]"),
    ("missing_comma_object", "{a: 1 b: 2}"),
    ("no_digit_before_point", "[.5]"),
    ("no_digit_after_point", "[1.]"),
    ("single_quotes", "['x']"),
    ("leading_zero", "[01]"),
    ("empty_exponent", "[1e]"),
    ("number_key", "{1: 2}"),
    ("raw_control_in_string", '["\u0001"]'),
    ("newline_in_string", '["a\nb"]'),
    ("unterminated_string", '"x'),
    ("unterminated_block_comment", "/* x"),
    ("unterminated_object", "{a: 1"),
    ("unterminated_array", "[1"),
    ("empty_document", ""),
    ("only_a_comment", "// nothing"),
    ("interior_bom", "[1, \ufeff2]"),
    ("missing_value", "{a:}"),
    ("empty_element", "[,]"),
    ("double_comma", "[1,,2]"),
    ("top_level_colon", "a:1"),
    ("bad_escape", r'["\x41"]'),
    ("short_unicode_escape", r'["\u12"]'),
    ("plus_without_digits", "[+]"),
    # These parse as numbers, but the result is infinity, and infinity has no
    # spelling: `Infinity` lexes as a symbol, not a number.  Accepting them
    # would produce a value that could not be written back out.
    ("float_overflow", "[5E547]"),
    # A `str` cannot hold an unpaired surrogate, and substituting U+FFFD would
    # lose data silently.
    ("lone_surrogate", r'["\ud800"]'),
    ("lone_low_surrogate", r'["\udc00"]'),
    ("half_a_pair", r'["\ud83d\u0041"]'),
    ("float_overflow_negative", "-1e400"),
    ("integer_overflow", "1" + "0" * 400),
    ("trailing_content", "1 2"),
    ("error_on_line_three", "[\n  1,\n  2 3\n]"),
]

corpus = {
    "valid": [{"name": n, "src": s, "out": o} for n, s, o in VALID],
    "invalid": [{"name": n, "src": s} for n, s in INVALID],
}

path = pathlib.Path(__file__).parent / "corpus.json"
path.write_text(json.dumps(corpus, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"{path}: {len(VALID)} valid, {len(INVALID)} invalid")
