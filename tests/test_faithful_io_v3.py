"""CPU tests for the v2 3-field contract in scripts/faithful_io.py.

Mirrors tests/test_faithful_io.py: round-trip (serialize -> parse -> equal) for
answerable AND abstain rows, malformed-by-failure-type classification (incl. the new
boolean field), and the cross-field contradiction cases. Plus a real-data round-trip
on derived records (build_3field.derive_3field + build_target_v3).
"""

import json

import pytest

from scripts.faithful_io import derive_3field
from scripts.faithful_io import (
    FaithfulOutput3Field,
    build_target_v3,
    parse_tolerant_v3,
    serialize_v3,
)
from scripts.format_sft import load_master


# ── round-trip ────────────────────────────────────────────────────────────────

def test_roundtrip_answerable():
    s = serialize_v3("0.28", True, [8])
    r = parse_tolerant_v3(s)
    assert r.ok and r.failure is None
    assert r.value.found_answer == "0.28"
    assert r.value.answerable_from_menu is True
    assert r.value.evidence_ids == [8]


def test_roundtrip_abstain_keeps_found_answer():
    """The new behavior: an abstain target carries a NON-empty found_answer with the
    menu-flag false and empty evidence, and that survives the round-trip."""
    s = serialize_v3("29", False, [])
    r = parse_tolerant_v3(s)
    assert r.ok and r.failure is None
    assert r.value.found_answer == "29"            # non-empty — used to be thrown away
    assert r.value.answerable_from_menu is False
    assert r.value.evidence_ids == []


def test_serialize_is_canonical_and_keeps_raw_casing():
    # fixed key order, compact separators, RAW casing preserved (deliberate)
    assert serialize_v3("ITC Limited", False, []) == \
        '{"found_answer":"ITC Limited","answerable_from_menu":false,"evidence_ids":[]}'
    assert parse_tolerant_v3(serialize_v3("ITC Limited", False, [])).value.found_answer == "ITC Limited"


def test_build_target_v3_from_v2_contract():
    ans = {"record_id": "a", "v2_contract": {"found_answer": "Paris",
           "answerable_from_menu": True, "evidence_ids": [3, 4]}}
    abst = {"record_id": "b", "v2_contract": {"found_answer": "29",
            "answerable_from_menu": False, "evidence_ids": []}}
    assert build_target_v3(ans) == serialize_v3("Paris", True, [3, 4])
    rb = parse_tolerant_v3(build_target_v3(abst))
    assert rb.value.found_answer == "29" and rb.value.answerable_from_menu is False and rb.value.evidence_ids == []
    # a record missing the derived block fails loudly, not silently
    with pytest.raises(KeyError):
        build_target_v3({"record_id": "c"})


# ── malformed, classified by failure type ─────────────────────────────────────

def test_malformed_no_json():
    assert parse_tolerant_v3("sorry, I could not find it").failure == "no_json"


def test_malformed_invalid_json():
    # balanced braces but not valid JSON (`tru`) -> json.loads fails -> invalid_json.
    # (an *unterminated* object has no balanced {...} and is classified no_json instead.)
    assert parse_tolerant_v3('{"found_answer":"x","answerable_from_menu":tru,"evidence_ids":[]}').failure == "invalid_json"
    assert parse_tolerant_v3('{"found_answer": "x", "answerable_from_menu":').failure == "no_json"


def test_malformed_extra_key():
    raw = json.dumps({"found_answer": "x", "answerable_from_menu": True,
                      "evidence_ids": [1], "confidence": 0.9})
    assert parse_tolerant_v3(raw).failure == "schema"


def test_malformed_bool_as_string():
    """answerable_from_menu = "false" (string) is wrong-type, not coerced -> schema."""
    raw = json.dumps({"found_answer": "x", "answerable_from_menu": "false", "evidence_ids": []})
    assert parse_tolerant_v3(raw).failure == "schema"


def test_malformed_evidence_not_int():
    raw = json.dumps({"found_answer": "x", "answerable_from_menu": True, "evidence_ids": ["1"]})
    assert parse_tolerant_v3(raw).failure == "schema"


def test_malformed_missing_field():
    raw = json.dumps({"found_answer": "x", "answerable_from_menu": True})  # no evidence_ids
    assert parse_tolerant_v3(raw).failure == "schema"


# ── contradiction cases (cross-field contract) ────────────────────────────────

def test_contradiction_false_flag_with_evidence():
    raw = json.dumps({"found_answer": "x", "answerable_from_menu": False, "evidence_ids": [5]})
    r = parse_tolerant_v3(raw)
    assert r.value is None and r.failure == "contradiction"


def test_contradiction_true_flag_with_empty_evidence():
    raw = json.dumps({"found_answer": "x", "answerable_from_menu": True, "evidence_ids": []})
    r = parse_tolerant_v3(raw)
    assert r.value is None and r.failure == "contradiction"


def test_build_target_v3_rejects_contradictory_record():
    bad = {"record_id": "z", "v2_contract": {"found_answer": "x",
           "answerable_from_menu": False, "evidence_ids": [5]}}
    with pytest.raises(AssertionError):
        build_target_v3(bad)


# ── real-data round-trip (derive -> build_target_v3 -> parse) ──────────────────

def test_real_data_roundtrip():
    """On real master rows (a few answerable + abstain): derive the contract, build
    the target, parse it back, and confirm it matches the derivation."""
    recs = load_master()
    ans = [r for r in recs if not r["target_is_abstain"]][:5]
    abst = [r for r in recs if r["target_is_abstain"]][:5]
    assert ans and abst
    for r in ans + abst:
        r = {**r, "v2_contract": derive_3field(r)}
        parsed = parse_tolerant_v3(build_target_v3(r))
        assert parsed.ok, f"{r['record_id']} did not round-trip ({parsed.failure})"
        c = r["v2_contract"]
        assert parsed.value.found_answer == c["found_answer"]
        assert parsed.value.answerable_from_menu == c["answerable_from_menu"]
        assert parsed.value.evidence_ids == c["evidence_ids"]
        # the new behavior, on real abstain rows: found_answer is non-empty
        if r["target_is_abstain"]:
            assert parsed.value.found_answer and parsed.value.answerable_from_menu is False \
                and parsed.value.evidence_ids == []
