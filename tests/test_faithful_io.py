"""CPU tests for scripts/faithful_io.py — the SFT output schema, serialization,
tolerant parser, and target builder. Data-driven tests run over the real frozen
data/master.jsonl so the contract is checked against actual records.
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.faithful_io import (
    ABSTAIN_ANSWER,
    FaithfulOutput,
    build_target,
    parse_tolerant,
    serialize,
)

ROOT = Path(__file__).resolve().parent.parent
MASTER = ROOT / "data" / "master.jsonl"


@pytest.fixture(scope="module")
def records() -> list[dict]:
    assert MASTER.exists(), f"missing frozen dataset: {MASTER}"
    return [json.loads(line) for line in MASTER.open()]


# ── test_json_roundtrip ───────────────────────────────────────────────────────

def test_json_roundtrip(records):
    """Every target is valid JSON, parses back, and re-serializes identically."""
    crafted = [("2013", [5]), ("unanswerable", []), ("San Diego", [1, 2, 3]),
               ("a{b}c", [9]), ("café", [0])]
    for ans, ids in crafted:
        s = serialize(ans, ids)
        obj = json.loads(s)                                  # valid JSON
        assert list(obj.keys()) == ["answer", "evidence_ids"]  # fixed key order
        p = parse_tolerant(s)
        assert p is not None and p.answer == ans and p.evidence_ids == ids
        assert serialize(p.answer, p.evidence_ids) == s       # idempotent round-trip

    # data-driven: every record's target round-trips
    for r in records:
        t = build_target(r)
        json.loads(t)                                         # parses as JSON
        p = parse_tolerant(t)
        assert p is not None, f"target did not parse: {t!r}"
        assert serialize(p.answer, p.evidence_ids) == t


# ── test_parser_robustness ────────────────────────────────────────────────────

def test_parser_robustness():
    """Tolerant extractor recovers the object from messy inputs; None on broken."""
    base = serialize("2013", [5])

    recoverable = [
        base,
        f"```json\n{base}\n```",
        f"```\n{base}\n```",
        f"   \n{base}\n   ",
        f"Here is the answer: {base} -- done.",
        f'prefix {{"answer":"a{{b}}c","evidence_ids":[1]}} suffix',  # braces in string
    ]
    for g in recoverable:
        assert parse_tolerant(g) is not None, f"should recover: {g!r}"

    assert parse_tolerant(f"Here: {base}").answer == "2013"
    p = parse_tolerant('x {"answer":"a{b}c","evidence_ids":[1]} y')
    assert p is not None and p.answer == "a{b}c" and p.evidence_ids == [1]

    broken = [
        "",
        "no json here at all",
        "{ broken json",
        '{"answer":"x"}',                                   # missing evidence_ids
        '{"evidence_ids":[1]}',                             # missing answer
        "[1,2,3]",                                          # list, not object
        '{"answer":"x","evidence_ids":[1],"extra":1}',      # extra key forbidden
        '{"answer":"x","evidence_ids":["1"]}',              # string ids
        '{"answer":"x","evidence_ids":[1.5]}',              # float id
        '{"answer":"x","evidence_ids":[true]}',             # bool id
        '{"answer":5,"evidence_ids":[]}',                   # non-str answer
        None,                                               # not a string
    ]
    for b in broken:
        assert parse_tolerant(b) is None, f"expected None for {b!r}"


# ── test_abstain_target ───────────────────────────────────────────────────────

def test_abstain_target(records):
    """Abstain records serialize to exactly {"answer":"unanswerable","evidence_ids":[]}."""
    expected = '{"answer":"unanswerable","evidence_ids":[]}'
    assert serialize(ABSTAIN_ANSWER, []) == expected

    abstains = [r for r in records if r["target_is_abstain"]]
    assert abstains, "no abstain records in master.jsonl"
    for r in abstains:
        assert build_target(r) == expected


# ── test_citation_validity ────────────────────────────────────────────────────

def test_citation_validity(records):
    """Answerable targets' evidence_ids are non-empty and ⊆ that record's menu ids."""
    answerables = [r for r in records if not r["target_is_abstain"]]
    assert answerables, "no answerable records in master.jsonl"
    for r in answerables:
        menu_ids = {m["id"] for m in r["ocr_menu"]}
        p = parse_tolerant(build_target(r))
        assert p is not None
        assert p.evidence_ids, f"answerable {r['record_id']} produced empty evidence"
        assert set(p.evidence_ids).issubset(menu_ids), (
            f"record {r['record_id']} cites ids outside its menu"
        )


# ── test_split_integrity (CLAUDE.md required) ─────────────────────────────────

def test_split_integrity(records):
    """No doc_id appears in both train and eval."""
    doc_to_splits: dict[int, set] = defaultdict(set)
    for r in records:
        doc_to_splits[r["doc_id"]].add(r["split"])
    leaked = {d: s for d, s in doc_to_splits.items() if len(s) != 1}
    assert not leaked, f"doc_ids in multiple splits: {list(leaked.items())[:5]}"
