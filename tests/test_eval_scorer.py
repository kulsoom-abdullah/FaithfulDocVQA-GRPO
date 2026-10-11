"""CPU tests for the live three-field evaluation path.

The unit cases exercise canonical parsing, exclusive bucket classification,
conditional citation-label F1, and fail-loud joins. The integration case feeds
real frozen records through ``eval_runner.evaluate_output`` and ``score`` without
loading a model or requiring page images.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_runner import evaluate_output
from scripts.eval_scorer import (
    CONTRADICTION,
    FABRICATION,
    FALSE_ABSTAIN,
    FALSE_GROUNDING,
    HONEST_ABSTAIN,
    MALFORMED,
    VALID_CITATION,
    aggregate,
    classify,
    score,
)
from scripts.faithful_io import derive_3field, serialize_v3
from scripts.format_sft import load_master

ROOT = Path(__file__).resolve().parent.parent


def rec(
    record_id: str,
    *,
    menu_answerable: bool,
    cited_ids: list[int],
    menu_ids: list[int],
    gold_answers: list[str] | None = None,
    doc_id: int = 1,
) -> dict:
    """Build the minimal frozen-record shape consumed by the live scorer."""
    return {
        "record_id": record_id,
        "doc_id": doc_id,
        "doc_type": ["table/list"],
        "target_is_abstain": not menu_answerable,
        "gold_localized": menu_answerable,
        "cited_ids": cited_ids,
        "gold_answers": gold_answers or ["answer"],
        "ocr_menu": [
            {"id": menu_id, "text": f"t{menu_id}", "bbox": [0, 0, 1, 1]}
            for menu_id in menu_ids
        ],
    }


def target_from_master(record: dict) -> str:
    """Serialize the live contract derived from one frozen ``master.jsonl`` row."""
    contract = derive_3field(record)
    return serialize_v3(
        contract["found_answer"],
        contract["answerable_from_menu"],
        contract["evidence_ids"],
    )


def test_answerable_three_field_buckets() -> None:
    record = rec("a", menu_answerable=True, cited_ids=[1, 2], menu_ids=[0, 1, 2, 3])

    valid = classify(record, serialize_v3("answer", True, [1, 2]))
    assert valid["bucket"] == VALID_CITATION
    assert valid["evidence_reward"] == pytest.approx(1.0)

    fabricated = classify(record, serialize_v3("answer", True, [1, 99]))
    assert fabricated["bucket"] == FABRICATION
    assert fabricated["fabricated_ids"] == [99]

    false_refusal = classify(record, serialize_v3("answer", False, []))
    assert false_refusal["bucket"] == FALSE_ABSTAIN


def test_menu_unanswerable_three_field_buckets() -> None:
    record = rec("b", menu_answerable=False, cited_ids=[], menu_ids=[0, 1, 2])

    honest = classify(record, serialize_v3("answer", False, []))
    assert honest["bucket"] == HONEST_ABSTAIN

    false_grounding = classify(record, serialize_v3("answer", True, [1]))
    assert false_grounding["bucket"] == FALSE_GROUNDING


def test_contract_failures_are_exclusive_buckets() -> None:
    record = rec("c", menu_answerable=True, cited_ids=[1], menu_ids=[0, 1])

    contradiction = classify(
        record,
        '{"found_answer":"answer","answerable_from_menu":false,"evidence_ids":[1]}',
    )
    assert contradiction["bucket"] == CONTRADICTION
    assert contradiction["parse_failure"] == "contradiction"

    malformed = classify(record, "not JSON")
    assert malformed["bucket"] == MALFORMED
    assert malformed["malformed"] is True


def test_bucket_rates_partition_each_frozen_population() -> None:
    records = [
        rec("a1", menu_answerable=True, cited_ids=[1], menu_ids=[1, 2, 3]),
        rec("a2", menu_answerable=True, cited_ids=[1], menu_ids=[1, 2, 3]),
        rec("a3", menu_answerable=True, cited_ids=[1], menu_ids=[1, 2, 3]),
        rec("b1", menu_answerable=False, cited_ids=[], menu_ids=[1, 2, 3]),
        rec("b2", menu_answerable=False, cited_ids=[], menu_ids=[1, 2, 3]),
    ]
    raw = {
        "a1": serialize_v3("answer", True, [1]),
        "a2": serialize_v3("answer", True, [99]),
        "a3": serialize_v3("answer", False, []),
        "b1": serialize_v3("answer", False, []),
        "b2": serialize_v3("answer", True, [1]),
    }
    result = aggregate([classify(record, raw[record["record_id"]]) for record in records])
    answerable = result["bucket_rates"]["answerable"]
    abstention = result["bucket_rates"]["abstain"]

    assert sum(answerable.values()) == pytest.approx(1.0)
    assert sum(abstention.values()) == pytest.approx(1.0)
    assert answerable[VALID_CITATION] == pytest.approx(1 / 3)
    assert answerable[FABRICATION] == pytest.approx(1 / 3)
    assert answerable[FALSE_ABSTAIN] == pytest.approx(1 / 3)
    assert abstention[HONEST_ABSTAIN] == pytest.approx(0.5)
    assert abstention[FALSE_GROUNDING] == pytest.approx(0.5)


def test_conditional_citation_f1_uses_only_eligible_outputs() -> None:
    records = [
        rec("r1", menu_answerable=True, cited_ids=[1, 2, 3], menu_ids=[1, 2, 3, 4, 5, 6]),
        rec("r2", menu_answerable=True, cited_ids=[4], menu_ids=[1, 2, 3, 4, 5, 6]),
        rec("r3", menu_answerable=True, cited_ids=[1], menu_ids=[1, 2, 3, 4, 5, 6]),
    ]
    outcomes = [
        classify(records[0], serialize_v3("answer", True, [1, 2, 3])),
        classify(records[1], serialize_v3("answer", True, [4, 5, 6])),
        classify(records[2], serialize_v3("answer", False, [])),
    ]
    citation = aggregate(outcomes)["citation_f1_gated"]

    assert citation["n_valid_citation"] == 2
    assert citation["micro_precision"] == pytest.approx(4 / 6, abs=1e-4)
    assert citation["micro_recall"] == pytest.approx(1.0)
    assert citation["micro_f1"] == pytest.approx(0.8, abs=1e-4)
    assert citation["macro_f1"] == pytest.approx(0.75, abs=1e-4)


def test_score_rejects_missing_output_and_manifest_rows() -> None:
    records = [
        rec("a", menu_answerable=True, cited_ids=[1], menu_ids=[1, 2], doc_id=10),
        rec("b", menu_answerable=False, cited_ids=[], menu_ids=[1, 2], doc_id=20),
    ]
    raw = {
        "a": serialize_v3("answer", True, [1]),
        "b": serialize_v3("answer", False, []),
    }

    with pytest.raises(ValueError, match="no model output"):
        score(records, {"a": raw["a"]})

    with pytest.raises(KeyError, match="doc_id 20"):
        score(records, raw, cap_manifest={"10": {"was_capped": True}})


def test_runner_uses_canonical_three_field_scorer_on_real_records() -> None:
    records = load_master()
    answerable = next(record for record in records if record["gold_localized"])
    abstention = next(record for record in records if not record["gold_localized"])

    answerable_facts = evaluate_output(answerable, target_from_master(answerable))
    abstention_facts = evaluate_output(abstention, target_from_master(abstention))

    assert answerable_facts["bucket"] == VALID_CITATION
    assert answerable_facts["answerable_from_menu"] is True
    assert abstention_facts["bucket"] == HONEST_ABSTAIN
    assert abstention_facts["answerable_from_menu"] is False


def test_real_frozen_targets_score_perfectly_without_images() -> None:
    records = [record for record in load_master() if record["split"] == "test"]
    assert records, "no test records found"
    manifest = json.loads((ROOT / "data" / "images_cap_manifest.json").read_text())
    raw = {record["record_id"]: target_from_master(record) for record in records}

    result = score(records, raw, cap_manifest=manifest)
    overall = result["overall"]
    answerable = overall["bucket_rates"]["answerable"]
    abstention = overall["bucket_rates"]["abstain"]
    citation = overall["citation_f1_gated"]

    assert answerable[VALID_CITATION] == pytest.approx(1.0)
    assert answerable[FALSE_ABSTAIN] == pytest.approx(0.0)
    assert answerable[FABRICATION] == pytest.approx(0.0)
    assert abstention[HONEST_ABSTAIN] == pytest.approx(1.0)
    assert abstention[FALSE_GROUNDING] == pytest.approx(0.0)
    assert citation["n_valid_citation"] == overall["n_answerable"]
    assert citation["micro_f1"] == pytest.approx(1.0)
    assert citation["macro_f1"] == pytest.approx(1.0)
    assert result["capped"]["n"] + result["uncapped"]["n"] == overall["n"]
