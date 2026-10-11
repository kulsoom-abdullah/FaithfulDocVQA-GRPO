"""CPU validation for the compact public evidence artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.validate_public_artifacts import (
    EXPECTED_ARTIFACTS,
    EXPECTED_MANUFACTURED_GROUPS,
    EXPECTED_POPULATIONS,
    FORBIDDEN_LEDGER_KEYS,
    LEDGER_METADATA_KEYS,
    OUTCOME_KEYS,
    SYSTEM_IDS,
    compute_cluster_sensitivity,
    compute_duplicate_sensitivity,
    compute_headline_metrics,
    compute_paired_comparisons,
    compute_secondary_comparisons,
    validate_public_artifacts,
)

ROOT = Path(__file__).resolve().parent.parent
PUBLIC_DIR = ROOT / "artifacts/public"
LEDGER_PATH = PUBLIC_DIR / "evaluation/evaluation_ledger.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def ledger() -> list[dict]:
    return _read_jsonl(LEDGER_PATH)


def test_public_validator_passes_without_private_data() -> None:
    result = validate_public_artifacts(PUBLIC_DIR)
    assert set(result["headline_metrics"]) == set(SYSTEM_IDS)
    assert result["operational_scan"] == {
        "uuid": 0,
        "email": 0,
        "phone": 0,
        "absolute_user_path": 0,
        "provider_identifier": 0,
    }


def test_nested_ledger_schema_and_complete_joins(ledger: list[dict]) -> None:
    assert len(ledger) == 1_006
    assert len({row["row_id"] for row in ledger}) == 1_006
    assert [row["row_id"] for row in ledger] == sorted(row["row_id"] for row in ledger)
    for row in ledger:
        assert set(row) == LEDGER_METADATA_KEYS | set(SYSTEM_IDS)
        assert not (set(row) & FORBIDDEN_LEDGER_KEYS)
        for system_id in SYSTEM_IDS:
            assert set(row[system_id]) == OUTCOME_KEYS
            assert not (set(row[system_id]) & FORBIDDEN_LEDGER_KEYS)


def test_exact_public_populations(ledger: list[dict]) -> None:
    observed = {
        population: sum(row["population"] == population for row in ledger)
        for population in EXPECTED_POPULATIONS
    }
    assert observed == EXPECTED_POPULATIONS
    assert sum(row["included_in_primary"] for row in ledger) == 991
    assert sum(row["population"] == "manufactured_excluded" for row in ledger) == 15
    manufactured_groups = {
        group: sum(row["manufactured_group"] == group for row in ledger)
        for group in EXPECTED_MANUFACTURED_GROUPS
    }
    assert manufactured_groups == EXPECTED_MANUFACTURED_GROUPS


def test_manufactured_groups_reproduce_run2_mechanism_counts(ledger: list[dict]) -> None:
    def bucket_count(group: str, bucket: str) -> int:
        return sum(
            row["manufactured_group"] == group and row["run2"]["bucket"] == bucket
            for row in ledger
        )

    assert bucket_count("string_absent", "honest_abstain") == 189
    assert bucket_count("string_survives_retained", "honest_abstain") == 15
    assert bucket_count("string_survives_retained", "false_grounding") == 11
    assert bucket_count("survivor_excluded", "honest_abstain") == 2


def test_final_system_metrics_reproduce_manifest(ledger: list[dict]) -> None:
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    metrics = compute_headline_metrics(ledger)
    assert metrics == manifest["expected_headline_metrics"]
    expected_counts = {
        "base": (4, 8, 14, 638),
        "sft_seed": (68, 62, 26, 650),
        "run1": (62, 59, 23, 655),
        "run2": (204, 77, 120, 557),
        "balanced_sft": (212, 71, 78, 601),
        "run4a": (217, 76, 99, 580),
        "run4b": (217, 74, 91, 588),
    }
    for system_id, expected in expected_counts.items():
        system = metrics[system_id]
        assert (
            system["clean_manufactured_honest_refusal"]["count"],
            system["natural_ocr_miss_honest_refusal"]["count"],
            system["natural_answerable_false_refusal"]["count"],
            system["conditional_citation_label_f1"]["eligible_count"],
        ) == expected


def test_paired_comparisons_and_exact_probabilities(ledger: list[dict]) -> None:
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    comparisons = compute_paired_comparisons(ledger)
    assert comparisons == manifest["expected_paired_comparisons"]
    assert comparisons["sft_seed_to_run2_clean"]["exact_two_sided_p"] == pytest.approx(
        2.0**-135
    )
    assert comparisons["sft_seed_to_balanced_clean"] == {
        "improved": 144,
        "declined": 0,
        "both_positive": 68,
        "both_negative": 23,
        "discordant": 144,
        "exact_two_sided_p": pytest.approx(8.96831017167883e-44),
        "population": "manufactured_clean",
        "denominator": 235,
        "before": "sft_seed",
        "after": "balanced_sft",
        "endpoint_bucket": "honest_abstain",
        "lower_is_better": False,
    }
    secondary = compute_secondary_comparisons(ledger)
    assert secondary == manifest["secondary_comparisons"]
    assert secondary["run1_to_run2_full_manufactured"]["exact_two_sided_p"] == pytest.approx(
        2.0**-143
    )
    assert secondary["run1_to_run2_full_manufactured"]["denominator"] == 250
    assert secondary["run1_to_run2_full_manufactured"][
        "focused_review_exclusions_included"
    ] == 15
    assert secondary["run1_to_run2_full_manufactured"]["used_in_public_notebook"] is False
    assert comparisons["run2_to_balanced_clean"]["exact_two_sided_p"] == pytest.approx(
        0.11531829833984375
    )
    assert comparisons["balanced_to_run4a_clean"]["exact_two_sided_p"] == 0.0625
    assert comparisons["run4a_to_run4b_false_refusal"]["exact_two_sided_p"] == pytest.approx(
        0.03857421875
    )


def test_document_cluster_and_duplicate_sensitivities(ledger: list[dict]) -> None:
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    cluster = compute_cluster_sensitivity(ledger)
    duplicate = compute_duplicate_sensitivity(ledger)
    assert cluster == manifest["expected_document_cluster_sensitivity"]
    assert cluster["document_count"] == 75
    assert (
        cluster["improved_documents"],
        cluster["declined_documents"],
        cluster["tied_documents"],
    ) == (60, 0, 15)
    assert cluster["exact_two_sided_sign_test_p"] == pytest.approx(2.0**-59)
    assert cluster["bootstrap"]["lower_95"] == 0.502
    assert cluster["bootstrap"]["upper_95"] == 0.653
    assert duplicate == manifest["expected_duplicate_excluded_sensitivity"]
    assert (
        duplicate["full_count"],
        duplicate["full_denominator"],
        duplicate["filtered_count"],
        duplicate["filtered_denominator"],
    ) == (120, 679, 119, 675)


def test_public_manifest_hashes_sizes_and_row_counts() -> None:
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == set(EXPECTED_ARTIFACTS)
    for relative, row_count in EXPECTED_ARTIFACTS.items():
        path = PUBLIC_DIR / relative
        entry = manifest["artifacts"][relative]
        assert entry["sha256"] == _sha256(path)
        assert entry["size_bytes"] == path.stat().st_size
        assert entry["row_count"] == row_count == len(_read_jsonl(path))


def test_private_source_text_and_identifier_attestations_are_clean() -> None:
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["private_validation"]["source_text_match_counts"] == {
        "questions": 0,
        "accepted_answers": 0,
        "ocr_text": 0,
        "raw_completions": 0,
    }
    assert manifest["private_validation"]["opaque_mapping_retained_outside_repository"] is True
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(PUBLIC_DIR.rglob("*"))
        if path.is_file()
    )
    for forbidden in ("/workspace/", "/root/", "/Users/", "request_id", "call_uuid"):
        assert forbidden not in public_text


def test_compact_manifest_summaries_are_exact() -> None:
    manifest = json.loads((PUBLIC_DIR / "manifest.json").read_text(encoding="utf-8"))
    ambiguity = manifest["citation_label_ambiguity_summary"]
    assert ambiguity["all_answerable_rows"] == {
        "affected": 533,
        "denominator": 4_496,
        "rate": 0.1185,
        "display_percent": 11.9,
    }
    assert ambiguity["answerable_test_rows"] == {
        "affected": 80,
        "denominator": 679,
        "rate": 0.1178,
        "display_percent": 11.8,
    }
    assert ambiguity["counterfactual_f1_deltas"] == {
        "sft_seed": 0.0148,
        "run1": 0.0117,
        "run2": 0.0183,
        "balanced_sft": 0.0133,
    }
    assert ambiguity["system_ordering_unchanged"] is True

    checkpoint = manifest["checkpoint_selection_summary"]
    assert checkpoint["original_sft"]["selected_checkpoint"] == 642
    assert checkpoint["original_sft"]["test_generation_artifact_count"] == 6
    assert checkpoint["balanced_sft"]["checkpoint_648"] == {
        "false_refusal_count": 79,
        "malformed_count": 1,
        "answerable_non_valid_output_count": 80,
        "evaluation_loss": 0.092343,
    }
    assert checkpoint["balanced_sft"]["checkpoint_1296"] == {
        "false_refusal_count": 75,
        "malformed_count": 0,
        "answerable_non_valid_output_count": 75,
        "evaluation_loss": 0.095555,
    }
    assert checkpoint["balanced_sft"]["honest_refusal_count"] == 44
    assert checkpoint["balanced_sft"]["honest_refusal_denominator"] == 46

    training = manifest["training_design_summary"]
    assert training["balanced_sft"]["manufactured_abstention_rows"] == 1_764
    assert training["run4"]["same_ordered_training_examples"] is True
    split = manifest["split_audit_summary"]
    assert split["image_level_audit"] == {
        "exact_train_test_duplicate_pages": 1,
        "near_duplicate_train_validation_pages": 1,
        "page_images_required_to_repeat_discovery": True,
    }
