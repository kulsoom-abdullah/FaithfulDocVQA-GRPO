"""Validate the compact, text-free public evidence artifacts.

This validator reads only ``artifacts/public``. It checks schemas, hashes,
population sizes, headline metrics, paired comparisons, cluster sensitivity,
and operational-identifier hygiene without requiring the private DocVQA data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.reward import compute_grpo_reward  # noqa: E402
from scripts import public_metrics  # noqa: E402
from scripts.eval_scorer import (  # noqa: E402
    BUCKETS,
    FALSE_ABSTAIN,
    FALSE_GROUNDING,
    HONEST_ABSTAIN,
    VALID_CITATION,
    classify,
)

SCHEMA_VERSION = 1
SYSTEM_IDS = (
    "base",
    "sft_seed",
    "run1",
    "run2",
    "balanced_sft",
    "run4a",
    "run4b",
)
EXPECTED_POPULATIONS = {
    "natural_menu_answerable": 679,
    "natural_ocr_miss": 77,
    "manufactured_clean": 235,
    "manufactured_excluded": 15,
}
EXPECTED_ARTIFACTS = {
    "evaluation/evaluation_ledger.jsonl": 1_006,
    "examples/evaluator_cases.jsonl": 9,
    "training/sft_validation.jsonl": 13,
    "training/run4_reward_dynamics.jsonl": 300,
    "training/run4_outcome_dynamics.jsonl": 600,
    "training/run4_validation.jsonl": 30,
}
LEDGER_METADATA_KEYS = {
    "schema_version",
    "row_id",
    "document_cluster_id",
    "population",
    "included_in_primary",
    "exclusion_code",
    "duplicate_sensitivity_exclude",
    "manufactured_group",
}
EXPECTED_MANUFACTURED_GROUPS = {
    None: 756,
    "string_absent": 209,
    "string_survives_retained": 26,
    "survivor_excluded": 15,
}
OUTCOME_KEYS = {
    "bucket",
    "citation_eligible",
    "citation_true_positive_count",
    "citation_prediction_count",
    "citation_label_count",
    "found_answer_correct",
    "parse_failure_code",
}
FORBIDDEN_LEDGER_KEYS = {
    "record_id",
    "doc_id",
    "question",
    "answer",
    "gold_answers",
    "target_answer",
    "ocr_menu",
    "ocr_text",
    "evidence_ids",
    "cited_ids",
    "bbox",
    "image_path",
    "source",
    "source_path",
    "raw",
    "raw_output",
    "completion",
    "provider",
    "request_id",
    "call_uuid",
    "uuid",
}
ROW_ID_RE = re.compile(r"^row_[0-9a-f]{24}$")
DOCUMENT_ID_RE = re.compile(r"^doc_[0-9a-f]{24}$")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")
OPERATIONAL_PATTERNS = {
    "uuid": UUID_RE,
    "email": EMAIL_RE,
    "phone": PHONE_RE,
    "absolute_user_path": re.compile(r"(?:/Users/|/home/|/root/|/workspace/)", re.I),
    "provider_identifier": re.compile(r"\b(?:runpod|modal|request[_-]?id|call[_-]?uuid)\b", re.I),
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_binomial_two_sided(improved: int, declined: int) -> float:
    """Return the exact two-sided binomial probability for discordant pairs."""

    return public_metrics.exact_binomial_two_sided(improved, declined)


def compute_headline_metrics(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recompute the seven final-system headline summaries from the public ledger."""

    return public_metrics.aggregate_final_system_metrics(ledger)


PAIR_SPECS = (
    ("sft_seed_to_run1_clean", "manufactured_clean", "sft_seed", "run1", HONEST_ABSTAIN, False),
    ("sft_seed_to_run2_clean", "manufactured_clean", "sft_seed", "run2", HONEST_ABSTAIN, False),
    (
        "sft_seed_to_balanced_clean",
        "manufactured_clean",
        "sft_seed",
        "balanced_sft",
        HONEST_ABSTAIN,
        False,
    ),
    ("run2_to_balanced_clean", "manufactured_clean", "run2", "balanced_sft", HONEST_ABSTAIN, False),
    (
        "balanced_to_run4a_clean",
        "manufactured_clean",
        "balanced_sft",
        "run4a",
        HONEST_ABSTAIN,
        False,
    ),
    (
        "balanced_to_run4b_clean",
        "manufactured_clean",
        "balanced_sft",
        "run4b",
        HONEST_ABSTAIN,
        False,
    ),
    ("run4a_to_run4b_clean", "manufactured_clean", "run4a", "run4b", HONEST_ABSTAIN, False),
    (
        "run4a_to_run4b_natural_ocr_miss",
        "natural_ocr_miss",
        "run4a",
        "run4b",
        HONEST_ABSTAIN,
        False,
    ),
    (
        "run4a_to_run4b_false_refusal",
        "natural_menu_answerable",
        "run4a",
        "run4b",
        FALSE_ABSTAIN,
        True,
    ),
)
SECONDARY_PAIR_SPECS = (
    (
        "run1_to_run2_full_manufactured",
        "manufactured_all",
        "run1",
        "run2",
        HONEST_ABSTAIN,
        False,
    ),
)


def compute_paired_comparisons(ledger: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Recompute the named paired comparisons stored in the manifest."""

    results: dict[str, dict[str, Any]] = {}
    for name, population, before, after, bucket, lower_is_better in PAIR_SPECS:
        results[name] = public_metrics.paired_comparison(
            ledger,
            before=before,
            after=after,
            population=population,
            endpoint_bucket=bucket,
            lower_is_better=lower_is_better,
        )
    return results


def compute_secondary_comparisons(
    ledger: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Recompute non-headline comparisons retained for context."""

    results: dict[str, dict[str, Any]] = {}
    for name, population, before, after, bucket, lower_is_better in SECONDARY_PAIR_SPECS:
        comparison = public_metrics.paired_comparison(
            ledger,
            before=before,
            after=after,
            population=population,
            endpoint_bucket=bucket,
            lower_is_better=lower_is_better,
        )
        comparison.update(
            {
                "focused_review_exclusions_included": 15,
                "used_in_public_notebook": False,
            }
        )
        results[name] = comparison
    return results


def compute_cluster_sensitivity(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the clean-235 document-cluster sensitivity from opaque IDs."""

    return public_metrics.compute_document_cluster_sensitivity(ledger)


def compute_duplicate_sensitivity(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute run-2 false refusal with the identified duplicate cluster excluded."""

    return public_metrics.compute_duplicate_excluded_sensitivity(ledger)


def _assert_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise ValueError(f"{label}: {actual!r} != {expected!r}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{label}: dictionary keys differ")
        for key, value in expected.items():
            _assert_equal(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label}: list shape differs")
        for index, value in enumerate(expected):
            _assert_equal(actual[index], value, f"{label}[{index}]")
    elif actual != expected:
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def _validate_ledger(ledger: list[dict[str, Any]]) -> None:
    if len(ledger) != 1_006:
        raise ValueError(f"evaluation ledger must contain 1,006 rows, got {len(ledger)}")
    if [row["row_id"] for row in ledger] != sorted(row["row_id"] for row in ledger):
        raise ValueError("evaluation ledger must be sorted by opaque row_id")
    if len({row["row_id"] for row in ledger}) != len(ledger):
        raise ValueError("evaluation ledger contains duplicate opaque row IDs")
    population_counts = Counter(row["population"] for row in ledger)
    if population_counts != Counter(EXPECTED_POPULATIONS):
        raise ValueError(f"unexpected ledger populations: {population_counts}")
    manufactured_group_counts = Counter(row["manufactured_group"] for row in ledger)
    if manufactured_group_counts != Counter(EXPECTED_MANUFACTURED_GROUPS):
        raise ValueError(f"unexpected manufactured groups: {manufactured_group_counts}")

    for row in ledger:
        expected_keys = LEDGER_METADATA_KEYS | set(SYSTEM_IDS)
        if set(row) != expected_keys:
            raise ValueError(f"ledger row {row.get('row_id')} has unexpected keys")
        if FORBIDDEN_LEDGER_KEYS & set(row):
            raise ValueError(f"forbidden top-level ledger key in {row['row_id']}")
        if row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"wrong ledger schema version in {row['row_id']}")
        if not ROW_ID_RE.fullmatch(row["row_id"]):
            raise ValueError(f"invalid opaque row ID: {row['row_id']!r}")
        if not DOCUMENT_ID_RE.fullmatch(row["document_cluster_id"]):
            raise ValueError(f"invalid opaque document ID: {row['document_cluster_id']!r}")
        if row["population"] == "manufactured_excluded":
            if row["included_in_primary"] or row["exclusion_code"] != "focused_survivor_review":
                raise ValueError(f"invalid excluded manufactured metadata in {row['row_id']}")
        elif not row["included_in_primary"] or row["exclusion_code"] is not None:
            raise ValueError(f"invalid primary-inclusion metadata in {row['row_id']}")
        if row["duplicate_sensitivity_exclude"] and row["population"] != "natural_menu_answerable":
            raise ValueError(f"duplicate sensitivity flag on wrong population in {row['row_id']}")
        manufactured_group = row["manufactured_group"]
        if row["population"].startswith("natural_") and manufactured_group is not None:
            raise ValueError(f"natural row has manufactured group in {row['row_id']}")
        if row["population"] == "manufactured_clean" and manufactured_group not in {
            "string_absent",
            "string_survives_retained",
        }:
            raise ValueError(f"clean manufactured row has invalid group in {row['row_id']}")
        if (
            row["population"] == "manufactured_excluded"
            and manufactured_group != "survivor_excluded"
        ):
            raise ValueError(f"excluded manufactured row has invalid group in {row['row_id']}")

        for system_id in SYSTEM_IDS:
            outcome = row[system_id]
            if not isinstance(outcome, dict) or set(outcome) != OUTCOME_KEYS:
                raise ValueError(f"invalid outcome schema for {system_id} in {row['row_id']}")
            if FORBIDDEN_LEDGER_KEYS & set(outcome):
                raise ValueError(f"forbidden outcome key for {system_id} in {row['row_id']}")
            if outcome["bucket"] not in BUCKETS:
                raise ValueError(f"unknown bucket for {system_id} in {row['row_id']}")
            eligible = outcome["citation_eligible"]
            if eligible is not (outcome["bucket"] == VALID_CITATION):
                raise ValueError(f"citation eligibility disagrees with bucket in {row['row_id']}")
            count_fields = (
                "citation_true_positive_count",
                "citation_prediction_count",
                "citation_label_count",
            )
            counts = [outcome[field] for field in count_fields]
            if not all(isinstance(value, int) and value >= 0 for value in counts):
                raise ValueError(f"invalid citation counts for {system_id} in {row['row_id']}")
            if not eligible and any(counts):
                raise ValueError(f"ineligible outcome contains citation counts in {row['row_id']}")
            if outcome["found_answer_correct"] not in (True, False, None):
                raise ValueError(f"invalid found-answer correctness in {row['row_id']}")
            if outcome["parse_failure_code"] is not None and not isinstance(
                outcome["parse_failure_code"], str
            ):
                raise ValueError(f"invalid parse-failure code in {row['row_id']}")

    duplicate_rows = [row for row in ledger if row["duplicate_sensitivity_exclude"]]
    if len(duplicate_rows) != 4:
        raise ValueError(f"expected four duplicate-sensitivity rows, got {len(duplicate_rows)}")

    run2_group_counts = {
        group: Counter(
            row["run2"]["bucket"]
            for row in ledger
            if row["manufactured_group"] == group
        )
        for group in ("string_absent", "string_survives_retained", "survivor_excluded")
    }
    expected_run2_group_counts = {
        "string_absent": Counter({HONEST_ABSTAIN: 189, FALSE_GROUNDING: 20}),
        "string_survives_retained": Counter({HONEST_ABSTAIN: 15, FALSE_GROUNDING: 11}),
        "survivor_excluded": Counter({HONEST_ABSTAIN: 2, FALSE_GROUNDING: 13}),
    }
    if run2_group_counts != expected_run2_group_counts:
        raise ValueError(f"unexpected run-2 manufactured-group outcomes: {run2_group_counts}")


def _validate_training_tables(public_dir: Path) -> None:
    sft = _read_jsonl(public_dir / "training/sft_validation.jsonl")
    expected_sft = {
        "original_sft": [214, 428, 642, 856, 1070, 1284],
        "balanced_sft": [324, 648, 972, 1296, 1620, 1944, 1947],
    }
    for system_id, checkpoints in expected_sft.items():
        rows = [row for row in sft if row["system_id"] == system_id]
        if [row["checkpoint"] for row in rows] != checkpoints:
            raise ValueError(f"unexpected SFT checkpoint set for {system_id}")
        if sum(bool(row["selected"]) for row in rows) != 1:
            raise ValueError(f"missing selected SFT checkpoint for {system_id}")
        for row in rows:
            if row["abstention_denominator"] != 46 or row["answerable_denominator"] != 707:
                raise ValueError(f"wrong SFT validation denominator for {system_id}")

    reward = _read_jsonl(public_dir / "training/run4_reward_dynamics.jsonl")
    for run in ("run4a", "run4b"):
        rows = [row for row in reward if row["system_id"] == run]
        if [row["optimizer_step"] for row in rows] != list(range(1, 151)):
            raise ValueError(f"unexpected reward steps for {run}")
        if any(row["completion_count"] != 32 for row in rows):
            raise ValueError(f"wrong completion count in reward dynamics for {run}")

    composition = _read_jsonl(public_dir / "training/run4_outcome_dynamics.jsonl")
    for run in ("run4a", "run4b"):
        for target_type in ("abstention_target", "answerable"):
            rows = [
                row
                for row in composition
                if row["system_id"] == run and row["target_type"] == target_type
            ]
            if [row["optimizer_step"] for row in rows] != list(range(1, 151)):
                raise ValueError(f"unexpected outcome steps for {run}/{target_type}")
            for row in rows:
                if sum(row["bucket_counts"].values()) != row["completion_count"]:
                    raise ValueError(f"outcome counts do not partition {run} step {row['optimizer_step']}")
        by_step = defaultdict(int)
        for row in composition:
            if row["system_id"] == run:
                by_step[row["optimizer_step"]] += row["completion_count"]
        if by_step != {step: 32 for step in range(1, 151)}:
            raise ValueError(f"run-4 outcome rows do not total 32 completions per step for {run}")

    validation = _read_jsonl(public_dir / "training/run4_validation.jsonl")
    for run, selected in (("run4a", 100), ("run4b", 150)):
        rows = [row for row in validation if row["system_id"] == run]
        if [row["checkpoint"] for row in rows] != list(range(10, 151, 10)):
            raise ValueError(f"unexpected run-4 validation checkpoint set for {run}")
        if [row["checkpoint"] for row in rows if row["selected"]] != [selected]:
            raise ValueError(f"wrong selected run-4 checkpoint for {run}")
        if any(
            row["total_denominator"] != 112
            or row["abstention_denominator"] != 64
            or row["answerable_denominator"] != 48
            for row in rows
        ):
            raise ValueError(f"wrong run-4 validation denominators for {run}")


def _validate_manifest_summaries(
    manifest: dict[str, Any],
    public_dir: Path,
    headline_metrics: dict[str, dict[str, Any]],
) -> None:
    """Validate compact curator attestations and their public-data cross-checks."""

    citation_summary = {
        "evidence_status": "curator_validated_aggregate_attestation",
        "definition": "answerable rows with at least two citable whole-token accepted-answer sites",
        "all_answerable_rows": {
            "affected": 533,
            "denominator": 4_496,
            "rate": 0.1185,
            "display_percent": 11.9,
        },
        "answerable_test_rows": {
            "affected": 80,
            "denominator": 679,
            "rate": 0.1178,
            "display_percent": 11.8,
        },
        "counterfactual_f1_deltas": {
            "sft_seed": 0.0148,
            "run1": 0.0117,
            "run2": 0.0183,
            "balanced_sft": 0.0133,
        },
        "system_ordering_unchanged": True,
    }
    _assert_equal(
        manifest.get("citation_label_ambiguity_summary"),
        citation_summary,
        "citation_label_ambiguity_summary",
    )

    sft_rows = _read_jsonl(public_dir / "training/sft_validation.jsonl")
    sft_by_key = {(row["system_id"], row["checkpoint"]): row for row in sft_rows}
    original_selected = sft_by_key[("original_sft", 642)]
    balanced_648 = sft_by_key[("balanced_sft", 648)]
    balanced_1296 = sft_by_key[("balanced_sft", 1296)]
    checkpoint_summary = {
        "evidence_status": "curator_validated_aggregate_attestation",
        "original_sft": {
            "validation_checkpoints": [214, 428, 642, 856, 1070, 1284],
            "selected_checkpoint": 642,
            "selection_basis": "honest_refusal_argmax_on_46_validation_ocr_miss_rows",
            "selected_validation": {
                "honest_refusal_count": original_selected["honest_refusal_count"],
                "honest_refusal_denominator": original_selected["abstention_denominator"],
                "false_refusal_count": original_selected["false_refusal_count"],
                "false_refusal_denominator": original_selected["answerable_denominator"],
                "conditional_citation_label_f1": original_selected["citation_micro_f1"],
                "citation_eligible_count": original_selected["citation_eligible_count"],
            },
            "selected_test": {
                "natural_ocr_miss_honest_refusal_count": headline_metrics["sft_seed"][
                    "natural_ocr_miss_honest_refusal"
                ]["count"],
                "natural_ocr_miss_denominator": 77,
                "false_refusal_count": headline_metrics["sft_seed"][
                    "natural_answerable_false_refusal"
                ]["count"],
                "answerable_denominator": 679,
                "conditional_citation_label_f1": headline_metrics["sft_seed"][
                    "conditional_citation_label_f1"
                ]["micro_f1"],
                "citation_eligible_count": headline_metrics["sft_seed"][
                    "conditional_citation_label_f1"
                ]["eligible_count"],
            },
            "test_generation_artifact_count": 6,
            "test_artifacts_existed_before_grpo_seed_was_fixed": True,
            "validation_curve_characterization": "noisy_and_non_monotonic",
            "selected_checkpoint_had_highest_false_refusal_in_series": True,
            "documented_forward_estimate": {"lower": 0.65, "upper": 0.68},
        },
        "balanced_sft": {
            "primary_metric": "honest_refusal_on_46_validation_ocr_miss_rows",
            "tied_checkpoints": [648, 1296],
            "honest_refusal_count": 44,
            "honest_refusal_denominator": 46,
            "checkpoint_648": {
                "false_refusal_count": balanced_648["false_refusal_count"],
                "malformed_count": balanced_648["answerable_bucket_counts"].get("malformed", 0),
                "answerable_non_valid_output_count": (
                    balanced_648["answerable_denominator"]
                    - balanced_648["citation_eligible_count"]
                ),
                "evaluation_loss": 0.092343,
            },
            "checkpoint_1296": {
                "false_refusal_count": balanced_1296["false_refusal_count"],
                "malformed_count": balanced_1296["answerable_bucket_counts"].get("malformed", 0),
                "answerable_non_valid_output_count": (
                    balanced_1296["answerable_denominator"]
                    - balanced_1296["citation_eligible_count"]
                ),
                "evaluation_loss": 0.095555,
            },
            "selected_checkpoint": 1296,
            "selection_basis": "secondary_tie_break_on_fewer_non_valid_answerable_outputs",
        },
    }
    _assert_equal(
        manifest.get("checkpoint_selection_summary"),
        checkpoint_summary,
        "checkpoint_selection_summary",
    )

    training_design_summary = {
        "evidence_status": "curator_validated_aggregate_attestation",
        "original_sft": {
            "natural_training_rows": 3_423,
            "answerable_rows": 3_110,
            "natural_ocr_miss_rows": 313,
            "manufactured_abstention_rows": 0,
        },
        "balanced_sft": {
            "natural_training_rows": 3_423,
            "manufactured_abstention_rows": 1_764,
            "total_training_rows": 5_187,
        },
        "run4": {
            "training_pool_rows": 4_410,
            "abstention_target_rows": 1_764,
            "answerable_rows": 2_646,
            "optimizer_steps_per_variant": 150,
            "examples_per_optimizer_step": 4,
            "sampled_completions_per_example": 8,
            "sampled_completions_per_optimizer_step": 32,
            "same_ordered_training_examples": True,
        },
    }
    _assert_equal(
        manifest.get("training_design_summary"),
        training_design_summary,
        "training_design_summary",
    )

    split_audit_summary = {
        "evidence_status": "curator_validated_aggregate_attestation",
        "assignment_unit": "doc_id",
        "same_doc_id_cross_split": False,
        "image_level_audit": {
            "exact_train_test_duplicate_pages": 1,
            "near_duplicate_train_validation_pages": 1,
            "page_images_required_to_repeat_discovery": True,
        },
        "exact_train_test_duplicate": {
            "answerable_test_rows": 4,
            "clean_manufactured_rows": 0,
        },
        "run2_duplicate_excluded_false_refusal": {
            "full_count": 120,
            "full_denominator": 679,
            "filtered_count": 119,
            "filtered_denominator": 675,
        },
    }
    _assert_equal(
        manifest.get("split_audit_summary"),
        split_audit_summary,
        "split_audit_summary",
    )


def _validate_examples(public_dir: Path) -> None:
    rows = _read_jsonl(public_dir / "examples/evaluator_cases.jsonl")
    expected_cases = {
        "valid_citation",
        "honest_abstention",
        "false_grounding",
        "false_refusal",
        "fabrication",
        "supported_with_empty_evidence_contradiction",
        "unsupported_with_evidence_contradiction",
        "malformed_output",
        "repeated_string_lure",
    }
    if {row["case_id"] for row in rows} != expected_cases:
        raise ValueError("synthetic example set does not contain the nine required cases")
    for row in rows:
        outcome = classify(row["record"], row["raw_output"])
        if outcome["bucket"] != row["expected"]["bucket"]:
            raise ValueError(f"synthetic case bucket mismatch: {row['case_id']}")
        if (outcome["bucket"] == VALID_CITATION) != row["expected"]["citation_eligible"]:
            raise ValueError(f"synthetic case eligibility mismatch: {row['case_id']}")
        rewards = {
            "run1": compute_grpo_reward(row["record"], row["raw_output"], 1),
            "run2": compute_grpo_reward(row["record"], row["raw_output"], 2),
            "run4b": compute_grpo_reward(
                row["record"], row["raw_output"], 2, false_abstain_base=-0.2
            ),
        }
        _assert_equal(rewards, row["expected"]["reward"], f"example.{row['case_id']}.reward")


def _validate_operational_hygiene(public_dir: Path) -> dict[str, int]:
    counts = {name: 0 for name in OPERATIONAL_PATTERNS}
    for path in sorted(public_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for name, pattern in OPERATIONAL_PATTERNS.items():
            counts[name] += len(pattern.findall(text))
    if any(counts.values()):
        raise ValueError(f"operational identifier scan failed: {counts}")
    return counts


def validate_public_artifacts(public_dir: Path) -> dict[str, Any]:
    """Validate every public artifact and return the recomputed summaries."""

    public_dir = public_dir.resolve()
    manifest_path = public_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"public manifest not found: {manifest_path}")
    manifest = _read_json(manifest_path)
    if manifest.get("public_schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported public artifact schema version")
    if set(manifest.get("artifacts", {})) != set(EXPECTED_ARTIFACTS):
        raise ValueError("public manifest artifact set is incomplete or contains extras")

    for relative, expected_rows in EXPECTED_ARTIFACTS.items():
        path = public_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"public artifact not found: {path}")
        entry = manifest["artifacts"][relative]
        if entry["sha256"] != _sha256(path):
            raise ValueError(f"public artifact hash mismatch: {relative}")
        if entry["size_bytes"] != path.stat().st_size:
            raise ValueError(f"public artifact size mismatch: {relative}")
        if entry["row_count"] != expected_rows:
            raise ValueError(f"public manifest row-count mismatch: {relative}")
        if len(_read_jsonl(path)) != expected_rows:
            raise ValueError(f"public artifact physical row-count mismatch: {relative}")

    ledger = _read_jsonl(public_dir / "evaluation/evaluation_ledger.jsonl")
    _validate_ledger(ledger)
    _validate_training_tables(public_dir)
    _validate_examples(public_dir)

    headline_metrics = compute_headline_metrics(ledger)
    paired_comparisons = compute_paired_comparisons(ledger)
    secondary_comparisons = compute_secondary_comparisons(ledger)
    cluster_sensitivity = compute_cluster_sensitivity(ledger)
    duplicate_sensitivity = compute_duplicate_sensitivity(ledger)
    _assert_equal(
        headline_metrics,
        manifest["expected_headline_metrics"],
        "expected_headline_metrics",
    )
    _assert_equal(
        paired_comparisons,
        manifest["expected_paired_comparisons"],
        "expected_paired_comparisons",
    )
    _assert_equal(
        secondary_comparisons,
        manifest["secondary_comparisons"],
        "secondary_comparisons",
    )
    _assert_equal(
        cluster_sensitivity,
        manifest["expected_document_cluster_sensitivity"],
        "expected_document_cluster_sensitivity",
    )
    _assert_equal(
        duplicate_sensitivity,
        manifest["expected_duplicate_excluded_sensitivity"],
        "expected_duplicate_excluded_sensitivity",
    )
    if manifest.get("expected_populations") != EXPECTED_POPULATIONS:
        raise ValueError("manifest population expectations differ from the public schema")
    _validate_manifest_summaries(manifest, public_dir, headline_metrics)
    if manifest.get("private_validation", {}).get("source_text_match_counts") != {
        "questions": 0,
        "accepted_answers": 0,
        "ocr_text": 0,
        "raw_completions": 0,
    }:
        raise ValueError("private source-text validation attestation is absent or nonzero")
    operational_scan = _validate_operational_hygiene(public_dir)
    return {
        "headline_metrics": headline_metrics,
        "paired_comparisons": paired_comparisons,
        "secondary_comparisons": secondary_comparisons,
        "document_cluster_sensitivity": cluster_sensitivity,
        "duplicate_excluded_sensitivity": duplicate_sensitivity,
        "operational_scan": operational_scan,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-dir",
        type=Path,
        default=ROOT / "artifacts/public",
        help="Public artifact directory (default: artifacts/public)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_public_artifacts(args.public_dir)
    print("PASS public artifacts")
    print(f"  systems: {len(result['headline_metrics'])}")
    print(f"  paired comparisons: {len(result['paired_comparisons'])}")
    print(f"  secondary comparisons: {len(result['secondary_comparisons'])}")
    print(f"  operational scan: {result['operational_scan']}")


if __name__ == "__main__":
    main()
