"""Load and summarize the compact public evaluation artifacts.

This module intentionally supports only the current public ledger and figures.
It does not require DocVQA records, raw generations, or private mappings.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

PUBLIC_SCHEMA_VERSION = 1
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
EXPECTED_MANUFACTURED_GROUPS = {
    None: 756,
    "string_absent": 209,
    "string_survives_retained": 26,
    "survivor_excluded": 15,
}
VALID_CITATION = "valid_citation"
HONEST_ABSTAIN = "honest_abstain"
FALSE_ABSTAIN = "false_abstain"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_public_manifest(public_dir: Path) -> dict[str, Any]:
    """Load the public manifest and validate its schema version."""

    path = public_dir.resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"public manifest not found: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("public_schema_version") != PUBLIC_SCHEMA_VERSION:
        raise ValueError("unsupported public artifact schema version")
    if not isinstance(manifest.get("artifacts"), dict):
        raise ValueError("public manifest is missing its artifact index")
    return manifest


def load_public_jsonl(
    public_dir: Path,
    relative_path: str,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load one manifest-addressed JSONL artifact after hash and size checks."""

    root = public_dir.resolve()
    public_manifest = manifest or load_public_manifest(root)
    entry = public_manifest["artifacts"].get(relative_path)
    if entry is None:
        raise ValueError(f"artifact is not declared in the public manifest: {relative_path}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact path escapes the public directory: {relative_path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"public artifact not found: {path}")
    if _sha256(path) != entry["sha256"]:
        raise ValueError(f"public artifact hash mismatch: {relative_path}")
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError(f"public artifact size mismatch: {relative_path}")
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if len(rows) != entry["row_count"]:
        raise ValueError(f"public artifact row-count mismatch: {relative_path}")
    return rows


def load_evaluation_ledger(
    public_dir: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load the 1,006-row public evaluation ledger."""

    rows = load_public_jsonl(
        public_dir,
        "evaluation/evaluation_ledger.jsonl",
        manifest=manifest,
    )
    populations = Counter(row.get("population") for row in rows)
    if populations != Counter(EXPECTED_POPULATIONS):
        raise ValueError(f"unexpected evaluation-ledger populations: {populations}")
    manufactured_groups = Counter(row.get("manufactured_group") for row in rows)
    if manufactured_groups != Counter(EXPECTED_MANUFACTURED_GROUPS):
        raise ValueError(
            f"unexpected evaluation-ledger manufactured groups: {manufactured_groups}"
        )
    if [row.get("row_id") for row in rows] != sorted(row.get("row_id") for row in rows):
        raise ValueError("evaluation ledger must be sorted by opaque row_id")
    if len({row.get("row_id") for row in rows}) != len(rows):
        raise ValueError("evaluation ledger contains duplicate opaque row IDs")
    for row in rows:
        missing = set(SYSTEM_IDS) - set(row)
        if missing:
            raise ValueError(f"ledger row {row.get('row_id')} is missing systems: {sorted(missing)}")
        population = row["population"]
        manufactured_group = row["manufactured_group"]
        if population.startswith("natural_") and manufactured_group is not None:
            raise ValueError(f"natural row {row['row_id']} has a manufactured group")
        if population == "manufactured_clean" and manufactured_group not in {
            "string_absent",
            "string_survives_retained",
        }:
            raise ValueError(f"clean manufactured row {row['row_id']} has an invalid group")
        if population == "manufactured_excluded" and manufactured_group != "survivor_excluded":
            raise ValueError(f"excluded manufactured row {row['row_id']} has an invalid group")
    return rows


def exact_binomial_two_sided(improved: int, declined: int) -> float:
    """Return the exact two-sided binomial probability for discordant pairs."""

    total_discordant = improved + declined
    if total_discordant == 0:
        return 1.0
    weights = [comb(total_discordant, count) for count in range(total_discordant + 1)]
    observed_weight = weights[improved]
    return min(1.0, sum(weight for weight in weights if weight <= observed_weight) / sum(weights))


def conditional_citation_label_metrics(
    rows: list[dict[str, Any]], system_id: str
) -> dict[str, Any]:
    """Aggregate exact conditional citation-label counts and eligible coverage."""

    outcomes = [row[system_id] for row in rows]
    eligible = [outcome for outcome in outcomes if outcome["citation_eligible"]]
    true_positive = sum(outcome["citation_true_positive_count"] for outcome in eligible)
    predicted = sum(outcome["citation_prediction_count"] for outcome in eligible)
    labeled = sum(outcome["citation_label_count"] for outcome in eligible)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / labeled if labeled else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "bucket_counts": dict(sorted(Counter(outcome["bucket"] for outcome in outcomes).items())),
        "citation_eligible_count": len(eligible),
        "citation_true_positive_count": true_positive,
        "citation_prediction_count": predicted,
        "citation_label_count": labeled,
        "citation_micro_f1": round(micro_f1, 4),
    }


def aggregate_final_system_metrics(
    ledger: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Aggregate the seven selected systems' final metrics from exact row outcomes."""

    answerable = [row for row in ledger if row["population"] == "natural_menu_answerable"]
    natural_miss = [row for row in ledger if row["population"] == "natural_ocr_miss"]
    manufactured = [row for row in ledger if row["population"] == "manufactured_clean"]
    metrics: dict[str, dict[str, Any]] = {}
    for system_id in SYSTEM_IDS:
        citation = conditional_citation_label_metrics(answerable, system_id)
        metrics[system_id] = {
            "clean_manufactured_honest_refusal": {
                "count": sum(row[system_id]["bucket"] == HONEST_ABSTAIN for row in manufactured),
                "denominator": len(manufactured),
            },
            "natural_ocr_miss_honest_refusal": {
                "count": sum(row[system_id]["bucket"] == HONEST_ABSTAIN for row in natural_miss),
                "denominator": len(natural_miss),
            },
            "natural_answerable_false_refusal": {
                "count": sum(row[system_id]["bucket"] == FALSE_ABSTAIN for row in answerable),
                "denominator": len(answerable),
            },
            "conditional_citation_label_f1": {
                "micro_f1": citation["citation_micro_f1"],
                "eligible_count": citation["citation_eligible_count"],
                "eligible_denominator": len(answerable),
                "true_positive_count": citation["citation_true_positive_count"],
                "prediction_count": citation["citation_prediction_count"],
                "label_count": citation["citation_label_count"],
            },
        }
    return metrics


def paired_comparison(
    ledger: list[dict[str, Any]],
    *,
    before: str,
    after: str,
    population: str,
    endpoint_bucket: str,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    """Compute exact paired transitions for one population and outcome bucket."""

    if before not in SYSTEM_IDS or after not in SYSTEM_IDS:
        raise ValueError(f"unknown system comparison: {before} -> {after}")
    if population == "manufactured_all":
        rows = [row for row in ledger if row["population"].startswith("manufactured_")]
    else:
        rows = [row for row in ledger if row["population"] == population]
    if not rows:
        raise ValueError(f"no ledger rows for population: {population}")
    improved = declined = both_positive = both_negative = 0
    for row in rows:
        before_positive = row[before]["bucket"] == endpoint_bucket
        after_positive = row[after]["bucket"] == endpoint_bucket
        if before_positive and after_positive:
            both_positive += 1
        elif not before_positive and not after_positive:
            both_negative += 1
        elif lower_is_better:
            improved += int(before_positive and not after_positive)
            declined += int(not before_positive and after_positive)
        else:
            improved += int(not before_positive and after_positive)
            declined += int(before_positive and not after_positive)
    return {
        "improved": improved,
        "declined": declined,
        "both_positive": both_positive,
        "both_negative": both_negative,
        "discordant": improved + declined,
        "exact_two_sided_p": exact_binomial_two_sided(improved, declined),
        "population": population,
        "denominator": len(rows),
        "before": before,
        "after": after,
        "endpoint_bucket": endpoint_bucket,
        "lower_is_better": lower_is_better,
    }


def compute_document_cluster_sensitivity(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Reproduce the seed-to-run-2 clean-235 document-cluster sensitivity."""

    clean = [row for row in ledger if row["population"] == "manufactured_clean"]
    changes_by_document: dict[str, list[int]] = defaultdict(list)
    for row in clean:
        changes_by_document[row["document_cluster_id"]].append(
            int(row["run2"]["bucket"] == HONEST_ABSTAIN)
            - int(row["sft_seed"]["bucket"] == HONEST_ABSTAIN)
        )
    document_ids = sorted(changes_by_document)
    document_delta = np.asarray(
        [sum(changes_by_document[document_id]) for document_id in document_ids], dtype=int
    )
    document_size = np.asarray(
        [len(changes_by_document[document_id]) for document_id in document_ids], dtype=int
    )
    rng = np.random.default_rng(42)
    bootstrap_parts: list[np.ndarray] = []
    for _ in range(20):
        sampled = rng.integers(0, len(document_ids), size=(10_000, len(document_ids)))
        bootstrap_parts.append(
            document_delta[sampled].sum(axis=1) / document_size[sampled].sum(axis=1)
        )
    lower, upper = np.quantile(np.concatenate(bootstrap_parts), [0.025, 0.975])
    improved = int((document_delta > 0).sum())
    declined = int((document_delta < 0).sum())
    tied = int((document_delta == 0).sum())
    return {
        "comparison": "sft_seed_to_run2_clean_honest_refusal",
        "row_count": len(clean),
        "document_count": len(document_ids),
        "improved_documents": improved,
        "declined_documents": declined,
        "tied_documents": tied,
        "exact_two_sided_sign_test_p": exact_binomial_two_sided(improved, declined),
        "bootstrap": {
            "seed": 42,
            "replicates": 200_000,
            "estimand": "row_weighted_honest_refusal_rate_difference",
            "resampling_unit": "document_cluster",
            "lower_95": round(float(lower), 3),
            "upper_95": round(float(upper), 3),
        },
    }


def compute_duplicate_excluded_sensitivity(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute run-2 false refusal after excluding the flagged duplicate cluster."""

    answerable = [row for row in ledger if row["population"] == "natural_menu_answerable"]
    excluded = [row for row in answerable if row["duplicate_sensitivity_exclude"]]
    retained = [row for row in answerable if not row["duplicate_sensitivity_exclude"]]
    if len({row["document_cluster_id"] for row in excluded}) != 1:
        raise ValueError("duplicate sensitivity rows must share one opaque document cluster")
    return {
        "system_id": "run2",
        "endpoint_bucket": FALSE_ABSTAIN,
        "full_count": sum(row["run2"]["bucket"] == FALSE_ABSTAIN for row in answerable),
        "full_denominator": len(answerable),
        "excluded_row_count": len(excluded),
        "excluded_document_cluster_count": 1,
        "filtered_count": sum(row["run2"]["bucket"] == FALSE_ABSTAIN for row in retained),
        "filtered_denominator": len(retained),
    }
