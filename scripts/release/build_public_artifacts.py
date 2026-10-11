"""Build compact public artifacts from the retained private evaluation evidence.

The script is curator-facing. It reads an explicit, fixed set of private files,
uses the canonical scorer for every per-record outcome, writes no source text or
source identifiers, and validates the public result before returning success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.reward import (  # noqa: E402
    FALSE_ABSTAIN_BASE,
    FALSE_GROUNDING_RUN2,
    HONEST_ABSTAIN_RUN2,
    compute_grpo_reward,
)
from scripts.eval_scorer import (  # noqa: E402
    BUCKETS,
    FALSE_ABSTAIN,
    HONEST_ABSTAIN,
    VALID_CITATION,
    classify,
)
from scripts.build_manufactured_abstains import answer_recurs  # noqa: E402
from scripts.validate_public_artifacts import (  # noqa: E402
    SCHEMA_VERSION,
    SYSTEM_IDS,
    compute_cluster_sensitivity,
    compute_duplicate_sensitivity,
    compute_headline_metrics,
    compute_paired_comparisons,
    compute_secondary_comparisons,
    validate_public_artifacts,
)

ORIGINAL_SFT_STEPS = (214, 428, 642, 856, 1070, 1284)
BALANCED_SFT_STEPS = (324, 648, 972, 1296, 1620, 1944, 1947)
RUN4_VALIDATION_STEPS = tuple(range(10, 151, 10))
RUN4_SELECTED_STEPS = {"run4a": 100, "run4b": 150}
PUBLIC_ARTIFACT_PATHS = (
    "evaluation/evaluation_ledger.jsonl",
    "examples/evaluator_cases.jsonl",
    "training/sft_validation.jsonl",
    "training/run4_reward_dynamics.jsonl",
    "training/run4_outcome_dynamics.jsonl",
    "training/run4_validation.jsonl",
)

SYSTEM_SPECS = {
    "base": {
        "display_name": "Base Qwen3-VL",
        "checkpoint": "Qwen3-VL-4B-Instruct",
        "lineage": "base",
        "natural": "data/checkpoints/eval/sft_B_seed42/gen_base_test.json",
        "manufactured": "data/checkpoints/eval/sft_B_seed42/gen_base_manufactured.json",
        "expected_source": "base",
    },
    "sft_seed": {
        "display_name": "SFT seed",
        "checkpoint": "ck-642",
        "lineage": "original_sft",
        "natural": "data/checkpoints/eval/sft_B_seed42/gen_test_checkpoint-642.json",
        "manufactured": "data/checkpoints/eval/sft_B_seed42/gen_manufactured_checkpoint-642.json",
        "expected_source": "sft_v3_32g/sft_B_seed42/checkpoint-642",
    },
    "run1": {
        "display_name": "GRPO control",
        "checkpoint": "ck-150",
        "lineage": "original_sft_grpo",
        "natural": "data/checkpoints/eval/grpo_run1_seed42/gen_test_checkpoint-150.json",
        "manufactured": "data/checkpoints/eval/grpo_run1_seed42/gen_manufactured_checkpoint-150.json",
        "expected_source": "grpo_run1_seed42/checkpoint-150",
    },
    "run2": {
        "display_name": "Shaped GRPO",
        "checkpoint": "ck-150",
        "lineage": "original_sft_grpo",
        "natural": "data/checkpoints/eval/grpo_run2_seed42/gen_test_checkpoint-150.json",
        "manufactured": "data/checkpoints/eval/grpo_run2_seed42/gen_manufactured_checkpoint-150.json",
        "expected_source": "grpo_run2_seed42/checkpoint-150",
    },
    "balanced_sft": {
        "display_name": "Balanced SFT",
        "checkpoint": "ck-1296",
        "lineage": "balanced_sft",
        "natural": "data/checkpoints/eval/sft_v3_balanced_nodecoy/gen_test_checkpoint-1296.json",
        "manufactured": (
            "data/checkpoints/eval/sft_v3_balanced_nodecoy/"
            "gen_manufactured_checkpoint-1296.json"
        ),
        "expected_source": "sft_v3_balanced_nodecoy/sft_B_seed42/checkpoint-1296",
    },
    "run4a": {
        "display_name": "Run-4a GRPO",
        "checkpoint": "ck-100",
        "lineage": "balanced_sft_grpo",
        "natural": "data/checkpoints/eval/grpo_run4a_seed42/gen_test_checkpoint-100.json",
        "manufactured": "data/checkpoints/eval/grpo_run4a_seed42/gen_manufactured_checkpoint-100.json",
        "expected_source": "grpo_run4a_seed42/checkpoint-100",
    },
    "run4b": {
        "display_name": "Run-4b GRPO",
        "checkpoint": "ck-150",
        "lineage": "balanced_sft_grpo",
        "natural": "data/checkpoints/eval/grpo_run4b_seed42/gen_test_checkpoint-150.json",
        "manufactured": "data/checkpoints/eval/grpo_run4b_seed42/gen_manufactured_checkpoint-150.json",
        "expected_source": "grpo_run4b_seed42/checkpoint-150",
    },
}

EXPECTED_HEADLINES = {
    "base": (4, 8, 14, 0.6771, 638),
    "sft_seed": (68, 62, 26, 0.8412, 650),
    "run1": (62, 59, 23, 0.8296, 655),
    "run2": (204, 77, 120, 0.8694, 557),
    "balanced_sft": (212, 71, 78, 0.8758, 601),
    "run4a": (217, 76, 99, 0.8895, 580),
    "run4b": (217, 74, 91, 0.8775, 588),
}
EXPECTED_PAIR_DIRECTIONS = {
    "sft_seed_to_run1_clean": (1, 7),
    "sft_seed_to_run2_clean": (136, 0),
    "sft_seed_to_balanced_clean": (144, 0),
    "run2_to_balanced_clean": (14, 6),
    "balanced_to_run4a_clean": (5, 0),
    "balanced_to_run4b_clean": (6, 1),
    "run4a_to_run4b_clean": (1, 1),
    "run4a_to_run4b_natural_ocr_miss": (0, 2),
    "run4a_to_run4b_false_refusal": (10, 2),
}
EXPECTED_SECONDARY_PAIR_DIRECTIONS = {
    "run1_to_run2_full_manufactured": (144, 0),
}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_checksum_manifest(repo_root: Path) -> dict[str, str]:
    path = repo_root / "docs/analysis/ARTIFACTS.sha256"
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        digest, relative = line.split(maxsplit=1)
        if relative in entries:
            raise ValueError(f"duplicate checksum entry at {path}:{line_number}")
        entries[relative] = digest
    return entries


def _verify_manifest_covered_inputs(repo_root: Path, paths: Sequence[Path]) -> int:
    checksum_entries = _load_checksum_manifest(repo_root)
    verified = 0
    for path in paths:
        relative = path.relative_to(repo_root).as_posix()
        expected = checksum_entries.get(relative)
        if expected is not None:
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(f"private input checksum mismatch: {relative}")
            verified += 1
    return verified


def _assert_exact_named_files(
    directory: Path,
    prefix: str,
    expected_names: set[str],
    *,
    allowed_unrelated_names: set[str] | None = None,
) -> None:
    allowed_unrelated_names = allowed_unrelated_names or set()
    observed = {path.name for path in directory.iterdir() if path.is_file() and path.name.startswith(prefix)}
    if not expected_names <= observed or not observed <= expected_names | allowed_unrelated_names:
        missing = sorted(expected_names - observed)
        extra = sorted(observed - expected_names - allowed_unrelated_names)
        raise ValueError(f"unexpected files in {directory}: missing={missing}, extra={extra}")


def _load_generation(
    path: Path,
    records: Sequence[dict[str, Any]],
    *,
    expected_split: str,
    expected_source: str,
) -> dict[str, str | None]:
    payload = _read_json(path)
    rows = payload.get("records")
    if not isinstance(rows, list):
        raise ValueError(f"generation artifact has no records: {path}")
    if payload.get("split") != expected_split:
        raise ValueError(f"wrong generation split in {path}")
    if payload.get("n") != len(rows) or len(rows) != len(records):
        raise ValueError(f"wrong generation row count in {path}")
    source = str(payload.get("source") or payload.get("checkpoint") or "")
    if expected_source == "base":
        if source != "base":
            raise ValueError(f"wrong base source in {path}: {source!r}")
    elif expected_source not in source:
        raise ValueError(f"wrong checkpoint source in {path}: {source!r}")
    observed_ids = [str(row["record_id"]) for row in rows]
    expected_ids = [str(record["record_id"]) for record in records]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError(f"duplicate generation record IDs in {path}")
    if set(observed_ids) != set(expected_ids):
        raise ValueError(f"generation join is incomplete in {path}")
    return {str(row["record_id"]): row.get("raw") for row in rows}


def _citation_counts(outcomes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    eligible = [outcome for outcome in outcomes if outcome["bucket"] == VALID_CITATION]
    true_positive = sum(
        len(set(outcome["cited_ids"]) & set(outcome["gold_cited_ids"])) for outcome in eligible
    )
    predicted = sum(len(set(outcome["cited_ids"])) for outcome in eligible)
    labeled = sum(len(set(outcome["gold_cited_ids"])) for outcome in eligible)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / labeled if labeled else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "citation_eligible_count": len(eligible),
        "citation_true_positive_count": true_positive,
        "citation_prediction_count": predicted,
        "citation_label_count": labeled,
        "citation_micro_f1": round(micro_f1, 4),
    }


def _public_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    eligible = outcome["bucket"] == VALID_CITATION
    return {
        "bucket": outcome["bucket"],
        "citation_eligible": eligible,
        "citation_true_positive_count": (
            len(set(outcome["cited_ids"]) & set(outcome["gold_cited_ids"])) if eligible else 0
        ),
        "citation_prediction_count": len(set(outcome["cited_ids"])) if eligible else 0,
        "citation_label_count": len(set(outcome["gold_cited_ids"])) if eligible else 0,
        "found_answer_correct": outcome["found_answer_correct"],
        "parse_failure_code": outcome["parse_failure"],
    }


def _private_input_paths(repo_root: Path) -> list[Path]:
    relative_paths: list[str] = [
        "data/master.jsonl",
        "data/manufactured_test_abstains.jsonl",
        "data/manufactured_val_abstains.jsonl",
        "data/manufactured_train_abstains_nodecoy.jsonl",
        "data/grpo_val_instrument_v2_recursfalse.jsonl",
        "docs/run_configs/sft_v3_32g_training_config.json",
        "docs/run_configs/sft_v3_balanced_nodecoy_training_config.json",
        "data/checkpoints/grpo_run4a_seed42/grpo_run_config.json",
        "data/checkpoints/grpo_run4b_seed42/grpo_run_config.json",
        "data/checkpoints/grpo_run4a_seed42/reward_log.jsonl",
        "data/checkpoints/grpo_run4b_seed42/reward_log.jsonl",
        "data/checkpoints/grpo_run4a_seed42/reward_members.jsonl",
        "data/checkpoints/grpo_run4b_seed42/reward_members.jsonl",
        "scripts/eval_scorer.py",
        "scripts/faithful_io.py",
        "phase0/evidence_reward.py",
    ]
    for specification in SYSTEM_SPECS.values():
        relative_paths.extend([specification["natural"], specification["manufactured"]])
    relative_paths.extend(
        f"data/checkpoints/eval/sft_B_seed42/gen_checkpoint-{step}.json"
        for step in ORIGINAL_SFT_STEPS
    )
    relative_paths.extend(
        f"data/checkpoints/eval/sft_v3_balanced_nodecoy/gen_checkpoint-{step}.json"
        for step in BALANCED_SFT_STEPS
    )
    for run in ("run4a", "run4b"):
        relative_paths.extend(
            f"data/checkpoints/grpo_{run}_seed42/eval_step{step}.jsonl"
            for step in RUN4_VALIDATION_STEPS
        )
    paths = [repo_root / relative for relative in dict.fromkeys(relative_paths)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing private inputs: {missing}")
    return paths


def _write_private_input_manifest(
    repo_root: Path,
    private_dir: Path,
    input_paths: Sequence[Path],
    manifest_covered_count: int,
) -> None:
    value = {
        "schema_version": 1,
        "repository_head": _git_head(repo_root),
        "manifest_covered_input_count": manifest_covered_count,
        "inputs": [
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(input_paths)
        ],
    }
    _write_json(private_dir / "private_input_manifest.json", value)


def _random_public_id(prefix: str, used: set[str]) -> str:
    while True:
        candidate = f"{prefix}_{secrets.token_hex(12)}"
        if candidate not in used:
            used.add(candidate)
            return candidate


def _load_or_create_mapping(
    mapping_path: Path,
    natural_records: Sequence[dict[str, Any]],
    manufactured_records: Sequence[dict[str, Any]],
    *,
    create: bool,
) -> dict[str, Any]:
    row_keys = {
        *(f"natural:{record['record_id']}" for record in natural_records),
        *(f"manufactured:{record['record_id']}" for record in manufactured_records),
    }
    document_keys = {
        *(str(record["doc_id"]) for record in natural_records),
        *(str(record["doc_id"]) for record in manufactured_records),
    }
    if mapping_path.exists():
        if create:
            raise FileExistsError(f"refusing to overwrite existing opaque mapping: {mapping_path}")
        mapping = _read_json(mapping_path)
    else:
        if not create:
            raise FileNotFoundError(
                f"opaque mapping not found: {mapping_path}; use --create-mapping once"
            )
        used: set[str] = set()
        mapping = {
            "schema_version": 1,
            "row_ids": {
                key: _random_public_id("row", used) for key in sorted(row_keys)
            },
            "document_ids": {
                key: _random_public_id("doc", used) for key in sorted(document_keys)
            },
        }
        _write_json(mapping_path, mapping)
    if mapping.get("schema_version") != 1:
        raise ValueError("unsupported opaque mapping schema")
    if set(mapping.get("row_ids", {})) != row_keys:
        raise ValueError("opaque row mapping has missing or extra private records")
    if set(mapping.get("document_ids", {})) != document_keys:
        raise ValueError("opaque document mapping has missing or extra private documents")
    public_ids = list(mapping["row_ids"].values()) + list(mapping["document_ids"].values())
    if len(public_ids) != len(set(public_ids)):
        raise ValueError("opaque mapping contains a public-ID collision")
    return mapping


def _build_evaluation_ledger(
    repo_root: Path,
    mapping: dict[str, Any],
    master: Sequence[dict[str, Any]],
    manufactured: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    natural = [record for record in master if record["split"] == "test"]
    if len(natural) != 756 or Counter(bool(row["target_is_abstain"]) for row in natural) != {
        False: 679,
        True: 77,
    }:
        raise ValueError("natural test population is not exactly 679 answerable + 77 OCR-miss")
    if len(manufactured) != 250:
        raise ValueError("manufactured test instrument must contain 250 rows")
    if sum(row.get("exclusion_reason") is None for row in manufactured) != 235:
        raise ValueError("manufactured instrument must contain 235 primary + 15 excluded rows")

    raw_by_system: dict[str, dict[str, dict[str, str | None]]] = {}
    for system_id, specification in SYSTEM_SPECS.items():
        raw_by_system[system_id] = {
            "natural": _load_generation(
                repo_root / specification["natural"],
                natural,
                expected_split="test",
                expected_source=specification["expected_source"],
            ),
            "manufactured": _load_generation(
                repo_root / specification["manufactured"],
                manufactured,
                expected_split="manufactured",
                expected_source=specification["expected_source"],
            ),
        }

    rows: list[dict[str, Any]] = []
    for population_name, records, source_kind in (
        ("natural", natural, "natural"),
        ("manufactured", manufactured, "manufactured"),
    ):
        for record in records:
            if population_name == "natural":
                population = (
                    "natural_ocr_miss" if record["target_is_abstain"] else "natural_menu_answerable"
                )
                exclusion_code = None
                included = True
                manufactured_group = None
            else:
                included = record.get("exclusion_reason") is None
                population = "manufactured_clean" if included else "manufactured_excluded"
                exclusion_code = None if included else "focused_survivor_review"
                if not included:
                    manufactured_group = "survivor_excluded"
                elif answer_recurs(record["gold_answers"], record["ocr_menu"]):
                    manufactured_group = "string_survives_retained"
                else:
                    manufactured_group = "string_absent"
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "row_id": mapping["row_ids"][f"{population_name}:{record['record_id']}"],
                "document_cluster_id": mapping["document_ids"][str(record["doc_id"])],
                "population": population,
                "included_in_primary": included,
                "exclusion_code": exclusion_code,
                "duplicate_sensitivity_exclude": (
                    population == "natural_menu_answerable" and str(record["doc_id"]) == "4386"
                ),
                "manufactured_group": manufactured_group,
            }
            for system_id in SYSTEM_IDS:
                raw = raw_by_system[system_id][source_kind][str(record["record_id"])]
                row[system_id] = _public_outcome(classify(record, raw))
            rows.append(row)
    rows.sort(key=lambda row: row["row_id"])
    if len(rows) != 1_006:
        raise ValueError(f"evaluation ledger must contain 1,006 rows, got {len(rows)}")
    expected_groups = {
        None: 756,
        "string_absent": 209,
        "string_survives_retained": 26,
        "survivor_excluded": 15,
    }
    if Counter(row["manufactured_group"] for row in rows) != Counter(expected_groups):
        raise ValueError("manufactured-group counts do not match the frozen instrument")
    return rows


def _build_sft_validation(repo_root: Path, master: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    validation = [record for record in master if record["split"] == "validation"]
    if len(validation) != 753 or Counter(bool(row["target_is_abstain"]) for row in validation) != {
        False: 707,
        True: 46,
    }:
        raise ValueError("SFT validation population must be 707 answerable + 46 OCR-miss")
    train_count = sum(record["split"] == "train" for record in master)
    if train_count != 3_423:
        raise ValueError("master training population must contain 3,423 rows")
    manufactured_train = _read_jsonl(repo_root / "data/manufactured_train_abstains_nodecoy.jsonl")
    if len(manufactured_train) != 1_764:
        raise ValueError("balanced-SFT manufactured training population must contain 1,764 rows")

    ignored_balanced_names = {f"gen_checkpoint-{step}.json" for step in BALANCED_SFT_STEPS}
    _assert_exact_named_files(
        repo_root / "data/checkpoints/eval/sft_B_seed42",
        "gen_checkpoint-",
        {f"gen_checkpoint-{step}.json" for step in ORIGINAL_SFT_STEPS},
        allowed_unrelated_names=ignored_balanced_names,
    )
    for name in ignored_balanced_names:
        ignored_copy = repo_root / "data/checkpoints/eval/sft_B_seed42" / name
        if ignored_copy.is_file():
            payload = _read_json(ignored_copy)
            source = str(payload.get("source") or payload.get("checkpoint") or "")
            if "sft_v3_balanced_nodecoy" not in source:
                raise ValueError(f"unexpected unrelated checkpoint artifact: {ignored_copy}")
    _assert_exact_named_files(
        repo_root / "data/checkpoints/eval/sft_v3_balanced_nodecoy",
        "gen_checkpoint-",
        {f"gen_checkpoint-{step}.json" for step in BALANCED_SFT_STEPS},
    )
    configs = {
        "original_sft": _read_json(repo_root / "docs/run_configs/sft_v3_32g_training_config.json"),
        "balanced_sft": _read_json(
            repo_root / "docs/run_configs/sft_v3_balanced_nodecoy_training_config.json"
        ),
    }
    for system_id, config in configs.items():
        if config.get("per_device_train_batch_size") != 1:
            raise ValueError(f"unexpected SFT batch size for {system_id}")
        if config.get("gradient_accumulation_steps") != 8 or config.get("num_train_epochs") != 3.0:
            raise ValueError(f"unexpected SFT geometry for {system_id}")
    steps_per_epoch = {
        "original_sft": math.ceil(train_count / 8),
        "balanced_sft": math.ceil((train_count + len(manufactured_train)) / 8),
    }
    if steps_per_epoch != {"original_sft": 428, "balanced_sft": 649}:
        raise ValueError(f"unexpected SFT steps per epoch: {steps_per_epoch}")

    specifications = {
        "original_sft": {
            "steps": ORIGINAL_SFT_STEPS,
            "directory": repo_root / "data/checkpoints/eval/sft_B_seed42",
            "source": "sft_v3_32g/sft_B_seed42/checkpoint-",
            "selected": 642,
        },
        "balanced_sft": {
            "steps": BALANCED_SFT_STEPS,
            "directory": repo_root / "data/checkpoints/eval/sft_v3_balanced_nodecoy",
            "source": "sft_v3_balanced_nodecoy/sft_B_seed42/checkpoint-",
            "selected": 1296,
        },
    }
    rows: list[dict[str, Any]] = []
    for system_id, specification in specifications.items():
        for step in specification["steps"]:
            raw_by_id = _load_generation(
                specification["directory"] / f"gen_checkpoint-{step}.json",
                validation,
                expected_split="validation",
                expected_source=f"{specification['source']}{step}",
            )
            outcomes = [classify(record, raw_by_id[str(record["record_id"])]) for record in validation]
            answerable = [outcome for outcome in outcomes if outcome["gold_class"] == "answerable"]
            abstention = [outcome for outcome in outcomes if outcome["gold_class"] == "abstain"]
            citation = _citation_counts(answerable)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "system_id": system_id,
                    "checkpoint": step,
                    "normalized_epoch": step / steps_per_epoch[system_id],
                    "selected": step == specification["selected"],
                    "abstention_denominator": len(abstention),
                    "honest_refusal_count": sum(
                        outcome["bucket"] == HONEST_ABSTAIN for outcome in abstention
                    ),
                    "answerable_denominator": len(answerable),
                    "false_refusal_count": sum(
                        outcome["bucket"] == FALSE_ABSTAIN for outcome in answerable
                    ),
                    **citation,
                    "abstention_bucket_counts": dict(
                        sorted(Counter(outcome["bucket"] for outcome in abstention).items())
                    ),
                    "answerable_bucket_counts": dict(
                        sorted(Counter(outcome["bucket"] for outcome in answerable).items())
                    ),
                }
            )
    return rows


def _load_run4_reward_logs(repo_root: Path) -> dict[str, list[dict[str, Any]]]:
    configs = {
        run: _read_json(repo_root / f"data/checkpoints/grpo_{run}_seed42/grpo_run_config.json")
        for run in ("run4a", "run4b")
    }
    if configs["run4a"].get("run") != 2 or configs["run4b"].get("run") != 2:
        raise ValueError("run-4 configs must use canonical run=2 shaping")
    if "false_abstain_base" in configs["run4a"]:
        raise ValueError("run-4a must use the canonical false-refusal default")
    if configs["run4b"].get("false_abstain_base") != -0.2:
        raise ValueError("run-4b must use false_abstain_base=-0.2")
    if (FALSE_ABSTAIN_BASE, HONEST_ABSTAIN_RUN2, FALSE_GROUNDING_RUN2) != (0.0, 0.5, -0.6):
        raise ValueError("canonical reward constants no longer match the run-4 design")

    logs: dict[str, list[dict[str, Any]]] = {}
    for run in ("run4a", "run4b"):
        path = repo_root / f"data/checkpoints/grpo_{run}_seed42/reward_log.jsonl"
        rows = _read_jsonl(path)
        reward_rows = [row for row in rows if "reward" in row]
        if len(rows) != 151 or len(reward_rows) != 150:
            raise ValueError(f"expected 151 total rows and 150 reward rows in {path}")
        if [int(row["step"]) for row in reward_rows] != list(range(1, 151)):
            raise ValueError(f"reward steps are not exactly 1..150 in {path}")
        running_ema: float | None = None
        for row in reward_rows:
            running_ema = (
                float(row["reward"])
                if running_ema is None
                else 0.9 * running_ema + 0.1 * float(row["reward"])
            )
            if not math.isclose(float(row["reward_ema"]), round(running_ema, 4), abs_tol=1e-12):
                raise ValueError(f"reward EMA is not the alpha=0.1 recurrence in {path}")
        logs[run] = reward_rows
    return logs


def _build_run4_reward_dynamics(logs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "system_id": run,
            "optimizer_step": int(row["step"]),
            "completion_count": 32,
            "mean_reward": float(row["reward"]),
            "reward_ema_alpha_0_1": float(row["reward_ema"]),
            "selected": int(row["step"]) == RUN4_SELECTED_STEPS[run],
        }
        for run in ("run4a", "run4b")
        for row in logs[run]
    ]


def _build_run4_outcomes(
    repo_root: Path, logs: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    members: dict[str, list[dict[str, Any]]] = {}
    sequences: dict[str, list[tuple[Any, ...]]] = {}
    for run in ("run4a", "run4b"):
        path = repo_root / f"data/checkpoints/grpo_{run}_seed42/reward_members.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != 4_800:
            raise ValueError(f"expected 4,800 reward members in {path}")
        calls = Counter(int(row["call"]) for row in rows)
        if sorted(calls) != list(range(600)) or set(calls.values()) != {8}:
            raise ValueError(f"expected contiguous calls 0..599 with eight members in {path}")
        by_call: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_call[int(row["call"])].append(row)
            if row["bucket"] not in BUCKETS:
                raise ValueError(f"unknown reward-member bucket in {path}")
            if not isinstance(row["is_abstain"], bool):
                raise ValueError(f"non-boolean reward-member target label in {path}")
        if any(sorted(int(row["i"]) for row in call_rows) != list(range(8)) for call_rows in by_call.values()):
            raise ValueError(f"reward-member calls do not contain indices 0..7 in {path}")
        for row in rows:
            row["optimizer_step"] = int(row["call"]) // 4 + 1
        if Counter(int(row["optimizer_step"]) for row in rows) != Counter(
            {step: 32 for step in range(1, 151)}
        ):
            raise ValueError(f"reward members do not contain 32 completions per step in {path}")
        sequences[run] = [
            (
                int(row["call"]),
                int(row["i"]),
                str(row["record_id"]),
                row["is_abstain"],
                row.get("recurs"),
            )
            for row in rows
        ]
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_step[int(row["optimizer_step"])].append(row)
        for log_row in logs[run]:
            mean_reward = float(np.mean([float(row["reward"]) for row in by_step[int(log_row["step"])] ]))
            if not math.isclose(mean_reward, float(log_row["reward"]), abs_tol=6e-8):
                raise ValueError(f"reward-member mean does not reproduce {run} step {log_row['step']}")
        members[run] = rows
    if sequences["run4a"] != sequences["run4b"]:
        raise ValueError("run-4a and run-4b reward-member sequences are not identical")

    public_rows: list[dict[str, Any]] = []
    for run in ("run4a", "run4b"):
        by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in members[run]:
            by_step[int(row["optimizer_step"])].append(row)
        for step in range(1, 151):
            for target_type, is_abstain in (
                ("abstention_target", True),
                ("answerable", False),
            ):
                selected = [row for row in by_step[step] if row["is_abstain"] is is_abstain]
                counts = Counter(row["bucket"] for row in selected)
                public_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "system_id": run,
                        "optimizer_step": step,
                        "target_type": target_type,
                        "completion_count": len(selected),
                        "bucket_counts": {bucket: counts.get(bucket, 0) for bucket in BUCKETS},
                    }
                )
    return public_rows


def _build_run4_validation(
    repo_root: Path,
    master: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    instrument = _read_jsonl(repo_root / "data/grpo_val_instrument_v2_recursfalse.jsonl")
    if len(instrument) != 112 or len({str(row["record_id"]) for row in instrument}) != 112:
        raise ValueError("run-4 validation instrument must contain 112 unique rows")
    if Counter(row["kind"] for row in instrument) != {"abstain": 64, "answerable": 48}:
        raise ValueError("run-4 validation instrument must contain 64 abstention + 48 answerable")
    expected_sources = {"abstain": "manufactured_val_abstains", "answerable": "master_validation"}
    if any(row["source"] != expected_sources[row["kind"]] for row in instrument):
        raise ValueError("run-4 validation source labels disagree with the frozen design")
    master_by_id = {str(row["record_id"]): row for row in master}
    manufactured_by_id = {
        str(row["record_id"]): row
        for row in _read_jsonl(repo_root / "data/manufactured_val_abstains.jsonl")
    }
    records: list[dict[str, Any]] = []
    for item in instrument:
        is_abstain = item["kind"] == "abstain"
        source = manufactured_by_id if is_abstain else master_by_id
        record_id = str(item["record_id"])
        if record_id not in source:
            raise ValueError(f"validation instrument record missing from source: {record_id}")
        record = source[record_id]
        if bool(record["target_is_abstain"]) != is_abstain:
            raise ValueError(f"validation instrument label mismatch: {record_id}")
        if is_abstain and record.get("answer_recurs_in_menu") is not False:
            raise ValueError(f"validation abstention is not recurs-False: {record_id}")
        records.append(record)

    public_rows: list[dict[str, Any]] = []
    instrument_ids = [str(row["record_id"]) for row in instrument]
    for run in ("run4a", "run4b"):
        directory = repo_root / f"data/checkpoints/grpo_{run}_seed42"
        _assert_exact_named_files(
            directory,
            "eval_step",
            {f"eval_step{step}.jsonl" for step in RUN4_VALIDATION_STEPS},
        )
        for step in RUN4_VALIDATION_STEPS:
            path = directory / f"eval_step{step}.jsonl"
            saved_rows = _read_jsonl(path)
            if len(saved_rows) != 112 or [str(row["record_id"]) for row in saved_rows] != instrument_ids:
                raise ValueError(f"run-4 validation join is not 112/112 in instrument order: {path}")
            raw_by_id = {str(row["record_id"]): row["raw"] for row in saved_rows}
            if len(raw_by_id) != 112:
                raise ValueError(f"duplicate validation IDs in {path}")
            outcomes: list[dict[str, Any]] = []
            for record, saved in zip(records, saved_rows, strict=True):
                outcome = classify(record, raw_by_id[str(record["record_id"])])
                if outcome["bucket"] != saved["bucket"]:
                    raise ValueError(f"canonical scorer disagrees with saved bucket in {path}")
                outcomes.append(outcome)
            answerable = [outcome for outcome in outcomes if outcome["gold_class"] == "answerable"]
            abstention = [outcome for outcome in outcomes if outcome["gold_class"] == "abstain"]
            citation = _citation_counts(answerable)
            public_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "system_id": run,
                    "checkpoint": step,
                    "selected": step == RUN4_SELECTED_STEPS[run],
                    "total_denominator": len(outcomes),
                    "abstention_denominator": len(abstention),
                    "honest_refusal_count": sum(
                        outcome["bucket"] == HONEST_ABSTAIN for outcome in abstention
                    ),
                    "answerable_denominator": len(answerable),
                    "false_refusal_count": sum(
                        outcome["bucket"] == FALSE_ABSTAIN for outcome in answerable
                    ),
                    **citation,
                    "abstention_bucket_counts": dict(
                        sorted(Counter(outcome["bucket"] for outcome in abstention).items())
                    ),
                    "answerable_bucket_counts": dict(
                        sorted(Counter(outcome["bucket"] for outcome in answerable).items())
                    ),
                }
            )
    selected = {
        (row["system_id"], row["checkpoint"]): (row["honest_refusal_count"], row["false_refusal_count"])
        for row in public_rows
        if row["selected"]
    }
    if selected != {("run4a", 100): (60, 2), ("run4b", 150): (60, 2)}:
        raise ValueError(f"unexpected run-4 selected-checkpoint validation counts: {selected}")
    return public_rows


def _synthetic_record(
    case_id: str,
    *,
    menu_answerable: bool,
    gold_answers: list[str],
    cited_ids: list[int],
    menu: list[tuple[int, str, list[int]]],
) -> dict[str, Any]:
    return {
        "record_id": f"synthetic_{case_id}",
        "doc_id": f"synthetic_doc_{case_id}",
        "doc_type": ["fictional_fixture"],
        "target_is_abstain": not menu_answerable,
        "gold_localized": menu_answerable,
        "cited_ids": cited_ids,
        "gold_answers": gold_answers,
        "ocr_menu": [
            {"id": token_id, "text": text, "bbox": bbox}
            for token_id, text, bbox in menu
        ],
    }


def _build_synthetic_examples() -> list[dict[str, Any]]:
    definitions = [
        (
            "valid_citation",
            _synthetic_record(
                "valid_citation",
                menu_answerable=True,
                gold_answers=["Velora-Quartz-731"],
                cited_ids=[11],
                menu=[(11, "Velora-Quartz-731", [10, 20, 80, 34]), (12, "Fictitious total", [10, 40, 90, 54])],
            ),
            '{"found_answer":"Velora-Quartz-731","answerable_from_menu":true,"evidence_ids":[11]}',
        ),
        (
            "honest_abstention",
            _synthetic_record(
                "honest_abstention",
                menu_answerable=False,
                gold_answers=["Nimbrel-Archive-482"],
                cited_ids=[],
                menu=[(21, "Invented archive status", [5, 5, 120, 25])],
            ),
            '{"found_answer":"Nimbrel-Archive-482","answerable_from_menu":false,"evidence_ids":[]}',
        ),
        (
            "false_grounding",
            _synthetic_record(
                "false_grounding",
                menu_answerable=False,
                gold_answers=["Orivex-Signal-904"],
                cited_ids=[],
                menu=[(31, "Unrelated fictional footer", [0, 90, 120, 110])],
            ),
            '{"found_answer":"Orivex-Signal-904","answerable_from_menu":true,"evidence_ids":[31]}',
        ),
        (
            "false_refusal",
            _synthetic_record(
                "false_refusal",
                menu_answerable=True,
                gold_answers=["Tesselon-Delta-618"],
                cited_ids=[41],
                menu=[(41, "Tesselon-Delta-618", [12, 12, 112, 30])],
            ),
            '{"found_answer":"Tesselon-Delta-618","answerable_from_menu":false,"evidence_ids":[]}',
        ),
        (
            "fabrication",
            _synthetic_record(
                "fabrication",
                menu_answerable=True,
                gold_answers=["Kavora-Index-257"],
                cited_ids=[51],
                menu=[(51, "Kavora-Index-257", [4, 8, 104, 26])],
            ),
            '{"found_answer":"Kavora-Index-257","answerable_from_menu":true,"evidence_ids":[999]}',
        ),
        (
            "supported_with_empty_evidence_contradiction",
            _synthetic_record(
                "supported_with_empty_evidence_contradiction",
                menu_answerable=True,
                gold_answers=["Fendrix-Parcel-346"],
                cited_ids=[61],
                menu=[(61, "Fendrix-Parcel-346", [7, 7, 107, 27])],
            ),
            '{"found_answer":"Fendrix-Parcel-346","answerable_from_menu":true,"evidence_ids":[]}',
        ),
        (
            "unsupported_with_evidence_contradiction",
            _synthetic_record(
                "unsupported_with_evidence_contradiction",
                menu_answerable=False,
                gold_answers=["Solvane-Route-815"],
                cited_ids=[],
                menu=[(71, "Fictional routing note", [3, 60, 123, 80])],
            ),
            '{"found_answer":"Solvane-Route-815","answerable_from_menu":false,"evidence_ids":[71]}',
        ),
        (
            "malformed_output",
            _synthetic_record(
                "malformed_output",
                menu_answerable=True,
                gold_answers=["Praxilon-Code-573"],
                cited_ids=[81],
                menu=[(81, "Praxilon-Code-573", [9, 9, 109, 29])],
            ),
            "fictional response without a JSON object",
        ),
        (
            "repeated_string_lure",
            _synthetic_record(
                "repeated_string_lure",
                menu_answerable=True,
                gold_answers=["Zoramel-Token-692"],
                cited_ids=[91],
                menu=[
                    (91, "Zoramel-Token-692", [2, 2, 102, 22]),
                    (92, "Zoramel-Token-692", [2, 52, 102, 72]),
                ],
            ),
            '{"found_answer":"Zoramel-Token-692","answerable_from_menu":true,"evidence_ids":[92]}',
        ),
    ]
    rows: list[dict[str, Any]] = []
    for case_id, record, raw_output in definitions:
        outcome = classify(record, raw_output)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "record": record,
                "raw_output": raw_output,
                "expected": {
                    "bucket": outcome["bucket"],
                    "citation_eligible": outcome["bucket"] == VALID_CITATION,
                    "reward": {
                        "run1": compute_grpo_reward(record, raw_output, 1),
                        "run2": compute_grpo_reward(record, raw_output, 2),
                        "run4b": compute_grpo_reward(
                            record, raw_output, 2, false_abstain_base=-0.2
                        ),
                    },
                },
            }
        )
    return rows


def _all_string_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _all_string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_string_values(child)


def _normalized_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _source_text_scan(
    master: Sequence[dict[str, Any]],
    manufactured: Sequence[dict[str, Any]],
    private_input_paths: Sequence[Path],
    public_dir: Path,
) -> dict[str, int]:
    source_records = list(master) + list(manufactured)
    private_sets = {
        "questions": {
            _normalized_text(str(record["question"]))
            for record in source_records
            if record.get("question")
        },
        "accepted_answers": {
            _normalized_text(str(answer))
            for record in source_records
            for answer in record.get("gold_answers", [])
            if answer
        },
        "ocr_text": {
            _normalized_text(str(token["text"]))
            for record in source_records
            for token in record.get("ocr_menu", [])
            if token.get("text")
        },
        "raw_completions": set(),
    }
    for path in private_input_paths:
        relative = path.relative_to(ROOT).as_posix()
        if "/eval/" in relative and path.suffix == ".json":
            payload = _read_json(path)
            for row in payload.get("records", []):
                raw = row.get("raw")
                if isinstance(raw, str) and raw:
                    private_sets["raw_completions"].add(_normalized_text(raw))
        elif "eval_step" in path.name and path.suffix == ".jsonl":
            for row in _read_jsonl(path):
                raw = row.get("raw")
                if isinstance(raw, str) and raw:
                    private_sets["raw_completions"].add(_normalized_text(raw))

    public_strings: set[str] = set()
    for relative in PUBLIC_ARTIFACT_PATHS:
        for row in _read_jsonl(public_dir / relative):
            public_strings.update(
                normalized
                for value in _all_string_values(row)
                if (normalized := _normalized_text(value))
            )
    counts = {
        name: len(public_strings & private_values)
        for name, private_values in private_sets.items()
    }
    if any(counts.values()):
        raise ValueError(f"source-text scan found private values in public artifacts: {counts}")
    return counts


def _citation_label_ambiguity_summary() -> dict[str, Any]:
    """Return the curator-validated aggregate multiple-site sensitivity summary."""

    return {
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


def _checkpoint_selection_summary(
    sft_validation: Sequence[dict[str, Any]],
    headline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build compact checkpoint-selection metadata from public counts plus attestations."""

    by_key = {(row["system_id"], row["checkpoint"]): row for row in sft_validation}
    original_selected = by_key[("original_sft", 642)]
    balanced_648 = by_key[("balanced_sft", 648)]
    balanced_1296 = by_key[("balanced_sft", 1296)]
    if (
        original_selected["honest_refusal_count"],
        original_selected["false_refusal_count"],
        balanced_648["honest_refusal_count"],
        balanced_1296["honest_refusal_count"],
        balanced_648["false_refusal_count"],
        balanced_1296["false_refusal_count"],
    ) != (39, 12, 44, 44, 79, 75):
        raise ValueError("checkpoint-selection counts differ from the validated summaries")
    return {
        "evidence_status": "curator_validated_aggregate_attestation",
        "original_sft": {
            "validation_checkpoints": list(ORIGINAL_SFT_STEPS),
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
                "natural_ocr_miss_honest_refusal_count": headline["sft_seed"][
                    "natural_ocr_miss_honest_refusal"
                ]["count"],
                "natural_ocr_miss_denominator": 77,
                "false_refusal_count": headline["sft_seed"][
                    "natural_answerable_false_refusal"
                ]["count"],
                "answerable_denominator": 679,
                "conditional_citation_label_f1": headline["sft_seed"][
                    "conditional_citation_label_f1"
                ]["micro_f1"],
                "citation_eligible_count": headline["sft_seed"][
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
                "malformed_count": balanced_1296["answerable_bucket_counts"].get(
                    "malformed", 0
                ),
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


def _training_design_summary(
    repo_root: Path,
    master: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build the compact curator-facing training-design attestation."""

    train = [row for row in master if row["split"] == "train"]
    train_counts = Counter(bool(row["target_is_abstain"]) for row in train)
    manufactured_count = len(
        _read_jsonl(repo_root / "data/manufactured_train_abstains_nodecoy.jsonl")
    )
    configs = {
        run: _read_json(repo_root / f"data/checkpoints/grpo_{run}_seed42/grpo_run_config.json")
        for run in ("run4a", "run4b")
    }
    for run, config in configs.items():
        actual = (
            config["pool_abstain"],
            config["pool_answerable"],
            config["max_steps"],
            config["per_device_train_batch_size"],
            config["num_generations"],
        )
        if actual != (1_764, 2_646, 150, 4, 8):
            raise ValueError(f"unexpected run-4 training design for {run}: {actual}")
    if (len(train), train_counts[False], train_counts[True], manufactured_count) != (
        3_423,
        3_110,
        313,
        1_764,
    ):
        raise ValueError("SFT training-design counts differ from the frozen inputs")
    return {
        "evidence_status": "curator_validated_aggregate_attestation",
        "original_sft": {
            "natural_training_rows": len(train),
            "answerable_rows": train_counts[False],
            "natural_ocr_miss_rows": train_counts[True],
            "manufactured_abstention_rows": 0,
        },
        "balanced_sft": {
            "natural_training_rows": len(train),
            "manufactured_abstention_rows": manufactured_count,
            "total_training_rows": len(train) + manufactured_count,
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


def _split_audit_summary(duplicate: dict[str, Any]) -> dict[str, Any]:
    """Return the compact curator-validated split and duplicate attestation."""

    return {
        "evidence_status": "curator_validated_aggregate_attestation",
        "assignment_unit": "doc_id",
        "same_doc_id_cross_split": False,
        "image_level_audit": {
            "exact_train_test_duplicate_pages": 1,
            "near_duplicate_train_validation_pages": 1,
            "page_images_required_to_repeat_discovery": True,
        },
        "exact_train_test_duplicate": {
            "answerable_test_rows": duplicate["excluded_row_count"],
            "clean_manufactured_rows": 0,
        },
        "run2_duplicate_excluded_false_refusal": {
            "full_count": duplicate["full_count"],
            "full_denominator": duplicate["full_denominator"],
            "filtered_count": duplicate["filtered_count"],
            "filtered_denominator": duplicate["filtered_denominator"],
        },
    }


def _assert_known_results(
    headline: dict[str, dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    secondary_pairs: dict[str, dict[str, Any]],
    cluster: dict[str, Any],
    duplicate: dict[str, Any],
    ledger: Sequence[dict[str, Any]],
) -> None:
    for system_id, expected in EXPECTED_HEADLINES.items():
        clean, natural, false_refusal, citation_f1, eligible = expected
        metrics = headline[system_id]
        actual = (
            metrics["clean_manufactured_honest_refusal"]["count"],
            metrics["natural_ocr_miss_honest_refusal"]["count"],
            metrics["natural_answerable_false_refusal"]["count"],
            metrics["conditional_citation_label_f1"]["micro_f1"],
            metrics["conditional_citation_label_f1"]["eligible_count"],
        )
        if actual != expected:
            raise ValueError(f"headline mismatch for {system_id}: {actual} != {expected}")
    for name, expected in EXPECTED_PAIR_DIRECTIONS.items():
        actual = (pairs[name]["improved"], pairs[name]["declined"])
        if actual != expected:
            raise ValueError(f"paired comparison mismatch for {name}: {actual} != {expected}")
    for name, expected in EXPECTED_SECONDARY_PAIR_DIRECTIONS.items():
        actual = (secondary_pairs[name]["improved"], secondary_pairs[name]["declined"])
        if actual != expected:
            raise ValueError(f"secondary comparison mismatch for {name}: {actual} != {expected}")
    run2_group_counts = {
        group: Counter(
            row["run2"]["bucket"] for row in ledger if row["manufactured_group"] == group
        )
        for group in ("string_absent", "string_survives_retained", "survivor_excluded")
    }
    expected_run2_group_counts = {
        "string_absent": Counter({HONEST_ABSTAIN: 189, "false_grounding": 20}),
        "string_survives_retained": Counter({HONEST_ABSTAIN: 15, "false_grounding": 11}),
        "survivor_excluded": Counter({HONEST_ABSTAIN: 2, "false_grounding": 13}),
    }
    if run2_group_counts != expected_run2_group_counts:
        raise ValueError(f"run-2 manufactured-group mismatch: {run2_group_counts}")
    expected_cluster = (235, 75, 60, 0, 15, 1.734723475976807e-18, 0.502, 0.653)
    actual_cluster = (
        cluster["row_count"],
        cluster["document_count"],
        cluster["improved_documents"],
        cluster["declined_documents"],
        cluster["tied_documents"],
        cluster["exact_two_sided_sign_test_p"],
        cluster["bootstrap"]["lower_95"],
        cluster["bootstrap"]["upper_95"],
    )
    if actual_cluster != expected_cluster:
        raise ValueError(f"document-cluster sensitivity mismatch: {actual_cluster}")
    expected_duplicate = (120, 679, 4, 119, 675)
    actual_duplicate = (
        duplicate["full_count"],
        duplicate["full_denominator"],
        duplicate["excluded_row_count"],
        duplicate["filtered_count"],
        duplicate["filtered_denominator"],
    )
    if actual_duplicate != expected_duplicate:
        raise ValueError(f"duplicate-excluded sensitivity mismatch: {actual_duplicate}")


def build_public_artifacts(
    repo_root: Path,
    public_dir: Path,
    private_dir: Path,
    *,
    create_mapping: bool,
) -> dict[str, Any]:
    """Build, attest, and validate the requested public artifacts."""

    repo_root = repo_root.resolve()
    public_dir = public_dir.resolve()
    private_dir = private_dir.resolve()
    private_dir.mkdir(parents=True, exist_ok=True)
    input_paths = _private_input_paths(repo_root)
    manifest_covered_count = _verify_manifest_covered_inputs(repo_root, input_paths)
    _write_private_input_manifest(repo_root, private_dir, input_paths, manifest_covered_count)

    master = _read_jsonl(repo_root / "data/master.jsonl")
    if len(master) != 4_932 or len({str(row["record_id"]) for row in master}) != 4_932:
        raise ValueError("master dataset must contain 4,932 unique records")
    manufactured = _read_jsonl(repo_root / "data/manufactured_test_abstains.jsonl")
    natural = [row for row in master if row["split"] == "test"]
    mapping_path = private_dir / "opaque_id_mapping.json"
    mapping = _load_or_create_mapping(
        mapping_path,
        natural,
        manufactured,
        create=create_mapping,
    )

    ledger = _build_evaluation_ledger(repo_root, mapping, master, manufactured)
    sft_validation = _build_sft_validation(repo_root, master)
    reward_logs = _load_run4_reward_logs(repo_root)
    reward_dynamics = _build_run4_reward_dynamics(reward_logs)
    outcome_dynamics = _build_run4_outcomes(repo_root, reward_logs)
    run4_validation = _build_run4_validation(repo_root, master)
    examples = _build_synthetic_examples()

    outputs = {
        "evaluation/evaluation_ledger.jsonl": ledger,
        "examples/evaluator_cases.jsonl": examples,
        "training/sft_validation.jsonl": sft_validation,
        "training/run4_reward_dynamics.jsonl": reward_dynamics,
        "training/run4_outcome_dynamics.jsonl": outcome_dynamics,
        "training/run4_validation.jsonl": run4_validation,
    }
    for relative, rows in outputs.items():
        _write_jsonl(public_dir / relative, rows)

    source_text_counts = _source_text_scan(
        master,
        manufactured,
        input_paths,
        public_dir,
    )
    headline = compute_headline_metrics(ledger)
    pairs = compute_paired_comparisons(ledger)
    secondary_pairs = compute_secondary_comparisons(ledger)
    cluster = compute_cluster_sensitivity(ledger)
    duplicate = compute_duplicate_sensitivity(ledger)
    _assert_known_results(headline, pairs, secondary_pairs, cluster, duplicate, ledger)

    manifest = {
        "public_schema_version": SCHEMA_VERSION,
        "purpose": (
            "Compact, text-free evidence for the seven evaluated systems, headline counts, "
            "paired comparisons, sensitivity analyses, and approved public figures."
        ),
        "private_evidence_statement": (
            "These artifacts were derived and validated against a separately retained private "
            "evidence archive containing the source-derived records and raw predictions."
        ),
        "canonical_scorer": {
            "module": "scripts.eval_scorer",
            "sha256": _sha256(repo_root / "scripts/eval_scorer.py"),
            "contract": "three_field_v3",
        },
        "systems": [
            {
                "system_id": system_id,
                "display_name": SYSTEM_SPECS[system_id]["display_name"],
                "checkpoint": SYSTEM_SPECS[system_id]["checkpoint"],
                "lineage": SYSTEM_SPECS[system_id]["lineage"],
            }
            for system_id in SYSTEM_IDS
        ],
        "expected_populations": {
            "natural_menu_answerable": 679,
            "natural_ocr_miss": 77,
            "manufactured_clean": 235,
            "manufactured_excluded": 15,
        },
        "expected_headline_metrics": headline,
        "expected_paired_comparisons": pairs,
        "secondary_comparisons": secondary_pairs,
        "expected_document_cluster_sensitivity": cluster,
        "expected_duplicate_excluded_sensitivity": duplicate,
        "citation_label_ambiguity_summary": _citation_label_ambiguity_summary(),
        "checkpoint_selection_summary": _checkpoint_selection_summary(
            sft_validation,
            headline,
        ),
        "training_design_summary": _training_design_summary(repo_root, master),
        "split_audit_summary": _split_audit_summary(duplicate),
        "expected_checkpoint_sets": {
            "original_sft_validation": list(ORIGINAL_SFT_STEPS),
            "balanced_sft_validation": list(BALANCED_SFT_STEPS),
            "run4a_validation": list(RUN4_VALIDATION_STEPS),
            "run4b_validation": list(RUN4_VALIDATION_STEPS),
            "selected": {
                "original_sft": 642,
                "balanced_sft": 1296,
                "run4a": 100,
                "run4b": 150,
            },
            "run4_selected_validation_counts": {
                "run4a": {"honest_refusal": 60, "abstention_denominator": 64, "false_refusal": 2, "answerable_denominator": 48},
                "run4b": {"honest_refusal": 60, "abstention_denominator": 64, "false_refusal": 2, "answerable_denominator": 48},
            },
        },
        "private_validation": {
            "source_text_match_counts": source_text_counts,
            "manifest_covered_input_count": manifest_covered_count,
            "opaque_mapping_retained_outside_repository": True,
        },
        "artifacts": {
            relative: {
                "sha256": _sha256(public_dir / relative),
                "size_bytes": (public_dir / relative).stat().st_size,
                "row_count": len(rows),
            }
            for relative, rows in outputs.items()
        },
    }
    _write_json(public_dir / "manifest.json", manifest)
    validation = validate_public_artifacts(public_dir)
    return {
        "manifest": manifest,
        "validation": validation,
        "private_mapping": mapping_path,
        "private_input_manifest": private_dir / "private_input_manifest.json",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--public-dir", type=Path, default=ROOT / "artifacts/public")
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument(
        "--create-mapping",
        action="store_true",
        help="Create the one-time opaque ID mapping; refuses to overwrite an existing mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_public_artifacts(
        args.repo_root,
        args.public_dir,
        args.private_dir,
        create_mapping=args.create_mapping,
    )
    print("PASS public artifact build")
    print(f"  public artifacts: {len(result['manifest']['artifacts'])}")
    print(f"  private mapping: {result['private_mapping']}")
    print(f"  private input manifest: {result['private_input_manifest']}")


if __name__ == "__main__":
    main()
