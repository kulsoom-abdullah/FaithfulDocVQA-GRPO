"""Score the matched-pool SFT continuation from committed generation artifacts.

This script is CPU-only. It uses the repository's canonical scorer and existing
prediction artifacts to compute the final four evaluation axes and paired
improvement/regression counts against the named comparison systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_scorer import FALSE_ABSTAIN, HONEST_ABSTAIN, score
from scripts.format_sft import load_split

RUN_ID = "sft_ck642_matched_grpo_pool_seed42"
DEFAULT_GEN_DIR = ROOT / "data" / "checkpoints" / "eval" / RUN_ID
MANUFACTURED = ROOT / "data" / "manufactured_test_abstains.jsonl"

COMPARISON_SYSTEMS = {
    "sft_seed": {
        "display_name": "SFT seed ck-642",
        "manufactured": ROOT / "data/checkpoints/eval/sft_B_seed42/gen_manufactured_checkpoint-642.json",
        "test": ROOT / "data/checkpoints/eval/sft_B_seed42/gen_test_checkpoint-642.json",
    },
    "run1": {
        "display_name": "run-1 ck-150",
        "manufactured": ROOT / "data/checkpoints/eval/grpo_run1_seed42/gen_manufactured_checkpoint-150.json",
        "test": ROOT / "data/checkpoints/eval/grpo_run1_seed42/gen_test_checkpoint-150.json",
    },
    "run2": {
        "display_name": "run-2 ck-150",
        "manufactured": ROOT / "data/checkpoints/eval/grpo_run2_seed42/gen_manufactured_checkpoint-150.json",
        "test": ROOT / "data/checkpoints/eval/grpo_run2_seed42/gen_test_checkpoint-150.json",
    },
    "balanced_sft": {
        "display_name": "balanced SFT ck-1296",
        "manufactured": (
            ROOT
            / "data/checkpoints/eval/sft_v3_balanced_nodecoy/gen_manufactured_checkpoint-1296.json"
        ),
        "test": ROOT / "data/checkpoints/eval/sft_v3_balanced_nodecoy/gen_test_checkpoint-1296.json",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_generation(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    rows = payload.get("records", [])
    by_id = {str(row["record_id"]): row["raw"] for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate record_id in generation file: {path}")
    return by_id


def exact_two_sided_binomial(b: int, c: int) -> float | None:
    """Exact two-sided binomial probability for discordant paired counts."""
    n = b + c
    if n == 0:
        return None
    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _outcome_map(records: list[dict[str, Any]], raw: dict[str, str], bucket: str) -> dict[str, bool]:
    result = score(records, raw)
    return {row["record_id"]: row["bucket"] == bucket for row in result["per_record"]}


def paired_counts(
    records: list[dict[str, Any]],
    first_raw: dict[str, str],
    second_raw: dict[str, str],
    bucket: str,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    first = _outcome_map(records, first_raw, bucket)
    second = _outcome_map(records, second_raw, bucket)
    favor_second = favor_first = both_yes = both_no = 0
    for record in records:
        rid = record["record_id"]
        a = first[rid]
        b = second[rid]
        if a and b:
            both_yes += 1
        elif not a and not b:
            both_no += 1
        elif lower_is_better:
            if a and not b:
                favor_second += 1
            elif not a and b:
                favor_first += 1
        else:
            if not a and b:
                favor_second += 1
            elif a and not b:
                favor_first += 1
    return {
        "rows_favoring_matched_sft": favor_second,
        "rows_favoring_comparison": favor_first,
        "both_positive": both_yes,
        "both_negative": both_no,
        "discordant": favor_second + favor_first,
        "exact_two_sided_p": exact_two_sided_binomial(favor_second, favor_first),
    }


def metrics_from_generations(gen_dir: Path) -> dict[str, Any]:
    test_records = load_split("test", "B")
    answerable = [r for r in test_records if not r["target_is_abstain"]]
    natural_miss = [r for r in test_records if r["target_is_abstain"]]
    manufactured_all = read_jsonl(MANUFACTURED)
    clean_manufactured = [
        r for r in manufactured_all if r.get("exclusion_reason") is None
    ]
    if (len(clean_manufactured), len(natural_miss), len(answerable)) != (235, 77, 679):
        raise RuntimeError(
            "unexpected evaluation population sizes: "
            f"clean={len(clean_manufactured)} natural={len(natural_miss)} "
            f"answerable={len(answerable)}"
        )

    matched_test_path = gen_dir / "gen_test_checkpoint-150.json"
    matched_man_path = gen_dir / "gen_manufactured_checkpoint-150.json"
    matched_test = load_generation(matched_test_path)
    matched_man = load_generation(matched_man_path)
    test_score = score(test_records, matched_test)
    man_score = score(clean_manufactured, matched_man)

    clean_outcomes = man_score["per_record"]
    test_outcomes = test_score["per_record"]
    natural_ids = {r["record_id"] for r in natural_miss}
    answerable_ids = {r["record_id"] for r in answerable}

    clean_correct = sum(row["bucket"] == HONEST_ABSTAIN for row in clean_outcomes)
    natural_correct = sum(
        row["bucket"] == HONEST_ABSTAIN
        for row in test_outcomes
        if row["record_id"] in natural_ids
    )
    false_refusal = sum(
        row["bucket"] == FALSE_ABSTAIN
        for row in test_outcomes
        if row["record_id"] in answerable_ids
    )
    f1 = test_score["overall"]["citation_f1_gated"]

    comparison_raw = {
        key: {
            "manufactured": load_generation(paths["manufactured"]),
            "test": load_generation(paths["test"]),
        }
        for key, paths in COMPARISON_SYSTEMS.items()
    }
    paired = {}
    for key, paths in COMPARISON_SYSTEMS.items():
        paired[f"{key}_to_matched_sft_clean_manufactured"] = {
            "comparison": f"{paths['display_name']} → matched-pool SFT ck-150",
            "evaluation_outcome": "clean manufactured correct refusal",
            **paired_counts(
                clean_manufactured,
                comparison_raw[key]["manufactured"],
                matched_man,
                HONEST_ABSTAIN,
                lower_is_better=False,
            ),
        }
        paired[f"{key}_to_matched_sft_natural_ocr_miss"] = {
            "comparison": f"{paths['display_name']} → matched-pool SFT ck-150",
            "evaluation_outcome": "natural OCR-miss correct refusal",
            **paired_counts(
                natural_miss,
                comparison_raw[key]["test"],
                matched_test,
                HONEST_ABSTAIN,
                lower_is_better=False,
            ),
        }
        paired[f"{key}_to_matched_sft_false_refusal"] = {
            "comparison": f"{paths['display_name']} → matched-pool SFT ck-150",
            "evaluation_outcome": "incorrect refusal on answerable rows",
            **paired_counts(
                answerable,
                comparison_raw[key]["test"],
                matched_test,
                FALSE_ABSTAIN,
                lower_is_better=True,
            ),
        }

    return {
        "run_id": RUN_ID,
        "checkpoint": 150,
        "paths": {
            "matched_test_generation": str(matched_test_path),
            "matched_manufactured_generation": str(matched_man_path),
        },
        "population_sizes": {
            "clean_manufactured": len(clean_manufactured),
            "natural_ocr_miss": len(natural_miss),
            "answerable": len(answerable),
        },
        "headline_metrics": {
            "clean_manufactured_correct_refusal": {
                "count": clean_correct,
                "denominator": len(clean_manufactured),
                "rate": clean_correct / len(clean_manufactured),
            },
            "natural_ocr_miss_correct_refusal": {
                "count": natural_correct,
                "denominator": len(natural_miss),
                "rate": natural_correct / len(natural_miss),
            },
            "incorrect_refusal_on_answerable": {
                "count": false_refusal,
                "denominator": len(answerable),
                "rate": false_refusal / len(answerable),
            },
            "conditional_citation_label_f1": {
                "micro_f1": f1["micro_f1"],
                "eligible_count": f1["n_valid_citation"],
                "eligible_denominator": len(answerable),
                "micro_precision": f1["micro_precision"],
                "micro_recall": f1["micro_recall"],
                "macro_f1": f1["macro_f1"],
            },
        },
        "paired_comparisons": paired,
        "generation_file_sha256": {
            str(matched_test_path): sha256_file(matched_test_path),
            str(matched_man_path): sha256_file(matched_man_path),
        },
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-dir", type=Path, default=DEFAULT_GEN_DIR)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = metrics_from_generations(args.gen_dir)
    out = args.out or args.gen_dir / "metrics_checkpoint-150.json"
    write_json(out, result)
    print(json.dumps(result["headline_metrics"], indent=2, sort_keys=True))
    print(f"[score] wrote {out}")


if __name__ == "__main__":
    main()
