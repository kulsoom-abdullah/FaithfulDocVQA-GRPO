#!/usr/bin/env python3
"""Retroactively score the 27 Phase 0 / 0.5 audit records with the F1 evidence reward.

Also defines ``localize_gold(tokens, gold)`` — the single-gold localization entry
point imported by ``scripts/build_master_dataset.py`` (a LIVE import of the frozen
dataset build, not scratch).

Reads phase0/outputs/audit_results_annotated.jsonl (READ-ONLY; a gitignored phase0
output, regenerable only by re-running the paid Phase 0/0.5 audit) and writes
phase0/outputs/phase0_f1_results.jsonl — each original record plus three new
fields: evidence_reward, evidence_precision, evidence_recall.

Does NOT mutate the input. Does NOT re-run Phase 0. The output file is written
only after all records are processed without a fatal error.

Run: uv run python phase0/score_phase0_with_f1.py   (from faithfuldocvqa/)
"""

import json
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from phase0.pipeline import compute_evidence_reward
from phase0.phase05 import find_bbox, find_bbox_lenient

PHASE0_DIR  = Path(__file__).parent
OUTPUTS_DIR = PHASE0_DIR / "outputs"
ANNOTATED   = OUTPUTS_DIR / "audit_results_annotated.jsonl"
OUT_PATH    = OUTPUTS_DIR / "phase0_f1_results.jsonl"

REFUSAL_BEHAVIOR = "refuse with empty tokens"

# Matches one dump_ocr line:  ID  TEXT  (x1,y1,x2,y2)  CONF
# Anchored on the bbox tuple shape (not column widths) so a word wider than its
# 22-char column — which drops the trailing pad space — still parses correctly.
OCR_LINE_RE = re.compile(r"^\s*(\d+)\s+(.*?)\s*\((\d+),(\d+),(\d+),(\d+)\)\s+(-?\d+)\s*$")


def parse_ocr_file(path: Path) -> list[dict]:
    """Parse an audit_ocr/*.txt dump back into token dicts: id, word, x1,y1,x2,y2.

    Header and any non-matching lines are skipped. find_bbox/find_bbox_lenient
    only need word + bbox + id, so conf is parsed but discarded.
    """
    tokens = []
    for line in path.read_text().splitlines():
        m = OCR_LINE_RE.match(line)
        if not m:
            continue  # header row or blank line
        tid, word, x1, y1, x2, y2, _conf = m.groups()
        tokens.append({
            "id": int(tid), "word": word.strip(),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
        })
    return tokens


def localize_gold(tokens: list[dict], gold: str) -> set[int]:
    """Locate gold answer text in the OCR menu; return the matched token IDs (or ∅)."""
    bbox = find_bbox(tokens, gold, 0.85) or find_bbox_lenient(tokens, gold)
    return set(bbox["token_ids"]) if bbox else set()


def interpret(cited: set[int], gold: set[int], valid: set[int],
              reward: float, is_refusal: bool) -> str:
    """One-line plain-English reason for the F1 score this record received."""
    fabricated = sorted(cited - valid)
    fab = f" [fabricated IDs {fabricated} not in OCR menu]" if fabricated else ""
    if not gold and not cited:
        tag = "correct refusal" if is_refusal else "both empty"
        return f"{tag}: cited ∅ and gold ∅ → F1=1.0{fab}"
    if not gold and cited:
        why = "should have refused but cited evidence" if is_refusal else "gold ∅ but model cited"
        return f"over-citation: {why} {sorted(cited)} → F1=0{fab}"
    if gold and not cited:
        return f"under-citation: gold has {len(gold)} token(s) but model cited none → F1=0"
    inter = cited & gold
    if not inter:
        return f"no overlap: cited {sorted(cited)} disjoint from gold {sorted(gold)} → F1=0{fab}"
    if reward == 1.0:
        return f"exact match: cited == gold == {sorted(gold)}{fab}"
    return (f"partial overlap: {len(inter)}/{len(cited)} cited in gold, "
            f"{len(inter)}/{len(gold)} gold recalled{fab}")


def case_letter(case_id: str) -> str:
    m = re.search(r"_case([ABCD])", case_id)
    return m.group(1) if m else "?"


def sample_id(case_id: str) -> str:
    m = re.match(r"(img\d+)", case_id)
    return m.group(1) if m else "?"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main():
    records = [json.loads(l) for l in open(ANNOTATED)]
    print(f"Loaded {len(records)} records from {ANNOTATED.name}\n")

    # ── Pass 1: determine gold_cited_ids / cited_ids / valid_ids (no F1 yet) ──
    print("── Pass 1: gold determination ───────────────────────────────")
    enriched: list[dict] = []
    skipped: list[str] = []

    for rec in records:
        cid = rec["case_id"]
        ocr_str = rec.get("ocr_file_path")
        if not ocr_str or not Path(ocr_str).exists():
            print(f"  SKIP {cid}: ocr_file_path missing/unresolvable ({ocr_str!r})")
            skipped.append(cid)
            continue

        tokens   = parse_ocr_file(Path(ocr_str))
        valid    = {t["id"] for t in tokens}
        raw_cited = rec.get("parsed_evidence_tokens") or []   # None/missing → []
        cited    = {int(x) for x in raw_cited}

        is_refusal = rec.get("expected_behavior") == REFUSAL_BEHAVIOR
        if is_refusal:
            # Refusal-expected records: gold is the empty set because the correct
            # citation IS the empty set. Localizing the gold answer text would
            # punish correct refusals as failures.
            gold = set()
            src  = "refusal→∅"
        else:
            gold = localize_gold(tokens, rec.get("gold_answer", ""))
            src  = "localized" if gold else "localize-failed→∅"

        enriched.append({
            "rec": rec, "cited": cited, "gold": gold, "valid": valid,
            "is_refusal": is_refusal,
        })
        print(f"  {cid:<20} {src:<18} |menu|={len(valid):<4} "
              f"cited={sorted(cited)} gold={sorted(gold)}")

    # ── Localization / refusal accounting (before any F1) ──
    n_refusal   = sum(1 for e in enriched if e["is_refusal"])
    n_answer    = sum(1 for e in enriched if not e["is_refusal"])
    n_loc_ok    = sum(1 for e in enriched if not e["is_refusal"] and e["gold"])
    n_loc_fail  = sum(1 for e in enriched if not e["is_refusal"] and not e["gold"])
    n_gold_ne   = sum(1 for e in enriched if e["gold"])
    n_gold_emp  = sum(1 for e in enriched if not e["gold"])

    print("\n── Gold-set accounting ──────────────────────────────────────")
    print(f"  processed records           : {len(enriched)}  (skipped: {len(skipped)})")
    print(f"  refusal-expected → forced ∅ : {n_refusal}")
    print(f"  answer-expected (localize)  : {n_answer}")
    print(f"      └ localized (non-empty) : {n_loc_ok}")
    print(f"      └ localize failed (∅)   : {n_loc_fail}")
    print(f"  TOTAL non-empty gold        : {n_gold_ne}")
    print(f"  TOTAL empty gold            : {n_gold_emp}")

    # ── Pass 2: compute F1 reward ──
    print("\n── Pass 2: F1 evidence reward ───────────────────────────────")
    out_records: list[dict] = []
    for e in enriched:
        rec, cited, gold, valid = e["rec"], e["cited"], e["gold"], e["valid"]
        scored = compute_evidence_reward(sorted(cited), gold, valid,
                                         is_answerable=not e["is_refusal"])
        out = {**rec, **scored}
        out_records.append(out)
        print(f"  {rec['case_id']:<20} P={scored['evidence_precision']:.3f} "
              f"R={scored['evidence_recall']:.3f} F1={scored['evidence_reward']:.3f}")

    # ── Summary ──
    print_summary(enriched, out_records)

    # ── Write output ONLY after all records processed without error ──
    with open(OUT_PATH, "w") as f:
        for out in out_records:
            f.write(json.dumps(out) + "\n")
    print(f"\nWrote {len(out_records)} records → {OUT_PATH}")
    print(f"(input {ANNOTATED.name} untouched)")


def print_summary(enriched: list[dict], out_records: list[dict]):
    by_e = {e["rec"]["case_id"]: e for e in enriched}
    rewards = [r["evidence_reward"] for r in out_records]

    print("\n" + "=" * 64)
    print("SUMMARY — F1 evidence reward over Phase 0 / 0.5 audit records")
    print("=" * 64)
    print(f"\nMean evidence_reward (all {len(out_records)}): {mean(rewards):.3f}")

    # By case letter
    print("\nMean evidence_reward by case (suffix after _case):")
    buckets: dict[str, list[float]] = {}
    for r in out_records:
        buckets.setdefault(case_letter(r["case_id"]), []).append(r["evidence_reward"])
    for c in sorted(buckets):
        vals = buckets[c]
        print(f"  Case {c}: {mean(vals):.3f}  (n={len(vals)})")

    # By sample
    print("\nMean evidence_reward by sample:")
    sbuckets: dict[str, list[float]] = {}
    for r in out_records:
        sbuckets.setdefault(sample_id(r["case_id"]), []).append(r["evidence_reward"])
    for s in sorted(sbuckets):
        vals = sbuckets[s]
        print(f"  {s}: {mean(vals):.3f}  (n={len(vals)})")

    # Top 3 / bottom 3
    ranked = sorted(out_records, key=lambda r: r["evidence_reward"], reverse=True)
    print("\nTop 3 (highest F1):")
    for r in ranked[:3]:
        print(f"  {r['case_id']:<20} F1={r['evidence_reward']:.3f}")

    print("\nBottom 3 (lowest F1):")
    for r in ranked[-3:]:
        e = by_e[r["case_id"]]
        reason = interpret(e["cited"], e["gold"], e["valid"],
                           r["evidence_reward"], e["is_refusal"])
        print(f"  {r['case_id']:<20} F1={r['evidence_reward']:.3f} "
              f"cited={sorted(e['cited'])} gold={sorted(e['gold'])}")
        print(f"      → {reason}")

    # Explicit img2_caseC flag
    print("\n── img2_caseC (flagged) ─────────────────────────────────────")
    target = next((r for r in out_records if r["case_id"] == "img2_caseC"), None)
    if target:
        e = by_e["img2_caseC"]
        reason = interpret(e["cited"], e["gold"], e["valid"],
                           target["evidence_reward"], e["is_refusal"])
        print(f"  F1={target['evidence_reward']:.3f} "
              f"P={target['evidence_precision']:.3f} R={target['evidence_recall']:.3f}")
        print(f"  cited_ids={sorted(e['cited'])}  gold_cited_ids={sorted(e['gold'])}  "
              f"|menu|={len(e['valid'])}")
        print(f"  → {reason}")
    else:
        print("  (img2_caseC not found)")


if __name__ == "__main__":
    main()
