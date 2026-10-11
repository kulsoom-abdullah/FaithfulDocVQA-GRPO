"""Derive + verify the v2 3-field contract from the frozen v1 dataset.

STEP 1 of the 2->3 field change: BUILD AND VERIFY ONLY. Reads data/master.jsonl
READ-ONLY and writes a NEW file (data/master_3field.jsonl); it does NOT modify the
frozen dataset, regenerate SFT targets, or touch format_sft/train_sft.

The 3-field contract decouples visual perception from menu attribution:
  found_answer         — what the model perceived in the page IMAGE.
                         v1: the image is never masked, so the answer is always
                         visible => the gold answer string, on BOTH answerable and
                         abstain rows (the abstain rows used to throw this away).
                         v2 (masked/illegible): "unanswerable" — handled via the
                         `image_masked` branch so the transform is v2-ready, but no
                         v1 record sets it.
  answerable_from_menu — is the answer citable from the OCR menu == gold_localized.
  evidence_ids         — the gold cited token ids (== cited_ids; non-empty iff
                         answerable_from_menu).

Run:  python scripts/build_3field.py        # build + verify + examples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.faithful_io import build_target, derive_3field  # 2-field (side-by-side) + v2 derivation

MASTER = ROOT / "data" / "master.jsonl"
OUT = ROOT / "data" / "master_3field.jsonl"


def build(master_path: Path = MASTER, out_path: Path = OUT) -> list[dict]:
    """Write each frozen record + its derived 3-field block to a NEW file. Returns
    the in-memory rows (original record with a ``v2_contract`` key added)."""
    rows = []
    with master_path.open() as f:
        for line in f:
            rec = json.loads(line)
            rec["v2_contract"] = derive_3field(rec)
            rows.append(rec)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def verify(rows: list[dict]) -> dict:
    """Run the invariant checks. Returns a report dict (counts + any failures)."""
    fails: dict[str, list[str]] = {
        "answerable_bad_found": [], "answerable_not_menu": [], "answerable_empty_ev": [],
        "abstain_bad_found": [], "abstain_is_menu": [], "abstain_nonempty_ev": [],
        "contradiction_true_empty": [], "contradiction_false_nonempty": [],
        "alignment_abstain_vs_localized": [],
    }
    n_ans = n_abs = 0
    for r in rows:
        rid = r["record_id"]
        c = r["v2_contract"]
        fa, afm, ev = c["found_answer"], c["answerable_from_menu"], c["evidence_ids"]
        gold0 = r["gold_answers"][0] if r["gold_answers"] else None
        is_abstain = bool(r["target_is_abstain"])

        # consistency the derivation relies on (v1: abstain <=> not localized)
        if is_abstain != (not bool(r["gold_localized"])):
            fails["alignment_abstain_vs_localized"].append(rid)

        if not is_abstain:                                   # gold-ANSWERABLE
            n_ans += 1
            if not fa or fa != gold0:
                fails["answerable_bad_found"].append(rid)
            if afm is not True:
                fails["answerable_not_menu"].append(rid)
            if not ev:
                fails["answerable_empty_ev"].append(rid)
        else:                                                # gold-ABSTAIN
            n_abs += 1
            if not fa or fa != gold0:                         # NEW: non-empty, == gold
                fails["abstain_bad_found"].append(rid)
            if afm is not False:
                fails["abstain_is_menu"].append(rid)
            if ev:
                fails["abstain_nonempty_ev"].append(rid)

        # contradictions (any row)
        if afm is True and not ev:
            fails["contradiction_true_empty"].append(rid)
        if afm is False and ev:
            fails["contradiction_false_nonempty"].append(rid)

    total_fail = sum(len(v) for v in fails.values())
    return {"n_total": len(rows), "n_answerable": n_ans, "n_abstain": n_abs,
            "failures": fails, "total_failures": total_fail}


def _print_examples(rows: list[dict], n: int = 3) -> None:
    ans = [r for r in rows if not r["target_is_abstain"]]
    abs_ = [r for r in rows if r["target_is_abstain"]]
    for label, subset in (("ANSWERABLE", ans), ("ABSTAIN", abs_)):
        print(f"\n--- {label}: old 2-field target  vs  new 3-field contract ---")
        for r in subset[:n]:
            print(f"  record_id={r['record_id']}  q={r['question'][:54]!r}")
            print(f"    OLD 2-field : {build_target(r)}")
            print(f"    NEW 3-field : {json.dumps(r['v2_contract'], ensure_ascii=False)}")


def main() -> None:
    assert MASTER.exists(), f"missing {MASTER}"
    rows = build()
    print(f"[build] wrote {OUT.relative_to(ROOT)}  ({len(rows)} rows)  — master.jsonl untouched")
    rep = verify(rows)
    print(f"\n[verify] total={rep['n_total']}  answerable={rep['n_answerable']}  abstain={rep['n_abstain']}")
    print(f"[verify] TOTAL INVARIANT FAILURES: {rep['total_failures']} (expect 0)")
    for k, v in rep["failures"].items():
        flag = "" if not v else "   <-- FAILURES"
        print(f"    {k:<32} {len(v)}{flag}" + (f"  e.g. {v[:5]}" if v else ""))
    _print_examples(rows)


if __name__ == "__main__":
    main()
