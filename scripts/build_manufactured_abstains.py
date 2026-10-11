"""Canonical build for manufactured-abstain sets (train / val).

Mirrors the committed TEST set (``data/manufactured_test_abstains.jsonl``), changing only
the source split. The original test build was out-of-tree, so ``first_doctype`` and the
drop+renumber transform live HERE (their canonical committed home). Reuses the committed
``scripts.eval_scorer._norm``.

Selection (data-derived from the test-build recovery): answerable, readable doc_type,
cite-span 1-3, gold answer >= 3 normalized chars. NO cap (unlike the test build's 250) —
more data helps GRPO. The ``answer_recurs_in_menu`` decoy flag (which the test build
computed but did NOT persist) is persisted here.

Run:  uv run python scripts/build_manufactured_abstains.py
Writes data/manufactured_{train,val}_abstains.jsonl. Does NOT git-add — review first.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_scorer import _norm          # COMMITTED helper — reused, not reimplemented
try:                                            # reuse the committed menu hash if importable
    from scripts.build_master_dataset import menu_hash
except Exception:                               # fallback keeps the build self-contained
    import hashlib
    def menu_hash(menu):
        return hashlib.sha256(json.dumps(menu, sort_keys=True).encode()).hexdigest()[:16]

READABLE = {"table/list", "layout", "form", "free_text"}


def first_doctype(dt) -> str:
    """Primary doc_type label (doc_type is a multi-tag list). Canonical home for the
    test build's out-of-tree ``ft``."""
    return dt[0] if isinstance(dt, list) and dt else str(dt)


def drop_and_renumber(menu: list[dict], cited: set[int]) -> list[dict]:
    """Drop the gold cite tokens, renumber the remaining tokens 0..N-1 (text + bbox kept).
    Exactly the test build's transform. Renumbering does not affect the text substring used
    for recurrence — it exists so cited ids in the new menu are contiguous 0..N-1."""
    remaining = [m for m in menu if m["id"] not in cited]
    return [{"id": i, "text": m["text"], "bbox": m["bbox"]} for i, m in enumerate(remaining)]


def answer_recurs(golds: list[str], new_menu: list[dict]) -> bool:
    """Decoy rule, ANY accepted answer variant: some ``_norm(gold)`` is a substring of the
    space-joined ``_norm`` remaining-menu text (committed ``_norm``). Any-variant (not just
    ``gold_answers[0]``) because any grabbable answer form makes the abstain non-clean."""
    joined = " ".join(_norm(m.get("text", "")) for m in new_menu)
    return any(_norm(g) and _norm(g) in joined for g in (golds or []))


def dropped_span_matches_gold(golds: list[str], src_menu: list[dict], cited: set[int]) -> bool:
    """True iff the DROPPED cite-span text matches a gold answer variant (a "clean" drop).
    False = frozen-localizer quirk: the cited token(s) are not the literal answer string
    (~20% of sources), inherited from the source data — the abstain label is still valid."""
    span = " ".join(_norm(m.get("text", "")) for m in src_menu if m["id"] in cited)
    return any(_norm(g) and (_norm(g) in span or span in _norm(g)) for g in (golds or []))


def N(step: str, n: int) -> int:
    print(f"    N[{step}] = {n}" + ("   !! EMPTY POPULATION (bug?)" if n == 0 else ""))
    return n


def main() -> None:
    master = {json.loads(l)["record_id"]: json.loads(l) for l in open(_ROOT / "data/master.jsonl")}
    manuf_test = [json.loads(l) for l in open(_ROOT / "data/manufactured_test_abstains.jsonl")]

    # ── FIDELITY: my drop+renumber must reproduce the committed test menus exactly ──────────
    ok = bad = miss = 0
    for r in manuf_test:
        src = master.get(r["source_record_id"])
        if src is None:
            miss += 1; continue
        mine = drop_and_renumber(src["ocr_menu"], set(r["dropped_token_ids"]))
        if mine == r["ocr_menu"]:
            ok += 1
        else:
            bad += 1
    print(f"FIDELITY drop+renumber vs committed test menus: {ok}/{ok + bad} match "
          f"({'PASS' if bad == 0 else f'FAIL — {bad} mismatch'}); {miss} source(s) missing")


    def build(split_label: str, out_name: str):
        print(f"\n=== BUILD split=={split_label!r} ===")
        p = [r for r in master.values() if r.get("split") == split_label];                       N("split", len(p))
        p = [r for r in p if not r.get("target_is_abstain")];                                    N("+ answerable", len(p))
        p = [r for r in p if first_doctype(r.get("doc_type")) in READABLE];                       N("+ readable doc_type", len(p))
        p = [r for r in p if 1 <= len(r.get("cited_ids", [])) <= 3];                              N("+ cite-span 1-3", len(p))
        p = [r for r in p if r.get("gold_answers") and len(_norm(r["gold_answers"][0])) >= 3];    N("+ gold>=3 chars (ELIGIBLE)", len(p))

        recs = []
        for r in p:
            cited = set(r["cited_ids"])
            new_menu = drop_and_renumber(r["ocr_menu"], cited)
            golds = r["gold_answers"]
            rc = answer_recurs(golds, new_menu)                               # any-variant decoy rule
            clean = dropped_span_matches_gold(golds, r["ocr_menu"], cited)    # quirk flag (False = quirk)
            tok = len(cited)
            recs.append({
                "record_id": f"{r['record_id']}_mfab", "source_record_id": r["record_id"],
                "split": split_label, "doc_id": r["doc_id"], "doc_type": r["doc_type"],
                "question": r["question"], "image_path": r.get("image_path"),
                "gold_answers": r["gold_answers"], "target_answer": r.get("target_answer"),
                "target_is_abstain": True, "gold_localized": False, "cited_ids": [],
                "dropped_token_ids": sorted(cited), "token_count": tok,
                "answer_recurs_in_menu": rc,                                  # any-variant
                "dropped_span_matches_gold": clean,                          # quirk flag: False => localizer quirk
                "stratum": f"{'single' if tok == 1 else 'multi'}_{'decoy' if rc else 'unique'}",
                "ocr_menu": new_menu, "ocr_menu_hash": menu_hash(new_menu),
                "localizer_score": None, "abstain_reason": "manufactured_evidence_drop",
            })
        N("emitted (== eligible, no cap)", len(recs))
        outp = _ROOT / "data" / out_name
        with open(outp, "w") as f:
            for x in recs:
                f.write(json.dumps(x) + "\n")
        return recs, outp


    train, train_p = build("train", "manufactured_train_abstains.jsonl")
    val, val_p = build("validation", "manufactured_val_abstains.jsonl")   # spec 'val' == master label 'validation'

    # ── stratum distributions ───────────────────────────────────────────────────────────────
    for name, recs in (("train", train), ("val", val)):
        dist = dict(sorted(Counter(x["stratum"] for x in recs).items()))
        dec = sum(x["answer_recurs_in_menu"] for x in recs)
        quirk = sum(1 for x in recs if not x["dropped_span_matches_gold"])
        print(f"\n{name} stratum dist: {dist}")
        print(f"  decoy (any-variant recurs): {dec}/{len(recs)} = {dec / len(recs):.1%}" if recs else "  (empty)")
        print(f"  localizer-quirk (dropped span != gold): {quirk}/{len(recs)} = {quirk / len(recs):.1%}" if recs else "")

    # ── ASSERTIONS (PASS/FAIL) ──────────────────────────────────────────────────────────────
    test_docs = {str(r["doc_id"]) for r in manuf_test};                       N("test-manuf doc_ids", len(test_docs))
    tr_docs = {str(x["doc_id"]) for x in train};                             N("train doc_ids", len(tr_docs))
    va_docs = {str(x["doc_id"]) for x in val};                               N("val doc_ids", len(va_docs))
    PF = lambda b: "PASS" if b else "FAIL"
    print("\n=== ASSERTIONS ===")
    print(f"- train ∩ test-manuf doc_ids == ∅ : {PF(tr_docs.isdisjoint(test_docs))}  (overlap {len(tr_docs & test_docs)})")
    print(f"- val   ∩ test-manuf doc_ids == ∅ : {PF(va_docs.isdisjoint(test_docs))}  (overlap {len(va_docs & test_docs)})")
    print(f"- train ∩ val doc_ids == ∅        : {PF(tr_docs.isdisjoint(va_docs))}  (overlap {len(tr_docs & va_docs)})")
    print(f"- all train sources split=='train'      : {PF(all(master[x['source_record_id']]['split'] == 'train' for x in train))}")
    print(f"- all val sources split=='validation'   : {PF(all(master[x['source_record_id']]['split'] == 'validation' for x in val))}")

    # ── one full before/after example (a clean 'single_unique' so the answer is genuinely gone) ─
    ex = next((x for x in train if x["stratum"] == "single_unique"), train[0])
    src = master[ex["source_record_id"]]
    cited = set(src["cited_ids"])
    lo, hi = min(cited) - 3, max(cited) + 3
    print("\n=== BEFORE/AFTER EXAMPLE ===")
    print(f"source={ex['source_record_id']} doc={ex['doc_id']} type={first_doctype(ex['doc_type'])} | Q: {ex['question']}")
    print(f"gold_answers={ex['gold_answers']}  stratum={ex['stratum']}  answer_recurs={ex['answer_recurs_in_menu']}")
    print(f"DROPPED gold span ids {sorted(cited)}: {[(m['id'], m['text']) for m in src['ocr_menu'] if m['id'] in cited]}")
    print("SOURCE menu around span:", [(m['id'], m['text']) for m in src['ocr_menu'] if lo <= m['id'] <= hi])
    kept_before = [m['id'] for m in src['ocr_menu'] if m['id'] < min(cited)]
    newpos = len(kept_before)
    print(f"NEW menu at the drop point (renumbered, gold token absent):",
          [(m['id'], m['text']) for m in ex['ocr_menu'] if newpos - 3 <= m['id'] <= newpos + 2])
    print(f"answer text present anywhere in NEW menu? {answer_recurs(ex['gold_answers'], ex['ocr_menu'])}  "
          f"dropped_span_matches_gold={ex['dropped_span_matches_gold']}")

    print(f"\nWROTE: {train_p.relative_to(_ROOT)}  ({len(train)} records)")
    print(f"WROTE: {val_p.relative_to(_ROOT)}  ({len(val)} records)")
    print("Outputs are deterministic from master.jsonl; review any diff before committing changes.")


if __name__ == "__main__":
    main()
