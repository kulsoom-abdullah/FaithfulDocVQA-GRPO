"""Image-ablation eval — does the trained model USE the page image, or has it
collapsed onto the OCR menu?

Run:  python scripts/eval_ablation.py --adapter <checkpoint dir> --split validation
      python scripts/eval_ablation.py --report <condition jsons>
(The raw condition JSONs behind docs/DECISIONS.md §2 live on the Modal volume and
are regenerable with the first form; they are not tracked.)

Per gold-ANSWERABLE record, generate under two conditions over the SAME image and
compare answer-correctness:

  (A) MENU    — the normal trained prompt: image + question + OCR menu, with the
                standard SYSTEM_INSTRUCTION (cite OCR ids / else unanswerable).
                This is exactly ``eval_runner.generate`` with its default prompt.
  (B) NO-MENU — image + question only, with a COHERENT no-menu prompt (below):
                "answer from the page image; no menu provided; evidence_ids=[]".
                NOT the menu prompt with the menu deleted (that would leave the
                instruction citing a menu that isn't there — a broken prompt that
                tests confusion, not image-use).

Large (A)->(B) correctness drop  => leans on the menu, under-uses the image.
Small drop                       => reads the image fine.

Diagnostic only: TRAIN and VALIDATION (not test), reported SEPARATELY — if train
no-menu correctness >> validation, that gap is a MEMORIZATION signal and the
validation number is the trusted one.

Inference only; reuses ``eval_runner`` load + generate (greedy, batch 1). GPU to
run, CPU-safe to import. Persists per-record JSON so it is re-scoreable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_runner import generate, load_adapted_model, render_prompt  # noqa: F401
from scripts.faithful_io import parse_tolerant_v3
from scripts.format_sft import load_split

# Coherent no-menu prompt: self-consistent (no dangling "cite the menu" instruction),
# same three-field JSON schema, with menu support false and evidence_ids empty
# because no OCR menu is provided.
NOMENU_SYSTEM = (
    "You are a document visual-question-answering system. You are given a page "
    "image and a question. No OCR token menu is provided.\n"
    "Answer the question using the page image. Output ONLY a single JSON object of "
    'the form {"found_answer": <string>, "answerable_from_menu": false, '
    '"evidence_ids": []}. No other text, no explanation.'
)


# ── answer-correctness matcher (NFC + curly/straight quote fold) ───────────────

def _norm(s: str) -> str:
    """NFC-normalize, fold curly quotes to straight, collapse whitespace, casefold.
    Applied identically to prediction and golds; the (A)->(B) delta is robust to the
    exact normalization since both conditions are scored the same way."""
    s = unicodedata.normalize("NFC", s or "")
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return " ".join(s.split()).casefold()


def answer_correct(pred: str | None, golds: list[str]) -> bool:
    """True iff the normalized prediction exactly matches any normalized gold."""
    p = _norm(pred or "")
    return any(p == _norm(g) for g in golds)


def _answer_of(raw: str) -> str | None:
    """Extract ``found_answer`` through the canonical three-field parser."""
    parsed = parse_tolerant_v3(raw)
    return parsed.value.found_answer if parsed.value is not None else None


# ── no-menu prompt ────────────────────────────────────────────────────────────

def nomenu_messages(record: dict) -> list[dict]:
    """Chat messages for the NO-MENU condition: system + (image, question)."""
    return [
        {"role": "system", "content": NOMENU_SYSTEM},
        {"role": "user", "content": [
            {"type": "image", "image": record["image_path"]},
            {"type": "text", "text": f"Question: {record['question']}"},
        ]},
    ]


def nomenu_prompt(processor, record: dict) -> str:
    return processor.apply_chat_template(
        nomenu_messages(record), tokenize=False, add_generation_prompt=True)


# ── record selection ──────────────────────────────────────────────────────────

def _doc_type_key(record: dict) -> str:
    return "|".join(sorted(record.get("doc_type") or ["?"]))


def pick_records(split: str, limit: int | None, seed: int = 42) -> list[dict]:
    """Gold-ANSWERABLE records for a split (mode B = full split). If ``limit`` is
    set and smaller than the split, return a seeded subsample **stratified by
    doc_type** (proportional allocation, largest-remainder to hit exactly ``limit``)
    so the sample preserves the split's doc_type mix."""
    import random
    from collections import defaultdict

    recs = [r for r in load_split(split, "B") if not r["target_is_abstain"]]
    if not (limit and limit < len(recs)):
        return recs

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[_doc_type_key(r)].append(r)

    n = len(recs)
    raw = {k: limit * len(v) / n for k, v in groups.items()}
    alloc = {k: int(x) for k, x in raw.items()}
    for k in sorted(groups, key=lambda k: raw[k] - alloc[k], reverse=True)[: limit - sum(alloc.values())]:
        alloc[k] += 1

    picked: list[dict] = []
    leftover: list[dict] = []
    for k, v in groups.items():
        vv = v[:]
        rng.shuffle(vv)
        a = min(alloc[k], len(vv))
        picked += vv[:a]
        leftover += vv[a:]
    if len(picked) < limit:                       # safety fill (rare rounding edge)
        rng.shuffle(leftover)
        picked += leftover[: limit - len(picked)]
    return sorted(picked, key=lambda r: r["record_id"])


# ── stats ─────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (0.0, 0.0, 0.0)
    z = 1.959963985
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, c - m), min(1.0, c + m)


# ── run one (adapter, split) ──────────────────────────────────────────────────

def run(adapter: str, split: str, limit: int | None, seed: int,
        max_new_tokens: int, out_path: Path) -> dict:
    records = pick_records(split, limit, seed)
    print(f"[ablation] adapter={adapter} split={split} answerable={len(records)}"
          f"{f' (subsampled, seed {seed})' if limit and limit < 10**9 else ''}")
    model, proc, resolved = load_adapted_model(adapter)

    rows: list[dict] = []
    t0 = time.time()
    for i, r in enumerate(records, 1):
        golds = r["gold_answers"]
        m = generate(model, proc, r, max_new_tokens=max_new_tokens)            # MENU (default prompt)
        nm = generate(model, proc, r, max_new_tokens=max_new_tokens,
                      prompt=nomenu_prompt(proc, r))                            # NO-MENU
        ma, na = _answer_of(m["raw"]), _answer_of(nm["raw"])
        rows.append({
            "record_id": r["record_id"], "gold_answers": golds,
            "menu_raw": m["raw"], "menu_answer": ma,
            "menu_correct": answer_correct(ma, golds), "menu_stop": m["stop_reason"],
            "nomenu_raw": nm["raw"], "nomenu_answer": na,
            "nomenu_correct": answer_correct(na, golds), "nomenu_stop": nm["stop_reason"],
        })
        if i % 25 == 0 or i == len(records):
            print(f"    {i}/{len(records)}  ({i/(time.time()-t0):.2f} rec/s)", flush=True)

    n = len(rows)
    mk = sum(x["menu_correct"] for x in rows)
    nk = sum(x["nomenu_correct"] for x in rows)
    mp, mlo, mhi = wilson(mk, n)
    npp, nlo, nhi = wilson(nk, n)
    summary = {
        "adapter": str(adapter), "base": resolved, "split": split, "n": n,
        "limit": limit, "seed": seed,
        "menu_correct": {"k": mk, "rate": round(mp, 4), "ci95": [round(mlo, 4), round(mhi, 4)]},
        "nomenu_correct": {"k": nk, "rate": round(npp, 4), "ci95": [round(nlo, 4), round(nhi, 4)]},
        "delta_menu_minus_nomenu": round(mp - npp, 4),
        "menu_parse_rate": round(sum(x["menu_answer"] is not None for x in rows) / n, 4) if n else None,
        "nomenu_parse_rate": round(sum(x["nomenu_answer"] is not None for x in rows) / n, 4) if n else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "records": rows}, indent=2))
    print(f"  menu={mp:.3f} [{mlo:.3f},{mhi:.3f}]  no-menu={npp:.3f} [{nlo:.3f},{nhi:.3f}]  "
          f"delta={mp-npp:+.3f}  -> {out_path}")
    return summary


def report(json_paths: list[str]) -> None:
    """Print the combined table from persisted ablation JSONs."""
    print(f"\n{'model/checkpoint':<34}{'split':<12}{'menu':<10}{'no-menu':<10}{'delta':<8}")
    print("-" * 74)
    for p in json_paths:
        s = json.loads(Path(p).read_text())["summary"]
        tag = Path(s["adapter"]).parent.name + "/" + Path(s["adapter"]).name
        print(f"{tag:<34}{s['split']:<12}{s['menu_correct']['rate']:<10.3f}"
              f"{s['nomenu_correct']['rate']:<10.3f}{s['delta_menu_minus_nomenu']:<+8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", help="checkpoint dir (e.g. data/checkpoints/sft/sft_B_seed42/checkpoint-642)")
    ap.add_argument("--split", choices=["train", "validation"])
    ap.add_argument("--limit", type=int, default=None, help="subsample N answerable (train is large)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report", nargs="+", metavar="JSON", help="print combined table from JSONs")
    a = ap.parse_args()

    if a.report:
        report(a.report)
        return
    if not a.adapter or not a.split:
        ap.error("--adapter and --split required (or use --report)")
    ad = Path(a.adapter)
    tag = f"{ad.parent.name}_{ad.name}_{a.split}"
    out = Path(a.out) if a.out else ROOT / "data" / "checkpoints" / "ablation" / f"{tag}.json"
    run(a.adapter, a.split, a.limit, a.seed, a.max_new_tokens, out)


if __name__ == "__main__":
    main()
