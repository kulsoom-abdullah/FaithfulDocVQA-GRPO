"""survivor_cases.py — enumerate manufactured abstains whose gold answer RECURS in the
renumbered menu ("survivors"), with full context to hand-judge each as a genuine
distractor (unrelated context) vs a legitimate second answer location.

CPU-only, pure DATA (no model output, no GPU). Reuses scripts.eval_scorer._norm.

Run:  uv run python scripts/survivor_cases.py

Emits:
  - counts of survivors / clean-remaining under exact / punctuation-tolerant / fuzzy defs
  - a detailed per-case dump (record_id, question, gold, dropped gold token + its original
    context, the surviving occurrence(s) id/text/bbox + its menu context) -> survivor_cases.txt
"""

from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_scorer import _norm

MASTER = _ROOT / "data" / "master.jsonl"
MANUF = _ROOT / "data" / "manufactured_test_abstains.jsonl"
OUT = _ROOT / "survivor_cases.txt"
FUZZY_THR = 0.80


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def exact_hits(golds: list[str], menu: list[dict]) -> list[list[int]]:
    """Menu index-spans whose _norm text == _norm(gold) (single token or contiguous window)."""
    toks = [_norm(t.get("text", "")) for t in menu]
    spans = []
    for g in golds:
        gn = _norm(g)
        if not gn:
            continue
        parts = gn.split()
        if len(parts) == 1:
            spans += [[i] for i, t in enumerate(toks) if t == gn]
        else:
            n = len(parts)
            for i in range(len(toks) - n + 1):
                if " ".join(toks[i:i + n]) == gn:
                    spans.append(list(range(i, i + n)))
    return spans


def punct_hits(golds: list[str], menu: list[dict]) -> list[list[int]]:
    """Index-spans matching under alnum-normalization: single-token exact/containment(len>=3)
    or an exact contiguous alnum-join window. This is the 'punctuation-tolerant' rule."""
    toks = [alnum(t.get("text", "")) for t in menu]
    spans = []
    for g in golds:
        ga = alnum(g)
        if not ga:
            continue
        for i, t in enumerate(toks):
            if t and (t == ga or (len(ga) >= 3 and ga in t)):
                spans.append([i])
        nw = len(g.split())
        if nw >= 2:
            for i in range(len(toks)):
                join = ""
                for j in range(i, min(i + nw + 1, len(toks))):
                    join += toks[j]
                    if join == ga:
                        spans.append(list(range(i, j + 1)))
                        break
                    if len(join) >= len(ga):
                        break
    # dedup
    uniq = []
    for s in spans:
        if s not in uniq:
            uniq.append(s)
    return uniq


def best_fuzzy(golds: list[str], menu: list[dict], win: int = 3) -> tuple[float, list[int]]:
    toks = [alnum(t.get("text", "")) for t in menu]
    best, span = 0.0, []
    for g in golds:
        ga = alnum(g)
        if not ga:
            continue
        for i in range(len(toks)):
            join = ""
            for j in range(i, min(i + win, len(toks))):
                join += toks[j]
                if not join:
                    continue
                r = difflib.SequenceMatcher(None, join, ga).ratio()
                if r > best:
                    best, span = r, list(range(i, j + 1))
    return best, span


def ctx(menu: list[dict], idxs: list[int], pad: int = 4) -> str:
    lo, hi = max(0, min(idxs) - pad), min(len(menu), max(idxs) + pad + 1)
    parts = []
    for i in range(lo, hi):
        tk = menu[i]
        mark = "»" if i in idxs else " "
        parts.append(f"{mark}{tk['id']}:{tk.get('text','')!r}")
    return "  ".join(parts)


def main() -> None:
    master = {r["record_id"]: r for r in load_jsonl(MASTER)}
    manuf = load_jsonl(MANUF)

    # ── counts under each definition ────────────────────────────────────────────────
    n = len(manuf)
    c_exact = c_punct = c_fuzzy = 0
    punct_recs = []
    fuzzy_only = []
    for r in manuf:
        golds = r.get("gold_answers", [])
        menu = r["ocr_menu"]
        e = bool(exact_hits(golds, menu))
        p = bool(punct_hits(golds, menu))
        fr, _ = best_fuzzy(golds, menu)
        f = fr >= FUZZY_THR
        c_exact += e
        c_punct += p
        c_fuzzy += f
        if p:
            punct_recs.append(r)
        elif f:
            fuzzy_only.append((r, fr))

    print("=" * 74)
    print("SURVIVOR COUNTS (manufactured n=250) — answer RECURS in the renumbered menu")
    print("=" * 74)
    print(f"{'definition':<26}{'survivors':>10}{'clean remaining':>18}")
    print(f"{'exact (normalized)':<26}{c_exact:>10}{n - c_exact:>18}")
    print(f"{'punctuation-tolerant':<26}{c_punct:>10}{n - c_punct:>18}")
    print(f"{'fuzzy (difflib>=%.2f)' % FUZZY_THR:<26}{c_fuzzy:>10}{n - c_fuzzy:>18}")
    print(f"\nfuzzy-only (caught by fuzzy but NOT punct): {len(fuzzy_only)}")

    # ── detailed dump of punct-survivors (+ fuzzy-only) ─────────────────────────────
    lines = [f"SURVIVOR CASES — {c_punct} punctuation-tolerant survivors (+{len(fuzzy_only)} fuzzy-only)",
             "For each: judge (a) genuine distractor in UNRELATED context (keep in clean set) or",
             "(b) legitimate SECOND valid answer location (exclude). » marks the matched token(s).",
             "=" * 90, ""]

    for r in punct_recs:
        src = r.get("source_record_id")
        golds = r.get("gold_answers", [])
        menu = r["ocr_menu"]
        tw = master.get(src, {})
        o_menu = tw.get("ocr_menu", [])
        o_by_id = {t["id"]: t for t in o_menu}
        o_idx = {t["id"]: i for i, t in enumerate(o_menu)}
        dropped_ids = r.get("dropped_token_ids", []) or tw.get("cited_ids", [])
        dropped = [(i, o_by_id.get(i, {}).get("text"), o_by_id.get(i, {}).get("bbox")) for i in dropped_ids]
        spans = punct_hits(golds, menu)
        surv = []
        for sp in spans:
            for i in sp:
                tk = menu[i]
                surv.append((tk["id"], tk.get("text"), tk.get("bbox")))
        lines.append(f"--- {r['record_id']}  (source={src})  doc_type={r.get('doc_type')}")
        lines.append(f"  Q: {r.get('question')}")
        lines.append(f"  gold_answers   : {golds}")
        lines.append(f"  DROPPED gold   : {dropped}")
        if dropped_ids and o_menu:
            di = [o_idx[i] for i in dropped_ids if i in o_idx]
            if di:
                lines.append(f"    orig context : {ctx(o_menu, di)}")
        lines.append(f"  SURVIVOR(s)    : {surv}")
        if spans:
            lines.append(f"    surv context : {ctx(menu, spans[0])}")
        lines.append("")

    if fuzzy_only:
        lines += ["", "=" * 90, f"FUZZY-ONLY survivors ({len(fuzzy_only)}) — weaker matches (difflib >= %.2f, not exact/punct):" % FUZZY_THR, ""]
        for r, fr in fuzzy_only:
            menu = r["ocr_menu"]
            _, sp = best_fuzzy(r.get("gold_answers", []), menu)
            lines.append(f"--- {r['record_id']}  gold={r.get('gold_answers')}  ratio={fr:.2f}")
            lines.append(f"    Q: {r.get('question')}")
            if sp:
                lines.append(f"    near-match ctx: {ctx(menu, sp)}")
            lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {c_punct} punct-survivor cases (+{len(fuzzy_only)} fuzzy-only) -> {OUT.name}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
