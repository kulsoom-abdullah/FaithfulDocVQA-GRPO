"""apply_survivor_audit.py — persist the hand-decided survivor audit + recompute.

CPU-only. (1) Annotates data/manufactured_test_abstains.jsonl with per-record
`exclusion_reason` (the (b) category for defensible-second-location survivors; null for
keeps), `gold_label_defect`, and `survivor_pending`. (2) Writes docs/findings/
survivor_audit.md (rule + per-case table + defect note). (3) Recomputes honest_abstain /
false_grounding over the survivor-audited clean set using the ck-642 manufactured gen.

This is what makes the clean-vs-decoy split reproducible from committed data. (The split
annotations regenerate from the committed jsonl alone; the rate RECOMPUTE reads the
ck-642 manufactured gen, committed at data/checkpoints/eval/sft_B_seed42/.)

Run:  uv run python scripts/apply_survivor_audit.py [--gen <manufactured gen json>]
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_scorer import _norm, classify, HONEST_ABSTAIN, FALSE_GROUNDING

MANUF = _ROOT / "data" / "manufactured_test_abstains.jsonl"
AUDIT_MD = _ROOT / "docs" / "findings" / "survivor_audit.md"

# The ck-642 manufactured gen is committed (force-added eval evidence); the default
# points at the in-repo copy. Canonical original: the Modal `faithfuldocvqa-ckpts`
# volume (eval/sft_B_seed42/). Override via --gen or FDV_GEN_MANUFACTURED.
_DEF_GEN = os.environ.get(
    "FDV_GEN_MANUFACTURED",
    "data/checkpoints/eval/sft_B_seed42/gen_manufactured_checkpoint-642.json")

# ── hand-audit decisions (keyed by source_record_id) ────────────────────────────
EXCLUDES = {  # survivor = defensible second answer location -> (b) category
    "15277": "abbrev definition site — '(NSE)' immediately follows 'National Stock Exchange'",
    "16341": "abbrev definition site — '(PLM)' follows 'Product Lifecycle Management'",
    "47496": "title repeated — the short title recurs at a second valid site",
    "52154": "recipient recurs — the addressee name appears at a second valid site",
    "50122": "answer restated — '6,400 net sq. ft.' appears in the SUMMARY block",
    "50939": "defined term recurs — 'electronically stored information' at a second valid site",
    "1013":  "gold alias at its own site — 'close to 2 years'",
    "31020": "answer recurs — 'University of Zurich' at additional valid CV entries",
    "51697": "answer recurs — 'First Class' on the Feb-13 flight entry (valid site)",
    "52157": "gold alias recurs — 'Director' at a second valid site",
    "43779": "second valid site — '7,502.53' beside 'Total' (the amount asked)",
    "47452": "answer recurs — deposit total '2,200.72' at a second valid site",
    "53439": "same value adjacent field — '$0.00' also the neighbouring Coupon Amount",
    "47535": "second valid location — 'Nashville' in the 'LOCATION OUTSIDE NYC' answer column (not the Vanderbilt purpose cell)",
    "51687": "hand-adjudicated 2026-08-08: surviving site (post-drop id 72, Feb-13 row) genuinely supports the answer; gold cite was a string-match quirk (Feb-7 row); sibling 51697_mfab was caught by the original audit, this one under-detected per §4's substring caveat",
}
PENDING = set()                       # STEP 1 fully resolved: 47535 exclude, 51687 KEEP (cited-token rule)
KEEP_NOTES = {  # per-record note for the retained table — exemplars of the disposition rule
    "51687": "EXEMPLAR (cited-token rule) — model cited the Feb-7 L.A. (id30) for a Feb-13 question; "
             "valid Feb-13 survivor (id72/83) went unused",
}
DEFECT = {"45817"}                    # gold-label defect: gold "Published" ≠ the date/journal asked
KEEP_EXPLICIT = {"38902", "16501", "51133", "51704", "43771", "16826", "4522",
                 "45817", "47487", "4968", "54911",
                 "55387", "55388", "55390", "55392", "55395", "55396"}
FUZZY_THR = 0.80


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def punct_span(golds, menu):
    toks = [alnum(t.get("text", "")) for t in menu]
    for g in golds:
        ga = alnum(g)
        if not ga:
            continue
        for i, t in enumerate(toks):
            if t and (t == ga or (len(ga) >= 3 and ga in t)):
                return [i]
        nw = len(g.split())
        if nw >= 2:
            for i in range(len(toks)):
                join = ""
                for j in range(i, min(i + nw + 1, len(toks))):
                    join += toks[j]
                    if join == ga:
                        return list(range(i, j + 1))
                    if len(join) >= len(ga):
                        break
    return None


def fuzzy_span(golds, menu, win=3):
    toks = [alnum(t.get("text", "")) for t in menu]
    best, span = 0.0, None
    for g in golds:
        ga = alnum(g)
        if not ga:
            continue
        for i in range(len(toks)):
            join = ""
            for j in range(i, min(i + win, len(toks))):
                join += toks[j]
                if join:
                    r = difflib.SequenceMatcher(None, join, ga).ratio()
                    if r > best:
                        best, span = r, list(range(i, j + 1))
    return (span, best) if best >= FUZZY_THR else (None, best)


def ctx(menu, idxs, pad=3):
    lo, hi = max(0, min(idxs) - pad), min(len(menu), max(idxs) + pad + 1)
    return " ".join(("»" if i in idxs else "") + str(menu[i].get("text", "")) for i in range(lo, hi))


def main() -> None:
    _ap = argparse.ArgumentParser(description="Persist the survivor audit + recompute clean-set rates (CPU).")
    _ap.add_argument("--gen", type=Path, default=Path(_DEF_GEN),
                     help="ck-642 manufactured gen JSON (env: FDV_GEN_MANUFACTURED)")
    GEN = _ap.parse_args().gen
    if not GEN.exists():
        sys.exit(f"missing manufactured gen: {GEN}\n  pull it from the Modal ckpts volume "
                 f"(see header) or pass --gen / set FDV_GEN_MANUFACTURED.")

    records = [json.loads(l) for l in open(MANUF) if l.strip()]
    gen = {r["record_id"]: r.get("raw") for r in json.loads(GEN.read_text())["records"]}

    # ── annotate + detect survivors ─────────────────────────────────────────────────
    survivors = []   # (rec, span, kind, disposition, reason)
    for rec in records:
        src = rec["source_record_id"]
        rec["exclusion_reason"] = EXCLUDES.get(src)
        rec["gold_label_defect"] = src in DEFECT
        rec["survivor_pending"] = src in PENDING

        golds, menu = rec.get("gold_answers", []), rec["ocr_menu"]
        sp = punct_span(golds, menu)
        kind = "punct"
        if sp is None:
            sp, _ = fuzzy_span(golds, menu)
            kind = "fuzzy"
        if sp is not None:
            disp = "EXCLUDE" if src in EXCLUDES else ("PENDING" if src in PENDING else "keep")
            reason = (EXCLUDES[src] if src in EXCLUDES else
                      "STEP-1 pending hand-review (provisionally exclude)" if src in PENDING else
                      "survivor in unrelated context — citing it is genuine false-grounding")
            survivors.append((rec, sp, kind, disp, reason))

    # ── consistency warnings ────────────────────────────────────────────────────────
    surv_ids = {r["source_record_id"] for r, *_ in survivors}
    for s in EXCLUDES:
        if s not in surv_ids:
            print(f"WARN: exclude {s} not detected as a survivor (check rule)")
    for s in KEEP_EXPLICIT:
        if s not in surv_ids and s != "45817":
            print(f"WARN: explicit-keep {s} not detected as a survivor")

    # ── persist annotated jsonl ─────────────────────────────────────────────────────
    with open(MANUF, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"persisted annotations -> {MANUF.name} "
          f"({len(EXCLUDES)} excluded, {len(PENDING)} pending, {len(DEFECT)} defect, "
          f"{len(survivors)} survivors total)")


    # ── recompute over the audited clean set ────────────────────────────────────────
    def rates(recs):
        facts = [classify(r, gen.get(r["record_id"])) for r in recs]
        n = len(facts)
        ha = sum(f["bucket"] == HONEST_ABSTAIN for f in facts) / n
        fg = sum(f["bucket"] == FALSE_GROUNDING for f in facts) / n
        return n, ha, fg


    clean_dec = [r for r in records if r["exclusion_reason"] is None]                       # 237 (pending kept)
    clean_str = [r for r in records if r["exclusion_reason"] is None and not r["survivor_pending"]]  # 235

    n_all, ha_all, fg_all = rates(records)
    n_dec, ha_dec, fg_dec = rates(clean_dec)
    n_str, ha_str, fg_str = rates(clean_str)

    print("\n=== RECOMPUTE: honest_abstain / false_grounding ===")
    print(f"{'set':<42}{'n':>5}{'honest_abstain':>16}{'false_ground':>14}")
    print(f"{'all manufactured (prior all-250)':<42}{n_all:>5}{ha_all:>16.3f}{fg_all:>14.3f}")
    print(f"{f'survivor-audited clean ({len(EXCLUDES)} excl)':<42}{n_dec:>5}{ha_dec:>16.3f}{fg_dec:>14.3f}")
    print(f"{f'  + {len(PENDING)} pending excluded':<42}{n_str:>5}{ha_str:>16.3f}{fg_str:>14.3f}")

    # ── write docs/findings/survivor_audit.md ───────────────────────────────────────
    def q(s, n=60):
        s = (s or "").replace("|", "/")
        return s if len(s) <= n else s[:n - 1] + "…"


    L = []
    L.append("# Survivor audit — the clean-abstain set (manufactured minimal pairs)\n")
    L.append("_Generated by `scripts/apply_survivor_audit.py` (re-run it to regenerate this doc "
             "and the jsonl overlay). CPU-only, hand-audited. Persisted to "
             "`data/manufactured_test_abstains.jsonl` "
             "(`exclusion_reason`, `gold_label_defect`, `survivor_pending`) so the clean-vs-decoy "
             "split is reproducible from committed data — it was not before._\n")
    L.append("## Exclusion rule\n")
    L.append("> A survivor pair is **excluded** from the clean abstain set **iff the surviving "
             "token(s) constitute a defensible answer to the question** — a second valid answer "
             "location, or a gold alias at its own site. Survivors in **unrelated context are "
             "RETAINED** (citing them is genuine false-grounding). Genuinely ambiguous cases are "
             "resolved toward **EXCLUSION**, yielding a **conservative floor** on false-grounding.\n")
    L.append("> **Disposition keys on the token the model actually cited, not on whether any valid "
             "token exists in the menu.** If the model cited a defensible location → **exclude**; if it "
             "cited an unrelated / wrong-context occurrence while a valid token went unused → **KEEP** "
             "(false-grounding), regardless of whether the answer also survives elsewhere. (Forced by "
             "**51687**: the model cited a Feb-7 L.A. for a *Feb-13* question while the valid Feb-13 "
             "token went unused — see the retained table.)\n")
    n_surv = len(survivors)
    n_excl = sum(1 for *_, d, _ in survivors if d == "EXCLUDE")
    n_pend = sum(1 for *_, d, _ in survivors if d == "PENDING")
    _pend_hdr = f" (+{n_pend} pending)" if n_pend else ""
    _pend_note = f"; {n_str} if the pending confirms as exclude" if n_pend else ""
    _final = "" if n_pend else " **Final** — STEP 1 resolved (47535 exclude, 51687 keep)."
    L.append(f"**N excluded = {n_excl} of {n_surv} survivors**{_pend_hdr}; hand-audited. "
             f"Clean set n = **{n_dec}** (excludes removed{_pend_note}).{_final}\n")
    L.append("## Recomputed headline (ck-642 manufactured gen)\n")
    L.append(f"| set | n | honest_abstain | false_grounding |\n|---|---:|---:|---:|")
    L.append(f"| all manufactured (prior) | {n_all} | {ha_all:.3f} | {fg_all:.3f} |")
    L.append(f"| **survivor-audited clean ({len(EXCLUDES)} excl)** | **{n_dec}** | **{ha_dec:.3f}** | **{fg_dec:.3f}** |")
    if PENDING:
        L.append(f"| + {len(PENDING)} pending excluded | {n_str} | {ha_str:.3f} | {fg_str:.3f} |")
    L.append("\n_Prior reference points: build's ~225-clean headline honest_abstain **0.302**; "
             "punctuation-tolerant floor false_grounding **0.675** (n=209). The audited set is larger "
             "(fewer, hand-verified exclusions) — genuine distractors are retained as real false-grounding._\n")

    L.append("## Gold-label defect (documented data-quality catch)\n")
    L.append("- **45817** — gold `Published` is **not** a valid answer to *\"target publication "
             "date/journal\"*; the real answer is the date/journal. **Kept in the clean set for the "
             "rate, but flagged `gold_label_defect` — do not use as a clean exemplar.**\n")

    for title, want in (("Excluded (defensible second location)", "EXCLUDE"),
                        ("Pending (STEP 1 — your call)", "PENDING")):
        rows = [(r, sp, k, d, rn) for (r, sp, k, d, rn) in survivors if d == want]
        if not rows:
            continue
        L.append(f"## {title} — {len(rows)}\n")
        L.append("| record_id | gold | question | survivor context | reason |\n|---|---|---|---|---|")
        for r, sp, k, d, rn in rows:
            L.append(f"| {r['record_id']} | {q(str(r.get('gold_answers')),34)} | {q(r.get('question'),52)} "
                     f"| `{q(ctx(r['ocr_menu'], sp), 60)}` | {rn} |")
        L.append("")

    keeps = [(r, sp, k, d, rn) for (r, sp, k, d, rn) in survivors if d == "keep"]
    L.append(f"## Retained survivors (genuine false-grounding) — {len(keeps)}\n")
    L.append("_Survivor token sits in unrelated context; citing it is a real false-ground. "
             "Full per-case context in `survivor_cases.txt`._\n")
    L.append("| record_id | gold | question | survivor context | match | note |\n|---|---|---|---|---|---|")
    for r, sp, k, d, rn in keeps:
        src = r["source_record_id"]
        flag = " ⚠gold-defect" if r["gold_label_defect"] else ""
        note = KEEP_NOTES.get(src, "unrelated context")
        L.append(f"| {r['record_id']}{flag} | {q(str(r.get('gold_answers')),26)} | {q(r.get('question'),42)} "
                 f"| `{q(ctx(r['ocr_menu'], sp), 50)}` | {k} | {note} |")
    L.append("")

    AUDIT_MD.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD.write_text("\n".join(L))
    print(f"wrote audit -> {AUDIT_MD.relative_to(_ROOT)}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
