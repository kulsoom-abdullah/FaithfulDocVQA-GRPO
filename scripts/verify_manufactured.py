"""verify_manufactured.py — sanity-check the ck-642 honest_abstain 0.805 -> 0.302 drop.

Run with:  uv run python scripts/verify_manufactured.py

Pure CPU. Reuses scripts.eval_scorer.classify (the model is NOT loaded here). It
re-classifies the ALREADY-GENERATED raw outputs for ck-642 over (a) the natural
test split and (b) the 250 manufactured minimal-pair abstains, and cross-checks the
0.805 -> 0.302 honest_abstain finding from three angles.

  STAGE 0  Schema probe — prints one manufactured record, one gen entry, one master
           twin, and ASSERTS the joins/fields the checks rely on (fails loud if wrong).
  CHECK 1  Free tripwires from the EXISTING manufactured gen (no new generation):
           found_answer accuracy, malformed/fabrication/contradiction rates,
           honest_abstain and false_grounding on manufactured. PLUS a paired
           found_answer check (natural twin -> manufactured): the image is untouched,
           so a big move would signal a format break, not a real behaviour change.
  CHECK 2  doc_type slice: per type n, honest_abstain, false_grounding, and the
           PAIRED flip rate (of twins that were valid_citation on the natural menu,
           what % flip to false_grounding once the evidence token is dropped).
  CHECK 3  Dumps up to 12 false-grounded cases to verify_check3_cases.txt for human
           review: dropped gold token, what the model cited instead, model
           found_answer, and an auto-flag for positional-id reuse.

WHAT DIFFERS FROM THE ORIGINAL DRAFT (resolved against the frozen data):
  * The manufactured/natural gens were pulled from the Modal `faithfuldocvqa-ckpts`
    volume to /private/tmp (see paths below), and are nested payloads
    ``{..., "records": [{record_id, raw, ...}]}`` — not flat ``{rid: raw}`` maps.
  * A manufactured record's twin is joined by ``source_record_id`` (its own
    ``record_id`` is ``<orig>_mfab`` and is NOT in master).
  * ``doc_type`` is a LIST (e.g. ``["free_text", "others"]``), so it is normalised
    to a hashable string label before it is used as a slice key.
  * No decoy flag survives in the committed data (``localizer_score`` is null,
    ``gold_localized`` uniformly False). The 225-clean / 25-decoy split was set by
    an uncommitted build and is NOT recoverable by a field lookup, so CHECK 1/2
    report over ALL 250 manufactured records. The DECOY-INDEPENDENT headline is the
    paired flip rate in CHECK 2, which reproduces the "~72% false-ground" finding
    without needing the split.
"""

from __future__ import annotations

import json
import sys
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure the repo root is importable when run as `python scripts/verify_manufactured.py`
# (same convention as scripts/gen_persist.py).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# eval_scorer.classify(record, raw_output) -> facts dict with:
#   bucket, found_answer, found_answer_correct, cited_ids, fabricated_ids,
#   gold_class, gold_localized, doc_type, record_id
from scripts.eval_scorer import classify
from scripts.eval_scorer import (
    HONEST_ABSTAIN, FALSE_GROUNDING, VALID_CITATION, FABRICATION,
    CONTRADICTION, MALFORMED,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────────
# Both input gens are committed (force-added eval evidence under
# data/checkpoints/eval/sft_B_seed42/); the defaults point at the in-repo copies.
# Canonical originals live on the Modal `faithfuldocvqa-ckpts` volume. Override
# via --gen-manufactured / --gen-test (or FDV_GEN_MANUFACTURED / FDV_GEN_TEST).
# Dated note (2026-08-09): the docstring's "225-clean/25-decoy split not
# recoverable" claim was true at write time; the survivor audit
# (apply_survivor_audit.py) has since persisted a principled clean/excluded
# annotation into data/manufactured_test_abstains.jsonl.
ROOT = Path(__file__).resolve().parents[1]

_DEF_GEN_MANUFACTURED = os.environ.get(
    "FDV_GEN_MANUFACTURED",
    str(ROOT / "data/checkpoints/eval/sft_B_seed42/gen_manufactured_checkpoint-642.json"))
_DEF_GEN_TEST = os.environ.get(
    "FDV_GEN_TEST",
    str(ROOT / "data/checkpoints/eval/sft_B_seed42/gen_test_checkpoint-642.json"))



def load_jsonl(p: Path) -> dict[str, dict]:
    """Read a records jsonl into ``{record_id: record}``."""
    d: dict[str, dict] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            d[r["record_id"]] = r
    return d


def load_gen(p: Path) -> dict[str, str]:
    """Return ``{record_id: raw_str}`` from a gen_persist payload.

    Accepts the nested payload ``{..., "records": [{record_id, raw, ...}]}`` (the
    real gen_persist shape) as well as a flat ``{rid: raw}`` map or a bare list of
    ``{record_id, raw}`` entries, so the probe survives either convention.
    """
    obj = json.loads(Path(p).read_text())
    out: dict[str, str] = {}
    if isinstance(obj, dict) and "records" in obj:          # gen_persist payload
        for v in obj["records"]:
            out[v["record_id"]] = v.get("raw")
    elif isinstance(obj, dict):                             # flat {rid: raw|{raw}}
        for rid, v in obj.items():
            out[rid] = v if isinstance(v, str) else v.get("raw")
    elif isinstance(obj, list):                             # [ {record_id, raw} ]
        for v in obj:
            out[v["record_id"]] = v if isinstance(v, str) else v.get("raw")
    else:
        raise TypeError(f"unrecognized gen shape in {p}: {type(obj)}")
    return out


def id2text(menu: list[dict]) -> dict[int, str]:
    """Map an OCR menu to ``{token_id: text}``."""
    return {tok["id"]: tok.get("text", "") for tok in menu}


def dt_key(doc_type: Any) -> str:
    """Normalise the list-valued ``doc_type`` to a hashable slice label."""
    if isinstance(doc_type, (list, tuple)):
        return "+".join(map(str, doc_type)) if doc_type else "<none>"
    return str(doc_type)


def is_decoy(rec: dict) -> bool:
    """True if a decoy marker is present. None survive in the committed data — the
    225-clean / 25-decoy split came from an uncommitted build — so this is always
    False here and CHECK 1/2 report over all 250 (see module docstring)."""
    if "is_decoy" in rec:
        return bool(rec["is_decoy"])
    if rec.get("category"):
        return "decoy" in str(rec["category"]).lower()
    return False



def main() -> None:
    _ap = argparse.ArgumentParser(description="Sanity-check the ck-642 honest_abstain 0.805->0.302 drop (CPU).")
    _ap.add_argument("--master", type=Path, default=ROOT / "data" / "master.jsonl")
    _ap.add_argument("--manufactured", type=Path, default=ROOT / "data" / "manufactured_test_abstains.jsonl")
    _ap.add_argument("--gen-manufactured", type=Path, default=Path(_DEF_GEN_MANUFACTURED),
                     help="raw gen over the manufactured set (env: FDV_GEN_MANUFACTURED)")
    _ap.add_argument("--gen-test", type=Path, default=Path(_DEF_GEN_TEST),
                     help="raw gen over the natural test split (env: FDV_GEN_TEST)")
    _ap.add_argument("--n-check3", type=int, default=12, help="how many false-grounded cases to dump")
    _ap.add_argument("--check3-out", type=Path, default=ROOT / "verify_check3_cases.txt")
    _args = _ap.parse_args()

    MASTER            = _args.master
    MANUFACTURED      = _args.manufactured
    GEN_MANUFACTURED  = _args.gen_manufactured
    GEN_TEST_NATURAL  = _args.gen_test
    N_CHECK3          = _args.n_check3
    CHECK3_OUT        = _args.check3_out

    for _label, _p in (("gen-manufactured", GEN_MANUFACTURED), ("gen-test", GEN_TEST_NATURAL)):
        if not _p.exists():
            sys.exit(f"missing {_label} gen: {_p}\n  pull it from the Modal ckpts volume "
                     f"(see header) or pass --{_label} / set the env var.")

    # ── STAGE 0 — schema probe ─────────────────────────────────────────────────────
    print("=" * 72)
    print("STAGE 0  SCHEMA PROBE")
    print("=" * 72)
    master  = load_jsonl(MASTER)
    manuf   = load_jsonl(MANUFACTURED)
    gen_m   = load_gen(GEN_MANUFACTURED)
    gen_nat = load_gen(GEN_TEST_NATURAL)

    sample_rid = next(iter(manuf))
    sample_src = manuf[sample_rid].get("source_record_id")
    print(f"\n-- manufactured record [{sample_rid}] keys: {sorted(manuf[sample_rid])}")
    print(json.dumps({k: manuf[sample_rid][k] for k in
                      ("record_id", "source_record_id", "target_is_abstain", "gold_localized",
                       "cited_ids", "dropped_token_ids", "doc_type", "gold_answers")
                      if k in manuf[sample_rid]}, indent=2, default=str)[:1000])
    print(f"\n-- manufactured gen entry [{sample_rid}] raw:")
    print(repr(gen_m.get(sample_rid))[:300])
    if sample_src in master:
        print(f"\n-- master twin [source_record_id={sample_src}] keys: {sorted(master[sample_src])}")
        print(json.dumps({k: master[sample_src][k] for k in
                          ("record_id", "doc_type", "target_is_abstain", "gold_localized",
                           "cited_ids", "gold_answers")
                          if k in master[sample_src]}, indent=2, default=str)[:800])
        print(f"\n-- natural gen entry [source_record_id={sample_src}] raw:")
        print(repr(gen_nat.get(sample_src))[:300])

    # Assertions (fail loud with a readable message)
    have_twin = sum(1 for r in manuf if manuf[r].get("source_record_id") in master)
    print(f"\nmanufactured records with a master twin (by source_record_id): {have_twin}/{len(manuf)}")
    if have_twin == 0:
        sys.exit("A1 FAILED: no manufactured source_record_id is in master — the twin link is "
                 "not source_record_id. Paste one manufactured record so we find the pointer field.")
    miss_gen = [r for r in manuf if r not in gen_m]
    if miss_gen:
        print(f"WARNING: {len(miss_gen)} manufactured records have no gen output (e.g. {miss_gen[:3]})")
    if not any("gold_answers" in manuf[r] for r in manuf):
        print("A2 WARNING: manufactured records lack gold_answers -> found_answer accuracy "
              "will be meaningless. Check the build.")
    if not any(is_decoy(manuf[r]) for r in manuf):
        print("A3 NOTE: no decoy flag in the data -> reporting over ALL 250 manufactured "
              "records (can't split the 225 clean from 25 decoys; that split lived in an "
              "uncommitted build). The all-250 honest_abstain will sit just below the "
              "225-clean 0.302 headline; the DECOY-INDEPENDENT check is CHECK 2's flip%.")

    # ── classify everything once ───────────────────────────────────────────────────
    clean = {r: rec for r, rec in manuf.items() if not is_decoy(rec)}
    print(f"\nusing {len(clean)} manufactured records "
          f"({len(manuf) - len(clean)} excluded by a decoy flag)")

    facts_m: dict[str, dict] = {r: classify(clean[r], gen_m.get(r)) for r in clean}

    # Twin (natural) facts, keyed by the MANUFACTURED record_id so the paired logic
    # lines up, but computed from the original answerable record + its natural-test gen.
    facts_nat: dict[str, dict] = {}
    for r in clean:
        src = clean[r].get("source_record_id")
        if src in master and src in gen_nat:
            facts_nat[r] = classify(master[src], gen_nat[src])


    def rate(facts: dict[str, dict], bucket: str) -> float:
        """Fraction of ``facts`` in ``bucket`` (nan if empty)."""
        n = len(facts)
        return (sum(1 for f in facts.values() if f["bucket"] == bucket) / n) if n else float("nan")


    # ── CHECK 1 — free tripwires ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHECK 1  TRIPWIRES (from the existing manufactured gen — no new generation)")
    print("=" * 72)
    n = len(facts_m)
    fa = [f["found_answer_correct"] for f in facts_m.values() if f["found_answer_correct"] is not None]
    print(f"n manufactured (all)           : {n}")
    print(f"honest_abstain  (225-clean=0.302): {rate(facts_m, HONEST_ABSTAIN):.3f}")
    print(f"false_grounding                : {rate(facts_m, FALSE_GROUNDING):.3f}")
    print(f"found_answer acc (expect ~0.91): {sum(fa) / len(fa):.3f}   <-- KEY: image untouched, "
          f"should hold. Collapse => format break.")
    print(f"malformed_rate  (expect ~0)    : {rate(facts_m, MALFORMED):.3f}")
    print(f"fabrication_rate(expect ~0)    : {rate(facts_m, FABRICATION):.3f}")
    print(f"contradiction_rate             : {rate(facts_m, CONTRADICTION):.3f}")
    print(f"valid_citation                 : {rate(facts_m, VALID_CITATION):.3f}")

    # paired found_answer: did reading the page change when only the menu changed?
    paired = [(facts_nat[r]["found_answer_correct"], facts_m[r]["found_answer_correct"])
              for r in facts_m if r in facts_nat]
    flips = sum(1 for a, b in paired if a != b)
    print(f"\npaired found_answer (natural twin -> manufactured), n={len(paired)}:")
    print(f"  changed correctness: {flips}/{len(paired)} = {flips / max(len(paired), 1):.3f}   "
          f"(near 0 = image reading stable => not a format break)")

    # ── CHECK 2 — doc_type slice + paired flip ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHECK 2  BY doc_type  (is the collapse everywhere, or one type?)")
    print("=" * 72)
    by_type: dict[str, list[str]] = defaultdict(list)
    for r, f in facts_m.items():
        by_type[dt_key(f.get("doc_type"))].append(r)

    print(f"{'doc_type':<22}{'n':>5}{'honest_abstain':>16}{'false_ground':>14}{'flip%':>8}")
    for dt, rids in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        sub = {r: facts_m[r] for r in rids}
        # flip: originally valid_citation on the natural menu, now false_grounding
        twins = [r for r in rids if r in facts_nat]
        orig_vc = [r for r in twins if facts_nat[r]["bucket"] == VALID_CITATION]
        flip = sum(1 for r in orig_vc if facts_m[r]["bucket"] == FALSE_GROUNDING)
        flip_pct = flip / len(orig_vc) if orig_vc else float("nan")
        print(f"{dt:<22}{len(rids):>5}{rate(sub, HONEST_ABSTAIN):>16.3f}"
              f"{rate(sub, FALSE_GROUNDING):>14.3f}{flip_pct:>8.3f}")

    # overall paired flip — the decoy-INDEPENDENT headline (handoff: 218 cited -> 72% flip)
    orig_vc_all = [r for r in facts_m if r in facts_nat and facts_nat[r]["bucket"] == VALID_CITATION]
    flip_all = sum(1 for r in orig_vc_all if facts_m[r]["bucket"] == FALSE_GROUNDING)
    abst_all = sum(1 for r in orig_vc_all if facts_m[r]["bucket"] == HONEST_ABSTAIN)
    print(f"\nOVERALL paired: of {len(orig_vc_all)} records ck-642 grounded (valid_citation) on "
          f"the NATURAL menu,")
    print(f"  false-ground when evidence dropped: {flip_all}/{len(orig_vc_all)} = "
          f"{flip_all / max(len(orig_vc_all), 1):.3f}   (handoff headline ~0.72)")
    print(f"  honest-abstain instead            : {abst_all}/{len(orig_vc_all)} = "
          f"{abst_all / max(len(orig_vc_all), 1):.3f}")
    print("\nREAD: if form/table/layout EACH sit near ~0.65-0.75 flip independently, the "
          "finding is real and mechanism-level. If it concentrates in one type, narrower story.")

    # ── CHECK 3 — what did it cite instead? (human-review dump) ─────────────────────
    print("\n" + "=" * 72)
    print(f"CHECK 3  dumping up to {N_CHECK3} false-grounded cases -> {CHECK3_OUT.name}")
    print("=" * 72)
    fg = [r for r, f in facts_m.items() if f["bucket"] == FALSE_GROUNDING]
    lines: list[str] = []
    for r in fg[:N_CHECK3]:
        rec_m, f_m = clean[r], facts_m[r]
        m_txt = id2text(rec_m["ocr_menu"])
        cited_m = [(i, m_txt.get(i, "<<not in menu>>")) for i in f_m["cited_ids"]]
        src = rec_m.get("source_record_id")
        # dropped gold token(s): the original gold cite, looked up in the ORIGINAL menu
        dropped: list[tuple[int, str]] = []
        orig_gold_ids: list[int] = []
        if src in master:
            o_txt = id2text(master[src]["ocr_menu"])
            orig_gold_ids = list(master[src].get("cited_ids", []))
            dropped = [(i, o_txt.get(i, "?")) for i in orig_gold_ids]
        # positional reuse: did the model cite an integer id that WAS gold originally?
        positional = sorted(set(f_m["cited_ids"]) & set(orig_gold_ids))
        lines += [
            f"--- {r}  (source={src})  doc_type={dt_key(f_m.get('doc_type'))} ------------",
            f"  gold_answers      : {rec_m.get('gold_answers')}",
            f"  model found_answer: {f_m['found_answer']!r}   (correct={f_m['found_answer_correct']})",
            f"  DROPPED gold token: {dropped}",
            f"  model CITED instead: {cited_m}",
            f"  positional-id reuse (cited an id that used to be gold): {positional or 'no'}",
            "",
        ]
    CHECK3_OUT.write_text("\n".join(lines))
    print(f"wrote {min(len(fg), N_CHECK3)} cases (of {len(fg)} false-grounded total). "
          f"Inspect {CHECK3_OUT.name} for binning: neighbor / positional / garbage.")
    print("\nDONE.")


if __name__ == "__main__":
    main()
