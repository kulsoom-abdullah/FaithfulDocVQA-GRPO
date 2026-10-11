"""Score run-4a ck-100 on the frozen test instruments — plan §(e)1 and §(f).

Executes ONLY the pre-registered comparison:
  §(e)1  run-4a-best (ck-100) vs seed ck-1296, honest_abstain on clean-235,
         paired exact two-sided binomial on discordant pairs.
  §(f)   both watch columns: false_abstain on the 679 answerable natural-test rows,
         and natural honest_abstain on the 77 natural-test abstain rows.

Discipline copied from run3_test_results.md, because a number is only comparable if
it is computed the same way as the numbers it is compared against:

  - **Baselines reproduced to the digit first.** The pipeline re-derives seed ck-642
    (68/235), run-2 ck-150 (204/235) and run-3 ck-1296 (212/235) from their committed
    gen files before computing anything new. A mismatch aborts.
  - **clean-235 asserted**, from ``exclusion_reason is None`` on the committed
    manufactured test file — never a stale ``in_clean_236`` flag.
  - **Join coverage asserted** for every generation. An absent record_id joins via
    ``.get()`` to None, classifies as a parse failure, and reports honest_abstain
    0.000 with no exception raised — a wrong answer that looks like a result.

CPU only. Reads committed files; opens no network and no GPU.

Usage:
    python3 scripts/score_run4a_test.py                       # baselines only (self-test)
    python3 scripts/score_run4a_test.py --gen-dir <dir>       # + score ck-100
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_scorer import classify  # noqa: E402
from scripts.format_sft import load_split  # noqa: E402

MANUFACTURED = ROOT / "data/manufactured_test_abstains.jsonl"
EVAL = ROOT / "data/checkpoints/eval"

# committed baselines, honest_abstain on clean-235 (run3_test_results.md)
BASELINES = {
    "seed ck-642": (EVAL / "sft_B_seed42/gen_manufactured_checkpoint-642.json", 68),
    "run-2 ck-150": (EVAL / "grpo_run2_seed42/gen_manufactured_checkpoint-150.json", 204),
    "run-3 ck-1296": (EVAL / "sft_v3_balanced_nodecoy/gen_manufactured_checkpoint-1296.json", 212),
}

# committed natural-test baseline (run3_test_results.md). The manufactured baselines
# above validate only the clean-235 path; the watch columns come off the natural
# split, which needs its own proof or half the pipeline is unvalidated.
NATURAL_BASELINE = (
    EVAL / "sft_v3_balanced_nodecoy/gen_test_checkpoint-1296.json",
    {"honest_abstain": 71, "false_abstain": 78, "valid_citation": 601, "malformed": 0},
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_gen(path: Path) -> dict[str, str | None]:
    """record_id -> raw completion, from a gen_persist payload."""
    payload = json.loads(path.read_text())
    return {str(r["record_id"]): r.get("raw") for r in payload["records"]}


def gen_meta(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return {k: payload.get(k) for k in ("n", "split", "stop_reason_counts", "prompt_parity")}


def exact_binomial_two_sided(b: int, c: int) -> float:
    """Two-sided exact binomial on discordant pairs, p=0.5 (the form used in
    mcnemar_seed_vs_grpo.md; authoritative at small counts).

    Sums the exact rational point masses whose probability does not exceed the
    observed one. Comparison is RELATIVE and done in integer arithmetic: an
    absolute epsilon (obs + 1e-12) silently sweeps in the whole distribution once
    the observed mass is smaller than the epsilon — at b=144, c=0 that turned the
    true 2^-143 = 8.97e-44 into 9.93e-13. Integer weights avoid the issue entirely.
    """
    n = b + c
    if n == 0:
        return 1.0
    weights = [comb(n, k) for k in range(n + 1)]   # exact ints, no float rounding
    total = sum(weights)
    obs = weights[b]
    tail = sum(w for w in weights if w <= obs)     # exact integer comparison
    return min(1.0, tail / total)


def bucket_map(records: dict[str, dict], gens: dict[str, str | None],
               label: str) -> dict[str, str]:
    """Classify every generated record. Asserts join coverage before classifying —
    a missing id must abort, never silently score as a parse failure."""
    missing = [rid for rid in gens if rid not in records]
    if missing:
        raise SystemExit(
            f"[{label}] JOIN FAILURE: {len(missing)} generated record_ids absent from "
            f"the instrument: {missing[:5]}")
    out = {}
    for rid, raw in gens.items():
        out[rid] = classify(records[rid], raw)["bucket"]
    return out


def score_manufactured(gens: dict[str, str | None], recs_by_id: dict[str, dict],
                       clean: set[str], label: str) -> tuple[dict[str, str], int]:
    buckets = bucket_map(recs_by_id, gens, label)
    covered = clean & set(buckets)
    if len(covered) != len(clean):
        raise SystemExit(f"[{label}] clean-set coverage {len(covered)}/{len(clean)}")
    ha = sum(1 for rid in clean if buckets[rid] == "honest_abstain")
    return buckets, ha


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-dir", type=Path, default=None,
                    help="dir holding gen_manufactured_checkpoint-<N>.json and "
                         "gen_test_checkpoint-<N>.json")
    ap.add_argument("--ckpt", default="100",
                    help="checkpoint number in the generation filenames (4a: 100, 4b: 150)")
    ap.add_argument("--vs-gen-dir", type=Path, default=None,
                    help="comparator arm's gen dir for plan §(e)2 (4b vs 4a)")
    ap.add_argument("--vs-ckpt", default=None, help="comparator checkpoint number")
    ap.add_argument("--label", default="run-4a ck-100", help="label for --gen-dir")
    ap.add_argument("--vs-label", default=None, help="label for --vs-gen-dir")
    args = ap.parse_args()

    # ── instruments, asserted from committed files ───────────────────────────────
    man_rows = read_jsonl(MANUFACTURED)
    recs_man = {str(r["record_id"]): r for r in man_rows}
    clean = {str(r["record_id"]) for r in man_rows if r.get("exclusion_reason") is None}
    print(f"[instrument] manufactured test: {len(man_rows)} rows, "
          f"clean-{len(clean)} (exclusion_reason is None), {len(man_rows)-len(clean)} excluded")
    if len(clean) != 235:
        raise SystemExit(f"expected clean-235, derived clean-{len(clean)}")

    nat = load_split("test", "B")
    recs_nat = {str(r["record_id"]): r for r in nat}
    nat_abstain = {rid for rid, r in recs_nat.items() if r.get("target_is_abstain")}
    nat_answerable = set(recs_nat) - nat_abstain
    print(f"[instrument] natural test: {len(nat)} rows = {len(nat_abstain)} abstain "
          f"+ {len(nat_answerable)} answerable")
    if (len(nat), len(nat_abstain), len(nat_answerable)) != (756, 77, 679):
        raise SystemExit("natural test split composition does not match the registered 756/77/679")

    # ── baselines reproduced to the digit, before anything new ───────────────────
    print("\n[validation] reproducing committed baselines on clean-235:")
    seed_buckets: dict[str, str] = {}
    ok = True
    for name, (path, expected) in BASELINES.items():
        if not path.is_file():
            print(f"    {name:<14} gen file MISSING: {path}")
            ok = False
            continue
        buckets, ha = score_manufactured(load_gen(path), recs_man, clean, name)
        match = ha == expected
        ok &= match
        print(f"    {name:<14} honest_abstain {ha}/235 = {ha/235:.3f}  "
              f"(committed {expected}/235) {'OK' if match else 'MISMATCH'}")
        if name == "run-3 ck-1296":
            seed_buckets = buckets
    # natural-test path: the watch columns come off this instrument, so it needs
    # its own baseline proof, not just clean-235's.
    nat_path, nat_expect = NATURAL_BASELINE
    print("\n[validation] reproducing the committed natural-test baseline (run-3 ck-1296):")
    if not nat_path.is_file():
        print(f"    gen file MISSING: {nat_path}")
        ok = False
    else:
        nb0 = bucket_map(recs_nat, load_gen(nat_path), "run-3 ck-1296 natural")
        got = {
            "honest_abstain": sum(1 for r in nat_abstain if nb0[r] == "honest_abstain"),
            "false_abstain": sum(1 for r in nat_answerable if nb0[r] == "false_abstain"),
            "valid_citation": sum(1 for r in nat_answerable if nb0[r] == "valid_citation"),
            "malformed": sum(1 for r in nb0 if nb0[r] == "malformed"),
        }
        dens = {"honest_abstain": 77, "false_abstain": 679,
                "valid_citation": 679, "malformed": 756}
        for k, exp in nat_expect.items():
            match = got[k] == exp
            ok &= match
            print(f"    {k:<16} {got[k]}/{dens[k]} = {got[k]/dens[k]:.4f}  "
                  f"(committed {exp}) {'OK' if match else 'MISMATCH'}")

    if not ok:
        raise SystemExit("[validation] FAILED — baselines did not reproduce; "
                         "no new number is computed from a pipeline that cannot "
                         "reproduce the numbers it compares against.")
    print("    -> pipeline validated on BOTH instruments")

    if args.gen_dir is None:
        print("\nNo --gen-dir given; baseline self-test only.")
        return 0

    # ── run-4a ck-100 ────────────────────────────────────────────────────────────
    man_path = args.gen_dir / f"gen_manufactured_checkpoint-{args.ckpt}.json"
    nat_path = args.gen_dir / f"gen_test_checkpoint-{args.ckpt}.json"
    for p in (man_path, nat_path):
        if not p.is_file():
            raise SystemExit(f"missing generation: {p}")

    print(f"\n[{args.label}] generations:")
    for p in (man_path, nat_path):
        m = gen_meta(p)
        print(f"    {p.name}: n={m['n']} stops={m['stop_reason_counts']} "
              f"parity={m['prompt_parity']}")

    man_gens = load_gen(man_path)
    if len(man_gens) != 250:
        raise SystemExit(f"manufactured gen has {len(man_gens)} records, expected 250")
    ck_buckets, ck_ha = score_manufactured(man_gens, recs_man, clean, args.label)

    print(f"\n=== §(e)1 PRIMARY — honest_abstain on clean-235 ===")
    print(f"    seed  run-3 ck-1296 : {sum(1 for r in clean if seed_buckets[r]=='honest_abstain')}/235"
          f" = {sum(1 for r in clean if seed_buckets[r]=='honest_abstain')/235:.4f}")
    print(f"    {args.label:<20}: {ck_ha}/235 = {ck_ha/235:.4f}")

    both_yes = both_no = b = c = 0
    fixes, breaks = [], []
    for rid in sorted(clean):
        s = seed_buckets[rid] == "honest_abstain"
        k = ck_buckets[rid] == "honest_abstain"
        if s and k:
            both_yes += 1
        elif not s and not k:
            both_no += 1
        elif k and not s:
            b += 1
            fixes.append(rid)
        else:
            c += 1
            breaks.append(rid)
    p = exact_binomial_two_sided(b, c)
    print(f"    paired: both-yes {both_yes}  both-no {both_no}  fixes b={b}  breaks c={c}")
    print(f"    exact two-sided binomial on {b+c} discordant pairs: p = {p:.4g}")
    if breaks:
        print(f"    regressions (abstained correctly at seed, not at {args.label}): {breaks[:10]}")

    # ── watch columns ────────────────────────────────────────────────────────────
    nat_gens = load_gen(nat_path)
    if len(nat_gens) != 756:
        raise SystemExit(f"natural gen has {len(nat_gens)} records, expected 756")
    nb = bucket_map(recs_nat, nat_gens, f"{args.label} natural")

    nat_ha = sum(1 for rid in nat_abstain if nb[rid] == "honest_abstain")
    fa = sum(1 for rid in nat_answerable if nb[rid] == "false_abstain")
    vc = sum(1 for rid in nat_answerable if nb[rid] == "valid_citation")
    mal = sum(1 for rid in nb if nb[rid] == "malformed")

    print(f"\n=== §(f) WATCH COLUMNS (not gates) ===")
    print(f"    over-refusal   false_abstain on n=679 answerable : {fa}/679 = {fa/679:.4f}")
    print(f"      references: run-1 ck-150 0.034 | run-3 ck-1296 0.115 | run-2 ck-150 0.177")
    print(f"    abstention    natural honest_abstain on n=77     : {nat_ha}/77 = {nat_ha/77:.4f}")
    print(f"      references: run-2 ck-150 1.000 | run-3 ck-1296 0.922")
    print(f"    valid_citation on n=679                          : {vc}/679 = {vc/679:.4f}")
    print(f"    malformed over all 756                           : {mal}/756 = {mal/756:.4f}")

    # ── plan §(e)2: this arm vs the comparator arm ──────────────────────────────
    if args.vs_gen_dir is not None:
        vs_ck = args.vs_ckpt or "100"
        vs_label = args.vs_label or str(args.vs_gen_dir.name)
        vm = args.vs_gen_dir / f"gen_manufactured_checkpoint-{vs_ck}.json"
        vn = args.vs_gen_dir / f"gen_test_checkpoint-{vs_ck}.json"
        for q in (vm, vn):
            if not q.is_file():
                raise SystemExit(f"comparator generation missing: {q}")
        vb_man = bucket_map(recs_man, load_gen(vm), vs_label)
        vb_nat = bucket_map(recs_nat, load_gen(vn), f"{vs_label} natural")

        print(f"\n=== §(e)2 — {args.label} vs {vs_label} ===")
        v_ha = sum(1 for r in clean if vb_man[r] == "honest_abstain")
        v_fa = sum(1 for r in nat_answerable if vb_nat[r] == "false_abstain")
        v_na = sum(1 for r in nat_abstain if vb_nat[r] == "honest_abstain")

        # PRIMARY per the plan: the watch column (over-refusal on 679 answerable)
        b = sum(1 for r in nat_answerable
                if vb_nat[r] == "false_abstain" and nb[r] != "false_abstain")
        c = sum(1 for r in nat_answerable
                if vb_nat[r] != "false_abstain" and nb[r] == "false_abstain")
        p = exact_binomial_two_sided(b, c)
        print(f"    PRIMARY — over-refusal, false_abstain on n=679:")
        print(f"      {vs_label:<20}: {v_fa}/679 = {v_fa/679:.4f}")
        print(f"      {args.label:<20}: {fa}/679 = {fa/679:.4f}")
        print(f"      paired: recovered b={b}  newly-refused c={c}  exact two-sided p = {p:.4g}")

        print(f"    ALONGSIDE — honest_abstain on clean-235:")
        bh = sum(1 for r in clean if nb_man_sel(ck_buckets, r) and not nb_man_sel(vb_man, r))
        ch = sum(1 for r in clean if nb_man_sel(vb_man, r) and not nb_man_sel(ck_buckets, r))
        ph = exact_binomial_two_sided(bh, ch)
        print(f"      {vs_label:<20}: {v_ha}/235 = {v_ha/235:.4f}")
        print(f"      {args.label:<20}: {ck_ha}/235 = {ck_ha/235:.4f}")
        print(f"      paired: fixes b={bh}  breaks c={ch}  exact two-sided p = {ph:.4g}")

        print(f"    ALONGSIDE — natural honest_abstain on n=77:")
        print(f"      {vs_label:<20}: {v_na}/77 = {v_na/77:.4f}")
        print(f"      {args.label:<20}: {nat_ha}/77 = {nat_ha/77:.4f}")

    return 0


def nb_man_sel(buckets: dict, rid: str) -> bool:
    """True when this record was scored honest_abstain."""
    return buckets[rid] == "honest_abstain"


if __name__ == "__main__":
    raise SystemExit(main())
