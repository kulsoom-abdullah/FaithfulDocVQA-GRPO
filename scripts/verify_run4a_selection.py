"""Reproduce and verify the run-4a checkpoint selection end to end.

Regenerates every number in `docs/analysis/run4a_checkpoint_selection.md` from the
run directory's own artifacts, and re-runs the integrity checks that document
claims. Deliberately does NOT trust `val_curve.jsonl`: the curve is recomputed from
the per-row `eval_step*.jsonl` dumps and diffed against the committed summary.

CPU only. No GPU, no generation, no network.

Checks performed:
  [1] adapter integrity  - sha256 + size of every adapter, zero-byte sweep,
                           byte-identity of final_adapter and the last checkpoint
  [2] instrument identity - same record_ids in the same order at every step
  [3] join coverage      - every eval row joins the registered instrument file,
                           keyed on the instrument's own `kind` field
  [4] v2 property        - all abstain rows are recurs-False (no 9/64 caveat)
  [5] curve recompute    - per-row rates vs committed val_curve.jsonl
  [6] selection rule     - plan (g): max honest_abstain, then min false_abstain,
                           then earliest step
  [7] tie depth          - per-row bucket diff across the tied checkpoints
  [8] frac_reward_zero_std - regrouped from reward_members.jsonl, split by side,
                           cross-validated against the trainer's logged mean

Usage:
    python3 scripts/verify_run4a_selection.py data/checkpoints/grpo_run4a_seed42
    python3 scripts/verify_run4a_selection.py <run_dir> --instrument <path>

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_INSTRUMENT = Path("data/grpo_val_instrument_v2_recursfalse.jsonl")
ADAPTER_FILE = "adapter_model.safetensors"


def sha256_file(path: Path) -> str:
    """Stream a file through sha256 and return the hex digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file into a list of dicts."""
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def discover_steps(run_dir: Path) -> list[int]:
    """Steps that have a per-row eval dump, ascending."""
    steps = []
    for p in run_dir.glob("eval_step*.jsonl"):
        try:
            steps.append(int(p.stem.removeprefix("eval_step")))
        except ValueError:
            continue
    return sorted(steps)


def check_adapters(run_dir: Path) -> tuple[bool, list[str]]:
    """Hash every adapter; verify sizes, distinctness and the final-adapter alias."""
    lines: list[str] = []
    ckpts = sorted(
        (p for p in run_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    targets = [*ckpts, run_dir / "final_adapter"]

    digests: dict[str, str] = {}
    sizes: set[int] = set()
    missing: list[str] = []
    for d in targets:
        f = d / ADAPTER_FILE
        if not f.is_file():
            missing.append(d.name)
            continue
        digests[d.name] = sha256_file(f)
        sizes.add(f.stat().st_size)

    zero_byte = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*")
                 if p.is_file() and p.stat().st_size == 0]

    distinct = len(set(digests.values()))
    lines.append(f"    adapters hashed: {len(digests)} (missing: {missing or 'none'})")
    lines.append(f"    distinct sha256 values: {distinct}")
    lines.append(f"    file sizes present: {sorted(sizes)}")
    lines.append(f"    zero-byte files in run dir: {len(zero_byte)} {zero_byte[:5]}")

    final = digests.get("final_adapter")
    last = digests.get(ckpts[-1].name) if ckpts else None
    alias = final is not None and final == last
    lines.append(f"    final_adapter == {ckpts[-1].name if ckpts else 'n/a'}: {alias}"
                 f" ({(final or '')[:12]}…)")

    ok = (not missing and not zero_byte and len(sizes) == 1
          and distinct == len(digests) - 1 and alias)
    return ok, lines


def side_counts(rows: list[dict], kind: str) -> tuple[Counter, int]:
    """Bucket counts and N for one side of the instrument."""
    side = [r for r in rows if r["kind"] == kind]
    return Counter(r["bucket"] for r in side), len(side)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--instrument", type=Path, default=DEFAULT_INSTRUMENT)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    steps = discover_steps(run_dir)
    if not steps:
        print(f"FATAL: no eval_step*.jsonl under {run_dir}", file=sys.stderr)
        return 2

    per_step = {s: read_jsonl(run_dir / f"eval_step{s}.jsonl") for s in steps}
    ok = True

    print(f"run dir   : {run_dir}")
    print(f"steps     : {len(steps)} -> {steps}")

    # [1] adapters
    print("\n[1] adapter integrity")
    a_ok, a_lines = check_adapters(run_dir)
    print("\n".join(a_lines))
    print(f"    => {'PASS' if a_ok else 'FAIL'}")
    ok &= a_ok

    # [2] instrument identity across steps
    print("\n[2] instrument identity across steps")
    ref_ids = [r["record_id"] for r in per_step[steps[0]]]
    ident = all([r["record_id"] for r in per_step[s]] == ref_ids for s in steps)
    print(f"    {len(ref_ids)} rows, {len(set(ref_ids))} unique; identical order at "
          f"all {len(steps)} steps: {ident}")
    print(f"    => {'PASS' if ident else 'FAIL'}")
    ok &= ident

    # [3] join coverage against the registered instrument
    print("\n[3] join coverage vs registered instrument")
    if args.instrument.is_file():
        inst = {r["record_id"]: r for r in read_jsonl(args.instrument)}
        print(f"    instrument: {args.instrument}")
        print(f"    sha256    : {sha256_file(args.instrument)}")
        print(f"    composition: {dict(Counter(v['kind'] for v in inst.values()))}")
        join_ok = True
        for s in steps:
            miss = [r["record_id"] for r in per_step[s] if r["record_id"] not in inst]
            # key on the instrument's OWN kind field; a .get() on an absent field
            # silently yields None and fabricates a plausible wrong number.
            mism = [r["record_id"] for r in per_step[s]
                    if r["record_id"] in inst and inst[r["record_id"]]["kind"] != r["kind"]]
            if miss or mism:
                join_ok = False
                print(f"    !! step {s}: missing={len(miss)} kind-mismatch={len(mism)}")
        print(f"    {len(ref_ids)}/{len(inst)} joined at every step, 0 mismatches: {join_ok}")
        print(f"    => {'PASS' if join_ok else 'FAIL'}")
        ok &= join_ok
    else:
        print(f"    !! instrument file not found: {args.instrument} — check SKIPPED")
        ok = False

    # [4] v2 recurs-False property
    print("\n[4] v2 property (abstain rows are recurs-False)")
    abst = [r for r in per_step[steps[0]] if r["kind"] == "abstain"]
    rec_true = [r["record_id"] for r in abst if r.get("recurs") is True]
    v2_ok = not rec_true
    print(f"    abstain rows={len(abst)}  recurs-True={len(rec_true)} (v2 requires 0)")
    print(f"    => {'PASS' if v2_ok else 'FAIL'}")
    ok &= v2_ok

    # [5] recompute the curve
    print("\n[5] recomputed curve vs committed val_curve.jsonl")
    committed = {r["step"]: r for r in read_jsonl(run_dir / "val_curve.jsonl")}
    curve: dict[int, dict] = {}
    curve_ok = True
    header = ("| step | honest_abstain | = n/{na} | false_grounding | false_abstain "
              "| = n/{nq} | valid_citation | malformed | mean_reward_abstain |")
    rows_md: list[str] = []
    for s in steps:
        rows = per_step[s]
        ca, n_a = side_counts(rows, "abstain")
        cq, n_q = side_counts(rows, "answerable")
        ha, fg = ca["honest_abstain"], ca["false_grounding"]
        fa, vc = cq["false_abstain"], cq["valid_citation"]
        mal = sum(1 for r in rows if r["bucket"] == "malformed")
        mra = sum(r["reward"] for r in rows if r["kind"] == "abstain") / n_a
        curve[s] = {"ha": ha, "n_a": n_a, "fa": fa, "n_q": n_q,
                    "ha_rate": round(ha / n_a, 4), "fa_rate": round(fa / n_q, 4),
                    "total_reward": sum(r["reward"] for r in rows)}
        c = committed.get(s)
        if c is not None:
            match = (curve[s]["ha_rate"] == c["abstain_rates"]["honest_abstain"]
                     and curve[s]["fa_rate"] == c["answerable_rates"]["false_abstain"]
                     and n_a == c["n_abstain"] and n_q == c["n_answerable"])
            if not match:
                curve_ok = False
                print(f"    !! step {s}: MISMATCH vs committed curve")
        rows_md.append(
            f"| {s} | {ha/n_a:.4f} | {ha}/{n_a} | {fg/n_a:.4f} | {fa/n_q:.4f} | "
            f"{fa}/{n_q} | {vc/n_q:.4f} | {mal} | {mra:.4f} |")
    n_a0, n_q0 = curve[steps[0]]["n_a"], curve[steps[0]]["n_q"]
    print("    " + header.format(na=n_a0, nq=n_q0))
    for line in rows_md:
        print("    " + line)
    print(f"    all {len(steps)} steps match committed curve: {curve_ok}")
    print(f"    => {'PASS' if curve_ok else 'FAIL'}")
    ok &= curve_ok

    # [6] selection rule, plan (g)
    print("\n[6] selection rule — run4_analysis_plan.md (g)")
    best_ha = max(v["ha_rate"] for v in curve.values())
    t1 = [s for s in steps if curve[s]["ha_rate"] == best_ha]
    best_fa = min(curve[s]["fa_rate"] for s in t1)
    t2 = [s for s in t1 if curve[s]["fa_rate"] == best_fa]
    winner = min(t2)
    print(f"    key1 max honest_abstain = {best_ha} -> {t1}")
    print(f"    key2 min false_abstain  = {best_fa} -> {t2}")
    print(f"    key3 earliest step      -> SELECT ck-{winner}")

    # [7] how deep the tie goes
    print("\n[7] tie depth — per-row bucket diff across tied checkpoints")
    base = {r["record_id"]: r["bucket"] for r in per_step[t2[0]]}
    tie_rows_identical = True
    for s in t2[1:]:
        cur = {r["record_id"]: r["bucket"] for r in per_step[s]}
        # An absent record_id is a real defect, not a silent no-diff: report it as
        # ABSENT rather than letting .get() paper over it or KeyError kill the report.
        diffs = [k for k in base if k not in cur or base[k] != cur[k]]
        absent = [k for k in base if k not in cur]
        if diffs:
            tie_rows_identical = False
        print(f"    ck-{t2[0]} vs ck-{s}: {len(diffs)} row(s) differ {diffs[:5]}"
              + (f"  [{len(absent)} ABSENT]" if absent else ""))
    print(f"    tied checkpoints are row-for-row identical: {tie_rows_identical}")
    for s in t2:
        print(f"    ck-{s}: total reward over {len(ref_ids)} rows = "
              f"{curve[s]['total_reward']:.4f}")

    # [8] frac_reward_zero_std, split by side
    print("\n[8] frac_reward_zero_std (plan amendment 2026-08-15)")
    members_path = run_dir / "reward_members.jsonl"
    if members_path.is_file():
        members = read_jsonl(members_path)
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for m in members:
            groups[(m["call"], m["record_id"])].append(m)
        gsizes = sorted({len(v) for v in groups.values()})
        print(f"    members={len(members)}  groups={len(groups)}  group sizes={gsizes}")

        def frac(sel) -> tuple[int, int, float]:
            g = [v for v in groups.values() if sel(v[0])]
            z = sum(1 for v in g if statistics.pstdev([m["reward"] for m in v]) == 0.0)
            return z, len(g), (z / len(g) if g else 0.0)

        for label, sel in (("abstain", lambda m: m["is_abstain"]),
                           ("answerable", lambda m: not m["is_abstain"]),
                           ("aggregate", lambda m: True)):
            z, n, f = frac(sel)
            print(f"    {label:<11}: {z}/{n} = {f:.4f}")
        agg = frac(lambda m: True)[2]

        log = read_jsonl(run_dir / "reward_log.jsonl")
        logged = [r["frac_reward_zero_std"] for r in log if "frac_reward_zero_std" in r]
        rstd = [r["reward_std"] for r in log if "reward_std" in r]
        if logged:
            lm = sum(logged) / len(logged)
            agree = abs(lm - agg) < 5e-4
            print(f"    trainer-logged mean over {len(logged)} steps: {lm:.4f} "
                  f"(regrouped {agg:.4f}) -> {'AGREE' if agree else 'DISAGREE'}")
            print(f"    trainer-logged mean reward_std: {sum(rstd)/len(rstd):.4f}")
            ok &= agree
    else:
        print(f"    !! {members_path} not found — check SKIPPED")
        ok = False

    print(f"\n=== OVERALL: {'PASS' if ok else 'FAIL'} — selected ck-{winner} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
