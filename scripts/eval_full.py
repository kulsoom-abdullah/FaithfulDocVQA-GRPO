"""Full per-checkpoint faithfulness eval for FaithfulDocVQA SFT.

NOTE (era): this is the v1-era full evaluator — it produced the committed 2-field
eval_checkpoint-*.json dumps under data/checkpoints/eval/sft_{A,B}_seed42/. For
the live 3-field contract use gen_persist.py (raw gens) + eval_scorer via
score_val_curve.py; this file is kept as the reproducible producer of the v1
evidence.

For EVERY checkpoint of a run it generates over the shared validation split,
scores with :mod:`scripts.eval_scorer`, and reports the gold-ANSWERABLE and
gold-ABSTAIN partitions **separately** (selection calibration reads answerable;
the SFT-A-vs-B headline reads abstain — they're consulted at different times).

Raw + parsed + scored output is persisted per record per checkpoint, so the
metrics can be re-derived (``--rescore``) WITHOUT regenerating.

Modes:
  (default) full   — every checkpoint × full validation split
  --shakeout       — one checkpoint × the LONGEST-menu / capped validation
                     records (opposite of the smoke's smallest-menu pick): a
                     fast long-sequence / VRAM / tokenization-parity probe.
  --rescore PATH   — re-score a persisted eval JSON without any generation.

Greedy, batch 1, same generation path as the smoke (``eval_runner.generate``).
GPU-only to run; CPU-safe to import.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_runner import generate, load_adapted_model, prompt_parity
from scripts.eval_scorer import _was_capped, score
from scripts.faithful_io import parse_tolerant
from scripts.format_sft import load_split

DEFAULT_RUN_DIR = "data/checkpoints/sft/sft_B_seed42"
DEFAULT_OUT_ROOT = "data/checkpoints/eval"
CAP_MANIFEST = ROOT / "data" / "images_cap_manifest.json"


# ── checkpoint + manifest helpers ─────────────────────────────────────────────

def enumerate_checkpoints(run_dir: str | Path, min_step: int = 0) -> list[Path]:
    """``checkpoint-N`` adapter dirs under a run with step >= ``min_step``, ordered.

    Deliberately explicit (not ``eval_runner.resolve_adapter_dir``, which picks
    the lowest only — that's smoke-specific). ``min_step`` excludes stray low-step
    smoke checkpoints (e.g. a step-6 leftover) that share the run dir. Raises if
    none remain.
    """
    run_dir = Path(run_dir)
    cks = sorted(
        (d for d in run_dir.glob("checkpoint-*")
         if (d / "adapter_config.json").exists() and int(d.name.split("-")[-1]) >= min_step),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not cks:
        raise FileNotFoundError(f"no checkpoint-*/ adapter dirs (step>={min_step}) under {run_dir}")
    return cks


def load_cap_manifest() -> dict:
    return json.loads(CAP_MANIFEST.read_text())


def _fmt(x) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


# ── generation over a record set ──────────────────────────────────────────────

def generate_over(model, processor, records: list[dict], max_new_tokens: int,
                  check_parity: bool = False, log_every: int = 50) -> dict[str, dict]:
    """Generate one output per record. Returns ``{record_id: gen_meta}`` where
    gen_meta has ``raw``/``stop_reason``/``n_new_tokens`` (+ ``parity`` if asked)."""
    out: dict[str, dict] = {}
    t0 = time.time()
    for i, r in enumerate(records, 1):
        meta = generate(model, processor, r, max_new_tokens=max_new_tokens)
        if check_parity:
            meta["parity"] = prompt_parity(processor, r)
        out[r["record_id"]] = meta
        if log_every and (i % log_every == 0 or i == len(records)):
            rate = i / (time.time() - t0)
            print(f"    generated {i}/{len(records)}  ({rate:.2f} rec/s)", flush=True)
    return out


# ── reporting ─────────────────────────────────────────────────────────────────

def print_partition_report(label: str, result: dict, gen: dict | None = None) -> None:
    """Print the ANSWERABLE and ABSTAIN partitions as separate, labeled blocks."""
    ov = result["overall"]
    a, ab, f1, er = ov["answerable"], ov["abstain"], ov["citation_f1"], ov["mean_evidence_reward"]
    print(f"\n=== {label}  (n={ov['n']}: {ov['n_answerable']} answerable / "
          f"{ov['n_abstain']} abstain) ===")
    if gen is not None:
        stops: dict[str, int] = {}
        for m in gen.values():
            stops[m["stop_reason"]] = stops.get(m["stop_reason"], 0) + 1
        print("  generation: " + " / ".join(f"{v} {k}" for k, v in sorted(stops.items()))
              + f" | malformed: {result['overall']['malformed_rate']:.3f}")

    print("  -- ANSWERABLE (selection calibration) --")
    print(f"     valid_citation {_fmt(a['valid_citation_rate'])}   "
          f"fabrication {_fmt(a['fabrication_rate'])}   "
          f"false_abstain {_fmt(a['false_abstain_rate'])}")
    print(f"     uncited_answer {_fmt(a['uncited_answer_rate'])}   "
          f"malformed {_fmt(a['malformed_rate'])}")
    print(f"     citation F1: micro {_fmt(f1['micro_f1'])} (P {_fmt(f1['micro_precision'])} "
          f"R {_fmt(f1['micro_recall'])})  macro {_fmt(f1['macro_f1'])}   "
          f"mean_evidence_reward {_fmt(er['answerable'])}")

    print("  -- ABSTAIN (A-vs-B headline) --")
    print(f"     honest_abstain {_fmt(ab['honest_abstain_rate'])}   "
          f"answered_anyway {_fmt(ab['answered_anyway_rate'])}   "
          f"malformed {_fmt(ab['malformed_rate'])}   "
          f"mean_evidence_reward {_fmt(er['abstain'])}")

    if "capped" in result and result["capped"]["n"]:
        c, u = result["capped"], result["uncapped"]
        print(f"  -- capped slice: capped n={c['n']} / uncapped n={u['n']} "
              "(see persisted JSON for full sliced metrics) --")


def print_cross_checkpoint(rows: list[dict]) -> None:
    """Two compact cross-checkpoint tables: answerable, then abstain."""
    print("\n" + "=" * 78)
    print("CROSS-CHECKPOINT SUMMARY — ANSWERABLE (selection calibration)")
    print("=" * 78)
    print(f"  {'checkpoint':<16}{'valid_cit':>10}{'fabric':>9}{'false_abs':>11}"
          f"{'micro_F1':>10}{'macro_F1':>10}")
    for r in rows:
        a, f1 = r["answerable"], r["citation_f1"]
        print(f"  {r['name']:<16}{_fmt(a['valid_citation_rate']):>10}{_fmt(a['fabrication_rate']):>9}"
              f"{_fmt(a['false_abstain_rate']):>11}{_fmt(f1['micro_f1']):>10}{_fmt(f1['macro_f1']):>10}")
    print("\n" + "=" * 78)
    print("CROSS-CHECKPOINT SUMMARY — ABSTAIN (A-vs-B headline)")
    print("=" * 78)
    print(f"  {'checkpoint':<16}{'honest_abs':>12}{'answered':>10}{'malformed':>11}{'ev_reward':>11}")
    for r in rows:
        ab = r["abstain"]
        print(f"  {r['name']:<16}{_fmt(ab['honest_abstain_rate']):>12}"
              f"{_fmt(ab['answered_anyway_rate']):>10}{_fmt(ab['malformed_rate']):>11}"
              f"{_fmt(r['ev_reward_abstain']):>11}")


# ── persistence ───────────────────────────────────────────────────────────────

def persist(out_path: Path, ckpt: Path, base_id: str, max_new_tokens: int,
            result: dict, gen: dict) -> None:
    """Write {raw, parsed, scored} per record + aggregate metrics to JSON, so a
    later ``--rescore`` never has to regenerate."""
    per_record = result["per_record"]
    for o in per_record:
        m = gen[o["record_id"]]
        o["raw"] = m["raw"]
        o["stop_reason"] = m["stop_reason"]
        o["n_new_tokens"] = m["n_new_tokens"]
        p = parse_tolerant(m["raw"])
        o["parsed"] = {"answer": p.answer, "evidence_ids": p.evidence_ids} if p else None
    payload = {
        "checkpoint": str(ckpt),
        "base": base_id,
        "split": "validation",
        "n": result["overall"]["n"],
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
        "metrics": {k: v for k, v in result.items() if k != "per_record"},
        "records": per_record,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"    persisted -> {out_path}")


# ── modes ─────────────────────────────────────────────────────────────────────

def run_full(run_dir: str, max_new_tokens: int, out_root: str, limit: int | None,
             min_step: int = 0) -> None:
    """Generate + score every checkpoint over the full shared validation split."""
    import gc
    import torch

    records = load_split("validation", "B")          # shared, full split
    if limit:
        records = records[:limit]
    manifest = load_cap_manifest()
    ckpts = enumerate_checkpoints(run_dir, min_step=min_step)
    run_name = Path(run_dir).name
    out_dir = ROOT / out_root / run_name
    print(f"[full] run={run_name}  checkpoints={[c.name for c in ckpts]}  "
          f"records={len(records)}")

    rows: list[dict] = []
    for ckpt in ckpts:
        print(f"\n[full] === {ckpt.name} ===")
        model, proc, resolved = load_adapted_model(ckpt)
        gen = generate_over(model, proc, records, max_new_tokens)
        raw_by_id = {rid: m["raw"] for rid, m in gen.items()}
        result = score(records, raw_by_id, cap_manifest=manifest)
        print_partition_report(ckpt.name, result, gen)
        persist(out_dir / f"eval_{ckpt.name}.json", ckpt, resolved, max_new_tokens, result, gen)
        ov = result["overall"]
        rows.append({"name": ckpt.name, "answerable": ov["answerable"],
                     "abstain": ov["abstain"], "citation_f1": ov["citation_f1"],
                     "ev_reward_abstain": ov["mean_evidence_reward"]["abstain"]})
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print_cross_checkpoint(rows)
    print(f"\n[full] done. per-checkpoint JSON under {out_dir}")


def pick_shakeout_records(manifest: dict, n: int = 15, min_capped: int = 5) -> list[dict]:
    """The ~``n`` validation records with the LARGEST menus, with the largest
    ``min_capped`` capped-image records guaranteed in (opposite of the smoke's
    smallest-menu pick) — a bounded long-sequence + capped-image stress set."""
    by_menu = sorted(load_split("validation", "B"),
                     key=lambda r: len(r["ocr_menu"]), reverse=True)
    capped_by_menu = [r for r in by_menu if _was_capped(manifest, r["doc_id"])]
    picks: list[dict] = []
    ids: set[str] = set()

    def add(r: dict) -> None:
        if r["record_id"] not in ids:
            picks.append(r)
            ids.add(r["record_id"])

    for r in capped_by_menu[:min(min_capped, n)]:    # guarantee capped representation
        add(r)
    for r in by_menu:                                # fill the rest with the longest overall
        if len(picks) >= n:
            break
        add(r)
    return picks


def run_shakeout(run_dir: str, max_new_tokens: int, n: int, which: int | None) -> None:
    """Generate on only the longest/capped records of one checkpoint and print the
    long-sequence behavior PER RECORD: stop reason, output length, parity, parse,
    and wall-time (so full-eval cost can be extrapolated from the worst case)."""
    import statistics as stat

    manifest = load_cap_manifest()
    records = pick_shakeout_records(manifest, n=n)
    ckpts = enumerate_checkpoints(run_dir)
    ckpt = (next((c for c in ckpts if int(c.name.split("-")[-1]) == which), None)
            if which is not None else ckpts[-1])    # default: most-trained
    if ckpt is None:
        raise SystemExit(f"checkpoint-{which} not found among {[c.name for c in ckpts]}")

    n_capped = sum(_was_capped(manifest, r["doc_id"]) for r in records)
    print(f"[shakeout] {ckpt.name} | {len(records)} records ({n_capped} capped) | "
          f"largest menu = {max(len(r['ocr_menu']) for r in records)} entries")

    t_load = time.time()
    model, proc, _ = load_adapted_model(ckpt)
    print(f"[shakeout] model load: {time.time() - t_load:.1f}s\n")

    n_parse = parity_ok = 0
    stops: dict[str, int] = {}
    secs: list[float] = []
    for r in records:
        t0 = time.time()
        m = generate(model, proc, r, max_new_tokens=max_new_tokens)
        dt = time.time() - t0
        secs.append(dt)
        parity = prompt_parity(proc, r)
        parsed = parse_tolerant(m["raw"]) is not None
        n_parse += int(parsed)
        parity_ok += int(parity)
        stops[m["stop_reason"]] = stops.get(m["stop_reason"], 0) + 1
        capped = _was_capped(manifest, r["doc_id"])
        print(f"  id={r['record_id']:<7} menu={len(r['ocr_menu']):<5} capped={str(capped):<5} "
              f"stop={m['stop_reason']:<10} n_new={m['n_new_tokens']:<4} "
              f"parity={'OK' if parity else 'MISMATCH':<8} "
              f"parses={'Y' if parsed else 'N'}  {dt:5.1f}s", flush=True)

    print(f"\n[shakeout] {ckpt.name}: parse {n_parse}/{len(records)} | "
          f"parity {parity_ok}/{len(records)} | stop "
          + " ".join(f"{v} {k}" for k, v in sorted(stops.items())))
    print(f"[shakeout] per-record gen wall-time: min {min(secs):.1f}s  "
          f"median {stat.median(secs):.1f}s  max {max(secs):.1f}s  "
          f"total {sum(secs):.1f}s over {len(secs)} records")
    if parity_ok < len(records):
        print("  *** parity MISMATCH on long inputs -> real train<->gen tensor "
              "divergence; FIX before the full run. ***")


def run_rescore(json_path: str) -> None:
    """Re-derive metrics from a persisted eval JSON — NO generation."""
    data = json.loads(Path(json_path).read_text())
    by_id = {r["record_id"]: r for r in load_split(data.get("split", "validation"), "B")}
    raw_by_id = {rec["record_id"]: rec["raw"] for rec in data["records"]}
    records = [by_id[rid] for rid in raw_by_id if rid in by_id]
    result = score(records, raw_by_id, cap_manifest=load_cap_manifest())
    print_partition_report(f"RESCORE {Path(json_path).name}", result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="full: cap #records (quick test)")
    ap.add_argument("--min-step", type=int, default=0,
                    help="full: skip checkpoints below this step (excludes smoke leftovers)")
    ap.add_argument("--shakeout", action="store_true", help="long-menu/capped probe, one ckpt")
    ap.add_argument("--n-shakeout", type=int, default=15)
    ap.add_argument("--checkpoint", type=int, default=None,
                    help="shakeout: checkpoint step to use (default: highest)")
    ap.add_argument("--rescore", metavar="JSON", default=None,
                    help="re-score a persisted eval JSON without generating")
    a = ap.parse_args()

    if a.rescore:
        run_rescore(a.rescore)
    elif a.shakeout:
        run_shakeout(a.run_dir, a.max_new_tokens, a.n_shakeout, a.checkpoint)
    else:
        run_full(a.run_dir, a.max_new_tokens, a.out_root, a.limit, min_step=a.min_step)


if __name__ == "__main__":
    main()
