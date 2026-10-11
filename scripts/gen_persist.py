"""Generate-only + persist RAW 3-field outputs over a split (val or test).

For each selected source — the trained adapter checkpoints OR the pre-SFT base model
(``--base``) — greedily generate one raw output per record on ``--split`` and persist
``{record_id, raw, stop_reason, n_new_tokens}`` to JSON. NO SCORING (``eval_scorer``
is never imported); the schema matches ``scripts/eval_full.py`` so outputs re-score
offline via ``--rescore``.

Output naming (under ``<out-root>/<run-name>/``):
  - adapter, validation -> ``gen_checkpoint-N.json``          (the original convention)
  - adapter, test       -> ``gen_test_checkpoint-N.json``     (alongside the val ones)
  - base, <split>       -> ``gen_base_<split>.json``

GPU-only to run (4-bit bitsandbytes); CPU-safe to import (heavy deps lazy in eval_runner).

Examples:
  # all checkpoints over validation (the original behaviour)
  python scripts/gen_persist.py --run-dir data/checkpoints/sft_v3/sft_B_seed42
  # one checkpoint over test, 5 records, with parity check (the Modal smoke)
  python scripts/gen_persist.py --run-dir <run> --split test --steps 1284 --limit 5 --check-parity
  # base model over test
  python scripts/gen_persist.py --run-dir <run> --split test --base
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.format_sft import load_split

DEFAULT_OUT_ROOT = "data/checkpoints/eval"


def enumerate_checkpoints(run_dir: str | Path, min_step: int = 0,
                          steps: set[int] | None = None) -> list[Path]:
    """``checkpoint-N`` adapter dirs under a run, ordered by step. ``min_step`` skips
    low-step leftovers; ``steps`` (if given) keeps only those exact step numbers.
    Inlined (not imported from eval_full) so this path never pulls in ``eval_scorer``.
    Raises if none qualify."""
    run_dir = Path(run_dir)
    cks = sorted(
        (d for d in run_dir.glob("checkpoint-*")
         if (d / "adapter_config.json").exists() and int(d.name.split("-")[-1]) >= min_step),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if steps is not None:
        cks = [c for c in cks if int(c.name.split("-")[-1]) in steps]
    if not cks:
        raise FileNotFoundError(
            f"no checkpoint-*/ adapter dirs (min_step={min_step}, steps={steps}) under {run_dir}")
    return cks


def gen_over(model, proc, resolved: str, records: list[dict], max_new_tokens: int,
             split: str, source: str, out_path: Path, check_parity: bool,
             log_every: int = 50) -> dict:
    """Generate one raw output per record with an already-loaded model + persist them.
    Frees the model afterward. If ``check_parity``, runs ``prompt_parity`` on every
    record (train<->gen prompt equality) and records the tally; a mismatch is a real
    divergence — surfaced loudly, and recorded in the payload."""
    import torch

    from scripts.eval_runner import generate, prompt_parity

    rows: list[dict] = []
    stops: dict[str, int] = {}
    par_ok = 0
    t0 = time.time()
    for i, r in enumerate(records, 1):
        if check_parity:
            par_ok += 1 if prompt_parity(proc, r) else 0
        m = generate(model, proc, r, max_new_tokens=max_new_tokens)
        rows.append({"record_id": r["record_id"], "raw": m["raw"],
                     "stop_reason": m["stop_reason"], "n_new_tokens": m["n_new_tokens"]})
        stops[m["stop_reason"]] = stops.get(m["stop_reason"], 0) + 1
        if log_every and (i % log_every == 0 or i == len(records)):
            print(f"    {source}: {i}/{len(records)} ({i / (time.time() - t0):.2f} rec/s)",
                  flush=True)

    parity = None
    if check_parity:
        parity = {"checked": len(records), "ok": par_ok}
        flag = "OK" if par_ok == len(records) else "!! MISMATCH"
        print(f"    [parity] {par_ok}/{len(records)} prompts match training render  {flag}",
              flush=True)

    elapsed = time.time() - t0
    payload = {
        "source": source,                  # checkpoint dir or "base"
        "base": resolved,
        "split": split,
        "n": len(rows),
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False},
        "scored": False,
        "prompt_parity": parity,
        "stop_reason_counts": stops,
        "records": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"    persisted -> {out_path}  ({len(rows)} records, {elapsed:.0f}s, stops={stops})",
          flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"n": len(rows), "elapsed_s": elapsed, "stops": stops, "parity": parity}


def run(run_dir: str, out_root: str, split: str, max_new_tokens: int,
        limit: int | None, min_step: int, steps: set[int] | None,
        base: bool, check_parity: bool, records_file: str | None = None,
        tag: str = "custom") -> None:
    """Generate + persist raw outputs over ``split`` (or a ``records_file`` of arbitrary
    records, e.g. manufactured abstains) for the base model or selected checkpoints."""
    if records_file:                          # arbitrary records — overrides --split
        records = [json.loads(l) for l in open(records_file)]
        split = tag                           # relabel payload + output filename gen_<tag>_<ckpt>.json
    else:
        records = load_split(split, "B")      # full held-out split (val or test)
    if limit:
        records = records[:limit]
    run_name = Path(run_dir).name
    out_dir = ROOT / out_root / run_name
    t0 = time.time()

    if base:
        from scripts.eval_runner import load_base_model
        print(f"[gen] BASE model | split={split} | records={len(records)} -> {out_dir}", flush=True)
        model, proc, resolved = load_base_model()
        gen_over(model, proc, resolved, records, max_new_tokens, split, "base",
                 out_dir / f"gen_base_{split}.json", check_parity)
    else:
        from scripts.eval_runner import load_adapted_model
        ckpts = enumerate_checkpoints(run_dir, min_step=min_step, steps=steps)
        prefix = "" if split == "validation" else f"{split}_"
        print(f"[gen] run={run_name} | split={split} | checkpoints={[c.name for c in ckpts]} "
              f"| records={len(records)} -> {out_dir}", flush=True)
        for ckpt in ckpts:
            print(f"\n[gen] === {ckpt.name} ({split}) ===", flush=True)
            model, proc, resolved = load_adapted_model(ckpt)
            gen_over(model, proc, resolved, records, max_new_tokens, split, str(ckpt),
                     out_dir / f"gen_{prefix}{ckpt.name}.json", check_parity)

    print(f"\n[gen] DONE in {(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="run dir holding checkpoint-*/ (also names the output subdir)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="cap #records (smoke)")
    ap.add_argument("--min-step", type=int, default=0, help="skip checkpoints below this step")
    ap.add_argument("--steps", default=None,
                    help="comma-separated checkpoint steps to include (e.g. 1284 or 214,1284)")
    ap.add_argument("--base", action="store_true",
                    help="generate from the pre-SFT BASE model (no adapter) instead of checkpoints")
    ap.add_argument("--check-parity", action="store_true",
                    help="run prompt_parity per record (train<->gen prompt equality)")
    ap.add_argument("--records-file", default=None,
                    help="jsonl of records to generate over (overrides --split; e.g. manufactured abstains)")
    ap.add_argument("--tag", default="custom",
                    help="output label for --records-file (writes gen_<tag>_<ckpt>.json)")
    a = ap.parse_args()
    steps = {int(s) for s in a.steps.split(",")} if a.steps else None
    run(a.run_dir, a.out_root, a.split, a.max_new_tokens, a.limit, a.min_step,
        steps, a.base, a.check_parity, a.records_file, a.tag)


if __name__ == "__main__":
    main()
