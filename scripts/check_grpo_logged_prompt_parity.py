"""check_grpo_logged_prompt_parity.py — sweep TRL's logged GRPO prompts against the repo render.

For every completions parquet TRL persisted during the two GRPO runs
(``data/modal_pull/run{1,2}/grpo_run{1,2}_seed42/completions/`` — a gitignored
volume-pull mirror; restage before running with
``modal volume get faithfuldocvqa-ckpts outputs/grpo_runs/grpo_run1_seed42 data/modal_pull/run1/grpo_run1_seed42``
and the run2 equivalent — one logged prompt-group per optimizer step, 150 per
run), this script:

  1. takes the unique ``prompt`` string TRL logged — the *decoded prompt tensor*
     the model actually received, decoded with ``skip_special_tokens=True``;
  2. matches it to its source record in the train pool, reconstructed from the
     volume-pulled run config (seed + pool sizes read from ``grpo_run_config.json``,
     asserted identical across the two runs);
  3. re-renders that record through the repo path — ``to_row`` →
     ``processor.apply_chat_template`` → tokenize → decode with
     ``skip_special_tokens=True`` (the same strip TRL applied when logging);
  4. requires byte-identity. Any unmatched, ambiguous, or divergent prompt is a
     hard FAILURE (non-zero exit), never a skipped row.

It also checks whether the two runs logged the identical step-aligned prompt
sequence (they should: same seed → same data order), and reports the number of
distinct prompts so the headline can be stated honestly.

WHAT THIS SWEEP DOES **NOT** VERIFY (stated assumptions, not oversights):

  - Special-token framing and image-token expansion. TRL logs prompts with
    ``skip_special_tokens=True``, so the logged text contains no chat-structure
    tokens and no vision tokens. Byte-identity here attests the *text content*
    end-to-end; the vision-token run length is invisible to it.
  - Image decoding parity. Eval loads images via PIL ``Image.open``
    (``eval_runner.py``); the GRPO path stores a path column cast to HF datasets
    ``Image()`` that TRL decodes internally. These are ASSUMED equivalent; this
    boundary is documented here precisely because no committed check covers it.
  - Prompt→image pairing inside TRL: ``to_row`` passes a bare ``{"type":"image"}``
    placeholder and the image as a separate column; text parity cannot show the
    right image accompanied each prompt.
  - Coverage: TRL persisted one prompt-group of the four per optimizer step
    (~25% of training prompt-groups). The sweep covers everything TRL logged.

Run:  uv run python scripts/check_grpo_logged_prompt_parity.py \
          [--out docs/analysis/grpo_logged_prompt_parity.md]

CPU-only (no model load — processor/tokenizer only, offline from the HF cache).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_Q_RE = re.compile(r"Question: (.*?)\n\nOCR tokens:\n", re.S)
_RUN_DIRS = {run: _ROOT / f"data/modal_pull/run{run}/grpo_run{run}_seed42" for run in (1, 2)}


def load_run_configs() -> dict:
    """Read both committed grpo_run_config.json files; assert the pool-defining
    fields (seed, pool_abstain, pool_answerable) are identical across runs and
    return them. The train-pool reconstruction is anchored to these committed
    values, not to code defaults."""
    cfgs = {run: json.loads((d / "grpo_run_config.json").read_text()) for run, d in _RUN_DIRS.items()}
    keys = ("seed", "pool_abstain", "pool_answerable")
    vals = {k: {cfgs[r][k] for r in cfgs} for k in keys}
    bad = {k: v for k, v in vals.items() if len(v) != 1}
    assert not bad, f"run configs disagree on pool-defining fields: {bad}"
    return {k: cfgs[1][k] for k in keys}


def build_pool(pool_cfg: dict) -> list[dict]:
    """Rebuild the exact seeded train pool both runs drew from (jsonl reads only)."""
    from grpo.train_grpo import GRPORunConfig, build_train_pool

    cfg = GRPORunConfig(seed=pool_cfg["seed"])
    cfg.pool_abstain = pool_cfg["pool_abstain"]
    cfg.pool_answerable = pool_cfg["pool_answerable"]
    pool, _ = build_train_pool(cfg)
    return pool


class Renderer:
    """Repo-side render of a record, decoded exactly the way TRL logged prompts:
    apply_chat_template → tokenize → decode(skip_special_tokens=True). Cached."""

    def __init__(self) -> None:
        from scripts.format_sft import load_processor

        self.proc, self.resolved = load_processor()
        self._cache: dict[str, tuple[str, str]] = {}

    def decoded_and_template(self, rec: dict) -> tuple[str, str]:
        from grpo.train_grpo import to_row

        rid = rec["record_id"]
        if rid not in self._cache:
            templ = self.proc.apply_chat_template(
                to_row(rec)["prompt"], tokenize=False, add_generation_prompt=True)
            ids = self.proc.tokenizer(templ, add_special_tokens=False)["input_ids"]
            self._cache[rid] = (self.proc.tokenizer.decode(ids, skip_special_tokens=True), templ)
        return self._cache[rid]


def match_record(logged: str, by_question: dict[str, list[dict]], renderer: Renderer
                 ) -> tuple[dict | None, str | None]:
    """Match a logged prompt back to its pool record.

    Primary key is the question text; same-question collisions are disambiguated
    by full render equality, then by first-menu-line containment. Returns
    (record, None) on a clean match, (None, reason) on failure, and
    (record, "ambiguous") only if no disambiguator separated the candidates."""
    m = _Q_RE.search(logged)
    if not m:
        return None, "no-question-parse"
    cands = by_question.get(m.group(1))
    if not cands:
        return None, "question-not-in-pool"
    if len(cands) == 1:
        return cands[0], None
    for r in cands:
        if renderer.decoded_and_template(r)[0] == logged:
            return r, None
    for r in cands:
        if r["ocr_menu"]:
            m0 = r["ocr_menu"][0]
            if f"{m0['id']}\t{m0['text']}\t" in logged:
                return r, None
    return cands[0], "ambiguous"


def sweep_run(run: int, by_question: dict[str, list[dict]], renderer: Renderer
              ) -> tuple[dict, dict[int, str], list[str]]:
    """Compare every logged prompt of one run against the repo render.

    Returns (stats, step→logged-prompt map, failure descriptions). Every parquet
    must hold exactly one unique prompt (one G-group); a violation is a failure,
    not a skip."""
    import pandas as pd

    files = sorted(glob.glob(str(_RUN_DIRS[run] / "completions" / "completions_*.parquet")))
    failures: list[str] = []
    step_prompt: dict[int, str] = {}
    n_ok = n_prompts = 0
    matched_ids: set[str] = set()
    for f in files:
        df = pd.read_parquet(f, columns=["step", "prompt"])
        uniq = df["prompt"].unique().tolist()
        steps = df["step"].unique().tolist()
        if len(uniq) != 1 or len(steps) != 1:
            failures.append(f"{Path(f).name}: expected 1 unique prompt/step, got "
                            f"{len(uniq)} prompts over steps {steps}")
            continue
        logged, step = uniq[0], int(steps[0])
        step_prompt[step] = logged
        n_prompts += 1
        rec, err = match_record(logged, by_question, renderer)
        if rec is None or err == "ambiguous":
            failures.append(f"step {step}: record match failed ({err})")
            continue
        matched_ids.add(rec["record_id"])
        decoded, _ = renderer.decoded_and_template(rec)
        if decoded == logged:
            n_ok += 1
        else:
            i = next((k for k in range(min(len(decoded), len(logged)))
                      if decoded[k] != logged[k]), min(len(decoded), len(logged)))
            failures.append(
                f"step {step} ({rec['record_id']}): first divergence at char {i}: "
                f"expected …{decoded[max(0, i - 60):i + 60]!r} "
                f"logged …{logged[max(0, i - 60):i + 60]!r}")
    stats = {"n_files": len(files), "n_prompts": n_prompts, "n_byte_identical": n_ok,
             "n_distinct_records": len(matched_ids)}
    return stats, step_prompt, failures


def write_report(out: Path, pool_cfg: dict, resolved_model: str, stats: dict[int, dict],
                 cross: dict, illustration: tuple[str, str], all_failures: list[str]) -> None:
    """Render the committed markdown report (numbers first, caveats stated)."""
    ok = not all_failures
    total = sum(s["n_prompts"] for s in stats.values())
    total_ok = sum(s["n_byte_identical"] for s in stats.values())
    templ_head, dec_head = illustration
    lines = [
        "# GRPO logged-prompt parity sweep — TRL's decoded prompt tensors vs the repo render",
        "",
        f"Generated by `scripts/check_grpo_logged_prompt_parity.py` on {date.today().isoformat()}. "
        f"Processor: `{resolved_model}` (offline, HF cache). Train pool reconstructed from the "
        f"volume-pulled run configs (seed {pool_cfg['seed']}, pools {pool_cfg['pool_abstain']}/"
        f"{pool_cfg['pool_answerable']}; the two runs' configs agree on these fields — asserted).",
        "",
        "## Result",
        "",
        f"**{total_ok}/{total} logged prompts byte-identical** to the repo render "
        f"(`to_row` → `apply_chat_template` → tokenize → decode with `skip_special_tokens=True`, "
        f"the same strip TRL applies when logging). " + ("PASS." if ok else "**FAILURES — see below.**"),
        "",
        "| run | parquets | prompts checked | byte-identical | distinct source records |",
        "|---|---|---|---|---|",
    ]
    for run, s in sorted(stats.items()):
        lines.append(f"| run{run} | {s['n_files']} | {s['n_prompts']} | "
                     f"{s['n_byte_identical']} | {s['n_distinct_records']} |")
    lines += [
        "",
        "## Honest coverage statement",
        "",
        f"- The two runs logged **step-identical prompt sequences** on their "
        f"{cross['n_common_steps']} common steps ({cross['n_equal']} equal / "
        f"{cross['n_common_steps']}): same seed → same data order. The union of distinct "
        f"logged prompts across both runs is **{cross['n_distinct_union']}** — so "
        f"\"{total}/{total}\" is {cross['n_distinct_union']} distinct prompts, each verified "
        "once per run, not 300 independent prompts.",
        "- TRL persisted **one prompt-group of the four per optimizer step** (~25% of "
        "training prompt-groups). The sweep covers everything TRL logged.",
        "- Byte-identity is **modulo special tokens**: the logged text carries no chat-structure "
        "tokens and no vision tokens (skip-special-tokens decode), so this verifies the text "
        "content end-to-end but cannot attest special-token framing or image-token expansion.",
        "",
        "### Illustration — template (with special tokens) vs the compared decode",
        "",
        "```",
        f"template head: {templ_head!r}",
        f"decoded  head: {dec_head!r}",
        "```",
        "",
        "## Stated assumptions this sweep does NOT verify",
        "",
        "- **Image-token run length / image decoding parity.** Eval loads images via PIL "
        "`Image.open` (`scripts/eval_runner.py`); GRPO passes a path column cast to HF datasets "
        "`Image()` that TRL decodes internally. Assumed equivalent — no committed check covers "
        "this boundary; it is documented here as an assumption. (Distinct from the "
        "PIL-vs-qwen_vl_utils resize note in `docs/findings/sft_pipeline_buildnotes.md` §4.)",
        "- **Prompt→image pairing inside TRL** (bare image placeholder + separate image column): "
        "text parity cannot show the right image accompanied each prompt.",
    ]
    if all_failures:
        lines += ["", "## FAILURES", ""] + [f"- {f}" for f in all_failures]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote report -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=_ROOT / "docs/analysis/grpo_logged_prompt_parity.md",
                    help="markdown report path (default: docs/analysis/grpo_logged_prompt_parity.md)")
    a = ap.parse_args()

    pool_cfg = load_run_configs()
    pool = build_pool(pool_cfg)
    by_question: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_question[r["question"]].append(r)
    renderer = Renderer()
    print(f"processor resolved: {renderer.resolved}")

    stats: dict[int, dict] = {}
    step_prompts: dict[int, dict[int, str]] = {}
    all_failures: list[str] = []
    for run in (1, 2):
        s, sp, fails = sweep_run(run, by_question, renderer)
        stats[run], step_prompts[run] = s, sp
        all_failures += [f"run{run}: {f}" for f in fails]
        print(f"run{run}: {s}" + (f"  FAILURES={len(fails)}" if fails else ""))

    common = sorted(set(step_prompts[1]) & set(step_prompts[2]))
    cross = {
        "n_common_steps": len(common),
        "n_equal": sum(step_prompts[1][s] == step_prompts[2][s] for s in common),
        "n_distinct_union": len(set(step_prompts[1].values()) | set(step_prompts[2].values())),
    }
    print(f"cross-run: {cross}")

    templ_dec = renderer.decoded_and_template(pool[0])
    write_report(a.out, pool_cfg, renderer.resolved, stats, cross,
                 (templ_dec[1][:200], templ_dec[0][:200]), all_failures)

    if all_failures:
        print(f"\nPARITY: FAIL — {len(all_failures)} failure(s)")
        sys.exit(1)
    total = sum(s["n_prompts"] for s in stats.values())
    print(f"\nPARITY: PASS — {total}/{total} logged prompts byte-identical "
          f"({cross['n_distinct_union']} distinct, each verified once per run)")


if __name__ == "__main__":
    main()
