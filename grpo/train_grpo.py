"""GRPO training for FaithfulDocVQA — Qwen3-VL-4B + LoRA, TRL GRPOTrainer, staged reward.

Stage 3 of the GRPO plan. CONTINUES the SFT seed adapter (ck-642) and optimizes the
staged bucket reward (``grpo.reward.compute_grpo_reward``) with ``--run`` selecting the
shaping stage:

  --run 1  BASELINE / control — no abstain shaping (honest_abstain and false_grounding
           both base 0.0). Expected result: valid_citation HOLDS on val while
           honest_abstain stays ~flat (run-1 abstain groups are reward-flat, so
           group-relative advantage on grounding discipline ≈ 0). Boring is correct.
  --run 2  adds abstain shaping (honest_abstain +0.5, false_grounding −0.6).

Design (reuse, don't reimplement):
  - prompt render = the SFT-identical ``format_sft.build_messages`` (image + question +
    OCR menu); normalized to typed-parts lists so pyarrow accepts the conversational
    ``prompt`` column (the banked Stage-2 gotcha).
  - reward = ``grpo.reward.compute_grpo_reward`` via TRL's per-record kwargs routing
    (a ``record_id`` column → module-level record lookup). Confirmed in Stage 2 TASK B.
  - eval hook = greedy generation over a fixed VAL INSTRUMENT + ``eval_scorer.classify``
    bucket rates (honest_abstain / valid_citation / false_grounding …) every N steps.

HARD FIREWALL: the frozen held-out TEST instrument (``manufactured_test_abstains.jsonl``,
the survivor-audited faithfulness test set) is NEVER loaded here. Every training and
val-instrument record is asserted to be split ∈ {train, validation}; a single test-split
record aborts the run. See ``assert_no_test_leak``.

Model differs from SFT: bf16 base + LoRA (NOT 4-bit) — Stage 2 proved bf16 generation on
Ada/cu128 with no cu124 throughput pathology. GPU-only (import heavy deps lazily).

Run (pod, cu128 venv):
  # smoke (20 steps, tiny pool, exercises eval+probe+throughput tripwire):
  FDV_ADAPTER=/workspace/ck-642 python grpo/train_grpo.py --run 1 --smoke \
      --output-dir /workspace/grpo_runs
  # full control run (the committed runs used --max-steps 150; see
  # docs/analysis/mcnemar_seed_vs_grpo.md — final = checkpoint-150):
  FDV_ADAPTER=/workspace/ck-642 python grpo/train_grpo.py --run 1 \
      --output-dir /workspace/grpo_runs --max-steps 150 --eval-every 50 --probe-every 25
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reduce CUDA fragmentation (same rationale as train_sft). Harmless off-GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# light, CPU-only imports (heavy torch/trl/peft/transformers/datasets are lazy in train())
from scripts.format_sft import MODEL_ID, build_messages, load_processor, load_split
from scripts.eval_runner import generate as greedy_generate  # SFT-identical greedy gen
from scripts.eval_scorer import (
    classify,
    VALID_CITATION, FABRICATION, CONTRADICTION, FALSE_GROUNDING,
    HONEST_ABSTAIN, FALSE_ABSTAIN, MALFORMED,
)
from grpo.reward import FALSE_ABSTAIN_BASE, compute_grpo_reward

# ── data files (train + validation ONLY — the test file is the firewall, never read) ──
MASTER = ROOT / "data" / "master.jsonl"
TRAIN_ABSTAINS = ROOT / "data" / "manufactured_train_abstains.jsonl"
VAL_ABSTAINS = ROOT / "data" / "manufactured_val_abstains.jsonl"
FROZEN_TEST_FILE = ROOT / "data" / "manufactured_test_abstains.jsonl"  # NEVER opened here

ABSTAIN_BUCKETS = (HONEST_ABSTAIN, FALSE_GROUNDING, FABRICATION, CONTRADICTION, MALFORMED)
ANSWERABLE_BUCKETS = (VALID_CITATION, FALSE_ABSTAIN, FABRICATION, CONTRADICTION, MALFORMED)


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class GRPORunConfig:
    """All GRPO run knobs. Logged verbatim to ``grpo_run_config.json``.

    LR/beta/loss_type are DECISIONS made for this control run (not previously locked)
    and are flagged as such in the report:
      - loss_type='grpo'  → classic GRPO (TRL 1.8 defaults to 'dapo'); classic is the
        canonical, most-interpretable choice for a control.
      - learning_rate=5e-6, beta=0.04 → small LR + mild KL anchor to ck-642 so the
        control barely drifts while still letting the loop optimize the answerable signal.
    """
    run: int = 1                       # 1 (control) | 2 (abstain shaping) — reward stage
    adapter: str = "/workspace/ck-642" # the SFT seed adapter to CONTINUE
    output_dir: str = "/workspace/grpo_runs"
    seed: int = 42
    # GRPO / generation
    num_generations: int = 8           # G (Stage-2 spec)
    temperature: float = 1.0           # (Stage-2 spec)
    top_p: float = 1.0
    max_completion_length: int = 128   # targets are ~30-60 tok JSON; matches TASK A cap
    beta: float = 0.04                 # KL coef — mild anchor to the ck-642 reference
    learning_rate: float = 5e-6
    loss_type: str = "grpo"            # classic GRPO (not TRL's 'dapo' default)
    # Optimizer-step semantics: 32 completions = 4 prompts × G=8 per optimizer step.
    # bs/accum/generation_batch_size only set HOW those 32 rows execute.
    # generation_batch_size MUST equal num_generations: trl 1.8's VLM scoring path
    # mis-slices the concatenated image tensors when a generation batch mixes
    # different-length prompts (shape-mismatch crash in _get_per_token_logps at
    # run-1 attempt 1, 2026-07-15); gen_batch == G keeps every generation batch
    # single-prompt. This geometry also needs >48 GB (both 48 GB-class attempts
    # OOM'd; Modal L40S usable VRAM varies 44.4–47.4 GiB between containers).
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # 4 × 8 → 32 gen rows / 8 = 4 prompts per step
    generation_batch_size: int = 8     # == num_generations: 1 prompt-group per gen batch
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "constant_with_warmup"
    max_steps: int = 500               # dataclass default; the committed runs overrode to 150
    # data mix (40% abstain / 60% answerable)
    pool_abstain: int = 2000
    pool_answerable: int = 3000
    # run-4 data restrictions (defaults reproduce runs 1-2 byte-for-byte).
    # train_abstains: filename under data/ to draw the abstain half from; run 4 uses the
    # recurs-False rows so the reward is never shaped against decoy-class labels
    # (docs/analysis/run4_analysis_plan.md §b).
    train_abstains: str | None = None
    # val_recurs_false_only: restrict the SCORED val instrument to recurs-False rows, so
    # checkpoint selection is not ranked on suspect gold (plan §g). The qualitative probe
    # still draws its decoy from outside that instrument.
    val_recurs_false_only: bool = False
    # run_name: overrides the run-dir / W&B name. Run 4's arms reuse run-2's REWARD
    # (--run 2 for 4a), so the derived "grpo_run2_seed42" would collide with the real
    # run-2 artifacts already on the ckpts volume. Default None reproduces runs 1-2.
    run_name: str | None = None
    # false_abstain_base: reward for refusing a citable answer. Default 0.0 reproduces
    # runs 1, 2 and 4a byte-for-byte; run 4b prices it negative (-0.2, plan §c) to move
    # the refusal threshold ~5pp of belief. Recorded here (asdict -> grpo_run_config.json)
    # so every run's reward is reconstructable from its own artifacts.
    false_abstain_base: float = FALSE_ABSTAIN_BASE
    # wandb: report_to="wandb" instead of the default "none". Default OFF — the long
    # runs stay on jsonl telemetry (robust unattended; an auth/network stall must not be
    # able to kill a 12h job). Used for the smoke, where a human is watching.
    wandb: bool = False
    # waive_gate3: exclude smoke gate [3] (abstain groups mixed-bucket) from the overall
    # verdict. The registered run-4 waiver (plan amendment 2026-08-15): the gate was
    # calibrated for the ck-642 era (0.289 manufactured abstention); ck-1296 sits at
    # 0.902, so determinism on two high-confidence probe rows is EXPECTED and says
    # nothing about pool-level gradient supply. Recorded in grpo_run_config.json so the
    # waiver is auditable, instead of a shell script silently tolerating a non-zero exit.
    waive_gate3: bool = False
    # eval instrument (fixed, seeded subset of the val split)
    eval_every: int = 50
    eval_abstain: int = 64
    eval_answerable: int = 48
    probe_every: int = 25
    # bookkeeping
    save_steps: int = 50
    logging_steps: int = 1
    smoke: bool = False


def smoke_overrides(cfg: GRPORunConfig) -> GRPORunConfig:
    """Shrink to a 20-step smoke that still exercises the FULL throughput path
    (same batch/G/completion-length as the real run) plus eval + probe hooks."""
    cfg.smoke = True
    cfg.max_steps = 20
    cfg.pool_abstain = 40
    cfg.pool_answerable = 60
    cfg.eval_every = 10
    cfg.eval_abstain = 16
    cfg.eval_answerable = 16
    cfg.probe_every = 10
    cfg.save_steps = 20
    return cfg


# ── frozen-test firewall ──────────────────────────────────────────────────────

def assert_no_test_leak(records: list[dict], where: str) -> None:
    """HARD-assert no held-out test-split record entered ``where``. A test record in
    train/eval invalidates the whole experiment — abort loudly, never silently drop."""
    leaked = [r["record_id"] for r in records if r.get("split") == "test"]
    assert not leaked, (
        f"FROZEN-TEST LEAK in {where}: {len(leaked)} test-split records "
        f"(e.g. {leaked[:5]}). Training/eval must be split ∈ {{train, validation}} only."
    )


# ── data builders ─────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def build_train_pool(cfg: GRPORunConfig) -> tuple[list[dict], dict]:
    """The 40/60 (abstain/answerable) GRPO training pool, seeded + shuffled.

    Abstain half = manufactured TRAIN abstains (gold_localized=False, the grounding-
    discipline signal). Answerable half = TRAIN answerable-citable records
    (gold_localized=True, load_split mode 'A'), the anti-collapse anchor for
    valid_citation. Returns (pool, {record_id: record})."""
    abst_path = (ROOT / "data" / cfg.train_abstains) if cfg.train_abstains else TRAIN_ABSTAINS
    abst_all = _read_jsonl(abst_path)
    print(f"[data] train abstains <- {abst_path.name} ({len(abst_all)} rows)")
    assert_no_test_leak(abst_all, "train abstains")
    ans_all = load_split("train", "A")                 # train + gold_localized==True
    assert_no_test_leak(ans_all, "train answerable")

    rng = random.Random(cfg.seed)
    n_abs = min(cfg.pool_abstain, len(abst_all))
    n_ans = min(cfg.pool_answerable, len(ans_all))
    abst = rng.sample(abst_all, n_abs)
    ans = rng.sample(ans_all, n_ans)
    pool = abst + ans
    rng.shuffle(pool)

    # record_ids must be globally unique so the reward lookup is unambiguous
    rec_by_id = {r["record_id"]: r for r in pool}
    assert len(rec_by_id) == len(pool), "duplicate record_id in train pool — reward lookup unsafe"
    frac = n_abs / len(pool)
    print(f"[data] train pool n={len(pool)}  abstain={n_abs} answerable={n_ans}  "
          f"abstain_frac={frac:.2f}  (target 0.40)")
    return pool, rec_by_id


def build_val_instrument(cfg: GRPORunConfig) -> list[dict]:
    """Fixed, seeded VAL INSTRUMENT for the eval hook: N_abstain manufactured val
    abstains + N_answerable val answerable-citable records. Deterministic (seed) so the
    curve across steps is comparable. Split-asserted (validation only)."""
    abst_all = _read_jsonl(VAL_ABSTAINS)
    if cfg.val_recurs_false_only:                     # plan §g: clean selection basis
        n_before = len(abst_all)
        abst_all = [r for r in abst_all if r.get("answer_recurs_in_menu") is False]
        print(f"[data] val instrument restricted to recurs-False: {n_before} -> {len(abst_all)} rows")
    assert_no_test_leak(abst_all, "val abstains")
    val_all = load_split("validation", "B")            # full val split
    assert_no_test_leak(val_all, "val split")
    ans_all = [r for r in val_all if (not r["target_is_abstain"]) and r["gold_localized"] is True]

    rng = random.Random(cfg.seed + 1)                  # distinct stream from the train pool
    abst = rng.sample(abst_all, min(cfg.eval_abstain, len(abst_all)))
    ans = rng.sample(ans_all, min(cfg.eval_answerable, len(ans_all)))
    inst = [{**r, "_kind": "abstain"} for r in abst] + [{**r, "_kind": "answerable"} for r in ans]
    print(f"[data] val instrument n={len(inst)}  abstain={len(abst)} answerable={len(ans)}")
    return inst


def pick_probe_records(pool: list[dict], instrument: list[dict], seed: int,
                       decoy_pool: list[dict] | None = None) -> list[dict]:
    """4 fixed probe prompts for the every-N-step qualitative dump: 2 abstain (≥1 with
    answer_recurs_in_menu=True decoy if available) + 2 answerable, from the val
    instrument (held-out, so drift is measured off-distribution).

    OBSERVATION ONLY — probes are dumped qualitatively and never scored, so a probe row
    drawn from outside the instrument cannot affect checkpoint selection.

    ``decoy_pool``: source for the decoy probe, used when the scored instrument has no
    decoy rows of its own (run 4's is recurs-False by construction, plan §g). Rows already
    in the instrument are excluded, so the probe never re-shows a scored row. Falls back to
    the instrument's own decoys when omitted — the runs 1-2 behaviour, unchanged."""
    rng = random.Random(seed + 2)
    abst = [r for r in instrument if r["_kind"] == "abstain"]
    ans = [r for r in instrument if r["_kind"] == "answerable"]
    in_instrument = {r["record_id"] for r in instrument}
    decoys = [r for r in abst if r.get("answer_recurs_in_menu")] or [
        {**r, "_kind": "abstain"}                 # stamp _kind: downstream dumps read it
        for r in (decoy_pool or [])
        if r.get("answer_recurs_in_menu") and r["record_id"] not in in_instrument
    ]
    nondecoy = [r for r in abst if not r.get("answer_recurs_in_menu")]
    probes = []
    if decoys:
        probes.append(rng.choice(decoys))
    probes += rng.sample(nondecoy or abst, min(2 - len(probes), len(nondecoy or abst)))
    probes += rng.sample(ans, min(2, len(ans)))
    return probes


# ── conversational-row builder (typed-parts normalized — the pyarrow gotcha) ──

def to_row(record: dict) -> dict:
    """One HF-dataset row: conversational ``prompt`` (bare image placeholder, all
    message content as typed-parts LISTS so Arrow infers one schema), a separate
    ``image`` path column, and the ``record_id`` metadata hook for the reward."""
    msgs = build_messages(record, include_target=False, image_ref=None)
    norm = []
    for m in msgs:
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image":
                c.pop("image", None)                   # {"type":"image"} placeholder only
        norm.append({"role": m["role"], "content": content})
    return {
        "prompt": norm,
        "image": str((ROOT / record["image_path"]).resolve()),
        "record_id": record["record_id"],
    }


def make_dataset(records: list[dict]):
    from datasets import Dataset, Image as HFImage
    ds = Dataset.from_list([to_row(r) for r in records]).cast_column("image", HFImage())
    return ds


# ── reward fn (bound to the run stage; routes record metadata) ────────────────

def _completion_text(comp) -> str:
    """Extract the raw assistant text from a TRL completion (str for text datasets,
    conversational list [{'role','content'}] for VLM/chat datasets — content may itself
    be a typed-parts list)."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list) and comp:
        content = comp[-1].get("content", "") if isinstance(comp[-1], dict) else ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(comp)


def make_reward_fn(run: int, rec_by_id: dict, member_log: "Path | None" = None,
                   false_abstain_base: float = FALSE_ABSTAIN_BASE):
    """Build the TRL reward callable. TRL passes extra dataset columns as per-prompt
    kwarg lists; we recover the full record via ``record_id`` and score with the staged
    ``compute_grpo_reward``. Returns a list[float] (one per completion).

    ``member_log``: when given, EVERY completion's reward is appended to that JSONL —
    one row per group member, not a sample. TRL's own parquet dump is capped by
    ``num_completions_to_print`` (8 of 32 rows/step observed in run 2 = 25%), and the
    scalar ``reward``/``reward_std`` in reward_log.jsonl are group aggregates. Neither
    can produce the abstain-vs-answerable split of ``frac_reward_zero_std`` that
    run4_analysis_plan.md §(amendment) registers as a monitored quantity, nor exact
    per-group variance. This log can: group by (call, record_id) for within-group
    spread, filter on ``is_abstain`` for the split."""
    call = {"n": 0}

    def faithful_reward(prompts, completions, **kwargs):
        ids = kwargs.get("record_id")
        if ids is None:
            raise RuntimeError("record_id did not route to the reward fn — check "
                               "remove_unused_columns=False and the dataset column.")
        lengths = {
            "prompts": len(prompts),
            "completions": len(completions),
            "record_id": len(ids),
        }
        if len(set(lengths.values())) != 1:
            rendered = ", ".join(f"{name}={length}" for name, length in lengths.items())
            raise ValueError(
                f"reward batch lengths must match exactly ({rendered}); refusing to "
                "truncate paired reward inputs."
            )
        rewards = []
        rows = []
        for k, (comp, rid) in enumerate(zip(completions, ids, strict=True)):
            rec = rec_by_id[rid]
            text = _completion_text(comp)
            r = compute_grpo_reward(rec, text, run=run,
                                    false_abstain_base=false_abstain_base)
            rewards.append(r)
            if member_log is not None:
                rows.append({
                    "call": call["n"], "i": k, "record_id": rid,
                    "is_abstain": bool(rec.get("target_is_abstain")),
                    "recurs": rec.get("answer_recurs_in_menu"),
                    "bucket": classify(rec, text)["bucket"],
                    "reward": r,
                })
        if member_log is not None and rows:
            try:
                with member_log.open("a") as fh:
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")
            except Exception as e:  # noqa: BLE001 — logging must never kill a paid run
                print(f"[reward-log] FAILED (continuing): {type(e).__name__}: {e}", flush=True)
        call["n"] += 1
        return rewards

    faithful_reward.__name__ = f"faithful_reward_run{run}"
    return faithful_reward


# ── eval-mode bracket (gc is training-only; just toggle eval + kv-cache) ──────

@contextlib.contextmanager
def _gen_mode(model):
    """Put a mid-training model into generation mode. Gradient checkpointing is gated
    on ``self.training`` in the forward, so eval() alone disables it — we only need to
    flip eval + kv-cache, then restore, leaving the training state untouched."""
    was_training = model.training
    was_cache = getattr(model.config, "use_cache", False)
    model.eval()
    model.config.use_cache = True
    try:
        yield
    finally:
        model.config.use_cache = was_cache
        if was_training:
            model.train()


# ── val-instrument eval (bucket rates via greedy gen + classify) ──────────────

def run_val_eval(model, proc, instrument: list[dict], run: int,
                 false_abstain_base: float = FALSE_ABSTAIN_BASE) -> tuple[dict, list[dict]]:
    """Greedy-generate over the val instrument, classify each, aggregate bucket rates
    split by gold class. Returns (summary, per_record). Never mutates training state
    beyond the eval bracket."""
    import torch  # noqa: F401 (ensures torch present; greedy_generate uses it)
    per = []
    with _gen_mode(model):
        for r in instrument:
            out = greedy_generate(model, proc, r, max_new_tokens=128)
            facts = classify(r, out["raw"])
            rew = compute_grpo_reward(r, out["raw"], run=run,
                                      false_abstain_base=false_abstain_base)
            per.append({
                "record_id": r["record_id"], "kind": r["_kind"],
                "recurs": r.get("answer_recurs_in_menu"),
                "bucket": facts["bucket"], "reward": rew,
                "found_answer_correct": facts["found_answer_correct"],
                "stop_reason": out["stop_reason"], "raw": out["raw"],
            })
    abst = [p for p in per if p["kind"] == "abstain"]
    ans = [p for p in per if p["kind"] == "answerable"]

    def rates(subset, buckets):
        n = len(subset) or 1
        return {b: round(sum(1 for p in subset if p["bucket"] == b) / n, 4) for b in buckets}

    summary = {
        "n": len(per), "n_abstain": len(abst), "n_answerable": len(ans),
        "abstain_rates": rates(abst, ABSTAIN_BUCKETS),
        "answerable_rates": rates(ans, ANSWERABLE_BUCKETS),
        "mean_reward_abstain": round(sum(p["reward"] for p in abst) / (len(abst) or 1), 4),
        "mean_reward_answerable": round(sum(p["reward"] for p in ans) / (len(ans) or 1), 4),
        "malformed_rate": round(sum(1 for p in per if p["bucket"] == MALFORMED) / (len(per) or 1), 4),
    }
    return summary, per


# ── telemetry callback (persist logs + EMA, probe dump, val eval, step timing) ─

def make_callback(model, proc, cfg: GRPORunConfig, out: Path, probes: list[dict]):
    from transformers import TrainerCallback

    class FaithfulTelemetry(TrainerCallback):
        """All artifacts to ``out`` on /workspace: reward_log.jsonl (per-log metrics +
        EMA), val_curve.jsonl + eval_step{N}.jsonl (bucket rates), probe_completions.jsonl
        (qualitative drift), step_times.jsonl (throughput tripwire)."""

        def __init__(self):
            self.ema = None
            self.step_times: list[float] = []
            self._t_begin = None
            self.reward_log = (out / "reward_log.jsonl").open("a")
            self.val_curve = (out / "val_curve.jsonl").open("a")
            self.probe_log = (out / "probe_completions.jsonl").open("a")
            self.step_log = (out / "step_times.jsonl").open("a")

        # per-log metrics (reward mean/std etc.) → persist + EMA
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            rew = logs.get("reward")
            if rew is not None:
                self.ema = rew if self.ema is None else 0.9 * self.ema + 0.1 * rew
            row = {"step": state.global_step, **logs}
            if self.ema is not None:
                row["reward_ema"] = round(self.ema, 4)
            self.reward_log.write(json.dumps(row) + "\n")
            self.reward_log.flush()

        def on_step_begin(self, args, state, control, **kw):
            self._t_begin = time.time()

        def on_step_end(self, args, state, control, **kw):
            # PURE training-step time (begin→end), measured BEFORE the eval/probe hooks
            # below so their overhead never bleeds into the throughput tripwire.
            if self._t_begin is not None:
                dt = time.time() - self._t_begin
                self.step_times.append(dt)
                self.step_log.write(json.dumps({"step": state.global_step, "sec": round(dt, 3)}) + "\n")
                self.step_log.flush()
            step = state.global_step
            # Peak VRAM, SMOKE ONLY. run-4a shipped with no memory instrumentation, so
            # its peak was never measured and card sizing rested on two prior OOMs.
            # Reported after a few real steps, by which point generation and fwd/bwd have
            # both run, so the number is the true peak rather than warmup.
            if cfg.smoke and step in (3, 5, cfg.max_steps):
                import torch as _t
                if _t.cuda.is_available():
                    print(f"[vram@{step}] peak_allocated="
                          f"{_t.cuda.max_memory_allocated()/(1024**3):.2f} GiB  "
                          f"peak_reserved={_t.cuda.max_memory_reserved()/(1024**3):.2f} GiB  "
                          f"device_total="
                          f"{_t.cuda.get_device_properties(0).total_memory/(1024**3):.2f} GiB",
                          flush=True)
            if cfg.probe_every and step % cfg.probe_every == 0:
                self._dump_probes(step)
            if cfg.eval_every and step % cfg.eval_every == 0:
                self._run_eval(step)

        def _dump_probes(self, step):
            try:
                with _gen_mode(model):
                    for r in probes:
                        out_ = greedy_generate(model, proc, r, max_new_tokens=128)
                        facts = classify(r, out_["raw"])
                        self.probe_log.write(json.dumps({
                            "step": step, "record_id": r["record_id"], "kind": r["_kind"],
                            "recurs": r.get("answer_recurs_in_menu"),
                            "bucket": facts["bucket"],
                            "reward": compute_grpo_reward(
                                r, out_["raw"], run=cfg.run,
                                false_abstain_base=cfg.false_abstain_base),
                            "stop_reason": out_["stop_reason"], "raw": out_["raw"],
                        }) + "\n")
                self.probe_log.flush()
            except Exception as e:  # noqa: BLE001 — a probe dump must not kill an overnight run
                print(f"[probe@{step}] FAILED (continuing): {type(e).__name__}: {e}", flush=True)

        def _run_eval(self, step):
            try:
                t0 = time.time()
                summary, per = run_val_eval(model, proc, INSTRUMENT, cfg.run,
                                            false_abstain_base=cfg.false_abstain_base)
                summary = {"step": step, "eval_sec": round(time.time() - t0, 1), **summary}
                self.val_curve.write(json.dumps(summary) + "\n")
                self.val_curve.flush()
                (out / f"eval_step{step}.jsonl").write_text(
                    "\n".join(json.dumps(p) for p in per) + "\n")
                ha = summary["abstain_rates"].get(HONEST_ABSTAIN)
                fg = summary["abstain_rates"].get(FALSE_GROUNDING)
                vc = summary["answerable_rates"].get(VALID_CITATION)
                print(f"[val@{step}] honest_abstain={ha} false_grounding={fg} "
                      f"valid_citation={vc} malformed={summary['malformed_rate']} "
                      f"({summary['eval_sec']}s)", flush=True)
            except Exception as e:  # noqa: BLE001 — eval failure logs loudly, run continues
                print(f"[val@{step}] FAILED (continuing): {type(e).__name__}: {e}", flush=True)

    # INSTRUMENT is closed over by _run_eval; set as attribute-free module local
    return FaithfulTelemetry


# ── model load (bf16 base + ck-642 LoRA, trainable) ───────────────────────────

def load_policy(cfg: GRPORunConfig):
    """Load the base VLM in bf16 and attach the ck-642 LoRA as a TRAINABLE adapter
    (continue-training, not a fresh adapter). Vision stays frozen (ck-642's LoRA is
    language-QKVO only). NO 4-bit — Stage 2 proved bf16 on Ada/cu128."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError("train_grpo needs a CUDA GPU; none found.")
    if not (Path(cfg.adapter) / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter_config.json in {cfg.adapter}")

    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, cfg.adapter, is_trainable=True)
    # sanity: adapter is trainable, vision frozen
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_vis = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and any(s in n for s in ("visual", "vision")))
    assert n_train > 0, "adapter loaded non-trainable — pass is_trainable=True"
    assert n_vis == 0, f"{n_vis} vision params trainable — vision must stay frozen"
    print(f"[model] bf16 base + {cfg.adapter} LoRA | trainable params={n_train:,} vision_trainable=0")
    return model


# ── build trainer ─────────────────────────────────────────────────────────────

INSTRUMENT: list[dict] = []  # set in train(); closed over by the callback's _run_eval


def build_trainer(cfg: GRPORunConfig, out: Path):
    from trl import GRPOConfig, GRPOTrainer

    global INSTRUMENT
    proc, _ = load_processor()
    pool, rec_by_id = build_train_pool(cfg)
    INSTRUMENT = build_val_instrument(cfg)
    probes = pick_probe_records(pool, INSTRUMENT, cfg.seed,
                                decoy_pool=_read_jsonl(VAL_ABSTAINS))
    ds = make_dataset(pool)
    model = load_policy(cfg)

    gcfg = GRPOConfig(
        output_dir=str(out),
        run_name=cfg.run_name or f"grpo_run{cfg.run}_seed{cfg.seed}",
        seed=cfg.seed,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        loss_type=cfg.loss_type,
        num_generations=cfg.num_generations,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_completion_length=cfg.max_completion_length,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        generation_batch_size=cfg.generation_batch_size,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        gradient_checkpointing=True,
        bf16=True,
        use_vllm=False,
        remove_unused_columns=False,     # keep record_id/image for the reward fn
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        log_completions=True,
        num_completions_to_print=4,
        report_to=("wandb" if cfg.wandb else "none"),  # jsonl by default (robust
                                         # unattended); --wandb for the watched smoke
    )
    reward_fn = make_reward_fn(cfg.run, rec_by_id, member_log=out / "reward_members.jsonl",
                               false_abstain_base=cfg.false_abstain_base)
    cb_cls = make_callback(model, proc, cfg, out, probes)
    cb = cb_cls()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=gcfg,
        train_dataset=ds,
        processing_class=proc,
        callbacks=[cb],
    )
    return trainer, proc, model, probes, cb


# ── throughput tripwire (same logic as Stage-2 TASK A) ────────────────────────

def throughput_verdict(step_times: list[float]) -> tuple[bool, str]:
    """(ok, message). Fires on a monotone climb or a >1.8× second-half/first-half ratio
    — the cu124 training pathology signature. Needs ≥6 steps to judge."""
    if len(step_times) < 6:
        return True, f"only {len(step_times)} step deltas — too few to judge (treat as inconclusive-OK)"
    mono = all(step_times[i] < step_times[i + 1] for i in range(len(step_times) - 1))
    h = len(step_times) // 2
    ratio = (sum(step_times[h:]) / len(step_times[h:])) / (sum(step_times[:h]) / len(step_times[:h]))
    patho = mono or ratio > 1.8
    tag = "!! PATHOLOGY (climbing s/it) — STACK FAIL" if patho else "FLAT/NOISY — Ada+cu128 train OK"
    return (not patho), f"{tag} | monotonic={mono} ratio={ratio:.2f} n={len(step_times)}"


# ── entry points ──────────────────────────────────────────────────────────────

def train(cfg: GRPORunConfig) -> str:
    import torch
    from transformers import set_seed
    set_seed(cfg.seed)

    _rn = cfg.run_name or f"grpo_run{cfg.run}_seed{cfg.seed}"
    out = Path(cfg.output_dir) / f"{_rn}{'_smoke' if cfg.smoke else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "grpo_run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    print(f"[start] GRPO run={cfg.run} smoke={cfg.smoke} -> {out}")
    print(f"[start] torch {torch.__version__} cuda {torch.version.cuda} "
          f"{torch.cuda.get_device_name(0)}")

    # our telemetry instance comes back from build_trainer — HF appends its own
    # default callbacks (ProgressCallback et al.) AFTER user callbacks, so
    # trainer.callback_handler.callbacks[-1] is NOT ours.
    trainer, proc, model, probes, cb = build_trainer(cfg, out)

    oom = False
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        oom = True
        print(f"[FATAL] CUDA OOM: {e}", flush=True)
    finally:
        try:
            model.save_pretrained(str(out / "final_adapter"))
            print(f"[done] adapter saved -> {out / 'final_adapter'}")
        except Exception as e:  # noqa: BLE001 — never let a save error bury the throughput report
            print(f"[warn] final adapter save failed: {type(e).__name__}: {e}", flush=True)

    ok, msg = throughput_verdict(cb.step_times)
    print("\n=== THROUGHPUT — cu124-pathology tripwire (TRAINING path) ===")
    print(f"  per-step seconds (first 12): {[round(t,1) for t in cb.step_times[:12]]}")
    print(f"  VERDICT: {msg}")

    if cfg.smoke:
        _smoke_gates(cfg, out, cb, model, proc, probes, oom, ok)
    return str(out)


def _smoke_gates(cfg, out, cb, model, proc, probes, oom, throughput_ok) -> None:
    """Print the 4 smoke gates + overall PASS/FAIL and set the process exit code.
    Gates: (1) no OOM, (2) reward mean/std populated, (3) abstain groups mixed-bucket,
    (4) s/it flat. A FAIL means: stop the pod; do not keep debugging on paid GPU time."""
    # gate 2: reward logging populated
    reward_rows = [json.loads(l) for l in (out / "reward_log.jsonl").open()] \
        if (out / "reward_log.jsonl").exists() else []
    has_reward = any(("reward" in r and r["reward"] is not None) for r in reward_rows)
    has_std = any((r.get("reward_std") is not None) for r in reward_rows)

    # gate 3: abstain groups mixed-bucket — explicit G=8 sample on 2 abstain probes
    mixed = False
    detail = []
    try:
        import torch
        abst_probes = [p for p in probes if p["_kind"] == "abstain"][:2]
        with _gen_mode(model):
            for r in abst_probes:
                from PIL import Image
                from scripts.eval_runner import render_prompt, _eos_ids
                prompt = render_prompt(proc, r)
                img = Image.open(ROOT / r["image_path"])
                inp = proc(text=[prompt], images=[img], return_tensors="pt").to(
                    next(model.parameters()).device)
                eos = _eos_ids(proc)
                pad = proc.tokenizer.pad_token_id or (eos[0] if eos else None)
                plen = inp["input_ids"].shape[1]
                with torch.inference_mode():
                    gen = model.generate(**inp, do_sample=True, temperature=1.0, top_p=1.0,
                                         num_return_sequences=8, max_new_tokens=128,
                                         eos_token_id=eos or None, pad_token_id=pad)
                buckets = Counter()
                for j in range(gen.shape[0]):
                    raw = proc.batch_decode([gen[j][plen:]], skip_special_tokens=True)[0].strip()
                    buckets[classify(r, raw)["bucket"]] += 1
                detail.append((r["record_id"], dict(buckets)))
                if len(buckets) > 1:
                    mixed = True
    except Exception as e:  # noqa: BLE001
        detail.append(("mixed-bucket check errored", str(e)))

    print("\n" + "=" * 68)
    print("SMOKE GATES (fail => STOP THE POD, write up — no debugging on paid GPU time)")
    print("=" * 68)
    g1 = not oom
    g2 = has_reward and has_std
    g3 = mixed
    g4 = throughput_ok
    print(f"  [1] no OOM ................... {'PASS' if g1 else 'FAIL'}")
    print(f"  [2] reward mean/std logged ... {'PASS' if g2 else 'FAIL'}  "
          f"(reward={has_reward} reward_std={has_std}, {len(reward_rows)} log rows)")
    g3_note = "  <- WAIVED (plan amendment 2026-08-15)" if (not g3 and cfg.waive_gate3) else ""
    print(f"  [3] abstain groups mixed .... {'PASS' if g3 else 'FAIL'}{g3_note}  {detail}")
    print(f"  [4] s/it flat ............... {'PASS' if g4 else 'FAIL'}")
    overall = g1 and g2 and g4 and (g3 or cfg.waive_gate3)
    print("-" * 68)
    print(f"  SMOKE: {'PASS — clear to launch run-1 detached' if overall else 'FAIL — STOP THE POD'}")
    sys.exit(0 if overall else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=int, choices=[1, 2], required=True,
                    help="reward stage: 1=control (no abstain shaping), 2=abstain shaping")
    ap.add_argument("--adapter", default=os.environ.get("FDV_ADAPTER", "/workspace/ck-642"))
    ap.add_argument("--output-dir", default="/workspace/grpo_runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--probe-every", type=int, default=None)
    # batch-geometry overrides: execution chunking per card class (48 GB needs
    # bs 2×16; 80 GB runs bs 4×8). Optimizer-step semantics must stay 32 rows =
    # 4 prompts × G=8 — keep (bs × accum) == 32 and gen-batch == num_generations.
    ap.add_argument("--per-device-bs", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--gen-batch", type=int, default=None)
    ap.add_argument("--run-name", default=None,
                    help="override the run-dir/W&B name; run 4 MUST set this "
                         "(--run 2 would otherwise collide with the real run-2 artifacts)")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="checkpoint cadence; run 4 uses 10-15 (plan §d) — without this "
                         "the default 50 yields only 3 checkpoints over 150 steps")
    ap.add_argument("--train-abstains", default=None,
                    help="filename under data/ for the abstain half (run 4: "
                         "manufactured_train_abstains_nodecoy.jsonl)")
    ap.add_argument("--pool-abstain", type=int, default=None)
    ap.add_argument("--pool-answerable", type=int, default=None)
    ap.add_argument("--val-recurs-false-only", action="store_true",
                    help="restrict the SCORED val instrument to recurs-False rows (plan §g)")
    ap.add_argument("--false-abstain-base", type=float, default=None,
                    help="reward for refusing a citable answer. Default 0.0 reproduces "
                         "runs 1/2/4a byte-for-byte; run 4b uses -0.2 (plan §c). Recorded "
                         "in grpo_run_config.json so the reward is reconstructable")
    ap.add_argument("--wandb", action="store_true",
                    help="report_to=wandb (default off: long runs use jsonl telemetry so "
                         "an auth/network stall cannot kill a 12h job)")
    ap.add_argument("--waive-gate3", action="store_true",
                    help="exclude smoke gate [3] from the verdict — the registered run-4 "
                         "waiver (plan amendment 2026-08-15)")
    ap.add_argument("--print-config", action="store_true",
                    help="resolve the config EXACTLY as a run would (same override order, "
                         "smoke_overrides included), build the val instrument, print both "
                         "as JSON, and exit. No GPU, no model, no training. Exists so a "
                         "launcher can assert the RESOLVED config instead of trusting the "
                         "argument string: run 4b's first smoke silently ran 150 steps on "
                         "an n=32 instrument because --max-steps re-overrode smoke_overrides.")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    cfg = GRPORunConfig(run=a.run, adapter=a.adapter, output_dir=a.output_dir, seed=a.seed)
    if a.smoke:
        cfg = smoke_overrides(cfg)
    if a.max_steps is not None:
        cfg.max_steps = a.max_steps
    if a.eval_every is not None:
        cfg.eval_every = a.eval_every
    if a.probe_every is not None:
        cfg.probe_every = a.probe_every
    if a.run_name is not None:
        cfg.run_name = a.run_name
    if a.save_steps is not None:
        cfg.save_steps = a.save_steps
    if a.false_abstain_base is not None:
        cfg.false_abstain_base = a.false_abstain_base
    cfg.wandb = a.wandb
    cfg.waive_gate3 = a.waive_gate3
    # smoke_overrides() runs BEFORE the explicit overrides below, so any of these flags
    # passed alongside --smoke silently reverts the corresponding shrink. Run 4b's first
    # smoke was built as "$TRAIN_ARGS --smoke" and therefore ran 150 steps on the full
    # pool while keeping the shrunken n=32 eval instrument — neither a smoke nor a usable
    # run, and ~4.5 GPU-hours were spent before it was noticed. Combining them is still
    # allowed (a longer smoke is legitimate); it is no longer allowed to be silent.
    if a.smoke:
        _shadowed = [n for n, v in (("--max-steps", a.max_steps),
                                    ("--eval-every", a.eval_every),
                                    ("--probe-every", a.probe_every),
                                    ("--save-steps", a.save_steps),
                                    ("--pool-abstain", a.pool_abstain),
                                    ("--pool-answerable", a.pool_answerable))
                     if v is not None]
        if _shadowed:
            print(f"[warn] --smoke given with {' '.join(_shadowed)}: these are applied "
                  f"AFTER smoke_overrides() and REVERT its shrink. The run will not have "
                  f"smoke shape. Drop them for a real smoke.", flush=True)
    if a.train_abstains is not None:
        cfg.train_abstains = a.train_abstains
    if a.pool_abstain is not None:
        cfg.pool_abstain = a.pool_abstain
    if a.pool_answerable is not None:
        cfg.pool_answerable = a.pool_answerable
    if a.val_recurs_false_only:
        cfg.val_recurs_false_only = True
    if a.per_device_bs is not None:
        cfg.per_device_train_batch_size = a.per_device_bs
    if a.grad_accum is not None:
        cfg.gradient_accumulation_steps = a.grad_accum
    if a.gen_batch is not None:
        cfg.generation_batch_size = a.gen_batch
    assert cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 32, (
        "optimizer-step semantics violated: bs × accum must equal 32 "
        f"(got {cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps})")
    if a.print_config:
        import hashlib
        _inst = build_val_instrument(cfg)
        _ids = sorted(str(r["record_id"]) for r in _inst)
        _payload = {
            "resolved_config": asdict(cfg),
            "val_instrument": {
                "n": len(_inst),
                "n_abstain": sum(1 for r in _inst if r["_kind"] == "abstain"),
                "n_answerable": sum(1 for r in _inst if r["_kind"] == "answerable"),
                "record_ids_sha256": hashlib.sha256("\n".join(_ids).encode()).hexdigest(),
            },
        }
        print("PRINT_CONFIG_JSON_BEGIN")
        print(json.dumps(_payload, indent=2, sort_keys=True))
        print("PRINT_CONFIG_JSON_END")
        return
    train(cfg)


if __name__ == "__main__":
    main()
