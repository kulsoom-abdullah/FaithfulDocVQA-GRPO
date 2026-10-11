"""Stage 2 TASK B — TRL GRPOTrainer multimodal CONSTRUCTION probe (no training).

Two questions for the Stage-2 report (construction + source inspection only — this
NEVER runs a train step or a generation):

  Q1 IMAGES    does GRPOTrainer.__init__ accept our VLM (ck-642 bf16 + LoRA), the
               Qwen3-VL processor, and a 4-row image-carrying dataset, no error?
  Q2 METADATA  is there a documented mechanism for a custom reward_func to receive
               per-record metadata (extra dataset columns), so
               compute_grpo_reward(record, output, run) can be called?

If either fails we REPORT the gap + workaround options — we do NOT build a workaround.

Run (pod, cu128 venv):  FDV_ADAPTER=/workspace/ck-642 python grpo/probe_stage2b.py
"""
import inspect, json, os, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset, Image as HFImage
import trl, transformers
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForImageTextToText
from peft import PeftModel

from scripts.format_sft import load_processor, MODEL_ID, build_messages
from grpo.reward import compute_grpo_reward

ADAPTER = os.environ.get("FDV_ADAPTER", "/workspace/ck-642")
RUN = 1

def main() -> None:
    print("=== stack ===")
    print(f"trl {trl.__version__} | transformers {transformers.__version__} | "
          f"torch {torch.__version__} | cuda {torch.version.cuda}")

    # ── A) API contract from source (deterministic, no GPU) ───────────────────────
    src = inspect.getsource(trl.trainer.grpo_trainer)
    lines = src.splitlines()

    print("\n=== Q2 metadata routing — how reward_kwargs are assembled (source) ===")
    for i, ln in enumerate(lines):
        if "reward_kwargs" in ln or ("reward_func(" in ln) or ("keys_to_exclude" in ln):
            print(f"  L{i}: {ln.strip()[:130]}")

    print("\n=== image handling refs in trainer source ===")
    for kw in ['"image"', '"images"', "pixel_values", "image_split_sizes", "apply_chat_template"]:
        print(f"    {kw:22} refs={src.count(kw)}")

    cfg0 = GRPOConfig(output_dir="/tmp/probe_out", num_generations=2)
    print("\n=== GRPOConfig defaults relevant to us ===")
    for f in ["remove_unused_columns", "num_generations", "max_prompt_length",
              "max_completion_length", "temperature", "steps_per_generation", "use_vllm"]:
        if hasattr(cfg0, f):
            print(f"    {f} = {getattr(cfg0, f)!r}")

    # ── B) model + processor + 4-row multimodal dataset ───────────────────────────
    print("\n=== loading model + processor (bf16 base + ck-642 LoRA) ===")
    proc, _ = load_processor()
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    print(f"    is_peft_model: {hasattr(model, 'peft_config')}  device: {next(model.parameters()).device}")

    manuf = [json.loads(l) for l in open(ROOT / "data/manufactured_train_abstains.jsonl")][:2]
    master = [json.loads(l) for l in open(ROOT / "data/master.jsonl")]
    answ = [r for r in master if r.get("split") == "train" and not r.get("target_is_abstain")][:2]
    recs = manuf + answ
    REC_BY_ID = {r["record_id"]: r for r in recs}


    def to_row(r):
        """Conversational prompt with a BARE image placeholder; PIL image lives in its
        own 'image' column (TRL VLM convention); record_id is the metadata hook.

        All message `content` is normalized to a typed-parts LIST (system too) so Arrow
        infers one consistent schema — mixing a string content and a list content in the
        same column is what pyarrow rejects ('cannot mix list and non-list')."""
        msgs = build_messages(r, include_target=False, image_ref=None)
        norm = []
        for m in msgs:
            content = m["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            for c in content:
                if c.get("type") == "image":
                    c.pop("image", None)            # {"type":"image"} placeholder only
            norm.append({"role": m["role"], "content": content})
        return {"prompt": norm,
                "image": str((ROOT / r["image_path"]).resolve()),
                "record_id": r["record_id"]}


    ds = Dataset.from_list([to_row(r) for r in recs]).cast_column("image", HFImage())
    print(f"    dataset cols={ds.column_names} n={len(ds)}  image feature={ds.features['image']}")


    # ── C) custom reward routing our metadata → compute_grpo_reward ───────────────
    def reward_fn(prompts, completions, **kwargs):
        """TRL passes every extra dataset column as a kwarg list (one per prompt).
        We recover the full record via record_id and score with compute_grpo_reward."""
        ids = kwargs.get("record_id")
        if ids is None:
            raise RuntimeError("record_id did NOT route to reward — metadata routing FAILS")
        outs = []
        for comp, rid in zip(completions, ids):
            text = comp if isinstance(comp, str) else comp[-1]["content"]
            outs.append(compute_grpo_reward(REC_BY_ID[rid], text, run=RUN))
        return outs


    # ── D) construct trainer (NO training / NO generation) ────────────────────────
    print("\n=== Q1 CONSTRUCTION attempt ===")
    try:
        cfg = GRPOConfig(
            output_dir="/tmp/probe_out",
            num_generations=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            max_completion_length=64,
            remove_unused_columns=False,   # keep record_id/image for the reward fn
            use_vllm=False,
            report_to="none",
            logging_steps=1,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=cfg,
            train_dataset=ds,
            processing_class=proc,
        )
        print("CONSTRUCTION: PASS — GRPOTrainer built (VLM + image dataset + custom reward)")
        print(f"    train_dataset cols after init: {trainer.train_dataset.column_names}")
        print(f"    reward_funcs: {[getattr(f, '__name__', type(f).__name__) for f in trainer.reward_funcs]}")
        keep = [c for c in ("record_id", "image") if c in trainer.train_dataset.column_names]
        print(f"    metadata columns SURVIVED init: {keep}  "
              f"({'record_id present → reward can route' if 'record_id' in keep else 'record_id STRIPPED → routing gap'})")
    except Exception:
        print("CONSTRUCTION: FAIL")
        traceback.print_exc()

    print("\n=== TASK B done ===")


if __name__ == "__main__":
    main()
