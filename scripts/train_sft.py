"""SFT training for FaithfulDocVQA v1 — Qwen3-VL-4B + QLoRA, completion-only loss.

SFT-A (control: answerable only) and SFT-B (treatment: answerable + ocr_miss
abstains) share EVERY hyperparameter; they differ ONLY in the data filter
(``--mode``). See CLAUDE.md hard rules 4 (completion-only loss) and 5 (A/B differ
only by data).

GPU vs CPU split
----------------
The pieces that DON'T need a GPU run on a laptop and are exercised now:
  - ``confirm_target_modules`` : load the model graph (empty weights, no download
    of the 8 GB checkpoint) and CONFIRM q_proj/k_proj/v_proj/o_proj live on the
    language tower — never guess target modules.
  - ``build_lora_config`` + ``apply_lora`` : the LoRA spec and trainable-param /
    vision-frozen verification.
  - ``tokenize_example`` : the EXPLICIT completion-only label mask (prompt -> -100,
    completion -> token ids). Verified by ``tests/test_loss_mask.py`` on CPU.

The pieces that need the GPU box (4-bit bitsandbytes, the Trainer loop) are
written here but import their heavy deps lazily; run them on RunPod.

Run the CPU inspection now:
    uv run python scripts/train_sft.py --inspect --mode B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Reduce CUDA fragmentation (the OOM trace showed ~4 GB reserved-but-unallocated).
# Set before torch initializes CUDA. Harmless off-GPU.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.format_sft import (
    FALLBACK_MODEL_ID,
    MODEL_ID,
    build_messages,
    load_processor,
    load_split,
)
from scripts.faithful_io import build_target

def build_train_records(cfg) -> tuple[list[dict], dict]:
    """Assemble the training records and the launch manifest. CPU-only, testable.

    Default path (extra_abstains == ""): identical to the documented recipe —
    load_split("train", mode), untouched. With --extra-abstains, merge the extra
    abstain jsonl into the TRAIN set behind three hard gates:
      1. every extra record has split == "train" (test/validation leak aborts);
      2. no extra record's doc_id appears in the master validation or test splits;
      3. the frozen test files are never opened here (only the given path is read).
    Returns (records, manifest) where manifest records n per source, class counts,
    abstain fraction, and sha256 over the sorted record_id list.
    """
    import hashlib

    records = load_split("train", cfg.mode)
    n_master = len(records)
    n_extra = 0
    extra_sha = None
    if cfg.extra_abstains:
        p = ROOT / cfg.extra_abstains
        raw = p.read_bytes()
        extra_sha = hashlib.sha256(raw).hexdigest()
        extra = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        leaked = [r["record_id"] for r in extra if r.get("split") != "train"]
        assert not leaked, (
            f"TRAIN-ONLY GATE: {len(leaked)} extra record(s) with split!='train' "
            f"(e.g. {leaked[:5]}) in {p.name} — aborting.")
        heldout_docs = {str(r["doc_id"])
                       for r in load_split("validation", "B") + load_split("test", "B")}
        doc_leak = sorted({str(r["doc_id"]) for r in extra} & heldout_docs)
        assert not doc_leak, (
            f"DOC-DISJOINT GATE: {len(doc_leak)} extra doc_id(s) also in "
            f"validation/test (e.g. {doc_leak[:5]}) — aborting.")
        seen = {r["record_id"] for r in records}
        dup = [r["record_id"] for r in extra if r["record_id"] in seen]
        assert not dup, f"record_id collision with master train rows: {dup[:5]}"
        records = records + extra
        n_extra = len(extra)

    n_abstain = sum(1 for r in records if r.get("target_is_abstain"))
    ids = sorted(r["record_id"] for r in records)
    manifest = {
        "n_total": len(records), "n_master_train": n_master, "n_extra": n_extra,
        "extra_abstains_file": cfg.extra_abstains or None,
        "extra_abstains_sha256": extra_sha,
        "n_abstain": n_abstain, "n_answerable": len(records) - n_abstain,
        "abstain_frac": round(n_abstain / len(records), 4),
        "record_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
    }
    return records, manifest


# LoRA spec — IDENTICAL for SFT-A and SFT-B (CLAUDE.md rule 5).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj")
# Path segments that mark the (frozen) vision tower in Qwen-VL module names.
VISION_SEGMENTS = {"visual", "vision", "vision_tower", "vision_model", "merger"}


@dataclass
class TrainConfig:
    """All training hyperparameters. Logged verbatim to the run dir."""

    mode: str = "B"                     # "A" or "B" — the ONLY arm difference
    model_id: str = MODEL_ID
    seed: int = 42
    output_dir: str = "data/checkpoints/sft"
    # LoRA
    lora_r: int = LORA_R
    lora_alpha: int = LORA_ALPHA
    lora_dropout: float = LORA_DROPOUT
    # optim / schedule
    num_train_epochs: float = 2.0       # 2 not 3: JSON-format imitation learns fast,
                                        # and fewer epochs guards against over-fitting
                                        # the easy "abstain" pattern. Same for A and B.
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1   # MUST be 1: completion_only_loss / the
                                          # logits_to_keep tail assume batch size 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"     # bitsandbytes paged 8-bit AdamW (GPU)
    logging_steps: int = 1              # per-step train loss + lr (streamed to W&B)
    # checkpoint every ~0.5 epoch and KEEP them all — selection is post-hoc via the
    # faithfulness eval, NOT by val loss (an abstain-collapsed model can have low val
    # loss); load_best_model_at_end is deliberately OFF. 214 ≈ half an epoch at the
    # rebuilt train size (B: 3423 rec / 8 = 428 steps/epoch -> ckpts at 214/428/642/856
    # = ~0.5/1.0/1.5/2.0 epochs). Applied identically to A and B.
    save_strategy: str = "steps"
    save_steps: int = 214
    save_total_limit: int = 5
    eval_strategy: str = "steps"
    eval_steps: int = 214
    # logging
    wandb: bool = True
    wandb_project: str = "faithfuldocvqa-sft"
    wandb_run: str = ""                 # "" -> default f"sft_{mode}_seed{seed}"
    # balanced-mix run-3: extra abstain records merged into the TRAIN set only
    # ("" = off, the documented A/B recipe). Targets derive through the same
    # 3-field path; every merged record is hard-gated to split=="train" and to
    # doc_ids disjoint from validation/test (see build_train_records).
    extra_abstains: str = ""
    # 4-bit QLoRA
    bf16: bool = True
    gradient_checkpointing: bool = True
    # dataloader throughput — fixes the input-pipeline starvation observed in SFT-B
    # (GPU idled between bursts at the default num_workers=0). NUMERICALLY NEUTRAL:
    # __getitem__ is deterministic (no augmentation) and shuffling is the seeded
    # main-process sampler, so these change only batch-prep overlap, NOT the trained
    # model — SFT-A-vs-B comparability is preserved (CLAUDE.md rule 5). B was trained
    # before this change; A (control arm) gets it. Nothing else differs from B.
    dataloader_num_workers: int = 6
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: bool = True
    # smoke test knobs
    smoke: bool = False
    max_steps: int = -1
    limit: int = -1                     # cap #train examples (smoke); -1 = all
    eval_limit: int = -1                # cap #eval examples (smoke); -1 = all


# ── module inspection (CPU; no weight download) ───────────────────────────────

def build_empty_model(model_id: str):
    """Instantiate the model GRAPH on the meta device (no weights, no 8 GB
    download) so we can read ``named_modules`` on a laptop.

    Returns (model, config). Tries the modern image-text-to-text auto class,
    then a couple of fallbacks; raises (STOP) if none resolve.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    last_err = None
    for auto_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                      "AutoModelForCausalLM"):
        try:
            import transformers
            auto_cls = getattr(transformers, auto_name)
            with init_empty_weights():
                model = auto_cls.from_config(config)
            return model, config
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Could not build model graph for {model_id}: {last_err}")


def is_vision(name: str) -> bool:
    """True if a module path belongs to the (frozen) vision tower."""
    return any(seg in VISION_SEGMENTS for seg in name.split("."))


def confirm_target_modules(model) -> dict:
    """Locate the LoRA target projections and CONFIRM they are language-tower only.

    Returns a report dict. Raises (STOP) if the expected q/k/v/o_proj leaves are
    absent — we never guess target modules.
    """
    by_leaf: dict[str, list[str]] = defaultdict(list)
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in TARGET_LEAVES:
            by_leaf[leaf].append(name)

    missing = [leaf for leaf in TARGET_LEAVES if leaf not in by_leaf]
    if missing:
        raise RuntimeError(
            f"Target leaves {missing} not found in {type(model).__name__}. "
            f"STOP — inspect named_modules and set target modules explicitly. "
            f"Leaves present: {sorted({n.split('.')[-1] for n, _ in model.named_modules()})}"
        )

    lang = {leaf: [n for n in paths if not is_vision(n)] for leaf, paths in by_leaf.items()}
    vis = {leaf: [n for n in paths if is_vision(n)] for leaf, paths in by_leaf.items()}

    # sample one vision attention module to show its projection naming (expected
    # to be a fused 'qkv', i.e. NOT q/k/v/o_proj -> the name list can't touch it)
    vision_attn_leaves = sorted({
        name.split(".")[-1]
        for name, _ in model.named_modules()
        if is_vision(name) and ("attn" in name or "attention" in name)
        and name.split(".")[-1] not in ("", )
    })

    return {
        "model_class": type(model).__name__,
        "n_modules": sum(1 for _ in model.named_modules()),
        "qkvo_language_counts": {k: len(v) for k, v in lang.items()},
        "qkvo_vision_counts": {k: len(v) for k, v in vis.items()},
        "example_language_paths": {k: (v[0] if v else None) for k, v in lang.items()},
        "vision_attn_leaf_names": vision_attn_leaves,
        "language_qkvo_paths": sorted(p for v in lang.values() for p in v),
        "vision_qkvo_paths": sorted(p for v in vis.values() for p in v),
    }


def resolve_target_modules(report: dict) -> list[str]:
    """Choose LoRA target_modules from the confirmation report.

    If q/k/v/o_proj appear ONLY on the language tower (the expected Qwen-VL
    layout, where vision attention uses a fused 'qkv'), the simple leaf-name list
    is safe and adapts language attention only. If any target leaf also appears
    on the vision tower, fall back to explicit full language paths (still
    language-only) and the caller should report it.
    """
    if not report["vision_qkvo_paths"]:
        return list(TARGET_LEAVES)
    return report["language_qkvo_paths"]


# ── LoRA config + application ─────────────────────────────────────────────────

def build_lora_config(target_modules: list[str], cfg: TrainConfig):
    """Build the PEFT ``LoraConfig`` (r=16, alpha=32, dropout=0.05, QKVO)."""
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )


def summarize_trainable(model) -> dict:
    """Trainable-param count, % of total, trainable leaf-module names, and a
    HARD assertion that no vision-tower parameter is trainable."""
    trainable, total = 0, 0
    trainable_leaves: set[str] = set()
    vision_trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
            # the LoRA-wrapped base module is the name minus the lora suffix
            trainable_leaves.add(name.split(".lora_")[0].split(".")[-1])
            if is_vision(name):
                vision_trainable += n
    assert vision_trainable == 0, (
        f"{vision_trainable} vision-tower params are trainable — vision must be frozen"
    )
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "trainable_leaf_modules": sorted(trainable_leaves),
    }


# ── completion-only tokenization (CPU-testable) ───────────────────────────────

def tokenize_example(processor, record: dict) -> dict:
    """Tokenize one record into model inputs with COMPLETION-ONLY labels.

    Prompt region (system + user + image + ``<|im_start|>assistant\\n``) -> label
    -100. Completion region (the JSON target + ``<|im_end|>``) -> its own token
    ids. The mask is computed EXPLICITLY (not via a TRL flag — CLAUDE.md rule 4):
    the prompt is rendered separately, tokenized with the SAME image, and its
    token length marks the boundary. A prefix check asserts the boundary is real.

    Returns a dict of tensors: ``input_ids``, ``attention_mask``, ``labels``, plus
    the vision tensors the model needs (``pixel_values``, ``image_grid_thw``).
    """
    import torch
    from PIL import Image

    img = Image.open(ROOT / record["image_path"])
    full_text = processor.apply_chat_template(
        build_messages(record, include_target=True), tokenize=False,
        add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(
        build_messages(record, include_target=False), tokenize=False,
        add_generation_prompt=True)

    full = processor(text=[full_text], images=[img], return_tensors="pt")
    prompt = processor(text=[prompt_text], images=[img], return_tensors="pt")

    input_ids = full["input_ids"][0]
    prompt_len = int(prompt["input_ids"].shape[1])

    # the prompt MUST be a token-level prefix of the full sequence, else the mask
    # boundary is wrong (e.g. tokenizer merged across the boundary) -> STOP.
    if not torch.equal(input_ids[:prompt_len], prompt["input_ids"][0]):
        raise RuntimeError(
            "prompt is not a token-prefix of the full sequence; completion-only "
            "mask boundary is unsafe — STOP and inspect the chat template/tokenizer."
        )

    labels = input_ids.clone()
    labels[:prompt_len] = -100

    # Drop the leading batch dim for PER-TOKEN fields only — those are the (1, seq)
    # tensors the processor emits: input_ids, attention_mask, and Qwen3-VL's
    # mm_token_type_ids (the text-vs-image mask M-RoPE needs). Keep the vision
    # tensors as-is: pixel_values is (num_patches, dim) and image_grid_thw is
    # (num_images, 3) — the collator concatenates those across the batch. (A blanket
    # "squeeze if len==1" wrongly flattened image_grid_thw (1,3) -> (3,) before.)
    seq_len = full["input_ids"].shape[1]
    out = {}
    for k, v in full.items():
        if hasattr(v, "ndim") and v.ndim == 2 and v.shape[0] == 1 and v.shape[1] == seq_len:
            out[k] = v[0]
        else:
            out[k] = v
    out["labels"] = labels
    out["_prompt_len"] = prompt_len  # debugging/testing aid
    return out


# ── batching collator (used by the GPU Trainer) ───────────────────────────────

class CompletionOnlyVLCollator:
    """Pads input_ids/attention_mask/labels and concatenates the Qwen vision
    tensors (pixel_values stacked along patches, image_grid_thw along images).
    Each example is pre-tokenized by :func:`tokenize_example`.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    # pad value per per-token field; padded positions are non-loss, non-attended,
    # text-type. Fields not listed pad with 0.
    _PAD = {"input_ids": None, "attention_mask": 0, "labels": -100,
            "mm_token_type_ids": 0, "token_type_ids": 0, "position_ids": 0}

    def __call__(self, features: list[dict]) -> dict:
        import torch

        maxlen = max(f["input_ids"].shape[0] for f in features)
        ref_len = features[0]["input_ids"].shape[0]

        def pad_stack(key: str) -> "torch.Tensor":
            pad_value = self.pad_token_id if key == "input_ids" else self._PAD.get(key, 0)
            rows = []
            for f in features:
                t = f[key]
                rows.append(torch.cat([t, t.new_full((maxlen - t.shape[0],), pad_value)]))
            return torch.stack(rows)

        batch: dict = {}
        for key in features[0]:
            sample = features[0][key]
            # per-token field (1-D, length == sequence length) -> pad + stack;
            # everything else (pixel_values, image_grid_thw) -> concat along dim 0.
            if hasattr(sample, "ndim") and sample.ndim == 1 and sample.shape[0] == ref_len:
                batch[key] = pad_stack(key)
            else:
                batch[key] = torch.cat([f[key] for f in features], dim=0)
        return batch


# ── completion-only loss (memory-bounded; CPU-testable) ───────────────────────

def completion_only_loss(model, inputs, labels):
    """Causal-LM cross-entropy over the COMPLETION tokens only, asking the model to
    materialize logits for just the completion tail via ``logits_to_keep`` rather
    than the full sequence. A full ~152k-vocab logits tensor over a ~10k-token page
    is several GB in fp32 and OOMs a 24 GB GPU; we only ever supervise the ~30
    completion positions. Mathematically identical to standard completion-only CE
    (asserted in ``tests/test_completion_loss.py``).

    Requires per-device batch size 1 — with right-padding the completion is the
    suffix, so the kept tail aligns with the labels. ``inputs`` must NOT contain
    ``labels``. Returns ``(loss, outputs)``.
    """
    import torch.nn.functional as F

    if labels.shape[0] != 1:
        raise ValueError(
            "completion_only_loss requires per_device_train_batch_size=1 "
            f"(got batch size {labels.shape[0]})"
        )
    n_comp = int((labels[0] != -100).sum())                 # completion tokens
    outputs = model(**inputs, logits_to_keep=n_comp + 1)    # +1 for next-token shift
    logits = outputs.logits.float()                         # (1, n_comp+1, vocab) — tiny
    shift_logits = logits[:, :-1, :]                        # predicts the completion
    shift_labels = labels[:, -n_comp:].to(logits.device)
    loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1), ignore_index=-100,
    )
    return loss, outputs


# ── QLoRA model load (GPU; bitsandbytes lazy) ─────────────────────────────────

def build_bnb_config():
    """The 4-bit QLoRA quantization config — the SINGLE source of truth shared by
    training (``load_qlora_model``) and the eval generation runner.

    The eval runner loads the base model with this same config so a 4-bit
    numeric-sanity check at eval time tests the EXACT quantization that was
    trained, not a divergent one. NF4 + double-quant, bf16 compute. GPU/bnb only.
    """
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_qlora_model(cfg: TrainConfig, target_modules: list[str]):
    """Load Qwen3-VL-4B in 4-bit NF4 + double-quant, attach LoRA, freeze vision.

    GPU-only (bitsandbytes). Returns the PEFT model. Not exercised on the laptop.
    """
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText

    bnb = build_bnb_config()
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id, quantization_config=bnb, torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing)
    model = get_peft_model(model, build_lora_config(target_modules, cfg))
    # vision is frozen by construction (LoRA only on language QKVO); verify.
    summarize_trainable(model)
    return model


# ── CPU inspection entry point ────────────────────────────────────────────────

def inspect(cfg: TrainConfig) -> dict:
    """Run the CPU-buildable checks: module-name dump + QKVO confirmation + LoRA
    trainable/vision-frozen verification, all on the meta-device model graph."""
    from peft import get_peft_model

    print(f"Building model graph for {cfg.model_id} (empty weights, no download)...")
    model, _ = build_empty_model(cfg.model_id)
    report = confirm_target_modules(model)

    print("\n" + "=" * 78)
    print(f"MODULE-NAME DUMP — {report['model_class']}  ({report['n_modules']} modules)")
    print("=" * 78)
    print("q/k/v/o_proj on LANGUAGE tower:", report["qkvo_language_counts"])
    print("q/k/v/o_proj on VISION tower  :", report["qkvo_vision_counts"], "(expect all 0)")
    print("example language paths:")
    for leaf, path in report["example_language_paths"].items():
        print(f"   {leaf:<8} -> {path}")
    print("vision attention projection leaf-names:", report["vision_attn_leaf_names"],
          "(fused 'qkv'/'proj' => the QKVO name list cannot touch vision)")

    target_modules = resolve_target_modules(report)
    print(f"\nresolved target_modules: {target_modules}")
    if report["vision_qkvo_paths"]:
        print("  NOTE: q/k/v/o_proj also found on vision -> using explicit "
              "language-only full paths.")

    # apply LoRA on the meta model to count trainable params + verify vision frozen
    lora = build_lora_config(target_modules, cfg)
    peft_model = get_peft_model(model, lora)
    summary = summarize_trainable(peft_model)
    print("\n" + "=" * 78)
    print("LoRA TRAINABLE-PARAM SUMMARY")
    print("=" * 78)
    print(f"  r={cfg.lora_r} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout}")
    print(f"  trainable params : {summary['trainable_params']:,}")
    print(f"  total params     : {summary['total_params']:,}")
    print(f"  trainable %      : {summary['trainable_pct']:.4f}%")
    print(f"  trainable leaves : {summary['trainable_leaf_modules']}")
    print("  vision-tower trainable params: 0 (asserted) -> VISION FROZEN")
    return {"report": report, "summary": summary, "target_modules": target_modules}


# ── full training (GPU; Trainer loop) ─────────────────────────────────────────

def train(cfg: TrainConfig) -> str:
    """Full SFT run on the GPU box. Saves adapter + config + log to output_dir.

    Not run on the laptop (4-bit + Trainer need CUDA/bitsandbytes). Returns the
    output dir path.
    """
    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    class CompletionOnlyTrainer(Trainer):
        """Thin wrapper that routes the loss through ``completion_only_loss`` (which
        supervises only the completion tokens and materializes logits for just the
        completion tail — see that function). Keeping the math in a module-level
        function makes it unit-testable on CPU without a real model."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            loss, outputs = completion_only_loss(model, inputs, labels)
            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            # eval path: same completion-only loss, loss-ONLY (no logit gathering
            # across the eval set). val loss is a monitor here, never a selector.
            labels = inputs.pop("labels")
            with torch.no_grad():
                loss, _ = completion_only_loss(model, inputs, labels)
            return (loss.detach(), None, None)

    class _LazySFTDataset(torch.utils.data.Dataset):
        """Tokenize each example on access, not all up front. A single example's
        pixel_values is ~90 MB, so eagerly tokenizing the full ~2200-example split
        would hold hundreds of GB in RAM; lazy access keeps only the in-flight
        batch resident."""

        def __init__(self, records, processor):
            self.records, self.processor = records, processor

        def __len__(self):
            return len(self.records)

        def __getitem__(self, i):
            ex = tokenize_example(self.processor, self.records[i])
            ex.pop("_prompt_len", None)
            return ex

    set_seed(cfg.seed)
    proc, resolved = load_processor(cfg.model_id)
    cfg.model_id = resolved

    # confirm target modules on the real architecture before training
    model_graph, _ = build_empty_model(cfg.model_id)
    target_modules = resolve_target_modules(confirm_target_modules(model_graph))
    del model_graph

    # Weights & Biases — init BEFORE the long run so a missing login fails fast.
    # Runs share project + group and are tagged by mode, so SFT-A and SFT-B overlay.
    report_to: list = []
    if cfg.wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run or f"sft_{cfg.mode}_seed{cfg.seed}",
                group="sft-A-vs-B",
                tags=[f"mode-{cfg.mode}", "sft", "qwen3vl-4b-lora"],
                config={
                    "mode": cfg.mode, "seed": cfg.seed,
                    "lora_r": cfg.lora_r, "lora_alpha": cfg.lora_alpha,
                    "lora_dropout": cfg.lora_dropout, "target_modules": target_modules,
                    "epochs": cfg.num_train_epochs, "learning_rate": cfg.learning_rate,
                    "per_device_batch_size": cfg.per_device_train_batch_size,
                    "grad_accum": cfg.gradient_accumulation_steps, "model_id": cfg.model_id,
                },
            )
            report_to = ["wandb"]
        except Exception as e:  # noqa: BLE001 — fail fast before a multi-hour run
            raise RuntimeError(
                f"W&B logging requested but init failed ({type(e).__name__}: {e}). "
                "Run `wandb login` (or set WANDB_API_KEY) on this machine, or pass "
                "--no-wandb to train without logging."
            ) from e

    records, train_manifest = build_train_records(cfg)
    if cfg.limit > 0:
        records = records[: cfg.limit]
    # validation is the SHARED, FULL held-out split (identical for A and B) — used
    # only as an overfitting/abstain-collapse loss monitor. The mode filter does
    # NOT apply here. `test` is reserved for the final faithfulness eval (separate).
    eval_records = load_split("validation", cfg.mode)
    if cfg.eval_limit > 0:
        eval_records = eval_records[: cfg.eval_limit]
    print(f"[train] mode={cfg.mode} train={len(records)} validation={len(eval_records)}")

    train_dataset = _LazySFTDataset(records, proc)
    eval_dataset = _LazySFTDataset(eval_records, proc)

    model = load_qlora_model(cfg, target_modules)
    collator = CompletionOnlyVLCollator(pad_token_id=proc.tokenizer.pad_token_id
                                        or proc.tokenizer.eos_token_id)

    out = ROOT / cfg.output_dir / f"sft_{cfg.mode}_seed{cfg.seed}"
    out.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim=cfg.optim,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_prefetch_factor=cfg.dataloader_prefetch_factor,
        dataloader_persistent_workers=cfg.dataloader_persistent_workers,
        dataloader_pin_memory=cfg.dataloader_pin_memory,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps,
        seed=cfg.seed,
        run_name=cfg.wandb_run or f"sft_{cfg.mode}_seed{cfg.seed}",
        report_to=report_to,
    )
    (out / "training_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    # merged-training-data manifest, persisted AT LAUNCH: what exactly was trained on
    # (class counts + sha256 of the sorted record_id list), so the mix is auditable
    # without re-deriving it from code.
    (out / "train_manifest.json").write_text(json.dumps(train_manifest, indent=2))

    trainer = CompletionOnlyTrainer(model=model, args=args,
                                    train_dataset=train_dataset, eval_dataset=eval_dataset,
                                    data_collator=collator)
    result = trainer.train()
    model.save_pretrained(str(out))
    (out / "train_result.json").write_text(json.dumps(result.metrics, indent=2))
    print(f"[train] done. final loss={result.metrics.get('train_loss')}  -> {out}")

    # smoke acceptance: prove the saved adapter reloads (reuses the in-memory base,
    # so no 8 GB re-download). A failure here does NOT mean training failed.
    if cfg.smoke:
        try:
            model.load_adapter(str(out), adapter_name="reloaded")
            print(f"[smoke] adapter reload OK from {out}")
        except Exception as e:  # noqa: BLE001
            print(f"[smoke] adapter reload check FAILED "
                  f"(training itself was fine): {type(e).__name__}: {e}")
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["A", "B"], default="B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="data/checkpoints/sft")
    ap.add_argument("--epochs", type=float, default=None,
                    help="override num_train_epochs (default: TrainConfig's 2.0). "
                         "The 3-field baseline uses 3 to read the faithfulness curve past 2.0.")
    ap.add_argument("--save-total-limit", type=int, default=None,
                    help="max checkpoints to keep (default: TrainConfig's 5). Pass a value "
                         ">= the number of checkpoints to keep them ALL — e.g. with "
                         "save_steps=214 a 3-epoch run makes 6 checkpoints, so use 6+.")
    ap.add_argument("--inspect", action="store_true",
                    help="CPU: dump modules + confirm QKVO + LoRA summary; no training")
    ap.add_argument("--smoke", action="store_true", help="short run on a few examples")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-wandb", action="store_true", help="disable Weights & Biases logging")
    ap.add_argument("--extra-abstains", default=None,
                    help="jsonl of extra TRAIN abstain records to merge (balanced-mix "
                         "run-3); hard-gated to split=='train' + val/test-disjoint doc_ids")
    ap.add_argument("--save-steps", type=int, default=None,
                    help="override checkpoint/eval cadence (default: TrainConfig's 214 "
                         "= half an epoch at the documented train size; scale it to keep "
                         "the same epoch fraction when the mix changes the step count)")
    ap.add_argument("--wandb-project", default=None, help="override W&B project")
    ap.add_argument("--wandb-run", default=None, help="override W&B run name")
    a = ap.parse_args()

    cfg = TrainConfig(mode=a.mode, seed=a.seed, output_dir=a.output_dir,
                      smoke=a.smoke, limit=a.limit, max_steps=a.max_steps)
    cfg.wandb = not a.no_wandb
    if a.extra_abstains is not None:
        cfg.extra_abstains = a.extra_abstains
    if a.save_steps is not None:
        cfg.save_steps = a.save_steps
        cfg.eval_steps = a.save_steps
    if a.wandb_project is not None:
        cfg.wandb_project = a.wandb_project
    if a.wandb_run is not None:
        cfg.wandb_run = a.wandb_run
    # explicit overrides — recorded verbatim in training_config.json (asdict(cfg)).
    # These are NOT model hyperparameters (epochs aside): they don't alter the A/B
    # training recipe, so overriding them here keeps the documented dataclass defaults
    # intact while letting this 3-field baseline run 3 epochs and keep every checkpoint.
    if a.epochs is not None:
        cfg.num_train_epochs = a.epochs
    if a.save_total_limit is not None:
        cfg.save_total_limit = a.save_total_limit
    if a.smoke:
        cfg.limit = a.limit if a.limit > 0 else 20
        cfg.max_steps = a.max_steps if a.max_steps > 0 else 6
        cfg.save_steps = 2          # exercise checkpointing + eval inside the smoke
        cfg.eval_steps = 2
        cfg.eval_limit = 8

    if a.inspect:
        inspect(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
