"""Matched-pool SFT continuation from the verified ck-642 adapter.

This is a narrow experiment runner for one comparison:

    ck-642 SFT adapter + continued SFT on the exact run-1/run-2 GRPO pool

It deliberately reuses the repository's existing prompt/target formatter and
completion-only SFT loss. The data pool is reconstructed through
``grpo.train_grpo.build_train_pool`` so the 5,000-record population matches the
original GRPO runs. The continuation uses a fresh optimizer/scheduler and a
deterministic 600-example supervised exposure sequence.

GPU-only for ``train`` and ``smoke`` modes; CPU-safe for manifest checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grpo.train_grpo import GRPORunConfig, build_train_pool
from scripts.format_sft import MODEL_ID, load_processor
from scripts.train_sft import (
    CompletionOnlyVLCollator,
    TrainConfig,
    build_bnb_config,
    completion_only_loss,
    summarize_trainable,
    tokenize_example,
)

EXPECTED_CK642_ADAPTER_SHA256 = (
    "d76fa2b4609ba126bb427e5f338ba5a9c65ef8111e0bb37df9d31d344ba09691"
)
EXPECTED_POOL_ORDERED_SHA256 = (
    "5e87988a38366c3b92f634815dc83af18868e830c4d3b51e09a1bbdc227f9238"
)
EXPECTED_POOL_SORTED_SHA256 = (
    "8305195507de039f245bcd3918d3fe40c18843002d2cf26304ef0494e9da7d7d"
)


@dataclass
class MatchedSFTConfig:
    """Resolved knobs for the single matched-pool SFT continuation."""

    run_name: str = "sft_ck642_matched_grpo_pool_seed42"
    model_id: str = MODEL_ID
    seed: int = 42
    adapter: str = "outputs/sft_v3_32g/sft_B_seed42/checkpoint-642"
    expected_adapter_sha256: str = EXPECTED_CK642_ADAPTER_SHA256
    output_dir: str = "outputs/sft_ck642_matched_grpo_pool_seed42"

    # Locked matched-exposure budget.
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 150
    total_supervised_examples: int = 600
    save_steps: int = 50
    save_total_limit: int = 3

    # Original SFT training settings retained unless explicitly incompatible.
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_new_tokens_eval: int = 128
    dataloader_num_workers: int = 6
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    dataloader_pin_memory: bool = True
    logging_steps: int = 1
    wandb: bool = False
    wandb_project: str = "faithfuldocvqa-sft"

    # Smoke mode keeps the same essential geometry but only two optimizer steps.
    smoke: bool = False


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file, streaming in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    """Hash record IDs exactly as newline-joined UTF-8 strings."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def resolve_under_root(path: str) -> Path:
    """Resolve an absolute path or a ROOT-relative path."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def verify_adapter_checkpoint(adapter_dir: Path, expected_sha256: str) -> dict[str, Any]:
    """Check ck-642 adapter files and the recorded adapter weight hash."""
    required = ["adapter_config.json", "adapter_model.safetensors"]
    missing = [name for name in required if not (adapter_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"adapter checkpoint missing required files: {missing}")

    observed = sha256_file(adapter_dir / "adapter_model.safetensors")
    if observed != expected_sha256:
        raise RuntimeError(
            "ck-642 adapter hash mismatch: "
            f"observed {observed}, expected {expected_sha256}"
        )

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    return {
        "adapter_dir": str(adapter_dir),
        "adapter_model_sha256": observed,
        "hash_method": "sha256(adapter_model.safetensors bytes)",
        "adapter_config": {
            "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
            "peft_type": adapter_config.get("peft_type"),
            "task_type": adapter_config.get("task_type"),
            "r": adapter_config.get("r"),
            "lora_alpha": adapter_config.get("lora_alpha"),
            "lora_dropout": adapter_config.get("lora_dropout"),
            "target_modules": adapter_config.get("target_modules"),
            "bias": adapter_config.get("bias"),
            "modules_to_save": adapter_config.get("modules_to_save"),
        },
    }


def build_exact_grpo_pool(seed: int = 42) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reconstruct the exact run-1/run-2 5,000-record GRPO pool."""
    cfg = GRPORunConfig(
        run=2,
        seed=seed,
        pool_abstain=2000,
        pool_answerable=3000,
        train_abstains=None,
        val_recurs_false_only=False,
        run_name=None,
    )
    pool, _ = build_train_pool(cfg)

    ordered_ids = [str(r["record_id"]) for r in pool]
    sorted_ids = sorted(ordered_ids)
    n_answerable = sum(1 for r in pool if not r.get("target_is_abstain"))
    n_abstain = sum(1 for r in pool if r.get("target_is_abstain"))
    n_natural_abstain = sum(
        1
        for r in pool
        if r.get("target_is_abstain") and not str(r["record_id"]).endswith("_mfab")
    )
    recurs_counts = Counter(
        r.get("answer_recurs_in_menu")
        for r in pool
        if r.get("target_is_abstain")
    )
    manifest = {
        "builder": "grpo.train_grpo.build_train_pool",
        "seed": seed,
        "pool_abstain": 2000,
        "pool_answerable": 3000,
        "n_total": len(pool),
        "n_answerable": n_answerable,
        "n_manufactured_refusal": n_abstain,
        "n_natural_ocr_miss": n_natural_abstain,
        "answer_recurs_in_menu_counts_for_manufactured": {
            str(k): v for k, v in sorted(recurs_counts.items(), key=lambda kv: str(kv[0]))
        },
        "ordered_record_ids_sha256": sha256_ids(ordered_ids),
        "sorted_record_ids_sha256": sha256_ids(sorted_ids),
        "hash_method": "sha256(newline-joined record_id strings)",
    }
    validate_pool_manifest(manifest)
    return pool, manifest


def validate_pool_manifest(manifest: dict[str, Any]) -> None:
    """Fail before training if the reconstructed pool does not match the lock."""
    expected = {
        "n_total": 5000,
        "n_answerable": 3000,
        "n_manufactured_refusal": 2000,
        "n_natural_ocr_miss": 0,
        "ordered_record_ids_sha256": EXPECTED_POOL_ORDERED_SHA256,
        "sorted_record_ids_sha256": EXPECTED_POOL_SORTED_SHA256,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"GRPO pool reconstruction mismatch: {mismatches}")


def pool_manifest_rows(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compact row-level manifest for the full 5,000-record pool."""
    rows: list[dict[str, Any]] = []
    for i, record in enumerate(pool):
        rows.append(
            {
                "pool_index": i,
                "record_id": str(record["record_id"]),
                "source_record_id": record.get("source_record_id"),
                "split": record.get("split"),
                "doc_id": record.get("doc_id"),
                "target_is_abstain": bool(record.get("target_is_abstain")),
                "gold_localized": bool(record.get("gold_localized")),
                "answer_recurs_in_menu": record.get("answer_recurs_in_menu"),
            }
        )
    return rows


def build_exposure_sequence(
    pool: list[dict[str, Any]], cfg: MatchedSFTConfig
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Return the deterministic 600-example SFT exposure sequence.

    The GRPO pool builder already returns a seeded shuffled pool. For this bounded
    continuation we use the first N rows of that fixed order and train over them
    sequentially exactly once. This matches the run-2 unique-prompt exposure count
    without claiming the same encountered prompt sequence.
    """
    expected_examples = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * cfg.max_steps
    )
    if expected_examples != cfg.total_supervised_examples:
        raise ValueError(
            "exposure arithmetic mismatch: "
            f"{cfg.per_device_train_batch_size} × {cfg.gradient_accumulation_steps} "
            f"× {cfg.max_steps} = {expected_examples}, expected "
            f"{cfg.total_supervised_examples}"
        )
    if expected_examples > len(pool):
        raise ValueError("requested exposure sequence is larger than the pool")

    exposure = pool[:expected_examples]
    rows = []
    for exposure_index, record in enumerate(exposure):
        rows.append(
            {
                "exposure_index": exposure_index,
                "pool_index": exposure_index,
                "record_id": str(record["record_id"]),
                "source_record_id": record.get("source_record_id"),
                "target_is_abstain": bool(record.get("target_is_abstain")),
                "gold_localized": bool(record.get("gold_localized")),
                "answer_recurs_in_menu": record.get("answer_recurs_in_menu"),
            }
        )

    ids = [row["record_id"] for row in rows]
    manifest = {
        "sequence_policy": "first 600 records from the seeded run-1/run-2 GRPO pool order",
        "actual_training_order": "sequential sampler over this exposure_sequence.jsonl",
        "same_as_run2_encountered_prompt_sequence": False,
        "n_examples": len(exposure),
        "n_answerable": sum(1 for r in exposure if not r.get("target_is_abstain")),
        "n_manufactured_refusal": sum(1 for r in exposure if r.get("target_is_abstain")),
        "ordered_record_ids_sha256": sha256_ids(ids),
        "sorted_record_ids_sha256": sha256_ids(sorted(ids)),
        "optimizer_steps": cfg.max_steps,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "supervised_examples_per_optimizer_step": (
            cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        ),
    }
    return exposure, manifest, rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def trainable_parameter_report(model: Any) -> dict[str, Any]:
    """Report and enforce trainability: LoRA params only, no vision params."""
    names = [name for name, p in model.named_parameters() if getattr(p, "requires_grad", False)]
    total = sum(p.numel() for _, p in model.named_parameters())
    trainable = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    non_lora = [name for name in names if "lora_" not in name]
    vision = [
        name
        for name in names
        if any(part in name for part in ("visual", "vision", "vision_tower", "vision_model"))
    ]
    if not names:
        raise RuntimeError("adapter loaded with zero trainable parameters")
    if non_lora:
        raise RuntimeError(f"non-LoRA parameters are trainable: {non_lora[:10]}")
    if vision:
        raise RuntimeError(f"vision/base parameters are trainable: {vision[:10]}")
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total if total else 0.0,
        "n_trainable_tensors": len(names),
        "trainable_parameter_names": names,
    }


def load_trainable_adapter(cfg: MatchedSFTConfig, adapter_dir: Path):
    """Load a 4-bit base model and attach ck-642 as a trainable adapter."""
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText

    base = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id,
        quantization_config=build_bnb_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    model.config.use_cache = False
    summarize_trainable(model)
    trainable_parameter_report(model)
    return model


class SequentialExposureTrainerMixin:
    """Force the persisted exposure sequence to be consumed in order."""

    def _get_train_sampler(self, dataset=None):  # noqa: D401 - signature matches Trainer hook
        from torch.utils.data import SequentialSampler

        return SequentialSampler(dataset if dataset is not None else self.train_dataset)


def build_training_arguments(cfg: MatchedSFTConfig, out: Path):
    """Build TrainingArguments from the locked config."""
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(out),
        num_train_epochs=1.0,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1,
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
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="no",
        seed=cfg.seed,
        run_name=cfg.run_name,
        report_to=(["wandb"] if cfg.wandb else []),
    )


def effective_config(
    cfg: MatchedSFTConfig,
    adapter_report: dict[str, Any],
    pool_manifest: dict[str, Any],
    exposure_manifest: dict[str, Any],
    trainable_report: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolved config persisted before the expensive run."""
    original_sft = TrainConfig()
    return {
        "experiment_name": "matched-pool and matched-exposure SFT",
        "comparison_question": (
            "Continue ck-642 with supervised learning on the run-1/run-2 GRPO pool "
            "using the same unique-prompt exposure count as run-2."
        ),
        "does_not_claim_identical_training_signal_to_grpo": True,
        "does_not_claim_exact_same_600_prompts_as_run2": True,
        "fresh_optimizer_scheduler": True,
        "resume_from_checkpoint": None,
        "old_optimizer_state_loaded": False,
        "fixed_final_checkpoint": 150,
        "checkpoint_selection_rule": (
            "step 150 is fixed before training; validation/test results do not change "
            "the checkpoint, budget, or hyperparameters."
        ),
        "matched_sft_config": asdict(cfg),
        "original_sft_reference_settings": {
            "learning_rate": original_sft.learning_rate,
            "warmup_ratio": original_sft.warmup_ratio,
            "lr_scheduler_type": original_sft.lr_scheduler_type,
            "weight_decay": original_sft.weight_decay,
            "max_grad_norm": original_sft.max_grad_norm,
            "optim": original_sft.optim,
            "bf16": original_sft.bf16,
            "gradient_checkpointing": original_sft.gradient_checkpointing,
            "per_device_train_batch_size": original_sft.per_device_train_batch_size,
            "dataloader_num_workers": original_sft.dataloader_num_workers,
            "dataloader_prefetch_factor": original_sft.dataloader_prefetch_factor,
            "dataloader_persistent_workers": original_sft.dataloader_persistent_workers,
            "dataloader_pin_memory": original_sft.dataloader_pin_memory,
        },
        "adapter": adapter_report,
        "pool_manifest": pool_manifest,
        "exposure_manifest": exposure_manifest,
        "trainable_parameter_report": trainable_report,
        "environment": environment,
    }


def collect_environment() -> dict[str, Any]:
    """Capture package and GPU information for the run artifact."""
    env: dict[str, Any] = {"python": sys.version}
    for module_name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes"):
        try:
            module = __import__(module_name)
            env[module_name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            env[module_name] = f"unavailable: {type(exc).__name__}: {exc}"
    try:
        import torch

        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_total_memory_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 3
            )
    except Exception as exc:  # noqa: BLE001
        env["gpu_probe_error"] = f"{type(exc).__name__}: {exc}"
    return env


class _LazyExposureDataset:
    """Map-style dataset that tokenizes the persisted exposure sequence lazily."""

    def __init__(self, records: list[dict[str, Any]], processor: Any):
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        ex = tokenize_example(self.processor, self.records[i])
        ex.pop("_prompt_len", None)
        return ex


def _optimizer_params(model: Any) -> list[Any]:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters for optimizer")
    return params


def run_smoke(cfg: MatchedSFTConfig, out: Path) -> dict[str, Any]:
    """Two optimizer-step smoke with finite-loss/gradient/weight-change checks."""
    import torch
    from bitsandbytes.optim import PagedAdamW8bit
    from transformers import get_scheduler, set_seed

    cfg.smoke = True
    cfg.max_steps = 2
    cfg.total_supervised_examples = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * cfg.max_steps
    )
    cfg.save_steps = 1
    cfg.save_total_limit = 2
    cfg.dataloader_num_workers = 0
    cfg.dataloader_prefetch_factor = 2
    cfg.dataloader_persistent_workers = False

    set_seed(cfg.seed)
    adapter_dir = resolve_under_root(cfg.adapter)
    adapter_report = verify_adapter_checkpoint(adapter_dir, cfg.expected_adapter_sha256)
    pool, pool_manifest = build_exact_grpo_pool(cfg.seed)
    exposure, exposure_manifest, exposure_rows = build_exposure_sequence(pool, cfg)
    environment = collect_environment()

    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "pool_manifest.jsonl", pool_manifest_rows(pool))
    write_json(out / "pool_manifest.summary.json", pool_manifest)
    write_jsonl(out / "exposure_sequence.jsonl", exposure_rows)
    write_json(out / "exposure_manifest.json", exposure_manifest)
    write_json(
        out / "resolved_training_config.json",
        effective_config(cfg, adapter_report, pool_manifest, exposure_manifest, environment=environment),
    )

    proc, resolved_model_id = load_processor(cfg.model_id)
    cfg.model_id = resolved_model_id
    model = load_trainable_adapter(cfg, adapter_dir)
    trainable = trainable_parameter_report(model)

    first_name, first_param = next(
        (name, p) for name, p in model.named_parameters() if p.requires_grad
    )
    before = first_param.detach().clone()

    collator = CompletionOnlyVLCollator(
        pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    )
    optimizer = PagedAdamW8bit(_optimizer_params(model), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    warmup_steps = int(cfg.warmup_ratio * cfg.max_steps)
    scheduler = get_scheduler(
        cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    losses: list[float] = []
    grad_checks: list[dict[str, Any]] = []
    model.train()
    examples_seen = 0
    for step in range(cfg.max_steps):
        optimizer.zero_grad(set_to_none=True)
        for j in range(cfg.gradient_accumulation_steps):
            record = exposure[step * cfg.gradient_accumulation_steps + j]
            encoded = tokenize_example(proc, record)
            encoded.pop("_prompt_len", None)
            batch = collator([encoded])
            labels = batch.pop("labels").to(next(model.parameters()).device)
            inputs = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            loss, _ = completion_only_loss(model, inputs, labels)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite smoke loss at step {step + 1}")
            (loss / cfg.gradient_accumulation_steps).backward()
            losses.append(float(loss.detach().cpu()))
            examples_seen += 1

        finite_grads = 0
        nonzero_grads = 0
        for p in _optimizer_params(model):
            if p.grad is None:
                continue
            if torch.isfinite(p.grad).all():
                finite_grads += 1
            if torch.any(p.grad != 0):
                nonzero_grads += 1
        if finite_grads == 0 or nonzero_grads == 0:
            raise RuntimeError(
                f"smoke gradients failed at optimizer step {step + 1}: "
                f"finite_tensors={finite_grads} nonzero_tensors={nonzero_grads}"
            )
        torch.nn.utils.clip_grad_norm_(_optimizer_params(model), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        grad_checks.append(
            {
                "optimizer_step": step + 1,
                "finite_gradient_tensors": finite_grads,
                "nonzero_gradient_tensors": nonzero_grads,
            }
        )

    after = first_param.detach()
    max_abs_delta = float((after - before).abs().max().detach().cpu())
    if max_abs_delta <= 0:
        raise RuntimeError(f"adapter parameter did not change: {first_name}")

    save_dir = out / "smoke_adapter"
    model.save_pretrained(str(save_dir))
    model.load_adapter(str(save_dir), adapter_name="smoke_reloaded")
    adapter_hashes = hash_adapter_outputs(out)

    metrics = {
        "smoke": True,
        "passed": True,
        "optimizer_steps": cfg.max_steps,
        "examples_seen": examples_seen,
        "losses": losses,
        "loss_min": min(losses),
        "loss_max": max(losses),
        "gradient_checks": grad_checks,
        "changed_parameter": first_name,
        "changed_parameter_max_abs_delta": max_abs_delta,
        "adapter_reload_ok": True,
        "trainable_parameter_report": trainable,
        "adapter_hashes": adapter_hashes,
    }
    write_json(out / "smoke_result.json", metrics)
    write_json(
        out / "resolved_training_config.json",
        effective_config(
            cfg,
            adapter_report,
            pool_manifest,
            exposure_manifest,
            trainable_report=trainable,
            environment=environment,
        ),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def hash_adapter_outputs(out: Path) -> dict[str, str]:
    """Hash every adapter_model.safetensors below an output directory."""
    hashes = {}
    for path in sorted(out.glob("**/adapter_model.safetensors")):
        hashes[str(path.relative_to(out))] = sha256_file(path)
    write_json(out / "adapter_hashes.json", hashes)
    return hashes


def train(cfg: MatchedSFTConfig) -> str:
    """Run the locked matched-pool SFT continuation."""
    import torch
    from transformers import Trainer, set_seed

    class CompletionOnlySequentialTrainer(SequentialExposureTrainerMixin, Trainer):
        """Trainer with existing completion-only loss and fixed exposure order."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            loss, outputs = completion_only_loss(model, inputs, labels)
            return (loss, outputs) if return_outputs else loss

    if cfg.max_steps != 150 or cfg.gradient_accumulation_steps != 4:
        raise ValueError("substantive run is locked to 150 steps and grad_accum=4")
    set_seed(cfg.seed)

    adapter_dir = resolve_under_root(cfg.adapter)
    adapter_report = verify_adapter_checkpoint(adapter_dir, cfg.expected_adapter_sha256)
    pool, pool_manifest = build_exact_grpo_pool(cfg.seed)
    exposure, exposure_manifest, exposure_rows = build_exposure_sequence(pool, cfg)
    environment = collect_environment()

    out = resolve_under_root(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "pool_manifest.jsonl", pool_manifest_rows(pool))
    write_json(out / "pool_manifest.summary.json", pool_manifest)
    write_jsonl(out / "exposure_sequence.jsonl", exposure_rows)
    write_json(out / "exposure_manifest.json", exposure_manifest)
    write_json(
        out / "resolved_training_config.json",
        effective_config(cfg, adapter_report, pool_manifest, exposure_manifest, environment=environment),
    )

    proc, resolved_model_id = load_processor(cfg.model_id)
    cfg.model_id = resolved_model_id
    model = load_trainable_adapter(cfg, adapter_dir)
    trainable = trainable_parameter_report(model)
    write_json(
        out / "resolved_training_config.json",
        effective_config(
            cfg,
            adapter_report,
            pool_manifest,
            exposure_manifest,
            trainable_report=trainable,
            environment=environment,
        ),
    )
    write_json(out / "trainable_parameters.json", trainable)

    train_dataset = _LazyExposureDataset(exposure, proc)
    collator = CompletionOnlyVLCollator(
        pad_token_id=proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
    )
    args = build_training_arguments(cfg, out)
    trainer = CompletionOnlySequentialTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    print(
        "[matched-sft] launching locked continuation: "
        f"steps={cfg.max_steps} examples={len(exposure)} "
        f"batch={cfg.per_device_train_batch_size} grad_accum={cfg.gradient_accumulation_steps}",
        flush=True,
    )
    print(
        f"[matched-sft] torch {torch.__version__} cuda {torch.version.cuda} "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda'}",
        flush=True,
    )
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=None)
    elapsed = time.time() - t0

    model.save_pretrained(str(out / "final_adapter"))
    metrics = {
        **result.metrics,
        "wall_time_sec": elapsed,
        "fixed_final_checkpoint": 150,
        "examples_seen": len(exposure),
        "fresh_optimizer_scheduler": True,
        "resume_from_checkpoint": None,
    }
    write_json(out / "train_result.json", metrics)
    hashes = hash_adapter_outputs(out)
    print(f"[matched-sft] done in {elapsed / 3600:.2f}h -> {out}", flush=True)
    print(f"[matched-sft] adapter hashes: {json.dumps(hashes, sort_keys=True)}", flush=True)
    return str(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default=MatchedSFTConfig.adapter)
    parser.add_argument("--expected-adapter-sha256", default=EXPECTED_CK642_ADAPTER_SHA256)
    parser.add_argument("--output-dir", default=MatchedSFTConfig.output_dir)
    parser.add_argument("--run-name", default=MatchedSFTConfig.run_name)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run the two-optimizer-step smoke")
    parser.add_argument("--manifest-only", action="store_true", help="verify pool/manifests without GPU")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = MatchedSFTConfig(
        run_name=args.run_name,
        seed=args.seed,
        adapter=args.adapter,
        expected_adapter_sha256=args.expected_adapter_sha256,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        wandb=not args.no_wandb,
        smoke=args.smoke,
    )
    cfg.total_supervised_examples = (
        cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * cfg.max_steps
    )

    if args.manifest_only:
        pool, pool_manifest = build_exact_grpo_pool(cfg.seed)
        _, exposure_manifest, _ = build_exposure_sequence(pool, cfg)
        print(json.dumps({"pool": pool_manifest, "exposure": exposure_manifest}, indent=2))
        return

    out = resolve_under_root(cfg.output_dir)
    if args.smoke:
        run_smoke(cfg, out)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
