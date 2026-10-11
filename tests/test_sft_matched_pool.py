"""CPU checks for the matched-pool SFT continuation experiment."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from scripts.faithful_io import parse_tolerant_v3
from scripts.format_sft import _v3_target
from scripts.train_sft_matched_pool import (
    EXPECTED_POOL_ORDERED_SHA256,
    EXPECTED_POOL_SORTED_SHA256,
    MatchedSFTConfig,
    SequentialExposureTrainerMixin,
    build_exact_grpo_pool,
    build_exposure_sequence,
    effective_config,
    load_trainable_adapter,
    trainable_parameter_report,
)


def test_exact_run1_run2_pool_reconstruction_hashes() -> None:
    pool, manifest = build_exact_grpo_pool(seed=42)

    assert len(pool) == 5000
    assert manifest["n_answerable"] == 3000
    assert manifest["n_manufactured_refusal"] == 2000
    assert manifest["n_natural_ocr_miss"] == 0
    assert manifest["answer_recurs_in_menu_counts_for_manufactured"] == {
        "False": 1623,
        "True": 377,
    }
    assert manifest["ordered_record_ids_sha256"] == EXPECTED_POOL_ORDERED_SHA256
    assert manifest["sorted_record_ids_sha256"] == EXPECTED_POOL_SORTED_SHA256


def test_matched_pool_targets_use_existing_three_field_formatter() -> None:
    pool, _ = build_exact_grpo_pool(seed=42)
    answerable = next(record for record in pool if not record["target_is_abstain"])
    decoy = next(
        record
        for record in pool
        if record["target_is_abstain"] and record.get("answer_recurs_in_menu") is True
    )
    clean_abstain = next(
        record
        for record in pool
        if record["target_is_abstain"] and record.get("answer_recurs_in_menu") is False
    )

    answerable_target = parse_tolerant_v3(_v3_target(answerable)).value
    decoy_target = parse_tolerant_v3(_v3_target(decoy)).value
    clean_target = parse_tolerant_v3(_v3_target(clean_abstain)).value

    assert answerable_target is not None
    assert answerable_target.answerable_from_menu is True
    assert answerable_target.evidence_ids == answerable["cited_ids"]
    assert decoy_target is not None
    assert decoy_target.answerable_from_menu is False
    assert decoy_target.evidence_ids == []
    assert clean_target is not None
    assert clean_target.answerable_from_menu is False
    assert clean_target.evidence_ids == []


def test_exposure_sequence_is_deterministic_and_accounted() -> None:
    cfg = MatchedSFTConfig()
    pool, _ = build_exact_grpo_pool(seed=42)
    exposure1, manifest1, rows1 = build_exposure_sequence(pool, cfg)
    exposure2, manifest2, rows2 = build_exposure_sequence(pool, cfg)

    assert len(exposure1) == 600
    assert manifest1["optimizer_steps"] == 150
    assert manifest1["supervised_examples_per_optimizer_step"] == 4
    assert manifest1["same_as_run2_encountered_prompt_sequence"] is False
    assert [r["record_id"] for r in exposure1] == [r["record_id"] for r in exposure2]
    assert manifest1 == manifest2
    assert rows1 == rows2


def test_effective_config_records_fresh_training_state() -> None:
    cfg = MatchedSFTConfig()
    resolved = effective_config(
        cfg,
        adapter_report={"adapter_model_sha256": "x"},
        pool_manifest={"n_total": 5000},
        exposure_manifest={"n_examples": 600},
    )

    assert resolved["fresh_optimizer_scheduler"] is True
    assert resolved["resume_from_checkpoint"] is None
    assert resolved["old_optimizer_state_loaded"] is False
    assert resolved["fixed_final_checkpoint"] == 150
    assert resolved["does_not_claim_identical_training_signal_to_grpo"] is True
    assert resolved["does_not_claim_exact_same_600_prompts_as_run2"] is True


def test_sequential_sampler_hook_accepts_transformers_dataset_argument() -> None:
    class DummyTrainer(SequentialExposureTrainerMixin):
        train_dataset = ["fallback"]

    sampler = DummyTrainer()._get_train_sampler(["a", "b", "c"])
    assert list(iter(sampler)) == [0, 1, 2]


class _Param:
    def __init__(self, n: int, requires_grad: bool):
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class _FakeModel:
    def __init__(self) -> None:
        self.config = types.SimpleNamespace(use_cache=True)
        self._params = [
            ("base_model.model.language.layers.0.self_attn.q_proj.base_layer.weight", _Param(100, False)),
            ("base_model.model.language.layers.0.self_attn.q_proj.lora_A.default.weight", _Param(10, True)),
            ("base_model.model.language.layers.0.self_attn.q_proj.lora_B.default.weight", _Param(10, True)),
            ("base_model.model.visual.blocks.0.weight", _Param(50, False)),
        ]

    def named_parameters(self):
        return iter(self._params)

    def parameters(self):
        return (param for _, param in self._params)


def test_trainable_parameter_report_rejects_non_lora_trainables() -> None:
    model = _FakeModel()
    report = trainable_parameter_report(model)
    assert report["trainable_params"] == 20
    assert report["n_trainable_tensors"] == 2

    model._params.append(("base_model.model.language.embed_tokens.weight", _Param(5, True)))
    with pytest.raises(RuntimeError, match="non-LoRA"):
        trainable_parameter_report(model)


def test_trainable_adapter_loader_requests_trainable_peft_adapter(monkeypatch, tmp_path: Path) -> None:
    adapter = tmp_path / "checkpoint-642"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "fake"}))
    (adapter / "adapter_model.safetensors").write_bytes(b"fake")

    calls = {}
    fake_model = _FakeModel()
    sentinel_bnb = object()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["from_pretrained"] = {"model_id": model_id, **kwargs}
            return fake_model

    class FakePeftModel:
        @classmethod
        def from_pretrained(cls, base, model_id, **kwargs):
            calls["peft_from_pretrained"] = {"base": base, "model_id": model_id, **kwargs}
            return fake_model

    def fake_prepare(model, use_gradient_checkpointing):
        calls["prepare"] = {
            "model": model,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }
        return model

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoModelForImageTextToText=FakeAutoModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftModel=FakePeftModel, prepare_model_for_kbit_training=fake_prepare),
    )
    import scripts.train_sft_matched_pool as matched

    monkeypatch.setattr(matched, "build_bnb_config", lambda: sentinel_bnb)
    monkeypatch.setattr(matched, "summarize_trainable", lambda model: {"ok": True})

    cfg = MatchedSFTConfig(model_id="fake-base")
    loaded = load_trainable_adapter(cfg, adapter)

    assert loaded is fake_model
    assert calls["from_pretrained"]["model_id"] == "fake-base"
    assert calls["from_pretrained"]["quantization_config"] is sentinel_bnb
    assert calls["from_pretrained"]["device_map"] == "auto"
    assert calls["prepare"]["use_gradient_checkpointing"] is True
    assert calls["peft_from_pretrained"]["model_id"] == str(adapter)
    assert calls["peft_from_pretrained"]["is_trainable"] is True
