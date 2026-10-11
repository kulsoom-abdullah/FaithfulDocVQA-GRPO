"""REQUIRED test (CLAUDE.md rule 4): completion-only loss mask.

On one real formatted example, build labels and assert the PROMPT region is
masked to -100 and the COMPLETION region carries the actual token ids. Runs on
CPU — it uses the processor + one image, no model weights / no GPU.

This guards the failure mode where prompt tokens leak into the loss (which
silently happened before with a TRL flag). Do not remove.

Prereqs: data/images/ materialized (scripts/build_master_dataset.py) plus a
one-time HF processor download — on a fresh clone without images this test
fails on the missing file rather than skipping; rebuild the images first.
"""

import torch
import pytest

from scripts.format_sft import load_master, load_processor, _v3_target
from scripts.train_sft import tokenize_example


@pytest.fixture(scope="module")
def processor():
    proc, _ = load_processor()
    return proc


@pytest.fixture(scope="module")
def records():
    return load_master()


def _example(records):
    """A compact-menu answerable record (cheap to tokenize, non-trivial target)."""
    ans = sorted((r for r in records if not r["target_is_abstain"]),
                 key=lambda r: len(r["ocr_menu"]))
    return next(r for r in ans if len(r["ocr_menu"]) >= 5)


def test_loss_mask_prompt_is_minus100_completion_is_ids(processor, records):
    rec = _example(records)
    out = tokenize_example(processor, rec)
    input_ids = out["input_ids"]
    labels = out["labels"]
    prompt_len = out["_prompt_len"]

    assert labels.shape == input_ids.shape
    # prompt region fully masked
    assert torch.all(labels[:prompt_len] == -100), "prompt tokens not masked to -100"
    # completion region equals the input ids (contributes to loss)
    assert torch.all(labels[prompt_len:] == input_ids[prompt_len:]), \
        "completion labels do not match input ids"
    # no -100 leaks into the completion region
    assert torch.all(labels[prompt_len:] != -100), "a -100 leaked into the completion"
    # the masked count is exactly the prompt length (contiguous prefix mask)
    assert int((labels == -100).sum()) == prompt_len


def test_completion_region_decodes_to_target(processor, records):
    rec = _example(records)
    out = tokenize_example(processor, rec)
    input_ids = out["input_ids"]
    prompt_len = out["_prompt_len"]
    completion = processor.tokenizer.decode(input_ids[prompt_len:])
    # the exact 3-field JSON target is contained in the trained region
    assert _v3_target(rec) in completion
    # and the turn-end marker is trained too (model must learn to stop)
    assert "<|im_end|>" in completion


def test_abstain_example_masks_correctly(processor, records):
    """The mask logic also holds for an abstain target."""
    rec = next(r for r in records if r["target_is_abstain"])
    out = tokenize_example(processor, rec)
    input_ids, labels, prompt_len = out["input_ids"], out["labels"], out["_prompt_len"]
    assert torch.all(labels[:prompt_len] == -100)
    assert torch.all(labels[prompt_len:] == input_ids[prompt_len:])
    completion = processor.tokenizer.decode(input_ids[prompt_len:])
    assert _v3_target(rec) in completion                       # 3-field abstain target
    assert '"answerable_from_menu":false' in completion
