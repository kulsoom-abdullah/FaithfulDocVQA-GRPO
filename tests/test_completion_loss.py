"""CPU test for the completion-only loss (logits_to_keep) used in SFT training.

Asserts (1) the memory-bounded `logits_to_keep` loss is bit-for-bit identical to a
full-sequence completion-only cross-entropy, and (2) it requires per-device batch
size 1. No GPU / no model weights — a tiny stub model stands in for the real
*ForCausalLM and returns exactly what `logits_to_keep` would (the last N rows).
"""

import pytest
import torch
import torch.nn.functional as F

from scripts.train_sft import completion_only_loss


class _StubOutput:
    def __init__(self, logits):
        self.logits = logits


class _StubModel:
    """Mimics a *ForCausalLM with `logits_to_keep`: returns the last N logit rows of
    a fixed full-sequence logits tensor (exactly what the real model computes)."""

    def __init__(self, full_logits):
        self.full_logits = full_logits

    def __call__(self, *, logits_to_keep, **inputs):
        return _StubOutput(self.full_logits[:, -logits_to_keep:, :])


def test_logits_to_keep_equals_full_completion_only_ce():
    torch.manual_seed(0)
    seq, prompt_len, vocab = 12, 8, 7
    input_ids = torch.randint(0, vocab, (1, seq))
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100                       # completion = last 4 tokens
    full_logits = torch.randn(1, seq, vocab)

    # reference: standard HF-style completion-only CE over the FULL sequence
    ref = F.cross_entropy(
        full_logits[:, :-1, :].reshape(-1, vocab),
        labels[:, 1:].reshape(-1), ignore_index=-100,
    )

    # the real code under test (memory-bounded path, stub model)
    mine, outputs = completion_only_loss(
        _StubModel(full_logits), {"input_ids": input_ids}, labels)

    assert torch.allclose(ref, mine, atol=1e-6), f"{ref.item()} != {mine.item()}"
    # the model only ever produced logits for the completion tail (n_comp + 1 rows)
    assert outputs.logits.shape[1] == int((labels[0] != -100).sum()) + 1


def test_requires_per_device_batch_size_one():
    labels = torch.zeros(2, 5, dtype=torch.long)        # batch size 2 -> must raise
    with pytest.raises(ValueError):
        completion_only_loss(
            _StubModel(torch.randn(2, 5, 7)),
            {"input_ids": torch.zeros(2, 5, dtype=torch.long)},
            labels,
        )
