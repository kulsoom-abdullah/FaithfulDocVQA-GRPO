"""CPU tests for scripts/format_sft.py data filtering and menu rendering.

These avoid loading the processor (no network/model needed); the processor-based
rendering is exercised by scripts/format_sft.py's CLI demo and by
scripts/train_sft.py's tokenization path (tests/test_loss_mask.py).
"""

import json
from pathlib import Path

import pytest

from scripts.format_sft import (
    load_master, load_split, render_menu, build_messages, _v3_target,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return load_master()


def test_train_mode_filter(records):
    """On TRAIN: A ⊆ B, A is exactly the answerable records, B−A is the abstains."""
    a = load_split("train", "A", records)
    b = load_split("train", "B", records)
    a_ids = {r["record_id"] for r in a}
    b_ids = {r["record_id"] for r in b}
    assert b_ids == {r["record_id"] for r in records if r["split"] == "train"}
    assert a_ids <= b_ids
    assert all(r["gold_localized"] is True and not r["target_is_abstain"] for r in a)
    assert (len(b) - len(a)) == sum(1 for r in b if r["target_is_abstain"])


def test_eval_splits_are_shared_and_full(records):
    """CRITICAL: validation and test are FULL (abstains included) and IDENTICAL for
    A and B — the mode filter must not leak into the held-out sets."""
    for split in ("validation", "test"):
        a = load_split(split, "A", records)
        b = load_split(split, "B", records)
        assert {r["record_id"] for r in a} == {r["record_id"] for r in b}, \
            f"{split} differs between A and B — comparison would be invalid"
        assert {r["record_id"] for r in a} == \
            {r["record_id"] for r in records if r["split"] == split}      # full split
        assert any(r["target_is_abstain"] for r in a), f"{split} has no abstains"


def test_load_split_rejects_bad_args(records):
    with pytest.raises(ValueError):
        load_split("train", "C", records)
    with pytest.raises(ValueError):
        load_split("holdout", "A", records)
    with pytest.raises(ValueError):
        load_split("eval", "B", records)        # old 2-way name no longer valid


def test_render_menu_matches_ids_and_text(records):
    """Rendered menu has one line per token, each carrying id, text, and bbox."""
    r = records[0]
    menu = r["ocr_menu"]
    s = render_menu(menu)
    lines = s.split("\n")
    assert len(lines) == len(menu)
    for line, m in zip(lines, menu):
        parts = line.split("\t")
        assert parts[0] == str(m["id"])
        assert parts[1] == m["text"]
        x1, y1, x2, y2 = m["bbox"]
        assert parts[2] == f"{x1},{y1},{x2},{y2}"


def test_build_messages_structure(records):
    """Messages carry image+question+menu in the user turn; target only when asked."""
    r = next(rec for rec in records if not rec["target_is_abstain"])
    prompt_only = build_messages(r, include_target=False)
    assert [m["role"] for m in prompt_only] == ["system", "user"]
    user_content = prompt_only[1]["content"]
    assert any(c.get("type") == "image" for c in user_content)
    text_blk = next(c["text"] for c in user_content if c.get("type") == "text")
    assert r["question"] in text_blk and "OCR tokens:" in text_blk

    with_target = build_messages(r, include_target=True)
    assert [m["role"] for m in with_target] == ["system", "user", "assistant"]
    assert with_target[2]["content"] == _v3_target(r)
    assert '"found_answer"' in with_target[2]["content"]   # 3-field contract
