"""Guards the per-group-member reward log.

`run4_analysis_plan.md` §(amendment) registers `frac_reward_zero_std` as a monitored
quantity **split by abstain vs answerable side**. That split cannot be computed from
what TRL emits: `reward`/`reward_std` in reward_log.jsonl are group aggregates, and the
completions parquet is capped by `num_completions_to_print` (8 of 32 rows/step observed
in run 2 = 25% of completions). `make_reward_fn(..., member_log=...)` writes every
completion, so these three properties must hold:

  1. EVERY completion is logged — not a sample;
  2. each row carries `is_abstain`, so the registered split is computable;
  3. a logging failure never propagates — a paid run must not die because a write failed.
"""
from __future__ import annotations

import json

import pytest

from grpo.train_grpo import make_reward_fn


def _records():
    """Two abstain rows and two answerable, shaped enough for classify()."""
    menu = [{"id": 0, "text": "Bone", "bbox": [0, 0, 10, 10]},
            {"id": 1, "text": "1978", "bbox": [0, 10, 10, 20]}]
    return {
        "a1_mfab": {"record_id": "a1_mfab", "doc_id": 1, "split": "train", "target_is_abstain": True, "gold_localized": False,
                    "cited_ids": [], "ocr_menu": menu, "gold_answers": ["Bone"],
                    "answer_recurs_in_menu": False},
        "a2_mfab": {"record_id": "a2_mfab", "doc_id": 2, "split": "train", "target_is_abstain": True, "gold_localized": False,
                    "cited_ids": [], "ocr_menu": menu, "gold_answers": ["Bone"],
                    "answer_recurs_in_menu": True},
        "q1": {"record_id": "q1", "doc_id": 3, "split": "train", "target_is_abstain": False, "gold_localized": True,
               "cited_ids": [0], "ocr_menu": menu, "gold_answers": ["Bone"]},
        "q2": {"record_id": "q2", "doc_id": 4, "split": "train", "target_is_abstain": False, "gold_localized": True,
               "cited_ids": [1], "ocr_menu": menu, "gold_answers": ["1978"]},
    }


ABSTAIN_OUT = '{"found_answer":"Bone","answerable_from_menu":false,"evidence_ids":[]}'
CITE_OUT = '{"found_answer":"Bone","answerable_from_menu":true,"evidence_ids":[0]}'


def test_matching_reward_batch_lengths_are_accepted() -> None:
    fn = make_reward_fn(2, _records())
    rewards = fn(
        prompts=[None, None],
        completions=[ABSTAIN_OUT, CITE_OUT],
        record_id=["a1_mfab", "q1"],
    )
    assert len(rewards) == 2


def test_missing_completion_fails_loudly() -> None:
    fn = make_reward_fn(2, _records())
    with pytest.raises(ValueError, match=r"prompts=2, completions=1, record_id=2"):
        fn(prompts=[None, None], completions=[ABSTAIN_OUT],
           record_id=["a1_mfab", "q1"])


def test_missing_record_id_column_fails_loudly() -> None:
    fn = make_reward_fn(2, _records())
    with pytest.raises(RuntimeError, match="record_id did not route"):
        fn(prompts=[None], completions=[ABSTAIN_OUT])


def test_extra_completion_fails_loudly() -> None:
    fn = make_reward_fn(2, _records())
    with pytest.raises(ValueError, match=r"prompts=2, completions=3, record_id=2"):
        fn(prompts=[None, None], completions=[ABSTAIN_OUT, CITE_OUT, CITE_OUT],
           record_id=["a1_mfab", "q1"])


def test_extra_record_id_fails_loudly() -> None:
    fn = make_reward_fn(2, _records())
    with pytest.raises(ValueError, match=r"prompts=2, completions=2, record_id=3"):
        fn(prompts=[None, None], completions=[ABSTAIN_OUT, CITE_OUT],
           record_id=["a1_mfab", "q1", "q2"])


def test_every_completion_is_logged_not_a_sample(tmp_path):
    recs = _records()
    log = tmp_path / "reward_members.jsonl"
    fn = make_reward_fn(2, recs, member_log=log)

    ids = ["a1_mfab"] * 8 + ["q1"] * 8          # two groups of G=8 = 32-style batch
    comps = [ABSTAIN_OUT] * 8 + [CITE_OUT] * 8
    rewards = fn(prompts=[None] * 16, completions=comps, record_id=ids)

    assert len(rewards) == 16
    rows = [json.loads(l) for l in log.open()]
    assert len(rows) == 16, f"expected every completion logged, got {len(rows)}"
    assert [r["record_id"] for r in rows] == ids
    assert [r["reward"] for r in rows] == rewards, "logged reward must equal the returned one"


def test_rows_carry_the_abstain_split_and_group_identity(tmp_path):
    recs = _records()
    log = tmp_path / "m.jsonl"
    fn = make_reward_fn(2, recs, member_log=log)
    fn(prompts=[None] * 4, completions=[ABSTAIN_OUT] * 2 + [CITE_OUT] * 2,
       record_id=["a1_mfab", "a2_mfab", "q1", "q2"])
    fn(prompts=[None] * 2, completions=[ABSTAIN_OUT] * 2, record_id=["a1_mfab", "a1_mfab"])

    rows = [json.loads(l) for l in log.open()]
    assert {"call", "i", "record_id", "is_abstain", "recurs", "bucket", "reward"} <= set(rows[0])
    # the registered split is computable
    assert sum(r["is_abstain"] for r in rows) == 4
    assert sum(not r["is_abstain"] for r in rows) == 2
    # decoy rows are distinguishable within the abstain side
    assert [r["recurs"] for r in rows if r["record_id"] == "a2_mfab"] == [True]
    # groups are separable across calls, so within-group variance is recoverable
    assert sorted({r["call"] for r in rows}) == [0, 1]
    assert len([r for r in rows if r["call"] == 1]) == 2


def test_logging_failure_never_kills_the_run(tmp_path):
    """A paid run must survive a broken log path — rewards still return."""
    recs = _records()
    unwritable = tmp_path / "nope" / "deep" / "m.jsonl"   # parent does not exist
    fn = make_reward_fn(2, recs, member_log=unwritable)
    rewards = fn(prompts=[None], completions=[ABSTAIN_OUT], record_id=["a1_mfab"])
    assert len(rewards) == 1 and isinstance(rewards[0], float)


def test_no_member_log_means_no_file_and_same_rewards(tmp_path):
    """Default (runs 1-2) behaviour is unchanged: no log, identical rewards."""
    recs = _records()
    log = tmp_path / "m.jsonl"
    with_log = make_reward_fn(2, recs, member_log=log)
    without = make_reward_fn(2, recs)
    ids = ["a1_mfab", "q1"]
    comps = [ABSTAIN_OUT, CITE_OUT]
    assert with_log(prompts=[None] * 2, completions=comps, record_id=ids) == \
           without(prompts=[None] * 2, completions=comps, record_id=ids)
