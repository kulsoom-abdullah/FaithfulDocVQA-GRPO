"""GRPO reward table — one case per bucket per run, plus the two control properties.

The reward reuses classify() for the bucket and compute_evidence_reward (via classify's
facts) for the valid_citation F1, so these cases also confirm the fixtures land in the
intended bucket before asserting the reward value.
"""
import json

import pytest

from grpo.reward import compute_grpo_reward
from scripts.eval_scorer import (
    classify,
    VALID_CITATION, FABRICATION, FALSE_ABSTAIN, HONEST_ABSTAIN,
    FALSE_GROUNDING, CONTRADICTION, MALFORMED,
)


def rec(rid, target_is_abstain, gold_localized, menu_ids, gold_cited, gold_answers):
    """One ANSWER KEY (ground truth) — same shape as tests/test_classify_cases.py::rec."""
    return {
        "record_id": rid, "doc_id": f"doc_{rid}", "doc_type": "form",
        "target_is_abstain": target_is_abstain, "gold_localized": gold_localized,
        "ocr_menu": [{"id": i} for i in menu_ids],
        "cited_ids": gold_cited, "gold_answers": gold_answers,
    }


def out(found_answer, flag, evidence_ids):
    """The MODEL's raw 3-field completion."""
    return json.dumps({"found_answer": found_answer,
                       "answerable_from_menu": flag, "evidence_ids": evidence_ids})


# fixtures: found_answer is CORRECT in every one (gold_answers=["ans"]) -> fa = +0.3
FIX = {
    "valid_citation":  (rec(1, False, True,  range(10), [0, 1, 2], ["ans"]), out("ans", True,  [0, 1, 2]), VALID_CITATION),
    "fabrication":     (rec(2, False, True,  range(10), [5],       ["ans"]), out("ans", True,  [99]),      FABRICATION),
    "false_abstain":   (rec(3, False, True,  range(10), [2, 3],    ["ans"]), out("ans", False, []),        FALSE_ABSTAIN),
    "honest_abstain":  (rec(4, True,  False, range(10), [],        ["ans"]), out("ans", False, []),        HONEST_ABSTAIN),
    "false_grounding": (rec(5, True,  False, range(10), [],        ["ans"]), out("ans", True,  [4]),       FALSE_GROUNDING),
    "contradiction":   (rec(6, False, True,  range(10), [3],       ["ans"]), out("ans", True,  []),        CONTRADICTION),
    "malformed":       (rec(7, False, True,  range(10), [1],       ["ans"]), "not json at all",            MALFORMED),
}

# expected reward per (bucket, run) with fa=+0.3.  valid_citation F1 = 1.0 (cited==gold).
EXPECT = {
    ("valid_citation", 1): 1.3,  ("valid_citation", 2): 1.3,
    ("fabrication", 1): -0.2,    ("fabrication", 2): -0.2,
    ("false_abstain", 1): 0.3,   ("false_abstain", 2): 0.3,
    ("honest_abstain", 1): 0.3,  ("honest_abstain", 2): 0.8,
    ("false_grounding", 1): 0.3, ("false_grounding", 2): -0.3,
    ("contradiction", 1): -1.0,  ("contradiction", 2): -1.0,
    ("malformed", 1): -1.0,      ("malformed", 2): -1.0,
}


@pytest.mark.parametrize("name,run", [(n, r) for n in FIX for r in (1, 2)])
def test_bucket_per_run(name, run):
    record, raw, exp_bucket = FIX[name]
    assert classify(record, raw)["bucket"] == exp_bucket           # fixture lands in the intended bucket
    assert compute_grpo_reward(record, raw, run) == pytest.approx(EXPECT[(name, run)], abs=1e-9)


def test_run1_control_honest_abstain_eq_false_grounding():
    """CONTROL: at equal fa, run-1 honest_abstain reward == run-1 false_grounding reward
    (run 1 has no abstain shaping — both bases 0.0)."""
    ha = compute_grpo_reward(FIX["honest_abstain"][0], FIX["honest_abstain"][1], 1)
    fg = compute_grpo_reward(FIX["false_grounding"][0], FIX["false_grounding"][1], 1)
    assert ha == pytest.approx(fg, abs=1e-9)


def test_run2_honest_vs_false_grounding_differential_is_1_1():
    """run-2 differential honest_abstain - false_grounding == 0.5 - (-0.6) == 1.1 (equal fa)."""
    ha2 = compute_grpo_reward(FIX["honest_abstain"][0], FIX["honest_abstain"][1], 2)
    fg2 = compute_grpo_reward(FIX["false_grounding"][0], FIX["false_grounding"][1], 2)
    assert (ha2 - fg2) == pytest.approx(1.1, abs=1e-9)


def test_fa_zero_additivity_and_control_hold():
    """found_answer WRONG -> fa=0; the control (run-1 equal) and run-2 differential still hold,
    confirming fa is a clean additive axis independent of the bucket base."""
    ha, ho = rec(8, True, False, range(10), [], ["ans"]), out("WRONG", False, [])
    fg, fo = rec(9, True, False, range(10), [], ["ans"]), out("WRONG", True, [4])
    assert compute_grpo_reward(ha, ho, 1) == pytest.approx(0.0, abs=1e-9)   # honest_abstain, fa=0
    assert compute_grpo_reward(fg, fo, 1) == pytest.approx(0.0, abs=1e-9)   # false_grounding, fa=0 (== control)
    assert compute_grpo_reward(ha, ho, 2) == pytest.approx(0.5, abs=1e-9)
    assert compute_grpo_reward(fg, fo, 2) == pytest.approx(-0.6, abs=1e-9)
    assert (compute_grpo_reward(ha, ho, 2) - compute_grpo_reward(fg, fo, 2)) == pytest.approx(1.1, abs=1e-9)


def test_invalid_run_raises():
    with pytest.raises(ValueError):
        compute_grpo_reward(FIX["valid_citation"][0], FIX["valid_citation"][1], 3)
