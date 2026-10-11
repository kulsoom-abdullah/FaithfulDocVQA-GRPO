"""GRPO scalar reward — bucket-based, staged. Pure CPU, NO new classification logic.

``compute_grpo_reward(record, raw_output, run)`` reuses:
  - ``scripts.eval_scorer.classify`` for the bucket AND the ``found_answer`` axis, and
  - the ``valid_citation`` F1 comes from ``compute_evidence_reward`` via
    ``classify``'s ``facts["evidence_reward"]`` (no re-classification, no re-scoring).

``run`` selects the staged rollout (DECISIONS §4 / behavior_axes_reference):
  - run 1 = BASELINE — no abstain shaping (honest_abstain and false_grounding both 0.0
    base), so we first debug the RL loop on the axes that already work.
  - run 2 = adds the abstain shaping: reward honest_abstain, penalize false_grounding
    at ``>= fabrication`` magnitude (the dangerous, inference-undetectable failure).

``found_answer`` correctness is a SEPARATE additive axis (anti-collapse: the model must
keep reading the image), applied to every bucket EXCEPT the flat format-level ones
(contradiction / malformed). Magnitudes below are the agreed starting point — the
ORDERING is fixed (group-relative advantage cares about within-prompt ordering); the
absolute values are to be tuned empirically during the staged rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_scorer import (
    classify,
    VALID_CITATION, FABRICATION, FALSE_ABSTAIN, HONEST_ABSTAIN,
    FALSE_GROUNDING, CONTRADICTION, MALFORMED,
)

# ── reward constants (starting point; magnitudes tuned during the staged rollout) ──────
FA_BONUS              =  0.3   # + when found_answer_correct: rewards image-use; anti-collapse (additive on non-flat buckets)
FABRICATION_PENALTY   = -0.5   # cited an out-of-menu id — strictly worse than any honest behavior
FALSE_ABSTAIN_BASE    =  0.0   # refused a citable answer — neutral base (over-abstention loses only the F1 it could have earned)
HONEST_ABSTAIN_RUN2   =  0.5   # run 2 rewards the win (correct refusal); run 1 = 0.0 (no abstain shaping yet)
FALSE_GROUNDING_RUN2  = -0.6   # run 2 penalizes the dangerous, inference-undetectable failure at >= fabrication; run 1 = 0.0
CONTRADICTION_PENALTY = -1.0   # flat, no found_answer bonus — internal flag<->evidence inconsistency (format-level)
MALFORMED_PENALTY     = -1.0   # flat, no found_answer bonus — unparseable output


def compute_grpo_reward(record: dict, raw_output: str | None, run: int,
                        false_abstain_base: float = FALSE_ABSTAIN_BASE) -> float:
    """Scalar reward for ONE sampled completion. ``run`` in {1, 2} selects the staged shaping.

    Args:
        record:     the frozen record (its gold fields drive the bucket via classify).
        raw_output: the model's raw completion string (parsed by classify).
        run:        1 (baseline) or 2 (with abstain shaping).
        false_abstain_base: base reward for the FALSE_ABSTAIN bucket. Defaults to the
            module constant (0.0), which reproduces runs 1, 2 and 4a byte-for-byte.
            Run 4b prices this negative (-0.2, run4_analysis_plan.md §c) to shift the
            refusal threshold. Passed explicitly rather than mutated at module level:
            editing the constant in place would retroactively change the rewards that
            already-committed runs reproduce from their frozen artifacts.
    Returns:
        float reward. Bucket base (per the finalized table) + found_answer bonus, except
        the flat format-level buckets which return a fixed penalty with no bonus.
    """
    if run not in (1, 2):
        raise ValueError(f"run must be 1 or 2, got {run!r}")

    facts = classify(record, raw_output)          # bucket + found_answer_correct + evidence_reward (F1)
    bucket = facts["bucket"]

    # flat, format-level buckets: no found_answer bonus
    if bucket == CONTRADICTION:
        return float(CONTRADICTION_PENALTY)
    if bucket == MALFORMED:
        return float(MALFORMED_PENALTY)

    fa = FA_BONUS if facts["found_answer_correct"] else 0.0

    if bucket == VALID_CITATION:
        base = facts["evidence_reward"]           # F1 from compute_evidence_reward (via classify) — no re-score
    elif bucket == FABRICATION:
        base = FABRICATION_PENALTY
    elif bucket == FALSE_ABSTAIN:
        base = false_abstain_base
    elif bucket == HONEST_ABSTAIN:
        base = HONEST_ABSTAIN_RUN2 if run == 2 else 0.0
    elif bucket == FALSE_GROUNDING:
        base = FALSE_GROUNDING_RUN2 if run == 2 else 0.0
    else:
        raise ValueError(
            f"unmapped bucket {bucket!r}: the reward table covers 7 buckets "
            "(uncited_answer is unreachable under the 3-field contract — answerable_from_menu "
            "is a required bool)."
        )

    return float(base + fa)
