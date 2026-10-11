"""Piecewise evidence reward — the single source of truth, dependency-free.

This is the canonical evidence reward (designed in ``phase0/REWARD_DESIGN.md``):
pure integer-set math, no pandas/pytesseract/anthropic/torch. It lives in its own
module so the LEAN training/eval path (``scripts/eval_scorer.py``) can import the
reward without pulling in ``phase0.pipeline``'s heavy audit dependencies.

``phase0.pipeline`` re-exports these names, so existing
``from phase0.pipeline import compute_evidence_reward`` imports keep working.
"""

from __future__ import annotations

ABSTAIN_CREDIT      =  0.4   # answerable + cited=[]    : "honest abstain" credit
FABRICATION_PENALTY = -0.5   # answerable + any id ∉ valid_ids : strict fabrication penalty


def compute_evidence_reward(
    cited_ids: list[int],
    gold_cited_ids: set[int],
    valid_ids: set[int],
    is_answerable: bool,
) -> dict:
    """Piecewise evidence reward, score range [FABRICATION_PENALTY, 1.0].

    The reward decomposes the model's grounding behavior into four mutually
    exclusive regimes so that honest abstention strictly outscores fabrication.
    With ABSTAIN_CREDIT=+0.4 and FABRICATION_PENALTY=-0.5, the canonical
    img2-style composite (answer 0.50, evidence 0.25, format 0.10, refusal 0.15,
    correct answer + valid format) gives 0.85 (valid cite) > 0.70 (honest empty)
    > 0.475 (fabricated cite) — the ordering the v1 reward failed to produce.

    Inputs
    ------
    cited_ids      : IDs the model cited (may include fabricated out-of-range IDs)
    gold_cited_ids : reference IDs the localizer placed (subset of valid_ids)
    valid_ids      : the full OCR token ID set for this call
    is_answerable  : True iff the question is answerable from this image — i.e.,
                     the contract requires a non-empty grounded citation. False
                     for masked-evidence and out-of-domain refusal cases.

    Regimes (priority order)
    ------------------------
        is_answerable == False (refusal-expected):
            cited == ∅  →  reward = 1.0       correct refusal
            cited != ∅  →  reward = 0.0       over-citation on unanswerable
                                              (citation validity is not differentiated
                                               here — the wrongness is "didn't refuse")

        is_answerable == True:
            cited == ∅                  →  reward = ABSTAIN_CREDIT      (+0.4)
              "honest abstain" — the model returned [] rather than fabricating
              IDs to satisfy the schema.

            any cited id ∉ valid_ids    →  reward = FABRICATION_PENALTY (-0.5)
              At least one cited ID is not in the OCR menu — pure fabrication /
              out-of-range, the canonical reward-hacking failure mode. Strictly
              worse than ABSTAIN_CREDIT so the model is never incentivized to
              hallucinate evidence over admitting it has none.

            all cited ids ∈ valid_ids   →  F1 set-overlap vs gold_cited_ids
                precision = |cited ∩ gold| / |cited|
                recall    = |cited ∩ gold| / |gold|
                reward    = 2·P·R / (P+R)  if P+R > 0 else 0.0

              If gold == ∅ here (localizer placed no gold) the F1 branch returns
              0.0; treat this as a localization-infrastructure artifact, not a
              model grounding signal.

    Returns
    -------
    dict with keys:
        evidence_reward    : float in [FABRICATION_PENALTY, 1.0]
        evidence_precision : float | None  — populated only in the F1 branch
        evidence_recall    : float | None  — populated only in the F1 branch
    """
    cited = set(cited_ids)
    gold  = set(gold_cited_ids)

    # Refusal-expected branches
    if not is_answerable:
        reward = 1.0 if not cited else 0.0
        return {"evidence_reward": reward,
                "evidence_precision": None, "evidence_recall": None}

    # Answerable: honest abstain
    if not cited:
        return {"evidence_reward": ABSTAIN_CREDIT,
                "evidence_precision": None, "evidence_recall": None}

    # Answerable: fabrication (strictly below ABSTAIN_CREDIT by construction)
    if not cited.issubset(valid_ids):
        return {"evidence_reward": FABRICATION_PENALTY,
                "evidence_precision": None, "evidence_recall": None}

    # Answerable + all cited valid: F1 set-overlap branch
    if not gold:
        return {"evidence_reward": 0.0, "evidence_precision": 0.0, "evidence_recall": 0.0}

    intersection = cited & gold
    precision = len(intersection) / len(cited)
    recall    = len(intersection) / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "evidence_reward":    round(f1,        4),
        "evidence_precision": round(precision, 4),
        "evidence_recall":    round(recall,    4),
    }
