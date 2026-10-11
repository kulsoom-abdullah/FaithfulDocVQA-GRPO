"""Faithfulness eval scorer for FaithfulDocVQA — 3-field contract.

PURE, CPU-only. Given the frozen records and one raw model output per record, it
parses the 3-field output {found_answer, answerable_from_menu, evidence_ids},
classifies each into a bucket, scores the found_answer (image-use) axis, and
aggregates the metrics. The model is NOT loaded here — the GPU runner feeds
``{record_id: raw_output}``.

STATE — COMPLETE
----------------
``classify()`` is IMPLEMENTED (the bucket cascade), and everything around it is built:
parse handling (``parse_tolerant_v3``), the bucket constants, the found_answer
correctness axis, and the aggregate with the **fabrication gate fixed** (aggregate
citation-F1 is computed ONLY over VALID_CITATION records, so a fabricated/
contradictory/false-grounding record no longer leaks partial F1 — it matches the
per-record reward's "fabrication ⇒ no F1" rule). ``_facts()`` pre-computes every
signal ``classify()`` routes on. Covered by ``tests/test_classify_cases.py``.

Design notes
------------
- Reuse, don't reimplement: the per-record piecewise reward is
  ``phase0.evidence_reward.compute_evidence_reward``; parsing is
  ``faithful_io.parse_tolerant_v3``.
- Two abstention axes are now distinct: MENU attribution (``answerable_from_menu``)
  drives the abstain/grounding buckets; visual PERCEPTION (``found_answer``) is its
  own correctness axis, scored separately and never folded into the citation logic.
- No silent failure: a missing generation or a doc_id absent from the cap manifest
  raises rather than being miscounted.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from phase0.evidence_reward import compute_evidence_reward
from scripts.faithful_io import ABSTAIN_ANSWER, parse_tolerant_v3

# ── Buckets (the finalized scoring table's rows) ──────────────────────────────
VALID_CITATION = "valid_citation"     # grounded: answerable_from_menu=true, cites, no fabrication
FABRICATION = "fabrication"           # cited an id not in the menu (gates F1 — checked first)
CONTRADICTION = "contradiction"       # flag↔evidence inconsistent (parse_failure == "contradiction")
FALSE_GROUNDING = "false_grounding"   # claims menu-citable when gold says it is NOT (dangerous case)
HONEST_ABSTAIN = "honest_abstain"     # gold not menu-citable AND model says answerable_from_menu=false
FALSE_ABSTAIN = "false_abstain"       # gold IS menu-citable but model says answerable_from_menu=false
UNCITED_ANSWER = "uncited_answer"     # (table-defined) answered without a usable citation
MALFORMED = "malformed"               # unparseable output (non-contradiction parse failure)

BUCKETS = (VALID_CITATION, FABRICATION, CONTRADICTION, FALSE_GROUNDING,
           HONEST_ABSTAIN, FALSE_ABSTAIN, UNCITED_ANSWER, MALFORMED)


# ── found_answer correctness (the image-use axis) ─────────────────────────────
# NOTE: this NFC + curly/straight-quote-fold + casefold matcher is duplicated in
# scripts/eval_ablation.py. Candidate to centralize in faithful_io later (same
# single-source move we did for derive_3field) — flagged, not done in 3b.

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        s = s.replace(a, b)
    return " ".join(s.split()).casefold()


def found_answer_correct(found_answer: str | None, gold_answers: list[str]) -> bool:
    """True iff the model's perceived answer matches any gold answer (the image-use
    axis — independent of citation/abstain). v1: the answer is always visible, so a
    ``found_answer == "unanswerable"`` here is simply wrong. (v2 masked-image rows,
    where gold-found_answer is "unanswerable", are not present in v1.)"""
    p = _norm(found_answer or "")
    return any(p == _norm(g) for g in gold_answers)


# ── fact extraction (scaffolding — everything classify() routes on) ───────────

def _facts(record: dict, raw_output: str | None) -> dict:
    """Parse the 3-field output and pre-compute every signal ``classify()`` needs.

    Returns the outcome dict with ``bucket`` left as None for ``classify()`` to set.
    Keys:
      record_id, doc_id, doc_type
      gold_class            "answerable" | "abstain"
      gold_localized        bool — the GOLD truth of menu-answerability (== not abstain in v1)
      gold_cited_ids        list[int] — the gold evidence ids
      parse_failure         None | "no_json"|"invalid_json"|"not_object"|"schema"|"contradiction"|"not_str"
      malformed             bool — parse produced no valid object
      found_answer          str | None   (model field)
      answerable_from_menu  bool | None  (model field — the menu-attribution flag)
      cited_ids             list[int]    (model field, evidence_ids)
      fabricated_ids        sorted list[int] — cited ids NOT in the menu
      found_answer_correct  bool | None — the image-use axis (None when malformed)
      evidence_reward/_precision/_recall — the piecewise per-record reward
      bucket                None  <- classify() assigns this
    """
    gold_abstain = bool(record["target_is_abstain"])
    gold_localized = bool(record["gold_localized"])
    valid_ids = {tok["id"] for tok in record["ocr_menu"]}
    gold_cited = set(record["cited_ids"])
    gold_answers = record.get("gold_answers", [])

    facts: dict[str, Any] = {
        "record_id": record["record_id"],
        "doc_id": record["doc_id"],
        "doc_type": record.get("doc_type"),
        "gold_class": "abstain" if gold_abstain else "answerable",
        "gold_localized": gold_localized,
        "gold_cited_ids": sorted(gold_cited),
        "parse_failure": None,
        "malformed": False,
        "found_answer": None,
        "answerable_from_menu": None,
        "cited_ids": [],
        "fabricated_ids": [],
        "found_answer_correct": None,
        "evidence_reward": None,
        "evidence_precision": None,
        "evidence_recall": None,
        "bucket": None,
    }

    res = parse_tolerant_v3(raw_output)          # handles non-str (→ failure "not_str")
    if res.value is None:
        facts["malformed"] = True
        facts["parse_failure"] = res.failure
        return facts

    out = res.value
    cited = list(out.evidence_ids)
    facts["found_answer"] = out.found_answer
    facts["answerable_from_menu"] = out.answerable_from_menu
    facts["cited_ids"] = cited
    facts["fabricated_ids"] = sorted(set(cited) - valid_ids)
    facts["found_answer_correct"] = found_answer_correct(out.found_answer, gold_answers)
    er = compute_evidence_reward(cited, gold_cited, valid_ids, is_answerable=not gold_abstain)
    facts["evidence_reward"] = er["evidence_reward"]
    facts["evidence_precision"] = er["evidence_precision"]
    facts["evidence_recall"] = er["evidence_recall"]
    return facts


def classify(record: dict, raw_output: str | None) -> dict:
    """Assign exactly one bucket from the finalized scoring table.

    Inputs:  ``record`` (a master/master_3field row) and ``raw_output`` (the model's
             raw string for it).
    Output:  the dict returned by ``_facts(record, raw_output)`` with ``bucket`` set
             to one of the BUCKETS constants. (``aggregate``/``score`` consume that.)

    The signals routed on are all in ``facts`` (see ``_facts`` docstring):
      gold_class / gold_localized, parse_failure / malformed, answerable_from_menu,
      cited_ids, fabricated_ids, found_answer_correct.

    Buckets are assigned in priority order:
      1. parse_failure == "contradiction"  → CONTRADICTION
         (flag↔evidence inconsistency is caught at parse; e.g. answerable_from_menu
          false with a non-empty cite, or true with empty.)
      2. malformed (any other parse_failure) → MALFORMED
      3. FABRICATION FIRST and it GATES F1: fabricated_ids non-empty → FABRICATION
         (do NOT compute F1 for these — handled by aggregate gating on VALID_CITATION).
      4. F1 path (→ VALID_CITATION) ONLY when: gold is menu-citable AND
         answerable_from_menu == true AND no fabrication.
      5. answerable_from_menu == false → gold-CONDITIONED abstain:
            gold_localized == false → HONEST_ABSTAIN ;  gold_localized == true → FALSE_ABSTAIN
      6. answerable_from_menu == true but gold_localized == false → FALSE_GROUNDING
         (claiming a menu grounding that the gold says does not exist — the dangerous case)
      7. UNCITED_ANSWER for the remaining table-defined "answered without a usable
         citation" case.
      found_answer correctness is a SEPARATE axis (facts["found_answer_correct"]) —
      it does NOT change the bucket; it is aggregated on its own.

    The cascade below implements exactly this priority order, sets
    ``facts["bucket"]``, and returns ``facts``. Covered by ``tests/test_classify_cases.py``.
    """
    facts = _facts(record, raw_output)

    if facts["parse_failure"] == "contradiction":
        facts["bucket"] = CONTRADICTION
    elif facts["malformed"]:
        facts["bucket"] = MALFORMED
    elif facts["fabricated_ids"]:            # non-empty list is truthy — no len()>0 needed
        facts["bucket"] = FABRICATION
    elif facts["gold_localized"] and facts["answerable_from_menu"]:
        facts["bucket"] = VALID_CITATION
    elif facts["answerable_from_menu"] is False:
        if facts["gold_localized"]:
            facts["bucket"] = FALSE_ABSTAIN      # gold WAS citable → refusing was wrong
        else:
            facts["bucket"] = HONEST_ABSTAIN     # gold NOT citable → refusing was right
    elif facts["answerable_from_menu"]:
        facts["bucket"] = FALSE_GROUNDING
    else:
        facts["bucket"] = UNCITED_ANSWER
        
    return facts


# ── aggregation helpers ───────────────────────────────────────────────────────

def _record_f1(cited: Iterable[int], gold: Iterable[int]) -> float:
    """Set-overlap F1 for one record (fabricated/out-of-menu ids hurt precision)."""
    c, g = set(cited), set(gold)
    if not c or not g:
        return 0.0
    inter = len(c & g)
    if inter == 0:
        return 0.0
    p = inter / len(c)
    r = inter / len(g)
    return 2 * p * r / (p + r)


def _mean(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) / len(vals) if vals else None


def _round_opt(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def _bucket_rates(subset: list[dict]) -> dict[str, float]:
    """Fraction of ``subset`` in each bucket (empty dict if subset empty)."""
    if not subset:
        return {}
    n = len(subset)
    return {b: sum(1 for o in subset if o["bucket"] == b) / n for b in BUCKETS}


def aggregate(outcomes: list[dict]) -> dict:
    """Aggregate classified outcomes.

    Bucket rates are reported per gold class + overall (they partition once
    ``classify`` assigns buckets). Citation-F1 is **gated**: micro + macro over
    ONLY ``VALID_CITATION`` records — fabricated/contradictory/false-grounding/
    abstained records are excluded from F1 (the fabrication-gate fix), and counted
    in their own buckets instead. found_answer accuracy is the image-use axis.
    """
    n = len(outcomes)
    ans = [o for o in outcomes if o["gold_class"] == "answerable"]
    abst = [o for o in outcomes if o["gold_class"] == "abstain"]

    # citation F1 over VALID_CITATION records only ('gated': fabrications, false
    # groundings, and abstains are excluded and counted in their own buckets)
    f1_pool = [o for o in outcomes if o["bucket"] == VALID_CITATION]
    tp = sum(len(set(o["cited_ids"]) & set(o["gold_cited_ids"])) for o in f1_pool)
    n_pred = sum(len(set(o["cited_ids"])) for o in f1_pool)
    n_gold = sum(len(set(o["gold_cited_ids"])) for o in f1_pool)
    micro_p = tp / n_pred if n_pred else 0.0
    micro_r = tp / n_gold if n_gold else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    macro_f1 = _mean([_record_f1(o["cited_ids"], o["gold_cited_ids"]) for o in f1_pool])

    def fa_acc(subset: list[dict]) -> float | None:
        """found_answer accuracy — the PERCEPTION axis (did it read the page right?),
        aggregated separately from the citation buckets and never mixed into them."""
        return _mean([o["found_answer_correct"] for o in subset])

    return {
        "n": n,
        "n_answerable": len(ans),
        "n_abstain": len(abst),
        "bucket_rates": {
            "answerable": _bucket_rates(ans),
            "abstain": _bucket_rates(abst),
            "overall": _bucket_rates(outcomes),
        },
        "citation_f1_gated": {
            "n_valid_citation": len(f1_pool),
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": _round_opt(macro_f1),
        },
        "found_answer_accuracy": {
            "answerable": _round_opt(fa_acc(ans)),
            "abstain": _round_opt(fa_acc(abst)),
            "overall": _round_opt(fa_acc(outcomes)),
        },
        "mean_evidence_reward": {
            "answerable": _round_opt(_mean([o["evidence_reward"] for o in ans])),
            "abstain": _round_opt(_mean([o["evidence_reward"] for o in abst])),
            "overall": _round_opt(_mean([o["evidence_reward"] for o in outcomes])),
        },
        "malformed_rate": _round_opt(sum(1 for o in outcomes if o["malformed"]) / n) if n else None,
    }


def _was_capped(cap_manifest: Mapping[str, dict], doc_id: Any) -> bool:
    """Look up ``was_capped`` for a doc_id (manifest keys are str(doc_id))."""
    entry = cap_manifest.get(str(doc_id))
    if entry is None:
        raise KeyError(f"doc_id {doc_id!r} not found in cap manifest")
    return bool(entry["was_capped"])


def score(
    records: list[dict],
    raw_by_record_id: Mapping[str, str],
    cap_manifest: Mapping[str, dict] | None = None,
) -> dict:
    """Classify every record and aggregate overall + (if a manifest is given) sliced.

    ``raw_by_record_id`` MUST cover every record (a dropped generation is a runner
    bug, not a malformed output) — a missing entry raises. Returns
    ``{"overall": <agg>, ["capped":..., "uncapped":...,] "per_record": [...]}``.
    (Runnable once ``classify`` is implemented.)
    """
    missing = [r["record_id"] for r in records if r["record_id"] not in raw_by_record_id]
    if missing:
        raise ValueError(
            f"{len(missing)} record(s) have no model output (e.g. {missing[:5]}); "
            "refusing to score — a dropped generation must not be silently counted."
        )

    outcomes: list[dict] = []
    for r in records:
        o = classify(r, raw_by_record_id[r["record_id"]])
        if cap_manifest is not None:
            o["was_capped"] = _was_capped(cap_manifest, r["doc_id"])
        outcomes.append(o)

    result: dict[str, Any] = {"overall": aggregate(outcomes)}
    if cap_manifest is not None:
        result["capped"] = aggregate([o for o in outcomes if o["was_capped"]])
        result["uncapped"] = aggregate([o for o in outcomes if not o["was_capped"]])
    result["per_record"] = outcomes
    return result
