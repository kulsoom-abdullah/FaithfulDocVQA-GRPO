"""`--false-abstain-base` safety: the default must reproduce run-4a byte-for-byte.

The flag exists so run 4b can price `false_abstain` negative (plan §c) WITHOUT editing
`grpo.reward.FALSE_ABSTAIN_BASE` in place. Editing that constant would retroactively
change the rewards that runs 1, 2 and 4a reproduce from their own frozen artifacts —
their committed `reward` fields would silently stop matching the code that claims to
produce them. These tests are what make the flag safe; the flag alone is not.

Three levels of evidence, weakest to strongest:

  1. unit — the default parameter IS the module constant, and only the
     FALSE_ABSTAIN bucket responds to the override.
  2. member log — every one of run-4a's 4,800 logged group-member rewards is
     consistent with the default. `reward_members.jsonl` carries no raw completion,
     so this is an exact interval check per bucket, not a recomputation.
  3. end-to-end — run-4a's per-row val dumps DO carry the raw completion, so every
     row is re-scored from raw text through classify() -> compute_grpo_reward() and
     compared to the committed value. 15 steps x 112 rows = 1,680 recomputations.

Level 3 is the test the commit rests on. CPU only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grpo.reward import (
    FA_BONUS,
    FABRICATION_PENALTY,
    FALSE_ABSTAIN_BASE,
    HONEST_ABSTAIN_RUN2,
    FALSE_GROUNDING_RUN2,
    CONTRADICTION_PENALTY,
    MALFORMED_PENALTY,
    compute_grpo_reward,
)

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "data/checkpoints/grpo_run4a_seed42"
INSTRUMENT = ROOT / "data/grpo_val_instrument_v2_recursfalse.jsonl"
VAL_ABSTAINS = ROOT / "data/manufactured_val_abstains.jsonl"
MASTER = ROOT / "data/master.jsonl"

RUN_4A = 2          # run-4a used the run-2 reward, byte-identical (plan §a)
STEPS = list(range(10, 160, 10))
REPRICE = -0.2      # the run-4b value (plan §c)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _rec(rid, target_is_abstain, gold_localized, menu_ids, gold_cited, gold_answers):
    """One answer key — same shape as tests/test_grpo_reward.py::rec."""
    return {
        "record_id": rid, "doc_id": f"doc_{rid}", "doc_type": "form",
        "target_is_abstain": target_is_abstain, "gold_localized": gold_localized,
        "ocr_menu": [{"id": i} for i in menu_ids],
        "cited_ids": gold_cited, "gold_answers": gold_answers,
    }


def _out(found_answer, flag, evidence_ids):
    return json.dumps({"found_answer": found_answer,
                       "answerable_from_menu": flag, "evidence_ids": evidence_ids})


# ── level 1: unit ────────────────────────────────────────────────────────────────

def test_default_is_the_module_constant():
    """Regression guard: if someone edits the constant to run 4b, this fails loudly."""
    assert FALSE_ABSTAIN_BASE == 0.0, (
        "FALSE_ABSTAIN_BASE was edited in place. Run 4b must pass "
        "--false-abstain-base instead; editing the constant breaks the reproducibility "
        "of every already-committed run."
    )


def test_override_moves_only_false_abstain():
    """The override must touch the FALSE_ABSTAIN bucket and nothing else."""
    cases = {
        "valid_citation":  (_rec(1, False, True, range(10), [0, 1, 2], ["ans"]), _out("ans", True, [0, 1, 2])),
        "fabrication":     (_rec(2, False, True, range(10), [5], ["ans"]),       _out("ans", True, [99])),
        "false_abstain":   (_rec(3, False, True, range(10), [2, 3], ["ans"]),    _out("ans", False, [])),
        "honest_abstain":  (_rec(4, True, False, range(10), [], ["ans"]),        _out("ans", False, [])),
        "false_grounding": (_rec(5, True, False, range(10), [], ["ans"]),        _out("ans", True, [4])),
        "contradiction":   (_rec(6, False, True, range(10), [3], ["ans"]),       _out("ans", True, [])),
        "malformed":       (_rec(7, False, True, range(10), [1], ["ans"]),       "not json at all"),
    }
    for name, (record, raw) in cases.items():
        at_default = compute_grpo_reward(record, raw, run=RUN_4A)
        at_reprice = compute_grpo_reward(record, raw, run=RUN_4A, false_abstain_base=REPRICE)
        delta = at_reprice - at_default
        expected = REPRICE if name == "false_abstain" else 0.0
        assert delta == pytest.approx(expected), (
            f"{name}: reprice moved reward by {delta}, expected {expected}")


def test_explicit_default_equals_implicit_default():
    """Passing the default explicitly must be indistinguishable from omitting it."""
    record = _rec(3, False, True, range(10), [2, 3], ["ans"])
    raw = _out("ans", False, [])
    assert (compute_grpo_reward(record, raw, run=RUN_4A)
            == compute_grpo_reward(record, raw, run=RUN_4A,
                                   false_abstain_base=FALSE_ABSTAIN_BASE))


# ── level 2: run-4a member log, all 4,800 rows ───────────────────────────────────

@pytest.mark.skipif(not (RUN_DIR / "reward_members.jsonl").is_file(),
                    reason="run-4a reward_members.jsonl not present")
def test_run4a_member_log_consistent_with_default():
    """Every logged group-member reward must be reachable at the DEFAULT base.

    reward_members.jsonl has no raw completion, so the reward cannot be recomputed
    from it. What is exactly checkable: reward == bucket_base + {0, FA_BONUS}, with
    valid_citation's base being an F1 in [0, 1].
    """
    rows = _read_jsonl(RUN_DIR / "reward_members.jsonl")
    assert len(rows) == 4800, f"expected 4800 member rows, got {len(rows)}"

    flat = {"contradiction": CONTRADICTION_PENALTY, "malformed": MALFORMED_PENALTY}
    based = {
        "false_abstain": FALSE_ABSTAIN_BASE,
        "fabrication": FABRICATION_PENALTY,
        "honest_abstain": HONEST_ABSTAIN_RUN2,
        "false_grounding": FALSE_GROUNDING_RUN2,
    }
    seen: set[str] = set()
    for row in rows:
        b, r = row["bucket"], row["reward"]
        seen.add(b)
        if b in flat:
            assert r == pytest.approx(flat[b]), f"{b}: {r}"
        elif b in based:
            assert r == pytest.approx(based[b]) or r == pytest.approx(based[b] + FA_BONUS), (
                f"{b} reward {r} unreachable at default base {based[b]} "
                f"(expected {based[b]} or {based[b] + FA_BONUS})")
        elif b == "valid_citation":
            assert 0.0 <= r <= 1.0 + FA_BONUS, f"valid_citation out of range: {r}"
        else:
            raise AssertionError(f"unmapped bucket in member log: {b!r}")

    # the bucket the flag targets must actually be present, or this proves nothing
    assert "false_abstain" in seen, "no false_abstain rows — test is vacuous"


@pytest.mark.skipif(not (RUN_DIR / "reward_members.jsonl").is_file(),
                    reason="run-4a reward_members.jsonl not present")
def test_run4a_member_log_would_break_under_reprice():
    """Known-positive control: at -0.2 the committed false_abstain values must NOT fit.

    Without this, the test above could pass for a reward function that ignores the
    base entirely.
    """
    rows = [r for r in _read_jsonl(RUN_DIR / "reward_members.jsonl")
            if r["bucket"] == "false_abstain"]
    assert rows, "no false_abstain rows to control on"
    allowed = {REPRICE, REPRICE + FA_BONUS}
    fits = sum(1 for r in rows
               if any(r["reward"] == pytest.approx(a) for a in allowed))
    assert fits == 0, (
        f"{fits}/{len(rows)} committed false_abstain rewards are consistent with the "
        f"repriced base {REPRICE} — the two configurations are not distinguishable, "
        "so the level-2 check cannot detect a reprice.")


# ── level 3: end-to-end recomputation against run-4a's committed val dumps ───────

def _load_val_records() -> dict[str, dict]:
    """Full gold records for the 112 instrument rows, from their committed sources."""
    inst = _read_jsonl(INSTRUMENT)
    want_abstain = {r["record_id"] for r in inst if r["source"] == "manufactured_val_abstains"}
    want_answerable = {r["record_id"] for r in inst if r["source"] == "master_validation"}

    recs = {r["record_id"]: r for r in _read_jsonl(VAL_ABSTAINS)
            if r["record_id"] in want_abstain}
    with MASTER.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r.get("record_id")) in want_answerable:
                recs[str(r["record_id"])] = r
    missing = (want_abstain | want_answerable) - set(recs)
    assert not missing, f"could not recover {len(missing)} instrument records: {sorted(missing)[:5]}"
    return recs


@pytest.mark.skipif(not (RUN_DIR / "eval_step150.jsonl").is_file(),
                    reason="run-4a val dumps not present")
def test_run4a_val_rewards_reproduce_exactly_at_default():
    """THE test: re-score every committed run-4a val row from its raw completion.

    1,680 recomputations (15 steps x 112 rows). Any drift in the reward table, the
    bucket cascade, or the default base shows up here as an exact mismatch.
    """
    recs = _load_val_records()
    checked = 0
    mismatches: list[str] = []
    for step in STEPS:
        path = RUN_DIR / f"eval_step{step}.jsonl"
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            rec = recs[row["record_id"]]
            got = compute_grpo_reward(rec, row["raw"], run=RUN_4A)
            checked += 1
            if got != pytest.approx(row["reward"]):
                mismatches.append(
                    f"step {step} {row['record_id']} ({row['bucket']}): "
                    f"committed {row['reward']} vs recomputed {got}")
    assert checked == 1680, f"expected 1680 recomputations, did {checked}"
    assert not mismatches, (
        f"{len(mismatches)}/{checked} committed rewards did not reproduce:\n  "
        + "\n  ".join(mismatches[:10]))


@pytest.mark.skipif(not (RUN_DIR / "eval_step150.jsonl").is_file(),
                    reason="run-4a val dumps not present")
def test_run4a_val_rewards_diverge_under_reprice():
    """Known-positive control for the test above.

    Re-scoring at -0.2 MUST break, and must break on exactly the false_abstain rows.
    A test that passes under both configurations is not testing the base at all.
    """
    recs = _load_val_records()
    changed: list[str] = []
    for step in STEPS:
        path = RUN_DIR / f"eval_step{step}.jsonl"
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            rec = recs[row["record_id"]]
            got = compute_grpo_reward(rec, row["raw"], run=RUN_4A,
                                      false_abstain_base=REPRICE)
            if got != pytest.approx(row["reward"]):
                changed.append(row["bucket"])
    assert changed, "repricing changed nothing — the override is not wired through"
    assert set(changed) == {"false_abstain"}, (
        f"reprice changed buckets {sorted(set(changed))}; expected only false_abstain")
