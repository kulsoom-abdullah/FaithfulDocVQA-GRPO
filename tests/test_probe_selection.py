"""Probe-selection guards for ``grpo.train_grpo.pick_probe_records``.

The probe is the every-N-step qualitative dump. It is OBSERVATION ONLY and never scored,
which is what allows its decoy row to come from outside the scored val instrument — needed
because run 4's instrument is recurs-False by construction and so contains no decoys
(``docs/analysis/run4_analysis_plan.md`` §g).

Three properties, all CPU, no data files:
  1. the out-of-instrument fallback engages ONLY when the instrument has no decoys of its
     own — otherwise runs 1-2 behaviour must be bit-identical;
  2. a fallback decoy is never a record that is already in the scored instrument;
  3. every probe carries ``_kind``, because the step dumps read it
     (``train_grpo.py`` step-dump sites) and a raw JSONL row does not have it.
"""
from __future__ import annotations

import pytest

from grpo.train_grpo import pick_probe_records

SEED = 42


def _rec(rid: str, kind: str, recurs: bool | None = None) -> dict:
    r = {"record_id": rid, "doc_id": int(rid.split("_")[0][1:]), "_kind": kind}
    if kind == "abstain":
        r["answer_recurs_in_menu"] = recurs
    return r


def _instrument(n_abstain: int = 6, n_answerable: int = 4, decoys: int = 0) -> list[dict]:
    abst = [_rec(f"a{i}_mfab", "abstain", recurs=(i < decoys)) for i in range(n_abstain)]
    ans = [_rec(f"q{i}", "answerable") for i in range(n_answerable)]
    return abst + ans


def _outside_decoys(n: int = 5) -> list[dict]:
    # raw manufactured-val rows: note NO "_kind" key, exactly as they come off the jsonl
    return [{"record_id": f"z{i}_mfab", "doc_id": 9000 + i, "answer_recurs_in_menu": True}
            for i in range(n)]


def test_fallback_engages_only_when_instrument_has_no_decoys():
    """A decoy-bearing instrument must ignore decoy_pool — runs 1-2 behaviour unchanged."""
    inst = _instrument(decoys=2)
    outside = _outside_decoys()
    with_pool = pick_probe_records([], inst, SEED, decoy_pool=outside)
    without_pool = pick_probe_records([], inst, SEED)
    assert [p["record_id"] for p in with_pool] == [p["record_id"] for p in without_pool]
    inst_ids = {r["record_id"] for r in inst}
    assert all(p["record_id"] in inst_ids for p in with_pool), "must not reach outside"

    # and with a decoy-free instrument the fallback DOES engage
    clean = _instrument(decoys=0)
    probes = pick_probe_records([], clean, SEED, decoy_pool=outside)
    outside_ids = {r["record_id"] for r in outside}
    assert any(p["record_id"] in outside_ids for p in probes), "fallback should engage"


def test_fallback_decoy_is_never_already_in_the_instrument():
    """The probe must not re-show a record the instrument scores."""
    inst = _instrument(decoys=0)
    inst_ids = {r["record_id"] for r in inst}
    # a pool whose decoys overlap the instrument entirely -> nothing eligible
    overlapping = [{"record_id": r["record_id"], "doc_id": r["doc_id"],
                    "answer_recurs_in_menu": True}
                   for r in inst if r["_kind"] == "abstain"]
    probes = pick_probe_records([], inst, SEED, decoy_pool=overlapping)
    chosen_outside = [p for p in probes if p["record_id"] not in inst_ids]
    assert not chosen_outside, "no eligible outside decoy existed; none may be invented"

    # a disjoint pool IS eligible, and what it contributes is genuinely outside
    disjoint = _outside_decoys()
    probes2 = pick_probe_records([], inst, SEED, decoy_pool=disjoint)
    picked = [p for p in probes2 if p["record_id"] not in inst_ids]
    assert len(picked) == 1
    assert picked[0]["record_id"] in {r["record_id"] for r in disjoint}


@pytest.mark.parametrize("decoys,pool", [(0, None), (0, _outside_decoys()), (2, _outside_decoys())])
def test_every_probe_carries_kind(decoys, pool):
    """The step dumps read p["_kind"]; a raw jsonl row has no such key, so the fallback
    must stamp it. Without this, a probe step would KeyError mid-training."""
    probes = pick_probe_records([], _instrument(decoys=decoys), SEED, decoy_pool=pool)
    assert probes, "probe selection returned nothing"
    for p in probes:
        assert "_kind" in p, f"probe {p['record_id']} is missing _kind"
        assert p["_kind"] in ("abstain", "answerable")
