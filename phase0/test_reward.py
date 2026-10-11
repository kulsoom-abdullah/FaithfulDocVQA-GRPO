# ═══ LEGACY — unused since the v3 contract. Kept for provenance, not maintained.
# Nothing imports this module; see phase0/README.md.
"""Unit tests for parse_response (Fix 1) and compute_evidence_reward (Fix 2).

Run with:  uv run python phase0/test_reward.py
       or: uv run pytest phase0/test_reward.py -v
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from PIL import Image

from phase0.pipeline import (
    parse_response,
    compute_evidence_reward,
    call_teacher_with_retry,
    ABSTAIN_CREDIT,
    FABRICATION_PENALTY,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_raw(tokens_str: str, answer: str = "foo") -> str:
    """Minimal well-formed XML response with custom evidence_tokens content."""
    return (
        "<rationale>"
        "<locate>x</locate>"
        f"<evidence_tokens>{tokens_str}</evidence_tokens>"
        '<evidence_text>""</evidence_text>'
        "<derive>x</derive>"
        "</rationale>"
        f"<answer>{answer}</answer>"
    )


VALID_IDS = set(range(10))  # IDs 0–9 for most tests


# ── Fix 1: parse_response JSON failure → evidence_valid=0 ─────────────────────

def test_parse_ok_empty_list():
    """[] with valid JSON → parse succeeds → evidence_valid=1 (correct refusal)."""
    r = parse_response(_make_raw("[]", answer="unanswerable"), VALID_IDS)
    assert r["parsed_evidence_tokens"] == []
    assert r["evidence_valid"] == 1

def test_parse_ok_valid_ids():
    """[1,2,3] all in valid_ids → evidence_valid=1."""
    r = parse_response(_make_raw("[1, 2, 3]"), VALID_IDS)
    assert r["parsed_evidence_tokens"] == [1, 2, 3]
    assert r["evidence_valid"] == 1

def test_parse_ok_out_of_range():
    """[975] not in valid_ids (0–9) → evidence_valid=0, no exception."""
    r = parse_response(_make_raw("[975]"), VALID_IDS)
    assert r["parsed_evidence_tokens"] == [975]
    assert r["evidence_valid"] == 0

def test_parse_fail_not_json():
    """Malformed JSON → evidence_tokens=[], evidence_valid=0 (not 1)."""
    r = parse_response(_make_raw("not json"), VALID_IDS)
    assert r["parsed_evidence_tokens"] == []
    assert r["evidence_valid"] == 0

def test_parse_fail_string_array():
    """["a","b"] → int("a") raises ValueError → evidence_valid=0."""
    r = parse_response(_make_raw('["a", "b"]'), VALID_IDS)
    assert r["parsed_evidence_tokens"] == []
    assert r["evidence_valid"] == 0

def test_parse_fail_partial_json():
    """Truncated JSON array → evidence_valid=0."""
    r = parse_response(_make_raw("[1, 2,"), VALID_IDS)
    assert r["parsed_evidence_tokens"] == []
    assert r["evidence_valid"] == 0

def test_parse_fail_does_not_propagate_as_refusal():
    """
    Fix 1 regression: before the fix, malformed JSON → [] → evidence_valid=1
    (false positive). After the fix, it must be 0.
    """
    raw_bad = _make_raw("not json", answer="unanswerable")
    r = parse_response(raw_bad, VALID_IDS)
    assert r["evidence_valid"] == 0, (
        "Silent parse failure must not produce evidence_valid=1"
    )

def test_parse_mixed_invalid():
    """[1, 975] — 1 valid, 975 invalid → evidence_valid=0."""
    r = parse_response(_make_raw("[1, 975]"), VALID_IDS)
    assert r["evidence_valid"] == 0


# ── Fix 2: compute_evidence_reward — piecewise contract ──────────────────────

# Refusal-expected branch (is_answerable=False)
# -----------------------------------------------------------------------------

def test_reward_unanswerable_correct_refusal():
    """Refusal-expected + cited=[] → 1.0 (correct refusal)."""
    out = compute_evidence_reward([], set(), VALID_IDS, is_answerable=False)
    assert out["evidence_reward"]    == 1.0
    assert out["evidence_precision"] is None
    assert out["evidence_recall"]    is None

def test_reward_unanswerable_overcite_valid():
    """Refusal-expected + non-empty cited (valid IDs) → 0.0 (refusal violation)."""
    out = compute_evidence_reward([2, 3], set(), VALID_IDS, is_answerable=False)
    assert out["evidence_reward"]    == 0.0
    assert out["evidence_precision"] is None
    assert out["evidence_recall"]    is None

def test_reward_unanswerable_overcite_fabricated():
    """Refusal-expected + cited contains fabricated ID → 0.0.
    Citation validity is not differentiated under unanswerable; the wrongness
    is 'didn't refuse', not 'cited badly'.
    """
    out = compute_evidence_reward([999], set(), VALID_IDS, is_answerable=False)
    assert out["evidence_reward"] == 0.0

# Answerable branch — honest abstain
# -----------------------------------------------------------------------------

def test_reward_answerable_empty_is_honest_abstain():
    """Answerable + cited=[] → ABSTAIN_CREDIT (+0.4).
    The old contract scored this 0.0 as 'lazy under-citation'. The new contract
    rewards it as honest abstention — strictly preferable to fabrication.
    """
    out = compute_evidence_reward([], {2, 3}, VALID_IDS, is_answerable=True)
    assert out["evidence_reward"]    == ABSTAIN_CREDIT
    assert out["evidence_precision"] is None
    assert out["evidence_recall"]    is None

# Answerable branch — fabrication
# -----------------------------------------------------------------------------

def test_reward_answerable_all_cited_invalid_is_fabrication():
    """Answerable + every cited ID ∉ valid_ids → FABRICATION_PENALTY (-0.5)."""
    valid = set(range(145))  # 0–144 as in img2
    out = compute_evidence_reward([975, 976], {2, 3}, valid, is_answerable=True)
    assert out["evidence_reward"]    == FABRICATION_PENALTY
    assert out["evidence_precision"] is None
    assert out["evidence_recall"]    is None

def test_reward_answerable_mixed_valid_invalid_is_fabrication():
    """Answerable + ANY cited ID ∉ valid_ids → FABRICATION_PENALTY (-0.5).
    Even one fabricated ID short-circuits to the penalty — set-overlap is not
    computed. This is the strictness the old soft-precision contract lacked.
    """
    valid = set(range(145))
    out = compute_evidence_reward([2, 975], {2, 3}, valid, is_answerable=True)
    assert out["evidence_reward"] == FABRICATION_PENALTY

def test_reward_answerable_empty_menu_fabrication():
    """Answerable + valid_ids=∅ + any cited → FABRICATION_PENALTY.
    Matches img1_caseB: Augraphy collapsed OCR to 0 tokens; the model cited
    [1,2,3,4] which cannot exist in an empty menu → fabrication.
    """
    out = compute_evidence_reward([1, 2, 3, 4], set(), set(), is_answerable=True)
    assert out["evidence_reward"] == FABRICATION_PENALTY

# Answerable branch — F1 set-overlap
# -----------------------------------------------------------------------------

def test_reward_answerable_perfect_match():
    """cited == gold, all valid → F1 = 1.0."""
    out = compute_evidence_reward([2, 3], {2, 3}, VALID_IDS, is_answerable=True)
    assert out["evidence_precision"] == 1.0
    assert out["evidence_recall"]    == 1.0
    assert out["evidence_reward"]    == 1.0

def test_reward_answerable_partial_overlap():
    """cited=[1,2,3], gold={2,3,4}, all valid: P=2/3, R=2/3, F1=2/3."""
    out = compute_evidence_reward([1, 2, 3], {2, 3, 4}, VALID_IDS, is_answerable=True)
    assert out["evidence_precision"] == round(2/3, 4)
    assert out["evidence_recall"]    == round(2/3, 4)
    assert out["evidence_reward"]    == round(2/3, 4)

def test_reward_answerable_superset_citation():
    """cited=[1,2,3,4,5], gold={2,3}, all valid: P=2/5, R=1.0, F1=4/7."""
    out = compute_evidence_reward([1, 2, 3, 4, 5], {2, 3}, VALID_IDS, is_answerable=True)
    p, r = 2/5, 1.0
    expected_f1 = round(2 * p * r / (p + r), 4)
    assert out["evidence_precision"] == round(p, 4)
    assert out["evidence_recall"]    == round(r, 4)
    assert out["evidence_reward"]    == expected_f1

def test_reward_answerable_single_correct():
    """cited=[3], gold={3} → F1 = 1.0."""
    out = compute_evidence_reward([3], {3}, VALID_IDS, is_answerable=True)
    assert out["evidence_reward"] == 1.0

def test_reward_answerable_deduplication():
    """cited=[2,2,3] dedupes to {2,3}, gold={2,3} → F1 = 1.0."""
    out = compute_evidence_reward([2, 2, 3], {2, 3}, VALID_IDS, is_answerable=True)
    assert out["evidence_reward"] == 1.0

def test_reward_answerable_gold_empty_returns_zero():
    """Answerable + all cited valid + gold=∅ (localizer failure) → 0.0.
    This is a localizer-infrastructure artifact, not a fabrication: the model
    cited real menu tokens but no gold was placed against which to compute F1.
    Documented so callers don't conflate it with the F1=0 disjoint-cite case.
    """
    out = compute_evidence_reward([2, 3], set(), VALID_IDS, is_answerable=True)
    assert out["evidence_reward"]    == 0.0
    assert out["evidence_precision"] == 0.0
    assert out["evidence_recall"]    == 0.0

# Contract invariant
# -----------------------------------------------------------------------------

def test_reward_honest_abstain_strictly_outscores_fabrication():
    """Core invariant the v1 reward violated: ABSTAIN_CREDIT > FABRICATION_PENALTY.
    This is the property that makes the new contract reward-hacking resistant.
    """
    assert ABSTAIN_CREDIT > FABRICATION_PENALTY
    valid = set(range(145))
    honest = compute_evidence_reward([], {2, 3}, valid, is_answerable=True)
    fab    = compute_evidence_reward([999], {2, 3}, valid, is_answerable=True)
    assert honest["evidence_reward"] > fab["evidence_reward"]


# ── Retry smoke test ──────────────────────────────────────────────────────────

def test_retry_smoke():
    """Mock client returns bad first response (fabricated ID), good second.

    Asserts: retry_triggered=True, retry_succeeded=True, failure_reason correct,
    final_parsed reflects the good second response.
    """
    # First response: valid XML but cites token ID 999 (not in 0–2 range)
    bad_raw = (
        "<rationale>"
        "<locate>Top of page.</locate>"
        "<evidence_tokens>[999]</evidence_tokens>"
        '<evidence_text>"foo"</evidence_text>'
        "<derive>Found it.</derive>"
        "</rationale>"
        "<answer>foo</answer>"
    )
    # Second response: valid XML with IDs that exist in the token list
    good_raw = (
        "<rationale>"
        "<locate>Top of page.</locate>"
        "<evidence_tokens>[1, 2]</evidence_tokens>"
        '<evidence_text>"foo bar"</evidence_text>'
        "<derive>Tokens 1 and 2 confirm the answer.</derive>"
        "</rationale>"
        "<answer>foo bar</answer>"
    )

    class _MockUsage:
        input_tokens = 100
        output_tokens = 50
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

    class _MockContent:
        def __init__(self, text): self.text = text

    class _MockResponse:
        def __init__(self, text):
            self.content = [_MockContent(text)]
            self.usage   = _MockUsage()

    call_index = [0]
    responses  = [bad_raw, good_raw]

    class _MockAPI:
        def create(self, **kwargs):
            resp = _MockResponse(responses[call_index[0]])
            call_index[0] += 1
            return resp

    class _MockClient:
        messages = _MockAPI()

    img    = Image.new("RGB", (100, 100), color=255)
    tokens = [
        {"id": 0, "word": "hello", "x1": 0,  "y1": 0,  "x2": 50,  "y2": 20},
        {"id": 1, "word": "foo",   "x1": 0,  "y1": 25, "x2": 50,  "y2": 45},
        {"id": 2, "word": "bar",   "x1": 55, "y1": 25, "x2": 100, "y2": 45},
    ]

    result = call_teacher_with_retry(img, tokens, "What does it say?",
                                     _MockClient(), case_id="smoke_test")

    assert result["retry_triggered"]  is True,  "retry should have fired"
    assert result["retry_succeeded"]  is True,  "second response is valid"
    assert result["failure_reason"]   == "out_of_range_ids"
    assert len(result["attempts"])    == 2
    assert result["attempts"][0]["parsed"]["evidence_valid"] == 0  # bad first
    assert result["attempts"][1]["parsed"]["evidence_valid"] == 1  # good second
    assert result["final_parsed"] is not None,  "final_parsed must be populated on success"
    assert result["final_parsed"]["parsed_answer"] == "foo bar"
    assert call_index[0] == 2  # exactly two API calls made


def test_retry_both_fail():
    """Both API calls return out-of-range IDs: final_parsed must be None, not bad data."""
    bad_raw = (
        "<rationale>"
        "<locate>Top of page.</locate>"
        "<evidence_tokens>[999]</evidence_tokens>"
        '<evidence_text>"foo"</evidence_text>'
        "<derive>Found it.</derive>"
        "</rationale>"
        "<answer>foo</answer>"
    )

    class _MockUsage:
        input_tokens = 100
        output_tokens = 50
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

    class _MockContent:
        def __init__(self, text): self.text = text

    class _MockResponse:
        def __init__(self, text):
            self.content = [_MockContent(text)]
            self.usage   = _MockUsage()

    class _MockAPI:
        def create(self, **kwargs):
            return _MockResponse(bad_raw)

    class _MockClient:
        messages = _MockAPI()

    img    = Image.new("RGB", (100, 100), color=255)
    tokens = [
        {"id": 0, "word": "hello", "x1": 0,  "y1": 0,  "x2": 50,  "y2": 20},
        {"id": 1, "word": "foo",   "x1": 0,  "y1": 25, "x2": 50,  "y2": 45},
        {"id": 2, "word": "bar",   "x1": 55, "y1": 25, "x2": 100, "y2": 45},
    ]

    result = call_teacher_with_retry(img, tokens, "What does it say?",
                                     _MockClient(), case_id="both_fail_test")

    assert result["retry_triggered"]  is True
    assert result["retry_succeeded"]  is False
    assert result["final_parsed"]     is None,  "final_parsed must be None when both attempts fail"
    assert len(result["attempts"])    == 2
    assert result["attempts"][0]["parsed"]["evidence_valid"] == 0
    assert result["attempts"][1]["parsed"]["evidence_valid"] == 0


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
