"""
classify() cascade test cases — pins the scoring-table cascade implemented in
scripts/eval_scorer.py::classify (the bucket table is documented in
docs/behavior_axes_reference.md, "The table" section).

WHAT EACH PIECE IS
  record   : the ANSWER KEY for one example (ground truth), built by rec() below.
             Its fields say what the CORRECT answer is and what the menu holds —
             none of them describe what the model did.
  raw      : the MODEL'S OUTPUT for that example, as the raw JSON string it emitted
             (found_answer / answerable_from_menu / evidence_ids live here).
  facts    : NOT built here. classify(record, raw) builds it on its first line via
             _facts(): one merged dict of gold fields + model fields + computed
             signals (fabricated_ids, found_answer_correct, ...), bucket=None.
             classify() routes on facts["..."] and sets facts["bucket"].
  expected : the expected bucket per the scoring-table cascade (one case per row;
             the A–G letters are local case labels defined in this file:
             A–E the five routed buckets, F1/F2 contradiction, G1/G2 malformed).

Run:  pytest tests/test_classify_cases.py -v
"""
import pytest

from scripts.eval_scorer import (
    classify,
    VALID_CITATION, FABRICATION, CONTRADICTION, FALSE_GROUNDING,
    HONEST_ABSTAIN, FALSE_ABSTAIN, UNCITED_ANSWER, MALFORMED,
)


def rec(record_id, target_is_abstain, gold_localized, menu_ids, gold_cited_ids, gold_answers):
    """Build one ANSWER KEY (record). Param names match the keys _facts() reads.
    Everything here is GROUND TRUTH about the example — not model behavior:
      target_is_abstain : should the correct behavior be to abstain?
      gold_localized    : is the answer citable from the menu? (v1: == not target_is_abstain)
      menu_ids          : the valid OCR-token ids; a cited id outside this set is fabricated
      gold_cited_ids    : which menu ids are the CORRECT citation (F1 is scored against these)
      gold_answers      : the correct answer text(s), a list
    """
    return {
        "record_id": record_id, "doc_id": f"doc_{record_id}", "doc_type": "form",
        "target_is_abstain": target_is_abstain, "gold_localized": gold_localized,
        "ocr_menu": [{"id": i} for i in menu_ids],
        "cited_ids": gold_cited_ids, "gold_answers": gold_answers,
    }


# (raw_output, record, expected_bucket) 
CASES = [
    # A  Correct cite - answerable + grounded, cites all gold ids and got the right found answer
    ('{"found_answer":"university of california","answerable_from_menu":true,"evidence_ids":[0,1,2]}',
     rec(24580, False, True, range(10), [0, 1, 2], ["university of california"]),
     VALID_CITATION), # v1 correct cite
     # A2  "wrong but real" citation — answer right, flag=true, cited ids are REAL
    #     (all in the menu) but not the gold ids. SAME bucket as A; F1 is ~0 in aggregate.
    ('{"found_answer":"university of california","answerable_from_menu":true,"evidence_ids":[7,8]}',
     rec(24580, False, True, range(10), [0, 1, 2], ["university of california"]),
     VALID_CITATION), #v1 wrong but real cite
    # B  abstain; found_answer is correct but flagged not menu-citable
    ('{"found_answer":"3293F","answerable_from_menu":false,"evidence_ids":[]}',
     rec(47368, True, False, range(10), [], ["3293F"]),
    HONEST_ABSTAIN), 
    # C  flag=true but gold says NOT citable; id 41 IS in the menu (so NOT fabrication)
    ('{"found_answer":"42","answerable_from_menu":true,"evidence_ids":[41]}',
     rec(57951, True, False, range(51), [], ["42"]),
     FALSE_GROUNDING),
    # D  flag=false but gold IS citable
    ('{"found_answer":"Initiate Priority Designation","answerable_from_menu":false,"evidence_ids":[]}',
     rec(51214, False, True, range(10), [2, 3, 4, 5], ["Initiate priority designation"]),
    FALSE_ABSTAIN), #v1
    # E  looks valid, but cited id 955 is NOT in the menu (ids 0..616)
    ('{"found_answer":"15","answerable_from_menu":true,"evidence_ids":[955]}',
     rec(36495, False, True, range(617), [5], ["15"]),
    FABRICATION), # v1 fabricated citation
    # F1  flag=false WITH a cite -> contradiction at parse
    ('{"found_answer":"x","answerable_from_menu":false,"evidence_ids":[3]}',
     rec(90001, False, True, range(10), [3], ["x"]),
     CONTRADICTION), # cited evidence correct, but boolean wrong
    # F2  flag=true with EMPTY cites -> contradiction at parse
    ('{"found_answer":"y","answerable_from_menu":true,"evidence_ids":[]}',
     rec(90002, False, True, range(10), [1], ["y"]),
     CONTRADICTION), #v1 boolean correct, but didnt cite anything
    # G1  not JSON at all -> malformed (no_json)
    ('not json at all',
     rec(90003, False, True, range(10), [1], ["z"]),
     MALFORMED), #unparseable
    # G2  stringified bool "false" -> schema failure -> malformed
    ('{"found_answer":"z","answerable_from_menu":"false","evidence_ids":[]}',
     rec(90004, True, False, range(10), [], ["z"]),
     MALFORMED), #this is partially malformed, not totally though
]


@pytest.mark.parametrize("raw,record,expected", CASES)
def test_classify_bucket(raw, record, expected):
    assert expected is not None, "Fill in the expected bucket for this case first."
    assert classify(record, raw)["bucket"] == expected
