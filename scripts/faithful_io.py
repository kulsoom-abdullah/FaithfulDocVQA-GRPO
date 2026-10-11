"""Output schema, serialization, and tolerant parsing for FaithfulDocVQA SFT.

NOTE: the v1 2-field contract below is historical; the LIVE 3-field v3 contract
(``FaithfulOutput3Field``, ``parse_tolerant_v3``, ``derive_3field``) lives in the
second half of this module.

The v1 model emitted STRICT JSON with exactly two fields:

    {"answer": "<text>", "evidence_ids": [<int>, ...]}

Abstain is the same schema with an empty evidence list and the sentinel answer
``"unanswerable"``. There is NO reasoning trace — v1 is JSON-only
(docs/faithfuldocvqa_scope.md §1: "SFT target = citation-only").

This module is the single source of truth for:
  - the EXACT target string the SFT completion is trained on (`serialize`),
  - tolerant parsing of model outputs at eval time (`parse_tolerant`),
  - building targets from frozen ``data/master.jsonl`` records (`build_target`).

Design decisions (deliberate, reported at CHECKPOINT 1):
  - ``extra="forbid"``: an output carrying keys beyond the two-field contract is
    treated as malformed (returns None), per CLAUDE.md "strict JSON, two fields".
    We do NOT repair it — the malformed rate is a measured result.
  - ``evidence_ids`` are validated as genuine ints (no bool, no float, no string
    coercion). A model emitting ``["1"]`` or ``[true]`` is malformed, not silently
    fixed.
  - ``serialize`` uses compact separators, fixed key order (answer, evidence_ids),
    and ``ensure_ascii=False`` so unicode answers tokenize as the page text rather
    than as ``\\uXXXX`` escapes. Output is deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

ABSTAIN_ANSWER = "unanswerable"


class FaithfulOutput(BaseModel):
    """The model's output contract: an answer string + a list of cited OCR ids."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    evidence_ids: list[int]

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _strict_int_list(cls, v: Any) -> Any:
        """Reject anything that is not a list of genuine integers.

        ``bool`` is a subclass of ``int`` in Python, so it is rejected explicitly;
        floats and numeric strings are rejected too (no silent coercion). This
        keeps the malformed/valid boundary honest at scoring time.
        """
        if not isinstance(v, list):
            raise ValueError("evidence_ids must be a JSON array")
        for x in v:
            if isinstance(x, bool) or not isinstance(x, int):
                raise ValueError("evidence_ids must contain only integers")
        return v


def serialize(answer: str, evidence_ids: list[int]) -> str:
    """Return the exact, deterministic JSON target string for the SFT completion.

    Args:
        answer: the answer text (or ``"unanswerable"`` for abstain).
        evidence_ids: cited OCR token ids (empty for abstain).

    Returns:
        Compact JSON, fixed key order ``answer`` then ``evidence_ids``, e.g.
        ``{"answer":"2013","evidence_ids":[5]}``. This is the string trained on.
    """
    obj = {"answer": answer, "evidence_ids": [int(i) for i in evidence_ids]}
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _extract_first_json_object(s: str) -> str | None:
    """Return the first balanced ``{...}`` substring, or None.

    Brace counting that ignores braces inside JSON string literals, so an answer
    containing ``{`` or ``}`` does not break extraction. Handles prose on either
    side of the object.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
    return None


def parse_tolerant(raw: str) -> FaithfulOutput | None:
    """Tolerantly parse a model output into a :class:`FaithfulOutput`, or None.

    Steps (per scope/CLAUDE.md):
      1. strip surrounding whitespace and ```json / ``` code fences,
      2. extract the first balanced ``{...}`` object if wrapped in prose,
      3. ``json.loads`` then Pydantic-validate.

    Returns None on any failure (the caller scores that as malformed). No
    aggressive repair beyond fence-stripping and object extraction.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    # 1. strip code fences (opening ```lang and closing ```)
    s = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()
    # 2. extract first JSON object if surrounded by prose
    obj_str = _extract_first_json_object(s)
    if obj_str is None:
        return None
    # 3. parse + validate
    try:
        data = json.loads(obj_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return FaithfulOutput.model_validate(data)
    except ValidationError:
        return None


def build_target(record: dict) -> str:
    """Build the exact SFT completion string for a ``master.jsonl`` record.

    answerable (``target_is_abstain`` False) -> ``serialize(gold_answers[0], cited_ids)``
    abstain    (``target_is_abstain`` True)  -> ``serialize("unanswerable", [])``

    Branches on the frozen ``target_is_abstain`` label. For abstain records the
    frozen build guarantees ``cited_ids == []``; this is asserted so a corrupt
    record fails loudly rather than producing a citing "unanswerable".
    """
    if record["target_is_abstain"]:
        assert record["cited_ids"] == [], (
            f"abstain record {record.get('record_id')!r} has non-empty cited_ids"
        )
        return serialize(ABSTAIN_ANSWER, [])
    return serialize(record["gold_answers"][0], record["cited_ids"])


# ══════════════════════════════════════════════════════════════════════════════
# v2 — the 3-field contract  {found_answer, answerable_from_menu, evidence_ids}
# ══════════════════════════════════════════════════════════════════════════════
# Decouples visual PERCEPTION (found_answer — what the model read in the image,
# populated even on abstain rows because v1 never masks the image) from menu
# ATTRIBUTION (answerable_from_menu == gold_localized; evidence_ids the cited tokens,
# non-empty iff answerable_from_menu). See docs/DECISIONS.md §3. The derivation +
# its dataset-wide invariant verification live in scripts/build_3field.py (step 1);
# this module owns the SERIALIZATION + PARSING (step 2), mirroring the 2-field code
# above. Prompt/chat-template wiring is step 3 (format_sft/train_sft — untouched here).


class FaithfulOutput3Field(BaseModel):
    """The v2 output contract: perceived answer + menu-attribution flag + cited ids.

    Strict like :class:`FaithfulOutput`: ``extra="forbid"``; ``answerable_from_menu``
    must be a genuine JSON boolean (``"false"`` / ``0`` are malformed, not coerced);
    ``evidence_ids`` genuine ints. The cross-field contract (flag vs evidence
    non-emptiness) is checked by :meth:`contradiction`, NOT in validation, so the
    tolerant parser can categorize a contradiction distinctly from a type/key error.
    """

    model_config = ConfigDict(extra="forbid")

    found_answer: str
    answerable_from_menu: bool
    evidence_ids: list[int]

    @field_validator("answerable_from_menu", mode="before")
    @classmethod
    def _strict_bool(cls, v: Any) -> Any:
        """Require a real JSON boolean — reject ``"false"``/``"true"`` strings and
        ``0``/``1`` ints (no coercion), so a sloppy flag is malformed, not silently
        accepted."""
        if not isinstance(v, bool):
            raise ValueError("answerable_from_menu must be a JSON boolean")
        return v

    @field_validator("evidence_ids", mode="before")
    @classmethod
    def _strict_int_list(cls, v: Any) -> Any:
        """Reject anything that is not a list of genuine integers (mirrors
        :class:`FaithfulOutput`; bool is a subclass of int so rejected explicitly)."""
        if not isinstance(v, list):
            raise ValueError("evidence_ids must be a JSON array")
        for x in v:
            if isinstance(x, bool) or not isinstance(x, int):
                raise ValueError("evidence_ids must contain only integers")
        return v

    def contradiction(self) -> str | None:
        """Return a reason string if the flag and evidence are mutually inconsistent,
        else None. Checked outside validation so the parser can flag it as its own
        failure category (not lumped with type/key errors)."""
        if self.answerable_from_menu and not self.evidence_ids:
            return "answerable_from_menu=true requires non-empty evidence_ids"
        if not self.answerable_from_menu and self.evidence_ids:
            return "answerable_from_menu=false requires empty evidence_ids"
        return None


@dataclass
class ParseResultV3:
    """Outcome of :func:`parse_tolerant_v3`. ``value`` is the parsed contract or None;
    ``failure`` is None on success else the malformed category — so callers can tally
    *why* outputs failed (broken JSON vs schema vs contract contradiction) rather than
    collapse everything to one malformed bucket. ``ok`` is a convenience."""

    value: FaithfulOutput3Field | None
    failure: str | None  # None | "not_str" | "no_json" | "invalid_json" | "not_object" | "schema" | "contradiction"

    @property
    def ok(self) -> bool:
        return self.value is not None


def serialize_v3(found_answer: str, answerable_from_menu: bool,
                 evidence_ids: list[int]) -> str:
    """Return the canonical, deterministic 3-field JSON target string.

    Fixed key order (found_answer, answerable_from_menu, evidence_ids), compact
    separators, ``ensure_ascii=False`` — same conventions as :func:`serialize`.

    NOTE (deliberate decision, not accident — see docs/DECISIONS.md): ``found_answer``
    keeps its RAW DocVQA casing; we do NOT normalize/casefold it in the target.
    Casing noise is mild and the eval answer-matcher casefolds at scoring time.
    Revisit at GRPO if casing turns out to matter.
    """
    obj = {
        "found_answer": found_answer,
        "answerable_from_menu": bool(answerable_from_menu),
        "evidence_ids": [int(i) for i in evidence_ids],
    }
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_tolerant_v3(raw: str) -> ParseResultV3:
    """Tolerantly parse a 3-field model output, classifying failures by type.

    Same recovery spirit as :func:`parse_tolerant` (strip ```fences```, extract the
    first balanced ``{...}``, ``json.loads`` + Pydantic-validate) — but returns a
    :class:`ParseResultV3` whose ``failure`` names the category so the malformed rate
    is decomposable. No aggressive repair: a contradiction or bad type is reported,
    never silently fixed.
    """
    if not isinstance(raw, str):
        return ParseResultV3(None, "not_str")
    s = raw.strip()
    s = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    s = s.strip()
    obj_str = _extract_first_json_object(s)
    if obj_str is None:
        return ParseResultV3(None, "no_json")
    try:
        data = json.loads(obj_str)
    except (json.JSONDecodeError, ValueError):
        return ParseResultV3(None, "invalid_json")
    if not isinstance(data, dict):
        return ParseResultV3(None, "not_object")
    try:
        obj = FaithfulOutput3Field.model_validate(data)
    except ValidationError:
        return ParseResultV3(None, "schema")     # extra key / missing / wrong type
    if obj.contradiction() is not None:
        return ParseResultV3(None, "contradiction")
    return ParseResultV3(obj, None)


def derive_3field(record: dict) -> dict:
    """Derive {found_answer, answerable_from_menu, evidence_ids} from a frozen record.

    The single source of the v2 derivation (used by build_target_v3, format_sft, and
    the build_3field step-1 transform). Pure; no mutation of ``record``.
    ``image_masked`` (absent in v1) gates the v2 "unanswerable" found_answer branch.
    """
    gold_answers = record["gold_answers"]
    image_masked = bool(record.get("image_masked", False))  # v2-ready; False in v1

    # found_answer: the visible answer (v1) or "unanswerable" if the image hides it (v2).
    found_answer = "unanswerable" if image_masked else (gold_answers[0] if gold_answers else "")

    answerable_from_menu = bool(record["gold_localized"])
    evidence_ids = list(record["cited_ids"])  # gold evidence; verified empty iff not answerable

    return {
        "found_answer": found_answer,
        "answerable_from_menu": answerable_from_menu,
        "evidence_ids": evidence_ids,
    }


def build_target_v3(record: dict) -> str:
    """Build the 3-field SFT completion string for a ``master_3field.jsonl`` record.

    Reads the step-1-derived ``v2_contract`` block (built + invariant-verified by
    scripts/build_3field.py) and serializes it. Guards against a contract
    contradiction (mirrors :func:`build_target`'s assert) so a corrupt record fails
    loudly rather than producing an inconsistent target.
    """
    c = record.get("v2_contract")
    if c is None:
        raise KeyError(
            f"record {record.get('record_id')!r} has no 'v2_contract' block — "
            "run scripts/build_3field.py (step 1) first"
        )
    fa, afm, ev = c["found_answer"], bool(c["answerable_from_menu"]), list(c["evidence_ids"])
    contra = FaithfulOutput3Field(found_answer=fa, answerable_from_menu=afm,
                              evidence_ids=ev).contradiction()
    assert contra is None, f"record {record.get('record_id')!r} contract contradiction: {contra}"
    return serialize_v3(fa, afm, ev)
