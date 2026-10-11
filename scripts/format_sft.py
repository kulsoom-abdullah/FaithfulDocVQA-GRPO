"""Prompt template + dataset formatting for FaithfulDocVQA SFT (Stage 2).

Turns a frozen ``data/master.jsonl`` record into a chat example fed to
Qwen3-VL-4B-Instruct:

  - **system**: the task contract (answer from the page, cite OCR token ids,
    abstain rule, JSON-only).
  - **user**: the page IMAGE, the QUESTION, and the OCR MENU rendered as the
    project's canonical ``<id>\\t<text>\\t<x1,y1,x2,y2>`` list.
  - **assistant**: the Stage-1 JSON target (``scripts.faithful_io.build_target``).

The chat string is produced ONLY by the official ``processor.apply_chat_template``
— we never hand-roll Qwen3-VL's template (its vision/chat format differs from
2.5-VL). The OCR menu string is rendered to the audit's exact
``<id>\\t<text>\\t<x1,y1,x2,y2>`` format locally (asserted in tests), so the
training path stays free of the audit module's heavy deps (pandas/pytesseract/
anthropic). The training pipeline does NO OCR/localization — it reads the frozen,
pre-extracted tokens from ``master.jsonl`` — so it does not import ``phase0``.

SFT-A vs SFT-B differ ONLY by data filter (CLAUDE.md hard rule 5):
  A = ``gold_localized == True`` records, B = all records — both within a split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.faithful_io import build_target_v3, derive_3field

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
FALLBACK_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MASTER = ROOT / "data" / "master.jsonl"

# Vision-token / memory cap. Qwen3-VL uses patch_size=16, merge_size=2 → each
# vision token covers (16*2)^2 = 1024 px, so max_pixels = MAX_VISION_TOKENS*1024.
# The processor's default ceiling is ~16.78 MP (≈16k tokens), which OOMs a 24 GB
# 4090 on outlier scans (one 38 MP page → 16,280 tokens — observed at train step
# 488). We cap at ~4.2 MP (4096 tokens), a PROVEN-SAFE budget (the full run handled
# ~4 MP fine): 90% of pages are ≤ 4.1 MP and pass through UNCLIPPED (full resolution
# → small text preserved); the ~38 oversized scans (>4.2 MP, mostly ~37 MP) are
# downscaled. Applied identically to SFT-A and SFT-B via load_processor, so the A/B
# comparison is unaffected. data/images_cap_manifest.json records per-doc
# original_mp / effective_mp / was_capped for capped-vs-uncapped eval slicing.
MAX_VISION_TOKENS = 4096
MAX_PIXELS = MAX_VISION_TOKENS * 1024  # = 4,194,304 px (~4.2 MP)

Mode = Literal["A", "B"]
Split = Literal["train", "validation", "test"]

# The task contract the model is trained to obey. JSON-only, no reasoning trace.
# v2 THREE-field contract (docs/DECISIONS.md §3): decouple visual PERCEPTION
# (found_answer) from menu ATTRIBUTION (answerable_from_menu + evidence_ids).
SYSTEM_INSTRUCTION = (
    "You are a document visual-question-answering system. You are given a page "
    "image, a question, and a numbered OCR token menu. Each menu line is "
    "`<id>\\t<text>\\t<x1,y1,x2,y2>` — the integer id, the token text, and its "
    "bounding box.\n"
    "First read the answer from the page IMAGE; then decide whether that answer "
    "can be cited from the OCR menu.\n"
    "Output ONLY a single JSON object with exactly three fields:\n"
    '  "found_answer": the answer read from the page image (a string); use '
    '"unanswerable" only if the answer cannot be determined from the image.\n'
    '  "answerable_from_menu": true if the answer is supported by OCR token id(s) '
    "in the menu, otherwise false (a boolean).\n"
    '  "evidence_ids": the menu token id(s) supporting the answer (a list of '
    "integers) — non-empty when answerable_from_menu is true, and [] when false.\n"
    "No other text, no explanation."
)


# ── data loading / filtering ──────────────────────────────────────────────────

def load_master() -> list[dict]:
    """Read every record from the frozen master dataset."""
    assert MASTER.exists(), f"missing frozen dataset: {MASTER}"
    return [json.loads(line) for line in MASTER.open()]


def load_split(split: Split, mode: Mode, records: list[dict] | None = None) -> list[dict]:
    """Return the records for one split, applying the per-mode filter to TRAIN ONLY.

    Args:
        split: ``"train"``, ``"validation"``, or ``"test"`` (split by docId).
        mode:  ``"A"`` (control) vs ``"B"`` (treatment). The filter — A keeps only
               ``gold_localized == True`` (answerable-only), B keeps all — is a
               **training-data** choice and is applied ONLY when ``split=="train"``.
        records: optional pre-loaded master (else loaded from disk).

    Returns:
        For ``train``: per-mode-filtered records. For ``validation``/``test``: the
        FULL held-out split (abstains included), **identical for SFT-A and SFT-B** —
        both arms are evaluated on exactly the same records.
    """
    if mode not in ("A", "B"):
        raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")
    if split not in ("train", "validation", "test"):
        raise ValueError(f"split must be 'train'/'validation'/'test', got {split!r}")
    recs = records if records is not None else load_master()
    recs = [r for r in recs if r["split"] == split]
    if split == "train" and mode == "A":      # mode filter: TRAINING data only
        recs = [r for r in recs if r["gold_localized"] is True]
    return recs


# ── menu rendering ────────────────────────────────────────────────────────────

def render_menu(ocr_menu: list[dict]) -> str:
    """Render a record's ``ocr_menu`` to the canonical menu string.

    One line per token, ``<id>\\t<text>\\t<x1,y1,x2,y2>`` — byte-identical to the
    audit's ``phase0.pipeline.tokens_to_str`` output, but rendered here so the
    training path does not import the audit module (and its pandas/pytesseract/
    anthropic deps). The format is asserted in
    ``tests/test_format_sft.py::test_render_menu_matches_ids_and_text``.
    """
    return "\n".join(
        f"{m['id']}\t{m['text']}\t"
        f"{m['bbox'][0]},{m['bbox'][1]},{m['bbox'][2]},{m['bbox'][3]}"
        for m in ocr_menu
    )


# ── chat message construction ─────────────────────────────────────────────────

def _v3_target(record: dict) -> str:
    """The 3-field SFT target string for a record. Uses the step-1-derived
    ``v2_contract`` block if present (master_3field.jsonl rows), else derives it on
    the fly from the raw master fields — so callers can pass either a master_3field
    row or a plain master.jsonl row."""
    rec = record if "v2_contract" in record else {**record, "v2_contract": derive_3field(record)}
    return build_target_v3(rec)


def build_messages(record: dict, include_target: bool, image_ref: Any | None = None) -> list[dict]:
    """Build the chat ``messages`` list for a record.

    Args:
        record: a master.jsonl record.
        include_target: if True, append the assistant turn (the JSON target);
            if False, stop after the user turn (for the prompt-only render used
            by completion-only masking in Stage 3).
        image_ref: what to put in the image content slot. Defaults to the
            record's ``image_path`` (a path string is fine for template
            rendering; Stage 3 swaps in the loaded image for tokenization).

    Returns:
        A Qwen-style messages list (system, user[, assistant]).
    """
    if image_ref is None:
        image_ref = record["image_path"]
    user_text = f"Question: {record['question']}\n\nOCR tokens:\n{render_menu(record['ocr_menu'])}"
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": [
            {"type": "image", "image": image_ref},
            {"type": "text", "text": user_text},
        ]},
    ]
    if include_target:
        messages.append({"role": "assistant", "content": _v3_target(record)})
    return messages


# ── processor loading + rendering ─────────────────────────────────────────────

def load_processor(model_id: str = MODEL_ID):
    """Load the official Qwen3-VL processor via ``AutoProcessor.from_pretrained``.

    Falls back to ``FALLBACK_MODEL_ID`` (Qwen2.5-VL-3B) if the primary id cannot
    be loaded with the installed tooling. Returns ``(processor, resolved_id)``.
    Raises if neither loads (caller should STOP and report — do not hand-roll).
    """
    from transformers import AutoProcessor

    last_err: Exception | None = None
    for cand in (model_id, FALLBACK_MODEL_ID):
        try:
            proc = AutoProcessor.from_pretrained(cand)
            _apply_pixel_cap(proc)
            return proc, cand
        except Exception as e:  # noqa: BLE001 - report the real error, don't improvise
            last_err = e
            print(f"[load_processor] {cand} failed: {type(e).__name__}: {str(e)[:200]}",
                  file=sys.stderr)
    raise RuntimeError(f"Could not load any processor; last error: {last_err}")


def _apply_pixel_cap(processor) -> None:
    """Lower the image processor's resize ceiling to ``MAX_PIXELS`` so outlier
    scans don't blow up the vision-token count (and VRAM). transformers 5.x stores
    the bound in ``image_processor.size['longest_edge']``; older builds use a
    ``max_pixels`` attribute — set whichever exists. Idempotent and harmless if the
    structure changes (it just won't lower a cap it can't find)."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    if hasattr(ip, "max_pixels"):
        ip.max_pixels = MAX_PIXELS
    size = getattr(ip, "size", None)
    if size is not None:
        try:
            if size.get("longest_edge", MAX_PIXELS + 1) > MAX_PIXELS:
                size["longest_edge"] = MAX_PIXELS
        except (AttributeError, TypeError):
            pass


def render_example(processor, record: dict, image_ref: Any | None = None) -> dict:
    """Render a record into the exact strings fed to the model.

    Uses ``processor.apply_chat_template`` for BOTH the full conversation
    (prompt + assistant target) and the prompt-only prefix (with a generation
    prompt). The completion is the suffix of full after the prompt — exactly the
    region Stage 3 trains on (everything before it is masked to -100).

    Returns dict: ``prompt_text``, ``completion_text``, ``full_text``, ``target``.
    """
    full_text = processor.apply_chat_template(
        build_messages(record, include_target=True, image_ref=image_ref),
        tokenize=False, add_generation_prompt=False,
    )
    prompt_text = processor.apply_chat_template(
        build_messages(record, include_target=False, image_ref=image_ref),
        tokenize=False, add_generation_prompt=True,
    )
    if not full_text.startswith(prompt_text):
        raise RuntimeError(
            "prompt render is not a prefix of full render — chat template "
            "assumption violated; STOP and inspect processor.apply_chat_template."
        )
    completion_text = full_text[len(prompt_text):]
    target = _v3_target(record)
    if target not in completion_text:
        raise RuntimeError(
            f"target {target!r} not found in completion {completion_text!r}; "
            "STOP and inspect the chat template."
        )
    return {
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "full_text": full_text,
        "target": target,
    }


# ── CLI: CHECKPOINT 2 demonstration ───────────────────────────────────────────

def _demo() -> None:
    """Print one answerable + one abstain example as fed to the model, plus the
    A/B train-set sizes. Run: ``uv run python scripts/format_sft.py``."""
    records = load_master()

    for split in ("train", "eval"):
        a = load_split(split, "A", records)
        b = load_split(split, "B", records)
        print(f"[{split}] SFT-A (gold_localized==True): {len(a)} | "
              f"SFT-B (all): {len(b)} | abstains in B: {len(b) - len(a)}")
    print()

    proc, resolved = load_processor()
    print(f"processor class : {type(proc).__name__}")
    print(f"resolved model  : {resolved}")
    print(f"has tokenizer   : {hasattr(proc, 'tokenizer')} "
          f"({type(getattr(proc, 'tokenizer', None)).__name__})")
    print(f"chat_template   : {'present' if getattr(proc, 'chat_template', None) else 'from tokenizer'}")

    # pick compact menus so the printed examples are readable
    train = load_split("train", "B", records)
    ans_pool = sorted((r for r in train if not r["target_is_abstain"]),
                      key=lambda r: len(r["ocr_menu"]))
    answerable = next((r for r in ans_pool if len(r["ocr_menu"]) >= 10), ans_pool[0])
    abstain = min((r for r in train if r["target_is_abstain"]),
                  key=lambda r: len(r["ocr_menu"]))

    for label, rec in (("ANSWERABLE", answerable), ("ABSTAIN", abstain)):
        ex = render_example(proc, rec)
        print("\n" + "=" * 80)
        print(f"{label}  record_id={rec['record_id']} doc_id={rec['doc_id']} "
              f"type={rec['doc_type']} | menu={len(rec['ocr_menu'])} tokens")
        print("=" * 80)
        print("---- PROMPT (masked to -100 in completion-only loss) ----")
        print(ex["prompt_text"])
        print("---- COMPLETION (the only region trained on) ----")
        print(repr(ex["completion_text"]))
        print(f"---- TARGET (Stage-1 build_target) ----\n{ex['target']}")


if __name__ == "__main__":
    _demo()
