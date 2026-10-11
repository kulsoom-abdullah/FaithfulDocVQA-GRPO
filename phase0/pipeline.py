#!/usr/bin/env python3
"""Phase 0 adversarial audit pipeline for FaithfulDocVQA-GRPO.

Run: uv run python phase0/pipeline.py   (from faithfuldocvqa/)
Requires: ANTHROPIC_API_KEY env var
"""

import json
import random
import base64
import io
import re
import difflib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytesseract
import anthropic

# ── Config ────────────────────────────────────────────────────────────────────

PHASE0_DIR = Path(__file__).parent
OUTPUTS_DIR = PHASE0_DIR / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"
OUTPUTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

SAMPLES_JSON = OUTPUTS_DIR / "samples.json"
AUDIT_JSONL = OUTPUTS_DIR / "audit_results.jsonl"

MODEL = "claude-sonnet-4-6"
BUDGET_CAP = 1.00
CALL_CAP = 0.10
COST_PER_INPUT_MTOK = 3.0    # $/MTok
COST_PER_OUTPUT_MTOK = 15.0  # $/MTok

total_cost_usd = 0.0

# Set global seeds before importing augraphy (it reads them at import time)
random.seed(42)
np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

# ── LEGACY — unused since the v3 contract (Anthropic-API-era cost accounting). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * COST_PER_INPUT_MTOK + output_tokens * COST_PER_OUTPUT_MTOK) / 1_000_000


def image_to_b64(img: Image.Image, max_px: int = 1568) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    import cv2
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    import cv2
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# ── LEGACY — unused since the v3 contract (Anthropic-API-era scoring helper). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def anls(pred: str, golds: list, threshold: float = 0.5) -> float:
    """Compute ANLS score; returns max score across all gold answers."""
    pred = pred.strip().lower()
    best = 0.0
    for gold in golds:
        gold = str(gold).strip().lower()
        ratio = difflib.SequenceMatcher(None, pred, gold).ratio()
        best = max(best, ratio if ratio >= threshold else 0.0)
    return best


# ── Step 1: Sample selection ──────────────────────────────────────────────────

CATEGORY_FILTERS = {
    "form_field":       lambda q: any(w in q.lower() for w in ["invoice", "number", "date", "name", "address"]),
    "table_cell":       lambda q: any(w in q.lower() for w in ["total", "amount", "revenue", "quantity"])
                                  or bool(re.search(r"\bcolumn\b|\brow\b", q.lower())),
    "handwritten":      lambda q: any(w in q.lower() for w in ["signature", "signed", "handwritten"]),
    "layout_position":  lambda q: any(w in q.lower() for w in ["top", "bottom", "left", "right", "header", "footer"]),
    "counting_multihop": lambda q: q.lower().strip().startswith("how many"),
}


def step1_load_samples() -> list[dict]:
    if SAMPLES_JSON.exists():
        print("[Step 1] Loading cached samples.json")
        with open(SAMPLES_JSON) as f:
            return json.load(f)

    print("[Step 1] Loading lmms-lab/DocVQA ...")
    from datasets import load_dataset

    for cfg in ("DocVQA", "docvqa"):
        try:
            ds = load_dataset("lmms-lab/DocVQA", cfg, split="validation", trust_remote_code=True)
            print(f"  Config '{cfg}' loaded: {len(ds)} samples")
            break
        except Exception as e:
            print(f"  Config '{cfg}' failed: {e}")
    else:
        raise RuntimeError("Could not load DocVQA with either config name.")

    chosen: dict[str, dict] = {}
    all_qs: list[tuple[int, str]] = []

    for i, row in enumerate(ds):
        q: str = row["question"]
        raw_ans = row.get("answers") or row.get("answer") or [""]
        gold_list: list[str] = [raw_ans] if isinstance(raw_ans, str) else list(raw_ans)
        all_qs.append((i, q))

        for cat, matcher in CATEGORY_FILTERS.items():
            if cat not in chosen and matcher(q):
                n = len(chosen) + 1
                img: Image.Image = row["image"]
                img_path = IMAGES_DIR / f"sample_{n}_clean.png"
                img.save(img_path)
                chosen[cat] = {
                    "sample_num": n,
                    "dataset_idx": i,
                    "category": cat,
                    "question": q,
                    "gold_answer": gold_list[0],
                    "gold_answers": gold_list,
                    "image_path": str(img_path),
                }
                print(f"  [{n}] {cat}: {q!r} → {gold_list[0]!r}")

        if len(chosen) == 5:
            break

    if len(chosen) < 5:
        print(f"  WARNING: Only found {len(chosen)}/5 categories. Proceeding with what we have.")

    samples = list(chosen.values())

    # Assign OOD questions from different documents
    used_idxs = {s["dataset_idx"] for s in samples}
    ood_pool = [(i, q) for i, q in all_qs if i not in used_idxs]
    for j, sample in enumerate(samples):
        sample["ood_question"] = ood_pool[j % max(len(ood_pool), 1)][1]

    with open(SAMPLES_JSON, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"  Saved {len(samples)} samples → {SAMPLES_JSON}")
    return samples


# ── Step 2: OCR extraction ─────────────────────────────────────────────────────

def extract_ocr_tokens(img: Image.Image) -> list[dict]:
    """Run Tesseract on img and return filtered word tokens with sequential IDs.

    Token IDs
    ---------
    IDs are dense integers 0, 1, 2, … assigned in Tesseract's reading order
    (top-to-bottom, left-to-right within a block). The ID is NOT Tesseract's
    internal row index — it is a counter over the filtered output only. This
    means the same image always produces the same IDs, and the set of valid IDs
    is exactly {0, 1, …, len(tokens)-1}.

    Filtering rules
    ---------------
    A word is excluded if:
      - word.strip() == ""  (blank/whitespace entry from layout analysis), OR
      - conf < 0            (Tesseract uses -1 for region-level rows with no word)
    Words with conf == 0 are INCLUDED — Tesseract produced a string, albeit at
    minimum confidence. The conf field is not stored in the returned dicts; call
    pytesseract.image_to_data directly if you need per-token confidence scores.

    Returned dict keys: id, word, x1, y1, x2 (=left+width), y2 (=top+height).
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    tokens = []
    tid = 0
    for i, word in enumerate(data["text"]):
        if not word.strip() or int(data["conf"][i]) < 0:
            continue
        tokens.append({
            "id": tid,
            "word": word,
            "x1": data["left"][i],
            "y1": data["top"][i],
            "x2": data["left"][i] + data["width"][i],
            "y2": data["top"][i] + data["height"][i],
        })
        tid += 1
    return tokens


def tokens_to_str(tokens: list[dict]) -> str:
    return "\n".join(
        f"{t['id']}\t{t['word']}\t{t['x1']},{t['y1']},{t['x2']},{t['y2']}"
        for t in tokens
    )


# ── Step 3: Answer bbox localization ──────────────────────────────────────────

# ── LEGACY — unused since the v3 contract (superseded by phase05.find_bbox_lenient). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def find_answer_bbox(tokens: list[dict], gold_answer: str, threshold: float = 0.85) -> dict | None:
    """Locate gold_answer in the token list via sliding-window fuzzy match.

    Matching algorithm
    ------------------
    gold_answer is split on whitespace into n words. A sliding window of sizes
    [max(1, n-1), n, n+1] slides over the token list; adjacent tokens are joined
    with a single space. Each window is scored with:
        difflib.SequenceMatcher(None, window.lower(), gold_answer.lower()).ratio()
    — character-level, case-insensitive. Returns the bbox of the highest-scoring
    window that meets the threshold.

    Tokenization note
    -----------------
    Split is whitespace-only; punctuation attached to a word (e.g. "$975.00" as
    a single Tesseract token) is treated as one unit. This is why "$975.00" often
    fails to match: Tesseract may emit it as "$", "975.00" or merge it with
    neighbouring text, while the gold string is the combined form.

    Threshold semantics
    -------------------
    Default 0.85 is used for localization (Step 3 / strict masking).
    verify_masking uses 0.7 — a lower bar chosen deliberately: post-mask OCR
    degrades, so 0.85 would produce false negatives where partial noise drops
    the ratio below 0.85 but the text is still visually readable by the model.

    Returns None if no window meets the threshold.
    """
    if not tokens or not gold_answer.strip():
        return None

    words = [t["word"] for t in tokens]
    gold_words = gold_answer.strip().split()
    n = len(gold_words)
    best_score, best_start, best_n = 0.0, 0, n

    # Sliding window over tokens
    for window_n in range(max(1, n - 1), n + 2):
        for i in range(len(words) - window_n + 1):
            window = " ".join(words[i : i + window_n])
            score = difflib.SequenceMatcher(None, window.lower(), gold_answer.lower()).ratio()
            if score > best_score:
                best_score, best_start, best_n = score, i, window_n

    if best_score < threshold:
        return None

    matched = tokens[best_start : best_start + best_n]
    return {
        "x1": min(t["x1"] for t in matched),
        "y1": min(t["y1"] for t in matched),
        "x2": max(t["x2"] for t in matched),
        "y2": max(t["y2"] for t in matched),
        "score": best_score,
        "token_ids": [t["id"] for t in matched],
    }


# ── Step 4: Noise pipeline ─────────────────────────────────────────────────────

# ── LEGACY — unused since the v3 contract (augraphy/masking — v1 is clean-images-only, CLAUDE.md rule 3). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def build_augraphy_pipeline():
    """Build a reproducible three-stage Augraphy document degradation pipeline.

    Stages: ink (InkBleed) → paper (NoiseTexturize, SubtleNoise) → post (Jpeg,
    BadPhotoCopy). Identical logic exists in phase05.py — consolidation deferred.
    See phase05.py:build_augraphy_pipeline for full stage-by-stage explanation.
    """
    from augraphy import AugraphyPipeline, InkBleed, Jpeg, NoiseTexturize, SubtleNoise, BadPhotoCopy

    ink_phase   = [InkBleed(p=0.6, intensity_range=(0.1, 0.3))]
    paper_phase = [NoiseTexturize(p=0.5), SubtleNoise(p=0.8)]
    post_phase  = [
        Jpeg(p=0.7, quality_range=(30, 50)),
        BadPhotoCopy(p=0.2, noise_concentration=(0.1, 0.3)),
    ]
    # random_seed may not exist in all 8.x builds; fall back to relying on global seeds
    try:
        return AugraphyPipeline(ink_phase, paper_phase, post_phase, random_seed=42)
    except TypeError:
        return AugraphyPipeline(ink_phase, paper_phase, post_phase)


def apply_augraphy(img: Image.Image, pipeline) -> Image.Image:
    """Apply the Augraphy pipeline to a PIL image; reset seeds first for reproducibility.

    Identical to phase05.py:apply_augraphy — consolidation deferred.
    """
    random.seed(42)
    np.random.seed(42)
    result = pipeline(pil_to_bgr(img))
    # Some augraphy versions return a dict; unpack if needed
    if isinstance(result, dict):
        result = result.get("output", result.get("image", list(result.values())[0]))
    return bgr_to_pil(result)


# ── Step 5: Refusal masking ────────────────────────────────────────────────────

def mask_answer_region(clean_img: Image.Image, bbox: dict, pipeline) -> Image.Image:
    """Erase the tight bbox of the answer and fill with background-sampled noise.

    Pads bbox by 15% on all four sides (not full-width), samples mean RGB from
    a 20px ring around the padded region, and fills with uniform ±15 + Gaussian
    σ=8 jitter to avoid a flat rectangle detectable by OCR or the model. Augraphy
    is re-applied after filling so the patch blends with surrounding degradation.

    Note: 15% tight padding was insufficient for Phase 0 — "Diego" remained
    visible in img1_caseC. Phase 0.5 introduced mask_full_line (phase05.py) with
    full-width horizontal masking and 25% vertical padding as the stricter
    alternative. These are two distinct functions with different trade-offs, not
    duplicates.
    """
    img = clean_img.copy().convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    pad_x = max(1, int((x2 - x1) * 0.15))
    pad_y = max(1, int((y2 - y1) * 0.15))
    mx1 = max(0, x1 - pad_x);  my1 = max(0, y1 - pad_y)
    mx2 = min(w, x2 + pad_x);  my2 = min(h, y2 + pad_y)

    # Sample mean RGB from 20px ring around the padded bbox
    ring = 20
    rows = np.arange(max(0, my1 - ring), min(h, my2 + ring))
    cols = np.arange(max(0, mx1 - ring), min(w, mx2 + ring))
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    in_ring = ~((rr >= my1) & (rr < my2) & (cc >= mx1) & (cc < mx2))
    ring_pixels = arr[rr[in_ring], cc[in_ring], :3]
    mean_rgb = ring_pixels.mean(axis=0) if len(ring_pixels) else np.array([255.0, 255.0, 255.0])

    # Paint with jitter + Gaussian noise
    rh, rw = my2 - my1, mx2 - mx1
    jitter = np.random.randint(-15, 16, (rh, rw, 3)).astype(np.float32)
    gauss  = np.random.normal(0, 8, (rh, rw, 3)).astype(np.float32)
    fill   = np.clip(mean_rgb + jitter + gauss, 0, 255).astype(np.uint8)
    arr[my1:my2, mx1:mx2, :3] = fill

    masked_img = Image.fromarray(arr)
    return apply_augraphy(masked_img, pipeline)


def verify_masking(img: Image.Image, gold_answer: str, threshold: float = 0.7) -> bool:
    """Return True if gold_answer is no longer fuzzy-matchable in img via OCR.

    Uses threshold=0.7 (not 0.85) because post-mask OCR quality degrades and a
    stricter threshold would produce false negatives: partial noise can drop the
    match ratio below 0.85 while the text remains visually readable to the model.
    A False return means the answer is still recoverable and the mask is invalid.
    """
    tokens = extract_ocr_tokens(img)
    return find_answer_bbox(tokens, gold_answer, threshold=threshold) is None


def filter_tokens_for_masked(tokens: list[dict], bbox: dict) -> list[dict]:
    """Remove tokens that overlap the masked region."""
    return [
        t for t in tokens
        if not (t["x1"] < bbox["x2"] and t["x2"] > bbox["x1"]
                and t["y1"] < bbox["y2"] and t["y2"] > bbox["y1"])
    ]


# ── Step 7: Teacher prompt ─────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a document-understanding tutor generating grounded training traces. You will see:
(1) one document image,
(2) a numbered OCR word list, each line formatted as <id>\t<word>\t<x1,y1,x2,y2>,
(3) one question about the document.

Produce output in this exact XML schema:

<rationale>
  <locate>1–2 sentences describing where evidence appears in layout terms. No pixel numbers.</locate>
  <evidence_tokens>[id1, id2, ...]</evidence_tokens>
  <evidence_text>"verbatim concatenation of cited tokens"</evidence_text>
  <derive>One sentence linking evidence to the answer.</derive>
</rationale>
<answer>SHORT DocVQA-style answer, lowercase unless proper noun, no trailing period.</answer>

Hard constraints:
- <evidence_tokens> MUST be a JSON array of integer IDs drawn ONLY from the provided list. Never fabricate IDs.
- If the answer cannot be found or the relevant region is unreadable:
    <evidence_tokens>[]</evidence_tokens>
    <evidence_text>""</evidence_text>
    <answer>unanswerable</answer>
- Keep <rationale> under 90 words."""


def call_teacher(
    img: Image.Image,
    tokens: list[dict],
    question: str,
    client: anthropic.Anthropic,
) -> dict:
    """Deprecated — use call_teacher_with_retry instead."""
    global total_cost_usd

    token_str = tokens_to_str(tokens)
    user_text = f"OCR tokens:\n{token_str}\n\nQuestion: {question}"

    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_to_b64(img),
                    },
                },
                {"type": "text", "text": user_text},
            ],
        }],
    )

    raw = response.content[0].text
    input_tok = response.usage.input_tokens
    output_tok = response.usage.output_tokens
    cost = compute_cost(input_tok, output_tok)
    total_cost_usd += cost

    return {"raw": raw, "input_tokens": input_tok, "output_tokens": output_tok, "cost": cost}


# ── Self-correction retry ─────────────────────────────────────────────────────

# System prompt as a cacheable block. Per-sample content (image/menu/question)
# is NOT included here — it changes every call and must not be cached.
CACHED_SYSTEM = [{"type": "text", "text": SYSTEM_PROMPT,
                  "cache_control": {"type": "ephemeral"}}]


def _build_user_content(img: Image.Image, tokens: list[dict], question: str) -> list:
    """Build the user-turn content block for one teacher API call.

    Returns a two-element list: [image block (base64 PNG), text block (OCR menu +
    question)]. This is the per-sample content that changes on every call and must
    NOT be placed under cache_control — only the system prompt (CACHED_SYSTEM) is
    eligible for caching. Called once per sample and reused across both the first
    attempt and the retry turn to guarantee identical inputs.
    """
    token_str = tokens_to_str(tokens)
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": image_to_b64(img)},
        },
        {"type": "text", "text": f"OCR tokens:\n{token_str}\n\nQuestion: {question}"},
    ]


def _raw_call(messages: list, client: anthropic.Anthropic) -> dict:
    """Single Messages API call with cached system prompt. Updates total_cost_usd."""
    global total_cost_usd
    response = client.messages.create(
        model=MODEL, max_tokens=1024, system=CACHED_SYSTEM, messages=messages,
    )
    raw           = response.content[0].text
    inp           = response.usage.input_tokens
    out           = response.usage.output_tokens
    cache_read    = getattr(response.usage, "cache_read_input_tokens",    0) or 0
    cache_created = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cost          = compute_cost(inp, out)
    total_cost_usd += cost
    return {"raw": raw, "input_tokens": inp, "output_tokens": out,
            "cache_read": cache_read, "cache_created": cache_created, "cost_usd": cost}


def _correction_message(parsed: dict, valid_ids: set[int]) -> tuple[str, str]:
    """Return (failure_reason, correction_text) for the retry turn.

    Priority order (same as call_teacher_with_retry docstring):
      1. out_of_range_ids — parse OK but cited IDs absent from the OCR menu
      2. malformed_json   — <evidence_tokens> was not parseable JSON/ints
      3. missing_tags     — format_valid == 0
    """
    evidence_tokens = parsed["parsed_evidence_tokens"]
    tokens_parse_ok = parsed["tokens_parse_ok"]

    if tokens_parse_ok and evidence_tokens:
        bad = [t for t in evidence_tokens if t not in valid_ids]
        if bad:
            max_id = max(valid_ids) if valid_ids else 0
            return (
                "out_of_range_ids",
                f"You cited token ID(s) {bad} in <evidence_tokens>. These IDs do not "
                f"exist in the OCR token menu for this image. The valid range is 0 to "
                f"{max_id}. Please review the menu and respond again with the same XML "
                f"schema, citing only IDs that exist in the menu. If the answer cannot "
                f"be grounded in the menu, respond with an empty evidence list and answer "
                f"'unanswerable'.",
            )

    if not tokens_parse_ok:
        return (
            "malformed_json",
            "The content inside <evidence_tokens> was not a valid JSON list of integers. "
            "Please respond again with the exact same schema, ensuring evidence_tokens "
            "contains a valid JSON list like [3, 7, 12] or [].",
        )

    return (
        "missing_tags",
        "Your response was missing one or more required tags. The required schema is:\n"
        "<rationale><locate>...</locate><evidence_tokens>[...]</evidence_tokens>"
        "<evidence_text>...</evidence_text><derive>...</derive></rationale>\n"
        "<answer>...</answer>\n"
        "Please respond again using this exact structure.",
    )


def call_teacher_with_retry(
    img: Image.Image,
    tokens: list[dict],
    question: str,
    client: anthropic.Anthropic,
    case_id: str = "",
) -> dict:
    """Call the teacher model with one optional self-correction retry.

    Retry trigger
    -------------
    A retry is issued when the first response has evidence_valid==0
    (fabricated/out-of-range ID or malformed JSON) OR format_valid==0
    (missing required XML tags).

    Correction message priority
    ---------------------------
    1. out_of_range_ids : parse OK but cited IDs absent from the OCR menu
    2. malformed_json   : <evidence_tokens> content was not valid JSON/ints
    3. missing_tags     : one or more required XML tags were missing

    Retry message list
    ------------------
    [user: original content, assistant: first raw response, user: correction]

    Prompt caching
    --------------
    CACHED_SYSTEM carries cache_control={"type":"ephemeral"} so the system
    prompt is served from cache on the second call. Per-sample content
    (image, OCR menu, question) is NOT cached — it changes every call.

    Returns
    -------
    {
      "final_parsed"    : dict | None  — parse_response output of the successful call,
                                         or None if both attempts failed validation.
                                         Always check retry_succeeded before accessing.
      "attempts"        : list  — one or two entries, each:
                                    {call_n, parsed, raw,
                                     usage: {input_tokens, output_tokens,
                                             cache_read, cache_created, cost_usd}}
      "retry_triggered" : bool
      "retry_succeeded" : bool  — True iff retry's parsed result passes validation
      "failure_reason"  : str | None
    }
    """
    valid_ids    = {t["id"] for t in tokens}
    user_content = _build_user_content(img, tokens, question)

    # ── Attempt 1 ─────────────────────────────────────────────────────────────
    res1    = _raw_call([{"role": "user", "content": user_content}], client)
    parsed1 = parse_response(res1["raw"], valid_ids)
    attempt1 = {
        "call_n": 1, "parsed": parsed1, "raw": res1["raw"],
        "usage": {k: res1[k] for k in
                  ("input_tokens", "output_tokens", "cache_read", "cache_created", "cost_usd")},
    }

    if parsed1["evidence_valid"] == 1 and parsed1["format_valid"] == 1:
        return {
            "final_parsed": parsed1, "attempts": [attempt1],
            "retry_triggered": False, "retry_succeeded": False, "failure_reason": None,
        }

    # ── Attempt 2 (retry) ─────────────────────────────────────────────────────
    failure_reason, correction_text = _correction_message(parsed1, valid_ids)
    first_sentence = correction_text.split(".")[0] + "."
    print(f"      [retry] {case_id} | reason={failure_reason} | {first_sentence}")

    res2 = _raw_call(
        [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": res1["raw"]},
            {"role": "user",      "content": correction_text},
        ],
        client,
    )
    parsed2 = parse_response(res2["raw"], valid_ids)
    attempt2 = {
        "call_n": 2, "parsed": parsed2, "raw": res2["raw"],
        "usage": {k: res2[k] for k in
                  ("input_tokens", "output_tokens", "cache_read", "cache_created", "cost_usd")},
    }

    retry_succeeded = parsed2["evidence_valid"] == 1 and parsed2["format_valid"] == 1
    return {
        "final_parsed": parsed2 if retry_succeeded else None,
        "attempts": [attempt1, attempt2],
        "retry_triggered": True, "retry_succeeded": retry_succeeded,
        "failure_reason": failure_reason,
    }


# ── Step 8: Parse + evaluate response ─────────────────────────────────────────

def parse_response(raw: str, valid_ids: set[int]) -> dict:
    """Parse model XML output and compute the four reward-component flags.

    evidence_valid contract
    -----------------------
        evidence_valid = 0  if JSON inside <evidence_tokens> is malformed
                             (json.JSONDecodeError or int-conversion ValueError)
        evidence_valid = 1  if parse succeeded AND evidence_tokens is []
                             (empty list → correct refusal, vacuously true)
        evidence_valid = 1  if parse succeeded AND every ID is in valid_ids
        evidence_valid = 0  if parse succeeded AND any ID is absent from valid_ids

    Out-of-range IDs (e.g. [975] when valid range is 0–144) set evidence_valid=0;
    no exception is raised.

    Empty-tokens edge case
    ----------------------
    evidence_tokens=[] + parse_ok=True → evidence_valid=1 (correct refusal).
    To distinguish a genuine refusal from a silent parse failure, check:
        evidence_valid==1 AND parsed_answer=="unanswerable" AND format_valid==1

    format_valid contract
    ---------------------
    Returns 1 iff the raw string contains all five structural tags:
        <rationale>, <locate>…</locate>, <evidence_tokens>…</evidence_tokens>,
        <derive>…</derive>, <answer>…</answer>
    Only tag presence is checked; malformed nesting is not caught.
    """
    tok_match  = re.search(r"<evidence_tokens>(.*?)</evidence_tokens>", raw, re.DOTALL)
    ans_match  = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    loc_match  = re.search(r"<locate>.*?</locate>", raw, re.DOTALL)
    drv_match  = re.search(r"<derive>.*?</derive>", raw, re.DOTALL)

    evidence_tokens: list[int] = []
    _parse_ok = True
    if tok_match:
        try:
            evidence_tokens = [int(x) for x in json.loads(tok_match.group(1).strip())]
        except (json.JSONDecodeError, ValueError):
            evidence_tokens = []
            _parse_ok = False

    parsed_answer = ans_match.group(1).strip() if ans_match else ""

    format_valid = int(bool(
        re.search(r"<rationale>", raw) and loc_match and tok_match and ans_match and drv_match
    ))
    if not _parse_ok:
        evidence_valid = 0
    elif evidence_tokens:
        evidence_valid = int(all(t in valid_ids for t in evidence_tokens))
    else:
        evidence_valid = 1  # [] with successful parse = correct refusal

    return {
        "parsed_evidence_tokens": evidence_tokens,
        "parsed_answer":          parsed_answer,
        "format_valid":           format_valid,
        "evidence_valid":         evidence_valid,
        "tokens_parse_ok":        _parse_ok,
    }


# ── Evidence reward (piecewise) ───────────────────────────────────────────────
# Extracted to phase0/evidence_reward.py (dependency-free) so the lean training/
# eval path can import it without pulling in this module's heavy audit deps.
# Re-exported here so existing `from phase0.pipeline import ...` imports still work.

from phase0.evidence_reward import (  # noqa: E402,F401
    ABSTAIN_CREDIT,
    FABRICATION_PENALTY,
    compute_evidence_reward,
)


# ── Step 9: Visual grid ────────────────────────────────────────────────────────

def save_grid(n: int, clean: Image.Image, noisy: Image.Image, masked: Image.Image):
    cell_w, cell_h = 450, 340
    labels = ["A: Clean", "B: Noisy", "C: Masked", "D: OOD (clean image)"]
    images = [clean, noisy, masked, clean]

    grid = Image.new("RGB", (cell_w * 2, cell_h * 2), (220, 220, 220))
    draw = ImageDraw.Draw(grid)
    positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]

    for (px, py), img, label in zip(positions, images, labels):
        thumb = img.copy().convert("RGB")
        thumb.thumbnail((cell_w - 10, cell_h - 30), Image.LANCZOS)
        tw, th = thumb.size
        grid.paste(thumb, (px + (cell_w - tw) // 2, py + 25 + (cell_h - 30 - th) // 2))
        draw.rectangle([px, py, px + cell_w - 1, py + cell_h - 1], outline=(80, 80, 80), width=2)
        draw.rectangle([px, py, px + cell_w - 1, py + 22], fill=(50, 50, 50))
        draw.text((px + 6, py + 4), label, fill=(255, 255, 255))

    out = OUTPUTS_DIR / f"sample_{n}_grid.png"
    grid.save(out)
    print(f"    Saved grid → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global total_cost_usd

    client = anthropic.Anthropic()
    pipeline = build_augraphy_pipeline()
    audit_records: list[dict] = []

    # Clear previous audit log
    if AUDIT_JSONL.exists():
        AUDIT_JSONL.unlink()

    # ── Step 1
    samples = step1_load_samples()

    for sample in samples:
        n = sample["sample_num"]
        print(f"\n── Sample {n} [{sample['category']}] ──────────────────────")

        clean_img = Image.open(sample["image_path"])

        # ── Step 2
        print("  [2] OCR on clean image...")
        clean_tokens = extract_ocr_tokens(clean_img)
        print(f"      {len(clean_tokens)} tokens")

        # ── Step 3
        print("  [3] Localizing answer bbox...")
        bbox = find_answer_bbox(clean_tokens, sample["gold_answer"])
        if bbox:
            print(f"      bbox={bbox} score={bbox['score']:.2f}")
        else:
            print(f"      WARNING: no bbox for {sample['gold_answer']!r} — Case C will use noisy image unmasked")
        sample["answer_bbox"] = bbox
        sample["answer_bbox_found"] = bbox is not None

        # ── Step 4
        print("  [4] Generating noisy image...")
        noisy_img = apply_augraphy(clean_img, pipeline)
        noisy_path = IMAGES_DIR / f"sample_{n}_noisy.png"
        noisy_img.save(noisy_path)
        noisy_tokens = extract_ocr_tokens(noisy_img)
        print(f"      {len(noisy_tokens)} noisy tokens → {noisy_path.name}")

        # ── Step 5
        print("  [5] Applying refusal mask...")
        if bbox:
            masked_img = mask_answer_region(clean_img, bbox, pipeline)
            masked_path = IMAGES_DIR / f"sample_{n}_masked.png"
            masked_img.save(masked_path)
            masking_ok = verify_masking(masked_img, sample["gold_answer"])
            print(f"      Masking verified (answer gone): {masking_ok}")
            masked_tokens = filter_tokens_for_masked(noisy_tokens, bbox)
        else:
            masked_img = noisy_img
            masked_path = noisy_path
            masked_tokens = noisy_tokens

        # ── Step 9: grids for samples 1 & 2
        if n in (1, 2):
            save_grid(n, clean_img, noisy_img, masked_img)

        # ── Steps 6 + 7: build cases and call teacher
        cases = [
            {
                "case":             "A",
                "img":              clean_img,
                "tokens":           clean_tokens,
                "question":         sample["question"],
                "expected_behavior": "answer correctly with cited tokens",
                "is_refusal_case":  False,
            },
            {
                "case":             "B",
                "img":              noisy_img,
                "tokens":           noisy_tokens,
                "question":         sample["question"],
                "expected_behavior": "answer correctly with cited tokens",
                "is_refusal_case":  False,
            },
            {
                "case":             "C",
                "img":              masked_img,
                "tokens":           masked_tokens,
                "question":         sample["question"],
                "expected_behavior": "refuse with empty tokens",
                "is_refusal_case":  True,
            },
            {
                "case":             "D",
                "img":              clean_img,
                "tokens":           clean_tokens,
                "question":         sample["ood_question"],
                "expected_behavior": "refuse with empty tokens",
                "is_refusal_case":  True,
            },
        ]

        for ci in cases:
            case_id = f"img{n}_case{ci['case']}"

            if total_cost_usd >= BUDGET_CAP:
                print(f"\n  HARD BUDGET CAP ${BUDGET_CAP:.2f} reached. Stopping.")
                break

            print(f"  [7] {case_id} → calling {MODEL}...", end=" ", flush=True)
            result    = call_teacher_with_retry(ci["img"], ci["tokens"], ci["question"],
                                                client, case_id=case_id)
            call_cost = sum(a["usage"]["cost_usd"] for a in result["attempts"])
            retry_tag = " [retried]" if result["retry_triggered"] else ""
            print(f"${call_cost:.4f} (total ${total_cost_usd:.4f}){retry_tag}")

            if call_cost > CALL_CAP:
                print(f"      ⚠ Call cost ${call_cost:.4f} exceeds cap ${CALL_CAP}. Logging but continuing.")

            if result["final_parsed"] is None:
                # Both attempts failed validation; log attempt-2 output for audit trail
                print(f"      [double-fail] {case_id} — both attempts invalid")
                parsed = result["attempts"][-1]["parsed"]
            else:
                parsed = result["final_parsed"]
            pred   = parsed["parsed_answer"].lower().strip()
            if ci["is_refusal_case"]:
                answer_correct      = int("unanswerable" in pred)
                refusal_appropriate = answer_correct
            else:
                answer_correct      = int(anls(pred, sample["gold_answers"]) >= 0.5)
                refusal_appropriate = int("unanswerable" not in pred)

            record = {
                "case_id":                case_id,
                "image_path":             str(IMAGES_DIR / f"sample_{n}_{'clean' if ci['case'] in ('A','D') else ('noisy' if ci['case']=='B' else 'masked')}.png"),
                "question":               ci["question"],
                "gold_answer":            sample["gold_answer"],
                "teacher_raw_response":   result["attempts"][-1]["raw"],
                "parsed_evidence_tokens": parsed["parsed_evidence_tokens"],
                "parsed_answer":          parsed["parsed_answer"],
                "answer_correct":         answer_correct,
                "evidence_valid":         parsed["evidence_valid"],
                "format_valid":           parsed["format_valid"],
                "refusal_appropriate":    refusal_appropriate,
                "expected_behavior":      ci["expected_behavior"],
                "actual_behavior":        f"answered '{parsed['parsed_answer']}'",
                "cost_usd":               call_cost,
                "input_tokens":           sum(a["usage"]["input_tokens"]  for a in result["attempts"]),
                "output_tokens":          sum(a["usage"]["output_tokens"] for a in result["attempts"]),
                "retry_triggered":        result["retry_triggered"],
                "retry_succeeded":        result["retry_succeeded"],
                "failure_reason":         result["failure_reason"],
            }
            audit_records.append(record)

            with open(AUDIT_JSONL, "a") as f:
                f.write(json.dumps(record) + "\n")

            print(f"      answer={parsed['parsed_answer']!r}  correct={answer_correct}"
                  f"  fmt={parsed['format_valid']}  ev_valid={parsed['evidence_valid']}")

        else:
            continue  # inner for completed normally — move to next sample
        break           # budget exceeded — exit outer loop

    # ── Step 10: Summary table
    print(f"\n── Total API cost: ${total_cost_usd:.4f} ──────────────────────")
    if audit_records:
        df = pd.DataFrame(audit_records)
        df["case_type"] = df["case_id"].str[-1]
        metrics = ["answer_correct", "evidence_valid", "format_valid", "refusal_appropriate"]
        summary = df.groupby("case_type")[metrics].mean().round(2)
        summary.index.name = "Case"
        print("\n── Pass rates by case ──")
        print(summary.to_string())
        summary.to_csv(OUTPUTS_DIR / "summary.csv")
        print(f"\nSaved summary → {OUTPUTS_DIR / 'summary.csv'}")

    print(f"\nAudit log → {AUDIT_JSONL}")
    return audit_records


if __name__ == "__main__":
    main()
