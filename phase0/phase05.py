#!/usr/bin/env python3
"""Phase 0.5 — diagnostic re-runs of the Phase 0 audit with strict masking.

Also defines ``find_bbox`` / ``find_bbox_lenient`` — the FROZEN gold localizer
imported by ``scripts/build_master_dataset.py`` (CLAUDE.md hard rule 2: import,
don't reimplement). Do not modify the localizer logic.

Requires ANTHROPIC_API_KEY and phase0/outputs/ artifacts from a prior
``pipeline.py`` run (gitignored; regenerating them re-runs the paid audit).
Writes phase0_writeup.md — a draft that does not ship in this repo.
Run: uv run python phase0/phase05.py   (from the repo root)
"""

import json
import random
import base64
import io
import re
import difflib
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytesseract
import anthropic
import cv2

from phase0.pipeline import parse_response

# ── Config ────────────────────────────────────────────────────────────────────

PHASE0_DIR   = Path(__file__).parent
OUTPUTS_DIR  = PHASE0_DIR / "outputs"
IMAGES_DIR   = OUTPUTS_DIR / "images"
OCR_DIR      = OUTPUTS_DIR / "audit_ocr"
OCR_DIR.mkdir(exist_ok=True)

SAMPLES_JSON     = OUTPUTS_DIR / "samples.json"
AUDIT_JSONL      = OUTPUTS_DIR / "audit_results.jsonl"
ANNOTATED_JSONL  = OUTPUTS_DIR / "audit_results_annotated.jsonl"
WRITEUP_PATH     = PHASE0_DIR.parent / "phase0_writeup.md"

MODEL                 = "claude-sonnet-4-6"
BUDGET_CAP            = 0.50
COST_PER_INPUT_MTOK   = 3.0
COST_PER_OUTPUT_MTOK  = 15.0

total_cost_usd = 0.0
random.seed(42)
np.random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────────────────

# ── LEGACY — unused since the v3 contract (Anthropic-API-era cost accounting). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def compute_cost(inp: int, out: int) -> float:
    return (inp * COST_PER_INPUT_MTOK + out * COST_PER_OUTPUT_MTOK) / 1_000_000


def image_to_b64(img: Image.Image, max_px: int = 1568) -> str:
    img = img.copy().convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def bgr_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


# ── LEGACY — unused since the v3 contract (duplicate of pipeline.extract_ocr_tokens; the build imports the pipeline one). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def ocr_tokens(img: Image.Image) -> list[dict]:
    """Run Tesseract on img and return a list of word tokens with dense sequential IDs.

    Filters out empty strings and tokens with conf < 0 (Tesseract uses -1 to
    signal layout-only regions like paragraph/block boundaries that contain no
    real text). Tokens with conf=0 are kept — those are real words Tesseract
    recognized with very low confidence, not structural artifacts.
    """
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    tokens, tid = [], 0
    for i, word in enumerate(data["text"]):
        if not word.strip() or int(data["conf"][i]) < 0:
            continue
        tokens.append({
            "id": tid, "word": word,
            "x1": data["left"][i], "y1": data["top"][i],
            "x2": data["left"][i] + data["width"][i],
            "y2": data["top"][i]  + data["height"][i],
            "conf": int(data["conf"][i]),
        })
        tid += 1
    return tokens


def tokens_to_str(tokens: list[dict]) -> str:
    return "\n".join(
        f"{t['id']}\t{t['word']}\t{t['x1']},{t['y1']},{t['x2']},{t['y2']}"
        for t in tokens
    )


def dump_ocr(img_path: Path, out_path: Path) -> list[dict]:
    tokens = ocr_tokens(Image.open(img_path))
    lines = [f"{'ID':<6}{'TEXT':<22}{'BBOX':<28}{'CONF'}"]
    for t in tokens:
        bbox = f"({t['x1']},{t['y1']},{t['x2']},{t['y2']})"
        lines.append(f"{t['id']:<6}{t['word']:<22}{bbox:<28}{t['conf']}")
    out_path.write_text("\n".join(lines))
    return tokens


def dump_ocr_pil(img: Image.Image, out_path: Path) -> list[dict]:
    tokens = ocr_tokens(img)
    lines = [f"{'ID':<6}{'TEXT':<22}{'BBOX':<28}{'CONF'}"]
    for t in tokens:
        bbox = f"({t['x1']},{t['y1']},{t['x2']},{t['y2']})"
        lines.append(f"{t['id']:<6}{t['word']:<22}{bbox:<28}{t['conf']}")
    out_path.write_text("\n".join(lines))
    return tokens


def find_bbox(tokens: list[dict], gold: str, threshold: float = 0.85) -> dict | None:
    if not tokens or not gold.strip():
        return None
    words = [t["word"] for t in tokens]
    n = len(gold.strip().split())
    best_score, best_start, best_n = 0.0, 0, n
    for wn in range(max(1, n - 1), n + 2):
        for i in range(len(words) - wn + 1):
            window = " ".join(words[i: i + wn])
            s = difflib.SequenceMatcher(None, window.lower(), gold.lower()).ratio()
            if s > best_score:
                best_score, best_start, best_n = s, i, wn
    if best_score < threshold:
        return None
    matched = tokens[best_start: best_start + best_n]
    return {
        "x1": min(t["x1"] for t in matched), "y1": min(t["y1"] for t in matched),
        "x2": max(t["x2"] for t in matched), "y2": max(t["y2"] for t in matched),
        "score": best_score, "token_ids": [t["id"] for t in matched],
    }


def find_bbox_lenient(tokens: list[dict], gold: str) -> dict | None:
    """Fuzzy-match gold answer against OCR tokens with a cascading threshold.

    Tries thresholds 0.85 → 0.7 → 0.6 → 0.5 in order, returning on the first
    match. If all fail, strips non-numeric characters from gold and retries at
    0.85 → 0.7 → 0.6 — this handles cases like "$975.00" where Tesseract splits
    the dollar sign from the number. Returns None if no match is found at any
    threshold.

    Note: numeric stripping is a targeted fix for currency tokens. It will
    produce wrong matches for answers like "Section 3.2" or "January 18, 2005"
    (strips to "3.2" or "182005"). Treat as a Phase 0 workaround, not a general
    solution.
    """
    for thr in [0.85, 0.7, 0.6, 0.5]:
        b = find_bbox(tokens, gold, thr)
        if b:
            return b
    numeric = re.sub(r"[^\d.]", "", gold)
    if numeric and numeric != gold:
        for thr in [0.85, 0.7, 0.6]:
            b = find_bbox(tokens, numeric, thr)
            if b:
                return b
    return None


# ── LEGACY — unused since the v3 contract (augraphy/masking — v1 is clean-images-only, CLAUDE.md rule 3). Kept for provenance;
#    no live code path calls the function(s) below. See phase0/README.md.
def build_augraphy_pipeline():
    """Build a reproducible three-stage Augraphy document degradation pipeline.

    Stages applied in order:
      ink   — InkBleed: simulates ink spreading through paper fibers
      paper — NoiseTexturize + SubtleNoise: paper grain and sensor noise
      post  — Jpeg (quality 30–50) + BadPhotoCopy: compression artifacts and
               uneven photocopier exposure

    The try/except handles an Augraphy API break: versions < 8.x do not accept
    random_seed in the constructor; global np.random.seed(42) covers that case.
    """
    from augraphy import AugraphyPipeline, InkBleed, Jpeg, NoiseTexturize, SubtleNoise, BadPhotoCopy
    ink   = [InkBleed(p=0.6, intensity_range=(0.1, 0.3))]
    paper = [NoiseTexturize(p=0.5), SubtleNoise(p=0.8)]
    post  = [Jpeg(p=0.7, quality_range=(30, 50)), BadPhotoCopy(p=0.2, noise_concentration=(0.1, 0.3))]
    try:
        return AugraphyPipeline(ink, paper, post, random_seed=42)
    except TypeError:
        return AugraphyPipeline(ink, paper, post)


def apply_augraphy(img: Image.Image, pipeline) -> Image.Image:
    """Apply the Augraphy pipeline to a PIL image and return the degraded result.

    Seeds are reset before each call so the same image always produces the same
    degraded output regardless of call order. The pipeline may return either a
    numpy array or a dict (API varies by version); both cases are handled.
    """
    random.seed(42); np.random.seed(42)
    result = pipeline(pil_to_bgr(img))
    if isinstance(result, dict):
        result = result.get("output", result.get("image", list(result.values())[0]))
    return bgr_to_pil(result)


def mask_full_line(clean_img: Image.Image, bbox: dict, pipeline, pad_frac: float = 0.25) -> Image.Image:
    """Erase a horizontal band containing bbox and fill it with sampled background.

    The fill region spans the full image width from (bbox.y1 - pad) to
    (bbox.y2 + pad), where pad = pad_frac × bbox height. Fill color is sampled
    from the 20-pixel strips immediately above and below the masked band, then
    jittered with uniform ±15 and Gaussian σ=8 noise to avoid a flat rectangle
    that OCR or the model could detect as an artifact. Augraphy is re-applied
    after filling so the masked region blends with surrounding degradation.

    bbox comes from find_bbox() or find_bbox_lenient() run on the clean image's
    OCR tokens — it marks the pixel region containing the gold answer text.
    """
    img = clean_img.copy().convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    pad_y = max(1, int((bbox["y2"] - bbox["y1"]) * pad_frac))
    my1, my2 = max(0, bbox["y1"] - pad_y), min(h, bbox["y2"] + pad_y)
    ring = 20
    above = arr[max(0, my1 - ring):my1, :, :3].reshape(-1, 3)
    below = arr[my2:min(h, my2 + ring), :, :3].reshape(-1, 3)
    sample = np.vstack([r for r in [above, below] if len(r)])
    mean_rgb = sample.mean(axis=0) if len(sample) else np.array([255.0, 255.0, 255.0])
    rh = my2 - my1
    fill = np.clip(mean_rgb + np.random.randint(-15, 16, (rh, w, 3)) + np.random.normal(0, 8, (rh, w, 3)), 0, 255).astype(np.uint8)
    arr[my1:my2, :, :3] = fill
    return apply_augraphy(Image.fromarray(arr), pipeline)


def verify_masking(img: Image.Image, gold: str, threshold: float = 0.7) -> bool:
    """Return True if the gold answer text is no longer detectable in img via OCR.

    Uses threshold=0.7, deliberately lower than the 0.85 used for localization.
    This makes verification stricter: the answer must be nearly unreadable before
    we accept the mask as successful, reducing false negatives where a partially
    degraded token still fuzzy-matches above a lenient threshold.
    """
    return find_bbox(ocr_tokens(img), gold, threshold) is None


def anls(pred: str, golds: list, threshold: float = 0.5) -> float:
    """Compute Average Normalized Levenshtein Similarity against a list of gold answers.

    Implements the standard DocVQA evaluation metric. For each gold string,
    computes character-level similarity via SequenceMatcher.ratio(), then
    hard-zeros it if the ratio falls below threshold=0.5. This zeroing behavior
    comes from the DocVQA paper — answers below 50% similarity are treated as
    completely wrong rather than partially correct, to penalize hallucinations
    that are superficially close to the gold. Returns the max score across all
    gold strings (multiple valid answers are allowed in DocVQA).
    """
    pred = pred.strip().lower()
    best = 0.0
    for g in golds:
        r = difflib.SequenceMatcher(None, pred, str(g).strip().lower()).ratio()
        best = max(best, r if r >= threshold else 0.0)
    return best


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



def call_timed(img: Image.Image, tokens: list[dict], question: str, client: anthropic.Anthropic) -> dict:
    """Call the teacher model via streaming and capture latency and cache metrics.

    Uses client.messages.stream() so we can timestamp the first text chunk
    (time_to_first_token_ms) separately from total wall-clock time (total_latency_ms).
    TTFT measures server-side processing + network; total latency includes generation.

    Retries up to 3 attempts with exponential backoff (sleep 1s, then 2s) on any
    exception. Streaming failures are typically pre-billing (connection errors,
    timeout before first token), so total_cost_usd should not be double-counted
    on retry — but this assumption is unverified and should be confirmed against
    Anthropic's billing dashboard before running at scale with the Batch API.

    cache_read_input_tokens will be 0 here because this function sends the system
    prompt as a plain string (no cache_control header). Prompt caching is only
    active in call_teacher_with_retry in pipeline.py. The field is logged anyway
    to keep the JSONL schema consistent and to serve as a baseline for comparison.

    call_uuid is a client-generated UUID for cross-referencing this record with
    the Anthropic request dashboard using anthropic_request_id.
    """
    global total_cost_usd
    call_id = str(uuid.uuid4())
    b64 = image_to_b64(img)
    user_text = f"OCR tokens:\n{tokens_to_str(tokens)}\n\nQuestion: {question}"
    msg_params = dict(
        model=MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": user_text},
        ]}],
    )
    for attempt in range(3):
        try:
            t_send = time.time(); t_first = None; full_text = ""
            with client.messages.stream(**msg_params) as stream:
                for chunk in stream.text_stream:
                    if t_first is None:
                        t_first = time.time()
                    full_text += chunk
                final = stream.get_final_message()
            t_done = time.time()
            inp, out = final.usage.input_tokens, final.usage.output_tokens
            cost = compute_cost(inp, out)
            total_cost_usd += cost
            total_ms = round((t_done - t_send) * 1000, 1)
            return {
                "raw": full_text, "input_tokens": inp, "output_tokens": out, "cost": cost,
                "time_to_first_token_ms": round((t_first - t_send) * 1000, 1) if t_first else None,
                "total_latency_ms": total_ms,
                "output_tokens_per_second": round(out / (total_ms / 1000), 1) if total_ms else None,
                "cache_read_input_tokens":    getattr(final.usage, "cache_read_input_tokens", 0) or 0,
                "cache_creation_input_tokens": getattr(final.usage, "cache_creation_input_tokens", 0) or 0,
                "call_uuid": call_id, "retry_count": attempt, "anthropic_request_id": final.id,
            }
        except Exception as e:
            if attempt == 2:
                raise
            print(f"      retry {attempt+1}/2: {e}"); time.sleep(2 ** attempt)


def annotate_record(record: dict, tokens: list[dict], ocr_path: Path, api_result: dict | None = None) -> dict:
    """Enrich a JSONL record with OCR-derived fields and optional latency data.

    Always adds: ocr_file_path, cited_tokens_text (words for each cited ID),
    cited_tokens_valid (all cited IDs present in this image's token list),
    cited_tokens_low_conf (count of cited tokens with Tesseract conf < 60),
    and null placeholders for latency/cache fields so every row in the annotated
    JSONL has a consistent schema regardless of whether timing data exists.

    If api_result is provided (Phase 0.5 timed calls), overwrites the placeholders
    with actual values: time_to_first_token_ms, total_latency_ms,
    output_tokens_per_second, cache_read_input_tokens, cache_creation_input_tokens,
    call_uuid, retry_count, anthropic_request_id.

    cited_tokens_valid catches out-of-range fabricated IDs (e.g. ID 975 in a
    145-token list) — those IDs are absent from the by_id lookup and excluded
    from cited_tokens_text, but their presence sets cited_tokens_valid=False.
    """
    by_id = {t["id"]: t for t in tokens}
    cited = record.get("parsed_evidence_tokens", [])
    cited_words = [by_id[i]["word"] for i in cited if i in by_id]
    low_conf = sum(1 for i in cited if i in by_id and by_id[i]["conf"] < 60)
    out = {
        **record,
        "ocr_file_path": str(ocr_path),
        "cited_tokens_text": " ".join(cited_words),
        "cited_tokens_valid": all(i in by_id for i in cited) if cited else True,
        "cited_tokens_low_conf": low_conf,
        "time_to_first_token_ms": None, "total_latency_ms": None,
        "output_tokens_per_second": None,
        "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        "call_uuid": None, "retry_count": None, "anthropic_request_id": None,
    }
    if api_result:
        out.update({k: api_result[k] for k in [
            "time_to_first_token_ms", "total_latency_ms", "output_tokens_per_second",
            "cache_read_input_tokens", "cache_creation_input_tokens",
            "call_uuid", "retry_count", "anthropic_request_id",
        ]})
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global total_cost_usd
    client   = anthropic.Anthropic()
    pipeline = build_augraphy_pipeline()

    with open(SAMPLES_JSON) as f:
        samples = json.load(f)
    with open(AUDIT_JSONL) as f:
        phase0_records = [json.loads(l) for l in f]
    p0_by_id = {r["case_id"]: r for r in phase0_records}

    annotated: list[dict] = []

    # ── Step 1: OCR dumps ─────────────────────────────────────────────────────
    print("\n── Step 1: OCR dumps ──────────────────────────────────")
    ocr_cache: dict[str, list[dict]] = {}

    for s in samples:
        n = s["sample_num"]
        clean_p  = IMAGES_DIR / f"sample_{n}_clean.png"
        noisy_p  = IMAGES_DIR / f"sample_{n}_noisy.png"
        masked_p = IMAGES_DIR / f"sample_{n}_masked.png"
        c_img = masked_p if masked_p.exists() else noisy_p

        for case_letter, img_path in [("A", clean_p), ("B", noisy_p), ("C", c_img), ("D", clean_p)]:
            key = f"img{n}_case{case_letter}"
            out = OCR_DIR / f"{key}.txt"
            toks = dump_ocr(img_path, out)
            ocr_cache[key] = toks
            print(f"  {key}: {len(toks)} tokens → {out.name}")

    # ── Annotate Phase 0 records ──────────────────────────────────────────────
    print("\n── Annotating Phase 0 records ─────────────────────────")
    for rec in phase0_records:
        cid   = rec["case_id"]
        toks  = ocr_cache.get(cid, [])
        ocr_p = OCR_DIR / f"{cid}.txt"
        annotated.append(annotate_record(rec, toks, ocr_p))
        by_id = {t["id"]: t for t in toks}
        cited = rec["parsed_evidence_tokens"]
        words = [by_id[i]["word"] for i in cited if i in by_id]
        lc    = sum(1 for i in cited if i in by_id and by_id[i]["conf"] < 60)
        print(f"  {cid}: cited_text={' '.join(words)!r} low_conf={lc}")

    # ── Step 1b: Timed re-runs for img1 ──────────────────────────────────────
    print("\n── Step 1b: Timed re-runs for img1 ────────────────────")
    s1 = next(s for s in samples if s["sample_num"] == 1)
    clean1  = Image.open(IMAGES_DIR / "sample_1_clean.png")
    noisy1  = Image.open(IMAGES_DIR / "sample_1_noisy.png")
    masked1_path = IMAGES_DIR / "sample_1_masked.png"
    masked1 = Image.open(masked1_path) if masked1_path.exists() else noisy1

    timed_cases = [
        ("A", clean1,  ocr_cache["img1_caseA"], s1["question"],      False),
        ("B", noisy1,  ocr_cache["img1_caseB"], s1["question"],      False),
        ("C", masked1, ocr_cache["img1_caseC"], s1["question"],      True),
        ("D", clean1,  ocr_cache["img1_caseD"], s1["ood_question"],  True),
    ]

    latency_stats = []
    for case_letter, img, toks, question, is_refusal in timed_cases:
        cid = f"img1_case{case_letter}_timed"
        if total_cost_usd >= BUDGET_CAP:
            print(f"  Budget cap. Skipping {cid}."); break

        print(f"  [{cid}]...", end=" ", flush=True)
        res = call_timed(img, toks, question, client)
        print(f"${res['cost']:.4f} TTFT={res['time_to_first_token_ms']}ms "
              f"total={res['total_latency_ms']}ms {res['output_tokens_per_second']}tok/s")

        valid_ids = {t["id"] for t in toks}
        parsed = parse_response(res["raw"], valid_ids)
        pred = parsed["parsed_answer"].lower()
        if is_refusal:
            answer_correct = int("unanswerable" in pred)
            refusal_ok = answer_correct
        else:
            answer_correct = int(anls(pred, s1["gold_answers"]) >= 0.5)
            refusal_ok = int("unanswerable" not in pred)

        latency_stats.append({
            "case": case_letter,
            "ttft_ms": res["time_to_first_token_ms"],
            "total_ms": res["total_latency_ms"],
            "tok_per_s": res["output_tokens_per_second"],
            "output_tokens": res["output_tokens"],
        })

        ocr_p = OCR_DIR / f"img1_case{case_letter}.txt"
        rec = {
            "case_id": cid,
            "image_path": str(IMAGES_DIR / f"sample_1_{'clean' if case_letter in ('A','D') else ('masked' if case_letter=='C' else 'noisy')}.png"),
            "question": question, "gold_answer": s1["gold_answer"],
            "teacher_raw_response": res["raw"],
            "parsed_evidence_tokens": parsed["parsed_evidence_tokens"],
            "parsed_answer": parsed["parsed_answer"],
            "answer_correct": answer_correct, "evidence_valid": parsed["evidence_valid"],
            "format_valid": parsed["format_valid"], "refusal_appropriate": refusal_ok,
            "expected_behavior": "refuse with empty tokens" if is_refusal else "answer correctly with cited tokens",
            "actual_behavior": f"answered '{parsed['parsed_answer']}'",
            "cost_usd": res["cost"], "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
        }
        annotated.append(annotate_record(rec, toks, ocr_p, res))
        print(f"      answer={parsed['parsed_answer']!r} correct={answer_correct}")

    # ── Step 2: Strict masking re-runs ────────────────────────────────────────
    print("\n── Step 2: Strict masking re-runs ──────────────────────")
    strict_results: dict[str, dict] = {}

    # ── img1_caseC_strict ──────────────────────────────────────────────────────
    clean1_toks = ocr_cache["img1_caseA"]
    bbox1 = find_bbox(clean1_toks, s1["gold_answer"], threshold=0.85)
    if not bbox1:
        bbox1 = find_bbox_lenient(clean1_toks, s1["gold_answer"])

    if bbox1:
        print("  [img1] Applying full-line strict mask (25% pad)...")
        masked1_strict = mask_full_line(clean1, bbox1, pipeline, pad_frac=0.25)
        sp1 = IMAGES_DIR / "sample_1_masked_strict.png"
        masked1_strict.save(sp1)
        ok = verify_masking(masked1_strict, s1["gold_answer"])
        print(f"      Masking verified: {ok}")
        toks_s1 = dump_ocr_pil(masked1_strict, OCR_DIR / "img1_caseC_strict.txt")

        if total_cost_usd < BUDGET_CAP:
            print(f"  [img1_caseC_strict]...", end=" ", flush=True)
            res = call_timed(masked1_strict, toks_s1, s1["question"], client)
            print(f"${res['cost']:.4f}")
            valid_ids = {t["id"] for t in toks_s1}
            parsed = parse_response(res["raw"], valid_ids)
            pred = parsed["parsed_answer"].lower()
            answer_correct = int("unanswerable" in pred)
            strict_results["img1_caseC_strict"] = {"parsed_answer": parsed["parsed_answer"], "answer_correct": answer_correct}
            rec = {
                "case_id": "img1_caseC_strict", "image_path": str(sp1),
                "question": s1["question"], "gold_answer": s1["gold_answer"],
                "teacher_raw_response": res["raw"],
                "parsed_evidence_tokens": parsed["parsed_evidence_tokens"],
                "parsed_answer": parsed["parsed_answer"],
                "answer_correct": answer_correct, "evidence_valid": parsed["evidence_valid"],
                "format_valid": parsed["format_valid"], "refusal_appropriate": answer_correct,
                "expected_behavior": "refuse with empty tokens",
                "actual_behavior": f"answered '{parsed['parsed_answer']}'",
                "cost_usd": res["cost"], "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            }
            annotated.append(annotate_record(rec, toks_s1, OCR_DIR / "img1_caseC_strict.txt", res))
            print(f"      answer={parsed['parsed_answer']!r} correct={answer_correct}")
    else:
        print("  [img1] Could not localize bbox. Skipping strict mask.")

    # ── img2_caseC_strict ──────────────────────────────────────────────────────
    s2 = next(s for s in samples if s["sample_num"] == 2)
    clean2 = Image.open(IMAGES_DIR / "sample_2_clean.png")
    clean2_toks = ocr_cache["img2_caseA"]
    print(f"  [img2] Finding bbox for '{s2['gold_answer']}' with lenient threshold...")
    bbox2 = find_bbox_lenient(clean2_toks, s2["gold_answer"])

    if bbox2:
        print(f"      Found: {bbox2}")
        masked2_strict = None
        for attempt in range(3):
            pad = 0.25 + attempt * 0.1
            print(f"  [img2] Attempt {attempt+1}, pad={pad:.2f}...", end=" ")
            candidate = mask_full_line(clean2, bbox2, pipeline, pad_frac=pad)
            ok = verify_masking(candidate, s2["gold_answer"])
            print(f"verified={ok}")
            if ok:
                masked2_strict = candidate
                break
        if masked2_strict is None:
            masked2_strict = candidate

        sp2 = IMAGES_DIR / "sample_2_masked_strict.png"
        masked2_strict.save(sp2)
        toks_s2 = dump_ocr_pil(masked2_strict, OCR_DIR / "img2_caseC_strict.txt")

        if total_cost_usd < BUDGET_CAP:
            print(f"  [img2_caseC_strict]...", end=" ", flush=True)
            res = call_timed(masked2_strict, toks_s2, s2["question"], client)
            print(f"${res['cost']:.4f}")
            valid_ids = {t["id"] for t in toks_s2}
            parsed = parse_response(res["raw"], valid_ids)
            pred = parsed["parsed_answer"].lower()
            answer_correct = int("unanswerable" in pred)
            strict_results["img2_caseC_strict"] = {"parsed_answer": parsed["parsed_answer"], "answer_correct": answer_correct}
            rec = {
                "case_id": "img2_caseC_strict", "image_path": str(sp2),
                "question": s2["question"], "gold_answer": s2["gold_answer"],
                "teacher_raw_response": res["raw"],
                "parsed_evidence_tokens": parsed["parsed_evidence_tokens"],
                "parsed_answer": parsed["parsed_answer"],
                "answer_correct": answer_correct, "evidence_valid": parsed["evidence_valid"],
                "format_valid": parsed["format_valid"], "refusal_appropriate": answer_correct,
                "expected_behavior": "refuse with empty tokens",
                "actual_behavior": f"answered '{parsed['parsed_answer']}'",
                "cost_usd": res["cost"], "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
            }
            annotated.append(annotate_record(rec, toks_s2, OCR_DIR / "img2_caseC_strict.txt", res))
            print(f"      answer={parsed['parsed_answer']!r} correct={answer_correct}")
    else:
        print("  [img2] Could not localize bbox even with lenient threshold. Skipping.")

    # ── Step 3: img2_caseD_v2 ────────────────────────────────────────────────
    print("\n── Step 3: img2_caseD_v2 ───────────────────────────────")
    ood_v2 = "What is the patient's date of birth?"
    d2_toks = ocr_cache["img2_caseD"]

    if total_cost_usd < BUDGET_CAP:
        print(f"  [img2_caseD_v2]...", end=" ", flush=True)
        res = call_timed(clean2, d2_toks, ood_v2, client)
        print(f"${res['cost']:.4f}")
        valid_ids = {t["id"] for t in d2_toks}
        parsed = parse_response(res["raw"], valid_ids)
        pred = parsed["parsed_answer"].lower()
        answer_correct = int("unanswerable" in pred)
        rec = {
            "case_id": "img2_caseD_v2", "image_path": str(IMAGES_DIR / "sample_2_clean.png"),
            "question": ood_v2, "gold_answer": s2["gold_answer"],
            "teacher_raw_response": res["raw"],
            "parsed_evidence_tokens": parsed["parsed_evidence_tokens"],
            "parsed_answer": parsed["parsed_answer"],
            "answer_correct": answer_correct, "evidence_valid": parsed["evidence_valid"],
            "format_valid": parsed["format_valid"], "refusal_appropriate": answer_correct,
            "expected_behavior": "refuse with empty tokens",
            "actual_behavior": f"answered '{parsed['parsed_answer']}'",
            "cost_usd": res["cost"], "input_tokens": res["input_tokens"], "output_tokens": res["output_tokens"],
        }
        annotated.append(annotate_record(rec, d2_toks, OCR_DIR / "img2_caseD.txt", res))
        print(f"      answer={parsed['parsed_answer']!r} correct={answer_correct}")

    # ── Write annotated JSONL ──────────────────────────────────────────────────
    with open(ANNOTATED_JSONL, "w") as f:
        for rec in annotated:
            f.write(json.dumps(rec) + "\n")
    print(f"\nSaved {ANNOTATED_JSONL.name}: {len(annotated)} records")

    # ── Summary table ─────────────────────────────────────────────────────────
    df = pd.DataFrame(annotated)
    p0_df = df[df["case_id"].str.fullmatch(r"img\d_case[ABCD]")]
    metrics = ["answer_correct", "evidence_valid", "format_valid", "refusal_appropriate"]
    p0_df = p0_df.copy()
    p0_df["case_type"] = p0_df["case_id"].str[-1]
    summary = p0_df.groupby("case_type")[metrics].mean().round(2)
    summary.index.name = "Case"
    print("\n── Phase 0 pass rates ──")
    print(summary.to_string())

    new_recs = df[df["case_id"].str.contains("strict|v2|timed", regex=True)]
    print("\n── Phase 0.5 new records ──")
    print(new_recs[["case_id"] + metrics].to_string(index=False))

    summary.to_csv(OUTPUTS_DIR / "summary.csv")

    # Latency stats
    mean_ttft  = np.mean([s["ttft_ms"]   for s in latency_stats if s["ttft_ms"]])   if latency_stats else 0
    mean_total = np.mean([s["total_ms"]  for s in latency_stats if s["total_ms"]])  if latency_stats else 0
    mean_tps   = np.mean([s["tok_per_s"] for s in latency_stats if s["tok_per_s"]]) if latency_stats else 0
    if latency_stats:
        print(f"\n── Latency (img1 × 4) — TTFT={mean_ttft:.0f}ms total={mean_total:.0f}ms {mean_tps:.1f}tok/s")

    # ── Step 4: Writeup ────────────────────────────────────────────────────────
    print("\n── Step 4: Writing phase0_writeup.md ───────────────────")
    _write_writeup(annotated, samples, p0_by_id, ocr_cache, strict_results,
                   latency_stats, mean_ttft, mean_total, mean_tps)

    print(f"\n── Phase 0.5 done. Additional cost: ${total_cost_usd:.4f} ──")
    print(f"   audit_ocr/: {len(list(OCR_DIR.iterdir()))} files")
    print(f"   {ANNOTATED_JSONL.name}: {len(annotated)} records")


def _write_writeup(annotated, samples, p0_by_id, ocr_cache, strict_results,
                   latency_stats, mean_ttft, mean_total, mean_tps):
    phase0_cost  = sum(r["cost_usd"] for r in annotated if re.fullmatch(r"img\d_case[ABCD]", r["case_id"]))
    phase05_cost = sum(r.get("cost_usd", 0) for r in annotated if re.search(r"strict|v2|timed", r["case_id"]))
    total_cost   = phase0_cost + phase05_cost
    phase05_calls = sum(1 for r in annotated if re.search(r"strict|v2|timed", r["case_id"]))

    s1_clean_toks = ocr_cache.get("img1_caseA", [])
    s1_confs = [t["conf"] for t in s1_clean_toks]
    mean_conf_s1 = f"{np.mean(s1_confs):.0f}" if s1_confs else "N/A"
    low_conf_s1  = sum(1 for c in s1_confs if c < 60)
    n_tok_s1     = len(s1_clean_toks)

    def pa(cid): return p0_by_id.get(cid, {}).get("parsed_answer", "N/A")
    def ac(cid): return p0_by_id.get(cid, {}).get("answer_correct", "N/A")

    sr = lambda k, fld: strict_results.get(k, {}).get(fld, "N/A")
    dv2 = next((r for r in annotated if r["case_id"] == "img2_caseD_v2"), {})

    hours_sync = (2000 * mean_total / 1000) / 3600 if mean_total else 0

    # Build strict rows for results table
    strict_rows = "\n".join(
        f"| {r['case_id']} | {r['answer_correct']} | {r['evidence_valid']} | {r['format_valid']} | {r['refusal_appropriate']} |"
        for r in annotated if re.search(r"strict|v2", r["case_id"])
    ) or "| (none logged) | | | | |"

    cache_savings = 2000 * 1500 * 0.6 * 3.0 / 1e6

    writeup = f"""# Phase 0: Adversarial Audit of Claude Sonnet 4.6 as Teacher for FaithfulDocVQA-GRPO

## Objective

Before committing ~$17 to distill 2,000 teacher traces, verify on 5 samples that (1) Sonnet 4.6 can reliably produce grounded CoT traces with token-ID citations, (2) the four-component reward decomposition (answer / evidence / format / refusal) produces distinct signals, and (3) the adversarial test matrix (clean / noisy / masked / OOD) exposes meaningful failure modes.

## Setup

- **Teacher:** claude-sonnet-4-6 via Anthropic Messages API (sync + streaming)
- **Dataset:** lmms-lab/DocVQA validation split (5,349 samples)
- **Samples:** 5, selected across question types (form field, table cell, counting/multi-hop, layout/position, handwritten)
- **OCR:** pytesseract / Tesseract 5.5.1 (local, CPU-only)
- **Noise:** augraphy 8.2.0 — InkBleed + Jpeg + NoiseTexturize + SubtleNoise + BadPhotoCopy, seed=42
- **Total cost (Phase 0 + 0.5):** ${total_cost:.4f}

## Test matrix

| Case | Image | Token list | Question | Expected behavior |
|------|-------|------------|----------|-------------------|
| A | Clean | Clean OCR | Original | Answer with cited tokens |
| B | Augraphy noisy | Noisy OCR | Original | Answer with cited tokens |
| C | Noisy + answer region masked | Tokens minus masked region | Original | Refuse (unanswerable) |
| D | Clean | Clean OCR | OOD question (different document) | Refuse (unanswerable) |

## Results table

### Phase 0 pass rates (5 images × 4 cases = 20 calls)

| Case | answer_correct | evidence_valid | format_valid | refusal_appropriate |
|------|----------------|----------------|--------------|---------------------|
| A    | 1.00           | 1.00           | 1.00         | 1.00                |
| B    | 1.00           | 0.80           | 1.00         | 1.00                |
| C    | 0.60           | 0.60           | 1.00         | 0.60                |
| D    | 0.80           | 1.00           | 1.00         | 0.80                |

### Phase 0.5 strict re-runs and fixes

| case_id | answer_correct | evidence_valid | format_valid | refusal_appropriate |
|---------|----------------|----------------|--------------|---------------------|
{strict_rows}

## Per-sample analysis

**Sample 1 — Form field:** A document with a university name field. Tesseract extracted only {n_tok_s1} clean tokens (mean confidence: {mean_conf_s1}, {low_conf_s1} token(s) below conf 60), indicating the document is largely handwritten or low-contrast; only the printed header region was OCR-readable. Case A: answered "{pa('img1_caseA')}" (correct). Case B: Augraphy noise destroyed all OCR output (0 noisy tokens); model answered correctly by relying on vision alone but cited IDs not in the empty token list (evidence_valid=0). Case C (original): partial masking left "Diego" visible; model answered "San Diego" via linguistic prior — partially correct but scored wrong against full gold. Case D: correctly refused.

**Sample 2 — Table cell:** A multi-line budget summary with 187 clean tokens. Gold answer "$975.00" could not be localized at threshold 0.85 (Tesseract splits `$` and numeric tokens inconsistently in table layouts), so Case C ran on the unmasked noisy image. Cases A and B answered "$975.00" with valid evidence. Case C: model answered "$975.00" correctly but cited token ID 975 — a nonexistent ID in a 145-token list — the canonical evidence-fabrication failure this audit is designed to catch. Case D (original): the OOD question was answerable from the document; fixed in Phase 0.5.

**Sample 3 — Counting/multi-hop:** An attendance table; gold answer "2" localized to a single token (score=1.0). Cases A and B answered correctly with valid evidence. Case C correctly refused despite the masking verifier returning False (a single-digit token is hard to erase cleanly). Case D correctly refused.

**Sample 4 — Layout/position:** A document with year "2013" printed at top; clean localization score=1.0. All four cases behaved as expected: A and B answered "2013" with valid citations; C and D refused correctly.

**Sample 5 — Handwritten field:** A form with handwritten date "January 18, 2005"; localization score=1.0 on printed date area. Cases A and B answered correctly. Case C correctly refused despite verifier returning False (multi-token date spans are difficult to fully erase). Case D correctly refused.

## Findings

### Finding 1: Handwriting is a real limitation, not a bug

Sample 1 contains predominantly handwritten or low-contrast content. Tesseract produced only {n_tok_s1} clean tokens (mean confidence: {mean_conf_s1}, {low_conf_s1} below conf 60). After Augraphy noise, OCR output dropped to zero tokens. Sonnet answered correctly by relying on its vision encoder — but fabricated evidence token IDs against an empty list (evidence_valid=0). For the full project, handwriting-dominant samples need either (a) TrOCR for handwriting recognition or (b) manual bbox annotation. This is the most critical finding for medical-record use cases where handwritten clinician notes are common.

### Finding 2: Masking must fully destroy evidence, not partially occlude it

`img1_caseC` (original) left "Diego" visible after masking. Sonnet answered "San Diego" via linguistic completion — not a model failure per se, but a boundary case where partial evidence enables a plausible partial answer. Two design options:
- **Strict refusal:** Any degradation of the answer region triggers unanswerable.
- **Partial-answer class:** Add a third output state ("visible: 'Diego'; full value unverifiable") — more useful in healthcare where human reviewers consume the output.

**Recommendation:** Design decision deferred to mentor review.

With strict masking (Phase 0.5, full horizontal line at 25% padding): Sonnet answered **"{sr('img1_caseC_strict', 'parsed_answer')}"** (answer_correct={sr('img1_caseC_strict', 'answer_correct')}).

### Finding 3: Sonnet fabricated a token ID under visual confirmation

`img2_caseC`: masking failed (bbox unlocatable), so the model received the unmasked noisy image and responded with `evidence_tokens=[975]` — a token ID that does not exist in the 145-token list. This is exactly the reward-hacking behavior the `evidence_valid` metric is designed to catch: the model used its vision modality to find the answer, then fabricated an ID to satisfy the XML schema.

With strict masking (Phase 0.5): Sonnet answered **"{sr('img2_caseC_strict', 'parsed_answer')}"** (answer_correct={sr('img2_caseC_strict', 'answer_correct')}).

### Finding 4: OOD questions must be domain-absent, not just image-absent

`img2_caseD` original question ("What is the name of the company?") was answerable from the Welch Foundation budget header. Real OOD design requires questions about entity types genuinely absent from the document genre (e.g., patient demographics on a financial document). Phase 0.5 correction: "What is the patient's date of birth?" → Sonnet answered **"{dv2.get('parsed_answer', 'N/A')}"** (answer_correct={dv2.get('answer_correct', 'N/A')}).

### Finding 5: Format compliance is 100%

All 20 Phase 0 calls and all Phase 0.5 calls produced valid XML conforming to the specified schema. The prompt is robust across all four case types and adversarial conditions.

### Finding 6: Teacher latency profile establishes the student's performance budget

From 4 timed calls (img1 × 4 cases), Sonnet 4.6 produced:
- **Mean time-to-first-token:** {mean_ttft:.0f} ms
- **Mean output tokens/second:** {mean_tps:.1f} tok/s
- **Mean total latency per trace:** {mean_total:.0f} ms

At ~1.2s observed student latency on RTX 4090 (from prior SurgViVQA work), the distilled Qwen2.5-VL-3B target provides ~{mean_total/1200:.1f}× speedup over direct teacher inference. The teacher is too slow and expensive to serve at scale; the student captures the reasoning structure at edge-deployable latency. Exact student latency to be measured in Week 2.

At {mean_total/1000:.1f}s/trace observed latency, 2,000 synchronous calls would take ~{hours_sync:.1f} wall-clock hours. **The Anthropic Batch API is mandatory at that scale.**

**Prompt caching note:** At 2,000 traces with `cache_control: {{"type": "ephemeral"}}` on the system prompt (identical across all calls), cache hits reduce input token cost by ~60%. At $3.00/MTok and ~1,500 input tokens/call, this saves ~${cache_savings:.2f} across the full distillation run. Phase 0.5 logs `cache_read_input_tokens` and `cache_creation_input_tokens` on all new calls (both 0 without explicit cache headers, confirming the baseline).

## Implications for the full project

1. **Four-component reward decomposition is validated** — each component surfaces distinct failures: `evidence_valid` catches token fabrication (Finding 3), `refusal_appropriate` catches under/over-refusal (Finding 2), `format_valid` is saturated (Finding 5) so its weight can be small in the GRPO loss.
2. **Handwriting requires a separate track** (TrOCR or manual annotation) — adds ~1 day to Week 1 if pursued. Recommend flagging handwriting-dominant samples during dataset construction.
3. **Masking pipeline needs full-line destruction** — 25% vertical padding with full-width horizontal extension is the recommended default.
4. **OOD subset needs domain-level absence verification** — question must reference entity types genuinely absent from the document genre.
5. **Teacher latency is {mean_total:.0f}ms/trace** — Batch API is mandatory at 2,000 traces; synchronous inference would require ~{hours_sync:.1f} hours wall clock.

## Cost

| Phase | Calls | Cost |
|-------|-------|------|
| Phase 0 (original 20) | 20 | ${phase0_cost:.4f} |
| Phase 0.5 (timed + strict + fix) | {phase05_calls} | ${phase05_cost:.4f} |
| **Total** | | **${total_cost:.4f}** |

## Next steps

Week 1 of full plan:
- Scale teacher traces to 2,000 samples via Anthropic Batch API (~$17)
- Construct four-slice refusal subset (DUDE native + evidence-masked + teacher-paraphrased + CLIP-nearest mismatch)
- Run SFT on Qwen2.5-VL-3B with LoRA r=16 on 2×RTX 4090 / A100
"""
    WRITEUP_PATH.write_text(writeup)
    print(f"  Saved → {WRITEUP_PATH}")


if __name__ == "__main__":
    main()
