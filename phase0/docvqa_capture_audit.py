# ═══ LEGACY — unused since the v3 contract. Kept for provenance, not maintained.
# Nothing imports this module; see phase0/README.md.
#!/usr/bin/env python3
"""Descriptive study: question-type distribution + Tesseract answer-capture rate on
DocVQA validation (5,349 samples).

Probes runtime budget; if estimated full run > BUDGET_MIN, falls back to a
question-type-stratified 2,000-sample subset and notes it in the summary.

Caches per-sample results to phase0/outputs/docvqa_capture_rates.jsonl
(re-runnable, incremental — already-scored questionIds are skipped).

Run: uv run python phase0/docvqa_capture_audit.py
"""

import argparse
import difflib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytesseract
from datasets import load_dataset
from PIL import Image

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

PHASE0_DIR  = Path(__file__).parent
OUTPUTS_DIR = PHASE0_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

CACHE_PATH    = OUTPUTS_DIR / "docvqa_capture_rates.jsonl"
SUMMARY_PATH  = OUTPUTS_DIR / "docvqa_descriptive_study.md"
CHART_QTYPE   = OUTPUTS_DIR / "docvqa_qtype_distribution.png"
CHART_CAPTURE = OUTPUTS_DIR / "docvqa_capture_by_qtype.png"

BUDGET_MIN  = 90.0
SUBSET_SIZE = 2000
PROBE_N     = 6

LOC_THRESH  = 0.85
NEAR_THRESH = 0.60


def extract_tokens(img: Image.Image) -> list[dict]:
    """Mirror of phase0/pipeline.extract_ocr_tokens — keep semantics identical."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    tokens = []
    tid = 0
    for i, word in enumerate(data["text"]):
        if not word.strip() or int(data["conf"][i]) < 0:
            continue
        tokens.append({
            "id": tid, "word": word,
            "x1": data["left"][i], "y1": data["top"][i],
            "x2": data["left"][i] + data["width"][i],
            "y2": data["top"][i]  + data["height"][i],
        })
        tid += 1
    return tokens


def best_match_score(tokens: list[dict], gold: str) -> tuple[float, str]:
    """Sliding-window fuzzy match — same algorithm as pipeline.find_answer_bbox,
    but returns the raw (best_score, best_match_text) without thresholding so we
    can bucket below 0.85."""
    if not tokens or not gold.strip():
        return 0.0, ""
    words = [t["word"] for t in tokens]
    n = len(gold.strip().split())
    best_score, best_match = 0.0, ""
    for wn in range(max(1, n - 1), n + 2):
        for i in range(len(words) - wn + 1):
            window = " ".join(words[i:i + wn])
            score = difflib.SequenceMatcher(None, window.lower(), gold.lower()).ratio()
            if score > best_score:
                best_score, best_match = score, window
    return best_score, best_match


def bucket_of(score: float) -> str:
    if score >= LOC_THRESH:  return "localized"
    if score >= NEAR_THRESH: return "near"
    return "fail"


def load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    cached = {}
    for line in CACHE_PATH.read_text().splitlines():
        if not line.strip(): continue
        rec = json.loads(line)
        cached[str(rec["questionId"])] = rec
    return cached


def score_sample(row) -> dict:
    tokens = extract_tokens(row["image"])
    best_score, best_match, best_gold = 0.0, "", ""
    for ga in row["answers"]:
        s, m = best_match_score(tokens, ga)
        if s > best_score:
            best_score, best_match, best_gold = s, m, ga
    return {
        "questionId":      str(row["questionId"]),
        "docId":           row["docId"],
        "question_types":  list(row["question_types"]),
        "n_tokens":        len(tokens),
        "n_answers":       len(row["answers"]),
        "best_score":      round(best_score, 4),
        "best_match":      best_match,
        "best_gold":       best_gold,
    }


def build_stratified_subset(ds, n: int) -> list[int]:
    """Pick `n` indices stratified by FIRST question_type (most are single-typed).
    Proportional allocation per stratum, deterministic ordering."""
    buckets = defaultdict(list)
    for i, qts in enumerate(ds["question_types"]):
        key = qts[0] if qts else "_none_"
        buckets[key].append(i)
    total = len(ds)
    chosen = []
    for k, ixs in buckets.items():
        quota = max(1, round(n * len(ixs) / total))
        chosen.extend(ixs[:quota])
    return chosen[:n]


def run_processing(ds, indices, cached) -> tuple[int, float]:
    n_done = 0
    t_start = time.time()
    out = open(CACHE_PATH, "a")
    try:
        for k, idx in enumerate(indices):
            row = ds[idx]
            qid = str(row["questionId"])
            if qid in cached:
                continue
            try:
                rec = score_sample(row)
                out.write(json.dumps(rec) + "\n")
                out.flush()
                n_done += 1
                if n_done % 100 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / elapsed if elapsed > 0 else 0
                    remaining = len(indices) - k - 1
                    eta = remaining / rate / 60 if rate else 0
                    print(f"  [{k+1:>5}/{len(indices)}] new={n_done} {rate:.2f} samp/s  "
                          f"ETA {eta:.1f} min", flush=True)
            except Exception as e:
                print(f"  ERROR qid={qid}: {type(e).__name__}: {e}", flush=True)
    finally:
        out.close()
    return n_done, time.time() - t_start


def analyze_and_report(subset_mode: bool, requested_n: int):
    recs = []
    for line in CACHE_PATH.read_text().splitlines():
        if line.strip():
            recs.append(json.loads(line))
    n_total = len(recs)

    # Question-type distribution
    qtype_counts = Counter()
    multi = 0
    for r in recs:
        qts = r["question_types"]
        if len(qts) > 1: multi += 1
        for qt in qts:
            qtype_counts[qt] += 1
    pct_multi = 100 * multi / n_total

    # Overall capture buckets
    overall = Counter()
    for r in recs:
        overall[bucket_of(r["best_score"])] += 1
    total_b = sum(overall.values())

    # Per-question_type capture
    per_qt = defaultdict(Counter)
    for r in recs:
        b = bucket_of(r["best_score"])
        for qt in r["question_types"]:
            per_qt[qt][b] += 1

    # === Charts ===
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qts_sorted = sorted(qtype_counts.items(), key=lambda x: -x[1])
    labels = [k for k, _ in qts_sorted]
    pcts = [100 * v / n_total for v in (v for _, v in qts_sorted)]

    fig, ax = plt.subplots(figsize=(9.5, 5))
    bars = ax.bar(labels, pcts, color="#3a6fa3")
    ax.set_ylabel("% of records (multi-label)")
    ax.set_title(f"Question-type distribution — DocVQA validation (n={n_total})")
    ax.set_ylim(0, max(pcts) * 1.18)
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{p:.1f}%",
                ha="center", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_QTYPE, dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    loc_p, near_p, fail_p = [], [], []
    for qt in labels:
        t = sum(per_qt[qt].values()) or 1
        loc_p.append(100 * per_qt[qt]["localized"] / t)
        near_p.append(100 * per_qt[qt]["near"]      / t)
        fail_p.append(100 * per_qt[qt]["fail"]      / t)
    base_near = loc_p
    base_fail = [a + b for a, b in zip(loc_p, near_p)]
    ax.bar(labels, loc_p,  color="#3a8a4a", label=f"localized (≥{LOC_THRESH:.2f})")
    ax.bar(labels, near_p, bottom=base_near, color="#d8a23d",
           label=f"near-miss ({NEAR_THRESH:.2f}–{LOC_THRESH:.2f})")
    ax.bar(labels, fail_p, bottom=base_fail, color="#c83434",
           label=f"fail (<{NEAR_THRESH:.2f})")
    ax.set_ylabel("% of records in this question_type")
    ax.set_title("OCR answer-capture rate by native question_type")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(CHART_CAPTURE, dpi=120)
    plt.close(fig)

    # === Markdown ===
    lines = []
    lines.append("# DocVQA validation — Tesseract answer-capture descriptive study")
    lines.append("")
    if subset_mode:
        lines.append(f"- **Subset mode:** runtime probe estimated full validation > {BUDGET_MIN:.0f} min; "
                     f"fell back to a question_type-stratified subset of **{n_total}** samples "
                     f"(target {requested_n}).")
    else:
        lines.append(f"- Full DocVQA validation split, **{n_total} / 5,349 samples** scored.")
    lines.append("- Pipeline per sample: pytesseract → token list → "
                 "sliding-window fuzzy match (`pipeline.find_answer_bbox` algorithm) "
                 "against every gold answer; best score wins.")
    lines.append(f"- Buckets: `localized` ≥ {LOC_THRESH}, `near-miss` {NEAR_THRESH}–{LOC_THRESH}, "
                 f"`fail` < {NEAR_THRESH}.")
    lines.append(f"- Cache: `{CACHE_PATH.name}` (incremental, re-runnable).")
    lines.append("")
    lines.append("## 1. Question-type distribution (multi-label, 9 native types)")
    lines.append("")
    lines.append("| question_type | count | % of records |")
    lines.append("|---------------|------:|------:|")
    for k, v in qts_sorted:
        lines.append(f"| `{k}` | {v} | {100*v/n_total:.1f}% |")
    lines.append(f"| **multi-typed records** | **{multi}** | **{pct_multi:.1f}%** |")
    lines.append("")
    lines.append(f"Bar chart: `{CHART_QTYPE.name}`.")
    lines.append("")
    lines.append("## 2. OCR answer-capture rate")
    lines.append("")
    lines.append("### Overall")
    lines.append("")
    lines.append("| bucket | count | % |")
    lines.append("|--------|------:|--:|")
    for b, label in [("localized", f"≥ {LOC_THRESH}"),
                     ("near",      f"{NEAR_THRESH}–{LOC_THRESH}"),
                     ("fail",      f"< {NEAR_THRESH}")]:
        c = overall[b]
        lines.append(f"| {b} ({label}) | {c} | {100*c/total_b:.1f}% |")
    lines.append("")
    lines.append("### By native question_type (multi-label, percentages within each type)")
    lines.append("")
    lines.append("| question_type | n | localized | near-miss | fail |")
    lines.append("|---------------|---:|------:|------:|----:|")
    for qt, _ in qts_sorted:
        t = sum(per_qt[qt].values()) or 1
        lines.append(f"| `{qt}` | {t} | "
                     f"{100*per_qt[qt]['localized']/t:.1f}% | "
                     f"{100*per_qt[qt]['near']/t:.1f}% | "
                     f"{100*per_qt[qt]['fail']/t:.1f}% |")
    lines.append("")
    lines.append(f"Stacked-bar chart: `{CHART_CAPTURE.name}`.")
    lines.append("")
    lines.append("## Reading the numbers")
    lines.append("")
    loc_pct  = 100 * overall["localized"] / total_b
    near_pct = 100 * overall["near"]      / total_b
    fail_pct = 100 * overall["fail"]      / total_b
    lines.append(f"- Overall localization rate is **{loc_pct:.1f}%** at the ≥0.85 threshold used by "
                 f"`find_bbox_lenient`'s first pass. **{near_pct:.1f}%** sit in the near-miss band "
                 f"(0.60–0.85) — fuzzy lookups may pick up some of these via lenient cascading. "
                 f"**{fail_pct:.1f}%** are unrecoverable at any reasonable text-match threshold.")
    lines.append("- For records in `localized` and partly `near-miss`, the model has a real gold-citation "
                 "to learn against. For records in `fail`, the new piecewise reward routes them through "
                 "the answerable-with-no-gold branch (F1 = 0.0) — that's the OCR-pipeline ceiling, "
                 "not a model defect.")
    lines.append("- The `ABSTAIN_CREDIT = +0.4` weight is calibrated for this distribution: if the "
                 "model cannot locate evidence on ~{:.0f}% of records (fail bucket), honest abstention "
                 "needs to be rewarded enough to dominate fabrication on that slice.".format(fail_pct))
    SUMMARY_PATH.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-subset", action="store_true",
                    help="Skip probe; go straight to stratified 2k subset.")
    ap.add_argument("--budget-min", type=float, default=BUDGET_MIN)
    args = ap.parse_args()

    print("Loading DocVQA validation split ...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    print(f"  {len(ds)} samples", flush=True)

    cached = load_cache()
    print(f"Cached: {len(cached)} already scored", flush=True)

    subset_mode = False
    requested_n = len(ds)

    if args.force_subset:
        subset_mode = True
        requested_n = SUBSET_SIZE
    else:
        # Probe
        n_uncached = len(ds) - len(cached)
        if n_uncached >= PROBE_N:
            print(f"\nProbing OCR rate on {PROBE_N} uncached samples ...", flush=True)
            t0 = time.time()
            seen = 0
            for i in range(len(ds)):
                if seen >= PROBE_N: break
                if str(ds[i]["questionId"]) in cached: continue
                extract_tokens(ds[i]["image"])
                seen += 1
            elapsed = time.time() - t0
            per = elapsed / max(seen, 1)
            eta = per * n_uncached / 60
            print(f"  {per:.2f} s/sample → ETA {eta:.1f} min for {n_uncached} samples", flush=True)
            if eta > args.budget_min:
                print(f"  Exceeds budget ({args.budget_min:.0f} min). "
                      f"Falling back to stratified {SUBSET_SIZE}-sample subset.", flush=True)
                subset_mode = True
                requested_n = SUBSET_SIZE

    if subset_mode:
        indices = build_stratified_subset(ds, SUBSET_SIZE)
    else:
        indices = list(range(len(ds)))

    n_to_run = sum(1 for i in indices if str(ds[i]["questionId"]) not in cached)
    print(f"\nProcessing: {len(indices)} target samples, {n_to_run} new (skipping cached).", flush=True)

    n_done, elapsed = run_processing(ds, indices, cached)
    print(f"\nProcessed {n_done} new samples in {elapsed/60:.1f} min.", flush=True)

    print("\nAnalyzing + writing summary + charts ...", flush=True)
    analyze_and_report(subset_mode, requested_n)
    print(f"  → {SUMMARY_PATH}")
    print(f"  → {CHART_QTYPE}")
    print(f"  → {CHART_CAPTURE}")


if __name__ == "__main__":
    main()
