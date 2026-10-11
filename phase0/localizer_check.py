# ═══ LEGACY — unused since the v3 contract. Kept for provenance, not maintained.
# Nothing imports this module; see phase0/README.md.
#!/usr/bin/env python3
"""LOCALIZER QUALITY CHECK (corrected invocation) — READ-ONLY measurement.

Runs the project's REAL gold localizer over a stratified sample of CLEAN DocVQA
validation images (excluding handwritten + Image/Photo) and reports:
  - localizer-miss rate (overall + per question_type)
  - fixable-formatting vs genuine-OCR-miss split
  - natural-abstain spread across question_type
  - image sizing for laptop-local work

OCR fidelity
------------
The scoring OCR path is `phase0.pipeline.extract_ocr_tokens` imported VERBATIM —
the exact `pytesseract.image_to_data(img, ...)` call the audit and capture study
used. The PIL image object is passed through unchanged (no mode conversion, no
re-encode, no stdin piping), so the token menu matches the rest of the project.

Run (from repo root, OUTSIDE any sandbox so pytesseract's temp-file path works):
    uv run python phase0/localizer_check.py
Options:
    --n 500            sample size (default 500)
    --probe-dpi 5      print DPI/resolution diagnostics for the first N images
    --set-dpi 300      EXPERIMENTAL: force Tesseract --dpi for scoring. This
                       DIVERGES from pipeline.py (which sets no DPI). Default OFF.
    --no-append        do not append results to PROJECT_STATUS.md

Writes: appends a results block to PROJECT_STATUS.md under
"LOCALIZER CHECK (corrected invocation, n=500)" and dumps per-row detail to
phase0/outputs/localizer_check_rows.json. Does NOT modify pipeline.py or any data.
"""
import argparse
import difflib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# --- REAL pipeline OCR + localizer, imported verbatim (no re-implementation) ---
from phase0.pipeline import extract_ocr_tokens            # exact audit/capture OCR call
from phase0.phase05 import find_bbox, find_bbox_lenient    # the real gold localizer

from datasets import load_dataset
import pytesseract

EXCLUDE = {"handwritten", "Image/Photo"}
PHASE0_DIR  = Path(__file__).resolve().parent
OUTPUTS_DIR = PHASE0_DIR / "outputs"
STATUS_MD   = _root / "PROJECT_STATUS.md"
CAPTURE_CACHE = OUTPUTS_DIR / "docvqa_capture_rates.jsonl"
ROWS_OUT    = OUTPUTS_DIR / "localizer_check_rows.json"

DATE_RE = re.compile(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{1,2}[/-]\d{1,2})", re.I)
CUR_RE  = re.compile(r"[$£€%]|\d[.,]\d")


# ── matching helpers (raw, unthresholded — for diagnosing WHY a match failed) ──

def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def best_window(tokens, gold):
    """Raw best sliding-window score + matched text (same algorithm as find_bbox,
    no threshold) so we can see what OCR had at the closest point."""
    if not tokens or not gold.strip():
        return 0.0, ""
    words = [t["word"] for t in tokens]
    n = len(gold.strip().split())
    bs, bm = 0.0, ""
    for wn in range(max(1, n - 1), n + 2):
        for i in range(len(words) - wn + 1):
            w = " ".join(words[i:i + wn])
            s = difflib.SequenceMatcher(None, w.lower(), gold.lower()).ratio()
            if s > bs:
                bs, bm = s, w
    return bs, bm


def norm_best(tokens, gold):
    """Best score under aggressive normalization (lowercase, alnum-only). Sizes how
    much a perfect currency/date normalizer COULD recover."""
    if not tokens or not gold.strip():
        return 0.0
    words = [t["word"] for t in tokens]
    g = norm(gold)
    if not g:
        return 0.0
    n = len(gold.strip().split())
    bs = 0.0
    for wn in range(max(1, n - 1), n + 3):
        for i in range(len(words) - wn + 1):
            w = norm(" ".join(words[i:i + wn]))
            if not w:
                continue
            s = difflib.SequenceMatcher(None, w, g).ratio()
            if s > bs:
                bs = s
    return bs


def cause(r):
    """Mechanism-first: if normalization can't recover it and OCR's best window is
    unrelated, it's a genuine miss; otherwise label by gold surface form."""
    if r["norm_best"] < 0.6 and r["raw_best"] < 0.45:
        return "genuine-OCR-miss"
    g = " ".join(r["golds"])
    if CUR_RE.search(g):
        return "currency/symbol(borderline)"
    if DATE_RE.search(g):
        return "date(borderline)"
    if max(len(x.split()) for x in r["golds"]) >= 4:
        return "multi-token-span"
    return "other-subthreshold"


# ── DPI diagnostic (does NOT touch the scoring path) ──────────────────────────

def _count_tsv_words(stdout_bytes):
    lines = stdout_bytes.decode("utf-8", "replace").splitlines()
    n = 0
    for ln in lines[1:]:
        f = ln.split("\t")
        if len(f) >= 12 and f[11].strip():
            try:
                if int(float(f[10])) >= 0:
                    n += 1
            except ValueError:
                pass
    return n


def probe_resolution(img):
    """Diagnostic only. Saves the image the way the pipeline's pytesseract would
    (PIL default, no dpi kwarg) and reports: image DPI metadata, the Tesseract
    'Estimating resolution as N' / 'Detected ...' stderr line, token count at the
    pipeline-default invocation, and token count when DPI is forced to 300. If the
    last two differ, DPI affects the menu. Uses a file path (works outside the
    sandbox); does not affect scoring."""
    dpi_meta = img.info.get("dpi")
    tf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    path = tf.name
    tf.close()
    try:
        img.save(path)  # PIL default, mirrors pipeline's no-DPI save
        r_def = subprocess.run(["tesseract", path, "stdout", "tsv"], capture_output=True)
        stderr = r_def.stderr.decode("utf-8", "replace")
        res_line = "; ".join(
            l.strip() for l in stderr.splitlines()
            if "resolution" in l.lower() or "Estimating" in l or "Detected" in l
        ) or "(no resolution line emitted)"
        n_def = _count_tsv_words(r_def.stdout)
        r_300 = subprocess.run(["tesseract", path, "stdout", "tsv", "--dpi", "300"],
                               capture_output=True)
        n_300 = _count_tsv_words(r_300.stdout)
    finally:
        os.unlink(path)
    return dpi_meta, res_line, n_def, n_300


def ocr_with_forced_dpi(img, dpi):
    """EXPERIMENTAL scoring OCR with forced --dpi. Mirrors extract_ocr_tokens'
    filtering exactly but adds `config='--dpi N'`. Use only with --set-dpi; this
    DIVERGES from pipeline.py and is not the default."""
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT,
                                     config=f"--dpi {dpi}")
    tokens, tid = [], 0
    for i, word in enumerate(data["text"]):
        if not word.strip() or int(data["conf"][i]) < 0:
            continue
        tokens.append({"id": tid, "word": word,
                       "x1": data["left"][i], "y1": data["top"][i],
                       "x2": data["left"][i] + data["width"][i],
                       "y2": data["top"][i] + data["height"][i]})
        tid += 1
    return tokens


# ── stratified sampling (same logic/shape as the n=100 run) ───────────────────

def stratified(eligible, qtypes_all, n_target):
    by_type = defaultdict(list)
    for i in eligible:
        key = qtypes_all[i][0] if qtypes_all[i] else "_none_"
        by_type[key].append(i)
    chosen = []
    for k, ixs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        quota = max(1, round(n_target * len(ixs) / len(eligible)))
        if len(ixs) > quota:
            step = len(ixs) / quota
            pick = [ixs[int(j * step)] for j in range(quota)]
        else:
            pick = ixs[:quota]
        chosen.extend(pick)
    return chosen[:n_target]


def unique_doc_count():
    if not CAPTURE_CACHE.exists():
        return None, None
    rows = [json.loads(l) for l in open(CAPTURE_CACHE)]
    docids = {r["docId"] for r in rows}
    return len(rows), len(docids)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--probe-dpi", type=int, default=5,
                    help="print DPI diagnostics for the first N scored images")
    ap.add_argument("--set-dpi", type=int, default=None,
                    help="EXPERIMENTAL: force Tesseract --dpi for scoring "
                         "(diverges from pipeline.py)")
    ap.add_argument("--no-append", action="store_true")
    args = ap.parse_args()

    if args.set_dpi:
        print(f"!! --set-dpi {args.set_dpi} set: scoring OCR DIVERGES from "
              f"pipeline.py (which sets no DPI). For a pipeline-faithful run, omit it.\n")

    print("Loading DocVQA validation ...", flush=True)
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    qtypes_all = ds["question_types"]
    N = len(ds)
    print(f"  {N} rows", flush=True)

    eligible = [i for i in range(N) if not (set(qtypes_all[i]) & EXCLUDE)]
    print(f"  eligible (excl handwritten+Image/Photo): {len(eligible)}", flush=True)

    chosen = stratified(eligible, qtypes_all, args.n)
    print(f"  selected {len(chosen)}; per first-type: "
          f"{dict(Counter(qtypes_all[i][0] for i in chosen))}\n", flush=True)

    # DPI diagnostic on the first --probe-dpi images
    if args.probe_dpi:
        print(f"── DPI / resolution diagnostic (first {args.probe_dpi} images) ──")
        print("   (scoring uses pipeline default = NO forced DPI; --dpi 300 column "
              "shows whether DPI would change the menu)")
        for idx in chosen[:args.probe_dpi]:
            img = ds[idx]["image"]
            dpi_meta, res_line, n_def, n_300 = probe_resolution(img)
            n_score = len(extract_ocr_tokens(img))
            flag = "" if n_def == n_300 else "  <-- DPI CHANGES MENU"
            print(f"  idx={idx} size={img.size} mode={img.mode} "
                  f"img.info['dpi']={dpi_meta}")
            print(f"      tesseract: {res_line}")
            print(f"      tokens: pipeline-call={n_score} file-default={n_def} "
                  f"forced-dpi300={n_300}{flag}")
        print()

    ocr = (lambda im: ocr_with_forced_dpi(im, args.set_dpi)) if args.set_dpi \
        else extract_ocr_tokens

    rows, img_bytes, img_dims, skipped = [], [], [], 0
    for k, idx in enumerate(chosen):
        row = ds[idx]
        img = row["image"]
        buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
        img_bytes.append(len(buf.getvalue())); img_dims.append(img.size)
        golds = list(row["answers"])
        try:
            toks = ocr(img)                       # <-- pipeline-identical OCR
        except Exception as e:
            print(f"  [skip idx={idx}] OCR error: {type(e).__name__}: {str(e)[:80]}",
                  flush=True)
            skipped += 1
            continue

        matched_ids, loc_score, loc_gold, loc_win = None, 0.0, "", ""
        for g in golds:
            b = find_bbox(toks, g, 0.85) or find_bbox_lenient(toks, g)
            bs, bm = best_window(toks, g)
            if b and matched_ids is None:
                matched_ids = b["token_ids"]
            if bs > loc_score:
                loc_score, loc_gold, loc_win = bs, g, bm
        localized = matched_ids is not None
        g0 = golds[0] if golds else ""
        loc_firstgold = (find_bbox(toks, g0, 0.85) or find_bbox_lenient(toks, g0)) is not None
        nb = max((norm_best(toks, g) for g in golds), default=0.0)

        rows.append({
            "idx": idx, "qid": str(row["questionId"]), "docId": row["docId"],
            "qtype": list(qtypes_all[idx]), "qtype1": qtypes_all[idx][0],
            "question": row["question"], "golds": golds,
            "n_tokens": len(toks), "localized": localized,
            "loc_firstgold": loc_firstgold,
            "raw_best": round(loc_score, 3), "norm_best": round(nb, 3),
            "best_gold": loc_gold, "best_window": loc_win,
        })
        if (k + 1) % 50 == 0:
            print(f"  [{k+1}/{len(chosen)}] localized so far: "
                  f"{sum(r['localized'] for r in rows)}", flush=True)

    # ── analysis ──
    n_scored = len(rows)
    loc = [r for r in rows if r["localized"]]
    fail = [r for r in rows if not r["localized"]]
    for r in fail:
        r["cause"] = cause(r)
    fixable = [r for r in fail if r["norm_best"] >= 0.85]
    genuine = [r for r in fail if r["norm_best"] < 0.85]
    fg = sum(1 for r in rows if r["loc_firstgold"])

    bt = defaultdict(lambda: [0, 0])
    for r in rows:
        bt[r["qtype1"]][1] += 1
        if r["localized"]:
            bt[r["qtype1"]][0] += 1

    miss_rate = 100 * len(fail) / n_scored if n_scored else 0.0
    lo, hi = wilson(len(fail), n_scored)

    out = []
    p = out.append
    p("=" * 70)
    p(f"RESULT: {len(loc)}/{n_scored} localized, {len(fail)} FAILED "
      f"(miss rate {miss_rate:.1f}%, 95% CI {lo:.1f}-{hi:.1f}%); skipped={skipped}")
    p("=" * 70)
    p(f"OCR invocation: {'extract_ocr_tokens (pipeline-identical)' if not args.set_dpi else f'forced --dpi {args.set_dpi} (DIVERGES from pipeline)'}")
    p("\nLocalization by first question_type:")
    for t, (a, b) in sorted(bt.items(), key=lambda kv: -kv[1][1]):
        p(f"  {t:<15} {a}/{b}  ({100*a/b:.0f}%)")
    p(f"\nFirst-gold-only (as-coded localize_gold) = {fg}/{n_scored}; "
      f"all-gold-variants = {len(loc)}/{n_scored}; "
      f"gap = {len(loc)-fg} recovered only via alternate gold spellings")
    p("\nFailure causes (mechanism-first):")
    for c, n in Counter(r["cause"] for r in fail).most_common():
        p(f"  {c:<28} {n}")
    p(f"\nFixable formatting (norm match >=0.85, text present): {len(fixable)}")
    p(f"Genuine OCR miss (norm match <0.85, true abstain):     {len(genuine)}")
    p("\nGenuine-miss spread by first question_type:")
    for t, n in Counter(r["qtype1"] for r in genuine).most_common():
        p(f"  {t:<15} {n}")
    p("\nFailure examples (up to 15):")
    for r in fail[:15]:
        p(f"  [{r['cause']}] qid={r['qid']} type={r['qtype']}")
        p(f"      Q: {r['question'][:80]}")
        p(f"      gold: {r['golds']}")
        p(f"      OCR best window: {r['best_window']!r}  raw={r['raw_best']} "
          f"norm={r['norm_best']}  n_tok={r['n_tokens']}")

    # sizing
    import statistics
    med_kb = statistics.median(img_bytes) / 1024
    n_qa, n_docs = unique_doc_count()
    p("\n--- IMAGE SIZING ---")
    p(f"  PNG bytes: median={med_kb:.0f}KB mean={statistics.mean(img_bytes)/1024:.0f}KB "
      f"min={min(img_bytes)/1024:.0f}KB max={max(img_bytes)/1024:.0f}KB")
    p(f"  dims: w median={statistics.median(d[0] for d in img_dims):.0f} "
      f"h median={statistics.median(d[1] for d in img_dims):.0f}")
    if n_docs:
        p(f"  validation: {n_qa} QA rows over {n_docs} unique docId "
          f"({n_qa/n_docs:.2f} Q/doc)")
        p(f"  download est: unique images ~{n_docs*med_kb/1024:.0f}MB ; "
          f"per-QA-row ~{n_qa*med_kb/1024:.0f}MB")

    report = "\n".join(out)
    print("\n" + report)

    ROWS_OUT.write_text(json.dumps(rows, indent=2))
    print(f"\n(wrote {ROWS_OUT})")

    if not args.no_append:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        invocation = ("extract_ocr_tokens (pipeline-identical, no DPI override)"
                      if not args.set_dpi else f"forced --dpi {args.set_dpi} (DIVERGES)")
        block = (
            f"\n\n---\n\n## LOCALIZER CHECK (corrected invocation, n={args.n})\n\n"
            f"_Run {stamp} by `phase0/localizer_check.py` outside the sandbox. "
            f"OCR = {invocation}. Kept separate from the earlier n=100 run for comparison._\n\n"
            "```\n" + report + "\n```\n"
        )
        with open(STATUS_MD, "a") as f:
            f.write(block)
        print(f"(appended results to {STATUS_MD} under "
              f"'LOCALIZER CHECK (corrected invocation, n={args.n})')")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return 100 * (c - m) / d, 100 * (c + m) / d


if __name__ == "__main__":
    main()
