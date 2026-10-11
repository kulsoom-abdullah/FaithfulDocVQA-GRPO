#!/usr/bin/env python3
"""Build the frozen v1 master dataset for FaithfulDocVQA.

This is a BUILD script. It does NOT modify pipeline.py / phase05.py / any phase0
code — it IMPORTS the real OCR + localizer so the dataset matches the audit and
localizer studies bit-for-bit.

Imported (no reimplementation) — quoted in the manifest too:
  - phase0.pipeline.extract_ocr_tokens        REAL Tesseract OCR -> token menu
  - phase0.phase05.find_bbox                  fuzzy sliding-window localizer core
  - phase0.phase05.find_bbox_lenient          cascade 0.85/0.7/0.6/0.5 + numeric fallback
  - phase0.score_phase0_with_f1.localize_gold single-gold entry point (used for the
        first-gold-vs-all-variants recovery metric); the build extends localization
        over ALL gold variants by calling the same find_bbox/find_bbox_lenient that
        localize_gold itself delegates to.

Source : lmms-lab/DocVQA, config "DocVQA", validation split.
Exclude: doc_types {handwritten, Image/Photo} (scope §1 / REWARD_DESIGN.md:69-71).
Images : CLEAN only — no augraphy, no masking (v1 clean-image build).
Policy : build to CURRENT localizer/OCR behavior (imperfect-but-frozen, scope §6).
         The localizer/OCR are NOT tuned or "fixed".

Outputs (written only if every self-check assertion passes):
  data/master.jsonl
  data/build_manifest.json
  data/images/<doc_id>.png                (materialized UNIQUE by doc_id)
  docs/findings/dataset_composition.md
"""

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

# ── REAL pipeline OCR + localizer, imported verbatim ──────────────────────────
from phase0.pipeline import extract_ocr_tokens               # Tesseract OCR menu
from phase0.phase05 import find_bbox, find_bbox_lenient       # localizer primitives
from phase0.score_phase0_with_f1 import localize_gold         # single-gold entry pt

# ── config ────────────────────────────────────────────────────────────────────
SEED          = 42
# Full build: ALL eligible records (no subsampling). 3-way split by docId,
# stratified by doc_type: ~70% train / ~15% validation / ~15% test. validation and
# test are FULL and SHARED (identical for SFT-A and SFT-B); the A=answerable-only
# filter applies to TRAIN only (done at load time, not here).
VAL_FRAC      = 0.15
TEST_FRAC     = 0.15
EXCLUDE       = {"handwritten", "Image/Photo"}
SOURCE_NAME   = "lmms-lab/DocVQA"
SOURCE_CONFIG = "DocVQA"
SOURCE_SPLIT  = "validation"

DATA        = ROOT / "data"
IMAGES      = DATA / "images"
FINDINGS    = ROOT / "docs" / "findings"
MASTER      = DATA / "master.jsonl"
MANIFEST    = DATA / "build_manifest.json"
COMPOSITION = FINDINGS / "dataset_composition.md"
LOG         = DATA / "build.log"

IMPORTED_FUNCTIONS = [
    "phase0.pipeline.extract_ocr_tokens",
    "phase0.phase05.find_bbox",
    "phase0.phase05.find_bbox_lenient",
    "phase0.score_phase0_with_f1.localize_gold",
]

_logf = None


def log(msg=""):
    print(msg, flush=True)
    if _logf:
        _logf.write(str(msg) + "\n")
        _logf.flush()


# ── helpers ───────────────────────────────────────────────────────────────────

def first_type(qtypes):
    return qtypes[0] if qtypes else "_none_"


def menu_from_tokens(tokens):
    """OCR token dicts -> citation menu rows {id, text, bbox:[x1,y1,x2,y2]}."""
    return [
        {"id": t["id"], "text": t["word"],
         "bbox": [t["x1"], t["y1"], t["x2"], t["y2"]]}
        for t in tokens
    ]


def menu_hash(menu):
    """sha256 over the canonicalized menu (ids + text + bbox). Stable, non-empty
    even for an empty menu ('[]')."""
    canon = json.dumps(
        [[m["id"], m["text"], *m["bbox"]] for m in menu],
        separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def localize_all_variants(tokens, golds):
    """Run the imported localizer over ALL gold variants.

    For each variant: find_bbox(0.85) or find_bbox_lenient(...) — exactly what
    localize_gold does for a single gold. cited_ids come from the FIRST variant
    that matches. localizer_score is the bbox score that produced those ids; if
    nothing matched it is the best sub-threshold window score attempted across
    variants (find_bbox at threshold 0.0 = best window, no filtering — the same
    imported function, not a reimplementation).

    Returns (localized: bool, cited_ids: sorted list[int], localizer_score: float).
    """
    cited, match_score = [], None
    best_attempt = 0.0
    for g in golds:
        b = find_bbox(tokens, g, 0.85) or find_bbox_lenient(tokens, g)
        if b and not cited:
            cited = sorted(b["token_ids"])
            match_score = float(b["score"])
        probe = find_bbox(tokens, g, 0.0)   # best window regardless of threshold
        if probe and probe["score"] > best_attempt:
            best_attempt = float(probe["score"])
    localized = bool(cited)
    score = match_score if localized else best_attempt
    return localized, cited, round(float(score), 4)


def xtab(records, key_first_type=True):
    """answerable / abstain / total counts per first question_type, sorted by total."""
    rows = defaultdict(lambda: [0, 0])  # ft -> [answerable, abstain]
    for r in records:
        ft = first_type(r["doc_type"])
        rows[ft][1 if r["target_is_abstain"] else 0] += 1
    ordered = sorted(rows.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
    return ordered


def render_xtab(title, records):
    ordered = xtab(records)
    lines = [f"### {title} (n={len(records)})", ""]
    lines.append("| doc_type (first) | answerable | abstain (ocr_miss) | total | abstain % |")
    lines.append("|---|---:|---:|---:|---:|")
    tot_ans = tot_abs = 0
    for ft, (a, b) in ordered:
        t = a + b
        tot_ans += a
        tot_abs += b
        pct = f"{100*b/t:.1f}%" if t else "—"
        lines.append(f"| {ft} | {a} | {b} | {t} | {pct} |")
    T = tot_ans + tot_abs
    tpct = f"{100*tot_abs/T:.1f}%" if T else "—"
    lines.append(f"| **TOTAL** | **{tot_ans}** | **{tot_abs}** | **{T}** | **{tpct}** |")
    lines.append("")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global _logf
    DATA.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    FINDINGS.mkdir(parents=True, exist_ok=True)
    _logf = open(LOG, "w")

    rng = random.Random(SEED)
    t_start = datetime.now()
    log(f"== FaithfulDocVQA v1 master build (full eligible, 3-way 70/15/15) ==  seed={SEED}")
    log(f"start {t_start.isoformat(timespec='seconds')}")

    # ── load source (column access only; no image decode here) ──
    log(f"\nLoading {SOURCE_NAME} [{SOURCE_CONFIG}] split={SOURCE_SPLIT} ...")
    ds = load_dataset(SOURCE_NAME, SOURCE_CONFIG, split=SOURCE_SPLIT)
    qtypes_all  = ds["question_types"]
    docids_all  = ds["docId"]
    qids_all    = ds["questionId"]
    quest_all   = ds["question"]
    answers_all = ds["answers"]
    N = len(ds)
    log(f"  {N} rows")

    # ── eligibility: drop excluded types, require non-empty gold ──
    elig = []
    for i in range(N):
        if set(qtypes_all[i] or []) & EXCLUDE:
            continue
        golds = [str(g) for g in (answers_all[i] or [])]
        if not golds or not any(g.strip() for g in golds):
            continue
        elig.append(i)
    elig_docs = {docids_all[i] for i in elig}
    log(f"  eligible records: {len(elig)} of {N}  over {len(elig_docs)} docs "
        f"(excluded types: {sorted(EXCLUDE)})")

    # ── FULL build: use ALL eligible records (no subsampling) ──
    selected = sorted(elig)
    ft_hist = Counter(first_type(qtypes_all[i]) for i in selected)
    log(f"  selected ALL {len(selected)} eligible records (full build); "
        f"first-type histogram: {dict(ft_hist.most_common())}")

    # unique docs among selected (image materialization key)
    sel_docs, seen, doc_repr_idx = [], set(), {}
    for i in selected:
        d = docids_all[i]
        if d not in seen:
            seen.add(d)
            sel_docs.append(d)
        doc_repr_idx.setdefault(d, i)
    log(f"  unique docs among selected: {len(sel_docs)} "
        f"(images materialized unique by doc_id)")

    # ── OCR each unique doc once; materialize its image ──
    log(f"\nOCR + image materialization over {len(sel_docs)} docs ...")
    doc_menu, doc_hash, doc_tokens = {}, {}, {}
    img_bytes_total = 0
    for k, d in enumerate(sel_docs):
        idx = doc_repr_idx[d]
        img = ds[idx]["image"]
        tokens = extract_ocr_tokens(img)          # REAL pipeline OCR
        doc_tokens[d] = tokens
        menu = menu_from_tokens(tokens)
        doc_menu[d] = menu
        doc_hash[d] = menu_hash(menu)
        path = IMAGES / f"{d}.png"
        try:
            img.save(path)
        except Exception:
            img.convert("RGB").save(path)
        img_bytes_total += path.stat().st_size
        if (k + 1) % 100 == 0 or (k + 1) == len(sel_docs):
            log(f"  [{k+1}/{len(sel_docs)}] docs OCR'd")

    # ── build records (localize over ALL gold variants) ──
    log(f"\nLocalizing {len(selected)} records over all gold variants ...")
    records = []
    fg_localized = 0
    for i in selected:
        d = docids_all[i]
        tokens = doc_tokens[d]
        golds = [str(g) for g in answers_all[i]]
        localized, cited, lscore = localize_all_variants(tokens, golds)
        if localize_gold(tokens, golds[0]):       # first-gold-only (as-coded)
            fg_localized += 1
        records.append({
            "record_id": str(qids_all[i]),
            "doc_id": int(d),
            "split": None,                         # assigned below, by doc_id
            "image_path": f"data/images/{d}.png",
            "question": quest_all[i],
            "gold_answers": golds,                 # ALL variants; [0] canonical
            "ocr_menu": doc_menu[d],
            "ocr_menu_hash": doc_hash[d],
            "gold_localized": bool(localized),
            "cited_ids": cited,
            "target_answer": golds[0] if localized else "unanswerable",
            "target_is_abstain": (not localized),
            "abstain_reason": None if localized else "ocr_miss",
            "doc_type": list(qtypes_all[i]),
            "localizer_score": lscore,
        })

    # ── 3-way split BY doc_id, stratified by each doc's dominant first-type ──
    # Per stratum (counting by records): fill ~TEST_FRAC into test, then ~VAL_FRAC
    # into validation, the rest train. A doc lands in exactly one split.
    doc_records = defaultdict(list)
    for r in records:
        doc_records[r["doc_id"]].append(r)
    doc_domtype, doc_reccount = {}, {}
    for d, rs in doc_records.items():
        doc_domtype[d] = Counter(first_type(r["doc_type"]) for r in rs).most_common(1)[0][0]
        doc_reccount[d] = len(rs)

    group_docs = defaultdict(list)
    for d in sel_docs:
        group_docs[doc_domtype[d]].append(d)

    doc_split = {}
    for ft, docs in group_docs.items():
        docs = docs[:]
        rng.shuffle(docs)
        group_records = sum(doc_reccount[d] for d in docs)
        target_test = round(TEST_FRAC * group_records)
        target_val = round(VAL_FRAC * group_records)
        test_acc = val_acc = 0
        for d in docs:
            if test_acc < target_test:
                doc_split[d] = "test"
                test_acc += doc_reccount[d]
            elif val_acc < target_val:
                doc_split[d] = "validation"
                val_acc += doc_reccount[d]
            else:
                doc_split[d] = "train"
    for r in records:
        r["split"] = doc_split[r["doc_id"]]

    # ── tallies ──
    n_total = len(records)
    n_ans   = sum(1 for r in records if not r["target_is_abstain"])
    n_abs   = n_total - n_ans
    splits = {name: [r for r in records if r["split"] == name]
              for name in ("train", "validation", "test")}
    train, val, test = splits["train"], splits["validation"], splits["test"]
    n_train, n_val, n_test = len(train), len(val), len(test)

    # ── self-check assertions ──
    doc_to_splits = defaultdict(set)
    for r in records:
        doc_to_splits[r["doc_id"]].add(r["split"])

    checks = {}
    checks["no_doc_in_multiple_splits"] = all(len(s) == 1 for s in doc_to_splits.values())
    checks["abstain_empty_cited_and_ocr_miss"] = all(
        r["cited_ids"] == [] and r["abstain_reason"] == "ocr_miss"
        for r in records if r["target_is_abstain"]
    )

    def answerable_ok(r):
        menu_ids = {m["id"] for m in r["ocr_menu"]}
        return bool(r["cited_ids"]) and set(r["cited_ids"]).issubset(menu_ids)
    checks["answerable_cited_nonempty_subset_of_menu"] = all(
        answerable_ok(r) for r in records if not r["target_is_abstain"]
    )
    checks["gold_answers_nonempty_list"] = all(
        isinstance(r["gold_answers"], list) and len(r["gold_answers"]) > 0 for r in records
    )
    checks["ocr_menu_hash_present"] = all(
        isinstance(r["ocr_menu_hash"], str) and r["ocr_menu_hash"] for r in records
    )
    # invariant guards (extra safety, not in the required list)
    checks["abstain_iff_not_localized"] = all(
        r["target_is_abstain"] == (not r["gold_localized"]) for r in records
    )
    checks["target_answer_consistent"] = all(
        (r["target_answer"] == "unanswerable") == r["target_is_abstain"] for r in records
    )

    all_pass = all(checks.values())

    # ── abstain-balance check on table/list + form (must have BOTH classes) ──
    def has_both(records_subset, ft):
        sub = [r for r in records_subset if first_type(r["doc_type"]) == ft]
        a = sum(1 for r in sub if not r["target_is_abstain"])
        b = sum(1 for r in sub if r["target_is_abstain"])
        return a, b
    balance = {}
    for ft in ("table/list", "form"):
        balance[ft] = {
            "all":        has_both(records, ft),
            "train":      has_both(train, ft),
            "validation": has_both(val, ft),
            "test":       has_both(test, ft),
        }

    # ── print summary ──
    log("\n" + "=" * 70)
    log("SELF-CHECK ASSERTIONS")
    log("=" * 70)
    for name, ok in checks.items():
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    log("")
    log(f"SUMMARY: total={n_total}  answerable={n_ans}  abstain={n_abs} "
        f"({100*n_abs/n_total:.1f}%)")
    for name in ("train", "validation", "test"):
        s = splits[name]
        na = sum(1 for r in s if r["target_is_abstain"])
        log(f"  {name:<11} records={len(s):5}  abstain={na:4} ({100*na/len(s):.1f}%)")
    log(f"  unique images={len(sel_docs)}  size={img_bytes_total/1e6:.1f} MB")
    log(f"  first-gold-only localized={fg_localized}  all-variant localized={n_ans}  "
        f"variant-recovery gap={n_ans - fg_localized}")
    log("\nAbstain balance (answerable, abstain):")
    for ft, by in balance.items():
        log(f"  {ft:<11} all={by['all']}  train={by['train']}  val={by['validation']}  test={by['test']}")

    if not all_pass:
        log("\n*** ONE OR MORE ASSERTIONS FAILED — NOT writing master.jsonl. ***")
        for name, ok in checks.items():
            if not ok:
                log(f"    FAILED: {name}")
                # show up to 3 offenders
                if name == "answerable_cited_nonempty_subset_of_menu":
                    bad = [r["record_id"] for r in records
                           if not r["target_is_abstain"] and not answerable_ok(r)][:3]
                    log(f"      offending record_ids: {bad}")
                if name == "abstain_empty_cited_and_ocr_miss":
                    bad = [r["record_id"] for r in records if r["target_is_abstain"]
                           and not (r["cited_ids"] == [] and r["abstain_reason"] == "ocr_miss")][:3]
                    log(f"      offending record_ids: {bad}")
        _logf.close()
        sys.exit(1)

    # ── write artifacts (only on full pass) ──
    with open(MASTER, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"\nwrote {MASTER.relative_to(ROOT)}  ({n_total} records)")

    image_count = len(list(IMAGES.glob("*.png")))
    manifest = {
        "dataset": "FaithfulDocVQA v1 master",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "build_seconds": round((datetime.now() - t_start).total_seconds(), 1),
        "seed": SEED,
        "source_dataset": SOURCE_NAME,
        "source_config": SOURCE_CONFIG,
        "source_split": SOURCE_SPLIT,
        "excluded_doc_types": sorted(EXCLUDE),
        "clean_images_only": True,
        "augraphy_or_masking_applied": False,
        "build": "full-eligible 3-way (70/15/15 by docId, stratified by doc_type)",
        "total_records": n_total,
        "answerable_records": n_ans,
        "abstain_records": n_abs,
        "abstain_pct": round(100 * n_abs / n_total, 2),
        "split_records": {name: len(splits[name]) for name in ("train", "validation", "test")},
        "split_docs": {name: sum(1 for d in sel_docs if doc_split[d] == name)
                       for name in ("train", "validation", "test")},
        "split_abstains": {name: sum(1 for r in splits[name] if r["target_is_abstain"])
                           for name in ("train", "validation", "test")},
        "val_test_shared_for_A_and_B": True,
        "train_mode_filter": "A=gold_localized only; B=all (applied to TRAIN only)",
        "image_dir": "data/images",
        "image_count": image_count,
        "image_total_bytes": img_bytes_total,
        "image_total_mb": round(img_bytes_total / 1e6, 1),
        "first_gold_localized": fg_localized,
        "all_variant_localized": n_ans,
        "variant_recovery_gap": n_ans - fg_localized,
        "imported_functions": IMPORTED_FUNCTIONS,
        "ocr": "pytesseract.image_to_data via phase0.pipeline.extract_ocr_tokens (Tesseract 5.5.1)",
        "localizer": ("phase0.phase05.find_bbox(0.85) -> find_bbox_lenient "
                      "(cascade 0.85/0.7/0.6/0.5 + numeric fallback), evaluated over "
                      "ALL gold variants; localize_gold used for first-gold metric"),
        "assertions": {k: bool(v) for k, v in checks.items()},
        "all_assertions_pass": all_pass,
        "scope_note": ("faithfuldocvqa_scope.md not found in the repo/workspace; built "
                       "from the task's inline scope citations (§1 exclude handwritten+"
                       "Image/Photo, §6 imperfect-but-frozen), PROJECT_STATUS LOCALIZER "
                       "CHECK (n=500), and REWARD_DESIGN.md:69-71."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log(f"wrote {MANIFEST.relative_to(ROOT)}")

    # ── composition cross-tab ──
    md = []
    md.append("# Dataset composition — FaithfulDocVQA v1 master\n")
    md.append(f"_Generated by `scripts/build_master_dataset.py` at "
              f"{manifest['built_at']} (seed {SEED})._\n")
    md.append(
        "Rows are keyed by the **first** DocVQA `question_type` (the primary label; "
        "`question_types` is multi-label). Columns: **answerable** = gold localized in "
        "the OCR menu; **abstain (ocr_miss)** = gold not recoverable by the frozen "
        "localizer. This is the stratification + abstain-balance check.\n")
    md.append(render_xtab("ALL", records))
    md.append(render_xtab("TRAIN", train))
    md.append(render_xtab("VALIDATION", val))
    md.append(render_xtab("TEST", test))
    md.append("### Abstain balance — table/list & form must carry BOTH classes\n")
    md.append("| doc_type | split | answerable | abstain |")
    md.append("|---|---|---:|---:|")
    for ft in ("table/list", "form"):
        for split_name in ("all", "train", "validation", "test"):
            a, b = balance[ft][split_name]
            md.append(f"| {ft} | {split_name} | {a} | {b} |")
    md.append("")
    md.append(f"- Total records: **{n_total}** — answerable **{n_ans}**, "
              f"abstain **{n_abs}** ({100*n_abs/n_total:.1f}%).")
    md.append(f"- 3-way split by doc_id (no doc in >1 split): train **{n_train}** / "
              f"validation **{n_val}** / test **{n_test}**.")
    md.append("- validation & test are FULL and SHARED (identical for SFT-A and SFT-B); "
              "the A=answerable-only filter applies to TRAIN only.")
    md.append(f"- Images materialized unique by doc_id: **{len(sel_docs)}** "
              f"({img_bytes_total/1e6:.1f} MB).\n")
    COMPOSITION.write_text("\n".join(md))
    log(f"wrote {COMPOSITION.relative_to(ROOT)}")

    log(f"\nDONE in {manifest['build_seconds']}s. all_assertions_pass={all_pass}")
    _logf.close()


if __name__ == "__main__":
    main()
