# phase0 — the frozen OCR + localizer layer

This directory is the **frozen** origin of the dataset. `scripts/build_master_dataset.py`
imports the OCR and localization functions from here verbatim so `data/master.jsonl`
matches the original audit bit-for-bit; per `CLAUDE.md` hard rules 1 and 2 they are never
tuned, patched, or re-implemented elsewhere.

**Names are kept stable for provenance, not tidiness.** Some modules here date from an
earlier Sonnet-API-era design and are no longer called by anything. They are retained so
that published numbers stay traceable to the code that produced them — renaming or
deleting them would break that chain. Nothing in this directory is reorganized.

## What is live

| symbol | module | used by |
|---|---|---|
| `extract_ocr_tokens` | `pipeline.py` | `build_master_dataset.py` — the real Tesseract token menu |
| `find_bbox`, `find_bbox_lenient` | `phase05.py` | `build_master_dataset.py` — fuzzy localizer primitives (0.85/0.7/0.6/0.5 cascade + numeric fallback) |
| `localize_gold` | `score_phase0_with_f1.py` | `build_master_dataset.py` — single-gold entry point |
| `compute_evidence_reward`, `ABSTAIN_CREDIT`, `FABRICATION_PENALTY` | `evidence_reward.py` | the eval scorer / GRPO reward path |

## What is legacy

Unused since the three-field format replaced the original two-field one. Kept for history;
each is marked in-file with a `LEGACY` banner.

- **`docvqa_capture_audit.py`** — the Tesseract answer-capture audit that established the
  OCR-miss rate. Standalone script, nothing imports it.
- **`localizer_check.py`** — the localizer study (resolution probes, forced-DPI retries,
  Wilson intervals). Standalone script, nothing imports it.
- **`test_reward.py`** — tests for the *old two-field* reward parser, superseded by
  `tests/` at the repo root. Not collected by the current suite.
- **Augraphy / masking helpers** inside `pipeline.py` and `phase05.py`
  (`build_augraphy_pipeline`, `apply_augraphy`, `mask_answer_region`) — v1 uses clean
  images only (`CLAUDE.md` hard rule 3). Retained because the v2 masked-image branch
  would start from them.
- **Anthropic-API-era helpers** (`compute_cost`, `image_to_b64`, `anls`,
  `step1_load_samples`, and the duplicated `ocr_tokens` / `find_answer_bbox` /
  `dump_ocr` variants) — from when phase0 called a hosted model directly.

Verified 2026-08-12: every symbol in the legacy list has **zero** references outside this
directory.
