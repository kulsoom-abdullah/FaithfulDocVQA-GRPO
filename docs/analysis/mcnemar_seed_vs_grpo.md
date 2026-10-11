# Paired McNemar: SFT seed vs final GRPO checkpoints — honest_abstain on the frozen manufactured test set

Date run: 2026-08-01. Scorer: `scripts/eval_scorer.py` as committed alongside this report (recorded 2026-08-01) (bucket `HONEST_ABSTAIN` = parsed `answerable_from_menu == false` on a gold-abstain row; malformed counts as not-honest). Per-record outcomes: [`mcnemar_seed_vs_grpo_per_record.jsonl`](mcnemar_seed_vs_grpo_per_record.jsonl).

**Historical population note.** The dated tables below use the original 236-example
clean manufactured population. One additional example was later excluded after review,
leaving the 235 used in current results. The historical tables and statistics are
preserved; the note at the end records the review and its effect on the comparison.

## Models compared

| role | adapter path | adapter_model.safetensors sha256 |
|---|---|---|
| Before — RL seed | `outputs/sft_v3_32g/sft_B_seed42/checkpoint-642` (volume) = `data/checkpoints/sft_v3_32g/sft_B_seed42/checkpoint-642` (local) | `d76fa2b4609ba126bb427e5f338ba5a9c65ef8111e0bb37df9d31d344ba09691` |
| After — GRPO run 2 (abstain-shaped reward), final | `outputs/grpo_runs/grpo_run2_seed42/checkpoint-150` (= `final_adapter`, byte-identical) | `59d335a22b34b43128b16daeef45e16d276afc0f40a12543fd104a35753459dc` |
| After — GRPO run 1 (control, no abstain shaping), final | `outputs/grpo_runs/grpo_run1_seed42/checkpoint-150` (= `final_adapter`, byte-identical) | `339236a663fa85bd0164a69d949819d6eab70d06ad94f0068143482bd6519a59` |

Both GRPO runs were initialized from the seed adapter above (`grpo_run_config.json`, both runs; hashes match `data/modal_pull/MANIFEST.txt` — config files and manifest live on the gitignored volume-pull mirror, not in this repo; the sha256s printed here are the in-tree record). The mirror-only `final_adapter_CRASHED_3STEPS_DO_NOT_USE` was excluded.

## Instrument

`data/manufactured_test_abstains.jsonl` — 250 rows, all `target_is_abstain=True`, `gold_localized=False`.

- **Primary: clean subset n=236** (`exclusion_reason` is None).
- Secondary: full file n=250; document-disjoint n=205 (clean rows excluding the 13 doc_ids shared with the SFT checkpoint-selection test set: 1165, 4202, 5324, 6275, 7010, 9305, 10622, 12709, 13097, 13650, 14685, 14723, 14757).

Frozen status, stated precisely: this set had been scored once before for the **seed** (the `verify_manufactured.py` recompute; the 0.288 clean-subset headline is recorded in `docs/findings/survivor_audit.md`); it had **never** been opened for any GRPO checkpoint before this run. The copy on the `faithfuldocvqa-data` volume is the June-30 pre-audit build: identical 250 record_ids and identical prompt-relevant fields (question, ocr_menu, image_path, doc_id); the committed file only adds the audit-overlay fields (`exclusion_reason`, `gold_label_defect`, `survivor_pending`; audit recorded 2026-07-02). Verified field-by-field before generating.

## Generations

- Seed: the committed volume artifact `eval/sft_B_seed42/gen_manufactured_checkpoint-642.json` (2026-06-30), reused — not regenerated. 250 records, greedy `do_sample=false`, `max_new_tokens=128`, stops eos 249 / length_cap 1.
- run1 ck-150 and run2 ck-150: generated 2026-08-01 on Modal (L40S) with a parameter-passing mirror of `modal_gen.py` (no repo script modified), same `gen_persist.py` → `eval_runner` path, same decoding, 250/250 eos both. Committed at `data/checkpoints/eval/grpo_run{1,2}_seed42/gen_manufactured_checkpoint-150.json` (originals on the `faithfuldocvqa-ckpts` volume; sha256 in [`ARTIFACTS.sha256`](ARTIFACTS.sha256)).
- Loader note: all three generations used `eval_runner.load_adapted_model` — base in 4-bit NF4 (the same `build_bnb_config` as SFT training) + PEFT adapter. This is identical across the three models (what the paired design requires) but differs from the GRPO training/val-eval precision (bf16, no bitsandbytes).

Malformed outputs (scored not-honest): seed 1/250 (`1901_mfab`), run1 1/250 (`51697_mfab`), run2 0/250.

## Raw honest_abstain rates

| model | clean 236 | full 250 | disjoint 205 |
|---|---|---|---|
| seed (ck-642) | 68/236 = 0.288 | 68/250 = 0.272 | 51/205 = 0.249 |
| run1 ck-150 (control) | 62/236 = 0.263 | 62/250 = 0.248 | 48/205 = 0.234 |
| run2 ck-150 (shaped) | 204/236 = 0.864 | 206/250 = 0.824 | 173/205 = 0.844 |

## Paired McNemar results

Cell convention: b = fixes (before-no / after-yes), c = breaks (before-yes / after-no). Exact test = two-sided binomial, min(b,c) against Binomial(b+c, ½), doubled and capped at 1 — used as the authoritative form because discordant counts can be small (and one cell is 0 in several tables), where the χ² approximation is unreliable. χ² = (b−c)²/(b+c) (df=1), continuity-corrected χ²cc = (|b−c|−1)²/(b+c), reported for comparison.

### seed → run2 ck-150 (shaped)

| subset | both-yes | both-no | fixes b | breaks c | exact p | χ² (p) | χ²cc (p) |
|---|---|---|---|---|---|---|---|
| clean 236 | 68 | 32 | 136 | 0 | 2.30e-41 | 136.0 (2.0e-31) | 134.0 (5.4e-31) |
| full 250 | 68 | 44 | 138 | 0 | 5.74e-42 | 138.0 (7.3e-32) | 136.0 (2.0e-31) |
| disjoint 205 | 51 | 32 | 122 | 0 | 3.76e-37 | 122.0 (2.3e-28) | 120.0 (6.3e-28) |

### seed → run1 ck-150 (control)

| subset | both-yes | both-no | fixes b | breaks c | exact p | χ² (p) | χ²cc (p) |
|---|---|---|---|---|---|---|---|
| clean 236 | 61 | 167 | 1 | 7 | 0.0703 | 4.50 (0.034) | 3.12 (0.077) |
| full 250 | 61 | 181 | 1 | 7 | 0.0703 | 4.50 (0.034) | 3.12 (0.077) |
| disjoint 205 | 47 | 153 | 1 | 4 | 0.375 | 1.80 (0.180) | 0.80 (0.371) |

Note: on the clean/full subsets the uncorrected χ² (p=0.034) and the exact test (p=0.070) **disagree** across the 0.05 line — expected at b+c=8; the exact form is authoritative here, and the continuity-corrected χ² agrees with it.

### run1 ck-150 → run2 ck-150

| subset | both-yes | both-no | fixes b | breaks c | exact p | χ² (p) | χ²cc (p) |
|---|---|---|---|---|---|---|---|
| clean 236 | 62 | 32 | 142 | 0 | 3.59e-43 | 142.0 (9.7e-33) | 140.0 (2.7e-32) |
| full 250 | 62 | 44 | 144 | 0 | 8.97e-44 | 144.0 (3.6e-33) | 142.0 (9.7e-33) |
| disjoint 205 | 48 | 32 | 125 | 0 | 4.70e-38 | 125.0 (5.1e-29) | 123.0 (1.4e-28) |

## Consistency across subsets

The seed→run2 paired effect is consistent in direction and similar in magnitude on all three cuts (b/n = 0.576, 0.552, 0.595; c = 0 in all three), including the document-disjoint subset, so it is not carried by the 13 documents shared with the SFT checkpoint-selection test set or by the 14 excluded rows. The control contrast (seed→run1) is likewise consistent across cuts: small, net-negative (7 breaks vs 1 fix on clean), not significant by the exact test.

## What this establishes — and what it does not

On these held-out records, the run-2 GRPO checkpoint changed the honest_abstain outcome in the positive direction on 136 of 236 records and in the negative direction on 0, relative to the seed (exact p ≈ 2e-41); the control run-1 checkpoint showed no significant paired change (1 fix, 7 breaks, exact p = 0.07). This is a statement about the **net direction of per-record change on this instrument and metric only**. It does not quantify effect magnitude beyond the reported counts and rates, does not speak to any other metric (answerable-side behavior, citation quality, natural-abstain performance), and does not distinguish *why* records flipped. (The mechanism question — string-matching vs site-level verification — is addressed with a three-cell breakdown in [`docs/PROVENANCE.md`](../PROVENANCE.md) §5: abstention tracks answer-string presence, 0.904 / 0.556 / 0.143 across absent / present-non-supporting / present-valid-site conditions.) The three pairings on three subsets are nine tests; the pre-registered primary is seed→run2 on the clean n=236 — the others are secondary context.

---

**Adjudication note (recorded 2026-08-09).** `51687_mfab` was hand-adjudicated
against the page and excluded (surviving Feb-13-row site genuinely supports the
answer; see `docs/label_audit.md` §4-note and `apply_survivor_audit.py`), so the
clean subset is now **n=235** (15 exclusions total). **The primary McNemar is
untouched**: the record is a both-no cell in every pairing here (all three arms
false-ground it), so seed→run2 remains **b=136 / c=0**, exact p ≈ 2.3e-41; only
denominators shift — seed 68/235 = 0.289, run1 62/235 = 0.264, run2 204/235 =
0.868. The per-record file's `in_clean_236` flag reflects the pre-adjudication
overlay (recorded 2026-08-01) and is intentionally left as written.
