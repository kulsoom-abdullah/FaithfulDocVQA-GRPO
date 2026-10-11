# Run-3 balanced SFT — pre-registered analysis plan (nodecoy primary)

Recorded 2026-08-09, before the run-3 launch it governs. This file was written at
the kill/relaunch decision point: the initial full-mix launch (all 2,175
manufactured train abstains) was stopped at step 282/2,100 and superseded by the
nodecoy arm below, for the reason in (a). The run's `train_manifest.json`
(persisted at launch) must match the mix in (b) exactly.

**Completion note.** Run-3 is finished; its results are in
[`run3_test_results.md`](run3_test_results.md). This dated plan retains the
historical counts and proposed extensions; those extensions are possibilities,
not current commitments.

## (a) Decision rationale — why recurs-False only

A survivor-audit mislabel class was hand-confirmed inside the kept decoy cell:
`51687_mfab` (adjudicated 2026-08-09 against the page, doc 1266) — the original
gold cite was a first-occurrence string-match quirk (Feb-7 row for a Feb-13
question), and a genuinely supporting site (the Feb-13 itinerary row, post-drop
id 72, corroborated by id 83) **survived the evidence drop**, so the row's
"clean abstain" label is wrong. Test-set calibration (`docs/label_audit.md` §3:
13/40 recurs-True rows hand-excluded) suggests **~32% of recurs-True rows may be
mislabeled**. Training therefore uses **only recurs-False rows**, where no accepted
answer string remains under the implemented string-matching rule. This checks text
presence; it does not prove that no remaining text supports the question.

## (b) Training mix (must match train_manifest.json at launch)

`data/manufactured_train_abstains_nodecoy.jsonl` — 1,764 rows (strata
single_unique 896 / multi_unique 868; all `answer_recurs_in_menu=False`, all
`split=="train"`), sha256 `28a0bf29ed7d201ef247529877b988a350b7a588cbce3421b5ea9ec28a6b9b87`.
Merged with mode-B train (3,423 incl. 313 natural ocr_miss abstains):
**5,187 records / 2,077 abstain / 40.0% abstain-frac** — matching the GRPO
pool's 40/60 ratio. ~1,945 optimizer steps at bs1×accum8×3ep; checkpoint/eval
cadence `--save-steps 324` (half-epoch) → 6 checkpoints (324…1944). All other
hyperparameters identical to the documented `sft_v3_32g` recipe.

## (c) Pre-specified comparisons (unchanged from the run-3 design)

1. **Primary McNemar**: best run-3 checkpoint vs. seed ck-642, honest_abstain on
   the clean-236 manufactured test subset (exact two-sided binomial on
   discordant pairs — same form as `mcnemar_seed_vs_grpo.md`).
2. **Balanced-SFT vs GRPO-run-2 contrast**: the same instrument and metric,
   run-3-best vs. GRPO run-2 ck-150 — does supervised exposure to the abstain
   distribution match what the reward differential bought?
3. **Watch column (not a gate): false_abstain on the 679 natural-test
   answerable rows** — GRPO run-2 paid 0.177 here (vs 0.034 control); run-3's
   value on the same column decides the run-4 question.
4. **Run-4 fork condition**: if run-3 shows material over-refusal on (3), run-4
   prices false_abstain negative in the GRPO reward (currently 0.0 base) and
   re-runs the shaping from the better seed.

## (d) Full-mix arm: relegated

The decoys-included arm (all 2,175) is now an **optional ablation** ("does decoy
exposure matter?") — not scheduled. Its 282-step partial was discarded unused.

## (e) Future work — recovering the 411 decoy rows

Two possible extensions proposed at the time for usable decoy training data:
- **Drop-all-occurrences manufacture**: scrub every occurrence of every gold
  variant (needs short-string over-drop guards). Realism cost, noted honestly:
  real documents legitimately repeat answers, so scrubbing all occurrences makes
  menus *less* representative of deployment inputs.
- **LLM-judge adjudication of surviving sites** (preferred): keep menus
  realistic, judge each surviving occurrence as supporting/non-supporting
  (51687_mfab-style row analysis automated), fix labels instead of deleting
  rows — the path that turns decoy rows into supervised lure-resistance data.

## Parked wording (for README + walkthrough notebook — recorded 2026-08-09)

> Manufactured abstains used for balanced SFT are restricted to records where,
> after removing the reference evidence, no accepted-answer string remains under
> the implemented string-matching rule. This checks text presence, not whether
> remaining text supports the question. Records where the answer text survives
> elsewhere (~19%) are excluded from balanced-SFT training: a
> surviving occurrence may genuinely support the answer (one hand-confirmed
> case; the test-set audit excluded 14 of 40 such rows), so their labels require
> review rather than an assumption of insufficient support.
