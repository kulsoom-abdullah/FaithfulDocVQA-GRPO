# GRPO finals on the natural master test split — transfer confirmed, over-refusal cost measured

Scored 2026-08-08 with the committed `scripts/eval_scorer.py`. Generations: fresh
greedy passes (recorded 2026-08-08, RunPod A6000 session) over the FULL master
test split (n=756: **77 natural abstains / 679 answerable**) for both GRPO final
checkpoints — the population the GRPO experiments had never measured
(`mcnemar_seed_vs_grpo.md` §"what it does not establish"). Adapters sha-gated to
`data/modal_pull/MANIFEST.txt` before generating (run-1 `339236a6…`, run-2
`59d335a2…`); same 4-bit NF4 + adapter loader as the McNemar generations.
Files: `data/checkpoints/eval/grpo_run{1,2}_seed42/gen_test_checkpoint-150.json`
(sha256 in [`ARTIFACTS.sha256`](ARTIFACTS.sha256)). Stop reasons: run-1 756 eos;
run-2 754 eos / 2 length_cap.

## Results (seed row from PROVENANCE §3 for reference)

| model | natural honest_abstain (n=77) | answerable false_abstain (n=679) | answerable valid_citation | gated F1 (citation F1 among valid citations only) micro (n) | malformed |
|---|---|---|---|---|---|
| seed ck-642 (reference) | 0.805 (62/77) | 0.038 | — | — | — |
| run-1 ck-150 (control) | 0.766 (59/77) | 0.034 | 0.965 | 0.8296 (655) | 0.0013 |
| **run-2 ck-150 (shaped)** | **1.000 (77/77)** | **0.177 (120/679)** | 0.820 | 0.8694 (557) | 0.0026 |

found_answer accuracy stayed high for both (run-1 0.913/0.909, run-2 0.916/0.922
answerable/abstain) — no image-reading collapse.

## Reading — both halves stated plainly

1. **The abstain shaping transferred.** Run-2 was trained and validated
   exclusively on manufactured (evidence-dropped) abstains, yet abstains on
   **77/77 natural OCR-miss test abstains** — up from the seed's 62/77 and the
   control's 59/77. This answers the open population-transfer question
   (`PROVENANCE.md` §5): the learned behavior is not tamper-signature-specific.
2. **It costs real over-refusal.** False abstains on gold-answerable rows rose
   **0.034 → 0.177** (23 → 120 of 679) versus the control — the anti-collapse
   concern `DECISIONS.md` §3(b) named. The earlier "valid_citation held at
   0.958" was the 48-row val instrument; on the 679-row natural test the
   answering rate pays visibly. Citation quality *among the answers it still
   gives* held (gated micro F1 0.870 vs control 0.830, over a smaller n=557 pool
   — the model answers less, but no worse).
3. Net: run-2 ck-150 is a **recall-heavy abstainer** — perfect refusal on both
   abstain populations (1.000 natural, 0.864 manufactured clean-236 — 204/235 =
   0.868 after the 2026-08-09 `51687_mfab` adjudication, see label_audit note) at ~14 pp
   of over-refusal on answerable rows. Operating-point selection between ck-50 /
   ck-100 / ck-150 (the deployable set) should weigh exactly this trade-off;
   only ck-150 is measured on the natural split so far.
