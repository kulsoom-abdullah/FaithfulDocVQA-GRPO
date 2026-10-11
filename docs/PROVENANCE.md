# Provenance & corrections — what each headline number is, and is not

This file maps the original SFT/GRPO headline numbers to their source artifacts.
For the current comparison, see the
[walkthrough notebook](../notebooks/faithfuldocvqa_walkthrough.ipynb).
It exists because an adversarial audit (2026-08-01) found
that the project's own working summaries had blended numbers from **two different
training runs under two different output contracts**. Nothing here retracts a
measurement — every number below is real and reproducible — but several needed
re-attachment to the model and instrument that actually produced them. Historical
status docs that remain in the repo are kept as written and carry a banner
pointing here.

## 1. Two contracts, two runs — never blend them

| | v1 (2-field) | v3 (3-field) |
|---|---|---|
| Output contract | `{answer, evidence_ids}` | `{found_answer, answerable_from_menu, evidence_ids}` |
| Runs | SFT-A (answer-only) **and** SFT-B (answer + ocr_miss abstains), June 16–18 | **SFT-B only** (`sft_v3_32g`), 3 epochs, June 23–24 — **no A control exists in v3** |
| "honest_abstain" means | output is the literal template `{"answer":"unanswerable","evidence_ids":[]}` | `answerable_from_menu == false` — the model still answers in `found_answer` (31/39 honest-abstain rows emit the gold answer) |
| Headline numbers | val honest_abstain **0.739** (34/46), macro-F1 **0.824** (ungated), malformed 0 — `data/checkpoints/eval/sft_B_seed42/eval_checkpoint-642.json` | val **0.848** (39/46), test **0.805** (62/77) — recomputable from committed `gen_checkpoint-*.json` / `gen_test_checkpoint-*.json` with `scripts/eval_scorer.py` |
| Adapter sha256 (ck-642) | `c300225c…` (`data/checkpoints/sft/sft_B_seed42/checkpoint-642`) | `d76fa2b4…` (`data/checkpoints/sft_v3_32g/sft_B_seed42/checkpoint-642`) — **this is the RL seed** (hash-matched to the GRPO ref adapters in `data/modal_pull/MANIFEST.txt` — a volume-pull mirror, gitignored, not in this repo; the inline sha256s are the checkable record) |

Consequences:
- The **A-vs-B causal claim** ("abstention must be trained": 0.000 vs 0.739, disjoint
  Wilson CIs) is a **v1 result with a control**. It was never re-run under v3.
- The **RL seed's numbers** (0.848 val / 0.805 test) are **v3, B-only, no control**.
- 0.74 → 0.805 is **not a trajectory**: different contracts, different honest_abstain
  definitions, different physical models. Under the v1 definition the v3 seed would
  score 0/46 (it never emits the literal "unanswerable" template).

## 2. Training data: natural abstains in the original June SFT runs

The original June SFT runs trained from `data/master.jsonl` only. B's treatment delta
over A is exactly **313 natural `ocr_miss` abstains** (9.1% of its 3,423-row train set). The
manufactured-abstain sets postdate those SFT runs (test set recorded 2026-06-30;
train/val 2026-07-12; last original June SFT run finished 2026-06-24). They were later
used by evaluation probes, GRPO, balanced SFT, and matched-pool SFT. Zero `_mfab`
records exist in master; all 436 master abstains have `abstain_reason == "ocr_miss"`.

Note the corollary: "natural abstain" is a **preprocessing stratum**, not question
unanswerability — the source DocVQA questions are all answerable from the image;
the abstain label marks the frozen Tesseract/localizer failing to find the gold
string (abstain rows are OCR-sparse: median menu 40 tokens vs 182 for answerable).

## 3. Checkpoint selection and the test split

ck-642 was selected as the honest_abstain argmax on the **n=46 validation abstain
rows** — the same rows behind the val headline (winner's-curse applies; neighbors
score 0.63–0.67). The test split was **not** touched once: test generations exist
for **all six** checkpoints (volume timestamps 2026-06-29, 15:37–18:05 EDT),
two weeks before GRPO fixed ck-642 as its seed (2026-07-13/14), and ck-642 is the
test argmax. Full test curve (current scorer, n=77 natural test abstains):

| ckpt | 214 | 428 | **642** | 856 | 1070 | 1284 |
|---|---|---|---|---|---|---|
| honest_abstain | 0.312 | 0.208 | **0.805** | 0.675 | 0.649 | 0.675 |
| false_abstain (n=679) | 0.010 | 0.007 | **0.038** | 0.021 | 0.018 | 0.021 |

Neighboring checkpoints score **0.649–0.675**, showing observed variation across
checkpoints, not a forward estimate of ck-642's future performance. ck-642 also
maximizes false_abstain on this test set.

## 4. F1 instruments are era-specific

The v1 "~0.82–0.84 unchanged between arms" macro-F1 is **ungated**: computed over
ALL gold-answerable rows, abstained/malformed rows scored as misses — a legitimate
fixed-population A-vs-B comparison *within v1*. The current v3 scorer reports
`citation_f1_gated`: pooled **only over rows the model itself grounded**
(`scripts/eval_scorer.py`, `aggregate()`). These are different quantities; never
compare across eras.

## 5. What the learned abstention is (mechanism, resolved on the test set)

Current reviewed breakdown of the 250 manufactured test examples, for run-2
checkpoint 150:

| OCR condition | n | run-2 checkpoint 150 outcome |
|---|---|---|
| Answer string absent | 209 | Correct refusal **189/209** |
| Matching text retained as unrelated | 26 | Correct refusal **15/26**; false grounding **11/26** |
| Excluded supporting-occurrence cases | 15 | Refusal **2/15** |

Matching text judged unrelated still drew false grounding on 11 of 26 examples.
The retained cases' non-supporting labels come from the saved review fields, not a
complete per-example review log.

**Earlier breakdown (before the 2026-08-09 review):**

| menu condition | n | abstain rate (GRPO run-2 ck-150) |
|---|---|---|
| answer string absent | 209 | **0.904** |
| string present, hand-cleared as non-supporting | 27 | **0.556** |
| string present at a hand-confirmed valid site (the 14 audit exclusions) | 14 | **0.143** |

One example moved from the 27 retained matching-text cases to the 14 exclusions
after the 2026-08-09 review; see the [label review](label_audit.md). This produced
the current 26/15 split above; the 209 answer-string-absent cases were unchanged.

What remains open is the **64-row
validation-instrument decoy audit** (`docs/label_audit.md`), which affects the GRPO
val curve, not the test results.

## 6. Original SFT versus GRPO: refusal results

The earlier analysis used 236 examples. One example was later excluded after review,
leaving the 235 used in the current results. This did not change the comparison:
run-2 corrected 136 refusal errors made by the original SFT model without introducing
new ones on these 235 manufactured examples. The
[detailed historical explanation](analysis/mcnemar_seed_vs_grpo.md)
records the review and the unchanged comparison.

Run-1 corrected one refusal error and introduced seven. The earlier analysis also
checked 205 examples from documents absent from the test set previously used to
inspect SFT checkpoints; the run-2 improvement remained similar. Run-1 and run-2
differed in their refusal rewards and how often they were evaluated during training.
Each training setting was run once; the exact historical settings are not distributed
in this repository.

## 7. Artifact index

Committed generation/eval dumps (sha256 in
[`docs/analysis/ARTIFACTS.sha256`](analysis/ARTIFACTS.sha256)):

- v1 A/B eval JSONs: `data/checkpoints/eval/sft_{A,B}_seed42/eval_checkpoint-*.json`
- v3 val gens (all six ckpts): `data/checkpoints/eval/sft_B_seed42/gen_checkpoint-*.json`
- v3 test gens (all six ckpts): `data/checkpoints/eval/sft_B_seed42/gen_test_checkpoint-*.json`
- v3 seed manufactured gen: `data/checkpoints/eval/sft_B_seed42/gen_manufactured_checkpoint-642.json`
- GRPO finals manufactured gens: `data/checkpoints/eval/grpo_run{1,2}_seed42/gen_manufactured_checkpoint-150.json`

Rescore any of them on CPU with `scripts/eval_scorer.py` against
`data/master.jsonl` / `data/manufactured_test_abstains.jsonl`.
