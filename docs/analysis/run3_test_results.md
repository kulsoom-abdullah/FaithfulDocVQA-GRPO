# Run-3 balanced SFT on the frozen test instruments — pre-registered results (2026-08-10)

Executes the comparisons pre-registered in [`run3_analysis_plan.md`](run3_analysis_plan.md)
§(c) for **ck-1296**, the checkpoint selected in
[`run3_checkpoint_selection.md`](run3_checkpoint_selection.md). Scored with the
committed `scripts/eval_scorer.py`.

## The pre-registered question, and the answer

> **Balanced-SFT vs GRPO-run-2 contrast**: the same instrument and metric, run-3-best
> vs. GRPO run-2 ck-150 — does supervised exposure to the abstain distribution match
> what the reward differential bought? — plan §(c)2

Balanced SFT achieved 212/235 correct manufactured refusals versus run-2’s 204/235,
with 78 versus 120 incorrect refusals on answerable questions.

On the clean-235 manufactured test subset, the paired comparison contains **20
discordant pairs**, with **14 improvements / 6 regressions** relative to run-2 and
exact two-sided **p = 0.115**. These are descriptive results for the saved runs;
they do not establish equivalence or superiority.

## Primary comparison — seed ck-642 → run-3 ck-1296, clean-235

| | honest_abstain on clean-235 |
|---|---|
| seed ck-642 | 68/235 = 0.289 |
| **run-3 ck-1296** | **212/235 = 0.902** |

| both-yes | both-no | fixes b | breaks c | exact two-sided p |
|---:|---:|---:|---:|---:|
| 68 | 23 | **144** | **0** | **8.97e-44** |

Scoped precisely: on this instrument and this metric, run-3 ck-1296 changed the
per-record outcome positively on 144 of 235 records and **produced zero abstain-side
regressions** — no record the seed abstained on correctly became a false grounding.
`c = 0` is the substantive part; the p-value only says a 144/0 split is not chance.

This is a statement about the net direction of per-record change on the manufactured
abstain instrument. It does not speak to answerable-side behaviour, which is measured
separately below and moves in the opposite direction.

## The trade-off surface — three axes

One instrument cannot express this result. Abstention gains and over-refusal costs are
different measurements on different populations, and the trade-off between them **is**
the finding.

| model | manufactured abstention (clean-235) | natural transfer, honest_abstain (n=77) | over-refusal, false_abstain (n=679) |
|---|---|---|---|
| seed ck-642 | 68/235 = 0.289 | 62/77 = 0.805 | 0.038 |
| run-1 ck-150 (control) | 62/235 = 0.264 | 59/77 = 0.766 | 0.034 |
| run-2 ck-150 (shaped) | 204/235 = 0.868 | **77/77 = 1.000** | 120/679 = 0.177 |
| **run-3 ck-1296 (balanced SFT)** | **212/235 = 0.902** | 71/77 = 0.922 | 78/679 = 0.115 |

Read across the row rather than down a column. Run-3 and run-2 make different
refusal trade-offs: run-3 carries about
two-thirds of run-2's over-refusal cost (0.115 vs 0.177), while run-2 retains the one
axis run-3 does not reach — **perfect** natural transfer, 77/77 against run-3's 71/77.
Neither dominates the other across all three.

Run-3 ck-1296 additionally: valid_citation **0.885** on the 679 answerable rows,
malformed **0/756 = 0.0000**.

Sources for the non-run-3 rows: manufactured column from
[`mcnemar_seed_vs_grpo.md`](mcnemar_seed_vs_grpo.md) (post-`51687_mfab` adjudication
denominators); natural and over-refusal columns from
[`grpo_finals_natural_test.md`](grpo_finals_natural_test.md).

## Watch column — false_abstain on the 679 natural-test answerable rows

Plan §(c)3 designates this a **watch column, not a gate**; it decides the run-4
question. All references labelled:

| reference | false_abstain (n=679) |
|---|---|
| seed ck-642 | 0.038 |
| run-1 ck-150 (control) | 0.034 |
| **run-3 ck-1296** | **0.115 (78/679)** |
| run-2 ck-150 (shaped) | 0.177 (120/679) |

### Fork verdict

Plan §(c)4: *"if run-3 shows material over-refusal on (3), run-4 prices false_abstain
negative in the GRPO reward (currently 0.0 base) and re-runs the shaping from the
better seed."*

**The condition is met.** Run-3 refuses 78 of 679 gold-answerable rows, 0.115 against
the control's 0.034 — a 3.4× increase, and material by any reading of the term.
**Run-4 is therefore confirmed as pre-registered: false_abstain priced negative,
shaping from ck-1296.** This records the verdict only — launch is a separate
budget and sequencing decision and is **not** authorised by this document.

The planned run-4 comparison tests whether penalizing incorrect refusal changes
the balance between correct and incorrect refusals. It does not establish whether
the model becomes better at identifying which OCR text supports a question.
Reviewing surviving answer occurrences with an LLM judge was proposed as a possible
data-quality extension.

## Scoring validation

The pipeline reproduces both committed baselines **to the digit** before computing
anything new — seed ck-642 **68/235 = 0.289** and run-2 ck-150 **204/235 = 0.868**,
matching `mcnemar_seed_vs_grpo.md`'s 2026-08-09 adjudication note exactly. The new
number is therefore computed the same way as the numbers it is compared against.

Further guards:
- **clean-235 asserted**, not assumed: derived from `exclusion_reason is None` on the
  committed `data/manufactured_test_abstains.jsonl` (250 rows, 15 excluded). The
  pre-adjudication `in_clean_236` flag in
  `mcnemar_seed_vs_grpo_per_record.jsonl` is deliberately stale and was not used.
- **Join coverage asserted** for every generation — 250/250 on all three manufactured
  files, 756/756 on the natural split. This closes the failure mode recorded as an
  open item in the pipeline field notes: an absent `record_id` joins via `.get()` to
  `None`, classifies as a parse failure, and reports honest_abstain **0.000** with no
  exception raised — a wrong answer that looks like a result.
- **Instrument identity verified**: the Modal data volume's copy of
  `manufactured_test_abstains.jsonl` is the June-30 pre-audit build. Same 250
  record_ids and zero mismatches on `question`, `image_path`, `doc_id`, `ocr_menu`,
  `gold_answers`, `target_is_abstain`, `gold_localized`, `cited_ids`; it lacks only
  the audit overlay. Generation used the volume copy, scoring used the committed file.

## Generations

`ck-1296` test-split and manufactured-test passes, Modal L40S, via
`modal_gen_run3.py` (parameter-passing mirror of `modal_gen.py`; `run_dir`
parameterized, generation path verbatim). Same `eval_runner.load_adapted_model`
loader as every other comparison generation — base in 4-bit NF4 + PEFT adapter,
greedy, `max_new_tokens 128` — which is what the paired design requires.

Stop reasons: **756/756 eos** and **250/250 eos**, no length-cap truncation. Prompt
parity checked 3/3 on the smoke pass. Files
`data/checkpoints/eval/sft_v3_balanced_nodecoy/gen_{test,manufactured}_checkpoint-1296.json`,
sha256 in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).

The pod that trained run-3 could not be restarted for generation — *"not enough free
GPUs on the host machine"* — so the passes moved to Modal, whose volumes already held
the data, the base model and the sha-verified adapter.
