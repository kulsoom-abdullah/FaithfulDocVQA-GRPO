# Run-4a on the frozen test instruments — pre-registered results (2026-08-17)

Executes the comparison pre-registered in [`run4_analysis_plan.md`](run4_analysis_plan.md)
§(e)1 and both watch columns of §(f), for **ck-100**, the checkpoint selected in
[`run4a_checkpoint_selection.md`](run4a_checkpoint_selection.md). Scored with the
committed `scripts/eval_scorer.py` via `scripts/score_run4a_test.py`.

The frozen test instrument was opened **once**, for this comparison only (plan §h).
Both generations were produced in a single pass each and are committed with sha256 in
[`ARTIFACTS.sha256`](ARTIFACTS.sha256).

## The pre-registered question, and the answer

> **Run-4a-best vs seed ck-1296** — *does RL still add anything once the seed already
> abstains well and the labels it learns from are construction-valid?* — plan §(e)1

All 31 bucket changes moved toward refusal: five additional correct manufactured
refusals, five additional correct natural OCR-miss refusals, and 21 additional
incorrect refusals on answerable questions.

## Primary comparison — seed ck-1296 → run-4a ck-100, clean-235

| | honest_abstain on clean-235 |
|---|---|
| seed run-3 ck-1296 | 212/235 = 0.9021 |
| **run-4a ck-100** | **217/235 = 0.9234** |

| both-yes | both-no | fixes b | breaks c | exact two-sided p |
|---:|---:|---:|---:|---:|
| 212 | 18 | **5** | **0** | **0.0625** |

All five fixes are the same transition, `false_grounding → honest_abstain`:
`1984_mfab`, `4527_mfab`, `44965_mfab`, `50720_mfab`, `55441_mfab`.
ck-100's residual on clean-235 is **18 false_grounding** and nothing else.

**The p-value is at its floor and cannot reach 0.05.** With 5 discordant pairs, the
smallest attainable two-sided exact binomial p is **0.0625** — the observed value. The
result is therefore *maximally* one-directional and still not significant at the
conventional threshold, because the discordant count is too small for significance to
be reachable at all. Reporting "p = 0.0625, not significant" without that fact would
misrepresent an underpowered test as an equivocal one.

What carries substance here is **c = 0**: no record the seed abstained on correctly
became a false grounding. That is the same structure as run-3's headline, at much
smaller magnitude — 5 of the 23 records of available headroom.

## Watch columns — §(f), reported not gated

**Over-refusal — false_abstain on the 679 answerable natural-test rows:**

| reference | false_abstain |
|---|---|
| run-1 ck-150 (control) | 0.034 |
| run-3 ck-1296 (the seed) | 0.115 (78/679) |
| **run-4a ck-100** | **0.1458 (99/679)** |
| run-2 ck-150 (shaped) | 0.177 (120/679) |

**Abstention retained — natural honest_abstain on the 77 natural-test abstain rows:**

| reference | honest_abstain |
|---|---|
| run-3 ck-1296 (the seed) | 71/77 = 0.922 |
| **run-4a ck-100** | **76/77 = 0.9870** |
| run-2 ck-150 (shaped) | 77/77 = 1.000 |

ck-100 additionally: valid_citation **580/679 = 0.8542** (seed: 601/679 = 0.8851),
malformed **0/756 = 0.0000**. On the answerable side every row is either
`valid_citation` or `false_abstain` — 580 + 99 = 679, no other bucket occurs.

## The shift is one-directional everywhere

Per-record movement against the seed. **Descriptive context, not a registered test.**

| population | n | gained | lost | net |
|---|---:|---:|---:|---:|
| manufactured abstain (clean-235), honest_abstain | 235 | **5** | **0** | +5 |
| natural abstain, honest_abstain | 77 | **5** | **0** | +5 |
| natural answerable, newly refused | 679 | **21** | **0** | +21 |

All 31 bucket changes moved toward refusal: five additional correct manufactured
refusals, five additional correct natural OCR-miss refusals, and 21 additional
incorrect refusals on answerable questions. The experiment does not establish an
internal threshold shift or whether evidence discrimination improved.

Read against the plan's framing: run 4a lands **between its seed and run 2 on both
axes** (over-refusal 0.115 → 0.1458 → 0.177; natural abstention 0.922 → 0.987 → 1.000).
That is what the same reward, applied from a better seed on construction-valid data,
buys: the same trade-off run 2 made, at a smaller step along the same line.

## What this settles, and what it does not

**Settles §(e)1 as registered.** RL from ck-1296 on the clean pool does still move the
policy, and it moves abstention up with no abstain-side regression. The gain is real
and small: 5 records manufactured, 5 natural, against 21 answerable rows given up.

**Does not establish superiority.** The primary test is at its p floor. The three
populations above are different denominators and must not be summed into one ledger;
whether +5/+5 is worth −21 is an operating-point judgement, not a measurement.

**Bears directly on 4b.** The over-refusal 4a introduces (78 → 99 answerable rows
refused, none recovered) is precisely the quantity `FALSE_ABSTAIN_BASE = −0.2` is
registered to price. 4a is now the comparator that isolates it: 4b differs from 4a in
that one constant alone.

**Carries the amendment's caveat forward.** Per the 2026-08-17 amendment, 4a's
`frac_reward_zero_std` of 0.650 aggregate / 0.594 abstain-side sits above run-1's
control 0.634. The amendment registered this gradient-supply caveat *in advance* for
a 4b null. Those gradient statistics and the observed bucket changes do not establish
the mechanism behind the refusal trade-off.

## Scoring validation

The pipeline reproduces **seven committed figures to the digit** before computing
anything new:

- clean-235 honest_abstain: seed ck-642 **68/235**, run-2 ck-150 **204/235**,
  run-3 ck-1296 **212/235** (`run3_test_results.md`).
- natural test, run-3 ck-1296: honest_abstain **71/77**, false_abstain **78/679**,
  valid_citation **601/679**, malformed **0/756**.

A mismatch on any of the seven aborts before a new number is produced. The
natural-test half of this validation was added specifically because the watch columns
come off that instrument and the clean-235 baselines do not exercise it.

Further guards:

- **clean-235 asserted**, from `exclusion_reason is None` on the committed
  `data/manufactured_test_abstains.jsonl` (250 rows, 15 excluded). The stale
  `in_clean_236` flag is absent from the file and was not used.
- **Join coverage asserted** — 250/250 and 756/756, with a missing record_id raising
  rather than joining via `.get()` to `None` and scoring as a parse failure.
- **Instrument identity verified**, reproducing run-3's finding independently: the
  Modal data volume's `manufactured_test_abstains.jsonl` has the same 250 record_ids
  and **zero mismatches** on `question`, `image_path`, `doc_id`, `ocr_menu`,
  `gold_answers`, `target_is_abstain`, `gold_localized`, `cited_ids`. It lacks only the
  audit-overlay keys (`exclusion_reason`, `gold_label_defect`, `survivor_pending`).
  Generation used the volume copy; scoring used the committed file.
- **The exact binomial is pinned** by `tests/test_exact_binomial.py` against all five
  committed p-values (`run3_test_results.md`: 8.97e-44 and 0.115;
  `run3_checkpoint_selection.md` §3c: 0.4049, 1.0000, 0.4244), plus symmetry,
  bounds, the b=c=0 case and monotonicity in imbalance.

  That test exists because of a real defect. The first implementation compared point
  masses with an **absolute** epsilon (`p/total <= obs + 1e-12`). It reproduced four of
  the five committed figures exactly, but at b=144, c=0 the observed mass is 2^-144,
  far below the epsilon, so the condition degenerated and swept in most of the
  distribution — returning **9.93e-13** against the true **2^-143 = 8.97e-44**. Wrong
  by 31 orders of magnitude, **in the direction that makes a result look less
  significant than it is**, and invisible to any test whose cases are all moderate. The
  fix compares exact integer binomial weights, so no float rounding enters.

  This did not affect the result above: at b=5, c=0 both the defective and the corrected
  implementation return **0.0625**, since the epsilon only distorts extreme tails. The
  defect was caught by validating against a committed extreme case before the primary
  number was computed, which is the reason that validation step exists.

  **The defect was never in the committed run-2/run-3 figures** — it was introduced in
  this session's fresh implementation and caught by them. Verified three ways: the
  committed 8.97e-44 equals the closed form for c=0, `2^-(b-1) = 2^-143`, to every digit
  quoted; it matches the independently-documented formulation at
  `mcnemar_seed_vs_grpo.md:42` (min(b,c) tail, doubled, capped), which agrees with the
  implementation here on all 3,600 (b,c) pairs over 0..59; and a second committed extreme
  case, b=136 / c=0 → 2.3e-41 = 2^-135, lands on the closed form too. The defective
  version returns 9.93e-13 at b=144 — 1.1e31 times too large to round to 8.97e-44, so
  "landed close by luck" is excluded.

## Generations

ck-100 natural-test and manufactured-test passes, Modal **L40S**, via `modal_gen_run3.py`
with `--run-dir` pointed at the run-4a run dir — the script unmodified, the same
`eval_runner.load_adapted_model` loader as every other comparison generation (base in
4-bit NF4 + PEFT adapter, greedy, `max_new_tokens 128`), which is what the paired design
requires. GRPO trained in bf16; generation precision is held constant across all
compared checkpoints regardless, exactly as for run-1/run-2/run-3.

Stop reasons **756/756 eos** and **250/250 eos** — no length-cap truncation; zero empty
completions. Prompt parity checked **3/3** on the smoke pass.

**Adapter verified before spending:** the Modal copy of
`grpo_run4a_seed42/checkpoint-100/adapter_model.safetensors` was downloaded and hashed
against the local copy — both `961554c27ad9d21e4f3df2dada5f1dda6b6632665886231936bbed029a5c0bd9`,
47,228,944 bytes. Under §4e.6 a Modal transfer has previously written correct-size,
wrong-content files, so generating from an unhashed adapter is not acceptable.

**Both outputs hash-verified at the source before pulling** (`run4_provisioning_runbook.md`
§4e.6): each file's sha256 was computed on the volume and compared to the pulled copy —
`gen_test_checkpoint-100.json` `d86d4233…5562b8` (151,718 B),
`gen_manufactured_checkpoint-100.json` `16af985c…a6e8ec` (50,558 B). Both match.

## Reproducing this document

```bash
python3 scripts/score_run4a_test.py --gen-dir data/checkpoints/eval/grpo_run4a_seed42
```

Baseline self-test alone (no ck-100 generations needed): omit `--gen-dir`.
