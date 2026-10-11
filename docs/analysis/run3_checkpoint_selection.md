# Run-3 checkpoint selection — ck-1296 (recorded 2026-08-10)

Selection of the run-3 balanced-SFT checkpoint to carry into the pre-registered
test-set comparisons of [`run3_analysis_plan.md`](run3_analysis_plan.md) §(c).
Everything here is val-only: the frozen test instrument was not opened for any
run-3 checkpoint before this document was written.

Run: `sft_v3_balanced_nodecoy` / `sft_B_seed42`, W&B `lqoekugp`, seed 42,
1,947 optimizer steps (3 epochs × ⌈5187/8⌉ = 649 steps/epoch), `--save-steps 324`.

## 1. The instrument, and what it can and cannot support

Scored with `scripts/score_val_curve.py` over the **natural validation split**
(`format_sft.load_split("validation", "B")`): **n = 753 = 46 natural `ocr_miss`
abstains + 707 answerable**.

Two limits, stated before the numbers:

- **n = 46 on the abstain side.** One row is 2.17 pp. The entire observed spread is
  three rows. This instrument ranks checkpoints; it does not establish differences
  between them.
- **The perception assumption is unvalidated precisely here.** On natural abstains
  `gold_localized=False` conflates "OCR missed visible text" with "the answer was
  never a literal string on the page" (see `docs/notes` field notes §2 reasoning and
  `docs/behavior_axes_reference.md`). Two of these 46 rows have a completely empty
  `ocr_menu`.

⚠ The lower-bound caveat recorded in [`../label_audit.md`](../label_audit.md) §5
("9 of 64 abstain rows recur, ≥3 hand-confirmed mislabelled") describes a
**different, manufactured-abstain validation instrument** — its row ids end `_mfab`
and `answer_recurs_in_menu` is a manufactured-only field. It does **not** apply to
the 46-row natural-abstain instrument used here.

## 2. Val curve — all seven checkpoints

`checkpoint-1947` is the trainer's end-of-training save, three optimizer steps past
the last `--save-steps` boundary; its adapter is byte-identical to the run-dir root
final adapter (`e39854306829c7637dbc5c4ae230575ca95ffa1f00ceab383922c1a18bfd428c`)
and differs from ck-1944. The pre-registered plan anticipated only the six
324-cadence saves, so ck-1947 is reported but was **not** admitted to the selection
pool; it would not have changed the outcome (42/46, mid-pack).

| ckpt | ~epoch | honest_abstain | = n/46 | false_grounding | false_abstain (n=707) | valid_citation | malformed | eval_loss |
|---|---|---|---|---|---|---|---|---|
| 324  | 0.50 | 0.891 | 41/46 | 0.109 | 0.089 | 0.909 | 0.001 | 0.097337 |
| 648  | 1.00 | **0.957** | **44/46** | 0.043 | 0.112 | 0.887 | 0.001 | **0.092343** |
| 972  | 1.50 | 0.891 | 41/46 | 0.109 | 0.106 | 0.894 | 0.000 | 0.096067 |
| **1296** | 2.00 | **0.957** | **44/46** | 0.043 | 0.106 | 0.894 | 0.000 | 0.095555 |
| 1620 | 2.50 | 0.913 | 42/46 | 0.087 | 0.100 | 0.900 | 0.000 | 0.111687 |
| 1944 | 3.00 | 0.913 | 42/46 | 0.087 | 0.100 | 0.900 | 0.000 | 0.108588 |
| _1947_ | 3.00 | _0.913_ | _42/46_ | _0.087_ | _0.100_ | _0.900_ | _0.000_ | _0.108741_ |

**The 1.5-epoch abstain peak does not replicate.** The prior series' peak sat at
ck-642 = 1.5 epochs; here 1.5 epochs is ck-972, tied for **lowest** honest_abstain
(41/46). Run-3's peaks fall at 1.0 and 2.0 epochs. This was the stated purpose of the
curve (`score_val_curve.py` docstring) and the answer is negative.

## 3. Discordance analysis, ck-648 vs ck-1296

Both lead at 44/46, so the tie was broken by a per-row diff on the same two committed
gen files. CPU only; no generation.

### 3a. Abstain side — 2 discordant rows of 46, one in each direction

| record | menu tokens | answer string in menu? | ck-648 | ck-1296 |
|---|---:|---|---|---|
| `55281` | 492 | no | honest_abstain | false_grounding → id 0 = `'4Tt'` |
| `57951` | 49 | no | false_grounding → id 42 = `'/id}!'` | honest_abstain |

- **`55281`** — *"What is written on the top left corner of the page?"*, gold `GTC`.
  Both checkpoints read the page correctly (`found_answer:"GTC"` from each). No menu
  token contains "GTC". ck-1296 cited **id 0** — first in reading order, i.e. the
  top-left token, exactly where the question points — whose OCR text is `'4Tt'`,
  Tesseract's mangling of GTC. A *spatially correct, textually failed* citation.
- **`57951`** — *"What is the page number?"*, gold `42`. Both read it correctly
  (`found_answer:"42"`). No menu token contains "42". ck-648 cited **id 42**, whose
  text is `'/id}!'` — i.e. it matched the answer string to the token **id number**
  rather than to any token text.

Context: only **3 of 46** abstain rows have the answer string present in the menu at
all, and **neither checkpoint false-grounds on any of them** (each has exactly 2
false-groundings, 0 with the string present, and they **share one** of the two). So
on this instrument there is no lure-driven false grounding to differentiate.

### 3b. Answerable side — 23 discordant rows of 707

Faces: *literal-string over-refusal* = declined although a gold answer variant is
literally present in the menu; *quirk-band over-refusal* = declined where the gold
cite span is not the literal answer string (the ~17–20 % frozen-localizer band);
*format break* = unparseable output.

| record | menu | ck-648 | ck-1296 | cite∩gold | face |
|---|---:|---|---|---|---|
| 37016 | 106 | valid_citation | false_abstain | 1/1 | literal-string over-refusal |
| 43634 | 50 | valid_citation | false_abstain | 0/1 | quirk-band over-refusal |
| 43734 | 160 | false_abstain | valid_citation | 0/3 | quirk-band over-refusal |
| 45142 | 48 | false_abstain | valid_citation | 0/5 | literal-string over-refusal |
| 46503 | 161 | false_abstain | valid_citation | 5/5 | literal-string over-refusal |
| 53730 | 235 | valid_citation | false_abstain | 3/3 | quirk-band over-refusal |
| 54341 | 70 | false_abstain | valid_citation | 0/2 | quirk-band over-refusal |
| 57087 | 238 | false_abstain | valid_citation | 2/2 | literal-string over-refusal |
| 5778 | 107 | false_abstain | valid_citation | 1/1 | literal-string over-refusal |
| 57784 | 600 | valid_citation | false_abstain | 1/1 | quirk-band over-refusal |
| 58673 | 374 | valid_citation | false_abstain | 1/1 | quirk-band over-refusal |
| 59468 | 211 | false_abstain | valid_citation | 0/1 | literal-string over-refusal |
| 60100 | 369 | false_abstain | valid_citation | 0/1 | quirk-band over-refusal |
| 61157 | 37 | false_abstain | valid_citation | 3/4 | quirk-band over-refusal |
| 63532 | 90 | false_abstain | valid_citation | 0/2 | quirk-band over-refusal |
| 64479 | 172 | valid_citation | false_abstain | 0/1 | quirk-band over-refusal |
| 64674 | 49 | valid_citation | false_abstain | 1/2 | quirk-band over-refusal |
| 64739 | 767 | false_abstain | valid_citation | 0/2 | literal-string over-refusal |
| 64741 | 767 | malformed | valid_citation | 2/3 | format break |
| 65115 | 139 | valid_citation | false_abstain | 0/1 | literal-string over-refusal |
| 65117 | 139 | valid_citation | false_abstain | 0/1 | literal-string over-refusal |
| 65399 | 323 | false_abstain | valid_citation | 0/4 | quirk-band over-refusal |
| 6991 | 244 | false_abstain | valid_citation | 5/6 | quirk-band over-refusal |

Tally: **13 quirk-band over-refusal, 9 literal-string over-refusal, 1 format break.**
Direction: ck-648 is the non-`valid_citation` side on **14**, ck-1296 on **9**.

Two negative controls:

- **No lookalike grabs among the discordances.** Every disagreement but one is
  over-refusal. Measured separately, rows scored `valid_citation` while citing **zero**
  gold tokens — a lookalike grab the bucket cascade does not penalise, since it routes
  on `gold_localized ∧ answerable_from_menu` without comparing cited ids to gold —
  run **55/707 (ck-648)** vs **52/707 (ck-1296)**. Indistinguishable.
- **No menu-density signal.** Over-refusals span 37→767-token menus in both faces.

### 3c. Exact tests

Two-sided binomial on discordant pairs (the form used in
[`mcnemar_seed_vs_grpo.md`](mcnemar_seed_vs_grpo.md), authoritative at small counts):

| partition | fixes (648→1296) | breaks | exact p |
|---|---:|---:|---:|
| answerable, `valid_citation` | 14 | 9 | **0.4049** |
| abstain, `honest_abstain` | 1 | 1 | 1.0000 |
| all 753 val rows | 15 | 10 | 0.4244 |

## 4. Decision

**ck-1296**, by this chain:

1. Pre-registered basis: val honest_abstain (`run3_analysis_plan.md` §(c), the
   abstain-side headline `score_val_curve.py` exists to produce).
2. That basis ties at **44/46** between ck-648 and ck-1296.
3. Tie broken by per-row discordance analysis (§3).
4. The two checkpoints' errors are **the same kind**: 22 of 23 answerable
   discordances are over-refusal, the abstain side is symmetric (1 unforced junk-token
   grab each, no lure present, perception intact in both), and bucket-invisible
   lookalike grabbing is 55 vs 52 of 707. Neither concentrates in a mode the other
   avoids.
5. The margin is **not significant** (p = 0.40 answerable, 0.42 overall). The choice
   is **documented-arbitrary**; ck-1296 is taken on **magnitude** — five fewer
   over-refusals and one fewer format break.

**Recorded counter-indication:** eval loss mildly favours ck-648
(**0.092343** vs **0.095555**; ck-648 is also the global eval-loss minimum, and loss
rises after ck-1296 — 0.1117 at 1620, 0.1086 at 1944 — consistent with epoch-3
overfitting). The **pre-registered selection rule is val honest_abstain, not eval
loss**, and it was followed as registered. This is noted so the alternative is
visible, not to reopen it.

**Bearing on the GRPO stage:** this instrument cannot inform that contrast. The axis
the staged reward targets is lure resistance, and only 3 of 46 val abstain rows
contain a lure at all, with neither checkpoint failing on any. That comparison lives
on the manufactured test instrument, where decoys exist by construction. The one
GRPO-relevant observation is that the shared dominant error mode **is** over-refusal —
exactly what `run3_analysis_plan.md` §(c)4 earmarks for negative pricing in run-4.

## 5. Prior-series cadence (verification of a derivation)

The prior SFT series `sft_v3_32g` trained **3 epochs at a 214-step half-epoch
cadence**, verified against its own config rather than inferred from checkpoint
arithmetic: `num_train_epochs: 3.0`, `save_steps: 214`, `eval_steps: 214`
(see [`../run_configs/`](../run_configs/)). Mode-B train = 3,423 records →
⌈3423/8⌉ = **428 steps/epoch** × 3 = **1,284 total**; 1284/214 = 6 checkpoints,
matching the 214/428/642/856/1070/1284 series in [`../PROVENANCE.md`](../PROVENANCE.md).
That series carried no `extra_abstains` key — plain mode-B, unlike run-3's balanced mix.

## 6. Reproducing this document

```bash
uv run python scripts/score_val_curve.py \
  --gens data/checkpoints/eval/sft_v3_balanced_nodecoy
```
Per-row discordance used `scripts/eval_scorer.classify`,
`scripts/format_sft.load_split`, and the committed recurrence rule
`scripts/build_manufactured_abstains.answer_recurs` (imported, not reimplemented).
Gen sha256s are in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).
