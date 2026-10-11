# Run-4b on the frozen test instruments — results (2026-08-20)

Executes the comparison specified in [`run4_analysis_plan.md`](run4_analysis_plan.md)
§(e)2 and both watch columns of §(f), for **ck-150**, the checkpoint selected in
[`run4b_checkpoint_selection.md`](run4b_checkpoint_selection.md). Scored with the
committed `scripts/eval_scorer.py` via `scripts/score_run4a_test.py`.

The frozen test instrument was opened **once** for this arm (plan §h). 4b differs from
4a in exactly one substantive config key, `false_abstain_base: −0.2`, verified by
diffing the two runs' own `grpo_run_config.json` files.

## The question, and the answer

> **Run-4b-best vs run-4a-best** — *what does the reprice buy?* Primary metric is the
> watch column (§f), with honest_abstain on clean-235 reported alongside so any
> abstention given up is visible next to the over-refusal recovered. — plan §(e)2

**The reprice moved over-refusal, measurably, and manufactured abstention did not
change.** Over-refusal falls **99/679 → 91/679** (paired exact two-sided **p = 0.0386**),
while honest_abstain on clean-235 is **identical at 217/235**. But natural abstention
**fell 76/77 → 74/77**, and both lost rows became **false grounding** — so the §(f)
pairing does not come out clean.

## §(e)2 PRIMARY — over-refusal, false_abstain on the 679 answerable natural-test rows

| | false_abstain |
|---|---|
| run-4a ck-100 | 99/679 = 0.1458 |
| **run-4b ck-150** | **91/679 = 0.1340** |

| recovered b | newly refused c | discordant | exact two-sided p |
|---:|---:|---:|---:|
| **10** | **2** | 12 | **0.0386** |

**All 10 recovered rows went to `valid_citation`**, not to another failure bucket —
`24505`, `38869`, `38876`, `38933`, `44139`, `58937`, `60498`, `63245`, `63246`, `6403`.
The 2 newly refused were `valid_citation → false_abstain`: `47823`, `61757`.

Net: **8 fewer wrongly-refused answerable rows**, and `valid_citation` rises
**580/679 → 588/679**.

At 12 discordant pairs the minimum attainable two-sided p is 0.000488, so **0.0386 is not
a floor artefact** — unlike 4a's §(e)1, which sat exactly at its floor. This is a real
movement in the metric the arm exists to move.

**Caveat carried with it:** n = 679, so **one row = 0.147 pp**. The effect is 8 net rows.

## §(f) WATCH COLUMNS — and the pairing does not come out clean

**Over-refusal — false_abstain on n=679:**

| reference | false_abstain |
|---|---|
| run-1 ck-150 (control) | 0.034 |
| run-3 ck-1296 (the seed) | 0.115 (78/679) |
| run-4a ck-100 | 0.1458 (99/679) |
| **run-4b ck-150** | **0.1340 (91/679)** |
| run-2 ck-150 (shaped) | 0.177 (120/679) |

**Abstention retained — natural honest_abstain on n=77:**

| reference | honest_abstain |
|---|---|
| run-3 ck-1296 (the seed) | 71/77 = 0.922 |
| run-4a ck-100 | 76/77 = 0.9870 |
| **run-4b ck-150** | **74/77 = 0.9610** |
| run-2 ck-150 (shaped) | 77/77 = 1.000 |

The plan states the test this pairing is for:

> *"the reprice is only interesting if false_abstain falls **without** natural
> honest_abstain falling with it. A drop in both is the reward buying refusal-averse
> behaviour wholesale rather than better discrimination."* — §(f)

**Both fell.** Over-refusal fell by 8 net rows; natural abstention fell by 2. The 2 lost
rows are `36850` and `47525`, and both moved **`honest_abstain → false_grounding`** — the
failure the plan's own harm ordering ranks worst among the recoverable ones, *"an
uncheckable lie"* (§c). 4b's natural abstain side is 74 honest_abstain / 3
false_grounding against 4a's 76 / 1.

The three populations, paired per record:

| population | n | 4a | 4b | improvements b | regressions c | exact paired p |
|---|---:|---|---|---:|---:|---:|
| manufactured abstention (clean-235) | 235 | 217 | 217 | 1 | 1 | 1.0 |
| natural abstention | 77 | 76 | 74 | 0 | 2 | 0.5 |
| answerable over-refusal | 679 | 99 | 91 | 10 | 2 | **0.0386** |

**The manufactured aggregate is unchanged, but the two arms' manufactured predictions are
not identical.** Both reach 217/235, and the pairing contains **one improvement**
(`22309_mfab`, `false_grounding → honest_abstain`) and **one regression** (`51702_mfab`,
`honest_abstain → false_grounding`) — two discordant rows, exact paired **p = 1.0**.

**An earlier draft of this section argued that a pure threshold displacement would push
abstention down on *every* abstain population, and that the flat manufactured aggregate
therefore ruled one out. That inference is wrong and is withdrawn.** A threshold or
operating-point shift need only affect rows near the relevant decision boundary; it
predicts nothing about rows far from it. An aggregate that does not move is therefore
consistent both with no shift and with a shift whose affected rows lie elsewhere. The
clean test subset excludes surviving text that supports the question; it retains 26
examples with matching text judged unrelated to the question. This does not measure
how close the model's decision on those rows sits to its own boundary.
Many may not lie near it; this instrument does not establish that, and the argument does
not depend on it either way.

**What the data supports.** Over-refusal fell measurably (p = 0.0386). Natural abstention
fell by two rows, both into false grounding, at p = 0.5. Manufactured abstention held in
aggregate with one row moving each way. **These observations cannot distinguish improved
discrimination from a shifted operating point, or from a combination of the two** — and
this experiment was not designed to identify that mechanism. Attributing the result to
either would be a claim the design cannot support.

## §(e)1 for completeness — run-4b ck-150 vs seed ck-1296, clean-235

Not the §(e)2 primary, reported because §(e)2 asks for clean-235 alongside.

| | honest_abstain on clean-235 |
|---|---|
| seed run-3 ck-1296 | 212/235 = 0.9021 |
| **run-4b ck-150** | **217/235 = 0.9234** |

| both-yes | both-no | fixes b | breaks c | exact two-sided p |
|---:|---:|---:|---:|---:|
| 211 | 17 | 6 | 1 | **0.125** |

4b reaches the **same 217/235 as 4a** by a slightly different route — 4a was b=5/c=0 from
the seed, 4b is b=6/c=1. The one regression is `51702_mfab`. One row = 0.426 pp here.

ck-150 additionally: **valid_citation 588/679 = 0.8660**, **malformed 0/756 = 0.0000**.

## What this settles, and what it does not

**Settles §(e)2 as specified.** −0.2 was registered with the risk that it *"may be too
weak to move the metric measurably"* (§c). It was not too weak: over-refusal moved by 8
net rows at p = 0.0386, and it moved in the intended direction, with recovered rows
landing in `valid_citation`.

**The §(f) pairing condition was not cleanly met.** The condition was that
false_abstain fall *without* natural abstention falling with it. Natural OCR-miss honest
refusal declined from 76/77 to 74/77 — 0 recoveries, 2 losses — and both losses became
false grounding. At n = 77 that decline is not statistically distinguishable (p = 0.5).
**Non-significance here is not evidence of no cost:** an instrument with 77 abstain rows,
where one row is 1.3 pp, has little power to resolve a two-row effect. The correct
statement is that the pairing condition was not cleanly met and that this instrument does
not establish whether a natural-transfer cost exists.

**Does not establish improved site-level verification.** The experiment does not show
whether repricing made the model better at checking *which row of the page* a matching
token sits in, the question raised in §(i). The 2 new false-groundings show an observed
cost of the changed refusal trade-off; they do not identify how the model made those
decisions.

**Concludes the arc.** Per the 2026-08-17 amendment, −0.2 stands and a result here is not
followed by a larger constant. Section §(i)'s LLM-judge adjudication of surviving sites
remains a possible extension for improving data quality.

**The gradient-supply caveat is not needed to explain this result**, but it remains on
record: 4b's `frac_reward_zero_std` was 0.620 aggregate / 0.5808 abstain-side (372 of 600
G=8 groups flat; 25 of 150 steps with every group flat). The amendment registered that
caveat for a *null*; this is not a null.

## Scoring validation

The pipeline reproduces **seven committed figures to the digit** before computing anything
new — clean-235 honest_abstain for seed ck-642 (**68/235**), run-2 ck-150 (**204/235**),
run-3 ck-1296 (**212/235**); and run-3 ck-1296 on natural test: honest_abstain **71/77**,
false_abstain **78/679**, valid_citation **601/679**, malformed **0/756**. A mismatch on
any of the seven aborts before a new number is produced.

Further guards:

- **clean-235 asserted** from `exclusion_reason is None` on the committed
  `data/manufactured_test_abstains.jsonl` (250 rows, 15 excluded).
- **Join coverage asserted** — 250/250 and 756/756, with a missing record_id raising
  rather than joining via `.get()` to `None` and scoring as a parse failure.
- **Instrument identity asserted BEFORE generation**, not after: the Modal data volume's
  `manufactured_test_abstains.jsonl` has the same 250 record_ids and **zero mismatches**
  on `question`, `image_path`, `doc_id`, `ocr_menu`, `gold_answers`, `target_is_abstain`,
  `gold_localized`, `cited_ids`. It is the **pre-audit build** — it lacks
  `exclusion_reason`, `gold_label_defect`, `survivor_pending`, so its file-level sha256
  differs from the committed file by design. Generation used the volume copy; scoring used
  the committed file. Same arrangement as run-3 and run-4a.
- **The exact binomial is pinned** by `tests/test_exact_binomial.py` against five
  committed p-values plus invariants.
- **The comparator is unchanged.** Re-scoring 4a through the generalised scorer reproduces
  its committed §(e)1 and §(f) numbers exactly (217/235, b=5/c=0, p=0.0625; 99/679;
  76/77), so the refactor did not move the arm 4b is compared against.

## Generations

ck-150 natural-test and manufactured-test passes, Modal **L40S**, via `modal_gen_run3.py`
unmodified with `--run-dir` pointed at the run-4b run dir — same
`eval_runner.load_adapted_model` loader as every other comparison generation (base in
4-bit NF4 + PEFT adapter, greedy, `max_new_tokens 128`).

Stop reasons **756/756 eos** and **250/250 eos**; zero empty completions. Prompt parity
checked **3/3** on the smoke pass.

**Adapter verified before spending:** ck-150 was mirrored to the Modal ckpts volume and
its sha256 computed **on the volume** — `cb810367…b06bc0`, 47,228,944 bytes — matching the
local copy. Under §4e.6 a Modal transfer has previously exited clean while writing
correct-size, wrong-content files, so a successful `put` is not evidence.

**Both outputs hash-verified at source before pulling:**
`gen_test_checkpoint-150.json` `44fbf22a…378f41` (151,613 B),
`gen_manufactured_checkpoint-150.json` `55026632…87e506` (50,506 B). Both match.

**Operator-reported, no repository artifact:** the manufactured pass was run twice. The
first attempt died at 200/250 when the local Modal client lost its heartbeat
(`App state is APP_STATE_STOPPED`) because the laptop slept mid-pass; the re-run used
`modal run --detach` and completed 250/250. Only the completed pass produced the artifact
scored here. This paragraph is session observation; the checkable record is the hashes
above.

## Reproducing this document

```bash
python3 scripts/score_run4a_test.py \
  --gen-dir data/checkpoints/eval/grpo_run4b_seed42 --ckpt 150 --label "run-4b ck-150" \
  --vs-gen-dir data/checkpoints/eval/grpo_run4a_seed42 --vs-ckpt 100 --vs-label "run-4a ck-100"
```

Baseline self-test alone (no generations needed): omit every argument.
