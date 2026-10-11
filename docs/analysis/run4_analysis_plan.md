# Run-4 — local analysis plan (two arms: balanced-seed control, then the reprice)

Recorded 2026-08-12 in this repository, before either run-4 arm is launched. This is a
**local** analysis plan under version control, not an externally timestamped
preregistration; its force comes from being committed before the runs, and from the
commit history that shows it. It records both arms, the reward
change, the training mechanics, and every comparison, so nothing below is chosen after
seeing an outcome. Launch is a separate budget decision and is **not** authorised by this
document.

**Completion note.** Run-4a and run-4b are finished; their results are in
[`run4a_test_results.md`](run4a_test_results.md) and
[`run4b_test_results.md`](run4b_test_results.md). The dated plan and amendments
below remain historical records; proposed extensions are possibilities,
not current commitments.

Antecedent: [`run3_test_results.md`](run3_test_results.md) met the fork condition in
[`run3_analysis_plan.md`](run3_analysis_plan.md) §(c)4 — run-3 ck-1296 refuses **0.115**
of the 679 answerable natural-test rows against the control's **0.034**.

## (a) Why two arms and not one

Run 4 changes **two** things at once relative to run 2: the seed (ck-642 → ck-1296) and
the training pool (decoy rows removed). Repricing `false_abstain` on top of that would
confound three variables, and no comparison could attribute the result.

- **Run-4a — the balanced-seed control.** GRPO from ck-1296 on the rebuilt pool with the
  run-2 reward **byte-identical** (`honest_abstain` +0.5, `false_grounding` −0.6,
  `false_abstain` 0.0, `fabrication` −0.5, flat −1.0, found-answer bonus +0.3). Isolates
  *new seed + clean data* from *new reward*. Answers: does RL still add anything once the
  seed already abstains well and manufactured examples are selected by the string-matching rule?
- **Run-4b — the reprice.** Identical to 4a in every respect except
  `FALSE_ABSTAIN_BASE` 0.0 → negative. Isolates the reprice alone.

Run-1 and run-2 changed two reward terms together: the correct-refusal bonus and the
false-grounding penalty. Run-4a and run-4b instead change only the incorrect-refusal
penalty. These comparisons do not by themselves establish why every individual
response changed.

## (b) Training pool (must match the manifest at launch)

`data/grpo_pool_v2_recursfalse.jsonl` + `.manifest.json`. Manufactured abstains restricted
to `answer_recurs_in_menu == False` — the same restriction run-3's SFT mix used, for the
same reason (`run3_analysis_plan.md` §(a): ~32% of recurs-True rows are estimated
mislabelled). The restriction means no accepted-answer string remains under the
implemented string-matching rule; it does not prove semantic absence of support.

| | original pool (runs 1–2) | v2 pool (run 4) |
|---|---:|---:|
| total | 5,000 | 4,410 |
| abstain / answerable | 2,000 / 3,000 | 1,764 / 2,646 |
| abstain fraction | 0.4000 | 0.4000 |
| abstain half: recurs-False | 1,623 | **1,764** |
| abstain half: recurs-True (decoy) | **377** | **0** |
| abstain strata | multi_unique 808, single_unique 815, multi_decoy 162, single_decoy 215 | multi_unique 868, single_unique 896 |

**The pool shrinks but the string-checked subset grows.** 377 decoy rows leave (18.9% of the
abstain half), yet the count of rows without accepted-answer text rises **1,623 → 1,764**,
because the v2 pool takes every available recurs-False row instead of sampling 2,000 from
a mixed set. Run 4 therefore trains on *more* examples satisfying this text-presence rule
than runs 1–2 did, not fewer. The 40/60 ratio is preserved exactly by scaling the
answerable half.

Config change required at launch: point `TRAIN_ABSTAINS` at the recurs-False rows and set
`pool_abstain=1764`, `pool_answerable=2646` (`grpo/train_grpo.py`). The original pool is a
runtime construction and remains reproducible at seed 42 — it is not overwritten.

## (c) The reward change, and the value

**Proposed: `FALSE_ABSTAIN_BASE = 0.0 → −0.2`.** Endorsed; the reasoning is below, with
the one quantification worth having in advance.

The justification is an ordering, not a feeling. Harm ranks: over-refusal is the mildest
failure (the user gets nothing, but is not misled), fabrication is worse (a checkable lie),
false grounding is worst of the recoverable ones (an *un*checkable lie), and incoherent
output is unusable. −0.2 preserves that ordering with clear margins:

| |false_abstain| | < | |fabrication| | < | |false_grounding| | < | flat penalties |
|---|---|---|---|---|---|---|
| 0.2 | | 0.5 | | 0.6 | | 1.0 |

At 0.2 the reprice is **one third** of `false_grounding`, so refusing wrongly never becomes
cheaper to fix than false grounding is to commit — the constraint that motivated the prior,
and it holds with room to spare.

**Illustrative reward calculation, not a measured model threshold.** If the found-answer
bonus is equal for answering and refusing, it cancels. Treating the observed valid-citation
F1 ≈ 0.87 as a fixed payoff for answering correctly, and assuming a belief *p* that the
record should be refused, gives these simplified expected-reward thresholds:

- at `false_abstain = 0.0` — refuse when *p* ≳ **0.442**
- at `false_abstain = −0.2` — refuse when *p* ≳ **0.493**

In this simplified calculation, the illustrated threshold changes by roughly
**5 percentage points of belief**. This is not a measurement of the model's belief or
decision threshold, and its responses need not follow this calculation.

**The risk to state in advance: −0.2 may be too weak to move the metric measurably.**
Whether the penalty changes false_abstain cannot be inferred from the illustrated
threshold difference; the run must measure it. **This is specified in advance as
an acceptable outcome, not a failure**: "the reprice at −0.2 did not measurably move
over-refusal" is a real result about the reward's leverage. If a stronger value is tried
afterwards it is a **new, separately specified arm** — not a re-run of 4b with a tuned
constant, which would turn a specified test into a search.

## (d) Training mechanics

Held identical across both arms except the one constant.

- **Total steps: 150**, matching run 2 so the arms are comparable to it and to each other.
- **Checkpoint + eval every 10–15 steps** (run 2 used a coarser cadence). This is the
  change that makes the reward-vs-step curve legible and gives checkpoint selection more
  than a handful of candidates.
- **TRL `GRPOTrainer` reward logging confirmed ON before launch** — verified in the smoke
  run, not assumed. Per-bucket reward components must appear in the logs, not just the
  scalar mean.
- **One W&B run per arm**, named `grpo_run4a_seed42` / `grpo_run4b_seed42`.
- **Required deliverable: reward-vs-step plots** — mean group reward, plus the per-bucket
  components on their own axes. Plotted **separately from any SFT figure**: SFT curves are
  cross-entropy against a fixed target and GRPO curves are reward against a moving policy,
  and putting them on shared axes would imply a comparability that does not exist.

## (e) Pre-specified comparisons

Instrument: the frozen manufactured test set, **clean-235** throughout
(`exclusion_reason is None`; never the stale `in_clean_236` flag). Paired exact two-sided
binomial on discordant pairs, the form used in
[`mcnemar_seed_vs_grpo.md`](mcnemar_seed_vs_grpo.md).

1. **Run-4a-best vs seed ck-1296** — *does RL still add on the right data?* honest_abstain
   on clean-235. Run-3 ck-1296 is at 212/235 = 0.902, so the headroom is 23 records; a
   null result here is informative and is recorded as such in advance.
2. **Run-4b-best vs run-4a-best** — *what does the reprice buy?* Primary metric is the
   watch column (§f), with honest_abstain on clean-235 reported alongside so any abstention
   given up is visible next to the over-refusal recovered.

Secondary context, not primary: 4a and 4b against run-2 ck-150. These cross the seed and
pool change together and are reported as context only.

## (f) Watch columns

Not gates. Both are reported for every selected checkpoint, with all references labelled.

**Over-refusal — false_abstain on the 679 answerable natural-test rows:**

| reference | false_abstain |
|---|---|
| run-1 ck-150 (control) | 0.034 |
| run-3 ck-1296 (the seed for run 4) | 0.115 |
| run-2 ck-150 (shaped) | 0.177 |

**Abstention retained — natural honest_abstain on the 77 natural-test abstain rows:**
run-2 ck-150's **77/77 = 1.000** is the bar; run-3 ck-1296 is at 71/77 = 0.922
(`# source: grpo_finals_natural_test.md`, `run3_test_results.md`).

The pairing is the point: the reprice is only interesting if false_abstain falls *without*
natural honest_abstain falling with it. A drop in both is the reward buying refusal-averse
behaviour wholesale rather than better discrimination.

## (g) Checkpoint selection basis — fixed in advance

**Highest honest_abstain on the GRPO val instrument**, ties broken by lower false_abstain
on the same instrument, ties then broken by earlier step. This is the run-3 rule with an
explicit second key, added because run-3's selection landed on a tie that the earlier
rule could not resolve (`run3_checkpoint_selection.md`).

### The selection instrument is rebuilt recurs-False — decided 2026-08-12, in advance

**Both arms select checkpoints on a val instrument restricted to
`answer_recurs_in_menu == False`**, built under the same rule as the v2 training pool and
fixed before launch.

The reason is the old instrument's composition. `grpo/train_grpo.build_val_instrument`
draws 64 manufactured val abstains + 48 answerable, and **9 of those 64 abstain rows are
decoy-class with unaudited labels** (`# source: docs/label_audit.md §5`) — the same class
that `run3_analysis_plan.md` §(a) estimates is ~32% mislabelled. Selecting checkpoints on
that instrument ranks them partly on rows whose gold is suspect, and the argument for
cleaning the training pool applies with equal force to the instrument that picks the
winner. Consistency between what a run learns from and what it is judged on is worth more
here than curve comparability with runs 1–2, which used a different seed and a different
pool regardless.

Build rule: same seed (`cfg.seed + 1`, the distinct stream), same answerable half, abstain
half drawn only from `manufactured_val_abstains.jsonl` rows with
`answer_recurs_in_menu == False`. **Built and persisted 2026-08-12** as
`data/grpo_val_instrument_v2_recursfalse.jsonl` + `.manifest.json`.

| | original (runs 1–2) | v2 (run 4) |
|---|---:|---:|
| abstain / answerable | 64 / 48 | 64 / 48 |
| decoy-class abstain rows | **9** | **0** |
| urn the abstain half is drawn from | 466 | 379 |

The replay of the original instrument at the same seed returns **exactly 9 decoy rows**,
independently reproducing the documented 9/64 — the caveat and the instrument agree.

**Consequence for the caveat — corrected.** The rebuilt instrument no longer carries the
9/64 lower-bound caveat, because the rows that caused it are gone. An earlier draft of this
section said it "gains a smaller denominator"; **that is wrong and is corrected here**. The
denominator is unchanged at **64** — 64 clean rows are comfortably drawable from the 379
available. What shrinks is the *urn* (466 → 379), not the instrument. Still do not compare
its honest_abstain to runs 1–2's val curve, which was measured on the unrestricted
instrument.

**The qualitative probe keeps its lure case — specified.** `pick_probe_records` prefers a
decoy row for its every-N-step dump so drift on lure behaviour stays observable, and a
recurs-False instrument offers none. **The probe is observation only, never scored, and its
decoy row is therefore drawn from the decoy-class val abstains held *outside* the scored
instrument** (source pool: `data/manufactured_val_abstains.jsonl` rows with
`answer_recurs_in_menu == True`, excluding any record already in the instrument). Selection
stays judged on string-filtered examples; the dump keeps the lure case. Because probes are never
scored, a probe row from outside the instrument cannot influence checkpoint selection.

**A checkpoint with no val score cannot be selected. Report it; never rescore it.**
The val curve is produced *in-process*, by `run_val_eval` against the live model held in
memory at that step. `_run_eval` catches its own failures and logs
`[val@N] FAILED (continuing)` so the run survives — which means a checkpoint can exist on
disk with no corresponding `val_curve.jsonl` row.

Such a checkpoint is **ineligible for selection** and is reported as unscored. It must
**not** be rescored afterwards: a post-hoc pass reloads the adapter through a different
path (different loader, precision, and generation entry point) than the in-run eval, so its
number is not commensurable with the points around it. Mixing the two would silently
corrupt the very curve selection reads. If enough checkpoints go unscored that selection is
compromised, that is a reportable run failure — not something to patch by rescoring.

Two distinct val instruments remain in the project and must not share a caveat:

- The **GRPO val instrument** (manufactured; rebuilt as above for run 4). The 9/64 caveat
  belongs to its *pre-rebuild* form and to any run-1/run-2 number quoted from it.
- The **natural validation split** (n=753, 46 abstain) carries a different caveat entirely:
  n=46, one row ≈ 2.2pp (`# source: docs/label_audit.md` §5 scope addendum). Never attach
  9/64 to it.

## (h) Test-set discipline

The frozen test instrument is opened **once per arm**, for the pre-designated comparison in
§(e) only. Not for checkpoint selection, not for exploration, not for a second look after a
disappointing result. Manufactured-test and natural-test generations for the selected
checkpoint of each arm are produced in one pass and committed with sha256 in
[`ARTIFACTS.sha256`](ARTIFACTS.sha256).

## Amendment — smoke gate [3] waived, and a new monitored quantity (2026-08-15)

Recorded after the run-4 smoke on the A100, before 4a launches. This amends §(d); nothing
else in the plan changes.

**The trainer's smoke gate [3] ("abstain groups mixed-bucket") FAILED and is waived for
run 4, on measured grounds.** The gate samples **G=8 completions on exactly two abstain
probe records** and passes only if at least one group yields mixed buckets. Both collapsed:
the decoy `5419_mfab` 8/8 `false_grounding`, the clean `1958_mfab` 8/8 `honest_abstain`.

That gate was calibrated for the run-1/run-2 era, whose seed (ck-642) sat at 0.289
manufactured abstention — an uncertain policy whose abstain groups varied by default.
ck-1296 is at 0.902. Determinism on two individual high-confidence rows is the *expected*
behaviour of a strong seed and says nothing about pool-level gradient supply, which is the
quantity GRPO actually depends on: a group whose G completions all score alike yields zero
group-relative advantage.

Measured instead of assumed, from `frac_reward_zero_std` in each run's `reward_log.jsonl`:

| run | seed | frac_reward_zero_std (mean) | reward_std (mean) | outcome |
|---|---|---:|---:|---|
| run-2 (shaped) | ck-642 | **0.477** | 0.184 | b=136 / c=0, the headline result |
| run-1 (control) | ck-642 | 0.634 | 0.092 | moved nothing, by design |
| **run-4 smoke** | **ck-1296** | **0.479** | 0.154 | — |

Run 4's flat-group fraction is **0.479 against the successful run's 0.477** — indistinguishable,
and well clear of the control's 0.634. Mean reward over the smoke rose **0.722 → 1.034**.
Roughly half of each optimizer step carrying no gradient is simply how this setup has always
run, including when it produced the result the project rests on.

**`frac_reward_zero_std` becomes a monitored quantity for both arms**, reported in the
run write-up **split by abstain and answerable side** (the smoke's aggregate cannot
distinguish them, and the abstain side is the axis the shaping targets). If the abstain-side
fraction runs materially above run-2's aggregate 0.477 for a sustained stretch, that is a
reportable finding about RL headroom on an already-strong seed — not a reason to intervene
mid-run.

**The sampling configuration is explicitly UNCHANGED from run 2**: `temperature=1.0`,
`top_p=1.0`, `num_generations=8`. Raising temperature or G would manufacture within-group
variance and might well "fix" gate [3] — and would confound the comparison this plan exists
to make, since 4a's whole purpose is isolating the *seed and pool* change against run 2's
reward. **A higher-temperature or larger-G configuration would be a new specified arm, not
an amendment to this one.**

Two defects found by the same smoke and fixed before launch, recorded for provenance:
`--save-steps` did not exist as a CLI flag (default 50 would have produced 3 checkpoints
over 150 steps, not the 10–15 §(d) requires and §(g) assumes); and the first smoke was
truncated by a `timeout` whose non-zero status was masked by a pipe, so it reported success
having written no checkpoint at all. The save path is now verified.

## Amendment — interpreting a 4b null, and the end of the arc (2026-08-17)

Recorded after run-4a completed and its checkpoint was selected
([`run4a_checkpoint_selection.md`](run4a_checkpoint_selection.md)), **before 4b
launches**, so neither line below can be chosen after seeing 4b's outcome.

**1. A 4b null may reflect gradient supply, not the magnitude of the constant.** Run-4a
measured `frac_reward_zero_std` at **0.650 aggregate / 0.594 abstain-side** — above even
run-1's control value of 0.634, against run-2's 0.477 — and its six top checkpoints are
bucket-identical on all 112 val rows while their adapters differ. A reward constant
cannot create within-group variance that is not there. If 4b shows no measurable
movement in `false_abstain`, "−0.2 is too weak" and "there was too little group-relative
signal to act on" are **both live explanations**, and the flat-group fractions reported
per §(d) are what distinguishes them. This does not retract §(c)'s registration of a
null as an acceptable outcome; it constrains what that null licenses anyone to conclude.

**2. Run 4 concludes this arc.** −0.2 stands as specified. A null is a result about the
reward's leverage and **will not be followed by a larger constant** — doing so would
convert a specified test into a search, which §(c) already forbids. The
option proposed at the time was §(i)'s LLM-judge review of surviving sites: a
**data-quality** route, not a reward-tuning one. It remains a possible extension,
not a current commitment.

## Amendment — §(d)'s W&B line recorded as UNMET, both arms (2026-08-17)

§(d) registers *"One W&B run per arm, named `grpo_run4a_seed42` / `grpo_run4b_seed42`."*
**Neither arm produces one, and this records that symmetrically rather than quietly.**

`grpo/train_grpo.py` set `report_to="none"` — *"jsonl telemetry instead of W&B (robust
unattended)"* — so **run 4a produced no W&B run**; there is no `wandb/` directory in its
run dir. The substance §(d) was reaching for (per-bucket reward telemetry, legible
reward-vs-step curves) was delivered by `reward_log.jsonl`, `reward_members.jsonl`,
`step_times.jsonl` and `val_curve.jsonl`, and by
[`../figures/run4a_reward_vs_step.png`](../figures/run4a_reward_vs_step.png). The W&B
line itself was not honoured.

**4b keeps it unmet, deliberately, on two grounds.** Enabling W&B for 4b's long run
would make the arms differ in a second way while *still* leaving §(d) unmet for 4a — so
it buys no compliance and costs comparability. And a W&B auth or network stall is a new
failure mode on a 12.4-hour run that cannot be repeated within budget.

A `--wandb` flag now exists, **default off**. It is used for the **smoke only**, where a
human is watching and a failure costs twenty minutes. Both arms' long runs are on jsonl.

Recorded as a deviation from the pre-registration, not as a silent omission: §(d)'s W&B
requirement is **unmet for 4a and 4b alike**, and the arms remain symmetric in this
respect, which is what the comparison actually depends on.

## (i) What run 4 cannot settle

Run 4 compares refusal trade-offs with those in `run3_test_results.md`, but it does not
establish that cited OCR text actually supports the question. The worked lure examples
show that a model can cite matching answer text at an unrelated location
([`multi_occurrence_audit.md`](multi_occurrence_audit.md),
[`failure_analysis.md`](failure_analysis.md)). Reviewing surviving locations with an
LLM judge was proposed as a data-quality option in `run3_analysis_plan.md` §(e);
it is a possible extension, not a current commitment.
