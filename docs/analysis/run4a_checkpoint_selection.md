# Run-4a checkpoint selection — ck-100 (recorded 2026-08-16)

Selection of the run-4a (balanced-seed control) checkpoint to carry into the
pre-registered test comparison of [`run4_analysis_plan.md`](run4_analysis_plan.md)
§(e)1. Everything here is **val-only**: the frozen test instrument was not opened
for any run-4a checkpoint before this document was written, and this document does
not authorise opening it.

Run: `grpo_run4a_seed42`, seed 42, GRPO from run-3 ck-1296 on
`data/grpo_pool_v2_recursfalse.jsonl`, 150 steps, `--save-steps 10`, reward
byte-identical to run 2. Config: `grpo_run_config.json` in the run dir.

## 1. The instrument, and what it can and cannot support

Scored **in-process** by `run_val_eval` at every 10th step against
`data/grpo_val_instrument_v2_recursfalse.jsonl`
(sha256 `18d28e6ec912a8cdfb4aa28f103e715f15fba3c710a3527b0c1d6890e2c0041b`, matching
the value its own manifest records):
**n = 112 = 64 manufactured abstains + 48 answerable**, all abstain rows
`answer_recurs_in_menu == False`.

Four limits, stated before the numbers:

- **n = 64 on the abstain side. One row is 1.6 pp.** The entire spread across all 15
  checkpoints is five rows (55→60). This instrument ranks checkpoints; it does not
  establish differences between them.
- **n = 48 on the answerable side. One row is 2.1 pp.** The entire observed spread is
  one row (1/48 → 2/48).
- **No 9/64 caveat.** This is the rebuilt v2 instrument; plan §(g) removed the
  decoy-class rows that caused it. Verified here, not assumed: all 64 abstain rows
  carry `recurs=false` in the eval dumps, and the instrument file's own strata are
  `multi_unique 40 / single_unique 24`, zero decoy. honest_abstain on this instrument
  is **not** a lower bound.
- **Do not compare these numbers to the run-1 / run-2 val curves.** Those were measured
  on the unrestricted 64-abstain instrument (9 decoy rows, different urn). No such
  comparison appears in this document.

There is **no step-0 baseline on this instrument**, so this document cannot say what RL
added over seed ck-1296. That is exactly the §(e)1 comparison, and it lives on
clean-235.

## 2. Val curve — all 15 checkpoints

Every row n=112 (64 abstain / 48 answerable); no checkpoint is unscored, so §(g)'s
ineligibility clause is not invoked and nothing was rescored.

| step | honest_abstain | = n/64 | false_grounding | false_abstain | = n/48 | valid_citation | malformed | mean_reward_abstain |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.8594 | 55/64 | 0.1406 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6313 |
| 20 | 0.8906 | 57/64 | 0.1094 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6656 |
| 30 | 0.8750 | 56/64 | 0.1250 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6484 |
| 40 | 0.8906 | 57/64 | 0.1094 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6656 |
| 50 | 0.8906 | 57/64 | 0.1094 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6656 |
| 60 | 0.9062 | 58/64 | 0.0938 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6875 |
| 70 | 0.8906 | 57/64 | 0.1094 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6656 |
| 80 | 0.9062 | 58/64 | 0.0938 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6828 |
| 90 | 0.9219 | 59/64 | 0.0781 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7000 |
| **100** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7219 |
| **110** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7219 |
| **120** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7219 |
| **130** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7219 |
| **140** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7219 |
| **150** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7219 |

`final_adapter` is **byte-identical to checkpoint-150** — verified by sha256, both
`da845c07…f083db`. It is one candidate, not two, and is not listed separately.

Malformed is **0/112 at every one of the 15 steps** (0/1680 row-evaluations), and
`stop_reason` is `eos` for 112/112 at every tied step — no length-cap truncation.

## 3. The tie, and how far it goes

honest_abstain peaks at **60/64 = 0.9375** and **six checkpoints reach it**: steps 100,
110, 120, 130, 140, 150. Applying §(g) in order:

1. **key 1 — max honest_abstain:** 0.9375 → {100, 110, 120, 130, 140, 150}. Six-way tie.
2. **key 2 — min false_abstain on the same instrument:** all six sit at **2/48 =
   0.0417**. Tie unbroken.
3. **key 3 — earliest step:** → **ck-100**.

The tie is stronger than equal aggregates. Diffing the per-row dumps, **all six
checkpoints assign the identical bucket to all 112 rows** — 0 rows differ between
ck-100 and each of ck-110/120/130/140/150. This is not six models that happen to score
alike; on this instrument they are behaviourally indistinguishable row for row.

They are **not** the same model. The 15 checkpoint adapters produce 15 distinct sha256
values (16 files, one collision, that being ck-150 = final_adapter). Weights kept moving
through step 150; the instrument stopped being able to see it after step 100.

Underneath the identical buckets, generations do drift slightly — **4 of 112 rows**
differ in raw text across the six, all cosmetically or within-bucket:

| record | kind | bucket (all six) | what varies |
|---|---|---|---|
| `61226_mfab` | abstain | honest_abstain | `PHILADELPHIA` vs `Philadelphia` |
| `25386` | answerable | valid_citation | `three` vs `Three` |
| `32543_mfab` | abstain | false_grounding | cites ids [155,156,157,…] vs [98,99,100,…] |
| `2211` | answerable | valid_citation | cites [42,43,…] vs [44,45,…] — reward 1.05 vs 1.3 |

`found_answer_correct` is unchanged on every row of all six.

## 4. Decision

**ck-100**, by this chain:

1. Pre-registered basis (§(g)): highest honest_abstain on the v2 recurs-False val
   instrument. That maxes at 60/64.
2. Six checkpoints tie there; key 2 (false_abstain) leaves all six tied at 2/48.
3. Key 3 — earliest step — resolves to **ck-100**, and the registered rule is followed
   as written.

**What this choice is and is not.** It is the registered rule applied to a tie the
instrument cannot break. It is **not** a finding that ck-100 is the best of the six: on
112 val rows they are row-for-row identical, so the instrument is silent on which
generalises best, and their weights genuinely differ. The tie-break is a
deterministic convention, chosen in advance precisely so this situation is not
resolved after seeing outcomes. Any of the six could behave differently on test;
that is a limitation of the selection, not a hidden reservation about ck-100.

The supporting argument for the earliest-step convention, stated for the record: at
identical measured behaviour, 50 fewer optimizer steps is less reward-directed drift
away from the seed for the same val result.

**Recorded counter-indication:** summed reward over the 112 val rows mildly favours
ck-120 and ck-150 (**102.4968** vs **102.2468** for the other four). The entire
difference is **one row** — `2211`, where the citation span scores 1.3 instead of 1.05.
The registered selection key is honest_abstain, not summed reward, and it was followed
as registered. Noted so the alternative is visible, not to reopen it.

## 5. Monitored quantity — `frac_reward_zero_std`, split by side

Required by the plan's 2026-08-15 amendment, reported split by abstain and answerable
side. Computed from `reward_members.jsonl` (**4,800 rows = 150 steps × 32 completions**)
by grouping into the **600 groups** of G=8 that GRPO actually forms, then taking the
fraction of groups with zero reward spread.

| side | groups | frac_reward_zero_std |
|---|---:|---:|
| abstain | 136/229 | **0.5939** |
| answerable | 254/371 | 0.6846 |
| **aggregate** | 390/600 | **0.6500** |

The aggregate **0.6500 reproduces the trainer's own logged mean over 150 steps to four
decimals**, computed by a completely separate path (per-member regrouping vs TRL's
in-step statistic). Mean `reward_std` over the run is **0.1144**.

**This triggers the amendment's pre-registered reportable finding.** Against the
references the amendment tabulates:

| run | seed | frac_reward_zero_std | reward_std |
|---|---|---:|---:|
| run-2 (shaped, the headline result) | ck-642 | 0.477 | 0.184 |
| run-1 (control, moved nothing by design) | ck-642 | 0.634 | 0.092 |
| run-4 smoke | ck-1296 | 0.479 | 0.154 |
| **run-4a (actual, this run)** | **ck-1296** | **0.650** | **0.1144** |

Run-4a's abstain-side fraction (0.5939) runs **materially above run-2's aggregate
0.477**, and sustained — which the amendment designates in advance as *"a reportable
finding about RL headroom on an already-strong seed."* The aggregate comparison, which
is like-for-like in composition, is starker still: run-4a's **0.650 exceeds even run-1's
0.634**, the control arm that moved nothing by design. Mean `reward_std` likewise sits
nearer the control (0.092) than the successful run (0.184).

**The smoke did not predict the full run.** The smoke measured 0.479 — indistinguishable
from run-2's 0.477, and the basis on which gate [3] was waived. Over 150 real steps the
quantity came in at 0.650. The waiver's reasoning was sound on the evidence available;
the evidence available was a short sample and it did not hold up. This is recorded as a
lesson about what a smoke can establish, not as an error in the waiver.

Consistent with all of the above, the val curve moved a total of five abstain rows
(55/64 → 60/64) and then flatlined completely from step 100. Mean group reward rose
0.8499 (first 10 steps) → 0.9784 (last 10).

**This does not modify 4b.** The plan states a high flat-group fraction is *"not a reason
to intervene mid-run"*, and §(d)/the amendment fix the sampling configuration
(`temperature=1.0`, `top_p=1.0`, `num_generations=8`) as **unchanged** — changing them
would be a new registered arm, not an amendment. 4b runs as registered.

## 6. Integrity checks run before any number above was written

Exit codes were not accepted as evidence for any of these.

- **16 adapter files hashed** (15 checkpoints + `final_adapter`): all exactly
  47,228,944 bytes, **15 distinct sha256 values**, the one collision being
  ck-150 ≡ `final_adapter`. **Zero zero-byte files** anywhere in the run dir.
- **The committed val curve was not trusted.** All 15 rows of `val_curve.jsonl` were
  recomputed from the per-row `eval_step*.jsonl` dumps; honest_abstain, false_abstain
  and both N values **match at all 15 steps**.
- **Join coverage asserted, 112/112 at every one of the 15 steps**, against the
  registered instrument file, with **0 kind mismatches** — closing the `.get()`-joins-to-
  `None` failure mode recorded in `run3_test_results.md`. (That failure mode is not
  hypothetical: the first version of this check keyed on a `target_is_abstain` field the
  instrument index does not carry, and silently reported a plausible 48/112 before the
  join key was corrected to the instrument's own `kind` field.)
- **Instrument identity across steps:** the same 112 record_ids in the same order at
  every step — required for the row-for-row diff in §3 to mean anything.
- **v2 property verified in the data**, not inferred from the filename: 64/64 abstain
  rows `recurs=false`, 0 recurs-True.
- **`frac_reward_zero_std` cross-validated** by an independent path against the
  trainer's logged value (§5).

## 7. Provenance — evidence force-added and hash-recorded

`data/checkpoints/` is gitignored, so the artifacts backing this document were initially
untracked and no number here traced to a committed file. Resolved under the
`.gitignore:47-49` precedent (the same one that tracks `data/checkpoints/eval/` dumps):
**19 files force-added, 1.2 MB total**, with sha256 for each recorded in
[`ARTIFACTS.sha256`](ARTIFACTS.sha256) —

`grpo_run_config.json`, `val_curve.jsonl`, the 15 `eval_step*.jsonl` dumps,
`reward_log.jsonl`, `reward_members.jsonl`. The adapters themselves stay out.

Every number in this document now traces to one of those files. The whole manifest —
**58 entries**, the 39 pre-existing plus these 19 — verifies with `shasum -a 256 -c`.

**One duplicate is genuine, not a copy error.** `eval_step20.jsonl` and
`eval_step40.jsonl` share sha256 `8d728ff6…`: byte-identical dumps, confirmed with
`cmp`. The ck-20 and ck-40 adapters are distinct (`134a3a35…` vs `00010c8c…`). This is
the §3 phenomenon appearing a second time, independently and 60 steps earlier — two
different weight states, identical behaviour on all 112 val rows.

## 8. Reproducing this document

```bash
python3 scripts/verify_run4a_selection.py data/checkpoints/grpo_run4a_seed42
```

Recomputes the §2 table from the per-row dumps, re-applies the §(g) keys, re-runs the
§6 integrity checks, and recomputes §5 from `reward_members.jsonl`. No GPU, no
generation, no network.

Evidence integrity, separately:

```bash
shasum -a 256 -c docs/analysis/ARTIFACTS.sha256
```

And the reward values these dumps carry are pinned by
`tests/test_false_abstain_base.py`, which re-scores all **1,680** committed val rows
(15 steps × 112) from their raw completions and requires exact reproduction at the
default `false_abstain_base`.
