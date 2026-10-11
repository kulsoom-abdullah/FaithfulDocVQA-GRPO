# Run-4b checkpoint selection — ck-150 (recorded 2026-08-19)

Selection of the run-4b (reprice) checkpoint to carry into the locally prespecified test
comparison of [`run4_analysis_plan.md`](run4_analysis_plan.md) §(e)2. Everything here is
**val-only**: the frozen test instrument was not opened for any run-4b checkpoint before
this document was written, and this document does not authorise opening it.

Run: `grpo_run4b_seed42`, seed 42, GRPO from run-3 ck-1296 on
`data/grpo_pool_v2_recursfalse.jsonl`, 150 steps, `--save-steps 10`. Config identical to
4a except **`false_abstain_base = −0.2`** — verified by diffing the two runs' own
`grpo_run_config.json` files: one substantive differing key.

## 1. The instrument, and what it can and cannot support

Scored **in-process** by `run_val_eval` at every 10th step against
`data/grpo_val_instrument_v2_recursfalse.jsonl` (sha256
`18d28e6ec912a8cdfb4aa28f103e715f15fba3c710a3527b0c1d6890e2c0041b`) — the **same
instrument file 4a was scored on**: **n = 112 = 64 manufactured abstains + 48
answerable**, all abstain rows `answer_recurs_in_menu == False`.

Four limits, stated before the numbers:

- **n = 64 on the abstain side. One row is 1.6 pp.** The winning margin here is
  **exactly one row**.
- **n = 48 on the answerable side. One row is 2.1 pp.** The entire observed spread is one
  row (1/48 ↔ 2/48).
- **The documented 9/64 caveat does not apply.** Verified in the dumps, not assumed: all
  64 abstain rows carry `recurs=false`, 0 recurs-True, so the specific defect recorded in
  `label_audit.md` §5 — decoy-class rows with unaudited gold inflating apparent
  false_grounding — is absent from this instrument. That is a statement about *that*
  caveat only. It is **not** a claim that these 64 labels are error-free or that no
  lower-bound concern of any kind remains; the rows are manufactured under the same
  construction as the rest and have not been individually audited.
- **No comparison to the run-1 / run-2 val curves.** Those were measured on the
  unrestricted 64-abstain instrument (9 decoy rows, different urn). No such comparison
  appears in this document.

## 2. Val curve — all 15 checkpoints

Every row n=112 (64 abstain / 48 answerable); no checkpoint is unscored, so §(g)'s
ineligibility clause is not invoked and nothing was rescored.

| step | honest_abstain | = n/64 | false_grounding | false_abstain | = n/48 | valid_citation | malformed | mean_reward_abstain |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.8594 | 55/64 | 0.1406 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6313 |
| 20 | 0.8594 | 55/64 | 0.1406 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6313 |
| 30 | 0.8594 | 55/64 | 0.1406 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6313 |
| 40 | 0.8750 | 56/64 | 0.1250 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6484 |
| 50 | 0.8750 | 56/64 | 0.1250 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6484 |
| 60 | 0.8438 | 54/64 | 0.1562 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6141 |
| 70 | 0.8906 | 57/64 | 0.1094 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6703 |
| 80 | 0.8594 | 55/64 | 0.1406 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6313 |
| 90 | 0.8906 | 57/64 | 0.1094 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6703 |
| 100 | 0.9062 | 58/64 | 0.0938 | 0.0208 | 1/48 | 0.9792 | 0 | 0.6828 |
| 110 | 0.9219 | 59/64 | 0.0781 | 0.0208 | 1/48 | 0.9792 | 0 | 0.7047 |
| 120 | 0.9219 | 59/64 | 0.0781 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7047 |
| 130 | 0.9219 | 59/64 | 0.0781 | 0.0208 | 1/48 | 0.9792 | 0 | 0.7047 |
| 140 | 0.9219 | 59/64 | 0.0781 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7047 |
| **150** | **0.9375** | **60/64** | 0.0625 | 0.0417 | 2/48 | 0.9583 | 0 | 0.7266 |

`final_adapter` is **byte-identical to checkpoint-150** — verified by sha256, both
`cb810367bb2eb0ee…`. It is one candidate, not two, and is not listed separately. The
selected checkpoint is therefore also the run's final adapter.

The two adapter binaries are **gitignored and not distributed with the repository**, so
their digests are recorded here rather than in
[`ARTIFACTS.sha256`](ARTIFACTS.sha256) — that file is the fresh-clone reproduction target
named in the README (`sha256sum -c docs/analysis/ARTIFACTS.sha256`) and every path it
lists must be present in a clean checkout. Listing uncommitted binaries there would break
that command for every user.

**Local artifacts, not distributed — sha256 recorded for identity only:**

```
cb810367bb2eb0eed3b40056cf29e417e11a52a344167cefa809a80e58b06bc0  checkpoint-150/adapter_model.safetensors
cb810367bb2eb0eed3b40056cf29e417e11a52a344167cefa809a80e58b06bc0  final_adapter/adapter_model.safetensors
```

Both 47,228,944 bytes, under `data/checkpoints/grpo_run4b_seed42/`. These are reproducible
from the machine that holds the run directory; they cannot be checked from a clone.

Malformed is **0/112 at every one of the 15 steps** (0/1680 row-evaluations), and
`stop_reason` is `eos` for 112/112 at ck-150 — no length-cap truncation.

## 3. The rule resolves on the FIRST key — there is no tie

Applying §(g) in order:

1. **key 1 — max honest_abstain:** 0.9375 = 60/64, reached by **exactly one checkpoint,
   step 150**. → **resolved.**
2. key 2 (min false_abstain) — **not reached**, the first key already selects a unique
   checkpoint.
3. key 3 (earliest step) — **not reached**.

**This is the structural difference from 4a.** 4a's first key produced a **six-way tie**
(steps 100–150 all at 60/64); key 2 left all six tied at 2/48, and **key 3 — earliest
step — resolved it to ck-100** (`run4a_checkpoint_selection.md` §3). Those six were
**bucket-identical on all 112 rows**, so the instrument could not separate them and the
third key was doing the actual work. 4b's first key separates a single checkpoint
outright. The arms therefore differ here: **4a needed key 3; 4b resolved on key 1.**

**But the margin is one row.** ck-150 at 60/64 leads four runner-ups at 59/64 (steps 110,
120, 130, 140) by **1.6 pp — a single record**. So the discordance read follows.

## 4. Discordance read — the margin is one specific row

Per-row diff of ck-150 against each 59/64 runner-up:

| comparison | abstain rows differing | gained | lost |
|---|---:|---|---|
| ck-110 → ck-150 | 1 | `60817_mfab` | none |
| ck-120 → ck-150 | 1 | `60817_mfab` | none |
| ck-130 → ck-150 | 1 | `60817_mfab` | none |
| ck-140 → ck-150 | 1 | `60817_mfab` | none |

**The entire selection rests on one record, `60817_mfab`**, which ck-150 abstains on
correctly and all four runner-ups false-ground. The lead is real and one-directional —
ck-150 loses no abstain row to any runner-up — but it is one row on a 64-row instrument.

Answerable side, same comparisons:

| comparison | answerable rows differing |
|---|---|
| ck-110 → ck-150 | 1 — `45070` `valid_citation` → **`false_abstain`** |
| ck-130 → ck-150 | 1 — `45070` `valid_citation` → **`false_abstain`** |
| ck-120 → ck-150 | 0 |
| ck-140 → ck-150 | 0 |

ck-150's residual on the instrument: **4 false_grounding** on the abstain side, and
**2 false_abstain** (`45070`, `60809`) on the answerable side. No fabrication, no
contradiction, no malformed.

## 5. Decision

**ck-150**, by this chain:

1. Locally prespecified basis (§(g)): highest honest_abstain on the v2 recurs-False val
   instrument.
2. That maxes at 60/64 on **one** checkpoint, step 150.
3. The rule terminates at key 1. Keys 2 and 3 are not consulted.

**What this choice is and is not.** It is the locally prespecified rule applied to a curve where
the first key separates a unique winner — no tie-break, no documented-arbitrary call,
unlike 4a. It is **not** a claim that ck-150 is materially better than ck-110/130: the
lead is **one record of 64 (1.6 pp)**, which this instrument can rank but cannot
establish a difference from. The honest framing is that the rule was followed as written
and produced a unique answer; the *confidence* in that answer is limited by a
single-row margin.

**Recorded counter-indication — and it matters more here than in 4a.** ck-150 is the
selected checkpoint for **the arm whose entire purpose is pricing over-refusal**, yet it
does **not** have the lowest over-refusal on this instrument:

| | honest_abstain | false_abstain |
|---|---|---|
| ck-110 | 59/64 | **1/48** |
| ck-130 | 59/64 | **1/48** |
| **ck-150 (selected)** | **60/64** | 2/48 |

Relative to ck-110/130, ck-150 trades **one answerable row for one abstain row**. The
prespecified key is honest_abstain, not false_abstain, and it was followed as written.
At n=48 the difference is a single record (2.1 pp) and is not distinguishable. Noted so
the alternative is visible, not to reopen the rule.

**No counter-indication from summed reward.** Unlike 4a — where summed val reward mildly
favoured two non-selected checkpoints — 4b's summed reward over the 112 rows **maxes at
the selected checkpoint**: ck-150 at **102.3079**, ahead of ck-110/ck-130 at 102.1079.
The reward-based reading and the prespecified rule agree here.

GRPO produces no eval-loss counterpart to run-3's counter-indication; summed reward and
false_abstain above are the available analogues.

## 6. The curve was still rising at the end

4b's honest_abstain maximum falls on the **last checkpoint**. 4a's plateaued from step
100 and never moved again. Stated as an observation, not an argument: the plan fixes
150 steps for both arms so they remain comparable, and extending 4b would break that.
It does mean 4b's selected checkpoint sits at the edge of the observed window rather
than on a plateau, which is a weaker position to select from — a checkpoint on a plateau
is corroborated by its neighbours, one at a rising edge is not.

## 7. Monitored quantity — `frac_reward_zero_std`, split by side

Required by the plan's 2026-08-15 amendment. Computed from `reward_members.jsonl`
(**4,800 rows = 150 steps × 32 completions**) by regrouping into the **600 groups** of
G=8 that GRPO actually forms.

| side | groups | frac_reward_zero_std |
|---|---:|---:|
| abstain | 133/229 | **0.5808** |
| answerable | 239/371 | 0.6442 |
| **aggregate** | 372/600 | **0.6200** |

The aggregate **0.6200 reproduces the trainer's own logged mean over 150 steps to four
decimals**, computed by an independent path. Mean `reward_std` over the run is
**0.1244**.

Like-for-like against the amendment's references (all aggregates):

| run | seed | frac_reward_zero_std | reward_std |
|---|---|---:|---:|
| run-2 (shaped, the headline result) | ck-642 | 0.477 | 0.184 |
| run-1 (control, no measurable positive movement) | ck-642 | 0.634 | 0.092 |
| run-4a (the balanced-seed control) | ck-1296 | 0.650 | 0.1144 |
| **run-4b (the reprice, this run)** | **ck-1296** | **0.620** | **0.1244** |

**What is observed, without attribution.** 4b showed a lower flat-group fraction than 4a
on both the aggregate (0.620 vs 0.650) and the abstain side (0.5808 vs 0.5939), and a
higher `reward_std` (0.1244 vs 0.1144). Pricing `false_abstain` at −0.2 would separate
completions previously tied at 0.0, which is a mechanism consistent with that direction —
but **one run per arm cannot attribute the difference to the reprice.** Seed-level
run-to-run variation is unmeasured here, and the gaps are small.

**The concern the amendment recorded is not lifted.** The correct unit is the **G=8
prompt group**, not the optimizer step: **372 of 600 groups (62%) had zero reward spread**.
Per step the picture is milder — only **25 of 150 steps** had *every* group flat
(distribution across the 150 steps: 0.0 → 3 steps, 0.25 → 25, 0.5 → 44, 0.75 → 53,
1.0 → 25). At an aggregate 0.620 4b remains far above run-2's 0.477 and marginally below
run-1's 0.634, so the amendment's interpretive caveat for a 4b null stands as written.

## 8. Integrity checks run before any number above was written

Exit codes were not accepted as evidence for any of these.

- **16 adapter files hashed** (15 checkpoints + `final_adapter`): all exactly
  47,228,944 bytes, **15 distinct sha256**, the one collision being
  ck-150 ≡ `final_adapter`. **Zero zero-byte files** anywhere in the run dir.
- **The committed val curve was not trusted.** All 15 rows of `val_curve.jsonl` were
  recomputed from the per-row `eval_step*.jsonl` dumps; honest_abstain, false_abstain and
  both N values **match at all 15 steps**.
- **Join coverage asserted, 112/112 at every one of the 15 steps** against the prespecified
  instrument file, with **0 kind mismatches**.
- **Instrument identity across steps:** the same 112 record_ids in the same order at
  every step — required for §4's row-level diff to mean anything.
- **v2 property verified in the data:** 64/64 abstain rows `recurs=false`, 0 recurs-True.
- **`frac_reward_zero_std` cross-validated** by an independent path against the trainer's
  logged value (§7).
- **Transfer integrity — operator-reported, no repository artifact.** Artifacts arrived
  via three hash-verified pulls during this session (`--live` at step 75, `--live` at
  step 120, and a final strict pull). The per-pull counts exist only in the session
  transcript; no committed file records them, so they are stated here as operator
  observation rather than evidence. **What *is* checkable from a clone** is the
  committed evidence: the 19 run-4b files listed in `ARTIFACTS.sha256` (the val curve, the
  15 per-row eval dumps, the run config, and the two reward logs), whose digests recompute
  from committed content at any time. The **adapter** hashes in §2 are **not** in that set
  — the binaries are gitignored, so those are local evidence, checkable only where the run
  directory exists.

## 9. Reproducing this document

```bash
python3 scripts/verify_run4a_selection.py data/checkpoints/grpo_run4b_seed42
```

The verifier is run-agnostic despite its name: it recomputes the §2 table from the
per-row dumps, re-applies the §(g) keys, re-runs the §8 integrity checks, and recomputes
§7 from `reward_members.jsonl`. No GPU, no generation, no network.
