# Design Decisions & Insights — FaithfulDocVQA

Why the project is built the way it is. These are decision *rationales* (for the
eventual README/story), not how-tos. Each section answers a "why," and where a claim
rests on a measurement, the number is stated.

---

> ## ⚠️ CONTRACT VERSIONS — READ BEFORE QUOTING ANY NUMBER
>
> Two different output contracts produced the numbers in this doc. They are **not
> the same experiment** and must never be blended in a writeup:
>
> - **v1 = 2-field** `{answer, evidence_ids}`. The **A-vs-B abstention experiment**
>   (§1 below: answer-only *never* abstains 0/46 vs answer+abstain 0.74) is **v1**.
>   It HAS a control arm (SFT-A, answer-only) — that contrast is the whole result.
> - **v3 = 3-field** `{found_answer, answerable_from_menu, evidence_ids}` (§3–4). The
>   **ck-642 headline** (honest_abstain val 0.848 / **test 0.805**, false_grounding
>   0.152 / 0.195) is **v3**, and it comes from a **mode-B-only** run
>   (`sft_v3_32g/sft_B_seed42`). **There is NO answer-only (SFT-A) control arm in v3.**
>
> So: the refusal improvement from adding abstention targets is a *v1* result with a
> control; the ck-642 faithfulness numbers are a *v3* result **without** a control.
> Do not present
> the v3 honest_abstain as if the v1 A/B contrast backs it — they are different
> contracts and different runs. (There is also a v2 = 2→3-field pivot, §3.)

---

## Limitations & rigor notes (read with the headline)

**How ck-642 was selected, and what it is not.** The 1.5-epoch checkpoint (ck-642) was
chosen post-hoc as the **best-available** faithfulness checkpoint on the **n=46**
gold-abstain *validation* rows. Its honest_abstain trajectory across checkpoints is
**noisy and non-monotonic** — 0.370 (0.5e) → 0.239 (1.0e) → **0.848 (1.5e)** → 0.630
(2.0e) → 0.630 (2.5e) → 0.674 (3.0e) — and at n=46 its Wilson interval **overlaps** the
2.0–3.0-epoch cluster, so the peak is a **point estimate, NOT statistically separated**.
It is also **single-seed (seed 42) and not multi-seed-replicated**. Read "1.5 epochs is
best" as best-available evidence, not an established result.

Test-split generations exist for **all six**
checkpoints (produced sequentially 2026-06-29 on the `faithfuldocvqa-ckpts` volume;
now committed under `data/checkpoints/eval/sft_B_seed42/gen_test_checkpoint-*.json`) —
two weeks before the GRPO stage fixed ck-642 as its seed (`modal_grpo.py`, 2026-07-14)
— and ck-642 is the **test argmax**. So the test honest_abstain **0.805 (62/77)** is a
max over six candidates on the reporting set, not a touched-once confirmation. The
neighboring checkpoints score **0.649–0.675**, showing observed variation across
checkpoints, not a forward estimate of ck-642's future performance; ck-642 also has
the **highest test false_abstain** (0.038 vs 0.007–0.021 for the others), so part of
the abstain peak is a global abstain-propensity spike rather than sharpened discrimination. Full curve and
recompute commands: `docs/PROVENANCE.md`.

**Multi-seed replication — attempted, blocked, deferred (documented systems bug).** A
replication run (seed 1, config-identical except `--seed`) was attempted on an RTX 6000
Ada / cu124 pod and **abandoned on cost/value grounds** after a reproducible **throughput
pathology**: training s/it climbed **monotonically and unboundedly** (~40 → 80+ s/it over
the first ~10 steps) on **torch 2.5.1+cu124 *and* 2.6.0+cu124** (torch 2.4.1 can't load
the model at all — transformers 5.12's 4-bit path calls `set_submodule`, added in torch
2.5). seed-42's original **cu128** stack ran **flat at 39.6 s/it**, so this is a **cu124 ×
variable-shape-VLM pathology on Ada** — *not* torch.compile (disabling dynamo changed
nothing) and *not* `cudnn.benchmark` (unset). Root-cause **deferred**; matched-**cu128**
replication is **future work**. A full seed on the degrading cu124 stack was ~35 h / ~$27
to firm up a secondary footnote — a **stated limitation was the disciplined call** over
the spend. (Aside: this also means any new seed on cu124 differs from seed-42's cu128
stack — an environment confound layered on top of the GPU-arch difference.)

---

## 1. Explicit refusal targets improved the early SFT comparison  ·  [v1 / 2-field · has SFT-A control]

*(These are **v1 2-field** numbers on the held-out validation set. The A-vs-B contrast
is the result; this experiment was not re-run under the v3 3-field contract.)*

The central experiment ran two SFT arms, identical except the training-data filter:
**answer-only** (the control) vs **answer+abstain** (the treatment). On the held-out
validation set (46 gold-abstain rows):

- The **answer-only** model **never abstains** — honest-abstain **0/46** at every
  checkpoint (Wilson 95% CI **[0.000, 0.077]**). It answers every unanswerable
  question.
- The **answer+abstain** model reaches **0.74** honest-abstain — a Wilson interval
  **disjoint** from the control's at every checkpoint — with **fabrication ≈ 0** and
  **answerable citation-F1 essentially unchanged between arms** (~0.82–0.84, differences within noise).
  (Instrument note: this v1 F1 is **ungated** — macro over ALL gold-answerable rows, with
  abstained/malformed rows contributing empty cite-sets as misses. The current v3 scorer's
  `citation_f1_gated` pools **only model-grounded rows** — a different quantity. Never
  compare a v1 F1 to a v3 gated F1; see `docs/PROVENANCE.md`.)

In this early v1 SFT comparison, explicit abstention examples improved refusal on the
held-out validation rows. The comparison does not establish that refusal is absent
from the base model or that every model must learn it through SFT.

---

## 2. The image-ablation — answering without the OCR menu

A grounded-citation model invites a sharp objection: maybe it has **collapsed onto the
OCR menu** — copying answers out of the token list and never actually reading the page.
If so, the "faithfulness" story is hollow.

I tested it directly: for each gold-answerable record, compare answer-correctness with
the **menu present** vs a **coherent no-menu prompt** (image + question only). Results:

- Deltas are **tiny — |Δ| ≤ 1.6 pp, within noise** — for both arms. Removing the menu
  barely changes correctness, supporting the model's ability to answer without that
  list. It does not establish which input it used when both were present.
- **train ≈ validation** (no-menu gap ≤ 2 pp), but similar accuracy does not rule out memorization.
- Both SFT arms show similarly small changes in answer accuracy when the menu is removed.

**Methodological note (why the prompt design mattered):** the no-menu condition had to
be a *coherent* prompt. Simply deleting the menu while still instructing the model to
"cite the OCR menu" would leave it obeying an instruction about a menu that isn't there
— that measures **prompt-confusion**, not image-blindness. The clean test removes both
the menu and the citation instruction together.

---

## 3. The 2→3-field contract change — decoupling perception from attribution

The v1 contract, `{answer, evidence_ids}`, forces the model to **conflate two distinct
failures** behind a single abstention: *"I can't read this"* (a perception failure) and
*"I read it, but it isn't citable from the OCR menu"* (an attribution failure). They are
different events with different downstream meaning, and v1 cannot tell them apart.

v2 splits them: **`{found_answer, answerable_from_menu, evidence_ids}`** — separating
**text-answer correctness** from **OCR-support and citation decisions**.
Three reasons this is the right pivot:

- **(a) Provenance.** Downstream consumers receive separate text-answer and OCR-support
  claims. These distinguish a claimed reading failure from missing OCR evidence;
  they do not independently verify either claim.
- **(b) Separate scoring.** `found_answer` is scored even when the model refuses OCR
  support, so text-answer correctness can be measured separately from refusal. The
  contract itself neither prevents excessive refusal nor establishes use of the image.
- **(c) It dissolves the old reward gap.** The ambiguous v1 cell — "answered a
  gold-abstain row with a real in-menu citation → scores 0" — decomposes cleanly into
  **separately-scored axes** under 3-field, so there's no longer a single undifferentiated
  zero hiding two different behaviors.

---

## 4. The 3-field reward axes (GRPO reward design)  ·  [v3 / 3-field]


The implemented reward combines an outcome-specific base value with a separate
text-answer bonus, following the
[notebook's reward table](../notebooks/faithfuldocvqa_walkthrough.ipynb#s7):

- **`found_answer` correctness:** +0.3 when the answer matches an accepted answer,
  except for malformed or contradictory outputs.
- **Citation-label F1:** for a structurally valid citation claim on a row labeled
  OCR-supported, the base reward is agreement between cited and reference IDs (0–1).
  Existing IDs that miss the reference evidence lower F1; ordinary citation-label
  misses do not receive a separate negative penalty.
- **False grounding and fabricated IDs:** claiming OCR support on a row labeled
  unsupported has base 0.0 in run-1 and −0.6 in run-2 and later GRPO. Citing an ID
  absent from the menu instead has base −0.5.
- **Refusal decisions:** correct refusal has base 0.0 in run-1 and +0.5 in run-2 and
  later GRPO. False refusal has base 0.0 in run-1, run-2, and run-4a, and −0.2 in run-4b.

Malformed or flag/citation-inconsistent outputs receive a fixed −1.0 with no answer bonus.


**Sequencing over magnitude-tuning:** introduce the axes in stages —
(Stage-2 feasibility receipts for the staged rollout: `grpo/probe_stage2.py` —
TASK A, 10-prompt G=8 generation+reward probe with the cu124-pathology tripwire —
and `grpo/probe_stage2b.py` — TASK B, TRL GRPOTrainer VLM construction and
metadata-routing probe. The ablation harness behind §2 is `scripts/eval_ablation.py`;
its raw JSONs live on the Modal volume and are regenerable, not tracked. The
deferred multi-seed replication's ready-to-run per-seed chain is
`scripts/seed_chain.sh`.)
get the **F1 axis** working first, then layer in **found_answer correctness** and the
**flag-judgment** axis — rather than trying to hand-tune all the reward magnitudes on
day one. Staged shaping is easier to debug and less prone to reward-hacking surprises.

**Figure — the GRPO training loop:** prompt → policy samples → reward → gradient step.

```text
prompt (image+question+menu)
      │
      ├─ sample G=8 at temp ──► 8 different 3-field JSONs   ← the variation (decoding noise)
      │
      ├─ eval_scorer scores each ──► 8 rewards              ← RLVR: programmatic, uses gold as key
      │
      ├─ advantage = reward − group_mean                    ← group-relative advantage (the GRPO core)
      │
      └─ upweight above-mean completions, down below        ← the policy-gradient step
```

---

## 5. Why OCR-menu grounding at all — the project's reason to exist

A plain VLM answers DocVQA **fluently whether it is grounded or hallucinating**, and you
**cannot tell which from the answer text alone**. That is the core trust problem.

Citations make the model's OCR-support claim inspectable: readers can locate the
cited words on the page. Code checks structure, flag/citation consistency, and whether
IDs exist in the supplied menu. These checks cannot establish that the cited text
semantically supports the answer; programmatic rewards compare outputs with the
frozen reference labels.

**The honest limitation:** grounding is bounded by **what OCR extracts**. When OCR misses
text (handwriting, photo cells, noisy scans), the answer isn't citable even if a human
could read it — and the model is intended to refuse OCR support while still reporting
any answer it can read from the image. We accept and *measure* that ceiling rather
than paper over it; the abstain class
is precisely this OCR-miss population.

---

## 6. `found_answer` edge case (v2)

When the answer is **masked out of the image or genuinely illegible**, `found_answer`
takes the value **`"unanswerable"`** — the model reports that it could not perceive the
answer on the page. In **v1 every answer is visible by construction**, so `found_answer`
is always populated (the masked / image-absent case is a v2 construction). In that masked/illegible case the contract strictly requires the other two fields as well: answerable_from_menu = false and evidence_ids = [] (nothing was perceived, so nothing is citable). This is the one row type where all three fields collapse to the "nothing found anywhere" state.
