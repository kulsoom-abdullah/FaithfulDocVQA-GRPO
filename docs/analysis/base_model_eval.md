# Base-model (no-adapter) eval — the pre-SFT baseline across instruments

Scored 2026-08-08 with the committed `scripts/eval_scorer.py` (no re-generation, no
scorer changes). Generations: the June-29 volume artifacts (`faithfuldocvqa-ckpts`
`eval/sft_B_seed42/gen_base_{validation,test}.json`, recorded 2026-06-29, pulled
2026-08-08 and committed alongside this report), produced by
`scripts/gen_persist.py --base` via `eval_runner.load_base_model` —
`Qwen/Qwen3-VL-4B-Instruct`, 4-bit NF4, greedy, `max_new_tokens=128`. The
manufactured-set row is generated fresh (2026-08-08 pod session, same loader path);
sha256 for every file in [`ARTIFACTS.sha256`](ARTIFACTS.sha256).

Stop reasons (report per policy — a length-cap ramble is not garbage):
validation 729 eos / 24 length_cap; test 729 eos / 27 length_cap.

## Bucket rates

| metric | validation (n=753: 46 abst / 707 ans) | test (n=756: 77 abst / 679 ans) |
|---|---|---|
| **abstain-side** honest_abstain | **0.1739** (8/46) | **0.1039** (8/77) |
| false_grounding | 0.7174 | 0.8052 |
| fabrication (abstain rows) | 0.0652 | 0.0649 |
| malformed (abstain rows) | 0.0435 | 0.0260 |
| **answerable-side** valid_citation | 0.9533 | 0.9396 |
| false_abstain | **0.0113** | **0.0206** |
| fabrication (answerable rows) | 0.0042 | 0.0044 |
| malformed (answerable rows) | 0.0311 | 0.0353 |
| **overall malformed** | **0.0319** | **0.0344** |
| found_answer accuracy (answerable / abstain) | 0.8336 / 0.7727 | 0.8718 / 0.8933 |
| citation_f1_gated (n_valid / micro / macro) | 674 / **0.6876** / 0.6309 | 638 / **0.6771** / 0.6679 |

The `found_answer` accuracy row above excludes outputs that cannot be parsed as a
consistent three-field contract. For the full-population comparison in the
[README](../../README.md), the base model produced correct answers on **638/756**
natural-test rows overall, **571/679** with reference OCR support, and **67/77**
natural OCR misses. Unparseable outputs remain in those denominators but supply no
correct scored answer.

Elevated malformed (~3%) is expected pre-SFT — the base model has never seen the
3-field contract; reported plainly, not repaired.

## Manufactured test set (phase 2 — fresh generation, recorded 2026-08-08)

Generated on the RunPod session (same `gen_persist.py --base` path and 4-bit
loader as above; greedy, 128 max new tokens). Stop reasons: 238 eos / 12 length_cap.

| metric | manufactured (n=250, all gold-abstain) |
|---|---|
| honest_abstain | **0.0160** (4/250) — clean subset: **0.0169** (4/236); 4/235 = 0.0170 after the 2026-08-09 `51687_mfab` adjudication (see label_audit note) |
| false_grounding | 0.9240 |
| fabrication | 0.0120 |
| malformed | 0.0480 |
| found_answer accuracy (abstain rows) | 0.8151 |

## What this adds

The v3-era results previously had **no base-model control** (PROVENANCE §1). These
rows document observed improvements from the prompted base model to ck-642:
natural-test correct refusal rises from **8/77 = 0.1039** to **62/77 = 0.805**, and
conditional test citation-label F1 rises from **0.6771** (638/679 eligible) to
**0.8412** (650/679 eligible). This compares the saved runs under this project's
prompt; it does not establish that prompting could never produce the behavior.

The historical clean-236 comparison was base **4/236 = 0.0169**, seed
**68/236 = 0.288**, and run-2 **204/236 = 0.864**. After the 2026-08-09 adjudication,
the current clean-235 comparison is base **4/235 = 0.0170**, seed **68/235 = 0.289**,
and run-2 **204/235 = 0.868**.
Natural and manufactured refusals remain different instruments.
