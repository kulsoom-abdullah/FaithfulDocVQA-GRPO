# Multi-occurrence audit — how often the answer string appears more than once (2026-08-12)

Quantification only. CPU, read-only, committed artifacts. **No script was modified, no
localizer was touched, and no label was changed.** Nothing here is a proposal; §5 states
explicitly what it does and does not license.

Question: on **answerable** rows, the gold answer string sometimes appears at more than one
place in the evidence table. Only one of those places is the gold citation. When a model
cites a different one, today's scorer counts it as a citation miss. This measures how
large that population is, and how much of the measured citation error it accounts for.

Normalization is the committed scorer's `_norm` throughout
(`# source: scripts/eval_scorer.py::_norm` — NFC, smart-quote folding, whitespace
collapse, casefold).

## 1 · Two measures, and why they differ

**String-level (the count-extension of the committed decoy rule).** Identical surface and
identical normalization to `answer_recurs`
(`# source: scripts/build_manufactured_abstains.py::answer_recurs`): the menu is
space-joined into one `_norm` string and `_norm(gold)` is searched in it. The only change
is boolean → count, taking disjoint (non-overlapping) matches, and the **max over gold
variants** (the counting analogue of that rule's any-variant test).

Verified equivalent at the boolean boundary: **string-level ≥1 = 3,555 rows** and
`answer_recurs` **= 3,555 rows**, over the same 4,496 answerable rows. Same rule.

**Site-level (whole-token spans).** A *site* is a run of consecutive menu tokens whose
joined `_norm` equals `_norm(gold)`. Here, "citable sites" means matching OCR token
locations that a model can reference by ID, not necessarily locations that support
the question. Tokens whose `_norm` is empty cannot participate in a text match and are
skipped, so adjacency is over non-empty-normalized tokens in menu order. Overlapping
candidate sites are reduced to a maximal pairwise-disjoint set, greedily in menu order.

**The two diverge, and the string-level one is the looser.** It inherits the substring
behaviour of `answer_recurs`: a one-character gold matches inside every longer word.

| record | gold | string-level occurrences | citable whole-token sites |
|---|---|---:|---:|
| `58395` | `'e'` | 245 | **0** |
| `57825` | `'A'` | 182 | 10 |
| `58393` | `'"a"'` | 109 | 2 |
| `55442` | `'0'` | 94 | 1 |

Read string-level as an **upper bound on textual recurrence**, and site-level as the
**count of matching whole-token locations**. The rest of this document uses site-level wherever
citation behaviour is at stake.

## 2 · Q1 — rows whose answer string occurs ≥2 times (answerable only)

Base populations: `data/master.jsonl`, 4,932 rows, **4,496 answerable**
(train 3,110 / validation 707 / test 679).

**String-level (`answer_recurs` normalization, disjoint occurrences, max over variants):**

| split | answerable | rows with ≥2 occurrences | % of answerable |
|---|---:|---:|---:|
| train | 3,110 | 573 | 18.4% |
| validation | 707 | 138 | 19.5% |
| test | 679 | 124 | 18.3% |
| **all** | **4,496** | **835** | **18.6%** |

**Site-level (citable whole-token spans):**

| split | answerable | rows with ≥2 citable sites | % of answerable | rows with 0 sites |
|---|---:|---:|---:|---:|
| train | 3,110 | 361 | 11.6% | 996 |
| validation | 707 | 92 | 13.0% | 222 |
| test | 679 | **80** | 11.8% | 218 |
| **all** | **4,496** | **533** | **11.9%** | 1,436 |

Two observations worth recording. The rate is **flat across splits** on both measures
(18.3–19.5% and 11.6–13.0%), so this is a property of the corpus, not of any split. And
the 18.6% string-level figure sits in the same band as the ~19–20%
`dropped_span_matches_gold` quirk rate and the 18.8%-of-answerable
cited-span-does-not-match-gold band already recorded for this dataset — related symptoms
of the same frozen-localizer looseness, not one number reappearing.

The **1,436 answerable rows with zero citable sites** are the mirror image: the gold
answer string does not appear as any whole-token span, so the gold citation is a
localizer match that is not the literal answer string. That is the known quirk band, and
it bounds what any string-matching account of citation quality can explain.

## 3 · Q2 — how often a model cites a non-gold occurrence (TEST)

Population: **679 answerable test rows**, of which **80 have ≥2 citable sites**.
Generations are the committed test-split files for the four models; join coverage was
asserted for all four (no missing `record_id`).

The scored pool is `valid_citation` rows only, mirroring
`eval_scorer.aggregate`'s gated pool. "Cited a non-gold site" means the model's cited ids
overlap some **other** same-string site more than they overlap the gold site.

| model | valid_citation pool | of which ≥2-site | cited a **non-gold** same-string site | % of the ≥2-site rows |
|---|---:|---:|---:|---:|
| seed ck-642 | 650 | 80 | **15** | 18.8% |
| run-1 ck-150 (control) | 655 | 79 | **13** | 16.5% |
| run-2 ck-150 (shaped) | 557 | 79 | **15** | 19.0% |
| run-3 ck-1296 (balanced SFT) | 601 | 80 | **13** | 16.3% |

These rows are scored as citation misses today. The behaviour is **stable across all four
models** at roughly one in six of the rows where a same-string alternative exists —
neither the reward shaping nor the balanced supervision moves it.

### Counterfactual: crediting same-string-different-site as a match

Rule: for each pooled row, credit the site the model actually cited when that site spells
the gold answer — formally, replace the gold id set with whichever same-string site
maximizes overlap with the cited ids, then recompute micro precision/recall/F1 exactly as
`aggregate` does (`tp = Σ|cited ∩ gold|`, `p = tp/Σ|cited|`, `r = tp/Σ|gold|`).

| model | gated citation F1 (as scored) | if same-string-different-site counts as a match | Δ |
|---|---:|---:|---:|
| seed ck-642 | 0.8412 | 0.8561 | +0.0148 |
| run-1 ck-150 (control) | 0.8296 | 0.8414 | +0.0117 |
| run-2 ck-150 (shaped) | 0.8694 | 0.8877 | +0.0183 |
| run-3 ck-1296 (balanced SFT) | 0.8758 | 0.8892 | +0.0133 |

**Scoring validation:** the as-scored column reproduces the committed figures exactly —
run-1 **0.8296 (n=655)** and run-2 **0.8694 (n=557)** match
[`grpo_finals_natural_test.md`](grpo_finals_natural_test.md), and run-3 **0.8758 (n=601)**
matches the walkthrough's §9 cell. Only the counterfactual column is new.

The effect is **real but small: ~1.2–1.8 F1 points**, and it does not reorder the models.

### Three examples (run-3 ck-1296, verbatim from the committed generation)

| record | question | gold | sites | gold site | model cited |
|---|---|---|---:|---|---|
| `57973` | *What is the classification for Project P3(2)?* | `Bone` | 2 | id 140 → `bone` | id 142 → `Bone` |
| `50719` | *What is the 'duration in months' for 'water on tobacco'?* | `4` | 11 | id 44 → `4` | id 131 → `4` |
| `1623` | *Is Michael the corresponding author?* | `No` | 3 | id 99 → `No` | id 233 → `No` |

`57973` is the cleanest case: two tokens spelling the same word, differing only in case,
which `_norm` folds away. `50719` is the opposite extreme — eleven citable `4`s on the
page, where "the gold one" is a property of the localizer's scan order, not of the answer.

## 4 · Stated Ns

| quantity | N |
|---|---:|
| master rows | 4,932 |
| answerable rows (all splits) | 4,496 |
| answerable with ≥1 occurrence (string-level ≡ `answer_recurs`) | 3,555 |
| answerable with ≥2 occurrences (string-level) | 835 |
| answerable with ≥2 citable sites (site-level) | 533 |
| answerable with 0 citable sites | 1,436 |
| TEST answerable | 679 |
| TEST answerable with ≥2 citable sites | 80 |
| TEST valid_citation pool (seed / run-1 / run-2 / run-3) | 650 / 655 / 557 / 601 |
| TEST non-gold-site citations (seed / run-1 / run-2 / run-3) | 15 / 13 / 15 / 13 |

## 5 · What this does and does not establish

**Does:** on ~12% of answerable rows the evidence table contains more than one citable
place spelling the gold answer; on the test split that is 80 rows; all four models cite a
non-gold one of those on 13–15 rows; and the citation F1 attributable to that choice is
about 1.2–1.8 points for every model.

**Does not:** it does not show the gold citation is wrong on those rows. A second site
spelling the answer is not automatically a *valid* site — the survivor audit's whole
finding was that the disposition depends on whether the surviving token actually answers
the question, judged against the page, not on string identity
(`# source: docs/findings/survivor_audit.md`; the `51687_mfab` adjudication is the worked
case). `50719` with eleven `4`s is exactly a row where string identity settles nothing.

Nor does it license a scoring change. The counterfactual column is a **measurement of how
much a different convention would move the number**, computed so the size of the question
is known. Adopting it would require adjudicating the alternative sites, which is the same
per-site verification work already parked as future work in
[`run3_analysis_plan.md`](run3_analysis_plan.md) §(e) — and doing it by string rule alone
is what that section rejects.

No training implication is drawn here.

## 6 · Reproduction

Computed with a standalone read-only script (kept out of the repo per the audit's
no-script-changes constraint) that imports `scripts.eval_scorer._norm` and `classify` and
`scripts.build_manufactured_abstains.answer_recurs` — no logic reimplemented except the
two counting functions defined in §1, whose boolean boundary is checked against
`answer_recurs` (3,555 = 3,555). Inputs: `data/master.jsonl` and the four committed
`gen_test_checkpoint-*.json` files, whose sha256s are in
[`ARTIFACTS.sha256`](ARTIFACTS.sha256).
