# A3 — Failure analysis across three populations

Recorded 2026-08-09. Committed per-record artifacts only — no new inference:
`gen_test_checkpoint-642.json` (seed), `gen_test_checkpoint-150.json` +
`gen_manufactured_checkpoint-150.json` (GRPO run-2), scored with the committed
`scripts/eval_scorer.py`; menus/bboxes from `data/master.jsonl` and
`data/manufactured_test_abstains.jsonl`. Denominators reflect the 2026-08-09
`51687_mfab` adjudication (clean subset = 235). Row identifications below use
bbox-line reconstruction (the 51687 lesson: site identity is decided by the row
a token sits in, not by its string). Mode counts within populations are
hand-classified from the per-record lists — approximate by nature; every
hypothesis is labeled as such.

## Population 1 — seed ck-642: false groundings on natural test abstains (n=15/77)

Feature contrasts vs the 77-row natural-abstain base: **form docs 40% vs 19%**;
**menus ≥150 tokens 47% vs 19%** (menus <50: 7% vs 47%); answer-string present
in menu 1/15 (base 12%); cited-span text ≈ gold answer only 1/15; and
**found_answer correct 15/15**.

**Dominant mode (≈13/15): dense-menu lookalike citation.** Perception never
fails — the model reads the true answer off the page every time — but on these
rows the answer itself is what OCR missed, so the citation lands on the nearest
lookalike or spatially adjacent token. *Hypothesis:* natural-abstain false
grounding is a dense-menu phenomenon — sparse menus (the majority of natural
abstains) offer no plausible token to grab, so the model abstains; dense menus
offer a lookalike, and the citation head takes it.

**Worked example 1 — `16835` (+ siblings `16837`, `16839`).** Conference-fee
questions; golds $2195 / $1795 / $1095. None of the fee amounts survived OCR.
The only dollar token in the region is id 207 `$200` — row y=1226, a *marketing
sentence* ("…receive **$200** …discount"), not the fee table. All three sibling
questions output the correct fee in `found_answer` and cite **the same id 207**.
Systematic, not random: one lookalike token absorbs three different questions'
citations.

**Worked example 2 — `60442`.** "What is the date stamped?" gold `Dec 14 1976`,
read correctly from the image. The stamp region OCR'd into a garbled line
(y=1686: "…Gotober (6 … **1975,** …"); the model cites id 24 `1975,` — the only
year-shaped token available. Right region, wrong year, no supporting text.

## Population 2 — GRPO run-2: residual false groundings on clean-235 (n=31)

Feature contrasts vs the 235-row clean base: **free_text 35% vs 15%**; menus
≥150 55% vs 43%; **answer-string-in-menu 11/31 = 35% vs 11% base**;
localizer-quirk rows 2/31 vs 16% base (quirk rows do *not* drive residuals);
found_answer correct 24/31.

Hand-classified modes:
1. **Exact-string second sites — 11/31 recurs-True — the mislabel-suspect
   class.** Cited texts include `3,000.00`→`3,000.00`, `500`→`500`,
   `sq. ft.`→`sq. ft.`, `60 calendar days`→`60 calendar days`, `Check`→`CHECKS`.
   Per the `51687_mfab` lesson (hand-confirmed mislabel; its exclusion is why
   the denominator is 235), **these 11 "failures" carry label suspicion** —
   test-audit calibration excluded 14/40 recurs-True rows. Some fraction of
   pop-2 is measurement artifact, not model failure.
2. **Near-miss lookalikes (≈10/31)** — same mechanism as population 1:
   `10:55 a.m.` for 10:15 a.m., `(336) 744-3763` for …741-7805, `November 1,
   1942` for November 19 1943, `Qualitative` for Quantitative, `February 7,
   1977` for February 13 (see example 1).
3. **Surviving variant/alias sites the substring predicate cannot see
   (≈6/31)** — the answer effectively survives in OCR-garbled or alias form,
   so recurs=False *under-detects*: `MICHAEL 0. JOHNSON` (zero-for-O, example
   2), `("EI Matador")`, `Corisumer information Center`, `Pearay L. Ogra,` (the
   full name of "Dr. Ogra"), `(ATP)` (the abbreviation of the gold phrase).
   **These extend the mislabel-suspicion beyond the recurs-True 11 — 11/31 is a
   lower bound on the label-artifact share.** The remainder (≈4) are semantic
   neighbor grabs (`FUNDS` for a dollar figure, `desk` for "paper documents").

**Worked example 1 — `51702_mfab` (doc 1266 — the 51687 itinerary, third
appearance).** "On which day does the passenger spend the longest time in air
travel?" gold `February 13, 1977`. The build dropped original ids 59-61 — the
"February 13, 1977" date-header row (y=922) — so the answer string is provably
absent: **clean label, genuine failure** (adjudicated like 51687, opposite
verdict). Run-2 answered "February 11, 1977" (wrong: the Feb-13 Honolulu→Denver
flight ≈ 12 h is the longest) citing post-drop ids 9-11 — the **February 7**
date row (y=279). Both fields land on wrong dates: date-family confusion inside
one document, in both perception and citation.

**Worked example 2 — `4926_mfab` (⚑ new mislabel candidate).** "Who is this
document from?" gold `Michael O. Johnson`; the build dropped the signature-block
tokens (source ids 194-196, `Michael` `O.` `Johnson`). But the **letterhead**
row y=138 (post-drop ids 13-16: `4` `0.` `JOHNSON` `MICHAEL`) is an OCR-garbled
duplicate of the same name — word-reordered, zero-for-O — invisible to the
substring predicate. Run-2 cites 13-15. By the 51687 standard (does the
surviving site genuinely evidence the answer?) the letterhead of a memo
plausibly does answer "who is this from" — **flagged for hand adjudication, not
decided here.**

## Population 3 — GRPO run-2: false abstains on answerable test (n=120/679)

Feature contrasts vs the 679-row answerable base: **answer-string present in
menu 33% vs 80%** — the dominant signal by far; gold answers are short (median
4 chars vs 9); figure/diagram 13% vs 6%; gold cite-spans similar (median 1 both;
spans>3: 8% vs 12%); and **found_answer correct in 102/120 refusals**.

**Dominant mode (and the population's headline): refuse-on-failed-string-match.**
The model reads the correct answer, then refuses because that string does not
literally appear in the evidence table (menu) — exactly the string-checking
mechanism `PROVENANCE.md` §5 measured on abstain rows, now seen from the other
side. *Hypothesis:* run-2's abstain shaping strengthened a literal
answer-in-menu check; on the ~20% of answerable rows whose gold evidence is a
non-literal match (paraphrase, case/punctuation variant, multi-token assembly),
the check over-fires. The remaining refusals include 37/120 "should-be-easy"
rows (span ≤2 *and* answer present) — see example 2.

**Worked example 1 — `50539`.** "What is the Heading of the table?" gold
"Appendix 1 - results of method development project survey", gold evidence span
= 9 tokens. Run-2's `found_answer` is the near-verbatim heading ("Appendix I -
Results of Method Development Project Survey" — I/1 variance) — then
`answerable_from_menu: false`, no citation. Reading a 9-token heading, matching
it against 9 separate menu tokens, and requiring a literal match is precisely
where a string-checker fails honest work. (`1625` is the same shape: 8-token
form name, near-verbatim found_answer, refusal.)

**Worked example 2 — `25680`.** "what day of the week is mentioned…" gold
`Friday`; the evidence table contains id 7 `FRIDAY,` (row y=196: "NOVEMBER
**FRIDAY,** 21, 1975" — the gold cite is exactly [7]). Run-2 outputs
`found_answer: "FRIDAY"` and refuses. The only gap between its answer and the
citable token is a trailing comma and case. *Hypothesis:* the learned matcher
is stricter than the scorer's own normalization — punctuation-level mismatches
trigger refusal.

## Cross-population synthesis

One mechanism, three faces. The model verifies by **string-matching its
perceived answer against the evidence table**: when the answer is missing and
the menu is dense, it grabs a lookalike (pop 1, pop 2 mode 2); when the answer
is missing and it has been abstain-shaped, it correctly refuses — but the same
check over-fires wherever gold evidence is non-literal (pop 3); and its residual
manufactured-set failures concentrate exactly where an exact-string second site
survives (pop 2 mode 1) — which is also where label validity is weakest. The
practical reading, stated as hypothesis: **site-level verification (which row
does this token live in?) is the capability gap on both sides of the ledger**,
and pop-2's residual is part measurement artifact — ≥11/31 rows carry label
suspicion, with mode 3 pushing the true artifact share higher pending the
LLM-judge adjudication pre-registered in `run3_analysis_plan.md` §e.
