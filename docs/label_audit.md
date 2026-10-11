# Manufactured-abstain label audit

The current evaluation excludes 15 of the 250 manufactured examples, leaving 235. The earlier review below excluded 14; a later review identified one additional example with supporting answer text.

Provenance: numbers regenerated from `data/manufactured_test_abstains.jsonl`, `data/master.jsonl`, and the validation instrument reconstructed via the committed `grpo/train_grpo.py::build_val_instrument` (seed 42, eval_abstain 64, eval_answerable 48). The decoy predicate `answer_recurs` was extracted from `scripts/build_manufactured_abstains.py` with `ast` and executed in isolation (at the time of this audit the module's module-level code rewrote the manufactured data files on import; as of the `__main__`-guard refactor (recorded 2026-08-02), importing it is side-effect-free). No data file was modified.

## 1. What the manufactured-abstain build does

`scripts/build_manufactured_abstains.py` takes answerable DocVQA records and constructs abstain examples: it drops the gold-cited OCR token id(s) from the source menu and renumbers the survivors from 0 (`drop_and_renumber`), yielding a **post-drop menu** in which the gold token is no longer present. The record is written with `target_is_abstain=true`, the `dropped_token_ids`, a `stratum`, and the `answer_recurs_in_menu` flag.

`answer_recurs_in_menu` is computed by `answer_recurs(golds, post_drop_menu)`, extracted verbatim:

```python
def answer_recurs(golds: list[str], new_menu: list[dict]) -> bool:
    """Decoy rule, ANY accepted answer variant: some ``_norm(gold)`` is a substring of the
    space-joined ``_norm`` remaining-menu text (committed ``_norm``). Any-variant (not just
    ``gold_answers[0]``) because any grabbable answer form makes the abstain non-clean."""
    joined = " ".join(_norm(m.get("text", "")) for m in new_menu)
    return any(_norm(g) and _norm(g) in joined for g in (golds or []))
```

It measures **substring presence of any accepted gold variant** in the space-joined, `_norm`-normalized text of the post-drop menu (committed `scripts.eval_scorer._norm`). Any-variant: it is True if `_norm(g)` is a substring of the joined normalized menu for any `g` in `gold_answers`. It is a string-substring test over the joined menu, not a per-token or image test.

## 2. Historical test-set label audit — the 14 excluded records

At the time of the earlier review, `data/manufactured_test_abstains.jsonl` held 250 records (all `target_is_abstain=true`). `exclusion_reason` was non-null on 14; null on 236. The 14 historical exclusions are listed below, with the `52157_mfab` explanation corrected against the saved OCR:

| record_id | question | exclusion_reason |
|---|---|---|
| 1013_mfab | How many years of delay does Novo Nordisk expect for filing the approval? | gold alias at its own site — 'close to 2 years' |
| 15277_mfab | What is the abbreviation for National Stock Exchange? | abbrev definition site — '(NSE)' immediately follows 'National Stock Exchange' |
| 16341_mfab | What is the abbreviation of Product Lifecycle Management? | abbrev definition site — '(PLM)' follows 'Product Lifecycle Management' |
| 31020_mfab | In which University Dr. Jean-pierre Habicht worked as Assistant in Anatomy? | answer recurs — 'University of Zurich' at additional valid CV entries |
| 43779_mfab | How much is the Total Amount? | second valid site — '7,502.53' beside 'Total' (the amount asked) |
| 47452_mfab | What is the total deposit amount? | answer recurs — deposit total '2,200.72' at a second valid site |
| 47496_mfab | What is the 'short title'  mentioned at the bottom of the page? | title repeated — the short title recurs at a second valid site |
| 47535_mfab | Where were the vacation days spend? | second valid location — 'Nashville' in the 'LOCATION OUTSIDE NYC' answer column (not the Vanderbilt purpose cell) |
| 50122_mfab | What is the 'net sq.ft. '? | answer restated — '6,400 net sq. ft.' appears in the SUMMARY block |
| 50939_mfab | Which type of documents(in broadest sense) are emails and attachments? | defined term recurs — 'electronically stored information' at a second valid site |
| 51697_mfab | What class does he travel on Feb 13? | answer recurs — 'First Class' on the Feb-13 flight entry (valid site) |
| 52154_mfab | To whom the ‘letter’ is written ? | recipient recurs — the addressee name appears at a second valid site |
| 52157_mfab | Who is Carolyn Walden? | other accepted answer variants remain with punctuation or spacing differences; 'Director' is absent |
| 53439_mfab | What is the check amount? | same value adjacent field — '$0.00' also the neighbouring Coupon Amount |

## 3. Historical `recurs` × `excluded` (recomputed over all 250)

`recurs` = recomputed `answer_recurs(master gold_answers, post-drop menu)`; `excluded` = `exclusion_reason is not None`.

|  | excluded=True | excluded=False | row total |
|---|---|---|---|
| **recurs=True** | 13 | 27 | 40 |
| **recurs=False** | 1 | 209 | 210 |
| **col total** | 14 | 236 | 250 |

- Exclusion rate within `recurs=True`: **13/40 = 0.3250**.
- Exclusion rate within `recurs=False`: **1/210 = 0.0048**.
- `recurs` base rate over 250: **40/250 = 0.1600**.
- Fisher exact two-sided **p = 8.64625e-11**; odds ratio **100.63**.

## 4. `answer_recurs_in_menu` as a label-error triage flag

Of the 14 human-excluded (mislabeled) test rows in the earlier review, 13 have recomputed `recurs=True` and 1 has `recurs=False`; the flag co-occurs with human exclusion at 32.5% within `recurs=True` versus 0.48% within `recurs=False` (Fisher p = 8.6e-11). The single `recurs=False` exclusion is `52157_mfab` (question: "Who is Carolyn Walden?"; gold variants include "Director", "TV, radio, film department", "Carl byoir& associates,Inc."). In its OCR list after the selected evidence was removed, "Director" is absent. Other accepted variants remain as "TV." "Radio," "Film" "Department" and "Carl" "Byoir" "&" "Associates," "Inc.", with punctuation or spacing differences that the normalized-substring check misses. The manual review identified these as supporting answer text.

## 5. Historical unfinished review — original GRPO validation instrument abstain rows

The 64 abstain rows of the reconstructed original GRPO validation instrument include **9** with `answer_recurs_in_menu=True`: 64689_mfab, 64158_mfab, 53475_mfab, 55106_mfab, 53476_mfab, 58518_mfab, 54259_mfab, 49577_mfab, 57304_mfab. A preliminary hand audit of these 9 found **at least 3 mislabeled** — i.e., the gold answer is in fact recoverable from the post-drop menu, so the row is not a clean abstain. Provenance: hand audit, not a recomputed metric. Incorrect labels can distort the reported refusal rate. Whether they raise or lower it depends on the model's predictions on the affected examples. At the time of this report, the full review of the 9 (and the remaining 55 non-recurs abstain rows) was unfinished.


---

**Manual review note (recorded 2026-08-09).** A later review of `51687_mfab`
found supporting answer text that the earlier manual review had missed. The
remaining OCR id-72 "L.A." sits in the February-13 itinerary row and supports
the answer; the original gold cite id 22 was in the Feb-7 row, a first-occurrence
string-match quirk. The recurrence string check already found "L.A."; this was
not a failure of that check. It is now excluded via `apply_survivor_audit.py`.
Updated §3 marginals: excluded 15 (14 recurs-True /
1 recurs-False), kept 235 (26 recurs-True / 209 recurs-False); exclusion rate
within recurs-True 14/40 = 0.35; Fisher two-sided p = 1.06e-11. The current
`survivor_audit.md` includes this 15th exclusion; §2 above preserves the earlier table.

---

**Scope note on §5 (recorded 2026-08-10).** §5's label-quality concern — 64 abstain
rows, 9 with `answer_recurs_in_menu=True`, ≥3 hand-confirmed mislabelled — applies
to the **reconstructed original GRPO manufactured-abstain validation instrument** and only to it.
Its rows are manufactured (ids ending `_mfab`) and `answer_recurs_in_menu` is a
manufactured-only field; natural `ocr_miss` abstains do not carry it at all.

It therefore does **not** apply to the **natural validation split** (n=753 = 46
natural `ocr_miss` abstains + 707 answerable) that `scripts/score_val_curve.py`
scores via `format_sft.load_split("validation", "B")` — the instrument used to rank
the run-3 checkpoints. Applying that 9/64 label-quality concern to a run-3 validation score would
attach one instrument's caveat to a different instrument's measurement.

The natural instrument has its own, different limitation: **n = 46 on the abstain
side, so one row ≈ 2.2 pp**. It supports ranking, not claims of difference between
checkpoints. Two of the 46 have a completely empty `ocr_menu`; that alone does
not establish a weakness in image perception.

Recorded while selecting the run-3 checkpoint; the reasoning and the numbers it
produced are in [`analysis/run3_checkpoint_selection.md`](analysis/run3_checkpoint_selection.md) §1–2.
