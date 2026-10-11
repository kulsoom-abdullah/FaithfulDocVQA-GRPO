# Reward Design — FaithfulDocVQA-GRPO

## 1. TL;DR

The reward is **verifiable**: deterministic, replayable offline, no judge model. Every component reduces to either a deterministic string function (ANLS, regex, substring) or integer set operations over Tesseract OCR token IDs (the evidence component). The model's `<evidence_tokens>` citation lives in the same ID space as the OCR menu generated offline, so every citation is checkable by `set.issubset(valid_ids)` at training time. Same input → same reward across runs, debuggable on a single GPU, no external API, no LLM-judge noise to fight under GRPO.

## 2. Components and weights

| Component | Weight | Implementation |
|---|---:|---|
| Answer | **0.50** | ANLS — character-level normalized Levenshtein ratio against each gold string, hard-zeroed below 0.5, max across golds. |
| Evidence | **0.25** | Piecewise function over integer sets; defined in §3. |
| Format | **0.10** | Regex: all five required XML tags (`<rationale>`, `<locate>`, `<evidence_tokens>`, `<derive>`, `<answer>`) present. |
| Refusal | **0.15** | Binary substring check on `"unanswerable"` in the parsed answer, matched against the refusal expectation. |

Correctness dominates because answer correctness is the user-visible failure mode. Evidence at 0.25 is large enough to shape grounding behavior but small enough that the model isn't pushed to cite plausible-looking junk when no real evidence exists. Format saturated at 100% across all 27 Phase 0 calls, so its weight is held low. Refusal at 0.15 reflects that calibrated abstention matters but is secondary to not fabricating positively.

## 3. The piecewise evidence component

Four mutually exclusive regimes:

```
is_answerable == False  (refusal-expected):
    cited == ∅            →  +1.0    correct refusal
    cited != ∅            →   0.0    over-citation on unanswerable

is_answerable == True:
    cited == ∅            →  +0.4    ABSTAIN_CREDIT      (honest abstain)
    any cited ∉ valid_ids →  -0.5    FABRICATION_PENALTY
    all cited ∈ valid_ids →   F1 set-overlap vs gold_cited_ids
```

The collision this fixes is the v1 reward's degeneracy on the canonical `img2_caseC` failure. Under pure F1, "correct answer with empty citation" and "correct answer with fabricated citation" both produced Evidence = 0 → composite 0.60 at the standard weights. The model had no reward incentive to abstain over fabricate. Under the piecewise reward, abstention strictly outscores fabrication by construction (`+0.4 > -0.5`), and the same ordering carries through to the composite.

Worked math for img2 (correct answer, valid XML, refusal neutral):

| Strategy | Evidence | Ev × 0.25 | **Composite** |
|---|---:|---:|---:|
| Valid citation matching gold | **+1.00** | +0.250 | **0.850** |
| Honest empty (`cited=[]`) | **+0.40** | +0.100 | **0.700** |
| Fabricated cite (`cited=[975]`) | **−0.50** | −0.125 | **0.475** |

Ordering 0.850 > 0.700 > 0.475 — the property the v1 reward violated.

Re-scoring the 27 Phase 0 records under the new function moved exactly two: `img1_caseB` and `img1_caseB_timed`. On both, Augraphy noise collapsed Tesseract output to zero tokens; the model answered correctly from vision but cited `[1,2,3,4]` against an empty menu. Under v1 they scored F1 = 0.0, and in my Phase 0 taxonomy I had labeled this **"correct under contract"** — meaning the F1=0 outcome was the right one even though the model had fabricated. That label was a workaround for the reward's degeneracy, not a real classification. Under v2 those records score **−0.5**: the model fabricated, the reward says so directly, the bucket retires. Designing the new reward function corrected my own Phase 0 taxonomy.

## 4. Empirical calibration

Across the full DocVQA validation split (n = 5,349; single-pass Tesseract on each image, sliding-window fuzzy match against every gold answer, best score wins):

- **71.7%** localized (≥ 0.85) — gold answer recoverable directly from the OCR menu
- **11.0%** near-miss (0.60–0.85) — recoverable via lenient threshold cascading
- **17.3%** hard fail (< 0.60) — gold not present at any reasonable threshold

The 17.3% hard-fail rate is what justifies abstention having its own reward weight rather than being treated as a special case of grounding. About one in six training records is structurally unfit for ID-based citation — the gold answer is genuinely not in the OCR menu (handwriting, photo-cells, OCR collapse on noisy paper). On that slice, the only correct grounding behavior is the empty citation. If empty citation didn't score strictly above fabrication, the model would learn to fabricate against those records — exactly the failure mode Phase 0 surfaced on `img2_caseC`, where Sonnet cited token ID `[975]` in a 145-token menu.

The per-question-type capture gradient is the second piece of empirical evidence and decides slice-level inclusion in the 2k SFT distribution:

| question_type | localized | near | fail |
|---|---:|---:|---:|
| `free_text` | 87.5% | 5.5% | 7.1% |
| `Yes/No` | 78.6% | 14.3% | 7.1% |
| `others` | 75.8% | 12.3% | 11.9% |
| `layout` | 74.5% | 8.4% | 17.1% |
| `table/list` | 70.2% | 13.6% | 16.2% |
| `form` | 63.6% | 13.7% | 22.7% |
| `figure/diagram` | 55.8% | 15.1% | 29.1% |
| `Image/Photo` | 44.9% | 11.2% | 43.9% |
| `handwritten` | **15.4%** | 25.7% | **58.9%** |

`handwritten` and `Image/Photo` are unfit for ID-grounded training: combined hard-fail above 40%, with `handwritten` localizing on only 15.4% of records. This substantiates the v1 scope decision to exclude both from the SFT distribution and route them through a separate track (TrOCR or manual bbox annotation) in v2. The remaining six categories all clear ≥55% localization and are viable.

## 5. Why ID-only, no text-match on `<evidence_text>`

The original Phase 0 write-up (an unshipped draft) described the evidence reward as "token-ID plus fuzzy text-match against the cited evidence string." The text-match was never implemented in code, and the capture study closes the case for adding it.

A `<evidence_text>`-vs-gold text-match could in principle recover some portion of the 11% near-miss band — records where the gold string is in the OCR menu but at sub-threshold similarity. It cannot recover any of the 17% hard-fail band, where the gold answer is not in the menu at all. The OCR pipeline, not the matching rule, sets the ceiling: **~83% of validation records have usable gold under any text-match scheme**, with the marginal recovery from text-match bounded by some fraction of 11%.

Against that, adding `<evidence_text>` to the reward expands the reward-hacking surface. The model can win the evidence component by emitting plausible text in `<evidence_text>` that was never grounded in cited IDs — a hallucinated quote that happens to look valid against the gold answer string. ID-based F1 closes that loophole: the model must cite real menu IDs whose words form the gold. The marginal capture recovery from text-match doesn't justify reopening that surface.

The reward stays ID-only.
