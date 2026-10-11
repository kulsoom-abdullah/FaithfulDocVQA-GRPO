# GRPO eval re-scoring — committed scorer (scripts.eval_scorer.score)

Instrument reconstructed via committed `build_val_instrument` (seed 42, eval_abstain 64, eval_answerable 48); record_ids match every dump exactly (asserted). n_abstain=64, n_answerable=48. Rates rounded to 4 decimals (val_curve convention).

## (a) Reproduction — re-derived vs val_curve.jsonl
### run1 step 50
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 9/64 | 0.1406 | 0.1406 |  |
| abstain | false_grounding | 55/64 | 0.8594 | 0.8594 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 48/48 | 1.0 | 1.0 |  |
| answerable | false_abstain | 0/48 | 0.0 | 0.0 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run1 step 100
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 9/64 | 0.1406 | 0.1406 |  |
| abstain | false_grounding | 55/64 | 0.8594 | 0.8594 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 48/48 | 1.0 | 1.0 |  |
| answerable | false_abstain | 0/48 | 0.0 | 0.0 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run1 step 150
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 8/64 | 0.125 | 0.125 |  |
| abstain | false_grounding | 56/64 | 0.875 | 0.875 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 48/48 | 1.0 | 1.0 |  |
| answerable | false_abstain | 0/48 | 0.0 | 0.0 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run2 step 25
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 14/64 | 0.2188 | 0.2188 |  |
| abstain | false_grounding | 50/64 | 0.7812 | 0.7812 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 48/48 | 1.0 | 1.0 |  |
| answerable | false_abstain | 0/48 | 0.0 | 0.0 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run2 step 50
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 28/64 | 0.4375 | 0.4375 |  |
| abstain | false_grounding | 36/64 | 0.5625 | 0.5625 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 48/48 | 1.0 | 1.0 |  |
| answerable | false_abstain | 0/48 | 0.0 | 0.0 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run2 step 75
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 39/64 | 0.6094 | 0.6094 |  |
| abstain | false_grounding | 25/64 | 0.3906 | 0.3906 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 47/48 | 0.9792 | 0.9792 |  |
| answerable | false_abstain | 1/48 | 0.0208 | 0.0208 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run2 step 100
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 40/64 | 0.625 | 0.625 |  |
| abstain | false_grounding | 24/64 | 0.375 | 0.375 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 46/48 | 0.9583 | 0.9583 |  |
| answerable | false_abstain | 2/48 | 0.0417 | 0.0417 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run2 step 125
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 47/64 | 0.7344 | 0.7344 |  |
| abstain | false_grounding | 17/64 | 0.2656 | 0.2656 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 46/48 | 0.9583 | 0.9583 |  |
| answerable | false_abstain | 2/48 | 0.0417 | 0.0417 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

### run2 step 150
| gold | bucket | re-derived k/n | re-derived rate | val_curve rate | differs |
|---|---|---|---|---|---|
| abstain | honest_abstain | 46/64 | 0.7188 | 0.7188 |  |
| abstain | false_grounding | 18/64 | 0.2812 | 0.2812 |  |
| abstain | fabrication | 0/64 | 0.0 | 0.0 |  |
| abstain | contradiction | 0/64 | 0.0 | 0.0 |  |
| abstain | malformed | 0/64 | 0.0 | 0.0 |  |
| answerable | valid_citation | 46/48 | 0.9583 | 0.9583 |  |
| answerable | false_abstain | 2/48 | 0.0417 | 0.0417 |  |
| answerable | fabrication | 0/48 | 0.0 | 0.0 |  |
| answerable | contradiction | 0/48 | 0.0 | 0.0 |  |
| answerable | malformed | 0/48 | 0.0 | 0.0 |  |

## (b) Citation quality — citation_f1_gated (never logged in val_curve)
| run | step | n_valid_citation | micro_f1 | macro_f1 |
|---|---|---|---|---|
| 1 | 50 | 48 | 0.8993 | 0.8544 |
| 1 | 100 | 48 | 0.9058 | 0.8596 |
| 1 | 150 | 48 | 0.9203 | 0.8805 |
| 2 | 25 | 48 | 0.8993 | 0.8544 |
| 2 | 50 | 48 | 0.9039 | 0.8572 |
| 2 | 75 | 47 | 0.8971 | 0.8513 |
| 2 | 100 | 46 | 0.9037 | 0.8698 |
| 2 | 125 | 46 | 0.9037 | 0.8698 |
| 2 | 150 | 46 | 0.9037 | 0.8698 |

## (c) found_answer_accuracy — abstain vs answerable, with counts
| run | step | abstain k/n | abstain rate | answerable k/n | answerable rate |
|---|---|---|---|---|---|
| 1 | 50 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 1 | 100 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 1 | 150 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 2 | 25 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 2 | 50 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 2 | 75 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 2 | 100 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 2 | 125 | 61/64 | 0.9531 | 44/48 | 0.9167 |
| 2 | 150 | 61/64 | 0.9531 | 44/48 | 0.9167 |

## (d) Decoy split — abstain rows: recurs x bucket (Fisher exact, two-sided)
### run2 step 150
abstain buckets present: ['false_grounding', 'honest_abstain']
| recurs | false_grounding | honest_abstain | row n |
|---|---|---|---|
| True | 8 | 1 | 9 |
| False | 10 | 45 | 55 |
| col n | 18 | 46 | 64 |
2x2 table [[recurs=True: false_grounding=8, honest_abstain=1], [recurs=False: false_grounding=10, honest_abstain=45]]
Fisher exact two-sided p = 7.48527e-05 (odds ratio 36); n(recurs=True)=9, n(recurs=False)=55

### run2 step 25
abstain buckets present: ['false_grounding', 'honest_abstain']
| recurs | false_grounding | honest_abstain | row n |
|---|---|---|---|
| True | 9 | 0 | 9 |
| False | 41 | 14 | 55 |
| col n | 50 | 14 | 64 |
2x2 table [[recurs=True: false_grounding=9, honest_abstain=0], [recurs=False: false_grounding=41, honest_abstain=14]]
Fisher exact two-sided p = 0.187018 (odds ratio inf); n(recurs=True)=9, n(recurs=False)=55

