# Manufactured-build recovery (read-only)

_norm helper: scripts.eval_scorer._norm (committed). first_doctype + drop/renumber replicated (NOT committed).

## PART A — recover the TEST build's selection filters (from committed data)

- source split dist   : {'test': 250}
- source doc_type[0]  : {'free_text': 42, 'form': 47, 'layout': 71, 'table/list': 90}
- source cite-span len: {1: 138, 2: 72, 3: 40}
- gold_answers: min normed char-len = 1 | isdigit-only answers = 72 | answers <3 chars = 2
- selected  : isdigit=28.80%  <3char=0.80%
- NOT-selected eligible: isdigit=33.55%  <3char=22.70%

**VERDICT:** Selection = doc_type∈['form', 'free_text', 'layout', 'table/list'], cite-span∈[1,3], min answer chars=1, isdigit answers present? Y. Extra short/numeric filter applied? **YES** — evidence: selected isdigit 29% vs not-selected 34%; short 1% vs 23%. (250 of 554 eligible → the shortfall is a short/numeric filter.)

## PART B — decoy/recurrence prevalence in TRAIN

recurrence rule = _norm(gold) is a substring of the space-joined _norm(remaining menu text) [build's rule]

- Pop1 recurs (decoy) overall : 562/2436 = 23.1%
- Pop1 single (1-token) decoy : 384/1389 = 27.6%
- Pop1 multi  (2-3tok) decoy  : 178/1047 = 17.0%
- Pop2 recurs in UNMODIFIED menu : 25/313 = 8.0% (expected small)

| population | N | recurs (decoy) rate |
|---|--:|--:|
| train eligible sources (Pop1) | 2436 | 23.1% |
| train natural abstains (Pop2)  | 313 | 8.0% |

## STEP Z — disjointness assertions

- assert train∩test doc_ids == ∅  : **PASS** (overlap=0 [])
- assert all train sources split==train : **PASS**
