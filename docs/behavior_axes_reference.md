# FaithfulDocVQA — Behavior, Ground Truth, and the Two Axes (v1 contract)

**Purpose.** One reference that separates three things people conflate: what the model
*did*, how we *measure* it (the eval bucket), and the two *different questions* we ask of
each behavior — can a deployed pipeline **catch** it without ground truth, and how should
the **reward** treat it during GRPO. The eval bucket and the two axes are computed over the
same facts but answer different questions, and they are allowed to disagree — **with one
exception, made explicit below: the citation-penalty magnitude is owned by the
detectability/deployment thesis, not by the learning axis.**

> **Scope: v1 contract.** In v1 the answer is **always present in the image** by
> construction. There are exactly two ground-truth situations (below). The v2 case —
> answer *masked out of / not perceivable in the image* (where `found_answer="unanswerable"`
> would be the correct output) — is **out of scope here** and is future work. So in v1, a
> model returning `"unanswerable"` is always a **perception miss**, never correct.

---

## Ground-truth situations (what is actually true)

| Situation | Meaning |
|---|---|
| **Answerable — in menu** | The answer to the question is in the image **and** the OCR menu captured it -> the model **should cite**. |
| **Abstain — in image, not in menu** | The answer is in the image but the OCR menu did **not** capture it (OCR miss) -> the model **should abstain** (`answerable_from_menu=false`, empty cite). This is the honest-abstain class. |

*(Excluded, v2: answer not perceivable in the image at all. Not modeled here.)*

## What the model returns (the 3 fields)

- **`found_answer`** (perception axis) — one of:
  - **right** — returned a string equal to a gold answer (read the page correctly).
  - **`"unanswerable"`** — the model reports it could not perceive the answer. **In v1 this is always wrong** (the answer is always in the image).
  - **wrong** — returned a string, but it is not a gold answer (misread).
- **`answerable_from_menu`** (attribution flag) — Boolean: does the model claim the answer is citable from the menu? *(Missing/blank -> malformed, not a value here.)*
- **`evidence_ids`** (the citation) — list of OCR token IDs. Non-empty **iff** the flag is true.

---

## The two axes (and the one rule that keeps them from colliding)

- **Detect @ inference (no GT)?** — Can a deployed pipeline flag this from the **output alone**, with **no ground truth**? A pure mechanical property, not a business-severity claim.
- **Learning treatment** — directional reward shape under GRPO. **Ordering fixed here; magnitudes deferred** to empirical tuning.

> **THE RULE (reconciles this doc with `DECISIONS.md` §4).** In GRPO the reward magnitude
> ordering **IS** the priority ordering — there is no separate priority knob. So the
> **citation-side penalty magnitude** is governed by the **detectability thesis**:
> **`false_grounding` >= `fabrication`** (false_grounding is penalized at least as hard,
> per `DECISIONS.md` §4 — it is the failure the cheap output-only checks CANNOT catch, and
> it is what fabrication *collapses into* if you penalize fabrication harder). The learning
> axis's "false_grounding cited a real token, so it's closer to correct" intuition is real,
> but it is expressed on the **separate `found_answer` axis** (it read the page -> partial
> credit there), **NOT** by softening the citation penalty. Dangerous-but-undetectable
> stays maximally penalized; the smooth learnable surface comes from the perception axis
> and the F1 gradient, not from making the worst failure cheaper.

---

## The table

### Gold = ANSWERABLE (answer is in the menu -> should cite)

| Model behavior | found_answer | flag | evidence_ids | Eval bucket | Detect @ inference (no GT)? | Learning treatment (reward) |
|---|---|---|---|---|---|---|
| **Correct / partial cite** | right | true | some-or-all correct **in-menu** IDs | `valid_citation` | No — looks valid; ID quality needs GT | **+ F1** (dense: all-correct > partial > real-but-none) |
| **Fabricated cite** | right | true | >=1 ID **not in menu** | `fabrication` | **YES** — ID out of menu range, output-only check | **heavy -** (didn't consult the menu; strictly below honest abstain) — **but <= false_grounding in magnitude** |
| **False abstain** | (perceived) | false | `[]` | `false_abstain` | No — can't tell a warranted refusal from an unwarranted one without GT | **-** (refused a citable answer; anti-over-abstention) |
| **Wrong answer** | **wrong** | true | any **in-menu** IDs | `valid_citation` | No — needs GT to know the answer is wrong | **-** on the **separate `found_answer` axis** (bucket unchanged) |
| **"Unanswerable" on answerable** | `"unanswerable"` | true | in-menu IDs | `valid_citation` | No | **-** on the **`found_answer` axis** (perception miss; a subtype of wrong-answer) |

### Gold = ABSTAIN (answer in image but NOT in menu -> should abstain)

| Model behavior | found_answer | flag | evidence_ids | Eval bucket | Detect @ inference (no GT)? | Learning treatment (reward) |
|---|---|---|---|---|---|---|
| **Honest abstain** (the win) | right | false | `[]` | `honest_abstain` | No — honest vs false abstain indistinguishable without GT | **+** reward the win |
| **Image miss** | `"unanswerable"` | false | `[]` | `honest_abstain` | No (bucket-wise) | **-** on the **`found_answer` axis** (didn't read the image -> anti-collapse guard) |
| **FALSE GROUNDING** | right | true | any real cite | `false_grounding` | **NO — identical to a valid citation on the output.** Needs GT to know the claim is false. | **heaviest -** (>= fabrication) — the undetectable failure; "read the page" credit goes to the `found_answer` axis, **not** here |

### Parse-level (checked first, before any ground truth is consulted)

| Model behavior | flag | evidence_ids | Eval bucket | Detect @ inference (no GT)? | Learning treatment |
|---|---|---|---|---|---|
| **Ungrounded** (flag yes, cited nothing) | true | `[]` empty | `contradiction` | **YES** — internal inconsistency, output-only | partial-credit-eligible by direction (got the flag right) |
| **Answered anyway** (flag no, cited) | false | non-empty | `contradiction` | **YES** — internal inconsistency, output-only | worse direction (flag wrong) |
| **Malformed** | — | — | `malformed` | **YES** — parse fails | format floor (separate) |

### The 7th bucket — `UNCITED_ANSWER` (the `else` branch)

| Aspect | Detail | Action |
|---|---|---|
| **Reachability** | The `else` in `classify()`: `answerable_from_menu` neither `True` nor `False` after a *successful* parse. Under the strict parser (`parse_tolerant_v3` yields a Boolean flag or a parse failure) this is unreachable. | **RESOLVED** — `grpo/reward.py:84` documents `uncited_answer` as unreachable under the 3-field contract and raises loudly if it ever fires. |
| **Reward** | The shipped reward treats it as unmapped and raises (`reward.py:81-86`) rather than silently scoring; `eval_scorer.py` deliberately keeps the defensive `else` branch so classification can never drop a record. | Every bucket accounted for: defined reward *or* a loud failure — as built. |

---

## The pair that makes the whole doc worth having: false_grounding vs fabrication

They rank **opposite** on the two axes — and the RULE above decides which axis owns the citation penalty:

| Failure | Detect @ inference (no GT)? | Citation penalty (owned by detectability) | Learning/perception surface |
|---|---|---|---|
| **Fabrication** | **YES** — free ID-range check; never reaches production un-flagged | heavy, but **<= false_grounding** | worst-behaved (invented tokens); no perception credit path here |
| **False grounding** | **NO** — looks exactly like a valid citation; slips the free check | **heaviest (>= fabrication)** | it read the page -> partial credit on the **`found_answer` axis**, not the citation penalty |

- **Why false_grounding is the train-away priority:** it's the only **hard** failure the cheap output-only checks **cannot** catch, *and* it's the reward-hacking sink fabrication collapses into once you penalize fabrication. Making it cheaper than fabrication would build a downhill path straight into it. (The Feb-7-L.A. survivor exemplar is that hack in miniature.)
- **Where the "warmth" goes:** onto the `found_answer` axis, so the model still gets a smooth gradient for reading the page correctly — without ever making the undetectable failure cheaper on the axis that steers grounding.

---

## Notes

- **Eval bucket** = *offline measurement*. Needs ground truth (`gold_localized`); exists so we can report "X% false_grounding, Y% honest_abstain." **Not** the training signal, **not** what a deployed pipeline sees.
- **Detect @ inference** = *output-only, no ground truth* — what a deployed pipeline can flag. Structure of `classify()`: its first three rungs — **contradiction -> malformed -> fabrication** — are precisely the ground-truth-free checks; everything below consults `gold_localized`. **That line is the flowchart** (output-only checks above, ground-truth checks below).
- **Learning treatment** = the GRPO reward shape. **Ordering fixed, magnitudes deferred.** Group-relative advantage means within-prompt *ordering* matters more than absolute values — which is exactly why the false_grounding >= fabrication ordering is the load-bearing decision, not the magnitudes.
- **Contradiction is not flat-bad.** "flag right, empty cite" is directionally more correct than "flag wrong, cited anyway"; the reward can give the first partial credit even though the eval bucket labels both `contradiction`. Measurement can be a flat label while the reward reads the underlying facts.
- **found_answer is a separate axis** (image-use). A wrong or `"unanswerable"` perception is penalized on that axis and does **not** change the citation bucket — the anti-collapse guard, so the model can't dodge penalties by abstaining on everything, and the channel that carries all "it read the page" partial credit.

---

*v1 contract; answer always in the image. Buckets/fields are the 3-field (v3) scorer; the
two ground-truth situations are the v1 answerable/abstain split. Citation-penalty ordering
(`false_grounding` >= `fabrication`) is fixed per `DECISIONS.md` §4; all learning-axis
magnitudes are placeholders pending the staged GRPO rollout.*
