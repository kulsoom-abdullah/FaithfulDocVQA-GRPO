# FaithfulDocVQA — Locked Scope (v1)

> **Historical record — locked June 2026.** Statements below were true at lock:
> pre-SFT, 2-field contract, Qwen2.5-VL-3B as base, GRPO deferred. The project
> later moved to Qwen3-VL-4B, the 3-field (v3) contract, and an executed GRPO
> stage — see `docs/PROVENANCE.md` + `docs/DECISIONS.md`. The locked body below
> is kept as written.

**Purpose of this document:** the fixed definition of "done" for v1. Hold every
proposed task against it. If a task isn't in §2 (v1 deliverables), it is either
deferred (§5) or a rabbit hole (§6). This file is the anti-drift artifact —
update it deliberately, not per-session.

**One-line project definition:** a document-VQA system that cites valid
OCR-extracted token IDs as evidence, abstains honestly when the answer cannot be
grounded in the OCR menu, and does not fabricate citations.

**Status at lock:** Phase 0 (adversarial audit) and capture/localizer studies
complete. No SFT data, training, or GRPO code exists yet. v1 builds the data →
SFT → eval axis from this base. GRPO is explicitly NOT part of v1.

---

## 1. Locked design decisions

These do not get relitigated. Each was decided with evidence; the evidence is noted.

**Data construction = Path B (programmatic gold-anchored), no teacher.**
Targets are built deterministically from DocVQA gold answers + Tesseract OCR +
the gold localizer. No teacher distillation. Justification: Phase 0 proved even a
frontier teacher (Sonnet) fabricated a token ID outside the menu, so for a
*faithfulness* task we anchor targets to verifiable localization rather than
distilling potentially-fabricated reasoning. Cheaper, deterministic, stronger story.

**SFT target = citation-only.** The model emits: answer text, cited OCR token
IDs, and an abstain flag. Cross-entropy on the completion only. The model does
NOT emit evidence-span text.

**Prompt/label interface (confirmed in pipeline.py):**
- Prompt carries the OCR menu as `(id, word, bbox)` per token. Bounding boxes are
  *context*, not a citation target.
- Label/citation unit is the token **ID only**. Reward scores ID set-overlap;
  bbox is never scored.

**Clean images only in v1.** No augraphy augmentation, no masking. Justification:
augmentation changes the OCR token set per example (label space moves),
confounding any faithfulness metric; it was also the sole cause of the Phase 0
reproducibility bug (unseeded augraphy). Dropping it makes the v1 pipeline
deterministic end-to-end.

**OCR menu frozen as-is; no DPI handling.** Confirmed empirically (n=5 probe):
forcing DPI 300 produced *identical* token counts to the pipeline default on all
probe images. DocVQA images are large enough in pixels that the DPI tag is
cosmetic. Tesseract is deterministic on fixed pixels → menu is reproducible.

**Abstain class = natural OCR-miss only (one type in v1).** When the gold answer
cannot be localized in the OCR menu, the example becomes an abstain target with
`abstain_reason = "ocr_miss"`. Measured size: **8.6% of the SFT-viable slice**
(n=500, 95% CI 6.4–11.4%), 0/43 fixable by normalization — all genuine misses.
The schema carries `abstain_reason` so a second type can be added later (§5)
as a data change, not a rewrite.

**Dataset = DocVQA validation split, unique-by-docId, split BY docId.**
1,286 unique docs behind 5,349 QA rows (4.16 Q/doc). Download unique images
(~655 MB), fits laptop. Split MUST be by docId, not by question — 4.16 Q/doc
means question-level splitting leaks the same document across train/eval, which
for a faithfulness project is a disqualifying bug.

**Excluded doc types:** handwritten (15.4% localized) and Image/Photo (44.9%)
are unfit for ID-grounded SFT and are excluded from v1 train/eval.

**Reward function = eval scorer in v1 (NOT used for SFT).** SFT is imitation
(cross-entropy on correct targets); the reward function is the *eval* scorer now
and the *GRPO* signal later. The 4-component weighted composite
(0.50/0.25/0.10/0.15) is DESIGNED, not implemented, and is deferred to GRPO. v1
uses the evidence component (as built) to report rates.

---

## 2. v1 deliverables — the "done" bar

v1 is finished when ALL of these exist and are honestly documented:

1. **Frozen dataset** — built once, persisted to disk, deterministic. Each record:
   question, image ref, OCR menu `(id, word, bbox)`, the SFT target (answer +
   cited IDs, or abstain + empty IDs + `abstain_reason="ocr_miss"`), and the
   frozen `gold_cited_ids`. Split by docId. This single artifact serves as
   reproducibility anchor + SFT training data + eval gold.

2. **SFT-A trained** — Qwen2.5-VL-3B + LoRA, trained on **answerable examples
   only** (OCR captured the answer → answer + cited IDs). The control.

3. **SFT-B trained** — same base, trained on **answerable + OCR-miss abstains**.
   The treatment.

4. **Eval framework** — runs both models on the same held-out (docId-split) set
   and reports, per model: valid-citation rate, honest-abstain rate, fabrication
   rate, and cited-ID F1. **Abstains tagged by reason** (v1: all `ocr_miss`).

5. **Audit/eval report** — SFT-A vs SFT-B compared on the metrics above. Numbers
   framed honestly against the ~83% OCR grounding ceiling and the 8.6% abstain
   class. The core finding is the A-vs-B contrast (does abstain training change
   fabrication/abstention behavior, and by how much).

6. **Repo presentation** — README, model card, one reproducible run. README
   includes the deployment-considerations prose (§4).

**The portfolio thesis (what the report argues):** anchoring SFT targets to
verifiable OCR localization (vs. teacher distillation) produces a model that
abstains where evidence is missing rather than fabricating; the SFT-A vs SFT-B
contrast measures whether explicit abstain training is necessary or whether the
base model already has latent abstention.

---

## 3. The central experiment (do not collapse this)

SFT-A and SFT-B are BOTH required. SFT-A is the control; without it there is no
reference point and "SFT-B abstains X%" is uninterpretable. Whichever way the
result falls — "abstain training was necessary" or "base model already abstained,
training sharpened it" — both are publishable findings, and neither is available
without running A. This is the same negative-result-is-a-contribution discipline
as the prior DPO study.

---

## 4. Data-composition requirement (from localizer findings)

The genuine-OCR-miss abstain class is ~84% tabular (table/list 18, layout 10,
form 8 of 43). Risk: SFT-B could learn "table → abstain" as a shortcut instead of
"missing evidence → abstain." **Mitigation (required, not optional):** SFT-B's
*answerable* set must include ample table/list examples where OCR DID capture the
answer (89% of tables localize, so supply is not a constraint). Tables must
appear in both classes so the abstain signal is "evidence present/absent," not
"is a table." This is a stratification rule for the data builder — no extra code.

---

## 5. Deferred — legitimate, queued, NOT v1

These are real and planned; they are not "done" criteria. Noting them here is what
keeps them off the v1 critical path.

- **GRPO** — the full reward function (composite, or the evidence reward as RL
  signal) consumes a *spread* of model generations. v1 ships SFT + eval first.
- **Second abstain type (true-absent via masking)** — teaches "answer is not in
  the document at all," distinct from OCR-miss. Needs a reward branch to separate
  the two abstain reasons. Schema's `abstain_reason` already supports it →
  v2 is a data + reward-branch change, not a rewrite. (Open question flagged for
  mentor review: is one abstain type enough for a complete faithfulness story, or
  does the two-type contrast make it materially stronger?)
- **Augmentation-robustness experiment** — seeded augraphy, as a *controlled*
  follow-up: train clean vs. clean+augmented, both eval'd on degraded images.
  Separate, named experiment — never folded into the v1 baseline.
- **Teacher rationale traces** — if reasoning-text targets are ever wanted, a
  Path-A add-on; schema allows it.
- **All-gold-variant localization** — using all gold spellings (not just first)
  recovered 2/500 in the study. Trivial optional improvement, not v1-blocking.

---

## 6. Rabbit holes — actively avoid

- **OCR/Tesseract tuning** — measure only. 0/43 misses were fixable by
  normalization; chasing OCR quality buys ~0 and is not the project.
- **DPI handling / resampling in v1** — confirmed cosmetic for this dataset. The
  production resolution-check logic goes in the README as prose (§4 deployment
  note), NOT in the v1 pipeline.
- **PaddleOCR swap, NemoTron Parse** — out.
- **Evidence-text matching** — correctly dropped already; do not reintroduce.
- **Composite reward implementation** — deferred to GRPO; do not build for v1.
- **Deleting Phase 0 augmentation/masking code** — leave it; Phase 0's Cases B/C
  used it. v1 simply doesn't call it. Cosmetic cleanup = zero job value.

---

## 7. README deployment-considerations (prose, not built)

Written into the README to show production awareness without expanding v1:
- **Open-weight (built) vs managed-closed (discussed):** v1's trained artifact is
  Qwen2.5-VL-3B + LoRA, self-hosted on RunPod. README discusses the alternative —
  managed hosted-LoRA (e.g. Gemini): closed weights, pay-per-call, no GPU hosting
  — with the host-your-own vs pay-per-call tradeoff. Demonstrates systems
  judgment; only the open-weight path is actually built.
- **Input resolution handling:** real-world inputs vary in resolution (this
  dataset happens to stay above Tesseract's usable pixel-height threshold; the
  n=500 sample ranged 850–2284 px wide). A production system would check
  character pixel-height on ingest and upscale low-res inputs toward ~300 DPI
  equivalent, passing through adequate ones and capping the upper end for speed.
  Described, not implemented.

---

## 8. Tooling (confirmed)

- Compute: RunPod (training); local Apple Silicon, Python 3.12, uv (dev).
- OCR: Tesseract 5.5.1 via pytesseract, `image_to_data`, pipeline-default
  invocation (no forced DPI), raw PIL image (mode L, no RGB convert/re-encode).
- Base model: Qwen2.5-VL-3B; LoRA via PEFT (r=16, language-tower Q/K/V/O,
  vision tower frozen — per prior design).
- Data: lmms-lab/DocVQA validation split, cached locally (~3.3 GB arrow;
  ~655 MB if materialized unique-by-docId).

---

## 9. Working pattern

Specs are agreed in design discussion first, then executed via Claude Code on
laptop or RunPod, and results brought back for review. Scope locked before code.
Honest framing, no overclaiming — audit every claim against actual code/data
before it is published anywhere.
