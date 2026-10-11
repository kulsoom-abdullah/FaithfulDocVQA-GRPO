# SFT pipeline — build notes & engineering decisions (v1)

Narrative record of building the FaithfulDocVQA v1 supervised-fine-tuning pipeline
and the non-obvious problems solved along the way. Intended as the "story" behind
the code for the repo README / portfolio. Every number here is reproduced from a
run, not estimated.

## What was built

A clean-room SFT pipeline that teaches `Qwen3-VL-4B-Instruct` (LoRA, vision
frozen) to answer a document-image question **and cite the OCR token id(s) it used
as evidence**, or **abstain** when the answer isn't recoverable from the OCR menu —
never fabricating a citation. Output is strict two-field JSON
(`{"answer": ..., "evidence_ids": [...]}`); there is no reasoning trace.

Three modules, each with CPU-runnable tests:

| file | role |
|---|---|
| `scripts/faithful_io.py` | Output schema (Pydantic), deterministic `serialize`, tolerant parser, `build_target` from a frozen record. |
| `scripts/format_sft.py` | Chat-message construction, the official Qwen3-VL `apply_chat_template`, the SFT-A/B data filters, the vision-pixel cap. |
| `scripts/train_sft.py` | Module/QKVO confirmation, LoRA config, **explicit completion-only loss mask**, 4-bit QLoRA load, Trainer loop. |

**The experiment:** SFT-A (answerable only, 2003 train) vs SFT-B (answerable +
natural OCR-miss abstains, 2207 train). *(Numbers in this paragraph are from the
initial subsampled build; the frozen master was later rebuilt from the full
eligible set — see Final configuration below: 3423 train / 313 abstains.)* Identical model / LoRA / hyperparameters /
seed — the arms differ **only** by data filter, so any behavioral difference is
attributable to abstain training. Of SFT-B's 204 train abstains, **187 (92%) are
"hard"** (a populated OCR menu where the answer simply isn't present) vs 17
empty-menu — the model is being taught "evidence absent → abstain," not "no tokens
→ give up."

## Locked design choices

- **Completion-only loss, computed explicitly.** Cross-entropy only on the JSON
  completion (+ the `<|im_end|>` turn-end); the prompt — system contract, expanded
  image tokens, question, OCR menu — is masked to `-100`. A past project had this
  silently fail via a TRL convenience flag, so here the boundary is derived by
  rendering the prompt separately, tokenizing it with the *same* image, and
  **asserting the prompt is a token-level prefix of the full sequence** before
  masking. Guarded by a required test (`tests/test_loss_mask.py`).
- **LoRA on language `q/k/v/o_proj` only; vision frozen.** Confirmed against the
  real module graph (not guessed): 36 language layers each expose
  `self_attn.{q,k,v,o}_proj`; the vision tower uses a *fused* `qkv`, so the
  leaf-name target list provably cannot touch vision. **11,796,480 trainable params
  = 0.2438%** of 4.84 B; a hard assertion fails the run if any vision param is
  trainable.
- **Strict JSON contract.** `extra="forbid"`, integer-only `evidence_ids` (no
  bool/float/string coercion). Malformed outputs are *counted*, not repaired — the
  malformed rate is a measured result.

## Problems solved (the debugging story)

These are the bugs that mattered. Most did **not** reproduce on the laptop and only
surfaced on a clean RTX 4090 pod — which is exactly why the staged CPU→GPU smoke
gate exists.

1. **Training silently coupled to the audit module's heavy deps.** `format_sft`
   imported `tokens_to_str` from `phase0/pipeline.py`, which does `import pandas` at
   module top — dragging pandas/pytesseract/anthropic into the *training* path. It
   never showed on the Mac (pandas installed); on the GPU box it was
   `ModuleNotFoundError: pandas`. **Fix:** the training pipeline does no
   OCR/localization (it reads pre-extracted tokens from `master.jsonl`), so it has
   no business importing the audit module. Inlined the 3-line menu renderer and
   removed the `phase0` import; verified **byte-identical** output to
   `tokens_to_str` across all 2600 records. Lesson: decouple at the seam, don't
   `pip install` the coupling away.

2. **transformers 5.12 needs torch ≥ 2.5 for 4-bit.** The RunPod image shipped
   torch 2.4.1; the bitsandbytes 4-bit replacement path calls
   `nn.Module.set_submodule`, added in torch 2.5 →
   `AttributeError: 'Qwen3VLForConditionalGeneration' object has no attribute
   'set_submodule'`. **Fix:** align the stack — torch 2.5.1+cu124 (kept the
   validated transformers). The bootstrap now auto-upgrades only when torch < 2.5.

3. **Threading the Qwen3-VL processor's output fields through tokenize → collate.**
   Two bugs of the same shape, hit in sequence in the model forward:
   - A blanket "drop the batch dim if `len==1`" flattened `image_grid_thw` `(1,3)`
     → `(3,)`; the vision forward did `for t, h, w in grid_thw.tolist()` over ints
     → `TypeError: cannot unpack non-iterable int object`.
   - The collator built a fresh dict with only 5 known keys and **silently dropped
     `mm_token_type_ids`** — a per-token text-vs-image mask Qwen3-VL needs for
     multimodal RoPE → `ValueError: ... mm_token_type_ids is missing`.

   **Fix (made general so a third field can't bite):** squeeze the batch dim only
   for *per-token* fields (1-D, length == sequence length: `input_ids`,
   `attention_mask`, `mm_token_type_ids`); keep the vision tensors
   (`pixel_values` `(num_patches, dim)`, `image_grid_thw` `(num_images, 3)`) as the
   processor returns them. The collator now **forwards every field present** —
   pad+stack the per-token ones, concatenate the vision ones along dim 0 — instead
   of hand-listing a subset. Verified on a mixed-length 2-example batch
   (`input_ids/labels/mm_token_type_ids → (2, 8195)`, `pixel_values → (18304, 1536)`,
   `image_grid_thw → (2, 3)`). Lesson: don't enumerate a processor's outputs; thread
   all of them.

4. **Resolution / `qwen_vl_utils` investigation.** Before adding any resize cap we
   verified what the processor actually does to large scanned pages:
   - The Qwen3-VL processor (`Qwen2VLImageProcessor`, patch 16 / merge 2 → resize
     factor **32**) **preserves document resolution**: its default ceiling is
     ~16.78 MP, and normal pages (~4 MP) pass through at **scale ≈ 1.00** — small
     text is not lost.
   - The common `qwen_vl_utils.process_vision_info` helper bakes in the
     **Qwen2/2.5-VL** factor of **28** (patch 14 / merge 2). Feeding its output to a
     Qwen3-VL processor double-resizes and yields a *different, slightly smaller*
     grid (e.g. 1687-px page → 1696 px via raw PIL vs 1664 px via `qwen_vl_utils`).
     **Conclusion: passing the raw PIL straight to the Qwen3-VL processor is the
     correct path; `qwen_vl_utils` would be the lossy one here.** We do not use it.
   - The only real cost is vision-token count: a normal page is ~3,600–3,900 vision
     tokens; one 38 MP outlier scan was **16,280** — enough to OOM a 24 GB 4090.
     **Fix:** cap `max_pixels` at ~5.24 MP (`= 5120 vision tokens × 1024 px/token`).
     Normal document pages are ≤ ~4.1 MP and pass **unclipped**; only oversized
     outliers downscale. The cap is for memory/throughput, **not** small-text
     fidelity, and is applied identically to SFT-A and SFT-B so the comparison is
     unaffected. *Refined in practice:* after a **train-step-488 OOM** traced to a
     stale `format_sft.py` running *without* the cap on a fresh pod, the cap was
     lowered **5.24 → 4.2 MP (4096 tokens)** for a proven-safe margin (the full run
     handled ~4 MP fine). 90% of pages still pass unclipped; 71/1060 are capped (9
     in eval). `data/images_cap_manifest.json` records per-doc `was_capped` /
     `effective_mp` so eval can report capped-vs-uncapped honestly.

5. **OOM at the loss, and a RAM trap at scale.** With the forward finally running,
   the smoke OOM'd at `logits.float()` in `ForCausalLMLoss`: the model materializes
   **full-sequence** logits over the ~152k vocab, so a ~10k-token page is
   `seq×vocab×4 ≈ 7 GB` in one fp32 tensor, on top of ~13 GB already resident.
   Completion-only training only supervises the ~30 completion tokens, so a custom
   `compute_loss` passes **`logits_to_keep = n_completion + 1`** — the model
   computes logits for just the tail (~18 MB), **mathematically identical** to
   full completion-only CE (verified by a synthetic alignment check before any GPU
   run). Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the ~4 GB of
   fragmentation the trace showed. A second, *scale-only* trap: `train()` eagerly
   tokenized every example into a list, but one example's `pixel_values` is ~90 MB
   → ~200 GB RAM for 2207 examples (invisible at the 20-example smoke). **Fix:** a
   lazy `Dataset` that tokenizes per `__getitem__`, so only the in-flight batch is
   resident. Lesson: the smoke gate catches *per-step* failures; *scale* failures
   (RAM held across the whole dataset) need to be reasoned about separately.

## Verified facts (reproduced from runs)

- Module graph: `Qwen3VLForConditionalGeneration`, 815 modules, 36 language layers;
  trainable **11,796,480 / 4,838,568,448 = 0.2438%**, vision trainable = 0.
- Completion-only mask: prompt region all `-100`, completion region == token ids,
  masked count == prompt length (3 tests).
- Menu renderer byte-identical to `phase0.pipeline.tokens_to_str` across 2600
  records.
- Pixel cap: 38 MP → 5.12 MP / 5002 tokens; 4.03 MP and 3.80 MP pages unchanged.
- Stack: torch 2.5.1+cu124 on RTX 4090, transformers 5.12.0, peft 0.19.1,
  bitsandbytes 0.49.2.
- **Smoke (SFT-B, 20 examples, 5 steps): PASSED** — finite loss (final 0.8895),
  adapter saved and reloaded (`load_adapter`). ~25 s/step at batch 1.

## Final configuration and outcome

**Dataset (frozen, committed):** `data/master.jsonl` = **4932 records / 1234 docs**,
rebuilt from the FULL eligible set (not a subsample). **3-way split by docId,
stratified by doc_type, seed 42:** train **3423** (313 abstain) / validation **753**
(46) / **test 756 (77 abstain)** — no docId leakage. The per-mode filter
(A=answerable-only / B=all) applies to **train only**; **validation + test are full
and byte-identical for SFT-A and SFT-B**. Vision cap **4.2 MP**;
`data/images_cap_manifest.json` records `was_capped`/`effective_mp` per doc for
capped-vs-uncapped eval slicing.

**Preflight tooling for the GPU workflow:** `scripts/check_bnb_4bit.py` is the
recommended 4-bit-kernels preflight on a fresh pod (import-passes-but-kernels-broken
is a real failure mode it catches); `scripts/inspect_sft_3field.py` renders v3
examples for a pre-launch eyeball of prompts/targets.

**Pipeline (final, 15 tests green):** Qwen3-VL-4B-Instruct + LoRA r16/α32 on language
`q/k/v/o_proj` (vision frozen), 4-bit QLoRA. Completion-only loss via
`logits_to_keep` (memory-bounded), lazy tokenization, **batch size 1 for train AND
eval**. `--mode {A,B}`, seed 42, **2 epochs**, checkpoints every **214 steps**
(~0.5 epoch, all kept), W&B (`faithfuldocvqa-sft`, tags `mode-A`/`mode-B`), val loss
logged as an overfitting alarm — **`load_best_model_at_end` OFF**; the best
checkpoint is chosen post-hoc by the faithfulness eval, never by val loss.

**Tooling (volume-less workflow):** `./launch_pod.sh` (Mac) syncs + runs
`scripts/runpod_bootstrap.sh` on any fresh GPU pod; `./launch_pod.sh --pull` retrieves
adapters. Bootstrap installs deps + torch≥2.5 (cu124; Blackwell RTX PRO cards would
need cu128), writes `/workspace/env.sh` (HF cache on disk, `HF_HUB_DISABLE_XET=1`,
TMPDIR), checks data/cap, runs the loss-mask test. Pod container disk ≥50 GB.

**How the runs were executed (for reproduction):**
1. Fresh ≥24 GB Ada/Ampere pod: `./launch_pod.sh` → `python scripts/train_sft.py --smoke --mode B` (confirms `eval_loss` logging + checkpoint + reload).
2. `tmux` → full **SFT-B** (`--mode B`, ~7–9 h, ~856 steps) then **SFT-A** (`--mode A`, ~778 steps) on the **same card type**.
3. `./launch_pod.sh --pull` to save adapters **before terminating the pod**.
4. Eval framework (built afterwards; see `scripts/eval_full.py` + `scripts/eval_scorer.py`):
   valid-citation / honest-abstain / fabrication rates + cited-ID F1, sliced by
   capped-vs-uncapped — SFT-A vs SFT-B results live in `data/checkpoints/eval/`.

**Deferred (NOT v1):** GRPO, the 4-component composite reward, reasoning/CoT targets,
a second (masking) abstain type, image augmentation.
