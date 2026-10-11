"""Generation core for the FaithfulDocVQA eval + a checkpoint-0.5 format smoke.

This is the reusable INFERENCE half of the eval framework: load a trained QLoRA
adapter under the EXACT 4-bit config training used, render the prompt-only chat
exactly as training did, and greedily generate one JSON answer per record. The
full per-checkpoint A-vs-B eval (later) wraps :func:`generate` and feeds the raw
outputs to ``scripts.eval_scorer.score``.

The CLI runs a small SMOKE on a checkpoint: 5 validation records (3 answerable +
2 abstain), printing each raw output, its canonical three-field classification,
and whether generation stopped on EOS or hit the length cap. PASS = all 5 satisfy
the current JSON contract, including cross-field consistency.

This checks **format coherence / 4-bit numeric sanity** (does the adapter emit
well-formed JSON on Blackwell sm_120) — NOT correctness. Do not read accuracy at
0.5 epoch.

GPU-only (bitsandbytes); heavy deps (torch/transformers/peft/PIL) are imported
lazily so the module imports and ``py_compile``s on a CPU laptop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_sft import build_bnb_config
from scripts.format_sft import (
    MODEL_ID,
    build_messages,
    load_processor,
    load_split,
    render_example,
)
from scripts.eval_scorer import classify


# ── model load (mirrors training exactly, inference adapter path) ─────────────

def resolve_adapter_dir(arg: str) -> Path:
    """Resolve ``--adapter`` to a concrete PEFT adapter directory.

    Accepts either a checkpoint dir (has ``adapter_config.json``) or a run dir;
    for a run dir the LOWEST-step ``checkpoint-N`` is chosen (the 0.5-epoch one).
    """
    p = Path(arg)
    if (p / "adapter_config.json").exists():
        return p
    cks = sorted(
        (d for d in p.glob("checkpoint-*") if (d / "adapter_config.json").exists()),
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not cks:
        raise FileNotFoundError(
            f"{p} is neither a PEFT adapter dir nor a run dir with checkpoint-*/ "
            f"adapters. Pass a checkpoint directory directly."
        )
    return cks[0]


def load_adapted_model(adapter_dir: str | Path, model_id: str = MODEL_ID):
    """Load the base in 4-bit (same ``build_bnb_config`` as training) + the adapter.

    Mirrors ``train_sft.load_qlora_model`` LOAD exactly (same base id, same NF4 +
    double-quant + bf16-compute quantization, same processor with the 4.2 MP pixel
    cap), then attaches the SAVED adapter via ``PeftModel.from_pretrained`` — the
    correct inference path (no ``prepare_model_for_kbit_training``/``get_peft_model``).

    Returns ``(model, processor, resolved_base_id)``.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError("eval_runner needs a CUDA GPU (4-bit bitsandbytes); none found.")

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter_config.json in {adapter_dir}")

    proc, resolved = load_processor(model_id)            # applies MAX_PIXELS cap
    base = AutoModelForImageTextToText.from_pretrained(
        resolved,
        quantization_config=build_bnb_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.use_cache = True                         # generation (training had it off)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, proc, resolved


def load_base_model(model_id: str = MODEL_ID):
    """Load the base VLM in 4-bit with NO adapter — the pre-SFT baseline generator.

    Identical to :func:`load_adapted_model`'s base load (same ``build_bnb_config`` NF4
    quantization, same processor + MAX_PIXELS cap) but WITHOUT the
    ``PeftModel.from_pretrained`` step, so generation reflects the untuned base model.
    Returns ``(model, processor, resolved_base_id)``.
    """
    import torch
    from transformers import AutoModelForImageTextToText

    if not torch.cuda.is_available():
        raise RuntimeError("eval_runner needs a CUDA GPU (4-bit bitsandbytes); none found.")

    proc, resolved = load_processor(model_id)
    base = AutoModelForImageTextToText.from_pretrained(
        resolved,
        quantization_config=build_bnb_config(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.use_cache = True
    base.eval()
    return base, proc, resolved


# ── prompt render (identical code path to training's prompt prefix) ───────────

def render_prompt(processor, record: dict) -> str:
    """Prompt-only chat render — the SAME call ``tokenize_example`` uses for its
    masked prefix (``include_target=False`` + ``add_generation_prompt=True``), so
    the train and generation prompts cannot diverge by construction."""
    return processor.apply_chat_template(
        build_messages(record, include_target=False),
        tokenize=False,
        add_generation_prompt=True,
    )


def prompt_parity(processor, record: dict) -> bool:
    """True iff the generation prompt equals the training-side render
    (``format_sft.render_example``'s ``prompt_text``). Used to RULE OUT a
    train↔gen divergence (cause b) before blaming 4-bit numerics (cause a)."""
    return render_prompt(processor, record) == render_example(processor, record)["prompt_text"]


def _eos_ids(processor) -> list[int]:
    """The stop-token id set: the tokenizer's eos plus ``<|im_end|>`` (the Qwen
    assistant-turn terminator), so greedy decoding actually stops."""
    tok = processor.tokenizer
    eos: set[int] = set()
    raw = tok.eos_token_id
    if isinstance(raw, (list, tuple)):
        eos.update(int(i) for i in raw if i is not None)
    elif raw is not None:
        eos.add(int(raw))
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0 and im_end != tok.unk_token_id:
        eos.add(int(im_end))
    return sorted(eos)


# ── generation ────────────────────────────────────────────────────────────────

def generate(model, processor, record: dict, max_new_tokens: int = 128,
             prompt: str | None = None) -> dict:
    """Greedily generate one answer for a record (batch 1).

    Returns ``{"raw": str, "stop_reason": "eos"|"length_cap"|"other",
    "n_new_tokens": int}``. ``stop_reason`` distinguishes a clean EOS stop from a
    length-cap ramble — targets are ~30-token JSON, so a ``length_cap`` result is
    most likely a missing stop token, NOT numeric garbage.

    ``prompt`` overrides the rendered chat string (default: the normal menu prompt
    via :func:`render_prompt`). The image is always loaded from ``record`` — this
    lets the image-ablation eval feed a coherent no-menu prompt over the same image
    without changing the generation path.
    """
    import torch
    from PIL import Image

    tok = processor.tokenizer
    if prompt is None:
        prompt = render_prompt(processor, record)
    img = Image.open(ROOT / record["image_path"])        # no .convert — matches training
    inputs = processor(text=[prompt], images=[img], return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)

    eos = _eos_ids(processor)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if eos else None)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos or None,
            pad_token_id=pad_id,
        )

    new_ids = gen[0][prompt_len:]
    raw = processor.batch_decode(
        [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    new_list = new_ids.tolist()
    n_new = len(new_list)
    if any(t in set(eos) for t in new_list):
        stop_reason = "eos"
    elif n_new >= max_new_tokens:
        stop_reason = "length_cap"
    else:
        stop_reason = "other"
    return {"raw": raw, "stop_reason": stop_reason, "n_new_tokens": n_new}


# ── smoke ─────────────────────────────────────────────────────────────────────

def evaluate_output(record: dict, raw_output: str | None) -> dict:
    """Parse and classify one current three-field output on CPU.

    This deliberately delegates to :func:`scripts.eval_scorer.classify`, whose
    first step is the canonical ``parse_tolerant_v3`` path. It gives the smoke
    runner and CPU tests one public entry point without duplicating contract or
    bucket logic.
    """
    return classify(record, raw_output)


def pick_smoke_records(n_answerable: int = 3, n_abstain: int = 2,
                       min_menu: int = 8, max_menu: int = 60) -> list[dict]:
    """Deterministically pick smoke records from the shared validation split.

    Both classes are drawn from records with a *legible* menu (``min_menu``..
    ``max_menu`` tokens) so the abstain cases test "answer absent from a populated
    menu" rather than the degenerate empty-menu page; falls back to the smallest
    available if too few clear the filter.
    """
    val = load_split("validation", "B")

    def smallest_legible(predicate, k: int) -> list[dict]:
        pool = sorted((r for r in val if predicate(r)), key=lambda r: len(r["ocr_menu"]))
        legible = [r for r in pool if min_menu <= len(r["ocr_menu"]) <= max_menu]
        return (legible or pool)[:k]

    pick_ans = smallest_legible(lambda r: not r["target_is_abstain"], n_answerable)
    pick_abs = smallest_legible(lambda r: r["target_is_abstain"], n_abstain)
    return pick_ans + pick_abs


def smoke(adapter_arg: str, max_new_tokens: int = 128, show_prompt: bool = False) -> bool:
    """Run the format/numeric-sanity smoke on a checkpoint. Returns True on PASS."""
    adapter = resolve_adapter_dir(adapter_arg)
    print(f"[load] adapter : {adapter}")
    model, proc, resolved = load_adapted_model(adapter)
    device = next(model.parameters()).device
    print(f"[load] base    : {resolved}")
    print(f"[load] quant   : 4-bit NF4 + double-quant, bf16 compute (mirrors train_sft)")
    print(f"[load] dtype   : torch.bfloat16   device: {device}")
    print("[note] FORMAT/NUMERIC SANITY ONLY — not a correctness check at 0.5 epoch\n")

    records = pick_smoke_records()
    if show_prompt and records:
        print("---- rendered prompt (record 1) ----")
        print(render_prompt(proc, records[0]))
        print("-" * 70)

    n_contract_valid = 0
    parity_ok = 0
    stops = {"eos": 0, "length_cap": 0, "other": 0}
    for i, r in enumerate(records, 1):
        cls = "abstain" if r["target_is_abstain"] else "answerable"
        par = prompt_parity(proc, r)
        parity_ok += int(par)
        out = generate(model, proc, r, max_new_tokens=max_new_tokens)
        stops[out["stop_reason"]] = stops.get(out["stop_reason"], 0) + 1
        facts = evaluate_output(r, out["raw"])
        contract_valid = not facts["malformed"]
        n_contract_valid += int(contract_valid)
        print(f"[{i}/{len(records)}] record_id={r['record_id']} class={cls} "
              f"menu={len(r['ocr_menu'])} | stop={out['stop_reason']} "
              f"n_new={out['n_new_tokens']} | parity={'OK' if par else 'MISMATCH'}")
        print(f"   RAW   : {out['raw']!r}")
        if contract_valid:
            print(f"   PARSED: found_answer={facts['found_answer']!r} "
                  f"answerable_from_menu={facts['answerable_from_menu']} "
                  f"evidence_ids={facts['cited_ids']} bucket={facts['bucket']}")
        else:
            print(f"   PARSED: None  ({facts['parse_failure'] or 'malformed'})")
        print()

    passed = n_contract_valid == len(records)
    other = f" / {stops['other']} other" if stops.get("other") else ""
    print("=" * 70)
    print(f"SMOKE: {n_contract_valid}/{len(records)} contract-valid | "
          f"stop: {stops.get('eos', 0)} eos / {stops.get('length_cap', 0)} length_cap{other} | "
          f"parity: {parity_ok}/{len(records)} -> {'PASS' if passed else 'FAIL'}")

    if not passed:
        print("\nDIAGNOSTIC FORK (read stop-reason FIRST):")
        if stops.get("length_cap"):
            print(" - length_cap present -> likely a MISSING-EOS ramble, NOT bnb garbage. "
                  "Re-read those raws (the JSON may be valid before the ramble); check the "
                  "stop-token wiring / lower max_new_tokens before suspecting numerics.")
        if parity_ok < len(records):
            print(" - parity MISMATCH -> cause (b): train<->gen prompt divergence. Fix the "
                  "render before blaming bnb.")
        elif stops.get("eos") and n_contract_valid < len(records):
            print(" - stopped on EOS, parity OK, still unparseable -> cause (b) ruled out; "
                  "suspect (a): degraded 4-bit numerics on Blackwell sm_120.")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke the eval generation core on a checkpoint.")
    ap.add_argument("--adapter", required=True,
                    help="PEFT checkpoint dir, or a run dir (lowest checkpoint-N is used)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--show-prompt", action="store_true",
                    help="print the rendered prompt for the first record")
    a = ap.parse_args()
    ok = smoke(a.adapter, max_new_tokens=a.max_new_tokens, show_prompt=a.show_prompt)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
