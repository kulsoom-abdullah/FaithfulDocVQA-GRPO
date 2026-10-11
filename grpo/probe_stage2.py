"""Stage 2 TASK A — 10-prompt generation probe + reward scoring (GPU, generation-only).

Reuses the committed SFT path: eval_runner.load_adapted_model (bf16 base + ck-642 LoRA)
and render_prompt/build_messages (SFT-identical prompt), then samples G=8 @ temp 1.0 and
scores each completion with grpo.reward.compute_grpo_reward(run=1). Logs per-prompt wall
clock as the cu124-pathology tripwire (monotonic climb => stack fail => switch to Modal).

Run (on the pod, in the cu128 venv):
  FDV_ADAPTER=/workspace/ck-642 python grpo/probe_stage2.py
"""
import json, os, sys, time, random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from transformers import AutoModelForImageTextToText
from peft import PeftModel

from scripts.eval_runner import render_prompt, _eos_ids   # SFT-identical prompt render
from scripts.format_sft import load_processor, MODEL_ID
from scripts.eval_scorer import classify
from grpo.reward import compute_grpo_reward

ADAPTER = os.environ.get("FDV_ADAPTER", "/workspace/ck-642")
G, TEMP, MAXNEW, SEED = 8, 1.0, 128, 0
READABLE = {"table/list", "layout", "form", "free_text"}


def first_doctype(dt):
    return dt[0] if isinstance(dt, list) and dt else str(dt)


def main() -> None:
    master = {json.loads(l)["record_id"]: json.loads(l) for l in open(ROOT / "data/master.jsonl")}
    manuf = [json.loads(l) for l in open(ROOT / "data/manufactured_train_abstains.jsonl")]

    random.seed(SEED)
    recurs = [r for r in manuf if r.get("answer_recurs_in_menu")]
    nonrec = [r for r in manuf if not r.get("answer_recurs_in_menu")]
    abstain5 = random.sample(recurs, 2) + random.sample(nonrec, 3)          # >=2 decoys
    ans_pool = [r for r in master.values() if r.get("split") == "train"
                and not r.get("target_is_abstain") and first_doctype(r.get("doc_type")) in READABLE]
    answerable5 = random.sample(ans_pool, 5)
    PROMPTS = [("abstain", r) for r in abstain5] + [("answerable", r) for r in answerable5]
    print(f"selected {len(abstain5)} abstain (>=2 recurs) + {len(answerable5)} answerable = {len(PROMPTS)} prompts")

    print(f"[load] {ADAPTER}  (bf16 base + LoRA, no bnb)")
    proc, _ = load_processor()
    base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    resolved = f"{MODEL_ID} (bf16)"
    tok = proc.tokenizer
    eos = _eos_ids(proc)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if eos else None)
    print(f"[stack] torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)} | base {resolved}")
    torch.manual_seed(SEED)

    all_facts, times = [], []
    for i, (kind, rec) in enumerate(PROMPTS):
        prompt = render_prompt(proc, rec)
        img = Image.open(ROOT / rec["image_path"])
        inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(model.device)
        plen = inputs["input_ids"].shape[1]
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, do_sample=True, temperature=TEMP, top_p=1.0,
                                 num_return_sequences=G, max_new_tokens=MAXNEW,
                                 eos_token_id=eos or None, pad_token_id=pad)
        torch.cuda.synchronize()
        dt = time.time() - t0
        times.append(dt)
        print(f"\n=== PROMPT {i+1}/10 [{kind}] {rec['record_id']} | {G} gens in {dt:.1f}s ===")
        print(f"    Q: {rec['question'][:88]!r} gold={rec.get('gold_answers')} recurs={rec.get('answer_recurs_in_menu')}")
        for j in range(G):
            raw = proc.batch_decode([gen[j][plen:]], skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0].strip()
            f = classify(rec, raw)
            r = compute_grpo_reward(rec, raw, run=1)
            all_facts.append((kind, f["bucket"]))
            print(f"    [{j}] {f['bucket']:<15} r={r:+.2f}  {raw[:88]!r}")

    print("\n=== TIMING — cu124-pathology tripwire ===")
    print("  per-prompt seconds:", [round(t, 1) for t in times])
    mono = all(times[i] < times[i + 1] for i in range(len(times) - 1))
    ratio = (sum(times[5:]) / 5) / (sum(times[:5]) / 5) if times[:5] else float("nan")
    patho = mono or ratio > 1.8
    print(f"  monotonic climb: {mono} | second-half/first-half ratio: {ratio:.2f}")
    print(f"  VERDICT: {'!! PATHOLOGY (climbing) — STACK FAIL, SWITCH TO MODAL' if patho else 'FLAT/NOISY — Ada+cu128 stack OK'}")

    print("\n=== bucket distribution over 80 completions ===")
    print("  overall:", dict(Counter(b for _, b in all_facts)))
    print("  abstain prompts:", dict(Counter(b for k, b in all_facts if k == "abstain")))
    print("  answerable prompts:", dict(Counter(b for k, b in all_facts if k == "answerable")))


if __name__ == "__main__":
    main()
