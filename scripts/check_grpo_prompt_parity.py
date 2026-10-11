"""check_grpo_prompt_parity.py — single-record GRPO prompt-parity spot check.

Renders one committed manufactured-val record through BOTH prompt paths — the
eval render (eval_runner.render_prompt) and the TRL-side chat template over
to_row's conversational messages — prints both in full, reports byte-identity
with the first divergence, and runs the one-record stratum-aware leak check.
The at-scale evidence is the 300-prompt sweep (check_grpo_logged_prompt_parity.py
-> docs/analysis/grpo_logged_prompt_parity.md); this is its quick precursor probe.

Run:  uv run python scripts/check_grpo_prompt_parity.py   (repo root; CPU, no model load)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import json
from scripts.format_sft import load_processor
from scripts.eval_runner import render_prompt
from scripts.eval_scorer import _norm
from grpo.train_grpo import to_row

proc, resolved = load_processor()
rec = json.loads(open("data/manufactured_val_abstains.jsonl").readline())
print(f"record={rec['record_id']} stratum={rec['stratum']} recurs={rec['answer_recurs_in_menu']}\n")

p_eval = render_prompt(proc, rec)                       # eval-side render
row = to_row(rec)                                       # TRL-side messages
p_trl = proc.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)

print("---- EVAL RENDER (repr, full) ----");  print(repr(p_eval))
print("---- TRL RENDER (repr, full) ----");   print(repr(p_trl))
print(f"\nBYTE-IDENTICAL: {p_eval == p_trl}")
if p_eval != p_trl:
    i = next(k for k in range(min(len(p_eval), len(p_trl))) if p_eval[k] != p_trl[k])
    print(f"first divergence at char {i}:")
    print(f"  eval: ...{p_eval[max(0,i-60):i+60]!r}")
    print(f"  trl : ...{p_trl[max(0,i-60):i+60]!r}")

# stratum-aware leak check: answer string may appear ONLY via menu text (decoy rows)
hit = any(_norm(g) and _norm(g) in _norm(p_eval) for g in rec["gold_answers"])
expected = rec["answer_recurs_in_menu"]
print(f"gold-string-in-prompt: {hit}  (expected {expected} for this stratum) "
      f"-> {'OK' if hit == expected else 'INVESTIGATE'}")