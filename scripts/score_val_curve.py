"""score_val_curve.py — per-checkpoint honest_abstain / false_grounding curve for a seed.

Run:  uv run python scripts/score_val_curve.py --gens <dir> [--split validation] [--out <json>]

Pure CPU. For every ``gen_checkpoint-*.json`` (gen_persist payload) in ``--gens``,
classify the split's records with ``eval_scorer`` and emit the abstain-side headline
metrics per checkpoint — the multi-seed peak-replication signal:

  checkpoint | n_abstain | honest_abstain | false_grounding | n_answerable | false_abstain | valid_citation | malformed

honest_abstain + false_grounding are read off the gold-ABSTAIN partition; false_abstain
+ valid_citation off the gold-ANSWERABLE partition (see eval_scorer.aggregate). This is
the artifact used to ask "does ck-642's 1.5-epoch abstain peak replicate across seeds?".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eval_scorer import aggregate, classify
from scripts.format_sft import load_split  # frozen split membership (same source as training)


def load_gen(p: Path) -> dict[str, str]:
    obj = json.loads(p.read_text())
    recs = obj["records"] if isinstance(obj, dict) and "records" in obj else obj
    return {r["record_id"]: r.get("raw") for r in recs}


def step_of(p: Path) -> int:
    m = re.search(r"checkpoint-(\d+)", p.name)
    return int(m.group(1)) if m else -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gens", type=Path, required=True,
                    help="dir of gen_checkpoint-*.json (or a single .json)")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--out", type=Path, default=None, help="write the curve as JSON here")
    a = ap.parse_args()

    # frozen split records (mode B = full split, abstains included), keyed by id
    records = {r["record_id"]: r for r in load_split(a.split, "B")}

    gen_files = ([a.gens] if a.gens.is_file()
                 else sorted(a.gens.glob("gen_checkpoint-*.json"), key=step_of))
    if not gen_files:
        sys.exit(f"no gen_checkpoint-*.json under {a.gens}")

    hdr = (f"{'ckpt':>6}{'n_abs':>7}{'honest_abstain':>16}{'false_ground':>14}"
           f"{'n_ans':>7}{'false_abstain':>15}{'valid_cite':>12}{'malformed':>11}")
    print(hdr)
    print("-" * len(hdr))
    curve = []
    for gf in gen_files:
        raw = load_gen(gf)
        outcomes = [classify(rec, raw[rid]) for rid, rec in records.items() if rid in raw]
        missing = [rid for rid in records if rid not in raw]
        agg = aggregate(outcomes)
        ab = agg["bucket_rates"]["abstain"]
        an = agg["bucket_rates"]["answerable"]
        row = {
            "checkpoint": step_of(gf),
            "n_abstain": agg["n_abstain"],
            "honest_abstain": round(ab.get("honest_abstain", float("nan")), 4),
            "false_grounding": round(ab.get("false_grounding", float("nan")), 4),
            "n_answerable": agg["n_answerable"],
            "false_abstain": round(an.get("false_abstain", float("nan")), 4),
            "valid_citation": round(an.get("valid_citation", float("nan")), 4),
            "malformed": agg["malformed_rate"],
            "n_missing_gen": len(missing),
        }
        curve.append(row)
        print(f"{row['checkpoint']:>6}{row['n_abstain']:>7}{row['honest_abstain']:>16.3f}"
              f"{row['false_grounding']:>14.3f}{row['n_answerable']:>7}"
              f"{row['false_abstain']:>15.3f}{row['valid_citation']:>12.3f}{row['malformed']:>11.3f}"
              + (f"   !! {len(missing)} missing gen" if missing else ""))

    if a.out:
        a.out.write_text(json.dumps(curve, indent=2))
        print(f"\nwrote curve -> {a.out}")


if __name__ == "__main__":
    main()
