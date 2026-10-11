"""Eyeball the 3-field SFT training data before launching a production run.

Read-only inspection (no training, no data mutation). Reports:
  - per-split (train / validation / test) counts + answerable/abstain breakdown,
  - N full rendered TRAIN examples (system + user + assistant target) for both
    answerable and abstain rows, exactly as the model is fed them,
  - explicit checks that the assistant target serializes `answerable_from_menu`
    as a JSON boolean (`true`/`false`, unquoted) and keeps `found_answer` casing raw.

The rendered example uses the official processor chat template when it loads
(network/cache available); otherwise it falls back to printing the raw
`build_messages` turns + the exact `build_target_v3` target string — the target
itself never depends on the processor, so the eyeball check holds either way.

Run:  python scripts/inspect_sft_3field.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.format_sft import (
    SYSTEM_INSTRUCTION,
    build_messages,
    load_master,
    load_split,
    render_menu,
    _v3_target,
)

SPLITS = ("train", "validation", "test")
N_EACH = 3  # examples per response type


def split_counts(records: list[dict]) -> None:
    """Print total / answerable / abstain (+ abstain %) for every split."""
    print("=" * 78)
    print("SPLIT COUNTS  (response-type balance)")
    print("=" * 78)
    print(f"{'split':<12}{'total':>8}{'answerable':>13}{'abstain':>10}{'abstain%':>11}")
    for split in SPLITS:
        recs = [r for r in records if r["split"] == split]
        n = len(recs)
        n_abs = sum(1 for r in recs if r["target_is_abstain"])
        n_ans = n - n_abs
        pct = 100 * n_abs / n if n else 0.0
        print(f"{split:<12}{n:>8}{n_ans:>13}{n_abs:>10}{pct:>10.1f}%")
    tot = len(records)
    tot_abs = sum(1 for r in records if r["target_is_abstain"])
    print("-" * 52)
    print(f"{'ALL':<12}{tot:>8}{tot - tot_abs:>13}{tot_abs:>10}"
          f"{100 * tot_abs / tot:>10.1f}%")


def _pick(records: list[dict], *, abstain: bool, n: int) -> list[dict]:
    """N TRAIN rows of one response type, smallest menus first (readable prints)."""
    pool = [r for r in records
            if r["split"] == "train" and bool(r["target_is_abstain"]) == abstain]
    pool.sort(key=lambda r: len(r["ocr_menu"]))
    # for answerable, skip the trivially tiny menus so the example is representative
    if not abstain:
        pool = [r for r in pool if len(r["ocr_menu"]) >= 8] or pool
    return pool[:n]


def _target_checks(target: str, record: dict) -> list[str]:
    """Return human-readable PASS/FAIL lines for the contract invariants we care
    about: real JSON bool (not a quoted string), correct bool value, raw casing,
    and the abstain rows carrying the real (thrown-away) answer."""
    obj = json.loads(target)  # must be valid JSON or this raises — that's the point
    is_abstain = bool(record["target_is_abstain"])
    gold0 = record["gold_answers"][0] if record["gold_answers"] else ""
    afm = obj["answerable_from_menu"]
    fa = obj["found_answer"]

    lines = []
    # JSON bool, not "false"/"true" string. `"false"` would survive json.loads as a
    # str; assert the parsed python type is bool AND the raw text has no quotes.
    raw_bool = '"answerable_from_menu":true' in target or '"answerable_from_menu":false' in target
    lines.append(
        f'  bool type    : {type(afm).__name__}={afm!r}  raw="{("true" if afm else "false")}" '
        f'unquoted={raw_bool}  -> {"PASS" if isinstance(afm, bool) and raw_bool else "FAIL"}'
    )
    exp = (not is_abstain)
    lines.append(
        f'  bool value   : answerable_from_menu={afm}  (expected {exp} for '
        f'{"abstain" if is_abstain else "answerable"})  -> {"PASS" if afm is exp else "FAIL"}'
    )
    lines.append(
        f'  found_answer : {fa!r}  == gold[0] {gold0!r} (raw casing)  '
        f'-> {"PASS" if fa == gold0 else "FAIL"}'
    )
    lines.append(
        f'  evidence_ids : {obj["evidence_ids"]}  '
        f'({"empty as required" if is_abstain else "non-empty as required"})  '
        f'-> {"PASS" if bool(obj["evidence_ids"]) == (not is_abstain) else "FAIL"}'
    )
    if is_abstain:
        lines.append(
            f'  ABSTAIN note : carries the real perceived answer {fa!r} while '
            f'answerable_from_menu=false / evidence_ids=[]  '
            f'-> {"PASS" if fa and fa != "unanswerable" and not afm and not obj["evidence_ids"] else "FAIL"}'
        )
    return lines


def _render_full(processor, record: dict) -> str:
    """The exact full training string (prompt + assistant target) via the official
    chat template if the processor loaded, else a manual system/user/assistant dump."""
    if processor is not None:
        from scripts.format_sft import render_example
        ex = render_example(processor, record)
        return ex["full_text"]
    # fallback: show the structured turns + the real target string
    msgs = build_messages(record, include_target=True)
    user_text = next(c["text"] for c in msgs[1]["content"] if c.get("type") == "text")
    return (
        "<<processor unavailable — showing build_messages turns + real target>>\n"
        "[SYSTEM]\n" + SYSTEM_INSTRUCTION + "\n\n"
        "[USER]\n<image: " + record["image_path"] + ">\n" + user_text + "\n\n"
        "[ASSISTANT  (the trained completion)]\n" + msgs[2]["content"]
    )


def show_examples(records: list[dict], processor) -> None:
    for abstain, label in ((False, "ANSWERABLE"), (True, "ABSTAIN")):
        for rec in _pick(records, abstain=abstain, n=N_EACH):
            target = _v3_target(rec)
            print("\n" + "=" * 78)
            print(f"{label}  record_id={rec['record_id']}  doc={rec['doc_id']} "
                  f"type={rec['doc_type']}  menu={len(rec['ocr_menu'])} tokens")
            print(f"  question     : {rec['question']!r}")
            print("=" * 78)
            print(_render_full(processor, rec))
            print("\n  -- assistant TARGET (exact trained completion string) --")
            print(f"  {target}")
            print("  -- contract checks --")
            for line in _target_checks(target, rec):
                print(line)


def _try_processor():
    """Load the official processor for a true chat-template render; None if it
    can't (offline / not cached) so the inspection still runs locally."""
    try:
        from scripts.format_sft import load_processor
        proc, resolved = load_processor()
        print(f"[processor] loaded {type(proc).__name__} ({resolved}) — full chat-template render")
        return proc
    except Exception as e:  # noqa: BLE001 — inspection must not hard-fail offline
        print(f"[processor] unavailable ({type(e).__name__}: {str(e)[:120]}) — "
              "falling back to build_messages dump (target string is identical)")
        return None


def main() -> None:
    records = load_master()
    split_counts(records)
    # the production run trains on mode B (all of train: answerable + abstain)
    b_train = load_split("train", "B", records)
    print(f"\n[production train set] mode B = ALL train records: {len(b_train)} "
          f"(answer + abstain over everything)")
    proc = _try_processor()
    show_examples(records, proc)


if __name__ == "__main__":
    main()
