#!/usr/bin/env python3
"""Build data/images_cap_manifest.json — per-doc record of whether the max_pixels
cap downscaled each image, so EVAL can slice results by capped vs uncapped.

Keyed by ``doc_id`` (str), each entry: ``original_mp``, ``effective_mp`` (the
resolution the model actually sees after the cap + factor-32 rounding),
``was_capped`` (original pixels > MAX_PIXELS), and ``split``. Ground truth is the
real Qwen3-VL image processor under the CURRENT ``format_sft.MAX_PIXELS`` cap —
regenerate this file whenever MAX_PIXELS changes.

Run: ``uv run python scripts/build_cap_manifest.py``
"""

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.format_sft import MAX_PIXELS, load_processor

MASTER = ROOT / "data" / "master.jsonl"
OUT = ROOT / "data" / "images_cap_manifest.json"


def main() -> None:
    recs = [json.loads(line) for line in MASTER.open()]
    docs: dict[int, tuple[str, str]] = {}
    for r in recs:
        docs.setdefault(r["doc_id"], (r["image_path"], r["split"]))

    proc, _ = load_processor()                 # applies the MAX_PIXELS cap
    ip = proc.image_processor
    patch, merge = ip.patch_size, ip.merge_size
    print(f"MAX_PIXELS = {MAX_PIXELS} (~{MAX_PIXELS/1e6:.2f} MP) | "
          f"cap longest_edge = {ip.size.get('longest_edge')} | {len(docs)} unique docs")

    manifest: dict[str, dict] = {}
    n_capped = 0
    for i, (docid, (path, split)) in enumerate(sorted(docs.items())):
        img = Image.open(ROOT / path).convert("RGB")
        w, h = img.size
        original_mp = w * h / 1e6
        was_capped = (w * h) > MAX_PIXELS
        # ground-truth effective resolution from the real processor grid
        g = ip(images=img, return_tensors="pt")["image_grid_thw"][0].tolist()  # t,h,w patches
        effective_mp = (g[1] * patch) * (g[2] * patch) / 1e6
        manifest[str(docid)] = {
            "original_mp": round(original_mp, 3),
            "effective_mp": round(effective_mp, 3),
            "was_capped": bool(was_capped),
            "split": split,
        }
        n_capped += was_capped
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(docs)}] processed")

    OUT.write_text(json.dumps(manifest, indent=2))

    capped_by_split = Counter(m["split"] for m in manifest.values() if m["was_capped"])
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(manifest)} docs)")
    print(f"capped (> {MAX_PIXELS/1e6:.2f} MP): {n_capped}  by split: "
          f"{ {s: capped_by_split.get(s, 0) for s in ('train', 'validation', 'test')} }")

    m = manifest.get("13402")
    if m:
        img = Image.open(ROOT / docs[13402][0]).convert("RGB")
        g = ip(images=img, return_tensors="pt")["image_grid_thw"][0].tolist()
        tokens = (g[0] * g[1] * g[2]) // (merge * merge)
        print(f"doc 13402: {m['original_mp']} MP -> effective {m['effective_mp']} MP "
              f"(grid {g}, vision_tokens={tokens}) was_capped={m['was_capped']} split={m['split']}")


if __name__ == "__main__":
    main()
