"""Modal GENERATION pass for FaithfulDocVQA — base model + SFT checkpoints over a split.

Companion to ``modal_train.py`` (training-only). Reuses the SAME image + the SAME
3-volume mounting (mirrored here, not imported, so Modal resolves a single app per
file), and reuses ``scripts/eval_runner`` + ``scripts/gen_persist`` for the actual
generation. Writes the ``{record_id, raw, stop_reason, n_new_tokens}`` schema to the
ckpts volume, alongside the existing validation gens.

Layout on the ckpts volume (mounted at REPO_ROOT/outputs):
  - checkpoints : outputs/sft_v3_32g/sft_B_seed42/checkpoint-*
  - val gens    : outputs/eval/sft_B_seed42/gen_checkpoint-*.json        (already there)
  - this writes : outputs/<out-root-tail>/sft_B_seed42/gen_*<split>*.json

Base Qwen3-VL-4B loads from the hf-cache volume (snapshot_download, already populated
by the training smoke); LoRA adapters attach via PeftModel.from_pretrained against it.

The passes that produced the committed gens (data/checkpoints/eval/) — kept as
the reproduction runbook:
  uv run modal run modal_gen.py --split validation --base      --out-root outputs/eval   # base over val
  uv run modal run modal_gen.py --split test       --base      --out-root outputs/eval   # base over test
  uv run modal run modal_gen.py --split test --steps 214,428,642,856,1070,1284 --out-root outputs/eval  # 6 ckpts over test

Smoke pattern used before the full passes (ck-1284, test, 5 records, isolated out-root):
  uv run modal run modal_gen.py --split test --steps 1284 --limit 5 --check-parity --out-root outputs/eval_smoke
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "faithfuldocvqa-gen"
REPO_ROOT = "/root/faithfuldocvqa"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LOCAL_SCRIPTS = str(Path(__file__).parent / "scripts")
RUN_DIR = f"{REPO_ROOT}/outputs/sft_v3_32g/sft_B_seed42"   # checkpoints on the ckpts volume

# ── image: same proven cu124 inference stack as modal_train.py ────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==5.12.0", "peft==0.19.1", "bitsandbytes==0.49.2",
        "accelerate>=1.1.0", "numpy==1.26.3", "pillow==10.2.0", "pydantic>=2",
        "huggingface_hub",
    )
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_dir(LOCAL_SCRIPTS, f"{REPO_ROOT}/scripts", ignore=["__pycache__", "*.pyc"])
)

app = modal.App(APP_NAME)

# ── volumes: identical names + mount paths as modal_train.py ──────────────────────
data_vol = modal.Volume.from_name("faithfuldocvqa-data")
ckpt_vol = modal.Volume.from_name("faithfuldocvqa-ckpts", create_if_missing=True)
hf_vol = modal.Volume.from_name("faithfuldocvqa-hf-cache", create_if_missing=True)
VOLUMES = {
    f"{REPO_ROOT}/data": data_vol,        # records (master.jsonl) + images/  (read-only in practice)
    f"{REPO_ROOT}/outputs": ckpt_vol,     # checkpoints in, generations out
    "/cache": hf_vol,                     # HF_HOME — base weights
}


@app.function(gpu="L40S", image=image, volumes=VOLUMES,
              timeout=6 * 60 * 60, cpu=8.0, memory=32 * 1024)
def generate_pass(split: str = "validation", steps: str | None = None,
                  limit: int | None = None, base: bool = False,
                  check_parity: bool = False, out_root: str = "outputs/eval",
                  records_file: str | None = None, tag: str = "custom") -> None:
    """Run scripts/gen_persist.py for one pass (base or selected checkpoints) over a split,
    or over an arbitrary --records-file (e.g. manufactured abstains on the data volume).
    Output lands on the ckpts volume under REPO_ROOT/<out_root>/sft_B_seed42/."""
    import subprocess

    from huggingface_hub import snapshot_download

    print(f"[modal-gen] ensuring {MODEL_ID} is cached on /cache ...", flush=True)
    snapshot_download(MODEL_ID)
    hf_vol.commit()

    cmd = ["python", "scripts/gen_persist.py", "--run-dir", RUN_DIR,
           "--out-root", out_root, "--split", split, "--max-new-tokens", "128"]
    if steps:
        cmd += ["--steps", steps]
    if limit:
        cmd += ["--limit", str(limit)]
    if base:
        cmd += ["--base"]
    if check_parity:
        cmd += ["--check-parity"]
    if records_file:
        cmd += ["--records-file", records_file, "--tag", tag]
    print(f"[modal-gen] launching: {' '.join(cmd)}  (cwd={REPO_ROOT})", flush=True)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        ckpt_vol.commit()   # persist whatever was written, success or not
    print("[modal-gen] done; outputs committed to faithfuldocvqa-ckpts", flush=True)


@app.local_entrypoint()
def main(split: str = "validation", steps: str | None = None, limit: int | None = None,
         base: bool = False, check_parity: bool = False, out_root: str = "outputs/eval",
         records_file: str | None = None, tag: str = "custom") -> None:
    """e.g. manufactured abstains through ck-642:
      uv run modal run modal_gen.py --steps 642 --records-file data/manufactured_test_abstains.jsonl --tag manufactured
    """
    generate_pass.remote(split=split, steps=steps, limit=limit, base=base,
                         check_parity=check_parity, out_root=out_root,
                         records_file=records_file, tag=tag)
