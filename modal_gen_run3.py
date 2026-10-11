"""Parameter-passing mirror of modal_gen.py: run_dir parameterized, generation path verbatim."""

# Why a mirror rather than an edit: the committed modal_gen.py hardcodes
# RUN_DIR = outputs/sft_v3_32g/sft_B_seed42, and its exact form produced the committed
# gens. Mirroring keeps that script untouched -- the same method used for the GRPO
# run-1/run-2 gens (docs/analysis/mcnemar_seed_vs_grpo.md: "a parameter-passing mirror
# of modal_gen.py (no repo script modified)"). Image, 3-volume mounting, and the
# gen_persist -> eval_runner path (greedy, --max-new-tokens 128) are copied verbatim,
# because the paired design depends on them.
#
# Runbook (the invocations that produced the run-3 test gens):
#   smoke : python -m modal run modal_gen_run3.py --split test --steps 1296 --limit 3 \
#               --check-parity --out-root outputs/eval_run3_smoke
#   full  : python -m modal run modal_gen_run3.py --split test --steps 1296 \
#               --out-root outputs/eval_run3
#           python -m modal run modal_gen_run3.py --steps 1296 \
#               --records-file data/manufactured_test_abstains.jsonl --tag manufactured \
#               --out-root outputs/eval_run3

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "faithfuldocvqa-gen-run3"
REPO_ROOT = "/root/faithfuldocvqa"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LOCAL = Path(__file__).parent
LOCAL_SCRIPTS = str(LOCAL / "scripts")
# The one parameter this mirror exists to change. The ckpt volume mounts at
# REPO_ROOT/outputs, so checkpoints mirrored to volume path
# sft_v3_balanced_nodecoy/sft_B_seed42/ land here. Proven by the preflight in
# generate_pass(), not assumed.
DEFAULT_RUN_DIR = f"{REPO_ROOT}/outputs/sft_v3_balanced_nodecoy/sft_B_seed42"

# ⚠ ALWAYS pass an isolated --out-root. gen_persist names its output directory after
# the run-dir's BASENAME -- "sft_B_seed42" -- which is the same basename the prior
# sft_v3_32g series uses. The default outputs/eval would therefore drop run-3 gens
# into the old series' directory on the volume, where the filenames differ only by
# step number. Same collision hazard as the local 214- vs 324-series gens.
DEFAULT_OUT_ROOT = "outputs/eval_run3"

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
    .add_local_dir(
        str(LOCAL / "phase0"),
        f"{REPO_ROOT}/phase0",
        ignore=["__pycache__", "*.pyc", "outputs", "*.html", "*.md"],
    )
)

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name("faithfuldocvqa-data")
ckpt_vol = modal.Volume.from_name("faithfuldocvqa-ckpts", create_if_missing=True)
hf_vol = modal.Volume.from_name("faithfuldocvqa-hf-cache", create_if_missing=True)
VOLUMES = {
    f"{REPO_ROOT}/data": data_vol,
    f"{REPO_ROOT}/outputs": ckpt_vol,
    "/cache": hf_vol,
}


@app.function(gpu="L40S", image=image, volumes=VOLUMES,
              timeout=6 * 60 * 60, cpu=8.0, memory=32 * 1024)
def generate_pass(split: str = "test", steps: str | None = None,
                  limit: int | None = None, base: bool = False,
                  check_parity: bool = False, out_root: str = DEFAULT_OUT_ROOT,
                  records_file: str | None = None, tag: str = "custom",
                  run_dir: str = DEFAULT_RUN_DIR, preflight_only: bool = False) -> None:
    """One generation pass for the run-3 checkpoints. Identical to modal_gen.generate_pass
    except that ``run_dir`` is a parameter and a path pre-flight runs first."""
    import os
    import subprocess

    from huggingface_hub import snapshot_download

    # ── pre-flight: prove the volume layout rather than assume it ────────────────
    print(f"[preflight] REPO_ROOT/outputs contains: {sorted(os.listdir(f'{REPO_ROOT}/outputs'))[:12]}", flush=True)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[preflight] FAIL: run_dir does not exist in-container: {run_dir}")
    entries = sorted(os.listdir(run_dir))
    print(f"[preflight] run_dir OK: {run_dir}", flush=True)
    print(f"[preflight] contents: {entries}", flush=True)
    if steps:
        for s in steps.split(","):
            ck = os.path.join(run_dir, f"checkpoint-{s.strip()}")
            ok = os.path.isfile(os.path.join(ck, "adapter_model.safetensors"))
            print(f"[preflight] checkpoint-{s.strip()} adapter present: {ok}", flush=True)
            if not ok:
                raise SystemExit(f"[preflight] FAIL: missing adapter for checkpoint-{s.strip()}")
    rf = os.path.join(REPO_ROOT, records_file) if records_file else None
    if rf:
        print(f"[preflight] records_file present: {os.path.isfile(rf)}  ({rf})", flush=True)
        if not os.path.isfile(rf):
            raise SystemExit(f"[preflight] FAIL: records_file missing: {rf}")
    if preflight_only:
        print("[preflight] preflight_only set — stopping before generation", flush=True)
        return

    print(f"[modal-gen] ensuring {MODEL_ID} is cached on /cache ...", flush=True)
    snapshot_download(MODEL_ID)
    hf_vol.commit()

    cmd = ["python", "scripts/gen_persist.py", "--run-dir", run_dir,
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
        ckpt_vol.commit()
    print("[modal-gen] done; outputs committed to faithfuldocvqa-ckpts", flush=True)


@app.local_entrypoint()
def main(split: str = "test", steps: str | None = None, limit: int | None = None,
         base: bool = False, check_parity: bool = False,
         out_root: str = DEFAULT_OUT_ROOT, records_file: str | None = None,
         tag: str = "custom", run_dir: str = DEFAULT_RUN_DIR,
         preflight_only: bool = False) -> None:
    generate_pass.remote(split=split, steps=steps, limit=limit, base=base,
                         check_parity=check_parity, out_root=out_root,
                         records_file=records_file, tag=tag, run_dir=run_dir,
                         preflight_only=preflight_only)
