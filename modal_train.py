"""Modal entrypoint for FaithfulDocVQA SFT — a pod-independent training path.

Runs the SAME training the RunPod pod runs (``scripts/train_sft.py``, mode B,
3 epochs, all checkpoints, no W&B) but on Modal's GPUs. Modal offers L40S (Ada,
sm_89) / A100 (Ampere, sm_80) / H100 (Hopper) — **NOT Blackwell** — so the image
pins the proven **cu124** torch stack (``torch 2.5.1``), not the pod's ``cu128``
(which is Blackwell-only). The cu124 wheels cover sm_80/sm_86/sm_89/sm_90, so the
SAME image runs on L40S and A100-40GB (a one-string GPU swap).

Data comes from the pre-uploaded Volume ``faithfuldocvqa-data``; checkpoints persist
to ``faithfuldocvqa-ckpts``; the ~8 GB HF base model caches in
``faithfuldocvqa-hf-cache`` (the latter two auto-create on first run).

Usage (nothing here runs until invoked):
    uv run modal run modal_train.py --smoke          # ~minutes: validate the pipeline cheaply
    uv run modal run --detach modal_train.py         # full ~13 h run (--detach: survive disconnect)
Inspect / pull persisted checkpoints:
    uv run modal volume ls  faithfuldocvqa-ckpts /sft_B_seed42
    uv run modal volume get faithfuldocvqa-ckpts /sft_B_seed42 ./data/checkpoints/modal/
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "faithfuldocvqa-sft"
# ROOT that train_sft.py computes as Path(__file__).resolve().parent.parent — the dir
# that must contain scripts/ AND (via volume mounts) data/ and outputs/.
REPO_ROOT = "/root/faithfuldocvqa"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"     # what format_sft.MODEL_ID resolves on the pod
LOCAL_SCRIPTS = str(Path(__file__).parent / "scripts")

# ── image: the PROVEN cu124 stack (NOT the pod's cu128 — Blackwell-only) ──────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch trio from the cu124 index — the validated non-Blackwell baseline
    # (runpod_bootstrap.sh's else-branch installs exactly these). Self-contained
    # wheels (bundle the CUDA runtime), so no CUDA base image is needed.
    .pip_install(
        "torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    # the rest, pinned to the pod's validated versions. transformers 5.12 requires
    # torch>=2.4 (satisfied by 2.5.1) and accelerate>=1.1.0 (hence >=1.1.0, not >=1.0).
    .pip_install(
        "transformers==5.12.0",
        "peft==0.19.1",
        "bitsandbytes==0.49.2",
        "accelerate>=1.1.0",
        "numpy==1.26.3",
        "pillow==10.2.0",
        "pydantic>=2",            # scripts/faithful_io.py uses pydantic v2 (BaseModel/field_validator)
        "huggingface_hub",        # snapshot_download for the base-model pre-cache
    )
    .env(
        {
            "HF_HOME": "/cache",                                   # base-model cache volume
            "HF_HUB_DISABLE_XET": "1",                            # classic dl, deterministic (matches pod)
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", # mirrors train_sft.py's os.environ.setdefault
        }
    )
    # The training CODE — scripts/ only. The train path never imports phase0/, and the
    # data lives on a Volume, so we deliberately do NOT add data/ (793 MB) to the image.
    .add_local_dir(LOCAL_SCRIPTS, f"{REPO_ROOT}/scripts", ignore=["__pycache__", "*.pyc"])
)

app = modal.App(APP_NAME)

# ── volumes ───────────────────────────────────────────────────────────────────────
data_vol = modal.Volume.from_name("faithfuldocvqa-data")                        # already uploaded
ckpt_vol = modal.Volume.from_name("faithfuldocvqa-ckpts", create_if_missing=True)
hf_vol = modal.Volume.from_name("faithfuldocvqa-hf-cache", create_if_missing=True)

# Mount the data volume at ROOT/data so train_sft's ROOT-relative reads resolve:
#   ROOT/data/master.jsonl, ROOT/data/images/<id>.png, ROOT/data/images_cap_manifest.json
# Checkpoints go to a SIBLING mount ROOT/outputs (NOT nested inside ROOT/data) via the
# relative --output-dir below, so the two volumes don't overlap.
VOLUMES = {
    f"{REPO_ROOT}/data": data_vol,        # NOTE: mounted read-WRITE (Modal has no per-mount RO flag);
                                          #       training only READS it — it never writes here.
    f"{REPO_ROOT}/outputs": ckpt_vol,     # -> ROOT/outputs/sft_B_seed42/checkpoint-*
    "/cache": hf_vol,                     # HF_HOME (base-model weights)
}


@app.function(
    gpu="L40S",                # 48 GB Ada (sm_89). Fallback: change to "A100-40GB" — same image works.
    image=image,
    volumes=VOLUMES,
    timeout=20 * 60 * 60,      # 20 h — generous for the ~13 h run + model download/eval headroom
    cpu=8.0,                   # train_sft uses dataloader_num_workers=6 + a main proc
    memory=32 * 1024,          # 32 GB RAM — big pixel_values (~90 MB each) × prefetch
)
def train(smoke: bool = False) -> None:
    """Run scripts/train_sft.py exactly as the pod does (mode B / 3 epochs / no-wandb),
    reading data from the mounted volume and writing checkpoints to the ckpt volume.

    smoke=True appends --smoke, which caps train_sft to max_steps=6 / save_steps=2 — so
    it writes checkpoint-2/4/6 + a final adapter in ~minutes. A full end-to-end pipeline
    validation (image, volumes, data paths, model cache, AND commit persistence) for ~$.
    """
    import subprocess
    import threading

    from huggingface_hub import snapshot_download

    # 1. Pre-cache the ~8 GB base model into the hf-cache volume + commit, so a later
    #    crash/timeout doesn't force a re-download on the next run. HF_HOME=/cache (image env).
    #    (Runs inside the GPU container, so the first run is billed for the download;
    #    every run after reuses the committed cache.)
    print(f"[modal] pre-caching {MODEL_ID} into /cache ...", flush=True)
    snapshot_download(MODEL_ID)
    hf_vol.commit()
    print("[modal] base model cached + committed", flush=True)

    # 2. Periodic ckpt-volume commit. UNLIKE the pod's network FS (writes land directly),
    #    Modal Volume writes only persist on .commit() — without this a timeout/crash
    #    loses every checkpoint since the last commit. Under --smoke the interval drops to
    #    60 s so the PERIODIC committer actually fires during the short run (so the smoke
    #    validates periodic AND final commit, not just the final one).
    commit_interval = 60 if smoke else 1800
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(commit_interval):
            try:
                ckpt_vol.commit()
                print("[modal] periodic ckpt commit", flush=True)
            except Exception as e:  # noqa: BLE001 — a commit hiccup must not kill training
                print(f"[modal] ckpt commit warning: {e}", flush=True)

    threading.Thread(target=committer, daemon=True).start()

    # 3. The SAME command the pod runs (+ --smoke when validating). --output-dir outputs
    #    is RELATIVE, so train_sft resolves ROOT/outputs/sft_B_seed42 (the ckpt volume).
    #    --no-wandb => wandb is never imported, so no W&B secret is needed.
    cmd = [
        "python", "scripts/train_sft.py",
        "--mode", "B", "--seed", "42",
        "--output-dir", "outputs",
        "--epochs", "3", "--save-total-limit", "8",
        "--no-wandb",
    ]
    if smoke:
        cmd.append("--smoke")  # train_sft caps to 6 steps / save_steps=2 -> checkpoint-2/4/6
    print(f"[modal] launching: {' '.join(cmd)}  (cwd={REPO_ROOT})", flush=True)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        stop.set()
        ckpt_vol.commit()  # final flush regardless of success/failure
    mode = "SMOKE" if smoke else "full"
    print(f"[modal] {mode} training finished; checkpoints committed to faithfuldocvqa-ckpts",
          flush=True)


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    """Spawns the GPU training function.
      uv run modal run modal_train.py --smoke     # cheap end-to-end pipeline validation
      uv run modal run --detach modal_train.py    # full ~13 h run (survives disconnect)
    """
    train.remote(smoke=smoke)
