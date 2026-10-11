"""Modal launcher for the matched-pool SFT continuation experiment.

This wrapper keeps the experiment isolated from the historical SFT launcher. It
mounts the same data/checkpoint/cache volumes, ships only the required local
code, and invokes ``scripts/train_sft_matched_pool.py`` with the locked config.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "faithfuldocvqa-sft-matched-pool"
REPO_ROOT = "/root/faithfuldocvqa"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
RUN_NAME = "sft_ck642_matched_grpo_pool_seed42"
ADAPTER = f"{REPO_ROOT}/outputs/sft_v3_32g/sft_B_seed42/checkpoint-642"
OUT_DIR = f"{REPO_ROOT}/outputs/{RUN_NAME}"
SMOKE_OUT_DIR = f"{REPO_ROOT}/outputs/{RUN_NAME}_smoke"
EXPECTED_ADAPTER_SHA256 = (
    "d76fa2b4609ba126bb427e5f338ba5a9c65ef8111e0bb37df9d31d344ba09691"
)
LOCAL = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "torchaudio==2.5.1",
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers==5.12.0",
        "peft==0.19.1",
        "bitsandbytes==0.49.2",
        "accelerate>=1.1.0",
        "numpy==1.26.3",
        "pillow==10.2.0",
        "pydantic>=2",
        "huggingface_hub",
        "datasets",
    )
    .env(
        {
            "HF_HOME": "/cache",
            "HF_HUB_DISABLE_XET": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(str(LOCAL / "scripts"), f"{REPO_ROOT}/scripts", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(str(LOCAL / "grpo"), f"{REPO_ROOT}/grpo", ignore=["__pycache__", "*.pyc"])
    .add_local_dir(
        str(LOCAL / "phase0"),
        f"{REPO_ROOT}/phase0",
        ignore=["__pycache__", "*.pyc", "outputs", "*.html", "*.md"],
    )
)

app = modal.App(APP_NAME)

data_vol = modal.Volume.from_name("faithfuldocvqa-data")
ckpt_vol = modal.Volume.from_name("faithfuldocvqa-ckpts")
hf_vol = modal.Volume.from_name("faithfuldocvqa-hf-cache")

VOLUMES = {
    f"{REPO_ROOT}/data": data_vol,
    f"{REPO_ROOT}/outputs": ckpt_vol,
    "/cache": hf_vol,
}


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes=VOLUMES,
    timeout=8 * 60 * 60,
    cpu=8.0,
    memory=32 * 1024,
)
def train_matched_pool(smoke: bool = False) -> None:
    """Run the two-step smoke or the single locked 150-step continuation."""
    import subprocess
    import threading

    from huggingface_hub import snapshot_download

    print(f"[modal-matched-sft] ensuring {MODEL_ID} cached on /cache ...", flush=True)
    snapshot_download(MODEL_ID)
    hf_vol.commit()

    stop = threading.Event()
    commit_interval = 60 if smoke else 600

    def committer() -> None:
        while not stop.wait(commit_interval):
            try:
                ckpt_vol.commit()
                print("[modal-matched-sft] periodic checkpoint-volume commit", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[modal-matched-sft] checkpoint commit warning: {exc}", flush=True)

    threading.Thread(target=committer, daemon=True).start()

    out_dir = SMOKE_OUT_DIR if smoke else OUT_DIR
    cmd = [
        "python",
        "scripts/train_sft_matched_pool.py",
        "--adapter",
        ADAPTER,
        "--expected-adapter-sha256",
        EXPECTED_ADAPTER_SHA256,
        "--output-dir",
        out_dir,
        "--run-name",
        f"{RUN_NAME}{'_smoke' if smoke else ''}",
        "--seed",
        "42",
        "--per-device-train-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "4",
        "--max-steps",
        "150",
        "--save-steps",
        "50",
        "--no-wandb",
    ]
    if smoke:
        cmd.append("--smoke")

    print(f"[modal-matched-sft] launching: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    finally:
        stop.set()
        ckpt_vol.commit()
    print(f"[modal-matched-sft] {'SMOKE' if smoke else 'RUN'} complete -> {out_dir}", flush=True)


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    train_matched_pool.remote(smoke=smoke)
