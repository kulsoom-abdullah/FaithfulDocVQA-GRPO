"""Modal entrypoint for FaithfulDocVQA GRPO Stage 3 — wraps ``grpo/train_grpo.py``.

Companion to ``modal_train.py`` / ``modal_gen.py`` (same 3-volume layout, same
REPO_ROOT convention). ``train_grpo.py`` itself is UNCHANGED — this file only
builds the environment and invokes it. Differences from the SFT image, on purpose:

  - **cu128 torch (2.11.0), not cu124.** The cu124 stack showed the unbounded
    s/it climb on Ada (see docs/findings); Stage 2 validated the cu128
    stack on Ada. GPU is **A100-80GB**: the unchunked 32-row generation geometry
    OOMs 48 GB-class cards, and Modal L40S containers vary 44.4–47.4 GiB usable.
  - **trl==1.8.0 installed --no-deps** (its runtime deps — datasets/pandas/rich —
    added explicitly), mirroring the pod venv recipe that Stage 2 proved, so trl
    can never bump the pinned transformers/peft/accelerate.
  - **no bitsandbytes.** GRPO continues ck-642 in bf16 + LoRA (no 4-bit); nothing
    on this import path loads bnb (train_sft.build_bnb_config lazy-imports).
  - ships ``grpo/`` and ``phase0/`` in addition to ``scripts/`` (the reward chain
    imports ``phase0.evidence_reward``; phase0's heavy ``outputs/`` is excluded).

Layout inside the container (all persisted state on volumes):
  /root/faithfuldocvqa/data     <- faithfuldocvqa-data   (master.jsonl, images/,
                                   manufactured_{train,val}_abstains.jsonl)
  /root/faithfuldocvqa/outputs  <- faithfuldocvqa-ckpts  (ck-642 in; GRPO runs out
                                   under outputs/grpo_runs/, incl. the tee'd log)
  /cache                        <- faithfuldocvqa-hf-cache (base Qwen3-VL weights)

FIREWALL note: manufactured_test_abstains.jsonl sits on the data volume but
train_grpo.py never opens it (assert_no_test_leak guards every pool/instrument).

Usage (nothing here runs until invoked):
  modal run modal_grpo.py --smoke                    # 20-step gate (~15 min GPU)
  modal run --detach modal_grpo.py                   # run-1 control (committed runs: --max-steps 150)
Runbook (after --detach):
  modal app list                                     # find the running app id
  modal app logs faithfuldocvqa-grpo                 # stream logs
  modal app stop faithfuldocvqa-grpo                 # kill the run
  modal volume ls faithfuldocvqa-ckpts grpo_runs/grpo_run1_seed42
A smoke-gate FAIL exits nonzero by design, so the Modal function call FAILS with
CalledProcessError — the gate block itself is in the streamed/tee'd log.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "faithfuldocvqa-grpo"
REPO_ROOT = "/root/faithfuldocvqa"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
# ck-642 = the SFT seed adapter (1.5-epoch checkpoint). Selected on the n=46 validation
# abstain rows; NOTE the test split was scored for all six checkpoints before this seed
# was fixed, and ck-642 is the test argmax (0.805; neighbors 0.65-0.68) — see
# docs/PROVENANCE.md. Already on the ckpts volume from the SFT run — nothing to upload.
ADAPTER = f"{REPO_ROOT}/outputs/sft_v3_32g/sft_B_seed42/checkpoint-642"
LOCAL = Path(__file__).parent

# ── image: the cu128 stack Stage 2 validated on Ada (NOT modal_train's cu124) ─────
image = (
    modal.Image.debian_slim(python_version="3.11")
    # torch trio from the cu128 index; torchvision/torchaudio float to the versions
    # pip resolves against torch==2.11.0 on that index.
    .pip_install(
        "torch==2.11.0", "torchvision", "torchaudio",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    # pinned to the pod-validated versions (+ trl's runtime deps, installed WITH
    # resolution here so the --no-deps trl layer below finds them present).
    .pip_install(
        "transformers==5.12.0",
        "peft==0.19.1",
        "accelerate>=1.1.0",
        "numpy==1.26.3",
        "pillow==10.2.0",
        "pydantic>=2",
        "huggingface_hub",
        "datasets",
        "pandas",
        "rich",
    )
    # the Stage-2-proven recipe: trl must never re-resolve the pinned stack
    .pip_install("trl==1.8.0", extra_options="--no-deps")
    .env(
        {
            "HF_HOME": "/cache",
            "HF_HUB_DISABLE_XET": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    # code: scripts/ (format/eval/reward deps) + grpo/ (trainer) + phase0/
    # (evidence_reward import chain; exclude its 20+ MB outputs/ artifacts)
    .add_local_dir(str(LOCAL / "scripts"), f"{REPO_ROOT}/scripts",
                   ignore=["__pycache__", "*.pyc"])
    .add_local_dir(str(LOCAL / "grpo"), f"{REPO_ROOT}/grpo",
                   ignore=["__pycache__", "*.pyc"])
    .add_local_dir(str(LOCAL / "phase0"), f"{REPO_ROOT}/phase0",
                   ignore=["__pycache__", "*.pyc", "outputs", "*.html", "*.md"])
)

app = modal.App(APP_NAME)

# ── volumes: identical names + mount paths as modal_train.py / modal_gen.py ───────
data_vol = modal.Volume.from_name("faithfuldocvqa-data")
ckpt_vol = modal.Volume.from_name("faithfuldocvqa-ckpts")
hf_vol = modal.Volume.from_name("faithfuldocvqa-hf-cache")
VOLUMES = {
    f"{REPO_ROOT}/data": data_vol,
    f"{REPO_ROOT}/outputs": ckpt_vol,
    "/cache": hf_vol,
}


@app.function(
    gpu="A100-80GB",           # 48 GB-class OOMs the unchunked 32-row geometry; L40S
                               # containers also vary 44.4–47.4 GiB usable (smoke #3 kill)
    image=image,
    volumes=VOLUMES,
    timeout=24 * 60 * 60,      # Modal's hard max (86400 s); A100 run should finish well inside
    cpu=8.0,
    memory=32 * 1024,
)
def grpo_train(run: int = 1, smoke: bool = False, max_steps: int | None = None,
               eval_every: int | None = None, probe_every: int | None = None,
               seed: int = 42) -> None:
    """Invoke grpo/train_grpo.py exactly as the pod would, volumes standing in for
    /workspace. Outputs land on the ckpts volume at outputs/grpo_runs/<run-name>/
    (train_grpo resolves the relative --output-dir against cwd=REPO_ROOT), and the
    full stdout is tee'd next to them as <run-name>.log."""
    import shlex
    import subprocess
    import threading

    from huggingface_hub import snapshot_download

    # base weights: no-op when already cached on the volume (they are, from SFT)
    print(f"[modal-grpo] ensuring {MODEL_ID} cached on /cache ...", flush=True)
    snapshot_download(MODEL_ID)
    hf_vol.commit()

    # periodic ckpt-volume commit — Modal volumes persist ONLY on .commit(); without
    # this a crash/timeout loses telemetry+checkpoints since the last commit.
    commit_interval = 60 if smoke else 600
    stop = threading.Event()

    def committer() -> None:
        while not stop.wait(commit_interval):
            try:
                ckpt_vol.commit()
                print("[modal-grpo] periodic ckpt commit", flush=True)
            except Exception as e:  # noqa: BLE001 — a commit hiccup must not kill the run
                print(f"[modal-grpo] ckpt commit warning: {e}", flush=True)

    threading.Thread(target=committer, daemon=True).start()

    name = f"grpo_run{run}_seed{seed}{'_smoke' if smoke else ''}"
    out_rel = "outputs/grpo_runs"
    cmd = ["python", "grpo/train_grpo.py", "--run", str(run),
           "--adapter", ADAPTER, "--output-dir", out_rel, "--seed", str(seed)]
    if smoke:
        cmd.append("--smoke")
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if eval_every is not None:
        cmd += ["--eval-every", str(eval_every)]
    if probe_every is not None:
        cmd += ["--probe-every", str(probe_every)]

    # pipefail so a smoke-gate FAIL (train_grpo exits 1) survives the tee
    shell = (f"set -o pipefail; mkdir -p {out_rel} && "
             f"{shlex.join(cmd)} 2>&1 | tee {out_rel}/{name}.log")
    print(f"[modal-grpo] launching: {shell}  (cwd={REPO_ROOT})", flush=True)
    try:
        subprocess.run(["bash", "-c", shell], cwd=REPO_ROOT, check=True)
    finally:
        stop.set()
        ckpt_vol.commit()  # final flush — log + artifacts persist even on FAIL
    print(f"[modal-grpo] done; artifacts committed under {out_rel}/{name}/", flush=True)


@app.local_entrypoint()
def main(run: int = 1, smoke: bool = False, max_steps: int | None = None,
         eval_every: int | None = None, probe_every: int | None = None,
         seed: int = 42) -> None:
    """Smoke gate:   modal run modal_grpo.py --smoke
    Run-1 control:   modal run --detach modal_grpo.py
    (train_grpo dataclass defaults: max-steps 500 / eval-every 50 / probe-every 25;
    the committed runs passed --max-steps 150.)
    """
    grpo_train.remote(run=run, smoke=smoke, max_steps=max_steps,
                      eval_every=eval_every, probe_every=probe_every, seed=seed)
