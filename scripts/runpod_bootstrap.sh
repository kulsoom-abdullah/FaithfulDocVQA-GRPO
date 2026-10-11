#!/usr/bin/env bash
# RunPod bootstrap for FaithfulDocVQA SFT — run ON the GPU pod, from the repo root
# (/workspace/FaithfulDocVQA-GRPO). Idempotent; safe to re-run on a fresh pod.
#
# Encodes every setup lesson so region/pod-hopping is one command:
#   - installs deps (incl. bitsandbytes) + upgrades torch to >=2.5 if needed,
#   - writes a persistent env file on the VOLUME (HF cache on the volume, xet OFF —
#     xet stages on the quota'd ephemeral root and caused EDQUOT downloads),
#   - installs tmux, checks the data, verifies the 4.2 MP vision cap is active,
#     and runs the completion-only mask test.
#
# Prereqs: the repo + data/ (master.jsonl AND images/) already rsync'd. IMPORTANT:
# rsync FROM INSIDE the repo dir (`cd .../faithfuldocvqa` then `rsync ./ ...`) so
# sibling project folders don't get swept onto the volume and blow the quota.
set -euo pipefail

echo "== FaithfulDocVQA RunPod bootstrap =="

# 1. persistent env on the VOLUME (survives this subshell; source it in every shell)
ENV_FILE=/workspace/env.sh
mkdir -p /workspace/hf-cache /workspace/tmp
cat > "$ENV_FILE" <<'EOF'
export HF_HOME=/workspace/hf-cache
export HF_HUB_DISABLE_XET=1     # xet stages on the ephemeral root -> EDQUOT; force classic dl
export TMPDIR=/workspace/tmp
EOF
# shellcheck disable=SC1090
source "$ENV_FILE"
echo "wrote $ENV_FILE -> HF_HOME=$HF_HOME, xet disabled, TMPDIR=$TMPDIR"
echo "   (run 'source $ENV_FILE' in each new shell / tmux window)"

# 2. tmux (for detachable long runs)
command -v tmux >/dev/null 2>&1 || { echo "installing tmux..."; apt-get update -qq && apt-get install -y -qq tmux; }

# 3. python deps — pin transformers to the version Qwen3-VL was validated on
python -m pip install -q \
  "transformers==5.12.0" "peft==0.19.1" "accelerate>=1.0" bitsandbytes pydantic pillow pytest wandb

# 3b. Install a torch build that ACTUALLY SUPPORTS this GPU's arch. Keep torch /
#     torchvision / torchAUDIO in LOCKSTEP — transformers imports torchaudio, so a
#     torch/torchaudio ABI mismatch breaks `import transformers`.
#     The decisive test is NOT "torch >= 2.5" but "does the installed torch's compiled
#     arch list cover this card's compute capability". RunPod images come PRE-LOADED
#     with a recent cu124 torch (>=2.5) that silently passed the old version-only check
#     on a Blackwell sm_120 card it CANNOT run — the no-op that left cu124 in place.
#     Detection is hardened: nvidia-smi compute_cap, then the GPU *name* ("Blackwell"),
#     then torch — so an empty compute_cap query no longer mis-routes to the cu124 path.
#       - Blackwell (cc major >= 12, e.g. RTX PRO 4000/4500/6000 Blackwell) -> cu128.
#       - older arch with torch too old / arch-unsupported -> torch 2.5.1 on cu124.
DETECT=$(python - <<'PY'
import subprocess

def smi(field):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
        return lines[0] if lines else ""
    except Exception:
        return ""

cc = smi("compute_cap")                                  # e.g. "12.0"
cc_major = int(cc.split(".")[0]) if cc[:1].isdigit() else 0
if cc_major == 0:                                        # fall back to the GPU name
    nl = smi("name").lower()
    if any(k in nl for k in ("blackwell", "b100", "b200", "rtx 50", "5090", "5080")):
        cc, cc_major = "12.0", 12

supported, torch_cuda, torch_old = "unknown", "none", "no"
try:
    import torch
    torch_cuda = torch.version.cuda or "cpu"
    maj, minr = map(int, torch.__version__.split("+")[0].split(".")[:2])
    torch_old = "yes" if (maj, minr) < (2, 5) else "no"
    if torch.cuda.is_available():
        if cc_major == 0:                                # last resort: ask the driver
            dmaj, dminr = torch.cuda.get_device_capability(0)
            cc, cc_major = f"{dmaj}.{dminr}", dmaj
        supported = "yes" if ("sm_" + cc.replace(".", "")) in torch.cuda.get_arch_list() else "no"
    else:
        supported = "no"                                 # torch can't use the GPU at all
except Exception:
    supported, torch_old = "no", "yes"
print(cc or "0.0", cc_major or 0, supported, torch_cuda, torch_old)
PY
)
read -r CC CC_MAJOR SUPPORTED TORCH_CUDA TORCH_OLD <<< "$DETECT"
echo "GPU cc=${CC} (major ${CC_MAJOR}) | current torch supports this arch: ${SUPPORTED} | torch cuda: ${TORCH_CUDA}"
if [ "$CC_MAJOR" -ge 12 ] && [ "$SUPPORTED" != "yes" ]; then
  echo "Blackwell (sm_${CC}) NOT supported by current torch (${TORCH_CUDA}) -> installing cu128 stack"
  python -m pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
elif [ "$CC_MAJOR" -lt 12 ] && { [ "$SUPPORTED" = "no" ] || [ "$TORCH_OLD" = "yes" ]; }; then
  echo "non-Blackwell, torch too old / arch-unsupported -> installing torch 2.5.1 stack (cu124)"
  python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
else
  echo "current torch (${TORCH_CUDA}) already supports sm_${CC} -> keeping it"
fi

# 4. sanity: GPU + versions + bitsandbytes import
python - <<'PY'
import torch, transformers, peft
print("torch       ", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")
print("transformers", transformers.__version__, "| peft", peft.__version__)
try:
    import bitsandbytes as bnb; print("bitsandbytes", bnb.__version__)
except Exception as e:
    print("bitsandbytes IMPORT FAILED:", e)
PY

# 5. data present + correct size?
test -f data/master.jsonl || { echo "!! MISSING data/master.jsonl — rsync from the repo root first"; exit 1; }
n_rec=$(wc -l < data/master.jsonl | tr -d ' ')
n_img=$(ls data/images/*.png 2>/dev/null | wc -l | tr -d ' ')
echo "data: master.jsonl=${n_rec} records (expect 4932) | images=${n_img} png (expect 1234)"

# 6. verify the 4.2 MP vision cap is active (downloads the processor on first run)
python - <<'PY'
import sys; sys.path.insert(0, ".")
from scripts.format_sft import load_processor, MAX_PIXELS
p, _ = load_processor()
edge = p.image_processor.size.get("longest_edge")
print(f"vision cap: MAX_PIXELS={MAX_PIXELS} longest_edge={edge}",
      "OK" if edge == MAX_PIXELS else "!! CAP MISMATCH")
PY

# 7. completion-only loss-mask test (CPU)
python -m pytest tests/test_loss_mask.py -q

echo
echo "BOOTSTRAP OK."
echo "  source /workspace/env.sh    # then, to validate end-to-end:"
echo "  python scripts/train_sft.py --smoke --mode B"
echo "  # full run in tmux:  tmux new -s sft ; source /workspace/env.sh ; python scripts/train_sft.py --mode B 2>&1 | tee data/checkpoints/sftB.log"
