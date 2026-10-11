#!/usr/bin/env bash
# launch_pod.sh — sync FaithfulDocVQA to a fresh RunPod GPU and bootstrap it.
# Run on your Mac (volume-less workflow: grab any 24GB+ GPU in any region, sync, go).
#
#   ./launch_pod.sh          sync project up + run the validated bootstrap on the pod
#   ./launch_pod.sh --pull   pull trained adapters (data/checkpoints/) BACK to the Mac
#
# Remembers the last pod IP/port in .last_pod so you don't retype. Edit LOCAL_DIR /
# REMOTE_DIR below if your paths differ.
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/workspace/FaithfulDocVQA-GRPO"
SSH_KEY="$HOME/.ssh/id_ed25519"
LAST="${LOCAL_DIR}/.last_pod"

# ---- pod IP/port: default to last-used ----
DEF_IP=""; DEF_PORT=""
[[ -f "$LAST" ]] && { . "$LAST"; DEF_IP="${POD_IP:-}"; DEF_PORT="${POD_PORT:-}"; }
read -rp "Pod IP [${DEF_IP}]: " IN_IP;     POD_IP="${IN_IP:-$DEF_IP}"
read -rp "Pod SSH port [${DEF_PORT}]: " IN_PORT; POD_PORT="${IN_PORT:-$DEF_PORT}"
[[ -z "$POD_IP" || -z "$POD_PORT" ]] && { echo "ERROR: IP and port are both required." >&2; exit 1; }
printf 'POD_IP=%s\nPOD_PORT=%s\n' "$POD_IP" "$POD_PORT" > "$LAST"

SSH_OPTS="-p ${POD_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"
REMOTE="root@${POD_IP}"
cd "${LOCAL_DIR}"   # rsync source is THIS dir only -> no sibling-folder sweep

# ---- preflight: connectivity + ensure rsync exists on the pod (many images lack it) ----
echo ">>> SSH check + ensure rsync on pod -> ${REMOTE} (port ${POD_PORT})"
ssh ${SSH_OPTS} "${REMOTE}" \
  "mkdir -p ${REMOTE_DIR}; command -v rsync >/dev/null 2>&1 || { echo 'installing rsync...'; apt-get update -qq && apt-get install -y -qq rsync; }; echo connected OK" || {
  echo 'ERROR: SSH failed or could not install rsync. Check IP/port/key.' >&2; exit 1; }

# ---- --pull: bring adapters back, then exit (do this BEFORE terminating a pod) ----
if [[ "${1:-}" == "--pull" ]]; then
  echo ">>> Pulling ${REMOTE}:${REMOTE_DIR}/data/checkpoints/ -> ./data/checkpoints/"
  mkdir -p data/checkpoints
  rsync -rltvP --no-owner --no-group -e "ssh ${SSH_OPTS}" \
    "${REMOTE}:${REMOTE_DIR}/data/checkpoints/" ./data/checkpoints/
  echo ">>> Adapters are safe locally. OK to terminate the pod."
  exit 0
fi

# ---- rsync up (data/images IN; checkpoints/wandb/logs OUT) ----
echo ">>> rsync project up (delta)..."
rsync -rltvP --no-owner --no-group \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' --exclude 'augraphy_cache' \
  --exclude 'wandb/' --exclude 'data/checkpoints/' --exclude 'data/*.log' \
  -e "ssh ${SSH_OPTS}" \
  ./ "${REMOTE}:${REMOTE_DIR}/"

# ---- run the ONE validated bootstrap (deps, torch>=2.5, env+xet-off, cap+data checks, loss-mask test) ----
echo ">>> Bootstrapping via scripts/runpod_bootstrap.sh ..."
ssh ${SSH_OPTS} "${REMOTE}" "cd ${REMOTE_DIR} && bash scripts/runpod_bootstrap.sh"

cat <<EOF

>>> Pod synced + bootstrapped. SSH in and run interactively (watch/detach):

  ssh ${SSH_OPTS} ${REMOTE}
  cd ${REMOTE_DIR} && source /workspace/env.sh

  nvidia-smi                                  # confirm the GPU is YOURS (used VRAM ~0)
  python scripts/train_sft.py --smoke --mode B   # validate before the long run

  tmux new -s sft                             # full run, detachable
  source /workspace/env.sh
  python scripts/train_sft.py --mode B 2>&1 | tee data/checkpoints/sftB.log
  #   detach: Ctrl-b d   |   reattach: tmux attach -t sft
  # after B: python scripts/train_sft.py --mode A 2>&1 | tee data/checkpoints/sftA.log

>>> When the runs finish, ON YOUR MAC:  ./launch_pod.sh --pull   (save adapters, THEN kill the pod)
EOF
