#!/usr/bin/env bash
# Unattended 3-field baseline chain (run ON the pod, inside tmux):
#
#     train 3 epochs  ->  generate+persist raw validation outputs per checkpoint  ->  idle
#
# Each stage runs ONLY if the prior stage exited 0. NO scoring (scoring is a separate
# CPU step run afterwards), NO second training run, and generation NEVER starts if training
# errored. Everything streams to logs AND a CHAIN_STATUS file under the run dir, so the
# end-state is visible at a glance on return — even after the pod is later stopped.
#
#   bash scripts/unattended_run.sh [OUT_DIR] [MODE] [SEED] [EPOCHS] [SAVE_LIMIT]
# Defaults are baked for this run; normally just:  bash scripts/unattended_run.sh
set -uo pipefail   # NOT -e: stages are gated on explicit exit codes, not auto-abort

OUT_DIR=${1:-data/checkpoints/sft_v3}
MODE=${2:-B}
SEED=${3:-42}
EPOCHS=${4:-3}
SAVE_LIMIT=${5:-8}                       # 6 checkpoints expected; 8 keeps them all w/ headroom
RUN_DIR="${OUT_DIR}/sft_${MODE}_seed${SEED}"
STATUS="${OUT_DIR}/CHAIN_STATUS"
TRAIN_LOG="${OUT_DIR}/train.log"
GEN_LOG="${OUT_DIR}/generate.log"
mkdir -p "$OUT_DIR"

stamp()      { date "+%Y-%m-%d %H:%M:%S"; }
set_status() { echo "$(stamp)  $*" | tee -a "$STATUS"; }

set_status "CHAIN START  out=${OUT_DIR} mode=${MODE} seed=${SEED} epochs=${EPOCHS} save_limit=${SAVE_LIMIT}"

# ── stage 1: train ────────────────────────────────────────────────────────────
set_status "STAGE 1 train: STARTED  (-> ${TRAIN_LOG})"
python scripts/train_sft.py --mode "$MODE" --seed "$SEED" \
    --output-dir "$OUT_DIR" --epochs "$EPOCHS" --save-total-limit "$SAVE_LIMIT" --no-wandb \
    2>&1 | tee "$TRAIN_LOG"
TRAIN_RC=${PIPESTATUS[0]}
if [ "$TRAIN_RC" -ne 0 ]; then
    set_status "STAGE 1 train: FAILED (rc=${TRAIN_RC}) -> NOT generating. CHAIN ABORTED."
    exit "$TRAIN_RC"
fi
set_status "STAGE 1 train: OK"

# ── stage 2: generate + persist RAW outputs (no scoring) ──────────────────────
set_status "STAGE 2 generate+persist: STARTED  (-> ${GEN_LOG})"
python scripts/gen_persist.py --run-dir "$RUN_DIR" 2>&1 | tee "$GEN_LOG"
GEN_RC=${PIPESTATUS[0]}
if [ "$GEN_RC" -ne 0 ]; then
    set_status "STAGE 2 generate+persist: FAILED (rc=${GEN_RC}). CHAIN ABORTED (trained checkpoints are safe)."
    exit "$GEN_RC"
fi
set_status "STAGE 2 generate+persist: OK"

# ── stage 3: idle ─────────────────────────────────────────────────────────────
set_status "CHAIN COMPLETE: trained ${EPOCHS} epochs + persisted raw validation outputs. IDLE."
set_status "  checkpoints -> ${RUN_DIR}/checkpoint-*"
set_status "  raw outputs -> data/checkpoints/eval/sft_${MODE}_seed${SEED}/gen_checkpoint-*.json"
set_status "  on return (CPU, no rush): score with  python scripts/score_val_curve.py --gens data/checkpoints/eval/sft_${MODE}_seed${SEED}"
echo ">>> CHAIN COMPLETE — pod left running (idle) for you. Reattach: tmux attach -t sft"
