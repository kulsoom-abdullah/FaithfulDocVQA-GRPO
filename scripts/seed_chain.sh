#!/usr/bin/env bash
# One-seed SFT-B chain for the multi-seed peak-replication study. Run DETACHED in
# tmux ON the pod so it survives ssh disconnect and holds the GPU through train->gen
# (no release between). Config is train_sft.py's mode-B default (identical to seed-42
# except --seed): bs1 x grad_accum 8, LoRA r16/a32 q/k/v/o, lr 1e-4 cosine+warmup,
# bf16, 4-bit, 3 epochs / save_steps 214 -> 6 checkpoints.
#
#   bash scripts/seed_chain.sh <SEED> [OUT_DIR]
#
# Stages (each gated on the prior exit code; a failure aborts, trained ckpts are kept):
#   1 train (GPU)  ->  2 val-gen over the 6 checkpoints (GPU)  ->  3 CPU score curve.
# Everything streams to logs + a CHAIN_STATUS file under OUT_DIR; the val curve
# (honest_abstain / false_grounding per checkpoint) lands at OUT_DIR/seed<N>_val_curve.json.
set -uo pipefail

# ── perf/env fix for the cu124 torch-2.5.1 stack on RTX 6000 Ada ────────────────
# On the first launch, torch inductor recompiled on dynamic shapes (variable menu/
# image sizes), spawning `_inductor/compile_worker --workers=32` that thrashed the
# ~13.6-CPU cgroup quota (load avg ~17.5) -> s/it climbed 40->78 monotonically.
# Force EAGER execution (seed-42 ran eager: torch_compile off) so there is no compile
# storm, and cap CPU thread pools. All numerically NEUTRAL — they change compilation
# / threading, not the trained model, so config-identity with seed-42 holds.
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=4
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

SEED=${1:?usage: seed_chain.sh <SEED> [OUT_DIR]}
OUT_DIR=${2:-data/checkpoints/sft_v3_seeds}
RUN_DIR="${OUT_DIR}/sft_B_seed${SEED}"
EVAL_DIR="data/checkpoints/eval/sft_B_seed${SEED}"
STATUS="${OUT_DIR}/CHAIN_STATUS_seed${SEED}"
CURVE="${OUT_DIR}/seed${SEED}_val_curve.json"
mkdir -p "$OUT_DIR"

stamp() { date "+%Y-%m-%d %H:%M:%S"; }
st()    { echo "$(stamp)  seed${SEED}  $*" | tee -a "$STATUS"; }

st "CHAIN START  out=${OUT_DIR}  (train 3ep -> val-gen 6 ckpts -> score)"

# ── stage 1: train ─────────────────────────────────────────────────────────────
st "STAGE 1 train: STARTED -> ${OUT_DIR}/train_seed${SEED}.log"
python scripts/train_sft.py --mode B --seed "$SEED" --output-dir "$OUT_DIR" \
    --epochs 3 --save-total-limit 8 --no-wandb 2>&1 | tee "${OUT_DIR}/train_seed${SEED}.log"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then st "STAGE 1 train: FAILED (rc=${RC}) -> ABORT"; exit "$RC"; fi
st "STAGE 1 train: OK"

# ── stage 2: val-only generation over all 6 checkpoints (GPU held) ─────────────
st "STAGE 2 val-gen: STARTED -> ${OUT_DIR}/gen_seed${SEED}.log"
python scripts/gen_persist.py --run-dir "$RUN_DIR" --split validation 2>&1 \
    | tee "${OUT_DIR}/gen_seed${SEED}.log"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then st "STAGE 2 val-gen: FAILED (rc=${RC}) -> ABORT (checkpoints safe)"; exit "$RC"; fi
st "STAGE 2 val-gen: OK"

# ── stage 3: CPU score -> per-checkpoint honest_abstain / false_grounding curve ─
st "STAGE 3 score: STARTED"
python scripts/score_val_curve.py --gens "$EVAL_DIR" --split validation --out "$CURVE" 2>&1 \
    | tee "${OUT_DIR}/score_seed${SEED}.log"
RC=${PIPESTATUS[0]}
if [ "$RC" -ne 0 ]; then st "STAGE 3 score: FAILED (rc=${RC})"; exit "$RC"; fi

st "CHAIN COMPLETE  seed=${SEED}  curve -> ${CURVE}"
echo ">>> seed ${SEED} val curve:"; cat "$CURVE"
