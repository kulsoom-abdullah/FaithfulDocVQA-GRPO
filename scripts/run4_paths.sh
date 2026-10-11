#!/usr/bin/env bash
# Single source of truth for every run-4 path, on the pod and locally.
#
# WHY THIS FILE EXISTS (runbook §4e.5): run-4a's pull script encoded its own guesses
# ($REMOTE/data/checkpoints/, $REMOTE/job4a.log) while the launch used
# --output-dir /workspace/grpo_runs and tee /workspace/job4a.log. Four consecutive
# failures, one path per attempt. Both scripts now source THIS file, so a path cannot
# disagree between them: there is only one definition to be wrong, and the preflight
# checks that one against the live pod.
#
# Usage:  . scripts/run4_paths.sh <arm>      # arm = 4a | 4b
# bash 3.2 compatible: no ${VAR^^}, no associative arrays, no mapfile (§4e.3).

FDV_ARM="${1:-}"
case "$FDV_ARM" in
  4a|4b) ;;
  *) echo "run4_paths.sh: arm must be 4a or 4b, got '${FDV_ARM}'" >&2; return 2 2>/dev/null || exit 2 ;;
esac

# ---- remote (pod) ----
# FDV_PATH_PREFIX relocates the whole remote tree. EMPTY in production, so the paths
# below are literally the pod's. Tests set it to a fixture root so the full flow —
# preflight, source manifest, hash acceptance — can be exercised end to end against a
# real tree without renting a pod. It is a test seam, not a path guess: the structure
# under the prefix is the single definition both scripts share.
FDV_PATH_PREFIX="${FDV_PATH_PREFIX:-}"
REMOTE_REPO="${FDV_PATH_PREFIX}/workspace/FaithfulDocVQA-GRPO"
REMOTE_OUT_ROOT="${FDV_PATH_PREFIX}/workspace/grpo_runs"
RUN_NAME="grpo_run${FDV_ARM}_seed42"
REMOTE_RUN_DIR="${REMOTE_OUT_ROOT}/${RUN_NAME}"
REMOTE_LOG="${FDV_PATH_PREFIX}/workspace/job${FDV_ARM}.log"
REMOTE_SEED_ADAPTER="${REMOTE_REPO}/seed/ck-1296"
TMUX_SESSION="grpo${FDV_ARM}"
MARKER_OK="JOB$(printf '%s' "$FDV_ARM" | tr '[:lower:]' '[:upper:]')_COMPLETE"
MARKER_BAD="JOB$(printf '%s' "$FDV_ARM" | tr '[:lower:]' '[:upper:]')_FAILED"

# ---- local ----
LOCAL_RUN_DIR="data/checkpoints/${RUN_NAME}"
LOCAL_LOG="data/checkpoints/${RUN_NAME}/job${FDV_ARM}.log"

# ---- data files the trainer must find on the pod ----
REQUIRED_DATA="data/master.jsonl data/manufactured_train_abstains_nodecoy.jsonl data/manufactured_val_abstains.jsonl data/grpo_val_instrument_v2_recursfalse.jsonl"

# ---- the stack run-4a ACTUALLY ran, from its own TRL-generated run-dir README ----
# (data/checkpoints/grpo_run4a_seed42/README.md "Framework versions"). 4b must match
# exactly, or the arms differ by more than the one reward constant. Asserted against
# `pip freeze` on the pod, through whatever venv env.sh resolves to — checking a
# hardcoded venv path would miss the exact trap that bit run 4b's first smoke, where
# env.sh still pointed at a corrupted venv. Overridable only so the assertion itself
# can be tested in the failing direction.
FDV_REQUIRED_PINS="${FDV_REQUIRED_PINS:-torch==2.11.0+cu128 transformers==5.12.0 trl==1.8.0 datasets==5.0.1 tokenizers==0.22.2}"

# ---- the run-4 config per the local analysis plan (4a's, with the one reward
# ---- constant changed for 4b) ----
#
# BASE_ARGS holds only what is invariant between a smoke and the real run. The full-run
# knobs live in TRAIN_ARGS ALONE, because main() applies CLI overrides AFTER
# smoke_overrides(): a SMOKE_ARGS built as "$TRAIN_ARGS --smoke" silently re-overrode
# max_steps 20 -> 150 and the pool 40/60 -> 1764/2646, so run 4b's "smoke" was a 150-step
# run on an n=32 val instrument. Keep full-run knobs out of BASE_ARGS.
BASE_ARGS="--run 2 --adapter ${REMOTE_SEED_ADAPTER} --output-dir ${REMOTE_OUT_ROOT} --seed 42 --train-abstains manufactured_train_abstains_nodecoy.jsonl --val-recurs-false-only"
if [ "$FDV_ARM" = "4b" ]; then
  BASE_ARGS="${BASE_ARGS} --false-abstain-base -0.2"
fi

# full-run knobs — TRAIN_ARGS only, never SMOKE_ARGS
TRAIN_ARGS="${BASE_ARGS} --max-steps 150 --eval-every 10 --probe-every 10 --save-steps 10 --pool-abstain 1764 --pool-answerable 2646 --run-name ${RUN_NAME}"

# smoke: let smoke_overrides() govern shape. --waive-gate3 is the documented run-4 waiver.
SMOKE_ARGS="${BASE_ARGS} --run-name ${RUN_NAME}_smoke --smoke --waive-gate3"

# ---- the RESOLVED config the launch must see, or refuse (assert, do not comment) ----
# Checked against `train_grpo.py --print-config`, which resolves the config through the
# same override order a real run uses and rebuilds the val instrument. The n=32 instrument
# silently replacing the specified 112-row one is the specific failure this catches.
EXPECTED_MAX_STEPS=150
EXPECTED_SAVE_STEPS=10
EXPECTED_EVAL_EVERY=10
EXPECTED_POOL_ABSTAIN=1764
EXPECTED_POOL_ANSWERABLE=2646
EXPECTED_TEMPERATURE=1.0
EXPECTED_TOP_P=1.0
EXPECTED_NUM_GENERATIONS=8
EXPECTED_FALSE_ABSTAIN_BASE=-0.2
EXPECTED_SMOKE=false
EXPECTED_INSTRUMENT_N=112
EXPECTED_INSTRUMENT_ABSTAIN=64
EXPECTED_INSTRUMENT_ANSWERABLE=48
# sha256 of the sorted record_ids in data/grpo_val_instrument_v2_recursfalse.jsonl
EXPECTED_INSTRUMENT_SHA="b4bc1708da1f31084ac061c223673a3376209081320baa7b178a57767035ea53"
EXPECTED_ADAPTER_SUFFIX="ck-1296"
