#!/usr/bin/env bash
# Pull a run-4 arm's artifacts off the pod and PROVE they arrived intact.
#
#   scripts/run4_pull_verify.sh <POD_IP> <POD_PORT> <arm> [--dry-run]
#
# Stages, in order (runbook §4e):
#   -1  preflight  : ONE listing of every real path, before any transfer (§4e.5)
#    0  manifest   : sha256 computed AT THE SOURCE, on the pod (§4e.6)
#    1  pull       : rsync
#    2  verify     : local sha256 vs the source manifest — the acceptance test (§4e.6)
#    3  completeness: checkpoint count, val_curve rows, zero-byte sweep
#
# Rules this script obeys, each earned the hard way:
#   §4e.1  PIPESTATUS captured on the very next line, never inside $( ), never after
#          an intervening command.
#   §4e.2  no `timeout` (absent on macOS); no pipe that can mask a non-zero status.
#   §4e.3  bash 3.2 clean — no ${VAR^^}, no associative arrays, no mapfile.
#   §4e.4  stderr is NEVER suppressed on a verification step. A `2>/dev/null` here
#          turns "Modal/ssh said why it failed" into "the thing is mysteriously empty".
#   §4e.6  hashes are the acceptance test. Counts and sizes are not: a batch put once
#          wrote 4 files with correct size and wrong content, and only hashes caught it.
#
# Testability: the remote runner is a function, overridable with FDV_REMOTE_SH, so the
# whole flow can be exercised end to end against a real path tree without renting a pod.
set -uo pipefail

usage() { echo "usage: $0 <HOST> <SSH_PORT> <4a|4b> [--dry-run] [--live]" >&2; exit 2; }
[ $# -ge 3 ] || usage
POD_IP="$1"; POD_PORT="$2"; ARM="$3"; DRY="${4:-}"
# --live: the run is STILL TRAINING. reward_log/reward_members/step_times/val_curve/
# probe_completions are appended continuously, so a file can legitimately differ between
# the source manifest and the pull. A mismatch is then re-checked AT THE SOURCE: if the
# source hash also moved, the file churned (expected); if the source hash is unchanged,
# the bytes really are corrupt. Completeness counts become informational, since a
# mid-run pull has fewer than 15 checkpoints by definition.
LIVE=0
case "${4:-}${5:-}" in *--live*) LIVE=1 ;; esac

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not a git repo" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2
. scripts/run4_paths.sh "$ARM" || exit 2

# Environment-specific default; override with FDV_SSH_KEY.
SSH_KEY="${FDV_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH_OPTS="-p ${POD_PORT} -i ${SSH_KEY} -o StrictHostKeyChecking=accept-new"
REMOTE="root@${POD_IP}"

# The one seam that makes this script testable. Default: real ssh. Tests override it.
remote_sh() {
  if [ -n "${FDV_REMOTE_SH:-}" ]; then
    "$FDV_REMOTE_SH" "$1"
  else
    ssh $SSH_OPTS "$REMOTE" "$1"
  fi
}

fail() { echo ""; echo "FAILED: $*" >&2; exit 1; }
say()  { echo ""; echo "=== $* ==="; }

# ── stage -1: preflight every real path in ONE pass (§4e.5) ─────────────────────
say "stage -1  preflight (arm=${ARM})"
echo "  remote run dir : ${REMOTE_RUN_DIR}"
echo "  remote log     : ${REMOTE_LOG}"
# Emit a machine-readable verdict per path. Matching on English error text would
# break under a different locale; MISS/OK is deterministic.
PRE=$(remote_sh "for p in '${REMOTE_RUN_DIR}' '${REMOTE_LOG}' '${REMOTE_REPO}'; do if [ -e \"\$p\" ]; then echo \"OK   \$p\"; else echo \"MISS \$p\"; fi; done; echo '--- run dir contents ---'; ls -1 '${REMOTE_RUN_DIR}' 2>&1 | head -40")
PRE_RC=$?
echo "$PRE" | sed 's/^/    /'
[ "$PRE_RC" -eq 0 ] || fail "preflight could not reach the pod (rc=${PRE_RC})."
MISS_N=$(printf '%s\n' "$PRE" | grep -c '^MISS ')
[ "$MISS_N" -eq 0 ] || fail "${MISS_N} required remote path(s) missing — see MISS lines above. Fix the PATHS in scripts/run4_paths.sh, do not paper over it (§4e.5)."

# ── stage 0: source-side manifest (§4e.6) ───────────────────────────────────────
say "stage 0   source-side sha256 manifest"
SRC_MANIFEST=$(remote_sh "cd '${REMOTE_RUN_DIR}' && find . -type f \\( -name '*.jsonl' -o -name '*.json' -o -name '*.md' \\) | LC_ALL=C sort | xargs sha256sum")
SRC_RC=$?
[ "$SRC_RC" -eq 0 ] || fail "could not compute source manifest (rc=${SRC_RC})"
SRC_N=$(printf '%s\n' "$SRC_MANIFEST" | grep -c '^[0-9a-f]\{64\}')
echo "  files in source manifest: ${SRC_N}"
[ "$SRC_N" -gt 0 ] || fail "source manifest is empty"

if [ "$DRY" = "--dry-run" ]; then
  printf '%s\n' "$SRC_MANIFEST" | head -8 | sed 's/^/    /'
  say "DRY RUN — stopping before transfer. Preflight and manifest both succeeded."
  exit 0
fi

# ── stage 1: pull ───────────────────────────────────────────────────────────────
say "stage 1   pull -> ${LOCAL_RUN_DIR}"
mkdir -p "${LOCAL_RUN_DIR}" || fail "cannot create ${LOCAL_RUN_DIR}"
if [ -n "${FDV_REMOTE_SH:-}" ]; then
  cp -R "${REMOTE_RUN_DIR}/." "${LOCAL_RUN_DIR}/"
else
  rsync -rltvP --no-owner --no-group -e "ssh ${SSH_OPTS}" \
    "${REMOTE}:${REMOTE_RUN_DIR}/" "${LOCAL_RUN_DIR}/"
fi
PULL_RC=$?
[ "$PULL_RC" -eq 0 ] || fail "pull returned rc=${PULL_RC}"

# the log lives outside the run dir — the exact thing §4e.5 got wrong
if [ -n "${FDV_REMOTE_SH:-}" ]; then
  cp "${REMOTE_LOG}" "${LOCAL_LOG}" 2>&1 || echo "  note: log copy failed (non-fatal)"
else
  rsync -rltvP --no-owner --no-group -e "ssh ${SSH_OPTS}" \
    "${REMOTE}:${REMOTE_LOG}" "${LOCAL_LOG}" || echo "  note: log copy failed (non-fatal)"
fi

# ── stage 2: verify against the SOURCE manifest — the acceptance test (§4e.6) ───
say "stage 2   verify local hashes against the source manifest"
VERIFIED=0; MISMATCH=0; MISSING=0; CHURNED=0
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  h=$(printf '%s' "$line" | awk '{print $1}')
  p=$(printf '%s' "$line" | awk '{print $2}'); p="${p#./}"
  lf="${LOCAL_RUN_DIR}/${p}"
  if [ ! -f "$lf" ]; then
    echo "  MISSING  ${p}"; MISSING=$((MISSING+1)); continue
  fi
  lh=$(shasum -a 256 "$lf" | awk '{print $1}')
  if [ "$lh" = "$h" ]; then
    VERIFIED=$((VERIFIED+1))
  else
    if [ "$LIVE" -eq 1 ]; then
      # re-hash at the source: did the FILE move on, or did the BYTES corrupt?
      rh=$(remote_sh "sha256sum '${REMOTE_RUN_DIR}/${p}' 2>/dev/null | cut -d' ' -f1")
      if [ "$rh" != "$h" ]; then
        echo "  CHURNED  ${p} (source advanced during pull — expected mid-run)"
        CHURNED=$((CHURNED+1))
        continue
      fi
    fi
    echo "  MISMATCH ${p}"; echo "    source=${h}"; echo "    local =${lh}"
    MISMATCH=$((MISMATCH+1))
  fi
done <<EOF
$(printf '%s\n' "$SRC_MANIFEST")
EOF
echo "  verified ${VERIFIED}/${SRC_N}   mismatched ${MISMATCH}   missing ${MISSING}   churned ${CHURNED}"
[ "$MISMATCH" -eq 0 ] || fail "${MISMATCH} file(s) differ from source. Re-put those paths individually (§4e.6) — the volume is not at fault."
[ "$MISSING"  -eq 0 ] || fail "${MISSING} file(s) never arrived."
if [ "$LIVE" -eq 1 ]; then
  [ $((VERIFIED+CHURNED)) -eq "$SRC_N" ] || fail "verified+churned $((VERIFIED+CHURNED)) of ${SRC_N}"
else
  [ "$VERIFIED" -eq "$SRC_N" ] || fail "verified ${VERIFIED} of ${SRC_N}"
fi

# ── stage 3: completeness of the artifacts themselves ───────────────────────────
say "stage 3   artifact completeness"
# find, not a glob+2>/dev/null: an unmatched glob makes ls error, and suppressing
# that stderr on a verification step is exactly what §4e.4 forbids.
CKPTS=$(find "${LOCAL_RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | wc -l | tr -d ' ')
echo "  checkpoints pulled     : ${CKPTS} (expect 15 at save-steps 10 over 150 steps)"
if [ -f "${LOCAL_RUN_DIR}/val_curve.jsonl" ]; then
  VROWS=$(grep -c . "${LOCAL_RUN_DIR}/val_curve.jsonl")
  echo "  val_curve.jsonl rows   : ${VROWS} (expect 15 — §(g): an unscored checkpoint is ineligible)"
else
  echo "  val_curve.jsonl        : ABSENT"; VROWS=0
fi
ZB=$(find "${LOCAL_RUN_DIR}" -type f -size 0 | wc -l | tr -d ' ')
echo "  zero-byte files        : ${ZB}"
if [ "$LIVE" -eq 1 ]; then
  echo "  (--live: counts are informational; a mid-run pull has <15 checkpoints by design)"
  [ "$ZB" -eq 0 ] || echo "  WARN ${ZB} zero-byte file(s) — re-check at the final pull"
else
  [ "$ZB" -eq 0 ] || fail "${ZB} zero-byte file(s) — the §4e.6 signature"
fi

say "PULL VERIFIED — ${VERIFIED} files hash-match source, ${CKPTS} checkpoints, ${VROWS} val rows"
echo "Adapters are safe locally. Only now is it OK to stop or terminate the pod."
