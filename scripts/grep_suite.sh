#!/bin/bash
# Pre-commit grep suite — scan staged additions for secrets, credential shapes,
# absolute-path leaks and PII before they enter a public commit.
#
#   bash scripts/grep_suite.sh              # scan staged additions (normal use)
#   bash scripts/grep_suite.sh --scan FILE  # scan an arbitrary file (self-test)
#
# Exit 0 = clean, 1 = at least one pattern hit (review, do not auto-dismiss),
# 2 = usage/environment error.
#
# ── Two-tier by design ────────────────────────────────────────────────────────
# This committed file carries only GENERIC shapes — no real key, host, account or
# path ever appears here, because publishing the identifiers the suite hunts would
# defeat the suite. Instance-specific literals (a particular pod IP, pod id,
# personal email, a real home-directory path) belong in an untracked file:
#
#   .grep_suite_local     one extended-regex per line; blank lines and # ignored
#
# It is gitignored. Without it the suite still runs, on generic shapes only, and
# says so — an unqualified "CLEAN" never silently means "half the patterns were
# missing".
#
# Patterns are word-bounded on purpose. An earlier draft matched bare `secret`
# and fired 20 times on the word "Secretary" in OCR'd document answers. A suite
# that cries wolf on ordinary content gets ignored, which is worse than no suite.
set -uo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not a git repo" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2

SCAN_FILE=""
if [ "${1:-}" = "--scan" ]; then
  SCAN_FILE="${2:-}"
  [ -n "$SCAN_FILE" ] && [ -f "$SCAN_FILE" ] || { echo "--scan needs an existing file" >&2; exit 2; }
fi

TMP=$(mktemp) || exit 2
trap 'rm -f "$TMP"' EXIT
if [ -n "$SCAN_FILE" ]; then
  cat "$SCAN_FILE" > "$TMP"
  echo "scanning file: $SCAN_FILE ($(wc -l < "$TMP" | tr -d ' ') lines)"
else
  # Strip the leading '+' diff marker before matching: it is not content, and leaving
  # it on made every Python decorator look like an email address -- the marker itself
  # satisfied the local-part class and the dotted decorator path satisfied the domain.
  # Genuine addresses are unaffected; they still match once the marker is removed.
  # (Deliberately phrased without literal example tokens, so this comment does not
  # trip the very pattern it documents.)
  git diff --cached -U0 | grep '^+' | grep -v '^+++' | sed 's/^+//' > "$TMP"
  echo "staged added lines: $(wc -l < "$TMP" | tr -d ' ')"
fi

# ── Generic patterns (safe to publish) ────────────────────────────────────────
PATTERNS=(
  'WANDB_API_KEY'
  'JUPYTER_PASSWORD'
  '\bapi[_-]?key\b'
  '\bpass(word|wd)\b'
  '\bsecret\b'
  '\btoken\s*[=:]\s*["'"'"'][A-Za-z0-9_-]{8}'
  'Bearer [A-Za-z0-9]'
  'sk-[A-Za-z0-9]{16}'
  'hf_[A-Za-z0-9]{16}'
  'AKIA[0-9A-Z]{16}'
  'ssh-(rsa|ed25519) AAAA'
  'BEGIN [A-Z ]*PRIVATE KEY'
  '/(Users|home)/[a-zA-Z][a-zA-Z0-9._-]*'       # home-dir path leak (shape only)
  '/private/tmp'                                 # scratchpad path leak
  '/workspace/'                                  # remote-box absolute path leak
  '\b([0-9]{1,3}\.){3}[0-9]{1,3}\b'              # IPv4 shape
  '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # email shape
)
n_generic=${#PATTERNS[@]}

# ── Instance-specific patterns, if present ────────────────────────────────────
LOCAL_FILE="$REPO_ROOT/.grep_suite_local"
n_local=0
if [ -f "$LOCAL_FILE" ]; then
  while IFS= read -r line; do
    case "$line" in ''|'#'*) continue ;; esac
    PATTERNS+=("$line"); n_local=$((n_local+1))
  done < "$LOCAL_FILE"
  echo "patterns: $n_generic generic + $n_local local (.grep_suite_local)"
else
  echo "patterns: $n_generic generic; .grep_suite_local ABSENT — instance-specific"
  echo "          identifiers (hosts, accounts, real home paths) are NOT being checked"
fi

hits=0
for p in "${PATTERNS[@]}"; do
  if n=$(grep -icE -- "$p" "$TMP") && [ "${n:-0}" -gt 0 ]; then
    echo "  !! HIT ($n)  /$p/"
    grep -inE -- "$p" "$TMP" | head -3 | cut -c1-170
    hits=$((hits+1))
  fi
done

if [ "$hits" -eq 0 ]; then
  echo "  CLEAN — no pattern matched"
  exit 0
fi
echo "  $hits pattern(s) hit — review each before committing"
exit 1
