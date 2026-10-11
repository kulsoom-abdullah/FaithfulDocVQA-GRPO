# Run-4 provisioning runbook — §4e, reconstructed

**Provenance, stated first.** The original runbook (§4c–4f) is **lost** — it is not on
disk, not in `git log --all` on any branch, and not stashed. This file reconstructs
**§4e only**, on 2026-08-17, from two sources: artifacts that survived the run
(`MARKER_CORRECTION.md`, `run4_analysis_plan.md`'s 2026-08-15 amendment) and the
operator's direct recollection of failures they witnessed. **§4c, §4d and §4f are not
reconstructed and should not be assumed to exist.**

Each pattern below is marked **[confirmed here]** where it was reproduced on this
machine while writing this file, or **[witnessed]** where it rests on the operator's
account of the run.

The theme is one failure mode with six faces: **a command reported success without
having done the work.** Exit codes are a hint; hashes are evidence.

---

## §4e.1 — `PIPESTATUS` lost across a shell layer, or overwritten

**[confirmed here]** Run 4a's launch emitted `JOB4A_FAILED` for a run that completed
150/150 steps and wrote every checkpoint. The autopilot behaved correctly on the signal
it was given: it refused to pull, refused to stop the pod, and the pod idled until the
marker was hand-corrected.

**Correction to `MARKER_CORRECTION.md`.** That note diagnoses the cause as *"`PIPESTATUS`
does not survive into a `$( )` command substitution."* **That mechanism does not
reproduce.** In a single bash shell, `$( )` reads `PIPESTATUS` correctly:

```
true | tee log; echo "JOB_$([ ${PIPESTATUS[0]} -eq 0 ] && echo COMPLETE || echo FAILED)"
  -> JOB_COMPLETE          # correct
```

The note's *conclusion* stands — the marker was wrong and the run had completed. Its
stated cause is imprecise. Two mechanisms do reproduce:

**(a) An extra shell layer empties the array** — the 4a symptom, a false FAILED. Any
`ssh host '…'`, `tmux send-keys`, or nested `bash -c` between the pipeline and the
reference:

```
false | tee log; echo "PIPESTATUS here: ${PIPESTATUS[0]}"   -> 1
                 bash -c 'echo "one layer down: [${PIPESTATUS[0]}]"'  -> []
```

Empty array → `[ -eq 0 ]` errors → `||` branch → `FAILED`.

**(b) Any intervening command overwrites it** — a false COMPLETE, and the more dangerous
direction:

```
false | tee log; mkdir -p /tmp/x; echo "JOB_$([ ${PIPESTATUS[0]} -eq 0 ] && … )"
  -> JOB_COMPLETE          # WRONG: mkdir's status, not the pipeline's
```

**Correct form** — capture on the very next line, before anything else runs, and use the
captured scalar thereafter:

```bash
python grpo/train_grpo.py ... 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 0 ] && echo "JOB_COMPLETE" || echo "JOB_FAILED (rc=$RC)"
```

Or `set -o pipefail` and read `$?` — the form `modal_grpo.py:169` already uses.
`scripts/unattended_run.sh:36` is the in-tree reference implementation.

## §4e.2 — `timeout` masked by a pipe, and `timeout` absent

**[confirmed here]** The first run-4 smoke was truncated by a `timeout` whose non-zero
status was masked by a pipe; it reported success having written no checkpoint at all
(`run4_analysis_plan.md`, 2026-08-15 amendment).

Compounding it: **`timeout` does not exist on macOS.** A local wrapper that pipes it
loses both the truncation status *and* the fact that the binary was never there:

```
timeout 180 python x.py | tail   ->  "command not found" (127), swallowed by the pipe
```

GNU coreutils installs it as `gtimeout`. Detect with `command -v timeout`, not a hopeful
`echo`. Same rule as §4e.1: capture `PIPESTATUS`/use `pipefail`, and assert the artifact
the command was supposed to produce.

## §4e.3 — `${VAR^^}` is bash-4 syntax; this Mac runs bash 3.2

**[confirmed here]** `/bin/bash` here is **3.2.57(1)-release**, and no newer bash is
installed:

```
ARM=4b; echo "${ARM^^}"   ->  bad substitution
```

Apple ships 3.2 for licensing reasons and this will not change. Case conversion,
`declare -A` associative arrays, and `mapfile`/`readarray` are all bash-4+. Portable
form: `$(printf '%s' "$ARM" | tr '[:lower:]' '[:upper:]')`.

Note the mirror-image trap in the other direction: the interactive shell here is **zsh**,
which does **not** word-split unquoted variable expansions. `git add -f $FILES` passes
one long filename in zsh where bash passes many; use an array (`"${FILES[@]}"`) so the
same line behaves identically in both. (Hit and corrected while force-adding the run-4a
evidence for this very document.)

## §4e.4 — never suppress stderr on a verification step

**[witnessed]** A `get` loop sent stderr to `/dev/null`. A Modal error therefore surfaced
as the symptom **"empty/absent"** rather than its cause, and the investigation chased the
wrong object — the volume's contents — instead of reading the error that was right there.

```
ls /nonexistent 2>/dev/null || echo "empty/absent"    # symptom only
ls /nonexistent 2>&1                                   # cause: No such file or directory
```

**Rule:** a step whose *purpose* is to verify may never discard stderr. Suppressing
stderr is acceptable only where the failure is expected and separately handled; on a
verification path it converts a diagnosable cause into an ambiguous symptom.

## §4e.5 — stage −1: preflight the real paths before anything else

**[witnessed]** `run4_pull_verify.sh` encoded assumed paths — `$REMOTE/data/checkpoints/`
and `$REMOTE/job4a.log` — while the launch actually used `--output-dir /workspace/grpo_runs`
and `tee /workspace/job4a.log`. Result: **four consecutive failures, one path per attempt**,
each fixed in isolation only to reveal the next.

A single `ls` against every real path, run before any transfer, catches all four at once:

```bash
ssh $SSH_OPTS "$REMOTE" 'set -x; ls -la /workspace/grpo_runs/ /workspace/job4a.log' || exit 1
```

**Rule:** paths are an interface between the launch script and the pull script. Assert
them from the *live* remote in one preflight pass; never infer them from what the pull
script wishes were true. Serial single-path debugging on a paid pod is the expensive way
to discover a set of assumptions.

## §4e.6 — a Modal batch `put` can exit success having written garbage

**[witnessed]** A Modal batch `put` **exited clean** while writing, of the files it
claimed to transfer, **31 zero-byte, 2 truncated, and 4 with correct size and wrong
content**. Count checks passed. Size checks passed. **Only hashes caught it** — the 4
correct-size-wrong-content files are invisible to any check weaker than a digest.

**Distrust the batch put, not the volume.** Each corrupted block cleared once its own
path was re-put individually; the volume was never at fault.

**Rule:** after any bulk transfer, verify by **sha256 of every file against a manifest
computed at the source** — never `ls`, never a file count, never a size comparison.
`ARTIFACTS.sha256` + `shasum -a 256 -c` is the in-tree form of this check. This is the
single most important entry in §4e: it is the only pattern here that a careful operator
reading exit codes and file listings would still miss entirely.

## §4e.7 — the newest writes are the least durable across a pod migration

**[confirmed here]** Run 4b's pod could not restart ("not enough free GPUs on the host
machine"). A replacement pod came up with the same `/workspace`, and almost everything
survived: `hf-cache` (8.3 GB, written Aug 16), `grpo_runs` (3.0 GB, Aug 16), the repo,
`data/images`, the seed adapter. **One thing did not: `venv-grpo2` came back as an empty
directory** — 11 GB gone, mtime restamped to the new pod's start.

That venv was the *newest* thing on the disk: built 04:36–04:55, verified, used for 4.5
hours of training, then lost across the stop and migration. Everything older persisted.

**The lesson is not "volumes are unreliable."** Older data on this same volume has
survived multiple stop/start cycles, and run-4a's checkpoints came off it intact. The
lesson is narrower and more useful:

> **Anything written during the current session is UNBACKED until it has been pulled and
> hash-verified.** Age is the best available proxy for durability across a migration:
> the most recent writes are the ones a migration is most likely to lose.

Consequences, all cheap:

- **Pull incrementally on long runs.** Do not wait for the completion marker. Run
  `run4_pull_verify.sh … --live` partway through (run 4b: ~step 75 and ~step 120), which
  bounds the loss to the work since the last verified pull.
- **`--live` distinguishes churn from corruption.** Five telemetry files
  (`reward_log`, `reward_members`, `step_times`, `val_curve`, `probe_completions`) are
  appended continuously, so a hash can legitimately move between the source manifest and
  the pull. On a mismatch, `--live` re-hashes AT THE SOURCE: if the source hash also
  moved, the file churned; if the source hash is unchanged, the bytes are genuinely
  corrupt. Without that distinction a mid-run pull produces five false alarms and trains
  the operator to ignore the check — worse than not running it.
- **Rebuild recipes belong on the volume AND in the repo.** `rebuild_env.sh` survived on
  the pod and was hash-matched to the local copy (`a399004a…`) before reuse, so the venv
  was reconstructible in one command. A recipe that exists only in a session transcript
  is lost with the session.
- **Assert the environment, never assume it.** The empty venv was caught by the stage-2b
  pin check reporting `PYTHON=/usr/bin/python` with `torch 2.4.1+cu124` — not by anything
  looking at the venv directly. Had it not been asserted, 4b would have launched on a
  different stack than 4a and the arms would have differed by far more than one reward
  constant, invisibly.

---

## Checklist form

Before 4b launches, and before any pull:

1. `command -v rsync tmux` on the pod. **Environment-specific:** the `runpod/pytorch:*`
   image this run used ships neither, and installs via `apt-get`. On another image the
   install command differs; the `command -v` verification is the part that generalises.
2. Preflight **every** remote path with one `ls`; do not proceed on any miss (§4e.5).
3. Launch detached, teeing to a log, capturing `RC=${PIPESTATUS[0]}` on the next line,
   with an explicit completion marker (§4e.1).
4. No `2>/dev/null` anywhere on a verification path (§4e.4).
5. Arrays for file lists; no `${VAR^^}`; `gtimeout` not `timeout` locally (§4e.2, §4e.3).
6. After transfer: **sha256 every file** against a source-side manifest (§4e.6).
7. Assert the artifacts exist and are non-empty — checkpoint count, `val_curve.jsonl`
   row count, zero zero-byte files — before trusting any completion marker.
8. On any run longer than an hour, pull incrementally with `--live` (§4e.7). Treat
   everything written this session as unbacked until pulled and hash-verified.
9. Assert the *environment* before launching, not just the data: venv pins against the
   reference run's, and the RESOLVED config (not the argument string) against the
   locally specified one (§4e.7, and stages 2b/2c of `launch_run4b.sh`).
