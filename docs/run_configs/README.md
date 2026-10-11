# Training run configs (provenance copies)

Verbatim copies of the `training_config.json` / `train_result.json` files each SFT run
persisted next to its checkpoints. They live here because the checkpoint tree itself is
gitignored (`.gitignore:32 data/checkpoints/`), so without these copies the exact
hyperparameters of a run are not recoverable from the repo.

**These are copies, not the originals.** Bytes are unmodified; only the filenames are
prefixed with the run name to disambiguate. Verified byte-identical to their sources at
copy time.

| file | source | sha256 | first tracked |
|---|---|---|---|
| `sft_v3_32g_training_config.json` | `faithfuldocvqa-ckpts` volume, `sft_v3_32g/sft_B_seed42/training_config.json` — fetched 2026-08-10 | `043f27cd708effb74150f422b12577140894ad794014629fb3f8268cd0ac7e52` | 2026-08-10 — **never previously tracked in git** |
| `sft_v3_balanced_nodecoy_training_config.json` | run-3 pod pull, `data/checkpoints/sft_v3_balanced_nodecoy/sft_B_seed42/training_config.json` | `c2436cd805b1629a59fdb6e892d92e91f1e66f065a7536176e6f665061916f98` | 2026-08-10 |
| `sft_v3_balanced_nodecoy_train_result.json` | as above, `train_result.json` | `d55a53ef41a6f5524778d5edd0ee4db5e0789c7fcbbe5889106fd556371db094` | 2026-08-10 |

Notes:

- **`sft_v3_32g`** is the prior SFT series (checkpoints 214/428/642/856/1070/1284;
  `PROVENANCE.md` §, `mcnemar_seed_vs_grpo.md`). Its config was only ever on the ckpts
  volume. It records `num_train_epochs: 3.0`, `save_steps: 214`, `eval_steps: 214`, and
  carries **no** `extra_abstains` key — plain mode-B, 3,423 train records.
- **`sft_v3_balanced_nodecoy`** is run-3 (W&B `lqoekugp`), the balanced 40 %-abstain
  arm. It records `extra_abstains: data/manufactured_train_abstains_nodecoy.jsonl` and
  `save_steps: 324`. Selection of its checkpoint is in
  [`../analysis/run3_checkpoint_selection.md`](../analysis/run3_checkpoint_selection.md).
- The run-3 launch manifest (`train_manifest.json`) is **not** copied here — its
  contents are already pinned in
  [`../analysis/run3_analysis_plan.md`](../analysis/run3_analysis_plan.md) §(b).
