# C4 untouched-shard evaluation v1

This is an evaluation-only follow-up for `caenl-c4-confirmatory-v2`. It does **not** train or alter any model. It evaluates the 24 saved checkpoints on `en/c4-validation.00002-of-00008.json.gz`, which was not used in calibration, controller feedback, configuration selection, or the original confirmatory evaluation.

## Requirements

- Existing project: `~/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory`
- Existing campaign and checkpoints: `/mnt/caenl/active/results/caenl-c4-confirmatory-v2`
- Mounted persistent volume: `/mnt/caenl`
- NVIDIA L40S and the existing tested `.venv`

## Run on the IBM server

```bash
cd ~/caenl-c4-holdout-eval-v1

~/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python \
  run_c4_holdout_eval.py --self-test

bash start_c4_holdout_tmux.sh
```

Monitor:

```bash
bash status_c4_holdout.sh
# or
tmux attach -t caenl-c4-holdout-v1
```

Detach with `Ctrl-b`, then `d`.

The run is resumable: verified per-checkpoint JSON outputs are skipped on a repeated launch.
