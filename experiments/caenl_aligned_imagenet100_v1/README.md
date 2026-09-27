# Aligned ImageNet-100 active learning

This experiment evaluates nine conditions across six paired seeds: six shared initializations and 54 method runs. [PROTOCOL.md](PROTOCOL.md) specifies the methods, annotation budgets, controls, evaluation, and implementation qualifications.

## Environment

The supplied shell wrappers expect a Linux host with an L40S, a mounted `/mnt/caenl` volume, and Python at `$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python`. The dataset must be present at `/mnt/caenl/persistent/datasets/imagenet100_cmc` and match `protocol.json`. `env.sh` defines the paths and runtime settings. Dependencies and licensed images must be provisioned before launch.

These wrappers retain the recorded environment layout. Adapting paths or hardware creates a distinct deployment that must pass the same tests and record its own source/environment identity. Use the exact source revision recorded by a checkpoint when resuming it.

## Start and monitor

From this experiment directory on the GPU host:

```bash
bash start.sh
bash status.sh
```

`start.sh` creates the detached `caenl-aligned-imagenet100-v1` tmux session. CPU self-tests use temporary synthetic fixtures; CUDA and dataset checks precede scientific training. The status command reads the fixed result directory, process state, and completed-method counter. Monitor epoch/step messages as well as worker status.

## Resume

After identifying a stopped process and resolving its error, run `bash start.sh` from the same checkout and environment. A live process or held lock prevents a duplicate runner. Completed jobs are verified and skipped; unfinished phases resume from the last committed checkpoint with model, optimizer, random-number, and controller state. Source/protocol mismatches stop execution.

## Collect results

Completion produces `ALIGNED IMAGENET100 COMPLETE` and `ALIGNED_AL_EXIT_CODE=0`. The result archive and checksum are written under `/mnt/caenl/persistent/archives/`. Verify them on the GPU host:

```bash
cd /mnt/caenl/persistent/archives
sha256sum -c imagenet100-aligned-results.tar.gz.sha256
```

Transfer the archive using the host's standard file-transfer mechanism if needed. Retain its checksum with the numerical records and provenance. Raw licensed images remain outside the public code distribution.

## Interpretation

This experiment evaluates Full MACC regularization control with a fixed acquisition threshold. It does not evaluate an adaptive query-threshold policy. Successful execution is independent of whether a treatment improves clean or robust accuracy.
