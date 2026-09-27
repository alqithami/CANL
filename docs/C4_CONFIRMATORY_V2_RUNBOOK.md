# C4 confirmatory-v2 execution

This historical pipeline evaluates four methods across six paired seeds, giving 24 sequential GPU jobs followed by strict aggregation and an archive of numerical outputs. [Calibration selection](C4_CALIBRATION_V2_SELECTION.md) records the selected settings.

## Prerequisites

Use a prepared Linux/L40S installation of v5.4.3 with the mounted storage configuration in `$HOME/.caenl-storage-env`, its Python environment, and the required C4 and model caches. The versioned code package is available under `archives/caenl_revision_pipeline_v5.4.3_c4_confirmatory.zip`. Its resolved job/environment records determine an exact reproduction; the following commands assume those prerequisites are already provisioned.

Do not replace a working installation to resume a campaign. Use the same source, configuration, and environment as its checkpoints. For a different environment, record a separate deployment and run its validation before scientific jobs.

## Validate and launch

On the GPU host, from the v5.4.3 installation directory:

```bash
source "$HOME/.caenl-storage-env"
. .venv/bin/activate
python -m caenl.cli validate --plan configs/plans/one_c4_confirmatory_v2.yaml
bash scripts/start_one_dataset_tmux.sh one_c4_confirmatory_v2
```

The result root is `/mnt/caenl/active/results/caenl-c4-confirmatory-v2`. The launcher starts a detached tmux session.

## Monitor and resume

```bash
bash scripts/status_one_dataset.sh one_c4_confirmatory_v2
tmux capture-pane -p -t caenl-one_c4_confirmatory_v2 | tail -n 60
```

Resume from the same installation after confirming that the previous worker has stopped and resolving its recorded error. The launcher checks existing jobs and preserves completed outputs.

## Inspect the export

After successful completion, the archive path is recorded on the GPU host:

```bash
cat /mnt/caenl/persistent/archives/LATEST_caenl-c4-confirmatory-v2.txt
```

Retain the archive and its `.sha256` companion. Verify the checksum in the directory containing both files with `sha256sum -c` on Linux, or `shasum -a 256 -c` on macOS. Use the host's standard file-transfer mechanism if analysis will be performed elsewhere.

The exported configuration, per-seed metrics, validation records, and source identities support result inspection. The separate unused-shard C4 evaluation is described under `tools/c4_holdout_eval/`; it is a distinct evaluation stage.
