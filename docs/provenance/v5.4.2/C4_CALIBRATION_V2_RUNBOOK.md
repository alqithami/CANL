# CÆNL C4 Calibration v2 — IBM L40S Runbook

This package corrects the defects identified by the C4 pilot-v1 audit before any 100M-token confirmatory run.

## Corrections in v5.4.2

- `eta_lambda` now controls MACC-Lite; it no longer silently falls back to the inherited `controller_step_size`.
- Per-layer language target ranks are supported (`h6` and `h12` can have different targets).
- Full MACC reward components are logged separately.
- Full MACC policy updates nested under `history[i].update` are recognised by the publication validator.
- Pilot tables no longer claim significance tests when inference is suppressed.
- Pilot summaries use “completed analysis jobs,” not “evidence-eligible jobs.”
- `/mnt/caenl/active` forcibly determines all derived data/cache/results/work roots, preventing stale `/dev/shm` variables.
- The launcher refuses a pilot/confirmatory run in `/dev/shm` when `/mnt/caenl` is mounted and writable.
- Plan fingerprints and full SHA-256 digests are stored under accurate names.
- Tmux session names no longer gain an accidental trailing hyphen.

## Calibration design

Two paired seeds (`201`, `202`) and six conditions:

1. Baseline.
2. Fixed DCR, ranks h6/h12 = 16/4, lambda = 0.005.
3. Fixed DCR, ranks h6/h12 = 16/4, lambda = 0.01.
4. Fixed DCR, ranks h6/h12 = 32/8, lambda = 0.005.
5. Corrected MACC-Lite, ranks 16/4, eta = 0.002, lambda in [0, 0.08].
6. Full MACC, ranks 16/4, smaller lambda actions and reduced geometry-reward weight.

Each run uses 10M requested C4 tokens. This is calibration only, not confirmatory evidence.

## IBM commands

### Mac terminal

```bash
scp ~/Downloads/caenl_revision_pipeline_v5.4.2_c4_calibration.zip \
  salqithami@161.156.166.216:~/
```

### IBM terminal

```bash
ssh salqithami@161.156.166.216

cd "$HOME"
sha256sum caenl_revision_pipeline_v5.4.2_c4_calibration.zip

rm -rf "$HOME/caenl-v5.4.2"
mkdir -p "$HOME/caenl-v5.4.2"
unzip -q "$HOME/caenl_revision_pipeline_v5.4.2_c4_calibration.zip" \
  -d "$HOME/caenl-v5.4.2"

cd "$HOME/caenl-v5.4.2/caenl_revision_pipeline_v5.4.2_c4_calibration"
source "$HOME/.caenl-storage-env"
bash scripts/upgrade_existing_ibm_v541.sh
```

Verify that the plan uses the mounted volume:

```bash
source scripts/ibm_one_dataset_env.sh
printf '%s\n' "$CAENL_DATA_ROOT" "$CAENL_RESULTS_ROOT" "$CAENL_PERSISTENT_ROOT"
python -m caenl.cli validate --plan configs/plans/one_c4_calibration_v2.yaml
```

Required roots begin with `/mnt/caenl/`.

Start the calibration:

```bash
bash scripts/start_one_dataset_tmux.sh one_c4_calibration_v2
```

Monitor:

```bash
bash scripts/status_one_dataset.sh one_c4_calibration_v2
```

Attach:

```bash
tmux attach -t caenl-one_c4_calibration_v2
```

Detach with `Ctrl-b`, then `d`.

## Retrieve the archive after completion

Run from the Mac, not the IBM prompt:

```bash
REMOTE_ARCHIVE="$(
  ssh salqithami@161.156.166.216 \
  'cat /mnt/caenl/persistent/archives/LATEST_caenl-c4-calibration-v2.txt'
)"

scp "salqithami@161.156.166.216:${REMOTE_ARCHIVE}" ~/Downloads/
scp "salqithami@161.156.166.216:${REMOTE_ARCHIVE}.sha256" ~/Downloads/

cd ~/Downloads
NAME="$(basename "$REMOTE_ARCHIVE")"
shasum -a 256 -c "${NAME}.sha256"
```

Upload the archive and checksum for audit. Do not start `configs/templates/one_c4_confirmatory_v2.yaml` until the calibration is reviewed and its selected settings are frozen.
