# C4 confirmatory-v2 on the IBM L40S

This package starts the frozen C4 confirmatory campaign selected from calibration-v2.
It runs 24 sequential GPU jobs: four methods across six paired seeds, followed by
strict aggregation and a verified lean archive.

## Mac: upload

```bash
ZIP="$(ls -t "$HOME"/Downloads/caenl_revision_pipeline_v5.4.3_c4_confirmatory*.zip | head -n 1)"
CHECKSUM="${ZIP}.sha256"
scp "$ZIP" salqithami@161.156.166.216:~/caenl_revision_pipeline_v5.4.3_c4_confirmatory.zip
scp "$CHECKSUM" salqithami@161.156.166.216:~/caenl_revision_pipeline_v5.4.3_c4_confirmatory.zip.sha256
```

## IBM: verify and install

```bash
ssh salqithami@161.156.166.216
cd "$HOME"
sha256sum -c caenl_revision_pipeline_v5.4.3_c4_confirmatory.zip.sha256
rm -rf "$HOME/caenl-v5.4.3"
mkdir -p "$HOME/caenl-v5.4.3"
unzip -q "$HOME/caenl_revision_pipeline_v5.4.3_c4_confirmatory.zip" -d "$HOME/caenl-v5.4.3"
cd "$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory"
source "$HOME/.caenl-storage-env"
bash scripts/upgrade_existing_ibm_v542.sh 2>&1 | tee "$HOME/caenl-v5.4.3-upgrade.log"
```

Required ending: `V5.4.3 UPGRADE PASS`.

## IBM: launch

```bash
cd "$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory"
source "$HOME/.caenl-storage-env"
bash scripts/start_one_dataset_tmux.sh one_c4_confirmatory_v2
```

Confirm the root is `/mnt/caenl/active/results/caenl-c4-confirmatory-v2`:

```bash
sleep 5
bash scripts/status_one_dataset.sh one_c4_confirmatory_v2
tmux capture-pane -p -t caenl-one_c4_confirmatory_v2 | tail -60
```

The run should take roughly 16--20 hours on one L40S. It is resumable: rerun the
same launcher only if the tmux session is absent and no CÆNL process is active.

## IBM: status

```bash
bash scripts/status_one_dataset.sh one_c4_confirmatory_v2
```

## Mac: download after completion

```bash
REMOTE_ARCHIVE="$(ssh salqithami@161.156.166.216 'cat /mnt/caenl/persistent/archives/LATEST_caenl-c4-confirmatory-v2.txt')"
scp "salqithami@161.156.166.216:${REMOTE_ARCHIVE}" "$HOME/Downloads/"
scp "salqithami@161.156.166.216:${REMOTE_ARCHIVE}.sha256" "$HOME/Downloads/"
cd "$HOME/Downloads"
NAME="$(basename "$REMOTE_ARCHIVE")"
shasum -a 256 -c "${NAME}.sha256"
```
