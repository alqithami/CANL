# Historical C4 calibration-v2 protocol

This record specifies the v5.4.2 calibration design and its execution interface. Use the matching source, resolved configuration, model/data cache, and environment when reproducing its outputs. The archived v5.4.3 package contains subsequent code; it is not a substitute for the original v5.4.2 runtime identity.

## Implementation requirements

- `eta_lambda` controls the MACC-Lite step size.
- Language targets can differ by monitored layer (`h6`, `h12`).
- Full MACC reward components and policy updates are recorded separately.
- Calibration summaries suppress confirmatory significance claims.
- Storage and result roots are bound to the mounted `/mnt/caenl` volume.
- Plan fingerprints and source digests identify the executed configuration.

## Calibration design

Two paired seeds (`201`, `202`) and six conditions:

1. Baseline.
2. Fixed DCR, ranks h6/h12 = 16/4, lambda = 0.005.
3. Fixed DCR, ranks h6/h12 = 16/4, lambda = 0.01.
4. Fixed DCR, ranks h6/h12 = 32/8, lambda = 0.005.
5. Corrected MACC-Lite, ranks 16/4, eta = 0.002, lambda in [0, 0.08].
6. Full MACC, ranks 16/4, smaller lambda actions and reduced geometry-reward weight.

Each run uses 10M requested C4 tokens. This is calibration only, not confirmatory evidence.

## Execute from a prepared v5.4.2 installation

On the GPU host, activate the matching environment and validate the plan:

```bash
source "$HOME/.caenl-storage-env"
. .venv/bin/activate
source scripts/ibm_one_dataset_env.sh
python -m caenl.cli validate --plan configs/plans/one_c4_calibration_v2.yaml
```

Start and monitor the calibration:

```bash
bash scripts/start_one_dataset_tmux.sh one_c4_calibration_v2
bash scripts/status_one_dataset.sh one_c4_calibration_v2
```

The session name is `caenl-one_c4_calibration_v2`. Resume with the same source/environment after a stopped worker's error has been resolved. Preserve existing outputs and their integrity records.

## Output and selection

The archive location is recorded at `/mnt/caenl/persistent/archives/LATEST_caenl-c4-calibration-v2.txt`. Retain the archive with its checksum and verify it before analysis. Calibration scores select settings; they are not independent confirmatory outcomes. Freeze the selected settings before evaluating new confirmatory seeds. The precalibration template must not be treated as the selected final configuration.
