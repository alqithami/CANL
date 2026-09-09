# CÆNL v5.4.3 C4 Confirmatory Package — Completion Report

Prepared from the audited C4 calibration-v2 archive.

- Active plan: `configs/plans/one_c4_confirmatory_v2.yaml`.
- 24 GPU jobs: 4 methods × 6 paired seeds.
- 100M C4 training tokens per run.
- Fixed DCR selected at rank 16/4 and lambda 0.01.
- Corrected MACC-Lite retained.
- Full MACC retained with learned-policy validation.
- Cross-modal significance report is mandatory for strict completion.
- Source snapshot is hashed into each campaign.
- Mounted IBM volume `/mnt/caenl` is required.
