# Reproducibility

This repository contains the CÆNL/CANL experimental implementation used for the journal revision.

## Included

- Neural-collapse and spectral metrics
- Dynamic Collapse Regularization (DCR)
- MACC-Lite and Full MACC controllers
- Active-learning acquisition methods and matched baselines
- ImageNet/ImageNet-100, C4, diffusion, and audio experiment pipelines
- AutoAttack robustness evaluation
- Statistics, reporting, resume, provenance, and campaign-integrity tooling
- Standalone preparation/evaluation utilities used during the revision

## Excluded

Licensed datasets, access tokens, browser cookies, SSH credentials, raw media, large checkpoints, and private cloud configuration are not committed.

## Experiment roles

Pilot/calibration runs are configuration-selection evidence and must not be mixed with confirmatory statistics. Resume/fault-injection runs are engineering checks and are excluded from scientific aggregation.

See `docs/PROTOCOL.md` and `configs/plans/` for experiment definitions.
