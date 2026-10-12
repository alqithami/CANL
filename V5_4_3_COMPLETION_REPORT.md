# CÆNL v5.4.3 C4 package specification

This historical filename identifies the prepared C4 calibration-v2 source package. Preparation of a source package is distinct from completion or scientific validation of an experiment. Use the [experiment map](docs/REPRODUCIBILITY.md) and [completed findings](docs/RESULTS.md) for released empirical evidence.

The package specifies:

- Plan `configs/plans/one_c4_confirmatory_v2.yaml`: four conditions and six paired seeds, for 24 training jobs.
- A nominal 100 million-token training budget per condition and seed; the executed runs used 99,999,744 tokens and 12,207 optimizer steps.
- Fixed spectral DCR at ranks 16/4 and strength 0.01.
- Spectral MACC-Lite and Full MACC implementations, including learned-policy updates in Full MACC.
- Hashed source snapshots and the original `/mnt/caenl` filesystem contract.

Completed C4 metrics are in `results/2026-09-26/c4_per_seed.json`. Additional unused-shard evaluation code is in [tools/c4_holdout_eval](tools/c4_holdout_eval/README.md), and recorded computation costs are reconstructed by [analysis/cost](analysis/cost/README.md). Each tool documents the model, data, checkpoint, and runtime prerequisites needed beyond this repository checkout.
