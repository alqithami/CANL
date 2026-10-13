# Reproducibility and experiment map

Three CPU workflows reconstruct the completed numerical records: [earlier comparisons](../README.md#reproduce-statistics-on-cpu), [feedback statistics and plots](../analysis/feedback/README.md), and [feedback/C4 costs](../analysis/cost/README.md). No GPU or licensed dataset is required for these calculations. The [findings guide](RESULTS.md) explains every completed study; the [deployment and asset map](DEPLOYMENT.md) distinguishes statistical reconstruction from training and inference.

## Feedback-study protocol

The completed feedback protocol is `caenl-feedback-study-v1`: 24 control-selection runs on ResNet-50, 40 fresh-seed confirmation runs on ResNet-50, and 40 transfer runs on ResNet-18. Its 23 seed-specific initializations are shared within each seed. All 104 completed method records are released in `results/2026-10-02-feedback/`. The feedback CPU workflow analyzes that snapshot separately from the earlier studies.

- [Launch, monitor, resume, and collect results on IBM](RUN_ON_IBM.md).
- [Frozen method definitions and statistical protocol](../experiments/caenl_feedback_study_v1/README.md).
- [Source, protocol, and resume code](../experiments/caenl_feedback_study_v1/engine/).

Reference implementation: `65df044feaf3bb6604416b1011ede5ea3b2a36ed`. The statistical workflows read only their specified numerical snapshots; they do not analyze unrelated campaign directories.

## Completed campaigns

| Study | Executed formulation | Source location | Numeric record |
|---|---|---|---|
| Initial classification | Nine conditions, aligned normalized scatter, seeds 801–806 | `experiments/caenl_aligned_imagenet100_v1/` | `classification_per_seed.json` |
| Entropy + Lite | Six follow-up branches, same seeds | `experiments/caenl-focused-completion-v1/runner.py` | same classification record |
| Uncertainty herding | Six fixed-feature adapted branches | `experiments/caenl-uherding-v1/runner.py` | same classification record |
| Entropy + fixed | Six matched fixed-coefficient branches | `experiments/caenl-entropy-fixed-v1/runner.py` | same classification record and budget curves |
| Feedback control selection, confirmation, and architecture transfer | Aligned classification DCR; 24 tuning and 80 fresh-seed method runs | `experiments/caenl_feedback_study_v1/` | `results/2026-10-02-feedback/` |
| GPT-2/C4 | Historical spectral implementation, seeds 301–306; separate unused-shard evaluation | root `src/caenl/language/` and `docs/C4_CONFIRMATORY_V2_RUNBOOK.md` | `c4_per_seed.json` |
| CIFAR-10 diffusion | Spectral confirmation, 24 trained runs plus six pretrained references | `experiments/caenl-diffusion-confirm-v1/` and spectral source snapshot | `diffusion_per_seed.json` |
| AudioCaps captioning | Spectral confirmation, 24 runs on available-recording subset | `experiments/caenl-audiocaps-confirm-v1/` and spectral source snapshot | `audio_per_seed.json` |

Earlier per-seed filenames in this table are under `results/2026-09-26/`; the later feedback row specifies its separate directory. `reported_contrasts.json` preserves the original comparison values, names and endpoint families, rather than silently pooling follow-ups. C4 perplexity is the exponential of NLL; it is not an independent primary outcome. Six diffusion references represent sampling variation of one pretrained model.

The completed cost records in `results/2026-10-12-cost/` add feedback method/initialization timings, the paired C4 monitoring microbenchmark, and all 24 successful C4 training-loop/task-job timers. The [cost workflow](../analysis/cost/README.md) reconstructs their arithmetic and keeps measurement boundaries separate. These are additional records of completed experiments, rather than new training runs.

The C4 [unused-shard evaluator](../tools/c4_holdout_eval/README.md) and [monitoring microbenchmark](../tools/c4_overhead_benchmark/README.md) include their recovered executable sources and source provenance. Frozen-checkpoint evaluation requires the corresponding completed training artifacts.

Training sources are preserved for inspection and reuse with the original dataset permissions and resolved environments. They are not a claim of a newly tested clean-machine GPU reproduction. The archived runner comments and manifests identify campaign-specific paths and prerequisites. Do not run every historical configuration as though it were a completed experiment. Original classification instructions are in its experiment directory; C4 instructions remain in the historical runbook; diffusion/audio protocol and environment JSONs are under `provenance/protocols/`.

`analysis/VERIFICATION.json` records the completed public-record reconstruction. Individual 95% paired-t intervals are unadjusted. Paired-t and exact sign-flip tests are Holm-adjusted separately within the declared endpoint families. With six pairs, exact-test resolution materially limits rejection after adjustment.

The later feedback study uses ten pairs for each confirmation/transfer contrast and one declared six-contrast family. Its sensitivity analysis excludes the same five duplicate-content validation images across all treatments; it does not alter the primary 5,000-image endpoint or undo training exposure. Its separate [analysis guide](../analysis/feedback/README.md) documents those boundaries.

The 206-method solidification study and separate robustness diagnostics are ongoing and have no completed findings in these snapshots. Their [source, fixed protocols, deployment and source-identity boundaries](PENDING_CAMPAIGNS.md) are available. Their partial outputs do not contribute to the statistical or cost reconstruction.

## Local result plots

[Plotting code and instructions](../analysis/plots/README.md) rebuild the existing numerical visualizations from their recorded data. The generated graphics remain local. Both the statistical and plotting workflows use the released numerical inputs.
