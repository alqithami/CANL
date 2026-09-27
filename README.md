# CÆNL: Collapse-Based Active Neural Learning

Research software, experiment configurations, run instructions, and numerical records for CAENL. The repository supports running experiments and reproducing their analysis. Manuscripts, journal templates, editorial correspondence, and submission packages are maintained separately.

## Start here

| Task | Instructions |
|---|---|
| Check the ongoing IBM feedback study | [IBM run guide](docs/RUN_ON_IBM.md#check-an-existing-run) |
| Start or resume the frozen GPU campaign | [IBM run guide](docs/RUN_ON_IBM.md) |
| Inspect the experimental design and analysis rules | [Feedback study protocol](experiments/caenl_feedback_study_v1/README.md) |
| Recompute statistics from completed numeric exports | [CPU analysis below](#reproduce-existing-results-on-cpu) |
| Rebuild result plots locally | [Plotting instructions](analysis/plots/README.md) |
| Find earlier experiment implementations | [Experiment map](docs/REPRODUCIBILITY.md) |
| Understand provenance and replication limits | [Verification scope](docs/PROVENANCE.md) |

## Current GPU campaign

`caenl-feedback-study-v1` compares unchanged MACC-Lite with independently selected fixed-strength and predetermined-schedule controls, followed by ResNet-18 transfer on the same ImageNet-100 dataset.

| Stage | Backbone | Seeds | Method runs |
|---|---|---|---:|
| Control selection | ResNet-50 | 41001–41003 | 24 |
| Confirmation | ResNet-50 | 42001–42010 | 40 |
| Architecture transfer | ResNet-18 | 43001–43010 | 40 |

The full campaign has 104 method runs, 23 shared initializations, and 520 post-initialization phases. It was reported running on IBM; a completed result report has not yet been added. The table describes the planned workload, not completed results.

The active release is pinned to commit `65df044feaf3bb6604416b1011ede5ea3b2a36ed`. The launcher and engine remain byte-identical to that release. Repository documentation updates do not update the running server process. Use the status command in the run guide; no restart, reinstall, or repository pull is needed for an existing run.

## Reproduce existing results on CPU

Use a separate local environment with Python 3.12. From the repository root:

```bash
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python3 -m pip install -r analysis/requirements.txt
python3 analysis/reproduce.py --output reproduced-results
```

Use a new output directory. This verifies input hashes and reconstructs 39 paired contrasts, the declared separate Holm families, means and sample standard deviations, 108 learning-curve rows, and 216 geometry rows. It uses the completed records in `results/2026-09-26/`; it does not incorporate the ongoing feedback study, retrain models, or regenerate raw predictions.

## Repository layout

- `experiments/caenl_feedback_study_v1/`: self-contained IBM launcher, frozen engine and protocol, software validation records.
- `experiments/caenl_aligned_imagenet100_v1/`: earlier aligned-classification implementation and run instructions.
- `experiments/caenl-*/`: completed follow-up campaign runners and available engines.
- `experiments/spectral_source_snapshot/`: recorded project source used with later spectral wrappers.
- `src/caenl/`, `configs/`, `scripts/`, `tests/`: historical pipeline, configurations, utilities, and tests.
- `analysis/`: statistical reconstruction and plotting code.
- `results/2026-09-26/`: numeric exports from completed earlier campaigns.
- `provenance/`, `archives/`: source identities, environment/protocol records, and an archived code package.

Historical plans include exploratory and unexecuted settings. Consult the experiment map before launching them. The IBM feedback launcher uses an existing prepared dataset and environment; it is not a general fresh-machine installer.

Raw licensed datasets, reference-caption text, model weights, and credentials are not distributed here. Exact reconstruction of all historical AudioCaps initial states from seeds alone is not established. Third-party notices remain applicable; this update grants no new license. See [contribution and repository scope](CONTRIBUTING.md).
