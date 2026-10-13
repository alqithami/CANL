# CÆNL: Collapse-Based Active Neural Learning with Objective-Aligned Representation Control

CAENL provides implementations of active-learning acquisition, representation regularization, and feedback control, together with experiment configurations and numerical analysis tools.

## Completed findings

The released evidence includes 72 earlier ImageNet-100 classification runs, a later 104-run feedback comparison, 24 GPT-2/C4 training runs with an additional unused-shard evaluation, 24 trained CIFAR-10 diffusion runs plus six pretrained-reference sampling streams, and 24 AudioCaps captioning runs. The [complete findings guide](docs/RESULTS.md) explains the formulations, controls, endpoint families, and limitations of each study.

In the later classification comparison, MACC-Lite improves mean clean accuracy over entropy alone by 1.742 percentage points on ResNet-50 and 1.834 on ResNet-18. Selected fixed regularization has slightly higher mean accuracy than Lite on both backbones; their differences are inconclusive, so these results establish neither adaptive superiority nor equivalence. The earlier clean-geometry improvements do not establish adversarial robustness. The spectral extensions have mixed outcomes: a small fixed-regularization C4 NLL reduction, worse diffusion FID under regularization, and inconclusive AudioCaps differences. Negative and inconclusive outcomes are retained in the released records.

The separate 206-method solidification study and robustness diagnostics are ongoing. Their [implementations, fixed protocols and deployment instructions](docs/PENDING_CAMPAIGNS.md) are released separately. Their partial outputs are excluded from the completed findings and numerical snapshots below.

## Choose a workflow

| Goal | Entry point |
|---|---|
| Read all completed findings and their scope | [Findings guide](docs/RESULTS.md) |
| Recompute the completed feedback-study statistics and plots | [Feedback CPU analysis](analysis/feedback/README.md) |
| Recompute the earlier numerical comparisons | [CPU analysis](#reproduce-statistics-on-cpu) |
| Recompute feedback and C4 computation costs | [Cost CPU analysis](analysis/cost/README.md) |
| Inspect or deploy the additional content-disjoint replication and checkpoint diagnostics | [Pending campaign protocols and deployment](docs/PENDING_CAMPAIGNS.md) |
| Identify execution dependencies and required assets | [Deployment and asset map](docs/DEPLOYMENT.md) |
| Run the ImageNet-100 feedback comparison on an L40S host | [GPU deployment guide](docs/RUN_ON_IBM.md) |
| Inspect methods, seed sets, and statistical comparisons | [Feedback study protocol](experiments/caenl_feedback_study_v1/README.md) |
| Export saved predictions, controller traces, and training records | [Evidence export guide](docs/FEEDBACK_EVIDENCE_EXPORT.md) |
| Rebuild numerical result plots | [Plotting guide](analysis/plots/README.md) |
| Find classification, language, diffusion, or audio experiments | [Experiment map](docs/REPRODUCIBILITY.md) |
| Check data provenance and reproducibility boundaries | [Verification scope](docs/PROVENANCE.md) |

## Obtain the code

```bash
git clone https://github.com/alqithami/CANL.git
cd CANL
git rev-parse HEAD
```

Record the commit used for each experiment. Use a fixed checkout throughout a run and when resuming its checkpoints. The feedback-study engine and self-contained launcher have a reference release at commit `65df044feaf3bb6604416b1011ede5ea3b2a36ed`.

## Reproduce statistics on CPU

With Python 3.12, run from the repository root:

```bash
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python3 -m pip install -r analysis/requirements.txt
python3 analysis/reproduce.py --output reproduced-results
```

The output directory must be new. The script verifies input hashes and reconstructs 39 paired contrasts, their declared Holm families, means and sample standard deviations, 108 learning-curve rows, and 216 geometry rows. Results are written as CSV and JSON files, including `VERIFICATION.json`.

This workflow uses the records in `results/2026-09-26/`. It reproduces exported-record arithmetic, without retraining models or regenerating predictions. The completed feedback study has a separate snapshot at `results/2026-10-02-feedback/`. Its [CPU workflow](analysis/feedback/README.md) reconstructs the six primary contrasts, control selection, and duplicate-exclusion sensitivity, and regenerates the measured plots.

The additional cost snapshot is `results/2026-10-12-cost/`. Its standard-library-only [CPU workflow](analysis/cost/README.md) reconstructs feedback timing and both C4 timing boundaries:

```bash
python3 analysis/cost/reproduce.py --output reproduced-cost
```

## GPU feedback experiment

The feedback study compares MACC-Lite with entropy acquisition alone, a selected fixed regularization strength, and a selected predetermined schedule. Control selection uses training holdouts; confirmation uses disjoint seeds. Selected settings are transferred from ResNet-50 to ResNet-18 on ImageNet-100.

| Stage | Backbone | Seeds | Method runs |
|---|---|---|---:|
| Control selection | ResNet-50 | 41001–41003 | 24 |
| Confirmation | ResNet-50 | 42001–42010 | 40 |
| Architecture transfer | ResNet-18 | 43001–43010 | 40 |

The protocol specifies 104 method runs, 23 shared initializations, and 520 post-initialization phases. Allow several days on one L40S and approximately 160 GiB of free result storage initially. The completed numerical records are released in `results/2026-10-02-feedback/`; see the [analysis guide](analysis/feedback/README.md) for verification boundaries and reconstruction commands.

The [deployment guide](docs/RUN_ON_IBM.md) describes dependencies, the dataset contract, preflight checks, execution, checkpoint recovery, and output files. The released runner enforces its L40S/CUDA environment and filesystem layout. Other hardware or dataset layouts require a separately validated adaptation.

## Repository layout

- `experiments/`: campaign-specific implementations, protocols, and validation records.
- `src/caenl/`: the v5.4.3 pipeline and task implementations.
- `configs/`, `scripts/`, `tests/`: plans, execution utilities, and software tests.
- `analysis/`: statistical reconstruction and plotting code.
- `tools/`: saved-evidence export, C4 frozen-checkpoint evaluation, and monitoring microbenchmark utilities.
- `results/2026-09-26/`: earlier numerical records.
- `results/2026-10-02-feedback/`: completed control-selection, confirmation, and transfer records.
- `results/2026-10-12-cost/`: completed feedback and C4 timing records with independent CPU reconstruction.
- `provenance/`: source identities, environment records, and protocol bindings.
- `archives/`: an immutable historical code package.

Configuration files can describe planned or exploratory experiments as well as completed ones. Use the [experiment map](docs/REPRODUCIBILITY.md) to identify the corresponding implementation and available result records.

Licensed datasets, reference-caption text, and model checkpoints must be obtained separately. Third-party notices remain applicable. See [contribution guidance](CONTRIBUTING.md) and [software citation metadata](CITATION.cff).
