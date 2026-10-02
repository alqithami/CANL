# CÆNL: Collapse-Based Active Neural Learning

CAENL provides implementations of active-learning acquisition, representation regularization, and feedback control, together with experiment configurations and numerical analysis tools.

## Choose a workflow

| Goal | Entry point |
|---|---|
| Recompute statistics from released numerical records | [CPU analysis](#reproduce-statistics-on-cpu) |
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

This workflow uses the records in `results/2026-09-26/`. It reproduces exported-record arithmetic, without retraining models or regenerating predictions. The feedback-study design is a separate experiment and is not part of that result snapshot.

## GPU feedback experiment

The feedback study compares MACC-Lite with entropy acquisition alone, a selected fixed regularization strength, and a selected predetermined schedule. Control selection uses training holdouts; confirmation uses disjoint seeds. Selected settings are transferred from ResNet-50 to ResNet-18 on ImageNet-100.

| Stage | Backbone | Seeds | Method runs |
|---|---|---|---:|
| Control selection | ResNet-50 | 41001–41003 | 24 |
| Confirmation | ResNet-50 | 42001–42010 | 40 |
| Architecture transfer | ResNet-18 | 43001–43010 | 40 |

The protocol specifies 104 method runs, 23 shared initializations, and 520 post-initialization phases. Allow several days on one L40S and approximately 160 GiB of free result storage initially. These counts define the experiment; they do not report completed outcomes.

The [deployment guide](docs/RUN_ON_IBM.md) describes dependencies, the dataset contract, preflight checks, execution, checkpoint recovery, and output files. The released runner enforces its L40S/CUDA environment and filesystem layout. Other hardware or dataset layouts require a separately validated adaptation.

## Repository layout

- `experiments/`: campaign-specific implementations, protocols, and validation records.
- `src/caenl/`: the v5.4.3 pipeline and task implementations.
- `configs/`, `scripts/`, `tests/`: plans, execution utilities, and software tests.
- `analysis/`: statistical reconstruction and plotting code.
- `tools/`: saved-evidence export utilities.
- `results/2026-09-26/`: released numerical records.
- `provenance/`: source identities, environment records, and protocol bindings.
- `archives/`: an immutable historical code package.

Configuration files can describe planned or exploratory experiments as well as completed ones. Use the [experiment map](docs/REPRODUCIBILITY.md) to identify the corresponding implementation and available result records.

Licensed datasets, reference-caption text, and model checkpoints must be obtained separately. Third-party notices remain applicable. See [contribution guidance](CONTRIBUTING.md) and [software citation metadata](CITATION.cff).
