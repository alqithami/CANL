# CÆNL: Objective-Aligned Representation Control for Active Learning

Research code, manuscript, protocols and numerical evidence for the CAENL study.

## Read the paper

- [Manuscript PDF](manuscript/CAENL_revised_fixed.pdf)
- [Editable LaTeX](manuscript/CAENL_revised.tex)
- [Reproducibility and campaign map](docs/REPRODUCIBILITY.md)
- [Evidence provenance and verification limits](docs/PROVENANCE.md)

The strongest ImageNet-100 treatment combines entropy acquisition, aligned DCR and MACC-Lite: **73.360 ± 0.208%** accuracy at 60% annotations across six paired seeds. Its paired gain over the specified fixed coefficient is **1.097 percentage points**. The sequential follow-ups, fixed-coefficient comparison, individual intervals and separate multiplicity families are explicit in the paper. Robust accuracy is near zero; diffusion deteriorates and audio gains are inconclusive. This repository does not support the earlier broad robustness or cross-task improvement claims.

## Reproduce the reported statistics (no GPU)

From the repository root with Python 3.12:

```bash
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python3 -m pip install -r analysis/requirements.txt
python3 analysis/reproduce.py --output reproduced-results
```

Use a new output directory. This verifies input hashes, reconstructs 39 paired contrasts with their separate Holm families, produces means/sample SDs, and checks 108 learning-curve and 216 geometry records. It does not retrain models or regenerate raw-data metrics.

## Source layout

- `src/caenl/`, `configs/`, `tests/`, `scripts/`: historical v5.4.3 pipeline. Missing source files are restored from its archived package; existing tracked configuration files are preserved.
- `experiments/caenl_aligned_imagenet100_v1/`: frozen aligned-classification implementation and its original run instructions.
- `experiments/caenl-*/`: later classification and spectral-confirmation runner/engine sources recovered from the completed result reports.
- `experiments/spectral_source_snapshot/`: hash-verified final project snapshot used with later spectral wrappers; its provenance is explicitly distinguished from a proof of every historical execution.
- `results/2026-09-26/`: public numeric records used by the offline reconstruction.
- `provenance/`: reported source identities, protocol/environment records and reference audit.
- `manuscript/`: revised paper and portable source build.

Historical plans include exploratory and unexecuted settings; they are not all claims of completed experiments. Follow the campaign map before attempting training. Some original runners use server-specific paths and controlled data caches. No additional training is necessary to inspect or reproduce the published statistical tables.

Raw licensed data, dataset-derived reference captions, model weights, credentials and editorial correspondence are excluded. Exact regeneration of all AudioCaps initial states from seeds alone is not established. See the verification limits before interpreting code availability as full experimental replication. Third-party notices remain applicable; no new blanket license is asserted.
