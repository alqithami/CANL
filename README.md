# CÆNL / CANL — Collapse-Based Active Neural Learning

Reference implementation and reproducibility repository for the CÆNL manuscript.

CÆNL treats neural-collapse dynamics as a controllable representation signal for active learning and robustness-oriented training. The repository includes core collapse metrics, spectral DCR regularization, MACC-Lite and Full MACC, active-learning baselines, ImageNet/C4/diffusion/audio experiment pipelines, robustness evaluation, and local reporting/statistics tooling.

## Repository layout

- `src/caenl/` — core library and task implementations
- `configs/plans/` — experiment plans, pilots, and confirmatory campaigns
- `configs/protocol/` — protocol-level configuration
- `scripts/` — IBM/RunPod execution and reproducibility helpers
- `tests/` — unit, integration, resume, and publication-gate tests
- `docs/` — protocol, baselines, changelog, and reproducibility notes
- `tools/` — standalone preparation/evaluation utilities used during revision

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements/core.txt
```

Install task-specific dependency groups as needed from `requirements/`.

## Reproducibility

See `docs/REPRODUCIBILITY.md` and `docs/PROTOCOL.md`.

Licensed datasets, access credentials, raw media, and large model checkpoints are not stored in GitHub.

## Citation

See `CITATION.cff`.
