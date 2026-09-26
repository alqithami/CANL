# Reproducibility and experiment map

The main entry point for the paper's numerical evidence is `analysis/reproduce.py`. It reads only the public numeric records in `results/2026-09-26/`; NumPy/SciPy requirements and exact steps are in the root README. No GPU is used.

| Study | Executed formulation | Source location | Numeric record |
|---|---|---|---|
| Initial classification | Nine conditions, aligned normalized scatter, seeds 801–806 | `experiments/caenl_aligned_imagenet100_v1/` | `classification_per_seed.json` |
| Entropy + Lite | Six follow-up branches, same seeds | `experiments/caenl-focused-completion-v1/runner.py` | same classification record |
| Uncertainty herding | Six fixed-feature adapted branches | `experiments/caenl-uherding-v1/runner.py` | same classification record |
| Entropy + fixed | Six matched fixed-coefficient branches | `experiments/caenl-entropy-fixed-v1/runner.py` | same classification record and budget curves |
| GPT-2/C4 | Historical spectral implementation, seeds 301–306; separate unused-shard evaluation | root `src/caenl/language/` and `docs/C4_CONFIRMATORY_V2_RUNBOOK.md` | `c4_per_seed.json` |
| CIFAR-10 diffusion | Spectral confirmation, 24 trained runs plus six pretrained references | `experiments/caenl-diffusion-confirm-v1/` and spectral source snapshot | `diffusion_per_seed.json` |
| AudioCaps captioning | Spectral confirmation, 24 runs on available-recording subset | `experiments/caenl-audiocaps-confirm-v1/` and spectral source snapshot | `audio_per_seed.json` |

All numeric record paths above are under `results/2026-09-26/`. `reported_contrasts.json` preserves the original comparison values, names and endpoint families, rather than silently pooling follow-ups. C4 perplexity is the exponential of NLL; it is not an independent primary outcome. Six diffusion references represent sampling variation of one pretrained model.

Training sources are preserved for inspection and reuse with the original dataset permissions and resolved environments. They are not a claim of a newly tested clean-machine GPU reproduction. The archived runner comments and manifests identify campaign-specific paths and prerequisites. Do not run every historical configuration as though it were a paper result. Original classification instructions are in its experiment directory; C4 instructions remain in the historical runbook; diffusion/audio protocol and environment JSONs are under `provenance/protocols/`.

`analysis/VERIFICATION.json` records the completed public-record reconstruction. Individual 95% paired-t intervals are unadjusted. Paired-t and exact sign-flip tests are Holm-adjusted separately within the declared endpoint families. With six pairs, exact-test resolution materially limits rejection after adjustment.
