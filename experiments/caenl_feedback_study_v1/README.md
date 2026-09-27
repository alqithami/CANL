# CAENL feedback study v1

This frozen campaign tests whether the unchanged MACC-Lite controller improves clean classification over separately tuned fixed-strength and predetermined-schedule controls. It then tests transfer of those settings from ResNet-50 to ResNet-18 on the existing ImageNet-100 dataset.

The experiment uses disjoint control-selection and evaluation seeds. ImageNet-100 and the Lite settings have prior development history, so the design does not constitute an untouched benchmark or equal lifetime development effort.

## Installation and execution

Use the [L40S deployment guide](../../docs/RUN_ON_IBM.md) for Python dependencies, required data manifests, preflight validation, execution, monitoring, and checkpoint recovery. The engine can be called from an explicitly selected virtual environment; the optional self-contained launcher also supports the original directory layout.

The protocol requires a mounted `/mnt/caenl` volume, the bound ImageNet-100 image collection, and PyTorch 2.6 / torchvision 0.21 with CUDA 12.4 on an L40S. Allow several days and approximately 160 GiB free initially. No dataset or dependency installation occurs inside the runner.

Outputs are written to `/mnt/caenl/active/results/caenl-feedback-study-v1`, including per-seed artifacts, `SELECTION.json`, `results.json`, a Markdown numerical report, and integrity/completion records. The run lock and source/configuration bindings guard checkpoint reuse.

## Frozen experimental matrix

| Stage | Backbone | Seeds | Methods per seed | Method runs |
|---|---|---|---:|---:|
| Control selection | ResNet-50 | 41001–41003 | 4 fixed + 4 preset schedules | 24 |
| Confirmation | ResNet-50 | 42001–42010 | entropy, selected fixed, selected schedule, Lite | 40 |
| Architecture transfer | ResNet-18 | 43001–43010 | the same four frozen methods | 40 |

Each seed has its own from-scratch initialization, shared across that seed's methods. There are **23 shared initializations, 104 complete method runs, 520 post-initialization training phases, and 4,155,880 optimizer steps**, including initializations. No checkpoint from the earlier paper runs is loaded. Method execution order is deterministically permuted within each seed.

### Controls and selection

All methods use entropy acquisition, the same seed-specific initial labeled set, the same annotation budgets, and the existing normalized alignment objective where applicable. The controls differ in how its strength is set:

- Fixed candidates: `0.003`, `0.01`, `0.03`, `0.10`.
- Preset schedules: linear increase or linear decrease, each with peak `0.03` or `0.10`. Progress spans all post-initialization optimizer steps; coefficients refresh at phase starts and nonterminal 100-step boundaries. No observed loss or geometry enters the schedule.
- MACC-Lite retains initial strength `0.03`, cap `0.10`, smoothing `0.20`, and Huber scale `0.25`. It is not retuned in this campaign.
- Entropy has no alignment regularization and serves as the acquisition-matched reference.

The two control families each receive four candidates evaluated on three development seeds. This equalizes the **current control-family search budgets**, not historical tuning effort between Lite and the controls. A broader search could find a stronger control; conclusions are conditional on this declared grid.

Selection uses the mean final top-1 accuracy on the seed's budgeted **training holdout**, independently for each control family. Accuracy ties within `1e-12` are resolved by lower mean holdout cross-entropy and then candidate identifier. All 24 tuning runs must finish. Neither the official validation images nor their predictions are evaluated during tuning. Selection is saved in an immutable `SELECTION.json` before confirmation. Neither control nor Lite is retuned for ResNet-18.

The holdout is drawn from the initial annotation budget: 12,669 annotated images include 633 holdout examples, leaving 12,036 examples for initial gradient training. Five acquisitions of 12,669 images lead to 76,014 total annotated images, including the unchanged holdout. Shared initialization uses 40 nominal epochs; each acquisition phase uses 20 nominal epochs. The balanced sampler uses 16 classes × 8 images. Batch-normalization running statistics are frozen after initialization for every method. Other training settings and data hashes are specified in `engine/protocol.json`.

### Outcomes and inference

The primary endpoint is final official-validation clean top-1 accuracy, with the training seed as the paired unit. On each backbone, the three contrasts are Lite minus entropy, Lite minus selected fixed, and Lite minus selected schedule. All **six contrasts across both backbones** form one Holm family, separately for paired-t and enumerated sign-flip tests. Primary tests are withheld until both complete matrices exist.

The report also gives each paired mean difference, seed-difference SD, individual unadjusted 95% paired-t interval, sign counts, and exact sign-flip p-value. Intervals are not simultaneous. Exact enumeration assumes sign exchangeability; it is not assumption-free. Ten seeds are fixed in advance. The previously observed paired SD of about 0.364 percentage points would imply an unadjusted 95% half-width of about 0.260 points if variability were similar; this is an illustration of precision, not a guaranteed power calculation. A gain of 0.5 percentage points is recorded as a practical reference, not a stopping rule.

Cross-entropy, calibration error, top-5 accuracy, geometry trajectories, and timing are secondary/descriptive. Intermediate means do not permit changing settings or ending the study based on favorable results. All planned runs and negative outcomes are retained; unsuccessful seeds are not replaced.

## What these results can resolve

- If Lite improves over both selected controls with adequate uncertainty bounds, the results support a benefit of feedback under the declared search and training protocol.
- If the fixed or scheduled control matches Lite, the claim must center on the alignment objective or practical control behavior, rather than asserting that adaptive feedback is necessary.
- A corresponding gain on ResNet-18 supports architecture transfer within this dataset. Failure to transfer narrows the claim to the tested ResNet-50 setting.
- An interval that includes both negligible and useful gains leaves the comparison uncertain. Statistical nonsignificance alone does not establish equivalence.

This campaign does **not** establish cross-dataset generalization, adversarial robustness, superiority to every alternative regularizer, or a causal isolation of the alignment term. It holds the existing objective and Lite implementation fixed. Conclusions are limited to the declared objective, control grids, dataset, and training protocol.

## Audit and validation

`engine/SOURCE_PROVENANCE.json` records the original module hashes and repository commit. `aligned.py`, acquisition, evaluation, and the vendored controller source remain unchanged. The new implementation adds ResNet-18 selection, predetermined strengths, suppression of official-validation evaluation during tuning, frozen selection, paired analysis, orchestration, and stronger configuration binding for resume.

`CPU_VALIDATION.json` records actual CPU tests run from the embedded delivery payload. They cover mathematical identities and gradients, both real torchvision backbones, a complete small training/acquisition/selection study, no validation leakage during tuning, idempotent completion, saved-result tampering, and bitwise recovery of model/optimizer/controller state after interruption. `LAUNCHER_VALIDATION.json` records extraction and integrity tests plus mocked process orchestration. These are software checks, not scientific experiment outcomes. The engine performs actual CUDA validation during preflight; the optional launcher checks that `tmux` is available.

The worker verifies saved logits against metrics and archived hashes before completion. Source, protocol, runtime, and dataset identities are bound to the run. Full checkpoints, acquisition indices, per-step logs, and per-seed outputs are retained. Reproducibility is scoped to this frozen software/hardware environment; CPU test success does not guarantee identical numerical results on a GPU.

Methodological references: [Cawley and Talbot (2010), selection bias](https://jmlr.org/papers/v11/cawley10a.html); [PyTorch 2.6 reproducibility guidance](https://docs.pytorch.org/docs/2.6/notes/randomness.html).
