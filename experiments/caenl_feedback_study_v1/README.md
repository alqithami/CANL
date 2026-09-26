# CAENL feedback study v1

This frozen campaign tests whether the unchanged MACC-Lite controller improves clean classification over separately tuned fixed-strength and predetermined-schedule controls. It then tests transfer of those settings from ResNet-50 to ResNet-18 on the existing ImageNet-100 dataset.

The campaign is prospective relative to these new seed outcomes. ImageNet-100 and the Lite settings have already been used in the research programme; this is not an untouched benchmark or a claim of equal lifetime development effort.

## Run on the IBM GPU server

Place `caenl_feedback_study_v1.py` in the server account's home directory, then run:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --start
```

The launcher embeds all engine code and extracts it after checksum verification. It starts a detached `tmux` worker; SSH can be disconnected. It does not install packages or download data. It uses the existing CAENL virtual environment and the dataset at `/mnt/caenl/persistent/datasets/imagenet100_cmc`. The expected environment is Python with PyTorch 2.6.x, torchvision 0.21.x, CUDA 12.4, and an NVIDIA L40S.

The default environment locations are:

1. `$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python`
2. `$HOME/caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python`

All campaign results are written under `/mnt/caenl/active/results/caenl-feedback-study-v1`. Existing experiments are not modified. The launcher refuses to run alongside an existing GPU compute process or to overwrite different source files. The worker checks the mounted volume, source hashes, dependency versions, available disk, all dataset image hashes, CPU integration tests, and actual batch-128 CUDA training steps for both backbones before the scientific runs.

Allow **several days of continuous GPU time** and approximately **160 GiB of free space initially**. This is a new 104-run campaign, not a short continuation of the six entropy-fixed runs. It runs sequentially on one GPU.

Check progress:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --status
tail -n 35 /mnt/caenl/active/results/caenl-feedback-study-v1/run.log
```

After an interruption, or after resolving an error shown by `--status`:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --resume
```

Resume verifies existing artifacts, skips completed work, and restores unfinished training from its last checkpoint. It preserves earlier failure records. A preflight check can also be run in the foreground using `--check`, but `--start` already performs it. `--unpack DIRECTORY` extracts inspectable source without starting experiments.

When complete, share the generated Markdown report:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --report
```

The report file is `/mnt/caenl/active/results/caenl-feedback-study-v1/caenl-feedback-study-v1.md`. Its sibling `.md.sha256` file allows verification using `sha256sum -c` on IBM. Share the text report; transferring model checkpoints or the dataset is unnecessary for reviewing these results.

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

This campaign does **not** establish cross-dataset generalization, adversarial robustness, superiority to every alternative regularizer, or a causal isolation of the alignment term. It holds the existing objective and Lite implementation fixed. It should not be described as restoring unsupported claims from the initial manuscript. Those require different evidence or narrower wording.

## Audit and validation

`engine/SOURCE_PROVENANCE.json` records the original module hashes and repository commit. `aligned.py`, acquisition, evaluation, and the vendored controller source remain unchanged. The new implementation adds ResNet-18 selection, predetermined strengths, suppression of official-validation evaluation during tuning, frozen selection, paired analysis, orchestration, and stronger configuration binding for resume.

`CPU_VALIDATION.json` records actual CPU tests run from the embedded delivery payload. They cover mathematical identities and gradients, both real torchvision backbones, a complete small training/acquisition/selection study, no validation leakage during tuning, idempotent completion, saved-result tampering, and bitwise recovery of model/optimizer/controller state after interruption. `LAUNCHER_VALIDATION.json` records extraction and integrity tests plus mocked process orchestration. These are software checks, not scientific experiment outcomes. Actual CUDA execution and IBM `tmux` are checked when the package is run on the server.

The worker verifies saved logits against metrics and archived hashes before completion. Source, protocol, runtime, and dataset identities are bound to the run. Full checkpoints, acquisition indices, per-step logs, and per-seed outputs are retained. Reproducibility is scoped to this frozen software/hardware environment; CPU test success does not guarantee identical numerical results on a GPU.

Methodological references: [Cawley and Talbot (2010), selection bias](https://jmlr.org/papers/v11/cawley10a.html); [PyTorch 2.6 reproducibility guidance](https://docs.pytorch.org/docs/2.6/notes/randomness.html).
