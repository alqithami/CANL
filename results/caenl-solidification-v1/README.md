# Content-filtered classification replication and component study

This release contains the completed numerical endpoints of the CÆNL
solidification study: 206 method–seed runs, 23 independently trained shared
initializations and 1,030 post-initialization training phases. It supports
inspection and independent statistical reconstruction without downloading
licensed images or large model checkpoints. It contains no manuscript and no
outcomes from unfinished diagnostic studies.

## Files and reconstruction

- `RESULTS.json`: all 206 endpoint records; the 17 planned contrasts; all control
  selection scores and choices; data-audit summaries; scientific configuration;
  frozen source digests; and provenance identifiers.
- `reconstruct_stats.py`: a self-contained statistical reconstruction using
  Python 3, NumPy and SciPy. It checks record counts, selections and annotation
  budgets, reconstructs every planned effect and test, and compares these with
  the released analysis. It reads files and prints results; it does not launch
  experiments or alter model states.

From this directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy scipy
python3 reconstruct_stats.py
```

The expected first line is `Endpoint statistical reconstruction: PASS`.
Every planned contrast is printed. For machine-readable reconstructed values:

```bash
python3 reconstruct_stats.py --json
```

The optional positional argument selects another compatible endpoint file:
`python3 reconstruct_stats.py RESULTS.json`.

## Scientific design

The study uses one previously evaluated ImageNet-100 subset after prospective
exact-content filtering. The retained inventory contains 126,590 training and
4,995 validation images. The deterministic data policy checks original byte
digests and decoded-RGB identities, quarantines conflicting-label and cross-split
identity groups, and retains one representative of same-label within-split
duplicates. A total of 104 images are excluded before creating new seed splits.
This does not establish absence of perceptual near-duplicates or authenticate
upstream images or labels. The records report zero filename/synset checks
(`filename_synset_checked = 0`); no filename-based authentication is established.

The annotation sequence is 12,659, 25,318, 37,977, 50,636, 63,295 and 75,954
images. All methods include the same 632-label controller holdout in their
annotation count; it remains outside gradient training. The annotation budget
describes simulated acquisition after common benchmark curation and does not
include the historical cost of obtaining or checking the benchmark labels.

Selection uses ResNet-50 seeds 61001–61003 and four candidates from each of three
families: fixed DCR, preset schedules and a matched contrastive-feature penalty.
These 12 candidates across three seeds produce 36 tuning runs. Selection uses
training-holdout accuracy, followed by holdout cross-entropy and candidate name
for ties. No tuning run evaluates official validation. The chosen fixed
coefficient is 0.10; the chosen schedule decreases from peak 0.10; the chosen
contrastive coefficient is 0.10. All are best within the tested grids, not
globally optimized controls. The inherited `study.selection` description retains
a reference to 24 fixed/scheduled tuning runs; the separate contrastive family
adds 12 runs, giving the complete 36-run selection stage.

Confirmation uses ResNet-50 seeds 62001–62010, with six end-to-end conditions and
five common-label replay conditions per seed (110 runs). Transfer uses ResNet-18
seeds 63001–63010 with the six end-to-end conditions (60 runs), transferring
selected settings without retuning. MACC-Lite retains its earlier coefficient
settings and receives no new search. This is architecture transfer on the same
dataset, not an untouched benchmark or evidence of reduced lifetime tuning cost.

The replay treatments share entropy's acquired-label sequence, batch identities
and augmentation seeds. They estimate component effects conditional on that
sequence. The ratio-only and separation-only ablations use the strength selected
for the combined objective; they are not independently tuned reduced objectives.
The end-to-end treatments permit later acquisitions to differ and estimate a
different, full-pipeline effect.

The contrastive comparator uses one view per example, temperature 0.1,
layer3/layer4 feature normalization, seven same-class partners per anchor and no
projection head. It is a matched feature-penalty comparison, not the original
two-view, projection-head, two-stage SupCon training recipe. The full-label
reference annotates the entire retained pool after shared initialization and
matches the active-learning optimizer steps. It is not separately optimized to
convergence and cannot establish a 40% annotation saving from a nonsignificant
contrast.

All 17 final-top-1 contrasts form one planned Holm family, applied separately to
paired Student t tests and exhaustive two-sided sign-flip tests. Seeds are the
paired inferential units; individual 95% paired-t confidence intervals are
unadjusted. The new family remains separate from earlier sequential studies and
does not supply programme-wide error control. Ten paired seeds permit the stated
exact-test resolution but do not guarantee power. Nonrejection establishes
neither equivalence nor absence of an effect.

## Verification and provenance

Frozen campaign binding:

```text
18a7164a39e295a27fe864a1d858fae3069a25a9f4997f6435d8d048f870e65e
```

Verified compact review archive: `CAENL_SOLIDIFICATION_REVIEW.zip`.
Its SHA-256 is:

```text
37b1d7d55f2bab65582f2109583381cf83433686c6a0517fd24f7364483565db
```

The original archived `RESULTS.json` payload SHA-256 is:

```text
ca0be2a3c3d9d05fd638ead4096b97ad069ca4c16e238e0940ec29fe5af42596
```

This public `RESULTS.json` is a normalized endpoint export. It omits machine
paths, runtime environment details, operational instructions and invocation timing.
Its bytes therefore differ from that original payload. The original binding
identifies the frozen campaign inputs and is not a hash of the normalized file.
File-level release checksums are listed in `SHA256SUMS.txt`.

Independent local reconstruction reproduces all 17 planned differences, seed
SDs, confidence intervals, raw tests and Holm adjustments from the 206 numerical
endpoint records. It also reconstructs selection and protocol budgets. This is
arithmetic verification of saved measurements, not independent inference,
training repetition, image authentication or a fresh checkpoint hash audit.

Two distinct checkpoint-verification scopes are recorded:

1. **Historical feedback study:** 104 original final checkpoints were replayed
   on 104 holdouts and 80 official-validation sets without retraining. These are
   historical models and are not 104 additional training runs in this study.
2. **New solidification study:** all 206 new endpoints report reload of their
   final portable model. Detailed replay receipts and evaluation arrays are in
   the verified review archive. Endpoint reconstruction alone does not repeat
   those model evaluations.

Checkpoint replay reuses the archived inference implementation with separate
NumPy metric/moment arithmetic; it is not an independently implemented model.
No adversarial-robustness or practical-cost diagnostic outcomes are included here.
