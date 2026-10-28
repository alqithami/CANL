# Focused CIFAR-100 confirmation — completed 28 October 2026

The separate focused confirmation contains 160 final models: four selected
recipes, ten new seeds and four uniform-label budgets, plus 40 shared one-epoch
preparations. All 54 six-method development outcomes are retained.

| Selected recipe | Normalized accuracy AULC, mean ± seed SD |
|---|---:|
| CE | 58.6276 ± 1.3015 |
| Ratio DCR | 58.7472 ± 0.9724 |
| Full DCR | 58.7216 ± 0.8575 |
| Fisher | 57.1147 ± 3.0141 |

Ratio DCR minus CE, full DCR and Fisher is +0.1196, +0.0256 and +1.6325
percentage points, respectively. The three primary Holm-adjusted exact p-values
are 1.0, 1.0 and 0.36328125. None establishes a superiority gain. Paired-t
sensitivity uses a separate three-contrast Holm family; individual 95% intervals
are unadjusted. Recorded ratio-DCR job time is 11.40% higher than CE on average
across paired seeds, without an established AULC gain. Nonrejection does not
demonstrate equivalence or noninferiority.

## Design and interpretation

A 45,000-image training pool and separate 5,000-label development set are fixed
before training. Each seed uses nested uniform identity-only subsets of 4,500,
9,000, 18,000 and 27,000 training labels. Ten new seeds, 92001–92010, supply the
paired experimental units. Each seed-budget pair has a fresh CIFAR-stem
ResNet-18 and shares acquired IDs, raw initialization, acquired-only balanced
minibatches and augmentation streams across methods. The 5,000 development
labels are additional to the stated gradient-training label count.

Standard training BatchNorm and final-layer normalized features differ from
the ImageNet implementation. These are independently selected supervised
recipes, not active acquisition or a repetition of the equal-strength ImageNet
floor ablation. Ratio DCR uses strength 0.03; full DCR uses ratio strength 0.10
and floor weight 0.01. DCR branches share a one-epoch CE prefix and reset SGD
for 199 epochs; CE and Fisher train continuously for 200 epochs. Fisher denotes
the specified CE-plus-stabilized-trace-ratio recipe, not all Fisher-style methods.

The selected Fisher development mean was highest at 9,000 labels. The earlier
six-method confirmation proposal was amended prospectively to these four
methods before confirmation. It was not executed or declared completed.
SupCon and NONL remain development-only. Configurations stay fixed across
budgets and confirmation seeds. Nonmonotone budget outcomes, including lower
27k than 18k means for CE and Fisher, remain in the exports and limit claims
about recipe reliability. No adverse seed is dropped and test outcomes do not
select configurations or checkpoints. No annotation saving, robustness or
causal mechanism is established by this experiment.

## Contents and verification

`RESULTS_INPUT.json`, `LOCKED_ANALYSIS_SPEC.json`, `ANALYSIS.json`, the release
receipt, test metadata and all 160 `models/*/RECEIPT.json` files retain original
bytes. `MODEL_BINDINGS.json` indexes checkpoint, prediction-array and training
receipt identities. `TRAINING_RECORDS.json` retains all 200 training receipt
objects using public JSON serialization; original receipt hashes are recorded
separately. `development/` contains all 54 scores, selections and the development
protocol. `exports/` retains every test score, ten paired AULCs, all three planned
contrasts, all 40 method-seed costs and the nonmonotonicity check.

`PROVENANCE.json` identifies original review archives and distinct verification
boundaries. `SHA256SUMS.json` covers every distributed numeric file. All 160
prediction arrays, model states, saved accuracy numerators and receipt bindings
were independently checked before publication. Public reconstruction repeats
released-score arithmetic and metadata checks, not GPU training or inference.

The first evaluator opened the test member, then stopped before any model
attempt or score because an overstrict filename-uniqueness check found 9,999
unique strings among 10,000 rows. A versioned loader amendment retained all
rows, used row indices as identities and preserved the original two-hour
deadline. No duplicate-image inference or untouched-prior-test claim is made.
No score was retried and no training or analysis decision changed.

From the repository root, install `analysis/cifar100_focused/requirements.txt`
and run:

```bash
python3 -B analysis/cifar100_focused/reproduce.py --output reproduced-cifar100-focused
```

The output directory must be new. The workflow verifies public hashes, the
complete 160/54 grids, frozen source identities, checkpoint/receipt declarations,
all primary and sensitivity statistics, and descriptive costs. It writes numeric
tables and a column-sized measured plot. It does not prove data authenticity or
independent implementation replication. Raw data, test labels, prediction arrays,
operational logs and model binaries are excluded; their recorded identities and
verification scope remain explicit.
