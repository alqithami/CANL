# CÆNL focused CIFAR-100 confirmation training

This is a new, explicitly amended confirmation protocol. It preserves all six
methods' development outcomes, including the finding that plain Fisher had the
highest selected development mean. It freezes four methods for independent
confirmation: CE, ratio-only DCR, full DCR and plain Fisher. The earlier proposed
six-method, five-contrast confirmation matrix is not executed or declared
completed. This stage makes no confirmed comparison with SupCon or NONL.

The focused stage retains the same hard negative comparator, ten independent
seeds 92001–92010, all four label budgets, and every selected configuration.
It reduces the measured confirmation cost while testing the main unresolved
objection: whether DCR adds predictive value beyond ordinary scatter reduction.
The full development results and exact selection remain bound in this source.

## Frozen matrix and cost

- 4,500, 9,000, 18,000 and 27,000 labels from the unchanged 45,000-image pool.
- One uniform identity-only permutation per seed; prefixes make nested sets.
- All methods share a seed-budget pair's labels, raw backbone initialization,
  balanced minibatches and augmentation streams. Raw states are fresh at each
  budget; models do not carry across budgets.
- Each epoch has 45, 90, 180 or 270 balanced batches of 100, respectively.
  A batch has ten classes and ten distinct acquired images per class. Repetition
  between batches is allowed. An unsupported frozen draw blocks the entire
  plan; there is no replacement seed, label draw or extra acquisition.
- CE and Fisher train for 200 epochs. Ratio/full DCR share one CE prefix epoch
  and then each train for 199 epochs with frozen, labeled-training-only targets.
  That prefix counts inside each treatment's advertised 200-epoch allocation.
- 160 treatment trajectories plus 40 shared preparations: 200 physical stages,
  no classifier probes, and exactly 4,674,150 physical optimizer updates.
- A 96-hour coordinator ceiling, two-hour treatment ceilings and 30-minute
  prefix ceilings are fixed before training. Preparation has a one-hour cap.
  No automatic retries or budget extensions are enabled.

A single host can run the entire matrix. Optional deterministic shards assign
all four budgets of seed index `i` to shard `i % shard_count`; allowed counts
are 1, 2, 5 and 10. Every shard freezes its own identity and explicit job list.
The global list remains unchanged. Hosts require independently verified idle
GPU capacity and the same source and train-cache byte bindings.

## Before launch

Keep the prior engineering, development and confirmation source directories
separate. Copy the completed development power/variance assessment as
`POWER_AND_DEVELOPMENT_ASSESSMENT.json` into this source tree, then generate
`SOURCE_MANIFEST.json` over every source, recipe, reference, assessment and
analysis file, excluding only the manifest and bytecode caches. Outputs and
validation reports belong outside the immutable source tree.

Use the existing PyTorch environment without changes. Set `TMPDIR` to a new
runtime directory on the spacious experiment filesystem. Reuse the original
verified pilot train cache without modifying it.

```bash
python -B source/test_methods.py --device cuda --report validation/METHODS_CUDA.json
python -B source/test_confirmation.py --with-torch --report validation/RUNNER_INTEGRATION.json
python -B source/test_analysis.py
python -B source/confirmation.py prepare --output output --shard-index 0 --shard-count 1
python -B source/confirmation.py launch --output output --shard-index 0 --shard-count 1
python -B source/confirmation.py status --output output
```

The default pilot cache is
`/mnt/caenl/active/caenl-cifar100-v1/pilot_output`; `--pilot-output` can identify
an independently byte-verified copy on another authorized host. No archive or
official-test reader is part of the training runner. Existing incomplete
attempts and reservations block silent reruns. Live process identity uses PID,
Linux start ticks and boot ID. A worker verifies that its coordinator remains
live; termination or timeout preserves the attempt and spend ledger.

## Analysis and release boundary

`ANALYSIS_SPEC.json` and `analyze.py` freeze seed-level normalized AULC with
weights 0.1, 0.3, 0.4 and 0.2. The primary family has three two-sided exact
paired sign-flip contrasts: ratio DCR minus CE, full DCR and Fisher, with Holm
correction across all three. Paired-t sensitivity results form a separate
three-contrast Holm family; their 95% intervals are unadjusted. Ten seeds have
limited sensitivity to small effects; the bound assessment records that limit.
A nonsignificant difference is not an equivalence or annotation-saving result.

Preparation emits a source-bound `ANALYSIS_FROZEN_TEMPLATE.json` with
`locked=false` and no test-release receipt. Final development accuracies are
saved only as diagnostics and cannot choose configurations, epochs or methods
again. Training completion does not evaluate or unlock the official test.
The analysis program rejects development partitions, unreleased test inputs,
changed seed families and mismatched release-receipt hashes. A later, separate
release must verify all global models and bind an explicit test-release receipt
before completing the locked analysis contract. The statistical decisions in
this frozen source cannot change at that release.

Completed shard review ZIPs preserve every receipt, history, source binding,
checkpoint hash, full diagnostic grid and actual cost. Full checkpoints remain
on the server. No superiority, annotation savings, active-acquisition,
robustness, or mechanism conclusion follows from training completion alone.
