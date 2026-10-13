# Frozen-model robustness and practical-cost protocol

Specification date: 11 October 2026. This is a separate prospective checkpoint
evaluation specification. It does not change the 206-method training campaign,
its selection rules, or its 17-contrast analysis family. Earlier research
development and historical outcomes were already known when this follow-up was
specified. It is not programme-wide preregistration or error control.

## Questions

1. Does the classification treatment show a benefit under smaller input
   perturbations, including comparison against tuned fixed regularization?
2. How do predictions, margins, and measured representations change on the
   examples returned by the attack suite?
3. What execution cost is required for fixed and feedback-controlled training
   once unused research monitoring is removed from the fixed baseline?

The independent value of adaptation for clean prediction is determined by the
completed training campaign's predefined fixed/scheduled and common-label
comparisons. The present diagnostics do not replace those comparisons.

## Checkpoints and data

Use the final scheduled checkpoint for every combination of ResNet-50
confirmation seeds 62001–62010 and conditions entropy, tuned_fixed, and lite.
There are exactly 30 models. No outcome selects a model or seed. Load only
`FINAL_MODEL.pt` after verifying its campaign receipt, byte hash, and tensor
digest. Reconstruct the retained catalog from the campaign's completed content
audit without rewriting that audit or any original data.

Use all cleaned validation images in their canonical order with the original
resize/center-crop transform and pixel-space [0,1] inputs. Check image byte
hashes. Before attacks, reproduce each model's clean predictions and logits
against its original round-5 payload. Require identical predictions and
logit agreement at absolute/relative tolerance 1e-4. Disagreement blocks.

## Attacks and inference

For every model, run unchanged standard AutoAttack at radii
0.5/255, 1/255, 2/255, 4/255, and 8/255 in L∞. The suite is
APGD-CE, targeted APGD-DLR, targeted FAB, and Square. Do not reduce iterations,
queries, target classes, or restarts. This gives exactly 150 full-validation
suites. Inference and attack gradients use FP32, evaluation mode, no autocast,
and disabled TF32. Classifier normalization stays inside the model.

Batch size is 32; independently resumable chunks contain at most 128 canonical
images. Corresponding method evaluations share deterministic seed/radius/chunk
attack seeds. Store original and adversarial logits, predictions, norms, and
per-image correctness/margins. Check finite outputs, pixel range, and the
specified perturbation bound for every chunk. Payload digests and source/data/
checkpoint bindings are required to reuse a completed chunk.

Primary robust accuracy uses all validation images as the denominator; initially
misclassified inputs do not become robust-correct. At 2/255, compare
tuned_fixed−entropy, lite−entropy, and lite−tuned_fixed across ten paired seeds.
Report mean effects, seed SDs, individual unadjusted paired-t 95% intervals,
paired-t tests, and exact two-sided sign-flip tests. Apply Holm separately to
each three-test family. Seed exchangeability/symmetry and paired-t assumptions
remain; exact enumeration is not assumption-free. No equivalence is inferred
from a nonsignificant result or zero difference variance.

All other radii and intermediate measurements are descriptive. A secondary
survival comparison uses the same intersection of clean-correct images across
the three methods within each seed. It addresses differing clean-correct
denominators; it is not a causal identification procedure. Empty intersections
are reported as undefined rather than assigned an accuracy.

## Representation and sensitivity diagnostics

At layer3 and layer4, normalize each feature vector using the original norm
floor. Accumulate true-label class counts, sum vectors, and scalar sums of
squared feature norms in float64. These are sufficient for the equal-class
within/between trace definition, even when cleaned class sizes differ.
Store sparse active-class moments per chunk and dense final aggregates.
No full per-image feature vectors are exported.

Report clean and returned-adversarial scatter, normalized feature displacement,
and true-class logit margins. AutoAttack returns initially incorrect examples
unchanged and uses successful examples from heterogeneous constituent attacks.
These moments describe that returned mixture, not a uniformly applied geometric
perturbation. Also summarize displacement/margins separately for initially
incorrect examples, successful attacks on clean-correct examples, and survivors.
These conditional groups do not prove geometric mediation.

For the first two canonical validation images per true class (at most 200),
measure the L1 norm of the input gradient of the true-versus-largest-competing
logit margin. The competitor and gradient can change away from the clean point.
This quantity is a local sensitivity proxy, not a robustness certificate or
neighborhood Lipschitz constant. Negative clean margins remain visible.

## Practical execution benchmark

Use seed62001 entropy's final weights as a common starting state. Obtain eight
real balanced augmented minibatches from its final labeled set, plus the full
original feedback holdout. Check the bytes of all used images before decoding.
Cache uint8 CPU tensors; report the cache input digest, size, construction time,
and the fact that the same eight transformed batches repeat cyclically.

Compare streamlined and instrumented variants of entropy, tuned fixed DCR, and
Lite. There are six conditions and five matched timing repetitions. Condition
order is predefined independently of timings, alternating reversed permutations.
For every condition, run 20 discarded warmup steps; then restore original
weights, fresh SGD state, RNG, and controller state before 200 measured steps.
Use the original BF16 forward path, FP32 parameters/geometry, frozen BatchNorm
statistics, original SGD parameter groups, and constant round learning rate.
This constant-rate short benchmark is not a training-performance experiment.

Streamlined entropy omits unused geometry/feedback monitoring. Streamlined fixed
DCR retains the exact differentiable loss but omits unused statistic outputs,
scalar trace extraction, EMA state, and holdout evaluation. Contract tests must
show matching loss and gradients. Lite retains its actual EMA/coefficient
recurrence and holdout-CE path; its coefficient formula does not consume CE,
but this benchmark does not introduce a different Lite implementation.
Instrumented conditions retain the research monitoring/holdout path for
comparison. No changed model or optimizer is saved.

Report synchronized measured-segment elapsed time, training compute, transfers,
monitoring, peak PyTorch allocated/reserved GPU memory, and serialized controller
state bytes. The serialized byte count is not controller RAM. Exclude and
separately report setup/cache construction; serialization, acquisition, attacks,
and complete application time are outside the measured segment.

Timing repetitions are descriptive runs on one starting model, not independent
scientific training seeds. Report mean/SD/median and paired timing differences;
do not use these timings to claim predictive superiority, full-job throughput,
or total engineering economics. Report actual completed grid cost separately
for fixed, scheduled, and contrastive families. Earlier development cost is
unknown. Frozen feedback parameters do not imply zero prior tuning.

## Completion and interpretation

The worker waits without CUDA allocation for verified completion of the active
campaign and an idle GPU. Waiting is bounded by 20 days and stops if the source
campaign is blocked. CPU and real-model contracts run before evaluation.
Completion requires every specified model/radius and all cost repetitions.
No metric-dependent stopping, replacement seeds, best checkpoint selection, or
radius selection is permitted. Negative and inconclusive outcomes are retained.

The result cannot establish general robustness, causality of geometry changes,
an adaptive-control advantage, or applicability beyond the specified models and
data. Those claims require their own supporting comparisons and assumptions.
