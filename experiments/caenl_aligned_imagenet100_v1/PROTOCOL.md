# Frozen aligned ImageNet-100 active-learning replication v1

## What is being tested
This is a new, full active-learning comparison of the audited aligned-DCR formulation. It is NOT a repetition of the spectral-DCR pilot, a continuation of seeds 601/602, or proof in advance of a positive result. Prior negative ImageNet results and completed C4 experiments are retained unchanged. Input-space robustness must be measured; improving feature geometry alone is not a certificate.

The objective implementation (`aligned.py`) and Full MACC class (`vendor/controllers.py`) are byte-identical to the aligned development package. Layer normalization, one-sided Huber objective, separation floor, lambda bounds, bounded leaky rule, and Full MACC reward components retain their audited values. End-to-end initialization, training schedule, and acquisition/evaluation integration are newly specified here; the short development continuations did not themselves validate these complete settings.

## Locked experiment
- Native CMC ImageNet-100, 126,689 training/5,000 official validation images; original byte hashes and class mapping are verified. No new download or reencoding.
- Six independent initial weights/splits: seeds 801--806. No development-model checkpoint is read.
- ResNet-50 from scratch, 224-pixel training RandomResizedCrop/flip; shorter-side 256 and center 224 for scoring, feedback and final evaluation. Pixel-space normalization is inside the model.
- Nine methods per seed (54 end-to-end method runs): random; entropy; exact-factorized BADGE; independently implemented Noise Stability; original Mahalanobis CÆNL acquisition-only; random + aligned fixed DCR; CÆNL + aligned fixed; CÆNL + bounded MACC-Lite-AD; CÆNL + Full MACC-AD regularization control.
- The two nontrivial literature baselines are independent implementations, not claimed byte-for-byte authors' reproductions. BADGE uses exact last-linear-weight gradient distances with no sketch and excludes bias. Noise Stability uses five Gaussian perturbations of the complete trainable parameter vector, relative norm 0.001, normalized softmax-output deviations, exact feature concatenation, and greedy farthest-first selection. Model parameters/buffers are restored exactly after noisy forwards.
- Initial label count 12,669 includes a fixed, stratified controller holdout of 633. Initial gradient-training count 12,036. Five acquisitions of 12,669 bring the final label count to 76,014 and gradient-training count to 75,381. No repeated acquisition or holdout/training overlap is allowed.

## Initialization, target calibration and BatchNorm
For each seed, train one shared CE-only initial model for 40 nominal epochs on the initial gradient set. All methods start their active acquisitions from that SAME seed-specific state, including identical optimizer momentum. BatchNorm running statistics update during this initial phase. At its end, calibrate layer3/layer4 targets once from 16 center-cropped, balanced anchor minibatches of TRAINING images; no official validation or controller-holdout labels enter target selection.

The same targets remain frozen for all methods and all five subsequent rounds. The layer target is mean anchor log((W+eps)/(B+eps))+log(0.9). The between-class floor is half the mean anchor B. Retaining an anchor target does not imply that a 100-class validation ratio can be interpreted as this 16-class minibatch target's attainment.

Freeze BatchNorm running statistics for EVERY subsequent method/round, matching the audited continuation's BN convention; affine BN weights still train. This is a specified common integration choice, not a hidden favorable-method-only adjustment. There is no reestimated target or learning-rate tuning after observing validation outcomes.

## Matching and optimization
A physical batch is 16 uniformly selected classes x eight distinct currently labeled examples. All methods, including baselines, use this recipe. A nominal epoch is ceil(current gradient-training count / 128) balanced sampled minibatches; sampling is with replacement across minibatches, not an exact once-per-epoch permutation. Exact example presentations are recorded. Common deterministic seed/phase/step/image streams are used; divergent acquired sets naturally lead to divergent sampled examples.

SGD, momentum 0.9, Nesterov, weight decay 1e-4 on multidimensional parameters, gradient clipping norm 5, FP32 weights with BF16 training autocast. Initial LR .05, cosine schedule with three-epoch warm-up. Five acquired-label phases each use 20 nominal epochs, base LR .025, cosine decay and one-epoch warm-up. Optimizer momentum is retained across phase boundaries for every method. Per-epoch/1,000-step checkpoints include model, optimizer, controller, pending action, RNG, data order, and accumulated records.

All branches compute geometry and use the same periodic controller-holdout evaluations. Report timing as compute plus monitoring/feedback and acquisition separately, with shared-initialization time explicitly allocated once per conceptual method run; do not label this as overhead relative to an uninstrumented plain network.

## Frozen regularization/control
The aligned objective unit-normalizes each pooled example representation and penalizes positive gaps in log within/between scatter with Huber delta .25. A fixed .01 separation-floor penalty remains enabled even when Full MACC disables a ratio term. Fixed lambda is .03. MACC-Lite starts at .03 and uses desired=.10*tanh(max(gap,0)/.25), then lambda=.8*lambda+.2*desired. Training geometry EMA beta .05. Both controller strengths lie in [0,.10]. These values are inherited, not retuned.

Full MACC retains hidden layers [128,128], delta choices [-.01,-.005,0,.005,.01], policy LR 3e-4, exploration .10 to .02 over the predeclared controller horizon, entropy .001 and reward-baseline EMA .05. Each action is rewarded AFTER its subsequent window of up to 100 training steps. Reward = clipped feedback-CE improvement / max(.1, phase-initial feedback CE) + .1*clipped decrease in aligned training-EMA potential - .01*mean effective lambda/.10. Terminal actions are rewarded before a round boundary; a new action is initialized with the new phase baseline, rather than assigning acquisition/restart effects to the old action. The learned state, private sampling RNG and pending action survive checkpoint/recovery.

**Full MACC's acquisition head is disabled in this replication.** The .90 query quantile is fixed for all CÆNL methods. This evaluates the learned regularization controller IN an active-learning pipeline, not a learned threshold policy or lower label count. Original Full MACC acquisition results remain in the previous pilot. No publication claim may conflate these components.

## Acquisition
At each round, infer pool predictions/features in eval mode and cache them in RAM for repeated passes (not repeated downloads). Candidate passes have at most 10,000 unlabeled examples; take up to 10% per pass with deterministic tie handling, repeating until the exact budget is met. Candidate IDs, chosen IDs, scores when scalar, pseudo-labels, and presentation counts are saved. Every method uses the same fixed pass-size/budget rules. Equal label budgets are not equal acquisition-compute costs, especially for Noise Stability.

For the CÆNL score, use the existing weighted raw-feature predicted-class Mahalanobis geometry proxy, with layer weights .5/.5. Diagonal class means/variances are reestimated from all CURRENTLY LABELED gradient-training examples at the current checkpoint, shrinkage 1e-5; pool true labels are NOT used. This explicit roundwise estimate replaces the old path's stale/EMA estimate and applies identically to acquisition-only and every aligned CÆNL branch. It is not a newly measured sample-level temporal-instability score.

## Endpoints
Every completed phase: full-precision clean top-1/top-5, pooled CE and 15-bin ECE on all 5,000 validation images; normalized geometry and sufficient class statistics recorded separately. No selection of a best-validation checkpoint: the final scheduled checkpoint is always used. Feedback uses only the initial training-pool holdout.

Final 60% checkpoint: pinned official standard AutoAttack, Linf epsilon=8/255, all 5,000 validation images, attack batch 64, restartable outer storage chunks 256. Standard attack list/iterations are not shortened. Record full-precision clean/adv logits, labels, IDs, robust masks and perturbation norms. No adversarial training, extra favorable epsilon search, robustness guarantee or positive-result pass criterion.

## Statistics and reporting
Six independent training seeds are the units of inference. Predeclared final-accuracy and final-robustness contrasts are in protocol.json. Report paired differences, 95% paired-t intervals, exact sign-flip tests, and Holm correction across the planned contrasts within each endpoint. All-zero effects yield an exact p of 1; degenerate nonzero paired variance is not displayed as a spuriously valid parametric p=0. Interim outputs have no final inferential labels. Geometry/ECE and label-budget curves are secondary. Neither a significance threshold nor a favorable outcome is required for completion.

## Scope and preservation
This is ImageNet-100 replication only, not the ImageNet-1K main experiment or diffusion/audio validation. It is a multi-day run on one L40S. Source/protocol hashes are frozen before results. If the protocol or implementation is changed after outcomes are inspected, those affected runs must be labeled development rather than silently combined with unchanged independent replication.

Existing datasets, C4, pilot, diagnostic and aligned development outputs are read-only and are not cleaned or overwritten. New data live under `/mnt/caenl/active/results/caenl-imagenet100-aligned-confirmation-v1`. Checkpoints remain on disk. The final lean archive excludes weights/raw images, contains exact source/configuration/metrics/indices/controller histories and internal SHA-256 checksums, and is verified before `COMPLETE` is written.

## Primary external sources
- BADGE: https://github.com/JordanAsh/badge (gradient-embedding method; implementation here is independent).
- Li et al. (2024), Deep Active Learning with Noise Stability: https://ojs.aaai.org/index.php/AAAI/article/view/29270.
- Croce and Hein, official pinned standard AutoAttack: https://github.com/fra31/auto-attack/tree/a39220048b3c9f2cca9a4d3a54604793c68eca7e.
- PyTorch reproducibility scope: https://docs.pytorch.org/docs/stable/notes/randomness.html. Same seeded streams do not promise bitwise reproducibility across different hardware/software versions.
