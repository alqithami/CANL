# Completed experimental findings

This page summarizes the completed experiments released with CÆNL. It covers classification, language modeling, diffusion generation, audio captioning, representation geometry, adversarial evaluation, and computational cost. Positive, negative, and inconclusive outcomes are retained. Numerical inputs, comparison families, and reconstruction code are linked below.

The strongest classification evidence supports the evaluated scatter-regularization intervention with entropy acquisition at the specified final annotation budget. Common-label replay now identifies a benefit from the scatter-ratio penalty conditional on the shared acquired-label sequence. The completed comparisons do **not** establish that adaptive feedback outperforms an independently tuned constant. The spectral formulation used for language, diffusion, and audio is a distinct objective, so its outcomes cannot be treated as tests of the aligned supervised classification objective.

## Records and statistical conventions

| Completed study | Experimental units | Public numerical records |
|---|---|---|
| Earlier ImageNet-100 classification | 72 method–seed runs: 12 conditions × 6 paired seeds, 360 post-initialization phases | [Earlier snapshot](../results/2026-09-26/), [classification records](../results/2026-09-26/classification_per_seed.json) |
| ImageNet-100 control selection, confirmation, and architecture transfer | 104 additional method runs, 23 shared initializations, 520 post-initialization phases: 24 tuning runs, 40 ResNet-50 confirmation runs, 40 ResNet-18 transfer runs | [Feedback snapshot](../results/2026-10-02-feedback/), [native-metric extracts](../results/2026-10-02-feedback/reconstructed_evaluations.json) |
| Content-filtered ImageNet-100 replication and components | 206 method runs, 23 shared initializations, 1,030 subsequent phases: 36 selection, 110 ResNet-50 confirmation/component, 60 ResNet-18 transfer runs | [Numerical release and reconstruction](../results/caenl-solidification-v1/), [all endpoint records and 17 contrasts](../results/caenl-solidification-v1/RESULTS.json) |
| GPT-2/C4 | 24 trained checkpoints: 4 conditions × 6 training seeds; the same checkpoints subsequently evaluated on another local validation shard | [C4 records](../results/2026-09-26/c4_per_seed.json) |
| CIFAR-10 diffusion | 24 trained runs plus 6 sampling streams from one unchanged pretrained reference | [Diffusion records](../results/2026-09-26/diffusion_per_seed.json) |
| AudioCaps captioning | 24 trained runs: 4 conditions × 6 training seeds | [Audio records](../results/2026-09-26/audio_per_seed.json) |

Development screens, implementation checks, reused initializations, and a second evaluation of existing checkpoints are not additional independent confirmation seeds. The 72-, 104- and 206-run classification studies have separate seed sets and inferential families. They are not pooled; the 206-run study also uses a separately filtered inventory.

Throughout this page, `mean ± SD` denotes the sample standard deviation across training seeds. Effect intervals are individual, unadjusted 95% paired-t intervals. They are not simultaneous intervals. `Holm t` and `Holm exact` are separately multiplicity-adjusted paired-t and exhaustive sign-flip p-values within the specified family. Classification accuracy effects use percentage points (pp); ECE values use proportions unless otherwise stated. A nonsignificant comparison does not demonstrate equivalence or non-inferiority.

With six pairs, the minimum two-sided exact sign-flip p-value is 0.03125. For a Holm family with at least two comparisons, no contrast can cross the first 0.05 threshold at this sample size. We retain the exact results alongside the paired-t results rather than treating either as interchangeable evidence. The ten-seed control study uses all 1,024 sign patterns and has finer resolution.

The earlier machine-readable comparison lists, including secondary C4 outcomes, are in [reported_contrasts.json](../results/2026-09-26/reported_contrasts.json). The 104-run study releases [primary contrasts](../results/2026-10-02-feedback/primary_contrasts.csv) and [sensitivity contrasts](../results/2026-10-02-feedback/sensitivity_contrasts.csv). The separate [206-run endpoint file](../results/caenl-solidification-v1/RESULTS.json) retains all 17 new contrasts and their raw/adjusted tests.

## Classification: completed content-filtered replication and components

The 206-method study filters exact content before new training and seed partitions. The fixed policy groups original file hashes and decoded-RGB identities, excludes every member of conflicting-label and cross-split groups, and retains one representative of same-label duplicates within a split. It excludes 104 images (99 training and five validation), retaining 126,590 training and 4,995 validation images. This is prospective exact-content filtering on a previously studied benchmark, not certification against perceptual near-duplicates or authentication of upstream images/labels. Benchmark labels used for common curation are outside the simulated acquisition budget.

Each active-learning condition uses exactly 60% annotations: 75,954 labels including a 632-label feedback holdout, leaving 75,322 gradient-training images. Selection uses seeds 61001–61003: four fixed strengths, four schedules and four contrastive-feature strengths, for 36 runs. Each family is selected by mean training-holdout accuracy with cross-entropy and identifier tie-breaks; official validation is not used during selection. The selected fixed coefficient is 0.10, the schedule decreases from peak 0.10, and the contrastive coefficient is 0.10. These are choices within finite grids, not global optima. Frozen Lite settings and selected controls transfer without retuning.

ResNet-50 seeds 62001–62010 supply 60 end-to-end and 50 common-label replay runs. ResNet-18 seeds 63001–63010 supply 60 end-to-end transfer runs. The experiment executes 23 shared initializations, 1,030 subsequent phases and 8,141,080 optimizer steps. Every condition shares its seed's initialization; each has 39,100 post-initialization steps. Final validation top-1 is primary; other endpoint measures below are descriptive.

| Backbone | Treatment | Top-1 (%) | Cross-entropy | ECE | Top-5 (%) | Mean log scatter |
|---|---|---:|---:|---:|---:|---:|
| ResNet-50 | Entropy | 71.493 ± 0.542 | 1.144 ± 0.016 | 0.079 ± 0.003 | 90.875 ± 0.216 | 1.389 ± 0.011 |
| ResNet-50 | Entropy + Lite | 73.393 ± 0.707 | 1.021 ± 0.021 | 0.055 ± 0.006 | 91.982 ± 0.333 | 0.499 ± 0.060 |
| ResNet-50 | Entropy + tuned fixed | 73.546 ± 0.747 | 1.012 ± 0.018 | 0.055 ± 0.006 | 92.130 ± 0.208 | 0.355 ± 0.066 |
| ResNet-50 | Entropy + schedule | 72.763 ± 0.808 | 1.056 ± 0.022 | 0.066 ± 0.006 | 91.784 ± 0.310 | 1.006 ± 0.033 |
| ResNet-50 | Entropy + contrastive | 72.300 ± 0.777 | 1.093 ± 0.024 | 0.074 ± 0.005 | 91.369 ± 0.314 | 1.007 ± 0.031 |
| ResNet-50 | Full labels, matched steps | 73.524 ± 0.606 | 0.992 ± 0.017 | 0.039 ± 0.004 | 92.012 ± 0.395 | 1.197 ± 0.017 |
| ResNet-18 | Entropy | 71.540 ± 0.567 | 1.196 ± 0.023 | 0.095 ± 0.004 | 90.454 ± 0.409 | 1.165 ± 0.014 |
| ResNet-18 | Entropy + Lite | 73.151 ± 0.557 | 1.090 ± 0.022 | 0.083 ± 0.005 | 91.508 ± 0.427 | 0.539 ± 0.026 |
| ResNet-18 | Entropy + tuned fixed | 73.243 ± 0.410 | 1.086 ± 0.017 | 0.081 ± 0.004 | 91.495 ± 0.237 | 0.424 ± 0.023 |
| ResNet-18 | Entropy + schedule | 72.871 ± 0.595 | 1.119 ± 0.021 | 0.084 ± 0.005 | 91.183 ± 0.385 | 0.816 ± 0.020 |
| ResNet-18 | Entropy + contrastive | 72.282 ± 0.484 | 1.144 ± 0.020 | 0.090 ± 0.004 | 91.023 ± 0.192 | 0.839 ± 0.021 |
| ResNet-18 | Full labels, matched steps | 74.819 ± 0.490 | 0.970 ± 0.018 | 0.052 ± 0.004 | 92.276 ± 0.319 | 0.997 ± 0.015 |

Common-label replay holds entropy's acquired-label sequence, batches and augmentation streams constant. Ratio-only and separation-only conditions retain the strengths selected for the combined objective; they are removal comparisons, not independently tuned reduced objectives. The entropy row is reused as the donor and is not another training run.

| ResNet-50 common-label intervention | Top-1 (%) |
|---|---:|
| No representation penalty (entropy donor) | 71.493 ± 0.542 |
| Ratio only, fixed 0.10 | 73.614 ± 0.703 |
| Separation only, weight 0.01 | 71.421 ± 0.587 |
| Ratio + separation, fixed | 73.481 ± 0.812 |
| Ratio + separation, Lite | 73.542 ± 0.785 |
| Ratio + separation, schedule | 73.037 ± 0.828 |

All 17 planned final-top-1 contrasts form one Holm family, applied separately to paired-t and exhaustive sign-flip tests. Each uses ten paired seeds. Intervals below are individual and unadjusted; interval exclusion alone does not establish adjusted significance. In replay rows, `fixed` means ratio plus separation at fixed strength. The interaction is fixed − ratio − separation + entropy.

| Design | Target minus reference | Effect (pp) | 95% interval (pp) | Holm t | Holm exact |
|---|---|---:|---|---:|---:|
| ResNet-50 | Lite − entropy | +1.900 | [+1.536, +2.264] | 0.00001315 | 0.03320 |
| ResNet-50 | Lite − tuned fixed | −0.152 | [−0.593, +0.289] | 1.0000 | 1.0000 |
| ResNet-50 | Lite − schedule | +0.631 | [+0.196, +1.065] | 0.08543 | 0.15820 |
| ResNet-50 | Lite − contrastive | +1.093 | [+0.691, +1.495] | 0.002030 | 0.04688 |
| ResNet-50 | Lite − full labels | −0.130 | [−0.531, +0.270] | 1.0000 | 1.0000 |
| ResNet-18 | Lite − entropy | +1.612 | [+1.247, +1.977] | 0.00005061 | 0.03320 |
| ResNet-18 | Lite − tuned fixed | −0.092 | [−0.403, +0.219] | 1.0000 | 1.0000 |
| ResNet-18 | Lite − schedule | +0.280 | [−0.152, +0.712] | 1.0000 | 1.0000 |
| ResNet-18 | Lite − contrastive | +0.869 | [+0.428, +1.309] | 0.01731 | 0.04688 |
| ResNet-18 | Lite − full labels | −1.668 | [−2.248, −1.087] | 0.001456 | 0.03320 |
| R50 common labels | Ratio − entropy | +2.120 | [+1.783, +2.457] | 0.000003050 | 0.03320 |
| R50 common labels | Separation − entropy | −0.072 | [−0.328, +0.183] | 1.0000 | 1.0000 |
| R50 common labels | Fixed − ratio | −0.132 | [−0.437, +0.173] | 1.0000 | 1.0000 |
| R50 common labels | Fixed − separation | +2.060 | [+1.672, +2.448] | 0.00001217 | 0.03320 |
| R50 common labels | Lite − fixed | +0.060 | [−0.199, +0.320] | 1.0000 | 1.0000 |
| R50 common labels | Schedule − fixed | −0.444 | [−0.730, −0.159] | 0.06514 | 0.11719 |
| R50 common labels | Factorial interaction | −0.060 | [−0.524, +0.404] | 1.0000 | 1.0000 |

The end-to-end gains over entropy survive both adjusted procedures on both backbones. The matched contrastive-feature comparison also favors Lite under both procedures, but uses one view, normalized layer3/layer4 features, temperature 0.1 and no projection head; it does not reproduce or establish general superiority over the original SupCon training recipe. Tuned fixed DCR has slightly higher mean accuracy than Lite on both backbones, and neither the adaptive-versus-fixed comparisons nor the common-label feedback increment establishes superiority, equivalence or non-inferiority.

The common-label ratio-only benefit and fixed-both advantage over separation-only survive both corrections. Separation's increment, feedback's increment and the factorial interaction remain inconclusive. Acquisition-sequence differences cannot explain these replay comparisons, but the results do not establish geometric mediation, isolate normalization, or allocate all gains of freely evolving acquisition pipelines to one component.

The full-label reference uses all 126,590 annotations after shared initialization with the same post-initialization optimizer-step budget; it is not separately optimized to convergence. ResNet-18 favors full labels, whereas ResNet-50 is inconclusive. Neither establishes an equal-performance 40% annotation saving. Geometry is also not a universal predictive ranking: full-label ResNet-18 has higher mean log scatter than Lite (0.997 versus 0.539) and higher accuracy (74.819% versus 73.151%). This is descriptive, not a mediation test.

The [release guide](../results/caenl-solidification-v1/README.md) documents frozen inputs, data filtering, original archive identity, the normalized public export and separate checkpoint-replay scopes. Run its CPU reconstruction to inspect all endpoints and raw/adjusted tests. The 206-run clean-classification study remains its own record. The subsequently completed [diagnostics](../results/caenl-diagnostics-v1/) supply the separately specified attack-radius and cached-path cost evidence below.

## Classification: earlier 104-run selected controls and architecture transfer

All four treatments use entropy acquisition on ImageNet-100. Control selection evaluates eight candidate fixed/scheduled settings using 633-example training holdouts on ResNet-50 seeds 41001–41003. Official validation is not evaluated during selection. The selected constant is 0.10; the selected schedule decreases linearly from peak 0.10. Selection is within the tested grid, not a global optimum. Their settings and MACC-Lite settings are frozen for ten ResNet-50 confirmation seeds 42001–42010 and ten ResNet-18 transfer seeds 43001–43010.

The final endpoint uses all 5,000 official validation images at approximately 60% annotations: 76,014 acquired labels including 633 feedback labels, leaving 75,381 gradient-training images. The 10% initialization is shared within each seed. See the [frozen protocol](../experiments/caenl_feedback_study_v1/README.md) for model, acquisition, training, and controller details.

| Backbone | Treatment | Clean top-1 (%) | Cross-entropy | ECE |
|---|---|---:|---:|---:|
| ResNet-50 | Entropy | 71.678 ± 0.808 | 1.137 ± 0.024 | 0.079 ± 0.007 |
| ResNet-50 | Entropy + aligned MACC-Lite | 73.420 ± 0.512 | 1.020 ± 0.014 | 0.058 ± 0.006 |
| ResNet-50 | Entropy + aligned tuned fixed | 73.608 ± 0.474 | 1.010 ± 0.019 | 0.054 ± 0.006 |
| ResNet-50 | Entropy + aligned tuned schedule | 73.022 ± 0.371 | 1.052 ± 0.019 | 0.064 ± 0.004 |
| ResNet-18 | Entropy | 71.252 ± 0.397 | 1.197 ± 0.022 | 0.095 ± 0.004 |
| ResNet-18 | Entropy + aligned MACC-Lite | 73.086 ± 0.397 | 1.091 ± 0.018 | 0.082 ± 0.003 |
| ResNet-18 | Entropy + aligned tuned fixed | 73.420 ± 0.673 | 1.086 ± 0.024 | 0.079 ± 0.006 |
| ResNet-18 | Entropy + aligned tuned schedule | 72.762 ± 0.483 | 1.113 ± 0.025 | 0.084 ± 0.004 |

All six primary contrasts form one Holm family across both backbones:

| Backbone | Lite minus comparator | Effect (pp) | 95% interval (pp) | Holm t | Holm exact |
|---|---|---:|---|---:|---:|
| ResNet-50 | Entropy | +1.742 | [+1.354, +2.130] | 0.0000157 | 0.01172 |
| ResNet-50 | Tuned fixed | −0.188 | [−0.529, +0.153] | 0.2436 | 0.25781 |
| ResNet-50 | Tuned schedule | +0.398 | [+0.073, +0.723] | 0.0650 | 0.08203 |
| ResNet-18 | Entropy | +1.834 | [+1.558, +2.110] | 0.000000674 | 0.01172 |
| ResNet-18 | Tuned fixed | −0.334 | [−0.741, +0.073] | 0.1929 | 0.20312 |
| ResNet-18 | Tuned schedule | +0.324 | [+0.155, +0.493] | 0.007541 | 0.03125 |

Lite exceeds entropy in every paired seed on each backbone, and both contrasts survive both adjusted test families. The tuned constant has the highest observed mean on both backbones. The Lite-versus-fixed intervals permit a disadvantage and a small advantage for Lite; neither direction of superiority, equivalence, nor non-inferiority is established. Only the ResNet-18 Lite-versus-schedule comparison survives adjustment. Both the selected fixed control and Lite transfer without backbone-specific retuning, so this study does not show that feedback saves control-selection effort.

The benefit is budget dependent. On ResNet-50, Lite minus entropy is −0.362 and −0.108 pp at 20% and 30% annotations, then +0.850, +1.552, and +1.742 pp at 40%, 50%, and 60%. ResNet-18 differences are +0.150, +0.882, +1.750, +2.040, and +1.834 pp. Intermediate endpoints are descriptive, not additional hypothesis tests or a universal annotation-saving rate. [Learning curves](../results/2026-10-02-feedback/learning_curves.csv) preserve all 400 post-acquisition outcomes.

## Classification: earlier six-seed comparisons

Seeds 801–806 evaluate 12 conditions under the earlier ResNet-50 protocol. The initial nine conditions and three six-run follow-ups total 72 runs; the follow-ups reuse the original initializations and paired seeds. Geometric acquisition conditions below use the CÆNL acquisition score. The fixed coefficient in these comparisons is 0.03, distinct from the later independently selected 0.10.

| Treatment | Clean top-1 (%) | Standard AutoAttack 8/255 (%) |
|---|---:|---:|
| Random | 69.293 ± 0.334 | 0.013 ± 0.010 |
| Entropy | 71.463 ± 0.320 | 0.010 ± 0.011 |
| BADGE | 71.110 ± 0.467 | 0.003 ± 0.008 |
| Noise Stability | 71.023 ± 0.249 | 0.003 ± 0.008 |
| Geometric acquisition only | 70.190 ± 0.240 | 0.013 ± 0.010 |
| Random + aligned fixed DCR | 70.113 ± 0.452 | 0.017 ± 0.008 |
| Geometric acquisition + aligned fixed DCR | 71.040 ± 0.427 | 0.017 ± 0.008 |
| Geometric acquisition + aligned Lite | 72.120 ± 0.490 | 0.007 ± 0.010 |
| Geometric acquisition + aligned Full MACC | 70.803 ± 0.318 | 0.007 ± 0.010 |
| Entropy + aligned Lite | 73.360 ± 0.208 | 0.003 ± 0.008 |
| UHerding-Margin adaptation | 71.747 ± 0.626 | 0.007 ± 0.010 |
| Entropy + aligned fixed DCR | 72.263 ± 0.381 | 0.000 ± 0.000 |

The initial seven-contrast clean-accuracy family is retained separately:

| Target minus reference | Effect (pp) | 95% interval (pp) | Holm t | Holm exact |
|---|---:|---|---:|---:|
| Geometric + fixed minus geometric acquisition only | +0.850 | [+0.430, +1.270] | 0.0208 | 0.2188 |
| Geometric + Lite minus geometric + fixed | +1.080 | [+0.447, +1.713] | 0.0309 | 0.2188 |
| Geometric + Full minus geometric + fixed | −0.237 | [−0.714, +0.241] | 0.2589 | 0.2813 |
| Geometric + Lite minus entropy | +0.657 | [+0.015, +1.299] | 0.0931 | 0.2188 |
| Geometric + Full minus entropy | −0.660 | [−0.869, −0.451] | 0.0032 | 0.2188 |
| Geometric acquisition only minus random | +0.897 | [+0.389, +1.405] | 0.0309 | 0.2188 |
| Random + fixed minus random | +0.820 | [+0.297, +1.343] | 0.0309 | 0.2188 |

Each follow-up has its own two-contrast Holm family:

| Family | Target minus reference | Effect (pp) | 95% interval (pp) | Holm t | Holm exact |
|---|---|---:|---|---:|---:|
| Acquisition-rule follow-up | Entropy + Lite minus entropy | +1.897 | [+1.625, +2.169] | 0.0000198 | 0.0625 |
| Acquisition-rule follow-up | Entropy + Lite minus geometric + Lite | +1.240 | [+0.707, +1.773] | 0.001875 | 0.0625 |
| UHerding follow-up | UHerding-Margin minus entropy | +0.283 | [−0.430, +0.997] | 0.3540 | 0.3125 |
| UHerding follow-up | Entropy + Lite minus UHerding-Margin | +1.613 | [+0.957, +2.270] | 0.002923 | 0.0625 |
| Fixed-strength follow-up | Entropy + fixed minus entropy | +0.800 | [+0.498, +1.102] | 0.001442 | 0.0625 |
| Fixed-strength follow-up | Entropy + Lite minus entropy + fixed | +1.097 | [+0.714, +1.479] | 0.001442 | 0.0625 |

These comparisons support favorable clean-accuracy directions for the aligned treatment under the specified controls, subject to the six-pair inference qualifications. The entropy-plus-fixed follow-up was specified after entropy and Lite outcomes were known and is not fresh-seed confirmation. Later acquisitions depend on model predictions and can differ between conditions: these earlier comparisons assess complete active-learning pipelines, not a common-label causal decomposition. UHerding uses the documented fixed-feature/selection-domain adaptation, so this is not a claim about an unmodified external benchmark implementation. Full MACC is active but has no established additional benefit. Geometric acquisition alone is below entropy in mean accuracy. These earlier experiments do not include an equal-performance full-supervision comparison establishing a 40% label saving; the new matched-step reference above also does not establish such a saving.

## Geometry, feedback dynamics, and adversarial evaluation

For the earlier classification runs, the all-class assessment statistic averages layer3/layer4 `log((W + epsilon)/(B + epsilon))` after normalization. Lower values mean less aggregate within-class scatter relative to between-class scatter.

| Earlier condition | Mean log trace ratio ± SD |
|---|---:|
| Random | 1.3881 ± 0.0197 |
| Entropy | 1.3916 ± 0.0170 |
| Geometric acquisition only | 1.3864 ± 0.0151 |
| Random + aligned fixed | 1.0607 ± 0.0218 |
| Geometric + aligned fixed | 0.9767 ± 0.0200 |
| Geometric + aligned Lite | 0.4937 ± 0.0390 |
| Geometric + aligned Full | 1.0825 ± 0.0982 |
| Entropy + aligned fixed | 1.0027 ± 0.0125 |
| Entropy + aligned Lite | 0.4725 ± 0.0434 |

In the later control study, mean log scatter ratios for entropy, Lite, fixed, and schedule are respectively 1.387, 0.471, 0.322, and 0.991 on ResNet-50, and 1.162, 0.528, 0.406, and 0.802 on ResNet-18. Step-weighted mean Lite strengths for layer3/layer4 are 0.0450/0.0866 and 0.0537/0.0789. The traces confirm the prescribed controller recurrence and changing layer-specific strengths. They do not isolate a causal benefit of adaptation or establish geometry as a mediator. The separation safeguard activates on only 5 layer3 and 102 layer4 minibatch records across the regularized post-initialization traces; its independent contribution is not identified. See [trace diagnostics](../results/2026-10-02-feedback/trace_diagnostics.csv) and [controller windows](../results/2026-10-02-feedback/controller_mean_windows.csv).

Standard AutoAttack at L-infinity radius 8/255 on all 5,000 images gives near-zero empirical robustness across the earlier conditions. Each final run retains zero or one adversarially correct image; mean condition accuracy is approximately 0.000%–0.017%. All seven original robustness comparisons are retained as a separate family:

| Target minus reference | Effect (pp) | 95% interval (pp) | Holm t | Holm exact |
|---|---:|---|---:|---:|
| Geometric + fixed minus geometric acquisition only | +0.0033 | [−0.0125, +0.0191] | 1.0000 | 1.0000 |
| Geometric + Lite minus geometric + fixed | −0.0100 | [−0.0215, +0.0015] | 0.5291 | 1.0000 |
| Geometric + Full minus geometric + fixed | −0.0100 | [−0.0215, +0.0015] | 0.5291 | 1.0000 |
| Geometric + Lite minus entropy | −0.0033 | [−0.0119, +0.0052] | 1.0000 | 1.0000 |
| Geometric + Full minus entropy | −0.0033 | [−0.0191, +0.0125] | 1.0000 | 1.0000 |
| Geometric acquisition only minus random | 0.0000 | [−0.0133, +0.0133] | 1.0000 | 1.0000 |
| Random + fixed minus random | +0.0033 | [−0.0052, +0.0119] | 1.0000 | 1.0000 |

Entropy plus fixed has zero adversarially correct images in every seed; entropy plus Lite has one in seed 802 and zero otherwise. The original two-seed spectral classification pilot also has zero robustness at 8/255. Its selected-checkpoint, post-hoc smaller-radius diagnostics do not establish an overall advantage over random acquisition and are distinct from the aligned confirmation study. No completed result here supports general adversarial robustness or a certified robust radius. Clean geometry does not by itself bound input sensitivity, and these experiments do not identify the mechanism behind the robustness outcome. Predictions on the same images across seeds are not independent new test examples.

## GPT-2 language modeling on C4

Seeds 301–306 fine-tune pretrained GPT-2 (124,439,808 parameters) for 99,999,744 tokens per condition, using 12,207 steps, sequence length 1,024, and eight sequences per step. Training uses one deterministic English C4 shard; feedback/development uses another fixed shard. All 24 frozen final checkpoints are subsequently evaluated on local validation shard 00002, with 1,999,872 next-token predictions each. This second shard was unused in local training/configuration development; absence of pretraining overlap is not established. It adds evaluation data, not training seeds.

Fixed spectral strength is 0.01. The adaptive spectral variants initialize at 0.005 within [0, 0.08]; these are task-specific settings and are not the aligned classification coefficients. NLL is primary. Perplexity is `exp(NLL)` and is not an independent endpoint; token ECE and accuracy are secondary.

| Condition | NLL (nats/token) | Perplexity | Pooled token ECE | Token accuracy |
|---|---:|---:|---:|---:|
| Baseline | 3.384212 ± 0.000239 | 29.49473 ± 0.00705 | 0.012324 ± 0.000426 | 0.380067 ± 0.000123 |
| Fixed spectral DCR | 3.383919 ± 0.000253 | 29.48611 ± 0.00746 | 0.012216 ± 0.000438 | 0.380029 ± 0.000083 |
| Spectral MACC-Lite | 3.386159 ± 0.000220 | 29.55222 ± 0.00651 | 0.013131 ± 0.000396 | 0.379920 ± 0.000098 |
| Spectral Full MACC | 3.384448 ± 0.000462 | 29.50170 ± 0.01362 | 0.012438 ± 0.000584 | 0.380040 ± 0.000111 |

The three NLL contrasts form one endpoint-specific family. Negative effects favor regularization:

| Condition minus baseline | NLL effect | 95% interval | Holm t | Holm exact |
|---|---:|---|---:|---:|
| Fixed spectral DCR | −0.00029250 | [−0.00032813, −0.00025688] | 0.000008848 | 0.09375 |
| Spectral MACC-Lite | +0.00194729 | [+0.00187898, +0.00201560] | 0.00000002690 | 0.09375 |
| Spectral Full MACC | +0.00023624 | [−0.00018778, +0.00066025] | 0.21152 | 0.21875 |

Fixed DCR lowers NLL in all six seeds, but the practical reduction is small: approximately 0.0086% in NLL and 0.0292% in perplexity. It establishes no token-accuracy or ECE benefit. Lite worsens NLL in all six seeds; secondary ECE worsens and accuracy decreases under adjusted paired-t analysis, while the adjusted exact tests retain the resolution limitation. Full is inconclusive. The original-shard fixed NLL difference is −0.00028734, close to the second-shard result. These outcomes do not support universal language-model, calibration, or memorization benefits.

Both controllers execute 488 events per seed. Lite spends much of training at the coefficient cap, and Full executes 428–445 learned-policy updates. Effective ranks increase under regularization, so the observed mechanism is spectral shaping rather than uniformly stronger low-rank collapse. Full secondary contrasts and reconstruction are retained in [reported_contrasts.json](../results/2026-09-26/reported_contrasts.json). See the [C4 experiment map and runbook links](REPRODUCIBILITY.md).

## CIFAR-10 diffusion generation

Confirmation uses seeds 1101–1106 and 50,000 generated images per condition and seed. Six pretrained-reference streams measure sampling variation for one unchanged model. Four FID contrasts form one family; KID, Inception score, and denoising MSE are secondary.

| Condition | FID, lower is better | KID × 1,000, lower is better | Inception score, higher is better |
|---|---:|---:|---:|
| Pretrained reference | 10.512 ± 0.100 | 9.153 ± 0.124 | 8.479 ± 0.039 |
| Unregularized fine-tuning | 8.057 ± 0.048 | 6.605 ± 0.084 | 8.692 ± 0.043 |
| Fixed spectral DCR | 8.213 ± 0.070 | 6.726 ± 0.105 | 8.724 ± 0.049 |
| Spectral MACC-Lite | 8.472 ± 0.100 | 6.939 ± 0.115 | 8.728 ± 0.055 |
| Spectral Full MACC | 8.211 ± 0.116 | 6.744 ± 0.088 | 8.729 ± 0.050 |

| Target minus reference | FID effect | 95% interval | Holm t | Holm exact |
|---|---:|---|---:|---:|
| Baseline minus pretrained reference | −2.45468 | [−2.58346, −2.32589] | 0.0000002677 | 0.125 |
| Fixed DCR minus baseline | +0.15601 | [+0.11357, +0.19845] | 0.0004481 | 0.125 |
| Lite minus baseline | +0.41472 | [+0.31931, +0.51012] | 0.0003004 | 0.125 |
| Full minus baseline | +0.15431 | [+0.06205, +0.24658] | 0.0077204 | 0.125 |

Every regularized condition worsens FID in every seed relative to matched unregularized fine-tuning. All adjusted paired-t tests pass; raw exact p-values are 0.03125 and adjusted values 0.125. Improvements over the pretrained reference are attributable to the complete fine-tuning treatment and do not establish an added regularization benefit. Slightly higher Inception scores do not replace the primary FID outcome. Official-test EMA MSE means are 0.02918539, 0.02918606, 0.02918634, and 0.02918632 for baseline, fixed, Lite, and Full, respectively; these small differences are descriptive.

Lite executes 200 controller events and ends at the 0.05 cap; Full executes 200 events and 183–189 learned updates. Representation ranks change without improved generation. The two-seed, shorter development pilot favored fixed/Full, but this direction does not persist in confirmation. Both training horizon and sample count changed, so the pilot-to-confirmation comparison does not identify the cause. The result concerns this spectral formulation, pretrained model, dataset, and schedule, not all diffusion regularization.

## AudioCaps captioning

Seeds 1301–1306 train four conditions for ten epochs on the fixed available-recording subset. Each run evaluates 866 retained official test clips with five references per clip and generates nonempty captions for all 866. This subset is not the full original benchmark. CIDEr-D is primary; the three comparisons form one family. All metrics retain their native scales.

| Condition | CIDEr-D | BLEU-4 | METEOR | SPICE | SPIDEr |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6736 ± 0.0220 | 0.2401 ± 0.0137 | 0.2328 ± 0.0043 | 0.1664 ± 0.0046 | 0.4200 ± 0.0132 |
| Fixed spectral DCR | 0.6745 ± 0.0185 | 0.2384 ± 0.0114 | 0.2335 ± 0.0022 | 0.1650 ± 0.0041 | 0.4198 ± 0.0110 |
| Spectral MACC-Lite | 0.6668 ± 0.0212 | 0.2378 ± 0.0108 | 0.2320 ± 0.0035 | 0.1664 ± 0.0020 | 0.4166 ± 0.0112 |
| Spectral Full MACC | 0.6703 ± 0.0175 | 0.2382 ± 0.0094 | 0.2319 ± 0.0023 | 0.1659 ± 0.0041 | 0.4181 ± 0.0100 |

| Condition minus baseline | CIDEr-D effect | 95% interval | Holm t | Holm exact |
|---|---:|---|---:|---:|
| Fixed spectral DCR | +0.000896 | [−0.016040, +0.017833] | 1.0000 | 1.0000 |
| Spectral MACC-Lite | −0.006781 | [−0.028950, +0.015389] | 1.0000 | 1.0000 |
| Spectral Full MACC | −0.003329 | [−0.015313, +0.008656] | 1.0000 | 1.0000 |

All intervals span zero. Raw paired-t p-values are 0.8971, 0.4673, and 0.5072; raw exact values are 0.84375, 0.50000, and 0.53125. The findings are inconclusive rather than evidence of equivalence, and do not rely solely on a coarse exact-test resolution. Mean test NLL is 2.026833, 2.029867, 2.033260, and 2.030362 for baseline, fixed, Lite, and Full.

Both adaptive conditions execute 278 events; Full executes 253–266 learned updates. Mismatched-audio development checks give higher NLL than correctly paired audio, demonstrating sensitivity to conditioning in that diagnostic, not added benefit from DCR. The two-epoch development study evaluates 376 development clips; its absolute scores cannot be directly compared with the ten-epoch official-test results to isolate duration. Audio retrieval, audio classification, and other architectures are outside the evaluated scope.

## Computational cost and practical use

Fixed DCR still computes the differentiable representation statistics used by its loss. Lite adds moving statistics and bounded coefficient state updates. The evaluated Lite path also measures held-out cross-entropy at control boundaries, even though that metric does not drive its coefficient action. Full adds policy optimization and uses held-out task feedback. These are training operations; ordinary classifier inference can use logits without the controller. Geometric acquisition between training rounds still requires features and class-statistic fitting.

| Measurement | Completed observation | Measurement boundary |
|---|---|---|
| Earlier classification synchronized training compute | Entropy 5,092.81 s; geometric + fixed 5,238.40 s; geometric + Lite 5,238.85 s; geometric + Full 5,240.13 s | Six-seed means including an allocated shared initialization; excludes later controller observation/policy operations, with feedback and acquisition separately timed |
| Earlier entropy phase duration | Entropy 5,255.02 s; entropy + Lite 5,391.31 s: +136.29 s, or +2.59% | Sum of five post-initialization phases; sequential campaigns, not interleaved timing trials |
| Earlier entropy phase peak allocation | 6,066,439,680 versus 6,066,701,824 bytes: +262,144 bytes (0.25 MiB) | PyTorch phase allocation, not isolated controller memory or total application memory |
| Earlier 104-run control-selection cost | 39.66 h training plus acquisition for 24 candidates; 40.07 h including three shared initializations | Excludes final reporting and other unrecorded overhead; earlier Lite development is additional |
| Earlier 104-run ResNet-50 per-run training plus acquisition | Entropy 1.626 h; Lite 1.669 h; fixed 1.674 h; schedule 1.674 h | Post-initialization mean; all paths record matched holdout/geometry measurements |
| Earlier 104-run ResNet-18 per-run training plus acquisition | Entropy 0.759 h; Lite 0.760 h; fixed 0.764 h; schedule 0.762 h | Same research-monitoring boundary as ResNet-50 |
| GPT-2 microbenchmark | Plain 33.5293 ± 0.0429 s; sampled monitoring 33.7633 ± 0.0731 s; paired increase 0.698%, SD 0.142 pp; about +0.089 GiB allocation | Five randomized-order pairs of 250 measured steps after 25 warmup steps; excludes loading/initialization; BF16 parameters and batch/accumulation differ from production |
| C4 production successful training-loop time | Baseline 2,332.71 s; fixed 2,516.53 s (+7.88%); Lite 2,511.91 s (+7.68%); Full 2,720.48 s (+16.62%) | Ratios of six-seed group mean successful training-loop times against the monitored baseline, with matched 12,207-step workloads |
| C4 production task-job wall time | Baseline 2,395.42 s; fixed 2,542.26 s (+6.13%); Lite 2,537.28 s (+5.92%); Full 2,743.34 s (+14.52%) | Broader task-driver job boundary; ratios of group means against the monitored baseline; all six seeds retained |
| Diffusion task-job wall time | Baseline 2,919.80 s; fixed 2,974.48 s; Lite 2,978.48 s; Full 2,977.43 s | Includes driver training, sampling, metrics; excludes later official-test EMA assessment and preparation |
| Audio successful training-loop time | Baseline 1,175.53–1,201.80 s; fixed 1,308.80–1,333.21 s; Lite 1,309.24–1,337.74 s; Full 1,336.39–1,375.73 s | Includes feedback/checkpoints; excludes setup, final decoding/metrics, audits and reused feature extraction |

The [cost-analysis workflow](../analysis/cost/README.md) reconstructs the later feedback-study timers and both GPT-2/C4 timing boundaries from the portable [cost records](../results/2026-10-12-cost/). Its checked reference reconstruction contains 105 numerical comparisons, with input hashes, workload and seed-set consistency checks. The earlier classification, fixed-entropy, diffusion, and audio cost summaries above are transcribed from completed source-export numerical summaries; this cost workflow does not independently reconstruct those costs from raw records. C4 successful training-loop and task-job percentages describe different boundaries, and neither can be combined with the separate microbenchmark to infer total overhead relative to plain production training.

The original nine-condition classification peak is 6,075,090,432 allocated bytes for every conceptual run, potentially dominated by shared initialization. Equal peaks do not show zero additional controller memory. The separately collected entropy-plus-fixed follow-up lacks a matching phase-duration/phase-peak aggregate, so it supplies no new comparable overhead ratio. Diffusion pretrained-reference sampling takes 1,616.81 s but performs no training updates and is not an overhead denominator.

These measurements do not establish a universal sub-10% overhead, lower total deployment cost, or tuning-free adaptation. Baselines in the research protocols also record representation/holdout measurements, so their timers do not measure a lean fixed-weight implementation. The distinction between selected-run cost, control-search cost, and earlier method development matters when choosing a deployment.

When representative labeled feedback data allow coefficient selection, tuned fixed DCR is the simplest regularized reference supported by the completed classification comparisons. Evaluate it alongside entropy alone before adding a controller. Lite supplies an explicit bounded feedback alternative whose coefficients vary with the measured geometry; assess whether this response benefits the intended budget, architecture, and changing data conditions. The present evidence does not justify recommending it generally over tuned fixed strength. Full MACC adds policy training without an established additional task benefit under the tested settings. The adverse diffusion and inconclusive audio outcomes warrant task-specific baseline evaluation rather than transferring a classification recommendation across tasks.

## Dataset sensitivity and verification boundaries

The earlier studies' frozen ImageNet-100 manifest contains five exact training/validation SHA-256 matches with conflicting labels, 51 duplicate-content pairs within training, and 43 within-training conflicting-label pairs. Seven of the 23 feedback-study holdouts contain one or two examples duplicated elsewhere in training; no tuning holdout is affected. Index-disjoint splits therefore do not guarantee content-disjoint splits.

The earlier feedback study's primary endpoint retains all 5,000 images. A post-hoc sensitivity analysis excludes the same five validation indices for every treatment, retaining 4,995. Lite-minus-entropy is +1.742/+1.836 pp on ResNet-50/ResNet-18, and Lite-minus-fixed is −0.188/−0.328 pp. All six adjusted-test decisions and selected controls remain unchanged. Evaluation-time exclusion does not undo training exposure or create a deduplicated training benchmark. The released [provenance](../results/2026-10-02-feedback/provenance.json) preserves the indices and overlap information. The separate 206-run study instead applies its documented prospective exclusions before all new partitions and training.

The public snapshots contain numerical metrics, comparisons, timing and trace summaries, source identities, and checksums. Licensed dataset bytes, large checkpoints, corpus/reference-caption text, generated images, and complete native prediction archives are not bundled. Statistical reconstruction checks released-record arithmetic and hashes; it is distinct from rerunning training or inference. The [provenance guide](PROVENANCE.md) documents upstream saved-array audits, source correspondence, and reproducibility limits, including accepted audio initial-state hashes without a guarantee that all corresponding binary states are publicly available.

## Recompute the released statistics on CPU

Use Python 3.12 from the repository root. No GPU or licensed datasets are needed. Choose new output directories; the analysis scripts refuse existing paths.

```bash
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python3 -m pip install -r analysis/requirements.txt
python3 analysis/reproduce.py --output reproduced-results
python3 -m pip install -r analysis/feedback/requirements.txt
python3 analysis/feedback/reproduce.py --output reproduced-feedback
python3 analysis/cost/reproduce.py --output reproduced-cost
python3 results/caenl-solidification-v1/reconstruct_stats.py
python3 analysis/feedback/plot.py --output feedback-plots
```

The first reconstruction checks the earlier snapshot and rebuilds 39 declared paired contrasts, their families, means/SDs, learning curves, and geometry rows. The feedback reconstruction verifies its eight numerical inputs and rebuilds the six primary contrasts, duplicate-exclusion sensitivity, selection rankings, and verification output. The standard-library cost reconstruction verifies its released input hashes and rebuilds feedback stage allocation, paired C4 microbenchmark measurements, and production successful-loop/job timing summaries. The separate solidification reconstruction checks 206 endpoint records, selections, counts and budgets, and rebuilds all 17 planned contrasts. Plots use measured curves and controller-window summaries. These workflows do not analyze unrelated campaign directories or count incomplete runs as findings.

For training entry points and prerequisites, use the [experiment map](REPRODUCIBILITY.md), [GPU deployment guide](RUN_ON_IBM.md), [feedback analysis guide](../analysis/feedback/README.md), and [cost-analysis guide](../analysis/cost/README.md). Record the Git commit and resolved environment before execution; keep a fixed source checkout when resuming checkpoints. Retrieve datasets and pretrained assets under their applicable terms; original accepted checkpoint states are not all supplied by this repository. A successful CPU reconstruction is not a claim of independently repeated GPU training.


## Completed frozen-checkpoint robustness and cost diagnostics

The separate diagnostic protocol is complete: entropy, tuned fixed and Lite, ten final ResNet-50 seeds 62001–62010, all 4,995 content-filtered validation images, five standard AutoAttack radii (150 suites). No training protocol or checkpoint was changed. All three primary effects at 2/255 have adjusted paired-t p=0.90699 and adjusted exact p=1; confidence intervals below are individual unadjusted 95% paired-t intervals.

| Target minus reference | Robust top-1 difference (pp) | 95% interval |
|---|---:|---|
| Lite − entropy | −0.01001 | [−0.03805, +0.01803] |
| Fixed − entropy | +0.00801 | [−0.03637, +0.05238] |
| Lite − fixed | −0.01802 | [−0.05527, +0.01924] |

| Radius | Entropy robust accuracy (%) | Fixed (%) | Lite (%) |
|---|---:|---:|---:|
| 0.5/255 | 22.30030 | 22.28028 | 22.19419 |
| 1/255 | 4.04004 | 3.98198 | 3.95596 |
| 2/255 (primary) | 0.31231 | 0.32032 | 0.30230 |
| 4/255 | 0.05806 | 0.06607 | 0.06006 |
| 8/255 | 0.01401 | 0.00801 | 0.01001 |

Other radii and the 15 common-clean-correct contrasts are descriptive. Every individual interval for the latter spans zero. There is no established primary robustness advantage or equivalence. Similar descriptive results at 0.5/255 address the objection that only the 8/255 floor was examined. Clean scatter decreases without a demonstrated robustness gain. Returned-output geometry mixes unchanged initially wrong/surviving inputs with successful perturbations from heterogeneous attacks; local gradient, margin and displacement summaries do not establish causal mediation or a certified radius. Only ResNet-50 checkpoints were attacked in this campaign.

Five repetitions of each 20-warm-up/200-measured-step cached-data path give mean elapsed seconds:

| Path | Entropy | Fixed | Lite |
|---|---:|---:|---:|
| Streamlined | 23.7322 | 24.7621 | 26.1157 |
| Instrumented | 25.5770 | 26.2770 | 26.2909 |

Mean paired overhead is 4.34% for streamlined fixed versus entropy, 5.47% for Lite versus fixed and 10.04% for Lite versus entropy. Instrumented Lite versus fixed is 0.05%. Shared monitoring changes the timing boundary. Timing repetitions are not training seeds; caching/setup, serialization, acquisition and attacks are excluded. The completed fixed, schedule and contrastive selection grids record 19.57, 19.60 and 19.48 training-plus-acquisition hours; Lite's prior development cost is unknown. These observations establish neither full-job efficiency nor lifetime tuning savings.

## Separate raw-source AudioCaps follow-up

All 58 runs completed: six fixed-strength candidates × three tuning seeds, then four conditions × ten paired confirmation seeds. The fixed audit retains 49,025 training, 487 validation and 956 official-test clips (about 98% coverage per split), with no retained all-zero clips. The earlier 24-run/866-test-clip study remains separate. Frozen AST features are newly extracted; every method shares the saved full initial model for its seed. Development-only selection chooses fixed strength 0.001 before confirmation; official-test captions/NLL do not enter tuning.

| Condition | CIDEr-D mean ± seed SD | SPIDEr mean | Test token NLL mean |
|---|---:|---:|---:|
| Baseline | 0.689205 ± 0.010920 | 0.430446 | 1.994744 |
| Tuned fixed | 0.686467 ± 0.012286 | 0.428578 | 1.994845 |
| Lite | 0.689326 ± 0.012525 | 0.430622 | 1.999819 |
| Full | 0.678143 ± 0.016335 | 0.424531 | 1.997075 |

CIDEr-D is primary. All five planned contrasts belong to one exact-sign-flip Holm family; a separate paired-t family is sensitivity only, never a second route to success. Intervals are individual and unadjusted.

| Contrast | Difference | 95% interval | Holm exact p | Holm t p |
|---|---:|---|---:|---:|
| Fixed − baseline | −0.002738 | [−0.011768, +0.006293] | 1 | 1 |
| Lite − baseline | +0.000121 | [−0.010363, +0.010606] | 1 | 1 |
| Full − baseline | −0.011062 | [−0.020274, −0.001851] | 0.136719 | 0.118677 |
| Lite − fixed | +0.002859 | [−0.008740, +0.014458] | 1 | 1 |
| Full − fixed | −0.008324 | [−0.019602, +0.002953] | 0.5 | 0.517169 |

None rejects after primary multiplicity correction. Full is numerically lower; its raw exact p=0.027344 and individual interval do not establish an adjusted-significant harm claim. No adaptive benefit, fixed benefit or equivalence is established. BLEU, SPIDEr and NLL are descriptive and cannot replace the endpoint. Job-wall means are approximately 55.01/57.39/57.34/57.97 minutes for baseline/fixed/Lite/Full, with 16.71 hours across tuning jobs; extraction, initialization and separate audit phases are outside those job means.

Wider coverage and tuned fixed control address limitations of the earlier comparison, while prior benchmark exposure, possible AST pretraining overlap, acoustic near-duplicates and upstream authenticity remain unresolved. This is spectral audio regularization, not the classification common-label scatter ablation or a test of one unchanged objective across modalities. No video result is claimed.

The [new CPU reconstruction](../analysis/completed/) verifies every new primary contrast, all 15 descriptive intersection contrasts, timing summaries and all audio outcomes. [Diagnostic records](../results/caenl-diagnostics-v1/) and [raw-source audio records](../results/caenl-audiocaps-raw-v1/) retain negative and inconclusive measurements. Statistical reconstruction remains distinct from independent training, attack generation or caption regeneration.
