# PROTOCOL — what is run, what changed relative to the submitted manuscript, and why

Authoritative file: `configs/protocol/revision_protocol.yaml` (frozen with SHA-256 into every campaign).
Entries marked **[paper]** are unchanged from FRAOPE-D-25-02381; **[revision]** entries are additions or
clarifications that the revised manuscript must state explicitly.

## 1. Benchmarks

| Role | Dataset / model | Seeds | Methods | Purpose |
|---|---|---|---|---|
| **Primary** | ImageNet-1K (1,281,167 / 50,000; 1,000 classes), ResNet-50 @ 224 px | **6** (paired) | full supervision, random, entropy, margin, Power-Margin, CoreSet, BADGE, BALD-MC, Noise Stability, CÆNL acq-only, random + fixed DCR, random + MACC-Lite, CÆNL + fixed DCR, CÆNL + MACC-Lite, CÆNL + Full MACC | every ImageNet claim of the manuscript: label efficiency, robustness/accuracy trade-off, ablations, statistics |
| Primary (confirmation) | ImageNet-1K, ViT-B/16 @ 224 px, **label-free MAE initialisation** (`facebook/vit-mae-base`, pre-trained on ImageNet-1K images without labels) | 3 (paired) | full supervision, random, entropy, BADGE, CÆNL acq-only, CÆNL + MACC-Lite, CÆNL + Full MACC | architecture transfer at a realistic accuracy level |
| Primary (confirmation) | ImageNet-1K, ViT-B/16 @ 224 px, **from scratch** | 3 (paired) | same arms | strictest reading of the manuscript (no pre-training at all) |
| Supporting | CIFAR-10 / CIFAR-100, ResNet-18 | 5 | 15 / 13 + full supervision + λ-sweep + PGD-AT references | cheap multi-seed ablations, Pareto, sensitivity |
| Supporting | ImageNet-100 (126,689 / 5,000), ResNet-50 @ 128 px | 3 | 10 + full supervision | intermediate-scale check |
| Extensions | C4 / GPT-2, CIFAR-10 DDPM, **AudioCaps** captioning + retrieval (Clotho supplementary) | 3 | baseline, fixed DCR, MACC-Lite, Full MACC | cross-modal claims |

**[revision]** The submitted manuscript reported single-seed ImageNet-1K numbers. The revision
re-runs ImageNet-1K with six paired seeds under one frozen schedule (plan
`configs/plans/franklin_complete_8gpu.yaml`, ≈ 975 H100-hours for everything; the ResNet-50
programme alone ≈ 445 H100-hours). CIFAR/ImageNet-100 results are presented as supporting
evidence only; they must not be used to back ImageNet claims.

**[revision] ViT-B/16 initialisation.** The manuscript does not state how its ViT was initialised.
Any ImageNet-1K *supervised* checkpoint would leak the labels the active learner must acquire, so
two programmes are run: (i) the encoder of MAE (He et al., CVPR 2022; `facebook/vit-mae-base`,
pre-trained on ImageNet-1K images **without labels**), converted tensor-for-tensor into the torchvision
ViT-B/16 (equivalence test in `tests/test_vision.py`) with a fresh classifier head and the MAE
fine-tuning recipe (AdamW, base lr 5e-4 per 256, layer-wise lr decay 0.65, wd 0.05, 3 warm-up epochs,
30 initial epochs, 20 per round, 50 for full supervision); (ii) random initialisation with the DeiT
recipe (AdamW lr 2.5e-4, wd 0.05, grad-clip 1.0, label smoothing 0.1, 60 initial epochs, 20 per
round, 90 for full supervision). The manuscript must report both and state that (i) uses label-free
pre-training.

**[revision] ImageNet-1K schedule.** Initial phase 60 epochs on the 10 % seed set, then 20 epochs after
every round (bs 256, SGD + Nesterov, lr 0.1, wd 1e-4, cosine with 2-epoch warm-up, round LR factor 0.5),
RandomResizedCrop-224 (scale 0.25–1) + flip from a 256 px cache, centre-224 evaluation; full supervision
90 epochs. The manuscript's batch size of 1,024 across 8 GPUs is replaced by the single-GPU torchvision
reference recipe (bs 256, lr 0.1); ViT-B/16 uses AdamW (lr 2.5e-4 = 5e-4 × 256/512, wd 0.05, 3-epoch
warm-up, grad-clip 1.0, label smoothing 0.1) from scratch — no pretrained weights, since any ImageNet
pretrained initialisation would leak the labels the active learner is supposed to acquire.

## 2. Active-learning protocol (Section 5.3 of the manuscript)

* **[paper]** class-balanced 10 % seed set; 5 rounds adding 10 % of the original training-set size each
  (→ 60 %); candidate passes of 50,000 examples; per-pass selection of the top 10 % (CÆNL: quantile τ = 0.9);
  warm start from the previous round with 20 fine-tuning epochs and a reduced, restarted learning rate.
* **[revision]** a stratified *controller-validation* subset (10 % of the seed set on CIFAR, 5 % on
  ImageNet-100, 2 % on ImageNet-1K) is carved out for **every** method (identical label budget), never
  receives gradients, and is the only validation signal Full MACC sees (the test set is touched only at
  evaluation). Every summary and table therefore distinguishes **labels available** (the budget, carve-out
  included) from **labels trained** (gradient steps); `supervised_full` has 100 % available and trains on
  all but the same carve-out.
* **[revision]** the initial phase (100 epochs CIFAR / 60 epochs ImageNet-100 and ImageNet-1K; not
  specified in the manuscript) is trained **once per seed and regulariser group** and shared by all
  acquisition functions in that group, so methods differ only by what they acquire (paired design).
* **[revision]** candidate passes are consecutive chunks of a per-round shuffle of the pool (no example is
  re-scored before the whole pool has been scored once); selected examples leave the pool immediately; pass
  sizes, caps, coverage and repeat rates are recorded (`summary.json → acquisitions`).
* **[revision]** mixup is disabled so that class-conditional statistics are computed from clean labels;
  standard crop/flip augmentation (RandomResizedCrop at 128 px for ImageNet-100, 224 px for ImageNet-1K).
* Schedules: SGD + Nesterov, cosine with warm-up, bs 256, lr 0.1, wd 5e-4 (CIFAR) / 1e-4 (ImageNet).

## 3. Collapse monitoring, DCR and controllers (Sections 3–4)

* **[paper]** κ_l = tr(Σ_W,l)/tr(Σ_B,l); instability Δκ_l = |κ_l − EMA(κ_l)|; EMA decay 0.99, shrinkage 1e-5;
  monitoring every N = 100 steps; truncated rank k = 128; spectrum-only DCR on normalised eigenvalues.
* **[revision] feasible ETF-like target.** The submitted rank-(C−1) target is not representable when only
  k = 128 eigenvalues are retained (C − 1 = 999 on ImageNet-1K). The implementation uses
  r = min(C − 1, d, k) equal components (CIFAR-10: 9, CIFAR-100: 99, ImageNet-100: 99, ImageNet-1K: 128);
  the rank is additionally capped at n − 1 for a batch of n examples.
* **[revision] covariance used by DCR.** Σ_l is the (unconditional) covariance of the batch activations
  (pooled block outputs), obtained with the Gram trick; class-conditional statistics are used for
  monitoring and acquisition. (Option `collapse.dcr_covariance: between` regularises the batch class-mean
  covariance instead.)
* **[revision] monitored layers** are `layer3` and `layer4` (penultimate) of the ResNets ("mid-to-late
  layers" in the manuscript); ViT taps are class tokens at 1/4, 1/2, 3/4 and the last block.
* **[revision] MACC-Lite error signal** uses the scale-free log-ratio log(κ_l/κ_l*) (the manuscript's
  κ_l − κ_l* is available as `error_mode: difference`); targets are expressed as collapse strength
  s* = 1/(1+κ*): 0.35 (layer3), 0.70 (layer4). λ ∈ [0, 0.5], η = 0.02, λ₀ = 0.03.
* **[paper] Full MACC** is the online contextual bandit of Algorithm 1 (two-layer MLP policy, ε-greedy
  0.20 → 0.02 over the whole run — the schedule is computed from the full plan so that the shared
  initial phase and the active-learning rounds form one consistent episode — REINFORCE with an EMA
  baseline; uniformly explored actions are not used for policy-gradient updates). No meta-pretraining;
  its policy parameters (≈ 20k) are counted as overhead. **[revision] Credit assignment.** The
  *regularisation* heads (λ deltas, layer flags) are rewarded at the next controller step with
  ΔVal − γ Σ|log(κ/κ*)| (ΔVal = 10 × decrease of the controller-validation loss, γ = 0.1), never across a
  phase boundary. The *acquisition* head (τ) is rewarded separately: the controller value is measured
  just before an acquisition round, the τ action is recorded as a pending action with the number of
  queries, and it is rewarded with 10 × (value after `acquisition_reward_steps` post-acquisition
  optimisation steps − value before) − β·n_q (β = 0; `acquisition_reward_steps` = 200 on CIFAR, 1,000 on
  ImageNet-1K, pre-registered in the protocol). Pending actions are serialised in every round checkpoint,
  so an interrupted and resumed run rewards exactly the same actions as an uninterrupted one (unit and
  integration tests). The quantile head is disabled for tasks without acquisition (C4, DDPM, audio).
* **[revision] acquisition score** s(x) = Σ_l α_l u_l(x) with α_l = 1/d_l and diagonal Mahalanobis
  distances to the predicted-class centre (the manuscript's "diagonal approximation"); class statistics
  are re-estimated exactly on the labeled set at acquisition time. Per-class variances receive the
  manuscript's absolute shrinkage (1e-5) plus a relative floor of 1 % of the mean channel variance so that
  dead channels cannot dominate the score (`mahalanobis_mode: full` uses shrunk full covariances instead).
* **[revision] generative / language collapse index** is the effective rank exp(H(p)); the DCR target is the
  rank-r near-isotropic template with r = min(r_cfg, d, k) (GPT-2: 128, DDPM mid-block: 64, audio: 96);
  MACC-Lite uses |log(κ/κ*)| − deadband so λ decays once the target rank is reached.

## 4. Baselines (identical settings)

Random; Entropy; Margin; Least confidence; Power-Margin (stochastic batch acquisition, Kirsch et al. TMLR
2023, β = 1); CoreSet (k-Center greedy, penultimate features); BADGE (k-means++ on gradient embeddings;
exact factorised distances); BALD with Monte-Carlo feature dropout (T = 20, p = 0.3); Noise Stability
(Li et al., AAAI 2024; K = 30 parameter perturbations on CIFAR, 10 on ImageNet, ζ = 1e-3, k-Center on
output deviations; independent reimplementation, see `docs/BASELINES.md`). All share the seed's initial
checkpoint, splits, schedules, candidate passes and evaluation.

CÆNL family (acquisition × regulariser): collapse acquisition only; random + fixed DCR; random +
MACC-Lite; collapse + fixed DCR (λ = 0.03); **collapse + MACC-Lite (the manuscript's CÆNL)**; collapse +
Full MACC; entropy + MACC-Lite; fixed-λ sweep {0.01, 0.1, 0.3, 1, 3, 10}; PGD-10 adversarial-training
references (random / CÆNL); full supervision.

## 5. Evaluation

* Clean accuracy, top-5, NLL, ECE on the full test set after every phase (label fractions 10–60 %).
* **[paper]** AutoAttack `standard` (APGD-CE, APGD-T, FAB-T, Square; `fra31/auto-attack` pinned to commit
  `a3922004`), L∞, ε = 8/255, normalisation inside the model, inputs in [0,1], at the final budget;
  per-example robust masks and the evaluated subset indices saved. **[revision]** CIFAR/ImageNet-100 use the
  full test set; ImageNet-1K uses a fixed, class-stratified 5,000-image validation subset (RobustBench
  ImageNet convention), identical for every method and seed and never a class-sorted prefix. The
  `aa_sanity` job evaluates the torchvision ResNet-50 (IMAGENET1K_V1) under the same protocol and must
  reproduce ≈ 76.1 % clean / ≈ 0 % robust accuracy (manuscript Sec. 6.2.1) before any CÆNL number is
  reported.
* **[revision]** per-round proxies on a fixed 2,000-image subset shared by all methods: APGD-CE (1 restart),
  PGD-20; ε-sweep {1, 2, 4}/255 and a synthetic common-corruption probe at the final budget; logit margins,
  input-gradient norms and empirical local Lipschitz estimates (Proposition 4 diagnostics); NC1–NC4 panel,
  effective rank and spectra of the penultimate layer on the test set; Spearman correlation and top-set
  overlap between the collapse score and entropy/margin on the scored candidates.
* Cross-modal: C4 validation NLL / perplexity / token ECE / effective rank; DDPM FID, KID, IS over 50k
  DDIM-100 samples vs. the CIFAR-10 training set; **AudioCaps** (official captions, pinned commit
  `d004db3e` of `cdjkim/audiocaps`; user-obtained audio with a strictly validated manifest — paths,
  non-empty splits, no clip in two splits, captions present, per-file SHA-256, ≥ 90 % coverage of every
  official split, coverage reported) CIDEr-D, BLEU-4, ROUGE-L, METEOR, SPICE from the official
  `pycocoevalcap` implementation (authoritative; the native BLEU/CIDEr/ROUGE implementations are only
  cross-checks, and a required metric that cannot be computed fails the job instead of producing NaN),
  validation loss; retrieval R@{1,5,10} both directions and median rank; Clotho v2.1 as a supplementary
  corpus. In every extension the small set used for Full MACC's reward (16 C4 sequences, 512 CIFAR-10
  test images with fixed noise, 64/128 validation clips) is disjoint from all reported evaluation sets.
  Retrieval draws one caption per clip per epoch so that a batch never contains two captions of the
  same clip (no in-batch false negatives).
* Overhead: ms/step, images/s, peak GPU memory and extra parameters for baseline / monitoring / fixed DCR
  / MACC-Lite / Full MACC / PGD-AT on the same training code path; acquisition scoring time per strategy;
  per-run training, acquisition and AutoAttack wall time and peak memory.

## 6. Statistics

Paired by seed (same split and initial checkpoint): mean ± std, t-based 95 % CI, paired t-test (primary),
exact sign-flip permutation (cannot go below p = 0.0625 with 5 seeds — stated in the tables), Wilcoxon
(n ≥ 6), bootstrap CI of the mean difference (10,000 replicates), Cohen's d_z, Holm correction within each
(dataset, metric) family; per-example McNemar tests on AutoAttack masks for each seed.

## 7. Reproducibility materials

Per job: `job.json` (fully resolved configuration), `environment.json` (pip freeze, CUDA, GPU),
`events.jsonl`, `metrics.jsonl`, `system.csv`, initial/ctrl-val/selected indices, candidate scores,
test logits, robust masks + AutoAttack subset indices, controller trajectories, checkpoints. Per
campaign: `PLAN.yaml` + SHA-256, frozen protocol + SHA-256, manifest, `STAGE_MANIFEST.json` per stage
(expected matrix, per-job validation, SHA-256 of `job.json` and `summary.json`), `PUBLICATION_CHECK.json`,
`aggregate/all_jobs.csv`. Dependencies are pinned per task group (`requirements/*.txt`, AutoAttack at
an exact commit) and frozen into `requirements.lock` after the GPU smoke; a Dockerfile records the
base image. Seeds for split, initialisation, data order, augmentation, acquisition sub-sampling,
attacks and the controller are derived separately from the run seed and logged.

## 8. Publication gate

A number may appear in the manuscript only if its job passed the per-stage validation (state
`succeeded`, every required metric finite, AutoAttack under the frozen protocol with saved masks,
round count and final label fraction as planned, labels block present; cross-modal metrics finite,
official caption metrics, manifest provenance) **and** the whole campaign passes
`caenl report --strict` (every expected seed × method succeeded, hashes unchanged since validation,
tables/figures/significance present). A stage that fails the gate is marked failed and blocks every
stage that depends on it (the final aggregation included), so no report can be declared
publication-ready while any expected job is missing or invalid.
