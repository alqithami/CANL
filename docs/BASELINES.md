# Baseline provenance

Every acquisition baseline in the campaign runs under **identical settings**: the same
architecture, initial checkpoint (shared per seed and regulariser group), class-balanced seed
set, candidate passes (50k candidates per pass, identical pass order per seed), training
schedule, evaluation subsets and AutoAttack protocol.  Only the scoring/selection rule differs.
This file records, for every baseline, the source it follows, whether the implementation is a
pinned external package or an independent reimplementation, every deviation from the source, and
the hyper-parameters used (`configs/protocol/revision_protocol.yaml`, `methods:` registry).
All implementations live in `src/caenl/vision/acquisition.py` and are covered by
`tests/test_vision.py`.

No baseline uses an external active-learning library: the published code bases (e.g. the BADGE
and CoreSet repositories) are tied to their own training loops, which would break the
identical-settings requirement.  Each rule below is therefore an *independent reimplementation
of the published algorithm*, kept deliberately small so that it can be audited line by line.

| Method key | Paper | Implementation | Notes / deviations |
|---|---|---|---|
| `random` | — | uniform sample of the candidate pass (seeded `numpy.random.Generator`) | reference lower bound |
| `entropy` | Settles (2009), *Active Learning Literature Survey* | softmax entropy, top-*b* | — |
| `margin` | Scheffer et al. (2001); strong-baseline evidence in Bahri et al. (2022) | smallest top-2 probability gap, top-*b* | — |
| `least_confidence` | Lewis & Gale (1994) | 1 − max probability, top-*b* | — |
| `power_margin` | Kirsch, Farquhar, Mukhoti, Gal (TMLR 2023), *Stochastic Batch Acquisition* | Gumbel-top-*k* on β·log(margin score), β = 1 | "PowerMargin" variant of the paper's stochastic acquisition; scores are the margin scores above |
| `coreset` | Sener & Savarese (ICLR 2018), *Active Learning for CNNs: A Core-Set Approach* | greedy k-Center (farthest-first) on penultimate features; distances are initialised to the current labelled set | the MIP refinement of the paper is not used (the greedy 2-approximation is what every subsequent comparison uses) |
| `badge` | Ash, Zhang, Krishnamurthy, Langford, Agarwal (ICLR 2020), *Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds* | k-means++ seeding on the hallucinated last-layer gradient embeddings g = (p − e_ŷ) ⊗ h | embeddings are never materialised: the squared distance ‖g_i − g_j‖² = ‖u_i‖²‖h_i‖² + ‖u_j‖²‖h_j‖² − 2⟨u_i,u_j⟩⟨h_i,h_j⟩ is computed exactly from the factors (necessary for C = 1000, d = 2048); sampling uses a seeded generator |
| `bald_mcd` | Gal, Islam, Ghahramani (ICML 2017), *Deep Bayesian Active Learning with Image Data* | BALD = H[E p] − E H[p] with T = 20 Monte-Carlo dropout masks (p = 0.3) applied to the penultimate features before the linear head | the backbone has no dropout layers, so the stochasticity is injected at the penultimate layer only (the "last-layer MC dropout" approximation); documented as such in the manuscript |
| `noise_stability` | Li, Yang, Gu, Zhan, Wang, Xu, Xu (AAAI 2024), *Deep Active Learning with Noise Stability* (arXiv:2205.13340) | Algorithm 1: K random unit directions u⁽ᵏ⁾ over the full parameter vector, perturbation Δθ = ζ‖θ‖₂ u⁽ᵏ⁾ with ζ = 10⁻³, deviation of the softmax output under each perturbation, concatenated into the uncertainty vector, greedy k-Center selection | **independent reimplementation** (the authors announced but, at the time of writing, had not released code that we could pin); the constant √(n/K) of the paper is omitted because it does not change k-Center selections; K = 30 on CIFAR, K = 10 on ImageNet (the paper reports K = 10 as "very competitive" for large datasets) |
| `collapse` (CÆNL) | this manuscript, Eq. 7–8 | layer-weighted (α_l = 1/d_l) diagonal Mahalanobis distance to the predicted-class centre of the monitored layers, class statistics recomputed from the labelled set at the end of every phase, quantile threshold τ (0.90; Full MACC chooses τ ∈ {0.80, 0.85, 0.90, 0.95}) | `mahalanobis_mode: full` (shrunk full covariance) is available; the variance floor is max(ε, 0.01·mean variance) |

## Regulariser arms

| Method key | Acquisition | Regulariser |
|---|---|---|
| `caenl_acq_only` | collapse | none |
| `dcr_fixed_random` | random | fixed DCR, λ = 0.03 |
| `dcr_lite_random` | random | MACC-Lite |
| `caenl_fixed_dcr` | collapse | fixed DCR, λ = 0.03 |
| `caenl_macc_lite` | collapse | MACC-Lite (projected proportional feedback on log κ/κ*, step 0.02, λ ∈ [0, 0.5]) |
| `caenl_full_macc` | collapse | Full MACC (contextual-bandit policy: λ deltas, layer flags, τ; acquisition action rewarded after a pre-registered number of post-acquisition steps) |
| `entropy_macc_lite` | entropy | MACC-Lite (isolates the regulariser's effect under a standard acquisition rule) |
| `supervised_full` | — (100 % labels) | none |

`supervised_full` trains on every training image except the controller-validation carve-out
(the same stratified carve-out that every active-learning method pays for from its budget), so
its "labels available" and "labels trained" counts differ by that carve-out; both numbers are
recorded in `summary.json` (`labels_available`, `labels_trained`) and reported in the tables.

## Reference implementations used for validation

* **AutoAttack** — `fra31/auto-attack`, pinned to the commit recorded in `requirements.txt` /
  `requirements.lock`; the standard version (APGD-CE, APGD-T, FAB-T, Square) with inputs in
  [0, 1] and normalisation inside the model.  The production plan contains an `aa_sanity`
  stage (`vision_aa_sanity` job) that evaluates the torchvision ImageNet ResNet-50 checkpoint on
  the same fixed 5,000-image subset and must reproduce ≈ 76 % clean / ≈ 0 % robust accuracy at
  8/255 (the manuscript's sanity check); the GPU smoke exercises the same job on CIFAR-10.
* **Caption metrics** — BLEU-4, ROUGE-L and CIDEr-D are computed natively and cross-checked
  against the official `pycocoevalcap` implementations; METEOR and SPICE come from
  `pycocoevalcap` (Java).  Where both are available the official values are authoritative.
* **FID / KID / IS** — `torch-fidelity` (pinned version), 50k samples, DDIM-100.
