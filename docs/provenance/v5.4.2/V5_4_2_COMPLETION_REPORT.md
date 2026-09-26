# CÆNL v5.4.2 C4 Calibration Package — Completion Report

## Purpose

This package is the audited continuation of the completed C4 pilot-v1. It corrects configuration, validation, storage-root, provenance, and reporting defects identified from the real IBM L40S outputs, and provides a bounded second calibration campaign before confirmatory training.

## Corrected defects

1. MACC-Lite now honours `eta_lambda`; the inherited `controller_step_size` no longer silently overrides the plan.
2. Language DCR and controller targets accept per-layer mappings such as `h6: 16` and `h12: 4`.
3. Full MACC reward components are recorded separately: validation term, query term, geometry term, target deviation, and clipped validation delta.
4. The cross-modal publication validator recognises learned policy updates stored under `history[i].update.policy_loss`.
5. Pilot tables omit significance-star notes when inferential output is suppressed.
6. Pilot summaries no longer describe pilot jobs as “evidence-eligible.”
7. All data/cache/results/work roots are forced to follow `/mnt/caenl/active`, preventing inherited `/dev/shm` variables.
8. The one-dataset launcher refuses pilot or confirmatory execution in `/dev/shm` when `/mnt/caenl` is mounted and writable.
9. Tmux session names no longer acquire an accidental trailing hyphen.
10. Plan fingerprints and full SHA-256 digests are stored under distinct, accurate filenames.
11. Archive export includes both the full digest and the short plan fingerprint.
12. A focused C4 calibration-v2 plan and a locked confirmatory template are included.
13. The obsolete C4 pilot-v1 and confirmatory-v1 campaign IDs are execution-blocked; they remain only for provenance.

## Calibration-v2 matrix

- Seeds: 201, 202.
- Baseline.
- Fixed DCR, target ranks 16/4, lambda 0.005.
- Fixed DCR, target ranks 16/4, lambda 0.01.
- Fixed DCR, target ranks 32/8, lambda 0.005.
- Corrected MACC-Lite, target ranks 16/4, `eta_lambda=0.002`, lambda in [0, 0.08].
- Full MACC, target ranks 16/4, smaller actions, lambda in [0, 0.08], larger controller-validation subset, and reduced geometry-reward weight.

The campaign contains 12 GPU jobs plus preparation and aggregation. It remains a pilot and cannot generate inferential claims.

## Validation performed

- All Python sources compiled.
- All shell scripts passed `bash -n`.
- Calibration plan validation: 14 total jobs, no errors or warnings.
- Regression/unit tests: 70 passed, 4 dependency-specific tests skipped in this CPU environment.
- Existing C4 pilot-v1 archive was reprocessed with strict reporting successfully.
- Custom checks verify MACC-Lite step-size precedence, per-layer rank propagation, nested Full MACC update validation, decomposed reward accounting, mounted-volume root enforcement, and plan checksum semantics.

## Environment-dependent validation still required on IBM

Before starting calibration-v2, the IBM upgrade script must verify:

- CÆNL version 5.4.2.
- CUDA availability.
- NVIDIA L40S detection.
- The mounted-volume roots.
- The full selected IBM test set.

The package does not claim that calibration-v2 results are known in advance. It only establishes that the planned campaign and corrected code are ready for the IBM validation and execution gate.
