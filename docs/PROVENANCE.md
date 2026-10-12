# Evidence provenance and verification scope

The earlier public numeric records are extracted from the completed server reports whose SHA-256 values are recorded in `provenance/source_report_hashes.json`. The later feedback and cost snapshots contain their own `provenance.json` and `SHA256SUMS.json`. Those digests identify source exports; they do not substitute for access to raw data or model states.

Before release, the source exports passed an offline verification of 228 embedded payloads, 13 entropy-follow-up payloads, 31 C4 manifest entries, and 54 current-project Python source digests. Byte reconstructions account explicitly for CSV CRLF normalization, terminal newlines, source-file UTF-8 BOMs, and NaN-to-null serialization of zero-update diffusion reference geometry. The paired statistics were independently reconstructed from the exported per-seed metrics. The released public analysis repeats the statistical calculation from numeric-only extracts and checks their release hashes.

`code_sources.json` maps recovered launcher/engine files to the report and exact code digest. `current_project_sources.json` identifies the separately exported current project copies. A current source snapshot is not automatically proof that every earlier run executed those bytes. The historical C4 source correspondence and other protocol bindings remain documented in [recorded experiment provenance](../provenance/recorded_provenance.md). `C4_recorded_settings.md` contains the resolved C4 configuration evidence.

The original server audits report saved-array reconstruction for classification and caption metric reconstruction for audio. The released offline verification does not regenerate adversarial examples, model predictions, diffusion images, or caption text. It therefore does not establish full raw-data/checkpoint replication.

Audio initial-state acceptance required a whole-model hash match before any training, with a bounded same-seed retry for an intermittent positional-parameter initialization mismatch. No training result or score selected the accepted attempt. Accepted hashes alone cannot recreate missing state tensors, and availability of every accepted initial binary is not established. Accepted hashes do not supply the missing state tensors.

Dataset sources, resolved settings, and subset identities are recorded in the campaign protocols under `provenance/protocols/`. They do not establish an independent audit of pretraining overlap or cross-dataset comparability. Raw ImageNet images, C4 corpus text, audio, reference-caption text and large checkpoints are excluded from this public release. Source files retain their applicable third-party notices and licensing terms.

## Completed feedback-study records

The completed 104-run feedback study has a separate native-evidence audit and portable numerical release. The upstream audit checked all 104 method records, 80 confirmation/transfer validation arrays, 543 training traces covering 4,155,880 optimizer steps, 520 acquisition receipts, and 40,872 controller windows. Its recorded evaluation reconstruction and controller diagnostics are preserved in `results/2026-10-02-feedback/`. The [CPU analysis](../analysis/feedback/README.md) repeats the six primary paired comparisons, control-selection rankings, and duplicate-exclusion sensitivity from the released numerical inputs.

The original dataset audit found five exact training/validation content matches with conflicting labels. The primary analysis retains all 5,000 validation images; a separate sensitivity analysis excludes the same five indices from every treatment. This checks evaluation sensitivity rather than providing a deduplicated training experiment. Index-disjoint holdouts may still contain content duplicates. Dataset and source overlap limits remain part of the findings.

## Timing and recovered C4 utilities

`results/2026-10-12-cost/` contains portable records of completed feedback and C4 timings. The release preserves the feedback method timings, shared-initialization summaries, and C4 microbenchmark observations; it projects production-job and successful training-loop fields without changing their numerical values. Each training-loop row retains its source-summary SHA-256. The [cost reconstruction](../analysis/cost/README.md) verifies nine released input digests and 105 numerical consistency checks.

C4 training-loop time and broader task-job wall time use different measurement boundaries. Their percentages are reconstructed separately; neither is combined with the monitoring microbenchmark to infer a plain-training production overhead. The cost reconstruction does not cover the raw timing records of earlier classification, diffusion, or audio runs.

Recovered C4 executable sources are accompanied by per-file digests in their tool directories. The holdout evaluator's Python digest matches the source digest recorded in its original execution metadata. For the monitoring benchmark, the recovered source identity is recorded separately from the original numerical configuration; an original execution-source digest has not been established. Source recovery alone is not a claim of a new independent GPU replication.

No in-progress solidification or robustness-diagnostics results contribute to these completed snapshots.
