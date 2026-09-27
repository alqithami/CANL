# Recorded experiment provenance

Source identities and verification boundaries from the completed historical campaign reports. These records do not describe outcomes of the ongoing feedback study.

## Original classification

The frozen source is `caenl_aligned_imagenet100_v1`. Campaign `caenl-imagenet100-aligned-confirmation-v1` has protocol/source binding `159fe52901f774e7abeb40a3afe96de81f1084a1bde0f63d070c7d2ecda325e1` Archive `imagenet100-aligned-results.tar.gz` has recorded digest `3aea89bd8139c60248d9cfd7b7ba68a1d255bb709fcda88ddc2db0b58b9cec76`

The exported integrity report records 7,861 passing manifest checks without mismatches or unresolved paths. A later IBM-side NumPy arithmetic audit covers 54 method–seed runs across nine conditions, 270 phases, 552 validation/feedback payloads, and 54 attack payloads. Recorded metrics and tests match the reconstructed saved-array arithmetic. It does not regenerate original predictions, rerun attacks, independently reconstruct perturbation norms from images, or rehash every later-phase checkpoint binary.

## Additional classification conditions

Campaign `caenl-focused-completion-v1` adds six entropy-plus-Lite branches and 30 phases, with binding `c6009897fdc2d87df4e257cd89e710bbb2465d6f80a98e4349e36ea01f2af058` Its report preserves the original audit, new outcomes, and the separate two-contrast family. The later `caenl-final-evidence.md` contains 24 C4 job configurations, 54 Python source files, and costs for the first 60 classification method–seed runs.

Campaign `caenl-uherding-v1` adds six adapted UHerding branches and 30 phases, with binding `05bb9cc074b108ff12abdefe68859da9f7a7c02fd9f44075d306066f6b57e545` Its report records passing CPU/integration/CUDA checks, phase and attack-payload arithmetic, and preservation of 111 bound inputs. The campaign report records digest `3ea627d9613ed27315011731962de6051b951a87411007f098529f41b5f8a608` The digest is retained as campaign provenance and is not presented as independent third-party verification of the historical upload.

Campaign `caenl-entropy-fixed-v1` adds the six entropy-plus-fixed runs and 30 phases, with protocol/source/input binding `1df7d40195f4ec15857687acb50014767a685469254b6f499b02f8643a1171fd` The supplied completion report has SHA-256 `a7f0093e3f9ba5ff63c38d4662f01f17016653014365ca16b040a1c71875cb40` It records passing CPU and CUDA checks, preservation of original and Lite inputs, and verification of all 131,689 dataset images. Its embedded launcher matches the bound delivered code. An independent report-level check verifies the report and 13 embedded payload digests, recomputes the two final paired comparisons, and checks 108 budget rows and 216 layer-geometry rows. Those rows cover three entropy treatments, six seeds, six budgets, and two layers for geometry. Recomputed statistics match the recorded values. The report-level check does not independently rerun checkpoint inference or attacks; their saved-array checks remain server-reported. Recovered Lite phase exports introduce no new training.

## Image source

The dataset source is `ILSVRC/imagenet-1k`, revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5` The CMC class-list and image-manifest digests are bound into the protocol. Preparation preserves 131,689 selected image files without resizing or reencoding. Locally recorded checksums are distinguished from upstream checksum certification.

## Language source correspondence

Campaign `caenl-c4-confirmatory-v2` uses `en/c4-train.00000-of-01024.json.gz` for training and `en/c4-validation.00000-of-00008.json.gz` for local feedback/development. The later `caenl-c4-confirmatory-v2-holdout-v1` uses `en/c4-validation.00002-of-00008.json.gz`. The read-only source-correspondence check verifies archive digest `e87436f3e57ccd6a152fc32bd859a24859c07382a3a8ff7167db1402dff76f8b` and matches its historical `src/caenl/language/task.py` manifest entry to `ffc8dbcd12db919f64066aeaccb70cb3bdaf99dc316cc1ee749c08cd3e7937f1` This establishes correspondence for that source file, not reconstruction of the entire runtime or an independent training rerun. The separate unused-shard export has SHA-256 `d8f6b681b89603ff51559b2f65d046b5de394c714c635b0e5247185237fb147d` Its 31 internal manifest entries match their payloads. Recalculation from the 24 exported checkpoint-level metric records reproduces the paired NLL effects and test values. This verifies exported metric arithmetic, not regeneration of token predictions from the model checkpoints.

## Diffusion

Campaign `caenl-diffusion-confirm-v1` is separate from `caenl-crossmodal-pilots-v1`. Its 24 trained runs and six references use seeds 1101–1106, excluding development seeds 1001 and 1002. The protocol binding is `eebc9142829c89141d942e781ff31c905ccb473017ecfd1b618833246f3d5d8b` The campaign record reports completion on 22 September 2026, passing implementation/configuration checks, all per-seed summaries, and preservation of 153 bound pilot inputs. The cache-path repair precedes scientific jobs and does not change training settings. Report inspection is not independent regeneration of images or recomputation of FID/KID from saved image arrays.

## Audio

Campaign `caenl-audiocaps-confirm-v1` uses seeds 1301–1306, separate from two-epoch pilot seeds 1201 and 1202. The mirror revision is `b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded` and the original annotation revision is `d004db3ea1b01cf4fd0347dd8d27db90cadc8809` The disjoint-subset manifest digest is `6a19d868feb14ab49ea2a0fba29e2ffa2e6c4b26f0335831db13a8ab7da915b1` The confirmation binding is `f9a511239012548a17b74c6cc936258e91b1f856055424d0f47b67c7388da757` The completion record reports 23 September 2026, 05:53 UTC, and report digest `901ca40d124fbdd8b00807a9c5481d88b5a641ea31a855fe9608ecb04daf58fd`

All 24 saved-result audits report matched original test references, caption-metric recomputation from saved predictions, pooled-NLL arithmetic, and 866 test rows. Initial-model digests agree across the four methods within every seed. These are recorded consistency checks, not independent prediction regeneration or third-party training.

## Initialization recovery and reproducibility boundary

An intermittent initialization mismatch is isolated during diagnosis to `encoder.pos`. The guard stops affected constructions before training. Recovery preserves the original runner, protocol, initial-model hash records, and ten completed outputs. Only that pre-training mismatch is eligible for bounded same-seed retry. Training, checkpoint, and evaluation failures are not retried under the rule. A construction can proceed only when the original whole-model hash comparison passes; no tensor values, seeds, records, or score-based selection criteria are substituted.

The final preservation record passes, but the workaround does not resolve the underlying nondeterminism. A digest identifies an accepted state but does not reconstruct its tensors. The recorded accepted-state hashes do not establish availability of the corresponding initial-state binaries; exact initialization reconstruction from released artifacts or seeds alone is therefore not established.
