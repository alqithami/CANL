# Run and monitor the separate AudioCaps raw-source study

This finite follow-up contains **58 method runs**: 18 development-only fixed-coefficient tuning runs and 40 confirmation runs. It also extracts frozen AST features once and saves one shared initial caption-model state for each of 13 seeds. It uses one dedicated L40S GPU and one sequential training worker. The execution-only amendment predates study outcomes and is recorded in `EXECUTION_AMENDMENT_20261018.json`. It has no result-dependent extension or stopping rule.

The previous 24 AudioCaps runs and classification campaigns remain separate evidence. This follow-up uses the previously studied AudioCaps benchmark with a newly audited raw-audio source; it does not create an untouched external dataset. See `protocol.json` and `TRAINING_README.md` for the scientific protocol and exact implementation.

## Prerequisites

Install the source directory intact. The default campaign directory is `/srv/iga/caenl-audiocaps-raw-v1`; the default interpreter is `/srv/iga/venv/bin/python`. Dependencies live in the campaign's own `runtime/pydeps` overlay. The coordinator does not install packages or alter the shared environment.

Stage the raw ZIP and pinned official caption CSVs under `incoming/`. Complete and verify the asset/runtime manifests, and run the real-model smoke check described in `TRAINING_README.md`. Required successful receipts are:

- `incoming/SOURCE_READY.json`: full source-file SHA-256 and byte size verified.
- `assets/ASSETS_READY.json`: exact frozen AST, GPT-2 and metric assets staged.
- `runtime/RUNTIME_READY.json`: private runtime overlay and Java 8 validated.
- `SMOKE.json`, `SMOKE_GENERATION.json`, `SMOKE_METRICS.json`: actual AST/caption-model training, exact checkpoint-resume, validation-only generation, and every frozen text metric passed. Smoke fixtures count as zero scientific runs.

The coordinator may be started while staging is underway. It waits for these verified inputs for at most **72 hours**, then stops with an explicit `BLOCKED` reason. This limit applies to input staging; it is not a limit on training duration. Readiness requires successful smoke fixtures; the coordinator does not silently replace a failed fixture.

## Start once

Run this from the installed source directory:

```bash
/srv/iga/venv/bin/python -B coordinator.py --start
```

This starts one detached coordinator, so the SSH session may disconnect. The root lock and Linux PID/start-time identity prevent duplicate coordinators. Repeating `--start` reports an existing live process rather than launching another one. The original protocol, source hashes and immutable data/assets binding must match on resume.

The coordinator runs these stages in order:

1. Wait for verified inputs and smoke receipts.
2. Audit, canonicalize for duplicate detection, and extract eligible original WAV bytes with the fixed data policy.
3. Extract frozen AST features on physical GPU 0; the extraction process exits before training workers begin.
4. Save the 13 shared CPU initializations.
5. Run all 18 tuning jobs sequentially on physical GPU 0; freeze the development-selected coefficient.
6. Run all 40 confirmation jobs sequentially on the same GPU, preserving the unchanged finite job list and paired randomness.
7. Verify all completed payloads, compute all five planned paired contrasts, report practical costs, and package the compact review.

Only child processes receive `CUDA_VISIBLE_DEVICES`; each child uses logical `--gpu 0`. The coordinator itself does not allocate CUDA memory. Before a GPU phase it records the GPU inventory and refuses to compete with an already active compute process on any selected GPU.

## Check progress

```bash
/srv/iga/venv/bin/python -B coordinator.py --status
tail -n 40 /srv/iga/caenl-audiocaps-raw-v1/logs/coordinator.log
tail -n 30 /srv/iga/caenl-audiocaps-raw-v1/logs/tuning_0.log
```

During confirmation, replace `tuning_0.log` with `confirm_0.log`. The status reports the current stage, live child processes and receipt counts out of 18 and 40. Receipt counts are progress indicators; they do not establish scientific success or full artifact verification. Current process identities are recorded in `PID.json` and `CHILDREN.json`.

`WAITING_FOR_VERIFIED_INPUTS` means staging or smoke verification is still pending. `RUNNING` means the named phase is active. `COMPLETE` means all scheduled jobs and analyses finished and the review ZIP was verified; it does not mean a hypothesis was supported. `BLOCKED` includes the exact failure reason and retains all evidence. A failed worker stops the finite campaign with its exact log and retained checkpoint; no outcome is discarded. The general coordinator also permits a separately frozen two-worker execution contract, and lets an assigned sibling finish if that configuration is used.

## Resume after an operational interruption

Inspect `STATUS.json`, the named child log, available disk space and `CHILDREN.json`. Do not remove checkpoints, receipts, failed-attempt records or original evidence. Fix the reported operational issue, then use the same `--start` command. Completed jobs are reverified and skipped; unfinished jobs use their retained full-state checkpoints. No replacement seed is introduced.

If an old child remains alive after its coordinator stopped, `--start` reports that child and starts nothing. Wait for that process to stop before resuming. A completed campaign is verified and reported without relaunching it. An expired staging deadline requires an explicit inspected extension of `READINESS_DEADLINE.json`; repeated starts do not reset its clock.

## Retrieve a complete review

The compact review is written only after all 18 tuning and 40 confirmation outcomes pass the locked completeness and artifact checks:

```text
review/CAENL_AUDIOCAPS_RAW_REVIEW.zip
review/CAENL_AUDIOCAPS_RAW_REVIEW.zip.sha256
review/ARCHIVE_RECEIPT.json
```

`RESULTS.json` contains the complete confirmation analysis, selected coefficient and all 18 tuning outcomes, data coverage/exclusions, practical-cost summaries, and explicit evidence scope. The exact paired sign-flip tests with Holm correction across five contrasts are primary; paired t tests with their separate Holm family are sensitivity analyses. Retain negative and inconclusive outcomes.

The review includes all planned contrast and conditioning summaries, original predictions/references, controller trajectories, job telemetry, source/protocol/data manifests and per-file SHA-256 hashes. Large audio, feature arrays, pretrained weights and training checkpoints stay on the server, with their hashes retained in the original receipts. ZIP CRC, payload inventory, each payload SHA-256 and the outer ZIP SHA-256 are verified before completion is reported.

This audio study informs adaptive-versus-tuned-fixed spectral regularization on a common labeled audio training set. It does not replace the separate classification component-attribution or smaller-radius robustness/geometry diagnostics, and it does not establish their conclusions.
