# Controlled raw-source AudioCaps follow-up

`train_study.py` imports the unchanged, SHA-256 verified `reference/src/caenl` audio/core/utils modules. It ports the historical audio configuration and compiles only the pinned historical `TerminalSafeLoop` class AST. It never imports or executes the old engine, diffusion task, old mirror cache, or old launchers. The spectral loss, architecture, controller settings, dropout reset and optimizer recipe are retained; the added fixed-control selection is a separate tuning procedure.

The execution-only amendment in `EXECUTION_AMENDMENT_20261018.json` changes concurrency from two GPUs to one before any new study outcome; all 58 jobs, 13 seeds, algorithms, budgets, selection and analysis remain fixed. The earlier 24 runs remain separate. AudioCaps has been studied before, and AST pretraining may contain underlying recordings. These results must not be described as an unseen benchmark or as evidence for the aligned classification-ratio objective. `protocol.json` specifies the finite study before model outcomes.

## Required layout

Place this source bundle at the study root (or keep its files together elsewhere and invoke its absolute path). All mutable outputs go beneath `--root`:

```
/srv/iga/caenl-audiocaps-raw-v1/
  assets/MODELS_FROZEN.json       exact historical file hashes; recorded old paths are ignored
  assets/ast/                    matching files, no extra model snapshots
  assets/gpt2/                   matching files
  METRIC_COPY.json                exact historical files mapping
  metric_vendor/pycocoevalcap/    exact historical Python/JAR/model files
  runtime/java8/                 copied Java 8 runtime
  data/DATA_READY.json            produced only by successful prepare_data.py audit
  data/manifest.jsonl             original encoded WAV paths, official memberships/captions
  data/manifest_meta.json
  data/AUDIT.jsonl
  data/EXCLUSIONS.jsonl
  data/SELECTION_REPORT.json
```

Use Python 3.12, torch 2.6.0+cu124, transformers 5.15.1 and the pinned environment overlay. No runner command downloads data/assets or installs dependencies. Set `PYTHONPATH` to the environment overlay and preserve it for subprocesses. Java must resolve to this study's `runtime/java8/bin/java`.

## Coordinator commands

The coordinator owns process launch and monitoring. The shell examples below are commands to run on the prepared server, not evidence that they have been executed:

```bash
export STUDY_ROOT=/srv/iga/caenl-audiocaps-raw-v1
export STUDY_PYTHON=/srv/iga/venv/bin/python3.12
export JAVA_HOME="$STUDY_ROOT/runtime/java8"
export PATH="$JAVA_HOME/bin:$PATH"
export PYTHONPATH="$STUDY_ROOT/runtime/pydeps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

A fixture smoke can run before the full archive arrives. Its manifest contains exactly two real training rows and at least two real validation rows (five original references each), with original file SHA-256 values and no test rows. It has its own feature cache and initialization and never contributes a scientific run:

```bash
CUDA_VISIBLE_DEVICES=0 "$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase smoke --gpu 0 --smoke-manifest "$STUDY_ROOT/smoke_input/manifest.jsonl"
```

After `data/DATA_READY.json` exists, extract features with a single process, let it exit, then initialize the 13 immutable CPU model states. A full-data smoke may be used instead of the fixture smoke:

```bash
CUDA_VISIBLE_DEVICES=0 "$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase features --gpu 0
CUDA_VISIBLE_DEVICES="" "$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase initialize
CUDA_VISIBLE_DEVICES=0 "$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase smoke --gpu 0
```

The frozen execution contract uses one dedicated L40S. Worker 0 executes every tuning job sequentially and owns its `fcntl` worker lock; every training/audit child also owns its job lock. The unchanged confirmation method order still rotates by seed. Wait for the worker to exit successfully before selection:

```bash
CUDA_VISIBLE_DEVICES=0 "$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase worker --stage tuning --worker-index 0 --workers 1 --gpu 0
"$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase analyze --stage tuning
```

The selection command requires all 18 complete, audited ten-epoch outcomes, selects maximum mean final-step development CIDEr-D (tie: smaller lambda), and freezes `SELECTED_FIXED.json`. It cannot compute test captions or test NLL. Then run all 40 confirmation jobs on the same worker and wait for successful completion before analysis:

```bash
CUDA_VISIBLE_DEVICES=0 "$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase worker --stage confirm --worker-index 0 --workers 1 --gpu 0
"$STUDY_PYTHON" "$STUDY_ROOT/train_study.py" --root "$STUDY_ROOT" --phase analyze --stage confirm
```

Restart the same worker command after an interruption. Bound checkpoints restore model, optimizer, controller/monitor/pending-action state and RNG. The data order is derived from seed and epoch; the dropout stream is reset by seed and optimizer step. A completed `DONE.json` is reused only when its input/configuration binding and every declared file hash match. Workers spawn one training child at a time and then a separate clean-checkpoint audit child; CUDA allocations are released when each child exits. Metrics are serialized by a shared lock to protect historical SPICE cache writes.

## Outputs and checks

- `INPUT_BINDING.json`: complete code/reference, raw readiness/full manifest, preprocessing, model/metric/Java and runtime binding.
- `features/`: new FP16 64×768 feature memmap, fixed AST FP32 batch-4 extraction, progress markers and first-window cross-split audit. No mirror features reused.
- `initial/s<seed>/initial.pt` and `INITIAL.json`: one immutable complete model state per seed, binary/tensor hashes and RNG after CPU construction.
- `tuning/lambda<value>/s8100*/`: 18 final-development CIDEr-D outcomes; all tuning candidates retained.
- `confirm/{baseline,tuned_fixed,macc_lite,full_macc}/s8200*/`: 40 final-test outcomes, candidate plus all five references, target-token NLL batches, development shuffled-audio diagnostic, controller trajectory, FP32 final model and resume checkpoint.
- `artifacts/RESULT_AUDIT.json`: clean-process full-state reload, bitwise development probe logits/NLL comparison and saved-caption metric/reference checks. Captions are not regenerated by this audit; this limitation is recorded explicitly.
- `analysis.json`: all 40 seed outcomes, descriptive secondary summaries and the five prespecified CIDEr-D contrasts. The primary inference is the two-sided exact 1024-sign-flip test, with Holm across the five contrasts at familywise alpha 0.05. Paired t tests receive their own five-contrast Holm adjustment as a sensitivity analysis; they are never an alternative criterion for declaring a benefit. Confidence intervals are individual unadjusted 95% t intervals. No seed/outcome deletion or gain threshold is allowed.
- `SMOKE.json`: real-model two-step/resume, initialization/gradient checks, generation on two validation rows with no quality threshold, fixed synthetic-text PTB/CIDEr-D/BLEU/ROUGE-L/METEOR/SPICE/SPIDEr runtime check, and terminal Full-MACC fixture; zero scientific runs and no test outcomes. `SMOKE_GENERATION.json` and `SMOKE_METRICS.json` preserve the separate fixture outputs.

The cause of the historical paired-initial-state mismatch remains unknown. In the pinned encoder, the position parameter is allocated with `torch.zeros` and initialized with `nn.init.trunc_normal_`. This runner avoids attributing the mismatch to an uninitialized parameter and enforces pairing through a saved binary instead.

Local source checks do not establish server GPU execution. Run `python -m unittest discover -s . -p 'test_study*.py'` for the available CPU fixtures; `--phase smoke` performs the actual-model server check. Required metric failures, hash mismatches, missing runs, nonfinite values, insufficient Full-MACC learned updates, or replay failures block completion without dropping an outcome.
