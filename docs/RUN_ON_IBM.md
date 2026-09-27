# Deploy the feedback study on an L40S host

This guide runs the frozen ImageNet-100 feedback experiment on a Linux host with an NVIDIA L40S. Commands are executed on the GPU host from the repository root unless stated otherwise. No personal server account or hostname is required.

## Runtime and storage requirements

| Requirement | Released implementation |
|---|---|
| Accelerator | NVIDIA L40S, accessible through `nvidia-smi` |
| PyTorch / torchvision | 2.6.x / 0.21.x, CUDA 12.4 build |
| Python dependencies | NumPy, SciPy, Pillow; recorded versions in each run's `INPUT_BINDING.json` |
| Storage | A real mounted volume at `/mnt/caenl`; approximately 160 GiB free for a new campaign |
| Dataset root | `/mnt/caenl/persistent/datasets/imagenet100_cmc` |
| Result root | `/mnt/caenl/active/results/caenl-feedback-study-v1` |
| Process supervision | A persistent terminal session or a scheduler that retains the process after SSH disconnects |

The engine checks the hardware, CUDA build, mount point, and result path. The result directory must be unused for a new experiment or contain checkpoints from exactly the same bound experiment. It is not a configurable multi-run workspace.

## Python environment

Create a separate environment for a new deployment. The following Python 3.12 environment combines the prescribed CUDA packages with the non-PyTorch versions recorded in the supplied CPU validation:

```bash
python3.12 -m venv .venv-feedback
. .venv-feedback/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install numpy==2.5.2 scipy==1.17.0 Pillow==12.3.0
python -m pip check
```

The PyTorch wheel combination follows the [official PyTorch 2.6 installation instructions](https://pytorch.org/get-started/previous-versions/#v260). The CPU validation is not a complete GPU environment lock: a new host must pass the CUDA preflight below. To reproduce a specific archived run, match its `INPUT_BINDING.json` environment and dependency record instead. Keep the environment fixed after the first preflight; resume checks bind it to the run.

## Dataset contract

Obtain ImageNet data under its applicable access terms. This repository does not distribute the images or automatically prepare the required subset. An arbitrary ImageNet-100 subset will not satisfy the frozen protocol.

The dataset directory must contain:

- `IMAGENET100_READY.json`, with `status: READY`, 126,689 training images, 5,000 validation images, 100 classes in each split, and `class_list_sha256` matching the protocol identity.
- `class_counts.csv`, with `class_id` and `wnid` fields defining the class mapping.
- `images_manifest.jsonl`, with one `path`, `class_id`, and `sha256` record per image. Relative image paths use `train/<wnid>/<filename>` or `val/<wnid>/<filename>`.
- The corresponding image files, with 50 validation images per class.

The expected image-manifest SHA-256 is `0ba19db4a9d1aeb633d1339a5f9ec11f55dc219d884a266bb4d805eca8d6dc1f`; the class-list identity is `5ee5db69d488bb799a286d8e181801c3d02798776c1ce9203e5450f64e086e07`. These values and all dataset counts are specified in [protocol.json](../experiments/caenl_feedback_study_v1/engine/protocol.json).

Exact execution requires access to the matching manifest and authorized image collection. Hashes identify those inputs but do not reconstruct missing files. If those inputs are unavailable, the released CPU analysis remains usable; constructing another subset is a separately identified experiment.

## Verify the environment

With the virtual environment activated and the required volume mounted:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
python -B experiments/caenl_feedback_study_v1/engine/campaign.py \
  --root /mnt/caenl/active/results/caenl-feedback-study-v1 \
  --check-only
```

This verifies source hashes, available storage, software tests, every image hash, and actual batch-128 CUDA training steps for ResNet-50 and ResNet-18. A successful preflight records `CHECK_COMPLETE`; it does not perform scientific training runs. Another GPU compute process causes the preflight to stop without terminating that process.

## Execute the experiment

Run inside a persistent terminal session or submit the command through a suitable process supervisor. A plain SSH terminal must remain connected if no supervisor is used.

```bash
set -o pipefail
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
python -B experiments/caenl_feedback_study_v1/engine/campaign.py \
  --root /mnt/caenl/active/results/caenl-feedback-study-v1 \
  2>&1 | tee -a /mnt/caenl/active/results/caenl-feedback-study-v1/run.log
```

Control selection, confirmation, and architecture transfer execute sequentially. The control-selection record is frozen before confirmation. The [experiment protocol](../experiments/caenl_feedback_study_v1/README.md) defines the 104-run matrix and statistical comparisons.

## Monitor and resume

From another terminal on the same host:

```bash
python3 experiments/caenl_feedback_study_v1/caenl_feedback_study_v1.py --status
tail -n 35 /mnt/caenl/active/results/caenl-feedback-study-v1/run.log
```

The completed-run counter advances only after all five phases of a method finish. Step and epoch messages show progress within a phase. Shared initialization, hashing, and data operations are distinct from completed method runs.

After an interruption, inspect `FAILED.json` and the log, resolve the recorded error, then rerun the same engine command with the same checkout and environment. Completed artifacts are verified and skipped; unfinished training resumes from its last checkpoint. The run lock prevents simultaneous campaign workers. Do not alter the protocol or remove integrity records to bypass a mismatch.

## Output files

| File | Meaning |
|---|---|
| `INPUT_BINDING.json` | Source, protocol, and runtime identities |
| `CPU_CHECK.json`, `CUDA_CHECK.json`, `DATA_CHECK.json` | Preflight validation results |
| `STATUS.json` | Most recently recorded operation |
| `FAILED.json` | Error details when an invocation fails |
| `SELECTION.json` | Controls selected from the complete tuning grid |
| `results.json` | Machine-readable summaries and statistical results |
| `caenl-feedback-study-v1.md` | Human-readable numerical report |
| `VERIFICATION.json`, `COMPLETE.json` | Final validation and completion records |

After completion, verify the report:

```bash
cd /mnt/caenl/active/results/caenl-feedback-study-v1
sha256sum -c caenl-feedback-study-v1.md.sha256
```

Retain the report, machine-readable results, configuration/environment records, and per-seed artifacts for reproducibility. A successful completion record establishes execution of the protocol; interpretation depends on the observed effect sizes and uncertainty.

## Optional launcher for the original directory layout

The self-contained `caenl_feedback_study_v1.py` launcher supports `--start`, `--resume`, `--status`, `--report`, and `--check`. It starts a detached `tmux` worker and searches for Python at these existing locations:

1. `$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python`
2. `$HOME/caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python`

On a host that uses this layout, verify the launcher and start it from the repository root:

```bash
(cd experiments/caenl_feedback_study_v1 && sha256sum -c caenl_feedback_study_v1.py.sha256)
python3 experiments/caenl_feedback_study_v1/caenl_feedback_study_v1.py --start
```

Use `--resume` for an interrupted launcher-managed run. With a different virtual-environment location, use the direct engine commands above. The launcher embeds the same frozen engine; it does not install dependencies or fetch data.
