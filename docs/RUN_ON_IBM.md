# Run the CAENL feedback study on IBM

These commands apply to the prepared `pot-gpu` server with one NVIDIA L40S. They use the existing ImageNet-100 dataset and CAENL virtual environment. Run server commands at the `[salqithami@pot-gpu ~]$` prompt.

## Check an existing run

For the campaign already in progress, use only:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --status
tail -n 35 /mnt/caenl/active/results/caenl-feedback-study-v1/run.log
```

Repository documentation updates require no action on the server. Do not replace the extracted engine, change its protocol, or reinstall dependencies during a run. The process uses frozen local files and its own checkpoints, independently of subsequent changes to the repository.

`0/104` means no complete method run has finished. Each method contains five acquisition/training rounds, and each seed first requires a shared initialization. Advancing step/epoch lines demonstrate training progress. A momentary zero-percent GPU reading can occur during hashing, data preparation, or other CPU work.

## Initial setup and launch

Skip this section if the campaign is already running. From a Mac terminal, connect first:

```bash
ssh salqithami@161.156.166.216
```

Then run the following on IBM. The download is pinned to the exact release used by the active experiment:

```bash
cd "$HOME" &&
curl -fL --retry 3 \
'https://raw.githubusercontent.com/alqithami/CANL/65df044feaf3bb6604416b1011ede5ea3b2a36ed/experiments/caenl_feedback_study_v1/caenl_feedback_study_v1.py' \
-o caenl_feedback_study_v1.py &&
printf '%s\n' '3d6b64de479c2426c2faab8991a9ec80058ed11a0aff813432a32ac1aa794c91  caenl_feedback_study_v1.py' |
sha256sum -c - &&
python3 "$HOME/caenl_feedback_study_v1.py" --start
```

The launcher embeds its engine. It checks source hashes, the mounted volume, dependency versions, available disk, all dataset image hashes, CPU integration tests, and real CUDA training steps for both backbones. It does not install packages or download a dataset. It starts a detached `tmux` worker, so SSH can be disconnected after launch.

Allow several days of GPU time and approximately 160 GiB free initially on `/mnt/caenl`. This is 104 new method runs plus 23 shared initializations. The launcher refuses to start alongside another GPU compute process; it does not stop that process.

Expected prerequisites:

- Dataset: `/mnt/caenl/persistent/datasets/imagenet100_cmc`, matching the frozen manifest.
- Primary Python: `$HOME/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python`.
- Fallback Python: `$HOME/caenl-src/caenl_revision_pipeline_v5.3_ibm_rhel_l40s/.venv/bin/python`.
- PyTorch 2.6.x, torchvision 0.21.x, CUDA 12.4, L40S, `tmux`, and `nvidia-smi`.
- Result directory: `/mnt/caenl/active/results/caenl-feedback-study-v1`.

Existing experiments remain intact. Running `--start` again while this worker is active does not launch a second copy.

## Resume an interrupted run

Read `--status` and the log first. After an interruption, or after the reported error is resolved:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --resume
```

Completed artifacts are verified and skipped. An unfinished phase resumes from its most recent checkpoint, including optimizer, controller, and random-number state. Failed records are preserved. Source/configuration mismatches stop the run instead of silently mixing experiments.

## Collect the completed report

Once `--status` reports `COMPLETE`, verify the report on IBM:

```bash
cd /mnt/caenl/active/results/caenl-feedback-study-v1 &&
sha256sum -c caenl-feedback-study-v1.md.sha256
```

Print the report:

```bash
python3 "$HOME/caenl_feedback_study_v1.py" --report
```

The report is `caenl-feedback-study-v1.md` in that result directory, with a companion `results.json`. These small text files contain the summaries needed for review; model checkpoints and the dataset do not need to be transferred. Completion confirms that the prescribed work finished; the findings can be positive, negative, or uncertain.

The [frozen protocol](../experiments/caenl_feedback_study_v1/README.md) specifies control selection, disjoint seeds, and the six-contrast analysis family. Control selection and all subsequent stages run automatically. Results from this study must be identified separately from the completed records in `results/2026-09-26/`.
