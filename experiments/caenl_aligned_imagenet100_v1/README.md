# Run the frozen aligned ImageNet-100 active-learning comparison

## Existing IBM resources reused
- Login: `ssh salqithami@161.156.166.216` from a MAC terminal only.
- Python: `~/caenl-v5.4.3/caenl_revision_pipeline_v5.4.3_c4_confirmatory/.venv/bin/python`.
- Dataset: `/mnt/caenl/persistent/datasets/imagenet100_cmc`.
- No pip install, no new download, no old file deletion, no old checkpoint loaded.

## Start (IBM terminal)
From this extracted directory run `bash start.sh`.
It starts one detached tmux session, `caenl-aligned-imagenet100-v1`.
The initial CPU self-tests use temporary synthetic fixtures and do not create scientific runs.
CUDA preflight and input checks then run; 6 fresh initializations and 54 full method runs follow sequentially.
This is full multi-day active learning, not another short development probe.

## Status (IBM terminal)
`bash status.sh`
The status reads the actual fixed result directory and live process, not a stale inherited CAENL_RESULTS_ROOT.
A tmux window is not itself proof of training. CPU dataset checks and selection/reporting can leave the GPU idle.
Watch current operation, epoch/step and completed-method count (target 54).

## Resume
Run `bash start.sh` again only after a stopped process has been identified.
A live PID/lock prevents a second runner. Completed jobs are verified and skipped.
The most recent committed epoch or 1,000-step checkpoint resumes with the saved model, optimizer, RNG and controller state.
Do not edit the frozen protocol to change the same campaign; binding mismatch causes a hard stop.

## Completion and retrieve (MAC terminal)
The server prints `ALIGNED IMAGENET100 COMPLETE` and `ALIGNED_AL_EXIT_CODE=0`.
Download (do not execute these commands inside IBM):
```
scp salqithami@161.156.166.216:/mnt/caenl/persistent/archives/imagenet100-aligned-results.tar.gz "$HOME/Downloads/"
scp salqithami@161.156.166.216:/mnt/caenl/persistent/archives/imagenet100-aligned-results.tar.gz.sha256 "$HOME/Downloads/"
cd "$HOME/Downloads"
shasum -a 256 -c imagenet100-aligned-results.tar.gz.sha256
```
Attach both files to the chat. No raw images or model checkpoints are required for the next artifact audit.

## What this does not claim
Software completion is independent of whether an aligned method improves accuracy/robustness.
This tests Full MACC regularization control with a fixed acquisition threshold, not a newly validated threshold policy.
The short aligned development screen has not established this end-to-end result in advance.
See PROTOCOL.md for the exact design and independent baseline implementation qualifications.
