# CÆNL C4 overhead benchmark v2

This package performs a short, controlled two-mode GPT-2 timing benchmark on
the existing IBM L40S environment. It does not retrain the 100M-token C4
confirmatory models.

Modes:

1. Plain GPT-2: no hidden-state capture and no spectral monitor.
2. Monitoring only.

Default design:

- 5 paired timing repeats.
- 25 untimed warm-up optimizer steps per mode.
- 250 measured optimizer steps per mode.
- 8,192 effective tokens per optimizer step.
- BF16 on CUDA GPU 0.
- Randomized mode order within each repeat.
- No concurrent GPU jobs.
- Model loading and report/archive generation excluded from timed intervals.

Results:
`/mnt/caenl/active/results/caenl-c4-overhead-v2`

Archive pointer:
`/mnt/caenl/persistent/archives/LATEST_caenl-c4-overhead-v2.txt`

Run:

```bash
bash start_c4_overhead_tmux.sh
bash status_c4_overhead.sh
```

The result is a controlled overhead benchmark and should be reported separately
from the six-seed 100M-token task-performance experiment. The benchmark refuses
to fall back to random tokens; it must find a local C4 shard.
