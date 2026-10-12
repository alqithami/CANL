# Reconstructing recorded computation costs

This CPU workflow reconstructs the timing arithmetic of the completed feedback
study and GPT-2/C4 measurements. It uses only the Python standard library; it
does not require a GPU, model checkpoints, dataset downloads, NumPy, or SciPy.
Run from the repository root with Python 3.9 or later:

```bash
python3 analysis/cost/reproduce.py --output reproduced-cost
```

The output directory must be new. To select another copy of the same numerical
snapshot, add `--records /path/to/records`. The command verifies all released
input SHA-256 digests before analysis and checks the recorded seed sets,
workloads, initializations, and original feedback/microbenchmark summaries.
It emits five CSVs, `cost_summary.json`, and `VERIFICATION.json`.

| Output | Measurement and denominator |
|---|---|
| `feedback_group_costs.csv` | Means and sample SDs of each recorded timing component for all 104 method runs. |
| `feedback_stage_allocation.csv` | Stage totals, with each of the 23 shared initializations allocated once. The tuning stage contains 24 runs. |
| `c4_microbenchmark_pairs.csv` | Five within-repeat monitor/plain time ratios and allocation differences. |
| `c4_microbenchmark_groups.csv` | Means and sample SDs of the 10 microbenchmark observations. The mean percentage is computed from the five paired percentages. |
| `c4_production_costs.csv` | Means and sample SDs across all six seeds per method for successful training-loop time and broader task-job wall time. Each percentage uses the matching boundary's group mean. |

## Timing boundaries

For the feedback study, the reported training-plus-acquisition quantity is
`(phase_duration_seconds + acquisition_seconds) / 3600`. Each method's phase
duration sums its five post-initialization phases. The separately recorded
synchronized training-compute timer overlaps the phase-duration timer and must
not be added again. The recorded tuning allocation is 39.6563 hours for those
phases and acquisitions, or 40.0671 hours with the three shared initializations.
These totals exclude final reporting and other unrecorded work. They do not
quantify the earlier development of MACC-Lite or the other methods.

All four confirmation and transfer conditions compute the same holdout feedback
at matched control windows. This includes the entropy and fixed references,
whose actions do not consume that feedback. MACC-Lite's measured path retains
its held-out cross-entropy observation, although cross-entropy does not enter
the coefficient action. The recorded differences therefore do not estimate
cost against the leanest fixed-weight or unregularized implementation.

The GPT-2/C4 microbenchmark uses five paired repeats with randomized execution
order within each repeat. Both paths use identical fixed token tensors, 25
untimed warmup optimizer steps, and 250 measured steps (2,048,000 tokens). The
monitor-only path performs ten monitor events and no learned policy update.
Mean measured times are 33.5293 and 33.7633 seconds. The mean paired increase is
0.6980% (sample SD 0.1415 percentage points); mean peak allocation increases
0.0890 GiB. Model/data loading and initialization are excluded. Its BF16
parameters, sequence length 1,024, microbatch two, and four accumulation passes
are specific to this controlled benchmark.

The production-task records include all 24 GPT-2/C4 scientific jobs, each with
12,207 optimizer steps and 99,999,744 training tokens. The per-seed summaries
record successful training-loop time in `compute.train_time_s`. Mean loop times
are 2,332.7091 seconds for the monitored baseline, 2,516.5304 for fixed spectral
regularization, 2,511.9142 for MACC-Lite, and 2,720.4757 for Full MACC. Their
ratios of group mean training-loop times increase by 7.8802%, 7.6823%, and
16.6230%, reproducing the paper's rounded 7.88%, 7.68%, and 16.62%.

The broader successful task-job
wall means are 2,395.4150 seconds for the monitored baseline, 2,542.2644 for
fixed spectral regularization, 2,537.2819 for MACC-Lite, and 2,743.3354 for Full
MACC. Their ratios of mean task-job wall times increase by 6.1304%, 5.9224%, and
14.5244%. All seeds, including the longer baseline seed 301, are retained.
The CPU workflow checks each training-loop summary's steps, tokens, seed, and
task-job wall value against the independent job table. Task-job wall time has a
broader boundary than the successful training-loop timer; percentages from
those boundaries are not interchangeable. These
production ratios cannot be combined with the compute microbenchmark to infer
total overhead relative to plain training.

None of these quantities is an isolated controller footprint, total application
overhead, or proof of economic savings. The nine-condition classification,
fixed-entropy follow-up, diffusion, and audio timing summaries are outside this
raw-record snapshot; this command does not independently reconstruct them.

## Numerical input provenance

`results/2026-10-12-cost/` includes byte-identical feedback method timings,
all shared-initialization summaries, and the ten original C4 microbenchmark
observations. Production job fields and all 24 successful training-loop summary
fields are projected without changing their numeric values. Each loop row
includes the SHA-256 digest of its source per-seed summary. Deployment paths
are removed from the C4 configuration and job
records. `provenance.json` identifies the original export hashes and each
projection; `SHA256SUMS.json` verifies the released portable inputs.

The verification checks arithmetic and record consistency. Matching input
hashes preserves the released values but does not independently establish
upstream timing authenticity or recreate the experiments. The in-progress
solidification and robustness-diagnostics campaigns contribute no measurements
to this completed snapshot.

