# Controlled GPT-2/C4 monitoring-cost benchmark

`run_c4_overhead.py` measures two paths: ordinary GPT-2 training (`plain`) and
hidden-state capture with sampled spectral monitoring (`monitor_only`). The
`MODES` registry contains these two conditions; extra helpers in the preserved
source do not establish additional measured conditions. This timing experiment
is separate from the six-seed C4 task-performance study.

Source and wrappers are byte-identical to the recovered server copies.
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records release hashes.
The historical result export contains no execution-time source digest. These
release hashes identify the recovered copies; they do not establish that these
bytes produced the historical measurements. The original ten observations and recorded configuration are in
[`results/2026-10-12-cost/`](../../results/2026-10-12-cost/); see the
[CPU cost reconstruction](../../analysis/cost/README.md).

## Workload and assets

Use a compatible CUDA environment with NumPy, transformers and pretrained
GPT-2/tokenizer assets. The recorded device is L40S with PyTorch `2.6.0+cu124`.
A local C4 JSONL gzip shard, writable result/persistent directories and an idle
GPU are required. No trained CÆNL checkpoint is needed; a random-token fallback
is refused.

Defaults are five paired repeats with randomized mode order, 25 untimed warmup
and 250 measured optimizer steps per mode, sequence length 1,024, microbatch
two and gradient accumulation four (8,192 tokens/step). Both paths share fixed
token tensors. Parameters are BF16; monitoring occurs every 25 steps using
128 sampled tokens and at most 64 eigenvalues. Production precision and batch
boundaries differ from this microbenchmark.

## Direct execution

Run from the repository root with a compatible environment activated. Select
the real shard and a new output directory. The recorded shard was the first
English C4 training shard.

```bash
CAENL_C4_SHARD="/mnt/caenl/persistent/datasets/c4/en/c4-train.00000-of-01024.json.gz"
CAENL_C4_COST="/mnt/caenl/active/results/caenl-c4-cost-reproduction-v1"

test -f "$CAENL_C4_SHARD"
test ! -e "$CAENL_C4_COST"
nvidia-smi --query-compute-apps=pid --format=csv,noheader

python -B tools/c4_overhead_benchmark/run_c4_overhead.py \
  --c4-shard "$CAENL_C4_SHARD" \
  --output-root "$CAENL_C4_COST" \
  --persistent-root /mnt/caenl/persistent \
  --model gpt2 --repeats 5 --warmup-steps 25 --measure-steps 250 \
  --seq-len 1024 --micro-batch 2 --grad-accum 4 \
  --monitor-cadence 25 --sample-tokens 128 --max-rank 64 \
  --lr 0.00005 --seed 20260909
```

Proceed only when the compute-process query shows no other GPU job. The Python
command does not itself enforce GPU exclusivity. Use a persistent terminal or
scheduler. Uncached model assets can be fetched through transformers; set
`HF_HUB_OFFLINE=1` when all required assets are cached and offline use is desired.
`--help` lists accepted options.

The preserved start/status wrappers use the historical `.venv`, output root and
tmux layout. The launcher moves a prior default output directory to a timestamped
`.previous` directory before starting. Direct execution with a new output root
gives a separate identity. The runner has no checkpoint-resume mode: rerun an
interrupted benchmark in a fresh directory and retain partial records separately.

## Outputs and interpretation

Outputs include `config.json`, `raw_runs.csv` (ten observations),
`grouped_metrics.csv`, `summary.json`, environment records, checksums and a lean
archive. Verify all five pairs and the archive checksum. Timed intervals exclude
model/data loading, initialization, warmup and report/archive generation.

The recorded mean paired monitoring increase is `0.6980%`, with sample SD
`0.1415` percentage points and about `0.0890 GiB` additional peak allocation.
This is a comparison of the two specified paths, rather than isolated controller
memory or total application cost.

Production C4 successful training-loop time increases by `7.88%`, `7.68%` and
`16.62%` for fixed, Lite and Full relative to its monitored baseline. Broader
task-job wall time increases by `6.13%`, `5.92%` and `14.52%`. These boundaries
differ. Neither set of ratios may be multiplied by the microbenchmark ratio
to infer production overhead relative to plain training.
