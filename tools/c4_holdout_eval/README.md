# GPT-2/C4 evaluation on an unused shard

`run_c4_holdout_eval.py` evaluates the 24 final checkpoints from
`caenl-c4-confirmatory-v2`: baseline, fixed spectral DCR, MACC-Lite and Full
MACC, at seeds 301–306. It uses
`en/c4-validation.00002-of-00008.json.gz`, excluded from the recorded training,
controller feedback and original evaluation. No model is trained or modified.
The released evaluator matches the historical `METADATA.json` script digest;
[SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json) records that correspondence.

## Prerequisites

- A compatible v5.4.3 project with `src/caenl/language/task.py` and its resolved
  Python environment. Exact replay requires the historical source/environment.
- CUDA; the recorded device and PyTorch build are L40S and `2.6.0+cu124`.
- All 24 source jobs at
  `<campaign-root>/stages/c4_confirmatory/jobs/<method>__seed<seed>/`, including
  `summary.json` and `checkpoints/state.pt`. Each must record 12,207 steps and
  99,999,744 trained tokens. Campaign `PLAN.yaml` and `PUBLICATION_CHECK.json`
  are also required for metadata generation.
- GPT-2/tokenizer assets, authorized C4 access and writable data/cache/result/
  archive directories. The data routine prepares a small source-training token
  cache as well as the evaluation shard, fetching uncached assets if needed.

The repository distributes numerical exports and source, but not the 24 trained
checkpoints or C4 text. A fresh clone alone cannot replay their inference.

## Direct execution

Run from the repository root on the GPU host. Set the compatible project and
completed campaign paths. Use a new output directory for a new evaluation.

```bash
CAENL_C4_PROJECT="/path/to/recorded-v5.4.3"
CAENL_C4_CAMPAIGN="/mnt/caenl/active/results/caenl-c4-confirmatory-v2"
CAENL_C4_HOLDOUT="/mnt/caenl/active/results/caenl-c4-holdout-reproduction-v1"

"$CAENL_C4_PROJECT/.venv/bin/python" -B \
  tools/c4_holdout_eval/run_c4_holdout_eval.py --self-test

"$CAENL_C4_PROJECT/.venv/bin/python" -B \
  tools/c4_holdout_eval/run_c4_holdout_eval.py \
  --project "$CAENL_C4_PROJECT" \
  --campaign-root "$CAENL_C4_CAMPAIGN" \
  --output-root "$CAENL_C4_HOLDOUT" \
  --data-root /mnt/caenl/active/data \
  --cache-root /mnt/caenl/active/cache \
  --persistent-root /mnt/caenl/persistent \
  --validation-shard en/c4-validation.00002-of-00008.json.gz \
  --validation-tokens 2000000 --batch-seqs 8 --device cuda:0
```

Use a persistent terminal or scheduler. The CPU self-test checks pooled metric
and paired-statistic arithmetic; it does not validate model loading or CUDA.
`--help` lists accepted options. Avoid simultaneous GPU training during replay.

The preserved start/status shell scripts use the original fixed directory and
tmux layout. The start wrapper's `CAENL_PROJECT` selects Python/PYTHONPATH but
does not pass `--project` to the evaluator; its own default still applies.
Use the direct command above for another installation layout.

## Resume, outputs and statistical boundaries

After confirming the prior process has stopped, repeat the same command with
unchanged inputs and settings. Existing records are skipped when checkpoint
digest and shard identity match. This historical skip check does not bind every
evaluation option: changing token counts, batches or devices requires a new
output directory. Inspect the process identified by a `.running` lock before
handling a stale lock.

Completion emits 24 `per_checkpoint/*.json` records, `per_seed_metrics.csv`,
`grouped_metrics.csv`, `paired_comparisons.csv`, `METADATA.json`, `REPORT.md`,
`c4_holdout_table.tex`, checksums and a lean archive. Verify the final exit status
and complete matrix. [Deployment](../../docs/DEPLOYMENT.md) explains checksum
collection; the [experiment map](../../docs/REPRODUCIBILITY.md) identifies the
released numerical reconstruction.

NLL is primary; perplexity is derived from it. The historical evaluator applies
Holm adjustment to paired-t tests within each three-comparison endpoint family
and emits raw exact sign-flip values. It does not emit Holm-adjusted exact
values. The public CPU reconstruction computes those adjustments separately:
for fixed and Lite NLL contrasts, raw exact `0.03125` becomes `0.09375`.
