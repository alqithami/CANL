# Deployment and asset requirements

Use the CPU numerical workflows to inspect completed findings without models
or data downloads. Run from the repository root with Python 3.12 and new output
directories:

```bash
python3 -m venv .venv-analysis
. .venv-analysis/bin/activate
python -m pip install -r analysis/requirements.txt
python analysis/reproduce.py --output reproduced-earlier
python -m pip install -r analysis/feedback/requirements.txt
python analysis/feedback/reproduce.py --output reproduced-feedback
python analysis/cost/reproduce.py --output reproduced-cost
```

These reconstruct released arithmetic, with no training or inference. See the
[experiment map](REPRODUCIBILITY.md), [verification scope](PROVENANCE.md) and
[cost guide](../analysis/cost/README.md).

## GPU host and environment

The recorded platform is Linux with one NVIDIA L40S, a working driver, PyTorch
`2.6.0+cu124` and a writable mounted volume at `/mnt/caenl`. Reserve space on
both the system disk and experiment volume, including temporary files. Frozen
campaigns with fixed output paths need an unused compatible layout.

For a new dedicated installation,
[`scripts/bootstrap_ibm_l40s.sh`](../scripts/bootstrap_ibm_l40s.sh) installs
system/Python dependencies, checks CUDA and runs software tests on supported
RHEL-compatible or Ubuntu/Debian hosts. It requires installation rights and an
already visible GPU and mounted volume:

```bash
bash scripts/bootstrap_ibm_l40s.sh
```

Provisioning is separate from scientific reproduction. For archived checkpoints,
retain their recorded source, configuration and resolved environment; do not
upgrade a resume environment with a bootstrap. The package under
[`archives/`](../archives/) preserves historical v5.4.3 source. Later engines
and wrappers have separate identities under `experiments/`.

The historical pipeline accepts explicit storage roots without requiring a
personal override file. From its compatible project installation:

```bash
export CAENL_ACTIVE_ROOT=/mnt/caenl/active
export CAENL_PERSISTENT_ROOT=/mnt/caenl/persistent
source scripts/ibm_one_dataset_env.sh
. .venv/bin/activate
```

Use storage with adequate capacity for caches and process temporary files.
If needed, create a dedicated writable directory and set `TMPDIR` before
launch, recording that runtime setting with the deployment.

## Workflow and asset map

| Workflow | Entry point | Prerequisites outside numerical release |
|---|---|---|
| Initial nine-condition ImageNet-100 study | [Aligned experiment](../experiments/caenl_aligned_imagenet100_v1/README.md) | Authorized images, exact class/image manifests and compatible runtime. |
| Completed 104-run tuned-control study | [L40S guide](RUN_ON_IBM.md) | Bound ImageNet-100 dataset/manifest and runtime; controls and initializations are generated within the campaign. |
| Earlier entropy/Lite, UHerding and entropy/fixed additions | `experiments/caenl-focused-completion-v1/runner.py`, `experiments/caenl-uherding-v1/runner.py`, `experiments/caenl-entropy-fixed-v1/runner.py` | Completed original classification source/results, shared initial checkpoints, manifests and audit inputs; the fixed addition also needs completed Lite records. |
| GPT-2/C4 training | [C4 runbook](C4_CONFIRMATORY_V2_RUNBOOK.md) | Compatible historical project, pretrained GPT-2/tokenizer, C4 shards/caches, resolved configuration and runtime. |
| C4 unused-shard evaluation | [Evaluator](../tools/c4_holdout_eval/README.md) | All 24 final checkpoints, summaries, campaign metadata and C4/model/tokenizer assets. |
| C4 monitoring microbenchmark | [Benchmark](../tools/c4_overhead_benchmark/README.md) | Pretrained GPT-2/tokenizer and real local C4 shard; no trained CÆNL checkpoint. |
| CIFAR-10 diffusion confirmation | `experiments/caenl-diffusion-confirm-v1/runner.py` | Bound pilot source/cache/manifests, CIFAR-10 data, pinned DDPM weights and metric assets. |
| AudioCaps confirmation | `experiments/caenl-audiocaps-confirm-v1/runner.py` | Bound pilot source/feature caches, authorized recordings/captions, pretrained assets, caption metrics/Java runtime and accepted initial-state evidence. |

Historical follow-up wrappers reuse hash-bound assets in fixed result trees.
A fresh clone cannot supply their checkpoints, features or manifests. Preserve
the integrity checks; a substituted subset or new asset set needs a separate
deployment/protocol identity. Audio accepted-state digests cannot reconstruct
missing state tensors; exact regeneration from seeds alone is not established.

The ImageNet contract is in [RUN_ON_IBM.md](RUN_ON_IBM.md); other task protocols
and recorded environments are under
[`provenance/protocols/`](../provenance/protocols/). Data/model access terms
remain applicable. Licensed raw data and large checkpoints are not bundled.
Exploratory or unexecuted configurations do not establish completed findings.

## Execution, resume and collection

Choose one workflow and pass its software, dataset and CUDA checks before
training. Use a persistent terminal or scheduler. Resume with unchanged inputs
only after the previous process has stopped and its error is understood. Timing
benchmarks need an idle GPU. Logs show progress; findings require a complete
verified matrix.

Retain the source revision, resolved configuration, environment, per-seed
outputs, integrity records and archive checksum. Verify the archive where both
it and its companion checksum are present, using its actual emitted filename:

```bash
sha256sum -c result-archive.zip.sha256
```

On macOS use `shasum -a 256 -c`. Transfer through your configured host's normal
file-transfer mechanism and keep the checksum with the archive.

The ongoing 206-method solidification and 30-model/150-suite robustness
diagnostic campaigns contribute no outcomes to the completed snapshots.
Their findings require separate completion, verification and release.

For the additional content-disjoint replication and checkpoint diagnostics, see [PENDING_CAMPAIGNS.md](PENDING_CAMPAIGNS.md). These public packages preserve scientific executable bytes but have newly bound public documentation; keep existing campaign installations unchanged when resuming.
