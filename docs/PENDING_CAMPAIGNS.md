# Content-filtered replication and checkpoint diagnostics

These packages release scientific implementations and fixed protocols. The 206-method content-filtered replication is complete, with its [verified numerical endpoints and reconstruction](../results/caenl-solidification-v1/) released separately. The 30-model/150-suite robustness and practical-cost diagnostics are complete, with a [verified numerical release](../results/caenl-diagnostics-v1/). The separate [58-run raw-source audio release](../results/caenl-audiocaps-raw-v1/) is also complete. Consult [RESULTS.md](RESULTS.md) for the completed evidence and its scope.

## Scientific questions and finite workload

| Campaign | Design | Records and evidence status |
|---|---|---|
| [Content-filtered replication](../experiments/caenl_solidification_v1/README.txt) | Prospective exact-content filtering; independently selected constants/schedules; entropy and representation-loss controls; common-label component attribution; matched-step full-label comparison; architecture transfer | Complete: 36 selection + 110 confirmation + 60 transfer method runs; 23 shared initializations; [numerical release](../results/caenl-solidification-v1/) |
| [Checkpoint diagnostics](../experiments/caenl_diagnostics_v1/README.md) | Frozen ResNet-50 checkpoints for entropy, tuned fixed DCR and MACC-Lite; AutoAttack at 0.5, 1, 2, 4 and 8/255; descriptive geometry/sensitivity measurements; six practical-cost paths | Complete: 30 models × five radii = 150 attack suites; five repetitions per cost path; [numerical release](../results/caenl-diagnostics-v1/) |

The 206-method protocol separates end-to-end adaptive-vs-fixed comparisons from component comparisons conditional on a common acquired-label sequence. The latter do not establish an unconditional acquisition-policy effect. The diagnostic primary endpoint is robust accuracy at 2/255 with three paired contrasts in its own Holm family. Smaller-radius geometry and local margin gradients are descriptive measurements, not robustness certificates or causal mediation proofs. Timing repetitions are not independent scientific training seeds. Both protocols retain negative and inconclusive outcomes.

## Source and resume identities

Every Python file and scientific protocol JSON is byte-identical to its original package, verified against the retained `ORIGINAL_SOURCE_MANIFEST.json`. Public documentation replaces personal transfer instructions. The public `SOURCE_MANIFEST.json` and `SHA256SUMS` bind this documentation as well as the code. The diagnostic campaign manifest consequently targets the public replication package. `PUBLIC_RELEASE.json` identifies the changed metadata and unchanged executable bytes.

These documentation changes create a different full source binding. Use this public package only for a separate deployment with new result directories. **Do not copy it over an active or resumable campaign or substitute its manifests into existing evidence.** Keep the original installed source, full input binding, resolved runtime and checkpoints for any original-campaign resume. The scientific code and protocols of that installation remain unchanged by this release.

`LOCAL_VALIDATION.json` preserves historical CPU fixture evidence from package preparation; it does not claim a new public-package GPU reproduction. Successful fixture checks validate implementation contracts rather than scientific hypotheses.

## Prerequisites

These are archival-asset workflows, not self-contained clean-machine training demos. They require:

- A compatible Linux L40S/CUDA host and the package versions recorded in the completed feedback campaign's input binding. The launcher matches torch, torchvision, numpy, scipy and Pillow; it does not install or upgrade them.
- The original complete feedback result tree at `/mnt/caenl/active/results/caenl-feedback-study-v1`, including its input binding, 104 final checkpoints, holdout/validation records, source and summaries. The input binding must equal the original protocol's recorded identity.
- Authorized ImageNet-100 images and their exact class and image manifests at `/mnt/caenl/persistent/datasets/imagenet100_cmc`. Released numerical summaries do not replace these assets.
- For diagnostics, the completed replication tree and both source packages, plus the original AutoAttack-compatible environment. Diagnostics verify source, dataset, model and evaluation identities before attacks.

Use [DEPLOYMENT.md](DEPLOYMENT.md) for broader asset requirements. The replication preflight re-executes original checkpoints, audits decoded content, freezes exclusions and derived counts, and verifies actual GPU steps before training. Incompatible or missing assets block execution instead of relaxing integrity checks.

## Install and verify on a separate compatible host

Run from the repository root. The destination directories must not already exist:

```bash
test ! -e "$HOME/CAENL_solidification_v1" &&
test ! -e "$HOME/CAENL_diagnostics_v1" &&
cp -R experiments/caenl_solidification_v1 "$HOME/CAENL_solidification_v1" &&
cp -R experiments/caenl_diagnostics_v1 "$HOME/CAENL_diagnostics_v1"

(cd "$HOME/CAENL_solidification_v1" && sha256sum -c SHA256SUMS)
(cd "$HOME/CAENL_diagnostics_v1" && sha256sum -c SHA256SUMS)
```

Use the compatible Python environment for bounded CPU fixture checks:

```bash
python3 -B "$HOME/CAENL_solidification_v1/caenl_solidification_v1.py" --self-test
python3 -B "$HOME/CAENL_diagnostics_v1/caenl_diagnostics_v1.py" --self-test \
  --source "$HOME/CAENL_solidification_v1"
```

Reserve about 64 GiB initially for replication and at least 8 GiB for diagnostic records. Also check system-disk and inode capacity: multiprocessing needs writable temporary storage even when result storage is spacious. Before launching, configure a dedicated temporary directory on adequate storage:

```bash
mkdir -p /mnt/caenl/active/caenl-runtime-tmp
export TMPDIR=/mnt/caenl/active/caenl-runtime-tmp
df -h / /mnt/caenl/active "$TMPDIR"
df -i / /mnt/caenl/active "$TMPDIR"
```

## Launch, monitor and resume

Only launch on an unused compatible deployment, with no existing workers or output trees at the selected locations:

```bash
python3 -B "$HOME/CAENL_solidification_v1/caenl_solidification_v1.py" --start
python3 -B "$HOME/CAENL_diagnostics_v1/caenl_diagnostics_v1.py" --queue
```

Diagnostics wait for verified replication completion, worker exit and an idle GPU; waiting uses no CUDA memory and has a finite 20-day deadline. The idle check is not an exclusive GPU reservation. Replication is finite at 206 method runs. Historical planning allowed roughly two weeks on one L40S; this is a capacity estimate, not a guaranteed duration. Diagnostic duration depends on measured attack throughput.

```bash
python3 -B "$HOME/CAENL_solidification_v1/caenl_solidification_v1.py" --status
python3 -B "$HOME/CAENL_diagnostics_v1/caenl_diagnostics_v1.py" --status
tail -n 40 -f /mnt/caenl/active/results/caenl-solidification-v1/run.log
# Or monitor diagnostics:
tail -n 40 -f /mnt/caenl/active/results/caenl-diagnostics-v1/run.log
```

Ctrl+C exits the log viewer only. Read live `PID.json` records rather than retaining historical process numbers. If interrupted, inspect the error, worker state, disk/inode capacity and temporary storage before resuming. After the old worker exits, resume the unchanged replication installation with `--start`, or diagnostics with `--queue`, retaining the same `TMPDIR` and input locations. Never delete experimental evidence to make space. Committed progress is reused only when bindings match; an incomplete validation may be re-executed from its saved training checkpoint.

## Verify and collect completed evidence

Each result tree writes a compact review ZIP in `review/` with a companion checksum:

```bash
(cd /mnt/caenl/active/results/caenl-solidification-v1/review && \
 sha256sum -c CAENL_SOLIDIFICATION_REVIEW.zip.sha256)
(cd /mnt/caenl/active/results/caenl-diagnostics-v1/review && \
 sha256sum -c CAENL_DIAGNOSTICS_REVIEW.zip.sha256)
```

Transfer each ZIP together with its checksum using the host's configured SSH connection; on macOS verify with `shasum -a 256 -c`. Review archives exclude licensed raw images and model binaries. Preserve their separate access arrangements and original weights. Each numbered numerical evidence ZIP is independently readable and has its own payload manifest.

A recovery review may exist without scientific completion. Before reporting results, confirm stopped workers, expected matrices and identities, the authoritative archive hash and internal payload manifests; inspect `RESULTS.json` and the full planned contrast or diagnostic summaries. Report adaptive-vs-tuned-fixed effects, common-label attribution, smaller-radius robustness/geometry and measured costs separately, including negative or inconclusive outcomes. A progress log or `COMPLETE` flag alone does not establish the hypothesis.

