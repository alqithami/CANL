> Source and protocol only: completed diagnostic findings are not released.
> This public deployment preserves the original Python and scientific protocol
> bytes. Its campaign source manifest targets the public solidification package,
> whose reader-facing documentation gives it a new full source identity.
> Do not replace either package in an active or resumable deployment.
> See [deployment and release boundaries](../../docs/PENDING_CAMPAIGNS.md).

# CÆNL checkpoint diagnostics

This package evaluates frozen final classifiers from the CÆNL content-disjoint
classification campaign and measures practical training-path costs. It does not
train another scientific model. The cost benchmark updates separate in-memory
copies for short timing segments and discards them.

Required inputs are the frozen `CAENL_solidification_v1` source, its verified
206-method completed result tree, the original image inventory, and the existing
PyTorch/torchvision/AutoAttack environment. No package installation, dataset
download, image deletion, or modification of trained checkpoints is performed.

## Evaluation

- ResNet-50 final models: entropy, independently selected fixed DCR, and
  MACC-Lite, each under seeds 62001–62010.
- Standard AutoAttack on the entire cleaned validation set at L∞ radii
  0.5/255, 1/255, 2/255, 4/255, and 8/255: 30 models and 150 finite suites.
- Primary endpoint: robust accuracy at 2/255. Three seed-paired comparisons form
  a separate Holm family, evaluated with paired-t and exact sign-flip tests.
- Secondary records: common-clean-correct survival, logit margins, normalized
  representation displacement, clean and returned-adversarial class moments,
  and a fixed 200-example-or-smaller local margin-gradient panel.
- Cost benchmark: six execution paths, five timing repetitions, 20 discarded
  warmup steps and 200 measured steps per repetition. Common starting weights,
  fresh optimizer/controller state, eight cached real balanced training batches,
  and the full original feedback holdout.

See `PROTOCOL.md` for interpretation and precise measurement boundaries.

## Run

Extract this package to a new `$HOME/CAENL_diagnostics_v1` directory on the
existing server. Verify the included source checksums, then queue it:

```bash
cd "$HOME/CAENL_diagnostics_v1" &&
sha256sum -c SHA256SUMS &&
python3 -B caenl_diagnostics_v1.py --queue
```

The detached worker waits for the solidification completion record, 206-method
analysis, worker exit, and an idle GPU. Waiting allocates no CUDA memory.
The waiting period has a fixed 20-day deadline. A blocked or missing
solidification worker stops the queue for inspection rather than restarting it.
The idle check is a process check, not an exclusive hardware reservation.

The default output is a separate tree:
`/mnt/caenl/active/results/caenl-diagnostics-v1`.
Custom locations may be provided on the first call with `--source`, `--campaign`,
and `--output`. An installed release remembers those locations.

Check status using the default system Python; it does not need to import CUDA:

```bash
python3 -B "$HOME/CAENL_diagnostics_v1/caenl_diagnostics_v1.py" --status
tail -n 40 /mnt/caenl/active/results/caenl-diagnostics-v1/run.log
```

The runtime selector reuses the original package versions and checks that
AutoAttack is available. Once the training campaign completes, the worker runs
CPU contracts, checks source/data/model identities, and performs a real model
input-gradient and fixed-loss-gradient parity check before the diagnostics.
At least 8 GiB free is required for new records; no existing files are removed.

Standard attacks at small radii can be expensive. After the first full-validation
suite, the log reports measured duration and a qualified extrapolation. There is
no verified GPU duration before that measurement.

## Resume and completion

Attack outputs commit between bounded chunks. An interrupted process can be
resumed with the same release and locations using `--queue`; committed chunks
are reused only when input bindings and payload hashes match. Cost repetitions
also commit individually. A mismatch blocks instead of accepting mixed inputs.
Do not replace the source of a running diagnostic process.

Successful completion prints `CAENL_DIAGNOSTICS_COMPLETE` and creates:

```text
/mnt/caenl/active/results/caenl-diagnostics-v1/review/CAENL_DIAGNOSTICS_REVIEW.zip
/mnt/caenl/active/results/caenl-diagnostics-v1/review/CAENL_DIAGNOSTICS_REVIEW.zip.sha256
```

The compact review contains protocol, runtime, checks, summaries, and timing
records. Numeric evidence is also exported as independently readable ZIP parts
below 30 MiB, each with its own payload manifest. Licensed raw images, complete
feature vectors, adversarial pixel tensors, and model weights are excluded from
the review archives and remain subject to their source access arrangements.

## Scope

The results concern this ImageNet-100 subset, these final ResNet-50 classifiers,
and the stated threat model. They do not establish ResNet-18 robustness or
cross-task gains. AutoAttack returns a mixture of initially incorrect unmodified
inputs, successful perturbations from different attacks, and surviving inputs;
the reported geometry describes that returned mixture. Outcome-group movement
and margin summaries are descriptive.

The pointwise margin-gradient norm is a local sensitivity measurement rather
than a neighborhood Lipschitz bound or certificate. Timing repetitions are not
independent training seeds. Cached timing data do not establish whole-job speed,
engineering labor, or adaptive performance. Serialized controller-state size is
not a measurement of RAM consumption. Earlier controller-development cost is
unknown and is not treated as zero.
