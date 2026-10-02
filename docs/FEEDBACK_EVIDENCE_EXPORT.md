# Inspecting saved feedback-study evidence

`tools/export_feedback_evidence.py` exports a completed feedback study using
Python's standard library. It reads the saved artifacts, checks the source
binding and available recorded payload hashes, and produces ordinary ZIP
volumes of at most 28 MiB. It does not import the experiment engine, load model
weights, run inference, or require a GPU.

From a checkout of this repository on the machine holding the results:

```bash
python3 tools/export_feedback_evidence.py \
  --root /mnt/caenl/active/results/caenl-feedback-study-v1 \
  --output /mnt/caenl/active/exports/caenl-feedback-study-v1-evidence
```

Use an output directory outside the result tree. An existing output directory
is never overwritten. On successful completion the utility prints
`EVIDENCE_EXPORT_COMPLETE`. Compression and verification use CPU and disk I/O;
elapsed time and archive count depend on the saved artifact sizes.

| Output | Contents |
|---|---|
| `CAENL_feedback_review_*.zip` | Original source and configuration, selection records, every method and phase summary, controller trajectories, acquisition selections, train/holdout splits, data catalog, logs, and completion records |
| `CAENL_feedback_predictions_*.zip` | Original final-round holdout and validation NPZ files, including logits, labels, indices, predictions, and class-scatter sufficient statistics |
| `CAENL_feedback_traces_*.zip` | Original per-step training CSV files for all methods and shared initializations |
| `EXPORT_MANIFEST.json` | SHA-256 hashes and archive locations for every exported file, plus an inventory of retained files and explicitly stated verification limits |
| `SHA256SUMS` | Checksums for every ZIP volume and the export records |
| `EXPORT_COMPLETE.json` | Completion record, written only after verifying all archived payloads |

All ZIP volumes are independently readable. Their paths retain the experiment
hierarchy beneath `results/`; the data catalog is under `dataset_metadata/`.
They can be inspected without extracting other volumes, or extracted together
into an empty directory to reconstruct the selected evidence tree. Model
weights and dataset images are not included. Keep the original result tree.

To include every earlier-round evaluation array and acquisition candidate-pass
array, add `--all-arrays` and choose a new output directory. These files are
placed in `CAENL_feedback_additional_arrays_*.zip`. All earlier-round summaries
and controller histories are already included in the default export.

Checkpoint paths and their previously recorded hashes are always inventoried.
By default, checkpoint bytes are not reread and those hashes are explicitly
marked as not verified by the exporter. Add `--verify-checkpoints` to hash all
checkpoint bytes against their saved completion records. This can require
substantial disk reads, but it does not deserialize or execute checkpoints.
Individual model checkpoints can subsequently be transferred when an
independent forward-pass check is needed.

After copying the export directory to another machine, verify the archives:

```bash
cd /path/to/caenl-feedback-study-v1-evidence
sha256sum -c SHA256SUMS
```

On macOS, the checksum command is `shasum -a 256 -c SHA256SUMS`. To also verify
each archived payload without accessing the original experiment directory:

```bash
python3 /path/to/CANL/tools/export_feedback_evidence.py \
  --output /path/to/caenl-feedback-study-v1-evidence --verify-export
```

## What an artifact audit can establish

The final arrays permit independent reconstruction of accuracy, cross-entropy,
calibration, and saved scatter summaries. Phase summaries and training traces
permit inspection of learning curves, applied coefficients, target residuals,
and optimization behavior. Splits and acquisition records permit checks of
annotation budgets, holdout exclusion, and shared initialization. Candidate
scores and earlier evaluation arrays are available through `--all-arrays`.

Hash agreement establishes consistency with the recorded artifacts. It does
not independently reproduce model inference, certify that an implementation
matches its mathematical description, or establish a scientific conclusion.
Inference reconstruction requires the corresponding checkpoint, images, and
compatible runtime. Diagnoses motivated by inspected results should be
identified as exploratory and kept distinct from the six predefined primary
comparisons. No result should be removed solely because it is unfavorable.

## Exporter validation

Run the standard-library integration checks with:

```bash
python3 tools/tests/test_export_feedback_evidence.py
```

The synthetic fixture uses the complete 104-method / 543-checkpoint directory
structure. Checks cover byte preservation, volume size limits, complete file
selection, optional checkpoint verification, verification without the source
tree, and rejection of changed reports, changed payloads, archive corruption,
manifest changes, symlinks, and output overwrites. These tests validate the
export utility; they are not scientific results or an audit of a GPU run.
