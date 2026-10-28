# Explicit official-test release and one-shot evaluator

`source/release_and_evaluate_v2.py` is the exact executed evaluator. The initial
v1 source is retained beside it to make the input-validation amendment
reviewable. `SOURCE_PROVENANCE.json` records original and public byte identities
and the synthetic tests' repository-relative path adaptations. No evaluator
runtime logic or scientific decision was edited for publication.

Before test access, the evaluator binds all 200 completed training stages,
all 160 final checkpoints, the exact dataset archive and the frozen three-contrast
analysis. It neither trains nor selects configurations, epochs or models. Every
one of the 10,000 original test rows is evaluated in archive order, without
augmentation, using fixed normalization and float32 evaluation/inference mode.
Saved numerators, arrays, receipts and unchanged model states are checked.
Existing evidence blocks replay. Project/GPU locks and per-model 180-second
and global two-hour ceilings remain active. Only the explicit analysis lock
and release hash are filled at this stage.

The v1 evaluator opened the test member before an overstrict filename-uniqueness
check stopped it. It produced zero model attempts, receipts or scores. V2 accepts
10,000 rows with 9,999 unique filename strings, retains every original row, and
binds the prior access and unchanged original deadline. Row indices define
identity. This is input-validation recovery, not a model-score retry or evidence
of duplicate images. The exact recovery record is in the numerical release.
The source root defaults deliberately remain pinned; no new generally portable
GPU deployment is claimed.

From the repository root, these tests use synthetic arrays and mocked workers
without opening real data or accessing a GPU:

```bash
python3 -B experiments/caenl_cifar100_official_test_v2/test_release.py
python3 -B experiments/caenl_cifar100_official_test_v2/test_release_v2.py
```

Original validation reports are retained in `validation_original/`. Public test
reports are written outside `source/`, to `validation/` or the directory set by
`CAENL_TEST_REPORT_DIR`. See `results/caenl-cifar100-focused-v1/` for every actual
score and provenance, and `analysis/cifar100_focused/` for CPU reconstruction.
