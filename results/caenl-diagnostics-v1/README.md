# Completed frozen-checkpoint robustness and cost

Read [RESULTS.json](RESULTS.json) for the complete numeric record, [PROTOCOL.json](PROTOCOL.json) for the separately fixed design and [PROVENANCE.json](PROVENANCE.json) for original archive/payload identities. Only absolute deployment-root strings are normalized; measurements and all comparisons are preserved. SHA256SUMS.txt identifies these public bytes, not the original bound archive.

From the repository root, install NumPy/SciPy if necessary and run:

```bash
python3 analysis/completed/reproduce.py
```

This checks both completed follow-up releases together, including counts, pairing, all eight primary contrasts, separate Holm families, all 15 descriptive intersection contrasts, cost rows and audio tuning. It reconstructs saved-record arithmetic; it does not repeat training, raw-data audits, adversarial generation or caption regeneration. No GPU, checkpoint or original data is needed for this arithmetic.

The [findings guide](../../docs/RESULTS.md) distinguishes adaptive versus tuned fixed evidence, common-label classification component attribution, robustness/geometry and practical cost. Nonrejection is not equivalence. Raw p-values and individual intervals do not override adjusted decisions. The older studies remain separate.
