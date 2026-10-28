# Reconstruct the focused CIFAR-100 results on CPU

This workflow verifies the public numerical release and repeats its frozen
statistics. It does not train models, open raw data or checkpoints, or regenerate
predictions. Python 3.10 or newer, NumPy, SciPy and Matplotlib are required.

From the repository root, run:

```bash
python3 -m venv .venv-cifar100-analysis
. .venv-cifar100-analysis/bin/activate
python3 -m pip install -r analysis/cifar100_focused/requirements.txt
python3 -B analysis/cifar100_focused/reproduce.py --output reproduced-cifar100-focused
```

The output directory must be new. The program checks every public numeric
payload hash and all 20 files in the original frozen source manifest. It
verifies the complete 160-model test grid, original test receipt bytes and
saved accuracy numerators, checkpoint/training-receipt declarations, all 54
development scores and selections, and all 200 training-stage objects. The
training objects use a public JSON serialization; their original receipt
hashes are recorded separately.

`ANALYSIS.json` is reconstructed with the exact frozen analyzer. A separate
implementation recomputes AULC by adjacent trapezoids, enumerates all 1,024
sign patterns, applies both Holm families, and checks every unadjusted paired-t
interval. The independent unit is one paired seed. These are ten seed pairs,
not 40 independent budget replicates.

The output includes `VERIFICATION.json`, the reconstructed analysis, independent
checks, and `exports/`: all 160 official-test accuracies, ten seed-level AULC
rows, all three planned contrasts, all 40 method-seed cost records, descriptive
summaries, four column-width LaTeX tables, and a 3.35 by 2.65 inch vector figure.
Error bars are sample seed standard deviations, not confidence intervals.
The curve is not smoothed or constrained to increase with the label budget.

The results do not establish a primary AULC advantage for ratio DCR over CE,
full DCR or the specified Fisher recipe. Nonrejection is not equivalence.
The reported job-time field excludes pipeline downtime, recovery, preparation,
test inference and transfer. DCR methods each attribute their shared prefix
when comparing isolated recipes; physical campaign totals count that prefix
only once. See the release README and provenance for scope and limitations.

The public release excludes raw images/test labels, prediction arrays and model
binaries. Original saved prediction arrays and model states were independently
checked before release; public reconstruction verifies their released metadata
and score arithmetic rather than repeating that model execution.
