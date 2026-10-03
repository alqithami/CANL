# Reproducing the feedback-study results

The numerical snapshot in `results/2026-10-02-feedback/` contains all 104 completed method runs: 24 control-selection runs, 40 ResNet-50 confirmation runs, and 40 ResNet-18 transfer runs. The three seed sets are disjoint. The snapshot adds to the earlier `results/2026-09-26/` records; the statistical families are not pooled.

From the repository root, with Python 3.12:

```bash
python3 -m venv .venv-feedback-analysis
. .venv-feedback-analysis/bin/activate
python3 -m pip install -r analysis/feedback/requirements.txt
python3 analysis/feedback/reproduce.py --output reproduced-feedback
python3 analysis/feedback/plot.py --output feedback-plots
```

Both output directories must be new. `--records /path/to/records` selects another copy of the same snapshot. The reconstruction verifies every numerical input checksum before analysis and emits group means and sample SDs, all six paired contrasts, the sensitivity contrasts, control-selection rankings, and `VERIFICATION.json`. The plotting command emits vector PDFs and PNG previews of the measured learning trajectories, final paired effects, and applied feedback coefficients. Run the verification command before plotting.

The six primary contrasts are Lite minus entropy, selected fixed strength, and selected schedule, on each of two backbones. Paired-t and exhaustive 1,024-pattern sign-flip tests each receive Holm adjustment across all six comparisons. Confidence intervals are individual paired-t intervals, not simultaneous intervals. The same family is retained in the post-hoc duplicate-exclusion analysis. Small-sample inference concerns seed variation under the frozen benchmark protocol, not uncertainty over new datasets.

| Record | Meaning |
|---|---|
| `reconstructed_evaluations.json` | 104 final training-holdout and 80 final official-validation metric records, reconstructed from native prediction arrays and saved scatter sufficient statistics. Accuracy and ECE are proportions. |
| `primary_contrasts.csv` | Reference contrasts for the planned 5,000-image final endpoint; effects and intervals are in percentage points. |
| `duplicate_exclusion_per_seed.csv` | Original and sensitivity accuracy for all 80 evaluation runs after excluding the same five validation indices. |
| `sensitivity_contrasts.csv` | Reference six-contrast family for the 4,995-image post-hoc sensitivity. |
| `learning_curves.csv` | All 400 measured post-acquisition outcomes. Earlier rounds use archived phase summaries; final outcomes were checked against native arrays. |
| `trace_diagnostics.csv` | Per-phase summaries of all 520 post-initialization traces; includes applied strengths, clipping frequency, and safeguard activation. |
| `controller_mean_windows.csv` | Ten-seed mean and sample SD of each layer's applied Lite strength in successive windows of at most 100 steps within each phase. Window means are computed per seed before aggregation. |
| `provenance.json` | Frozen source/report identities, completed counts, tuning rankings, timing boundaries, and content-overlap indices. |
| `SHA256SUMS.json` | SHA-256 digests of all eight numerical input files. |

The selected controls are constant strength 0.10 and a linear decreasing schedule with peak 0.10. Selection uses the final training-holdout score on seeds 41001–41003; official validation is not evaluated during tuning. Confirmation uses seeds 42001–42010 and transfer uses 43001–43010. Lite settings and selected controls are unchanged for ResNet-18.

The dataset manifest records five exact cross-split content matches with conflicting labels, 51 within-training duplicate pairs, and 43 within-training conflicting-label pairs. The primary endpoint retains all 5,000 validation examples. Sensitivity exclusion removes validation indices 114, 125, 476, 939, and 1793, using the frozen manifest's zero-based indexing. All six adjusted-test decisions remain unchanged. Evaluation-time exclusion does not undo any training exposure or establish a deduplicated training benchmark.

This CPU workflow reconstructs released-record arithmetic. It does not repeat model inference or training, certify upstream data authenticity, or imply that every historical checkpoint can be recreated from its seed alone. To inspect native saved arrays and traces, use the [evidence export guide](../../docs/FEEDBACK_EVIDENCE_EXPORT.md). For GPU execution, use the [frozen protocol](../../experiments/caenl_feedback_study_v1/README.md) and [deployment guide](../../docs/RUN_ON_IBM.md).
