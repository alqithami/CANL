# Completed-follow-up CPU reconstruction

With Python 3 and NumPy/SciPy installed, run `python3 analysis/completed/reproduce.py` from the repository root. It verifies the diagnostic/audio public checksums, counts and seed pairing; reconstructs all eight primary effects and both recorded test families; checks all 15 descriptive common-clean-correct effects; checks six timing paths × five repetitions; and rebuilds all audio metric means/SDs and development-only fixed selection from 18 tuning and 40 confirmation runs. The earlier 206-run study has its own unchanged `results/caenl-solidification-v1/reconstruct_stats.py`.

These are arithmetic checks from released measurements. They do not independently execute models, attacks, caption decoding, dataset authentication or pretraining-overlap audits. Timing repetitions are not independent training seeds; endpoint families are not pooled across sequential studies. See the [complete findings and limitations](../../docs/RESULTS.md).

With Matplotlib installed, `python3 analysis/completed/plot.py --output figure_completed_diagnostics.pdf` rebuilds the paper's column-sized two-panel measured figure. It retains all five radii and all five audio contrasts, with individual unadjusted intervals and no adjusted-significance annotation.
