# Rebuild recorded result plots

This directory contains plotting code and numerical inputs from the completed 2026-09-26 analysis snapshot. The feedback-study experiment is outside this snapshot.

From the repository root, activate the separate analysis environment described in the [root README](../../README.md#reproduce-statistics-on-cpu), then run:

```bash
python3 -m pip install -r analysis/plots/requirements.txt
python3 analysis/plots/make_figures.py
```

The script writes `figure2_budget`, `figure3_effects`, and `figure4_entropy` in PDF, SVG, and PNG formats into this directory. These generated files are ignored by Git and stay local. Each filename identifies one plot group.

`figure_data.json` contains the recorded curves, paired contrasts, and entropy-treatment seed records. Error bars and intervals retain their original definitions: seed standard deviations for the trajectory/endpoint summaries, and individual unadjusted paired-t intervals for the effect plot. The script does not train models or create additional evidence.

For independent recalculation of statistics from the public per-seed exports, run `analysis/reproduce.py` as described in the root README.
