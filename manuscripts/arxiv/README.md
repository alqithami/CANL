# CÆNL preprint — 23 October 2026

[Read the revised paper](CAENL_arXiv.pdf) · [Clean LaTeX source ZIP](CAENL_arXiv_source.zip) · [Version notes](CAENL_arXiv_version_notes.txt) · [Validation](CAENL_arXiv_validation.json)

43 pages, seven column-sized figures, 43 tables and two algorithms. The former Table 12 now fits a column. The author block contains only Saad Alqithami and his Gmail address. The existing AI-use declaration is preserved in the manuscript and supplied separately for reuse. All theoretical statements, proofs, algorithms and 48 bibliography entries are unchanged from the preceding revision.

The revision adds completed 30-model/150-suite robustness diagnostics, six practical-cost paths and a separate 58-run raw-source AudioCaps study. The earlier classification, C4, diffusion and audio results remain separate. The paper supports the evaluated classification scatter-ratio intervention; adaptive superiority, equivalence, robustness gains, cross-task improvement and total-cost savings remain unestablished. All planned comparisons and negative/inconclusive findings are retained.

The paper pins the completed new evidence to revision `51412af5df877a8b87746bd7a8a9422192895132`. Full numerical records are in [classification](../../results/caenl-solidification-v1/), [diagnostics](../../results/caenl-diagnostics-v1/) and [raw-source audio](../../results/caenl-audiocaps-raw-v1/). [CPU reconstruction and the measured figure script](../../analysis/completed/) distinguish endpoint arithmetic from independent model execution. No original experiment was changed or restarted by this publication update.

Unzip the clean source archive into a new directory and run `latexmk -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex`. It includes every external figure and required style; references and the workflow schematic are embedded. The source archive contains no compiled PDF, logs, numerical data, licensed data or model binaries. A fresh-directory rebuild matches the reading copy's page count and extracted text. This repository copy is a prepared preprint, not a claim of an arXiv submission or acceptance.
