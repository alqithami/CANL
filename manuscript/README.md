# Revised manuscript

The complete paper retains the original CÆNL name (Collapse-Based Active Neural Learning) and identifies objective-aligned representation control in its title. It contains one workflow diagram and three numerical figures, with no internal worklist or editorial correspondence.

Compile `CAENL_revised.tex` in a CAS project, keeping the `figures/` directory next to it, or run `python3 build.py` with a full TeX Live installation. CAS class/support files are included. Required packages include STIX, natbib, amsthm, TikZ, algorithm/algorithmicx, xurl, seqsplit, microtype and multicol. Do not replace the STIX text/math setup without checking every mathematical symbol.

The workflow is editable TikZ within the LaTeX source. Numerical figures are supplied as vector PDF and SVG. To redraw them, install NumPy and Matplotlib and run `python3 figures/make_figures.py`. The script reads the accompanying `figure_data.json`: existing manuscript trajectory summaries, the verified six-seed records, and the released paired contrasts. It performs no training and introduces no new statistical family. Source measurements, SD bars, individual confidence intervals and paired-seed lines remain explicitly distinguished.

The manuscript build uses the supplied figure PDFs; it does not require Python plotting dependencies or regenerate figures automatically.
