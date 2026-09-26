#!/usr/bin/env python3
"""Build with pdfLaTeX and standard CAS/STIX/PGFPlots/algorithm packages."""
from pathlib import Path
import subprocess,shutil,os
root=Path(__file__).resolve().parent;b=root/'build';b.mkdir(exist_ok=True)
for name in ['CAENL_revised.tex','cas-dc.cls','cas-common.sty']:shutil.copy2(root/name,b/name)
for i in range(3):
    with (b/f'pass-{i+1}.log').open('w') as f:
        subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-jobname=CAENL_revised_fixed',r'\pdfmapfile{+stix.map}\input{CAENL_revised.tex}'],cwd=b,stdout=f,stderr=subprocess.STDOUT,check=True)
log=(b/'CAENL_revised_fixed.log').read_text(errors='replace')
if any(x in log for x in ['undefined references','undefined citations','Overfull','LaTeX Error']):raise RuntimeError('Inspect build log before using PDF')
stage=root/'CAENL_revised_fixed.new.pdf';shutil.copyfile(b/'CAENL_revised_fixed.pdf',stage);os.replace(stage,root/'CAENL_revised_fixed.pdf')
print(root/'CAENL_revised_fixed.pdf')
