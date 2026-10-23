#!/usr/bin/env python3
"""Rebuild the column-sized completed-results figure from released measurements."""
from pathlib import Path
import argparse,json,statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('figure_completed_diagnostics.pdf'))
args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
root=Path(__file__).resolve().parents[2]
R=json.loads((root/'results/caenl-diagnostics-v1/RESULTS.json').read_text())['robustness']
AC=json.loads((root/'results/caenl-audiocaps-raw-v1/RESULTS.json').read_text())['confirmation']
plt.rcParams.update({'font.size':7,'axes.labelsize':7,'axes.titlesize':7.5,'legend.fontsize':6.5,'axes.linewidth':.6,'pdf.fonttype':42})
fig,axs=plt.subplots(2,1,figsize=(3.35,4.15),gridspec_kw={'height_ratios':[1,1.15]},layout='constrained')
for m,col,mark,ls,lbl in [('entropy','#222222','o','-','Entropy'),('tuned_fixed','#777777','s','--','Tuned fixed'),('lite','#176B73','^',':','Lite')]:
 vals=[100*st.mean(s['robust_accuracy'] for s in R['suites'] if s['epsilon_pixels']==e and s['method']==m) for e in R['radii_pixels']]
 axs[0].plot(R['radii_pixels'],vals,marker=mark,lw=.85,ms=3,color=col,ls=ls,label=lbl)
axs[0].set(xscale='log',yscale='log',xticks=R['radii_pixels'],xticklabels=['0.5','1','2','4','8'],xlabel=r'$L_\infty$ radius (pixel levels / 255)',ylabel='Robust top-1 (%)',title='(a) Frozen ResNet-50 checkpoints: ten-seed means')
axs[0].legend(frameon=False,ncol=1,loc='upper right');axs[0].grid(axis='y',color='.9',lw=.5)
short=['Fixed - baseline','Lite - baseline','Full - baseline','Lite - fixed','Full - fixed']
for i,c in enumerate(AC['primary_contrasts']):
 mu=c['difference_mean'];lo,hi=c['confidence_interval_95_unadjusted']
 axs[1].errorbar(mu,4-i,xerr=[[mu-lo],[hi-mu]],fmt='o',ms=3,color='#176B73',elinewidth=.9,capsize=2)
axs[1].axvline(0,color='.4',lw=.6,ls='--');axs[1].set(yticks=range(5),yticklabels=short[::-1],xlabel='Paired CIDEr-D difference (target - reference)',title='(b) Raw-source AudioCaps: individual 95% intervals')
axs[1].set_xticks([-.02,-.01,0,.01],['-0.02','-0.01','0','0.01'])
axs[1].grid(axis='x',color='.9',lw=.5)
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.minorticks_off()
fig.savefig(args.output,bbox_inches='tight',pad_inches=.025);plt.close(fig)
print(args.output)
