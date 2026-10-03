from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description="Plot the released feedback-study numerical records.")
parser.add_argument('--records',type=Path,default=Path(__file__).resolve().parents[2]/'results/2026-10-02-feedback')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists(): raise SystemExit('Choose a new output directory.')
args.output.mkdir(parents=True)
A=args.records;O=args.output;P=args.output
a={'primary_contrasts':pd.read_csv(A/'primary_contrasts.csv').to_dict('records')}
c=pd.read_csv(A/'learning_curves.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'legend.frameon':False})
names={'entropy':'Entropy','lite':'Lite','tuned_fixed':'Tuned fixed','tuned_schedule':'Tuned schedule'}
styles={'entropy':('#737373','--','o'),'lite':('#0072B2','-','o'),'tuned_fixed':('#151515','-','s'),'tuned_schedule':('#D55E00',':','^')}
fig=plt.figure(figsize=(7.05,5.65));gs=fig.add_gridspec(2,2,height_ratios=[1,1.04],hspace=.5,wspace=.21)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
for ax,stage,title in zip(axs,['confirmation','transfer'],['(a) ResNet-50 confirmation','(b) ResNet-18 transfer']):
 for method,(color,ls,mark) in styles.items():
  g=c[(c.stage==stage)&(c.method==method)].groupby('phase').accuracy.mean()*100
  ax.plot([20,30,40,50,60],g,color=color,ls=ls,marker=mark,ms=3,lw=1.4,label=names[method])
 ax.set(title=title,xlabel='Annotation budget (%)',xticks=[20,30,40,50,60],ylim=(35,76),yticks=[40,50,60,70]);ax.grid(axis='y',alpha=.18)
axs[0].set_ylabel('Validation top-1 accuracy (%)')
fig.legend(*axs[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.555),ncol=4,columnspacing=1.6,handlelength=2.5,fontsize=8)
ax=fig.add_subplot(gs[1,:]);ax.axvline(0,color='#888888',lw=.8,ls='--');ax.axhspan(2.5,5.5,color='#f2f4f6',zorder=0)
for i,d in enumerate(a['primary_contrasts']):
 y=5-i;v=d['difference_pp'];color=styles[d['reference']][0]
 ax.errorbar(v,y,xerr=[[v-d['ci95_low_pp']],[d['ci95_high_pp']-v]],fmt='o',color=color,capsize=3,markersize=4,lw=1.3)
 ax.text(2.42,y,f'{v:+.3f}',ha='right',va='center',fontsize=7.5)
labels=[('R50' if d['stage']=='confirmation' else 'R18')+'  Lite − '+names[d['reference']].lower() for d in a['primary_contrasts']]
ax.set(yticks=list(range(5,-1,-1)),yticklabels=labels,ylim=(-.6,5.6),xlim=(-.9,2.48),xticks=[-.5,0,.5,1,1.5,2],xlabel='Final accuracy difference (percentage points)',title='(c) Paired effects and individual 95% confidence intervals')
ax.grid(axis='x',alpha=.14);fig.subplots_adjust(left=.21,right=.98,top=.95,bottom=.09)
fig.savefig(O/'figure_feedback_confirmation.pdf');fig.savefig(P/'figure_feedback_confirmation.png',dpi=190);plt.close(fig)
w=pd.read_csv(A/'controller_mean_windows.csv')
fig,axs=plt.subplots(1,2,figsize=(7.05,2.45),sharey=True)
bounds=np.cumsum([3880,5840,7820,9800,11780])
for ax,stage,title in zip(axs,['confirmation','transfer'],['(a) ResNet-50 confirmation','(b) ResNet-18 transfer']):
 g=w[w.stage==stage].groupby(['phase','mid_step'],sort=True)
 for layer,color,ls in [('layer3','#0072B2','-'),('layer4','#151515','--')]:
  m=g[layer].mean();ax.plot(m.index.get_level_values('mid_step')/1000,m.values,color=color,ls=ls,lw=1.25,label=layer)
 for b in bounds[:-1]:ax.axvline(b/1000,color='#bbbbbb',lw=.6,ls=':')
 ax.axhline(.1,color='#D55E00',ls='-.',lw=1,label='Tuned fixed')
 ax.set(title=title,xlabel='Post-initialization steps (thousands)',xlim=(0,39.12),xticks=[0,10,20,30,39.12],xticklabels=['0','10','20','30','39.1'],ylim=(0,.105),yticks=[0,.025,.05,.075,.1]);ax.grid(axis='y',alpha=.16)
axs[0].set_ylabel('Applied regularization coefficient')
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.005),ncol=3,fontsize=8)
fig.subplots_adjust(left=.095,right=.985,top=.86,bottom=.26,wspace=.14)
fig.savefig(O/'figure_feedback_dynamics.pdf');fig.savefig(P/'figure_feedback_dynamics.png',dpi=190);plt.close(fig)
print('Generated confirmation and controller-dynamics plots in',O)
