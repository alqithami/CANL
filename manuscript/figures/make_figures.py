#!/usr/bin/env python3
"""Rebuild the vector result figures from figure_data.json. Requires numpy and matplotlib."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT=Path(__file__).resolve().parent
D=json.loads((ROOT/'figure_data.json').read_text())
ACCENT='#176B73'; INK='#24292d'; GREY='#6b7075'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
 'axes.edgecolor':'#878c90','axes.linewidth':.6,'grid.color':'#e0e3e5','grid.linewidth':.5,
 'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':3,'ytick.major.size':0,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def tidy(ax,grid='y'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True); ax.grid(axis=grid)
    ax.tick_params(pad=4)

def panel(ax,label,title):
    ax.set_title(f'{label}  {title}',loc='left',fontweight='bold',pad=10)

def save(fig,name):
    # Exact canvas sizes keep typography consistent when included at text width.
    fig.savefig(ROOT/f'{name}.pdf',metadata={'Creator':'CAENL recorded-data figure script'})
    fig.savefig(ROOT/f'{name}.svg')
    fig.savefig(ROOT/f'{name}.png',dpi=180)
    plt.close(fig)

styles=[('#91969a','o',':'),(INK,'s','-'),('#555b60','^','--'),
        ('#777d81','D','-.'),('#4a5055','v',(0,(4,2))),('#22272b','x',(0,(1,2)))]
curves=D['initial_curves']
fig=plt.figure(figsize=(7.3,6.4))
ax1=fig.add_axes([.09,.605,.38,.33]);ax2=fig.add_axes([.59,.605,.38,.33])
for ax,seq,is_reg in [(ax1,curves[:6],False),(ax2,curves[6:],True)]:
    for i,s in enumerate(seq):
        col,m,ls=styles[i]
        if is_reg:
            col,m,ls=[(INK,'s','-'),('#91969a','o',':'),('#555b60','^','--'),(ACCENT,'D','-'),('#4a5055','x','-.')][i]
        a=np.array(s['values'])
        ax.errorbar(a[:,0],a[:,1],yerr=a[:,2],color=col,marker=m,linestyle=ls,
                    markersize=3.2,lw=1.05,elinewidth=.55,capsize=1.7,
                    markerfacecolor='white' if m not in ['s','x'] else col,label=s['label'])
    ax.set(xlim=(8,63),ylim=(27,76),xticks=range(10,61,10),yticks=[30,40,50,60,70],xlabel='Annotated pool (%)')
    tidy(ax)
    ax.legend(loc='upper left',frameon=False,fontsize=7.2,handlelength=2.3,
              labelspacing=.35,borderaxespad=.15,handletextpad=.5)
ax1.set_ylabel('Clean top-1 accuracy (%)')
panel(ax1,'a','Acquisition without DCR');panel(ax2,'b','Aligned regularization')

# The enlarged endpoint comparison resolves curves that overlap at the full scale.
ax=fig.add_axes([.36,.085,.60,.415]);tidy(ax,'x')
entries=curves[:6]+curves[7:]
ys=[10,9,8,7,6,5,3.5,2.5,1.5,.5]
for s,y in zip(entries,ys):
    a=np.array(s['values'])[-1];color=ACCENT if s['label']=='Geometry + Lite' else INK
    ax.errorbar(a[1],y,xerr=a[2],fmt='D' if color==ACCENT else 'o',color=color,
                ms=4.2,capsize=3,lw=1.1,elinewidth=1)
ax.set(xlim=(68.5,74.15),ylim=(-.25,10.8),yticks=ys,yticklabels=[s['label'] for s in entries],
       xlabel='Clean top-1 accuracy at 60% annotations (%)')
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.axhline(4.25,color='#c9ced1',lw=.6)
ax.text(1.0,1.03,'Mean ± seed SD · n = 6',ha='right',transform=ax.transAxes,color=GREY,fontsize=8)
fig.text(.09,.529,'c  Final-budget comparison',fontweight='bold',fontsize=10)
save(fig,'figure2_budget')

# All seven contrasts in the original statistical family, with individual CIs.
cs=next(f['contrasts'] for f in D['contrasts'] if f['name']=='classification/original/clean_accuracy')
labels=['Fixed DCR − geometry only','MACC-Lite − fixed DCR','Full MACC − fixed DCR',
        'MACC-Lite − entropy','Full MACC − entropy','Geometry only − random','Random + fixed DCR − random']
fig=plt.figure(figsize=(7.3,3.55));ax=fig.add_axes([.36,.18,.37,.69]);tidy(ax,'x')
ys=np.arange(7)[::-1]
for c,y in zip(cs,ys):
    mean,lo,hi=np.array([c['mean_difference'],c['ci95_low'],c['ci95_high']])*100
    color=ACCENT if c['target']=='caenl_aligned_lite' else INK
    ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt='D' if color==ACCENT else 'o',
                color=color,ms=4.5,lw=1.2,capsize=3)
    fig.text(.97,.18+.69*(y+.6)/7.2,f'{mean:+.3f}  [{lo:+.3f}, {hi:+.3f}]',ha='right',va='center',fontsize=8.5)
ax.axvline(0,color=GREY,ls=(0,(3,3)),lw=.9)
ax.set(xlim=(-1.05,1.9),ylim=(-.6,6.6),yticks=ys,yticklabels=labels,
       xlabel='Target − reference (percentage points)')
ax.xaxis.set_major_locator(MultipleLocator(.5))
fig.text(.035,.925,'Target − reference',fontweight='bold',fontsize=9)
fig.text(.97,.925,'Difference [95% CI]',ha='right',fontweight='bold',fontsize=9)
fig.text(.97,.04,'Six paired seeds · individual, unadjusted paired-t intervals',ha='right',fontsize=8,color=GREY)
save(fig,'figure3_effects')

# Entropy treatment curves and matched seed-wise differences.
methods=['entropy','entropy_aligned_fixed','entropy_aligned_lite']
names=['Entropy','Entropy + fixed DCR','Entropy + MACC-Lite']
cols=[INK,GREY,ACCENT];markers=['s','^','D'];lss=['-','--','-']
seeds=sorted({int(r['seed']) for r in D['entropy_records']})
arrays={m:np.array([[float(next(r['accuracy'] for r in D['entropy_records'] if r['method']==m and int(r['seed'])==s and int(r['phase'])==p))*100 for p in range(6)] for s in seeds]) for m in methods}
fig=plt.figure(figsize=(7.3,5.75))
ax1=fig.add_axes([.09,.555,.38,.365]);ax2=fig.add_axes([.59,.555,.38,.365])
for i,m in enumerate(methods):
    a=arrays[m];ax1.errorbar(range(10,61,10),a.mean(0),yerr=a.std(0,ddof=1),
        color=cols[i],marker=markers[i],ls=lss[i],ms=3.5,lw=1.2,capsize=2,elinewidth=.65,label=names[i])
    if i:
        diff=a-arrays['entropy']
        ax2.errorbar(range(20,61,10),diff.mean(0)[1:],yerr=diff.std(0,ddof=1)[1:],
            color=cols[i],marker=markers[i],ls=lss[i],ms=4,lw=1.2,capsize=3,elinewidth=.7,label=['Fixed DCR','MACC-Lite'][i-1])
ax1.set(xlim=(8,63),ylim=(27,77),xticks=range(10,61,10),yticks=[30,40,50,60,70],xlabel='Annotated pool (%)',ylabel='Clean top-1 accuracy (%)')
ax2.set(xlim=(18,62),ylim=(-.75,2.45),xticks=range(20,61,10),yticks=[-.5,0,.5,1,1.5,2],xlabel='Annotated pool (%)',ylabel='Paired gain over entropy (pp)')
ax2.axhline(0,color='#8d9296',ls=(0,(2,3)),lw=.7)
for ax in [ax1,ax2]:tidy(ax);ax.legend(loc='upper left',frameon=False,fontsize=7.5,handlelength=2.4)
panel(ax1,'a','Matched entropy treatments');panel(ax2,'b','Budget-dependent gains')

ax=fig.add_axes([.12,.115,.83,.31]);tidy(ax)
final=np.array([arrays[m][:,-1] for m in methods]).T
for row in final:
    ax.plot(range(3),row,color='#b4babd',lw=.7,zorder=1)
    for j,v in enumerate(row):ax.plot(j,v,marker='o',ms=3.7,mfc='white',mec=cols[j],mew=.9,zorder=3)
for j,m in enumerate(methods):
    vals=arrays[m][:,-1];mu=vals.mean();sd=vals.std(ddof=1)
    ax.errorbar(j+.085,mu,yerr=sd,fmt=markers[j],ms=6,color=cols[j],lw=1.5,capsize=4,zorder=5)
    ax.text(j,74.25,f'{mu:.3f} ± {sd:.3f}%',ha='center',va='center',fontweight='bold',color=cols[j],fontsize=9)
ax.set(xlim=(-.3,2.32),ylim=(70.8,74.6),xticks=range(3),xticklabels=names,
       yticks=[71,72,73,74],ylabel='Clean top-1 accuracy (%)')
panel(ax,'c','Six paired seeds at the final 60% budget')
fig.text(.95,.015,'Open circles: individual seeds · Connected points: same seed · Filled markers and bars: mean ± SD',ha='right',fontsize=7.7,color=GREY)
save(fig,'figure4_entropy')

print('Rebuilt three numerical figures from fixed recorded data; no training or inference family changes.')
