import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.ticker
S,W=sys.argv[1],sys.argv[2]
runs=[('_noica','No ICA'),('_base','ICA, capped; EOG $z{>}3$'),('_v3','ICA, capped; EOG $z{>}4$')]
cols=['#2a78d6','#eb6834','#1baf7a']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e'})
def panel(ax,df,items,xlabel,title):
    for i,(lab,sel) in enumerate(items):
        for j,(r,rl) in enumerate(runs):
            row=df[sel(df)&(df.run==r)].iloc[0]
            se=row.est/row.t if 't' in row else row.se
            y=i+(j-1)*0.22
            ax.errorbar(row.est,y,xerr=1.96*abs(se),fmt='o',ms=4.5,color=cols[j],lw=1.6,capsize=0,label=rl if i==0 else None,mec='white',mew=0.8)
    ax.axvline(0,color='#9a9993',lw=0.8,zorder=0)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l,_ in items]); ax.set_ylim(len(items)-0.5,-0.5); ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel(xlabel); ax.set_title(title,loc='left',fontsize=9.5)
    ax.grid(axis='x',color='#e6e5e0',lw=0.6); ax.set_axisbelow(True)
h=pd.read_csv(f'{S}/headline.csv').rename(columns={'diff_dB':'est'})
fig,axs=plt.subplots(1,3,figsize=(7.6,2.9),sharey=False,width_ratios=[1,1,1.05],constrained_layout=True)
G=[(f'{g.upper()} $-$ CTR',lambda d,g=g:d.group==g) for g in ['htr','sny','vip']]
panel(axs[0],h[(h.band=='gamma60-110')&(h.cond=='trait(med2+think2)')],G,'difference (dB)','(a) Gamma 60–110 Hz, trait')
panel(axs[1],h[(h.band=='gamma60-110')&(h.cond=='med2')],G,'difference (dB)','(b) Gamma 60–110 Hz, MED')
A=[('VIP $-$ CTR',lambda d:d.group=='vip'),('VIP $-$\n(HTR+SNY)',lambda d:d.group=='vip-vs-htr+sny')]
panel(axs[2],h[(h.band=='alpha7-11')&(h.cond=='med2')],A,'difference (dB)','(c) Alpha 7–11 Hz, MED')
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.savefig(f'{W}/figures/headline_sensitivity.pdf'); fig.savefig(f'{S}/head.png',dpi=110)
c=pd.read_csv(f'{S}/crit.csv'); c=c[c.metric=='dfa']
tests=[('med2_vs_med1','MED2 $-$ MED1 (meditators)'),('med_vs_think','MED $-$ IMW (meditators)'),('rest_meditators_vs_ctr','Meditators $-$ CTR, IMW'),('med_meditators_vs_ctr','Meditators $-$ CTR, MED')]
fig,axs=plt.subplots(1,2,figsize=(7.4,2.9),constrained_layout=True,sharey=True)
for ax,b in zip(axs,['alpha','beta']):
    panel(ax,c[c.band==b],[(l,lambda d,t=t:d.test==t) for t,l in tests],'difference in DFA exponent',f'{"(a)" if b=="alpha" else "(b)"} {b} (8–13 Hz)' if b=='alpha' else '(b) beta (13–30 Hz)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
fig.savefig(f'{W}/figures/dfa_sensitivity.pdf')
fig.savefig(f'{S}/dfa.png',dpi=110)
plt.close('all')
import matplotlib.image
