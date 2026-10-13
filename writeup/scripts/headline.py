import sys, numpy as np, pandas as pd, warnings
from scipy import stats
from compute_psd_epochs import load_all, psd_dir_for, bids_root
warnings.simplefilter('ignore')
p=pd.read_csv(bids_root/'participants.tsv',sep='\t'); g=dict(zip(p.participant_id.str[4:],p.group))
PO=['O1','Oz','O2','PO7','PO3','POz','PO4','PO8']
bands={'alpha7-11':(7,11),'gamma60-110':(60,110)}
rows=[]
for v in ['_noica','','_v2','_v3']:
    bt,f,ch,subs=load_all(psd_dir_for(v),key='psd_mean')
    ci=[list(ch).index(c) for c in PO]
    def bp(t,b):
        lo,hi=bands[b]; m=(f>=lo)&(f<=hi)
        x=bt[t][:,ci][:,:,m].mean(-1)            # subj x chan (linear)
        return pd.Series(10*np.log10(np.nanmean(x,1)), index=subs[t])
    for b in bands:
        cond={'med2':bp('med2',b),'think2':bp('think2',b)}
        cond['trait(med2+think2)']=pd.concat([cond['med2'],cond['think2']],axis=1).mean(1)
        for cn,s in cond.items():
            df=pd.DataFrame({'y':s,'g':s.index.map(g)})
            c=df[df.g=='ctr'].y
            for grp in ['htr','sny','vip']:
                a=df[df.g==grp].y; t,pv=stats.ttest_ind(a,c,equal_var=False)
                rows.append(dict(run=v or '_base',band=b,cond=cn,group=grp,diff_dB=a.mean()-c.mean(),t=t,p=pv,n=len(a),nc=len(c)))
            if b.startswith('alpha'):
                a=df[df.g=='vip'].y; o=df[df.g.isin(['htr','sny'])].y; t,pv=stats.ttest_ind(a,o,equal_var=False)
                rows.append(dict(run=v or '_base',band=b,cond=cn,group='vip-vs-htr+sny',diff_dB=a.mean()-o.mean(),t=t,p=pv,n=len(a),nc=len(o)))
r=pd.DataFrame(rows); r.to_csv(sys.argv[1],index=False)
pd.set_option('display.width',200)
print(r.pivot_table(index=['band','cond','group'],columns='run',values=['diff_dB','p']).round(3).to_string())
