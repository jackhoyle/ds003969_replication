import sys, numpy as np, pandas as pd, warnings
import statsmodels.formula.api as smf
from compute_psd_epochs import bids_root
warnings.simplefilter('ignore')
p=pd.read_csv(bids_root/'participants.tsv',sep='\t'); g=dict(zip(p.participant_id.str[4:].astype(int),p.group))
rows=[]
for v in ['_noica','','_v2','_v3']:
    d=pd.read_csv(bids_root/f'outputs{v}'/'criticality.csv'); d=d[~d.bad]
    s=d.groupby(['subject','task','band'])[['dfa','fei']].mean().reset_index(); s['group']=s.subject.map(g)
    s=s[s.group!='tm']
    for band in ['theta','alpha','beta','broadband','gamma']:
        for m in ['dfa','fei']:
            b=s[s.band==band].dropna(subset=[m])
            med=b[b.group.isin(['htr','sny','vip'])].copy()
            med['med']=med.task.isin(['med1breath','med2']).astype(int)
            def fit(formula,data,term):
                try:
                    r=smf.mixedlm(formula,data,groups=data.subject).fit(reml=True)
                    return r.params[term],r.bse[term],r.pvalues[term]
                except Exception as e: return np.nan,np.nan,np.nan
            out={}
            out['med_vs_think']=fit(f'{m} ~ med',med,'med')
            out['med2_vs_med1']=fit(f'{m} ~ C(task, Treatment("med1breath"))',med,'C(task, Treatment("med1breath"))[T.med2]')
            th=b[b.task.isin(['think1','think2'])].copy(); th['medit']=(th.group!='ctr').astype(int)
            out['rest_meditators_vs_ctr']=fit(f'{m} ~ medit',th,'medit')
            mm=b[b.task.isin(['med1breath','med2'])].copy(); mm['medit']=(mm.group!='ctr').astype(int)
            out['med_meditators_vs_ctr']=fit(f'{m} ~ medit',mm,'medit')
            for k,(e,se,pv) in out.items(): rows.append(dict(run=v or '_base',band=band,metric=m,test=k,est=e,se=se,p=pv))
    print(v, 'recordings', s[['subject','task']].drop_duplicates().shape[0], 'subjects', s.subject.nunique(), s.drop_duplicates(['subject','task']).groupby('group').size().to_dict())
r=pd.DataFrame(rows); r.to_csv(sys.argv[1],index=False)
pd.set_option('display.width',250)
x=r[r.metric=='dfa'].pivot_table(index=['test','band'],columns='run',values=['est','p']); print(x.round(3).to_string())
x=r[r.metric=='fei'].pivot_table(index=['test','band'],columns='run',values=['est','p']); print(x.round(3).to_string())
