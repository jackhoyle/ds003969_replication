"""DFA/fEI contrasts on the paper's matched sample (both matched runs), as in the write-up: mixed models with a
random intercept per subject on subject-level means over good channels. Also the time-on-task control.

Usage (from code/): python ../writeup/scripts/matched_criticality.py <out_dir>
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from compute_criticality import criticality_dir_for, load_all
from compute_psd_epochs import bids_root

warnings.simplefilter('ignore')
out = Path(sys.argv[1])
p = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
group_of = dict(zip(p.participant_id.str.removeprefix('sub-'), p.group))
RUNS = {'matched, capped ICA': '_matchsample_v0', 'matched, no ICA': '_matchsample_noica_v0'}


def fit(formula, data, term):
    r = smf.mixedlm(formula, data, groups=data.subject).fit(reml=True)
    return r.params[term], r.bse[term], r.pvalues[term], data.subject.nunique()


rows = []
for run, sfx in RUNS.items():
    d = load_all(criticality_dir_for(sfx))
    d = d[~d.bad]
    s = d.groupby(['subject', 'task', 'band'])[['dfa', 'fei']].mean().reset_index()
    s['group'] = s.subject.map(group_of)
    s['medit'] = (s.group != 'ctr').astype(int)
    print(f'{run}: {s[["subject", "task"]].drop_duplicates().shape[0]} recordings with >=1 clean 180 s epoch, '
          f'{s.subject.nunique()} subjects')
    for band in ['theta', 'alpha', 'beta', 'broadband', 'gamma']:
        for metric in ['dfa', 'fei']:
            b = s[s.band == band].dropna(subset=[metric])
            med = b[b.medit == 1].copy()
            med['med'] = med.task.isin(['med1breath', 'med2']).astype(int)
            mm = b[b.task.isin(['med1breath', 'med2'])].copy()
            mm['late'] = (mm.task == 'med2').astype(int)
            th = b[b.task.isin(['think1', 'think2'])].copy()
            tests = {
                'MED2 - MED1 (meditators)': (f'{metric} ~ late', mm[mm.medit == 1], 'late'),
                'MED2 - MED1 (controls)': (f'{metric} ~ late', mm[mm.medit == 0], 'late'),
                'block x meditator interaction': (f'{metric} ~ late * medit', mm, 'late:medit'),
                'MED - IMW (meditators)': (f'{metric} ~ med', med, 'med'),
                'meditators - CTR, IMW': (f'{metric} ~ medit', th, 'medit'),
                'meditators - CTR, MED': (f'{metric} ~ medit', mm, 'medit'),
            }
            for name, (f, data, term) in tests.items():
                try:
                    est, se, pv, n = fit(f, data, term)
                except Exception:
                    continue
                rows.append(dict(run=run, band=band, metric=metric, test=name, est=est, se=se, p=pv, n_subjects=n))
r = pd.DataFrame(rows)
out.mkdir(parents=True, exist_ok=True)
r.to_csv(out / 'matched_criticality.csv', index=False)
r['cell'] = r.apply(lambda x: f'{x.est:+.3f} ({x.p:.3f})', axis=1)
pd.set_option('display.width', 220)
for metric in ['dfa', 'fei']:
    print(f'\n## {metric.upper()}: estimate (p)')
    print(r[r.metric == metric].pivot_table(index=['test', 'band'], columns='run', values='cell', aggfunc='first')
          .to_string())
print('\nContrasts with p < .05:')
print(r[r.p < .05][['run', 'metric', 'band', 'test', 'est', 'p', 'n_subjects']].round(3).to_string(index=False))
print(f'\nTotal contrasts: {len(r)}')
