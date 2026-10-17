"""Headline spectral results on the paper's matched sample, compared with the paper's own per-subject values.

Usage (from the code/ directory):
    python ../writeup/scripts/matched_sample.py <zenodo_dir> <out_dir>
<zenodo_dir> holds median_gamma_power.csv, median_alpha_power.csv and
mw_medit_combined_parieto_occip_Gamma_correl_exp.csv from doi 10.5281/zenodo.57911.

Per subject: mean power over an 8-electrode parieto-occipital cluster, in dB, for gamma (60-110 Hz) and alpha
(7-11 Hz), in MED (med2), MW (think2), BRE (med1breath) and trait (mean of MED and MW, as in the paper).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

from compute_psd_epochs import bids_root, load_all, paper_subjects, paper_subjects_csv, psd_dir_for

warnings.simplefilter('ignore')
zen, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

PO = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
BANDS = {'gamma': (60, 110), 'alpha': (7, 11)}
CONDS = {'MED': 'med2', 'MW': 'think2', 'BRE': 'med1breath'}
GROUP = {'ctr': 'CTR', 'htr': 'HYT', 'sny': 'ISY', 'vip': 'VIP'}
RUNS = {'matched, capped ICA': '_matchsample_v0', 'matched, no ICA': '_matchsample_noica_v0'}

participants = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
group_of = dict(zip(participants.participant_id.str.removeprefix('sub-'), participants.group.map(GROUP)))


def subject_values(run_suffix, key='psd_mean', only=None):
    """Long table: subject, group, band, cond, db (mean linear power over cluster and band, in dB)."""
    by_task, freqs, ch_names, subjects = load_all(psd_dir_for(run_suffix), key=key, only_subjects=only)
    ci = [list(ch_names).index(c) for c in PO]
    rows = []
    for cond, task in CONDS.items():
        for band, (lo, hi) in BANDS.items():
            m = (freqs >= lo) & (freqs <= hi)
            x = np.nanmean(by_task[task][:, ci][:, :, m].mean(-1), axis=1)
            rows += [dict(subject=s, band=band, cond=cond, db=10 * np.log10(v)) for s, v in zip(subjects[task], x)]
    df = pd.DataFrame(rows)
    trait = df[df.cond.isin(['MED', 'MW'])].groupby(['subject', 'band']).db.mean().reset_index().assign(cond='trait')
    df = pd.concat([df, trait])
    df['group'] = df.subject.map(group_of)
    return df


def paper_values():
    """The paper's per-subject values (their 'Posterior' column; trait = mean of MED and MW, like ours)."""
    ids = pd.read_csv(zen / 'mw_medit_combined_parieto_occip_Gamma_correl_exp.csv').Sujet
    mapping = pd.read_csv(paper_subjects_csv).set_index('lab_id').participant_id.str.removeprefix('sub-')
    subj = dict(zip(range(1, 65), ids.map(mapping)))               # paper row index -> ds003969 id (NaN: HYT26)
    g = pd.read_csv(zen / 'median_gamma_power.csv').rename(columns={'Posterior': 'db'}).assign(band='gamma')
    a = pd.read_csv(zen / 'median_alpha_power.csv').rename(columns={'Alpha': 'db'}).assign(band='alpha')
    df = pd.concat([g, a])[['Subjects', 'Group', 'Condition', 'band', 'db']]
    df = df.rename(columns={'Group': 'group', 'Condition': 'cond'})
    df['subject'] = df.Subjects.map(subj)
    trait = (df[df.cond.isin(['MED', 'MW'])].groupby(['Subjects', 'group', 'band', 'subject'], dropna=False)
             .db.mean().reset_index().assign(cond='trait'))
    return pd.concat([df, trait]).drop(columns='Subjects')


def group_tests(df, source):
    rows = []
    for (band, cond), d in df.groupby(['band', 'cond']):
        groups = [d[d.group == g].db.dropna() for g in ['CTR', 'HYT', 'ISY', 'VIP']]
        w = anova_oneway(groups, use_var='unequal')
        rows.append(dict(source=source, band=band, cond=cond, contrast='Welch ANOVA (4 groups)',
                         diff_db=np.nan, stat=w.statistic, p=w.pvalue))
        ctr = d[d.group == 'CTR'].db
        others = {g: d[d.group == g].db for g in ['HYT', 'ISY', 'VIP']}
        others['all meditators'] = d[d.group != 'CTR'].db
        if band == 'alpha':
            others['VIP vs HYT+ISY'] = None
        for name, x in others.items():
            if name == 'VIP vs HYT+ISY':
                x, ref = d[d.group == 'VIP'].db, d[d.group.isin(['HYT', 'ISY'])].db
            else:
                ref = ctr
                name = f'{name} - CTR'
            t, p = stats.ttest_ind(x.dropna(), ref.dropna(), equal_var=False)
            # Hedges' g: difference / pooled SD (bias-corrected)
            n1, n2 = x.notna().sum(), ref.notna().sum()
            sp = np.sqrt(((n1 - 1) * x.var() + (n2 - 1) * ref.var()) / (n1 + n2 - 2))
            g_ = (x.mean() - ref.mean()) / sp * (1 - 3 / (4 * (n1 + n2) - 9))
            rows.append(dict(source=source, band=band, cond=cond, contrast=name, diff_db=x.mean() - ref.mean(),
                             hedges_g=g_, stat=t, p=p))
    return pd.DataFrame(rows)


ours = {name: subject_values(sfx) for name, sfx in RUNS.items()}
paper = paper_values()

# 0. sanity: the matched runs reproduce the full-sample runs' PSDs for the same subjects
print('## 0. Matched runs vs full-sample runs, same subjects (max |diff| in dB)')
only = paper_subjects()
for new, old in [('_matchsample_noica_v0', '_noica'), ('_matchsample_v0', '_v2'), ('_matchsample_v0', '')]:
    a = subject_values(new).set_index(['subject', 'band', 'cond']).db
    b = subject_values(old, only=only).set_index(['subject', 'band', 'cond']).db
    print(f'  {new} vs {old or "_base"}: {(a - b).abs().max():.4f} dB over {len(a)} values')

# 1. group tests, ours and the paper's
tests = pd.concat([group_tests(df, name) for name, df in ours.items()] + [group_tests(paper, 'paper (Zenodo)')])
tests.to_csv(out / 'matched_group_tests.csv', index=False)
pd.set_option('display.width', 220)
for band in ['gamma', 'alpha']:
    print(f'\n## 1. {band}: group differences (dB), Welch t-tests; p in brackets')
    t = tests[(tests.band == band)].copy()
    t['cell'] = t.apply(lambda r: (f'{r.diff_db:+.2f} ({r.p:.3f})' if np.isfinite(r.diff_db)
                                   else f'F={r.stat:.2f} ({r.p:.3f})'), axis=1)
    conds = ['MED', 'MW', 'trait'] + (['BRE'] if band == 'gamma' else [])
    print(t[t.cond.isin(conds)].pivot_table(index=['cond', 'contrast'], columns='source', values='cell',
                                            aggfunc='first').to_string())

# 2. per-subject agreement with the paper's values
print('\n## 2. Per-subject agreement with the paper (Pearson r, Spearman rho, mean offset in dB)')
rows = []
for name, df in ours.items():
    j = df.merge(paper, on=['subject', 'band', 'cond'], suffixes=('', '_paper')).dropna(subset=['db', 'db_paper'])
    for (band, cond), d in j.groupby(['band', 'cond']):
        rows.append(dict(run=name, band=band, cond=cond, n=len(d), r=stats.pearsonr(d.db, d.db_paper)[0],
                         rho=stats.spearmanr(d.db, d.db_paper)[0], offset=(d.db_paper - d.db).mean(),
                         offset_sd=(d.db_paper - d.db).std()))
    j.to_csv(out / f'per_subject_vs_paper_{RUNS[name]}.csv', index=False)
agree = pd.DataFrame(rows)
agree.to_csv(out / 'matched_agreement.csv', index=False)
print(agree.round(3).to_string(index=False))

# 3. group means per source, to see whether differences come from CTR or meditators
print('\n## 3. Group means, gamma (dB; paper values carry a constant offset)')
gm = pd.concat([df.assign(source=n) for n, df in ours.items()] + [paper.assign(source='paper (Zenodo)')])
print(gm[(gm.band == 'gamma') & gm.cond.isin(['MED', 'MW'])].groupby(['source', 'cond', 'group']).db.mean()
      .unstack('group').round(2).to_string())
