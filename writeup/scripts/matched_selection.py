"""Why does restricting to the paper's matched sample strengthen the gamma effect? Compare included vs excluded
subjects (full-sample no-ICA run) and look at gamma vs age.

Usage (from code/): python ../writeup/scripts/matched_selection.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from compute_psd_epochs import bids_root, load_all, paper_subjects, psd_dir_for

PO = ['O1', 'Oz', 'O2', 'PO7', 'PO3', 'POz', 'PO4', 'PO8']
by_task, freqs, ch, subs = load_all(psd_dir_for('_noica'))
ci = [list(ch).index(c) for c in PO]
m = (freqs >= 60) & (freqs <= 110)
vals = {}
for task in ['med2', 'think2']:
    x = 10 * np.log10(np.nanmean(by_task[task][:, ci][:, :, m].mean(-1), axis=1))
    vals[task] = pd.Series(x, index=subs[task])
p = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
p['subject'] = p.participant_id.str.removeprefix('sub-')
d = p.set_index('subject').join(pd.DataFrame(vals)).dropna(subset=['med2', 'think2'])
d = d[d.group != 'tm']
d['gamma'] = d[['med2', 'think2']].mean(axis=1)            # trait, dB
d['included'] = d.index.isin(paper_subjects())
d['meditator'] = d.group != 'ctr'

print('Trait gamma (dB, no ICA), included vs excluded, by group:')
print(d.groupby(['group', 'included']).agg(n=('gamma', 'size'), gamma=('gamma', 'mean'), age=('age', 'mean'))
      .round(2).to_string())
for g in ['ctr', 'htr', 'sny', 'vip']:
    x = d[d.group == g]
    if (~x.included).sum() > 1:
        t, pv = stats.ttest_ind(x[x.included].gamma, x[~x.included].gamma, equal_var=False)
        print(f'  {g}: included - excluded = {x[x.included].gamma.mean() - x[~x.included].gamma.mean():+.2f} dB (p={pv:.3f})')

print('\nGamma vs age (Spearman), no ICA:')
for name, x in [('controls', d[~d.meditator]), ('meditators', d[d.meditator]), ('all', d)]:
    r, pv = stats.spearmanr(x.age, x.gamma, nan_policy='omit')
    print(f'  {name}: rho={r:+.2f} (p={pv:.3f}, n={x.age.notna().sum()})')

print('\nMeditator - control trait gamma, full sample, with and without age/sex adjustment (OLS, HC3):')
for f in ['gamma ~ meditator', 'gamma ~ meditator + age', 'gamma ~ meditator + age + C(gender)']:
    r = smf.ols(f, d.dropna(subset=['age'])).fit(cov_type='HC3')
    print(f'  {f:38s} meditator: {r.params["meditator[T.True]"]:+.2f} dB (p={r.pvalues["meditator[T.True]"]:.4f})')
print('  matched sample only:', end=' ')
r = smf.ols('gamma ~ meditator', d[d.included]).fit(cov_type='HC3')
print(f'{r.params["meditator[T.True]"]:+.2f} dB (p={r.pvalues["meditator[T.True]"]:.4f})')
