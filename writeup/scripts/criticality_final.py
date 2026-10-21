"""Criticality hypothesis tests for the write-up, as in exploration/261020_criticality_final.ipynb, for both
matched runs.

Per subject, task and band: DFA and fEI averaged over epochs and good channels. OLS with standard errors clustered by
subject, controlling for session order with second_pair (1 for the pair of blocks recorded after the break):
  H1 (trait group difference), med2 and think2, all four groups:  {dfa, fei} ~ C(group) + second_pair
     plus a joint Wald test that the three group coefficients are zero;
  H2 (DFA rises during meditation; DFA only), meditators, all four blocks:
     dfa ~ C(med_task) * C(second_in_pair) + second_pair
     where the interaction is (med2 - med1breath) - (think2 - think1); also reports med2 - med1breath.

Usage (from writeup/): python scripts/criticality_final.py [--epoch-len 90]
Writes results/matched/criticality_final.csv, results/tables/criticality_h{1,2}.tex and
figures/criticality_final{,_noica}.pdf (+ .png). With a non-default --epoch-len (a robustness check), writes only
results/matched/criticality_final_{len}s.csv (needs compute_criticality.main(..., epoch_len=90) run for both runs).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import mne
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from compute_criticality import EPOCH_LEN, criticality_dir_for, load_all  # noqa: E402
from compute_psd_epochs import bids_root  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument('--epoch-len', type=float, default=EPOCH_LEN, help='epoch length in s (default: %(default)s)')
epoch_len = parser.parse_args().epoch_len
TAG = '' if epoch_len == EPOCH_LEN else f'_{epoch_len:g}s'

RUNS = {'capped ICA': '_matchsample_v1', 'no ICA': '_matchsample_noica_v1'}
BANDS = ['alpha', 'beta', 'gamma']
GROUP = {'ctr': 'CTR', 'htr': 'HYT', 'sny': 'ISY', 'vip': 'VIP'}
TERMS = {
    'H1': [('C(group)[T.htr]', 'HYT $-$ CTR'), ('C(group)[T.sny]', 'ISY $-$ CTR'), ('C(group)[T.vip]', 'VIP $-$ CTR'),
           ('second_pair', 'Second pair')],
    'H2': [('C(med_task)[T.1]', 'MED $-$ IMW (first block)'), ('C(second_in_pair)[T.1]', 'think2 $-$ think1'),
           ('C(med_task)[T.1]:C(second_in_pair)[T.1]', r'(med2 $-$ med1) $-$ (think2 $-$ think1)'),
           ('med2 - med1breath', 'med2 $-$ med1breath'), ('second_pair', 'Second pair')],
}

participants = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
participants['subject'] = participants.participant_id.str.removeprefix('sub-')


def session_order(subjects):
    """time = rank of each block's recording start (meas_date) within the subject, 0-3."""
    rows = []
    for s in subjects:
        for bdf in sorted((bids_root / f'sub-{s}' / 'eeg').glob(f'sub-{s}_task-*_eeg.bdf')):
            info = mne.io.read_raw_bdf(bdf, preload=False, verbose='error').info
            rows.append(dict(subject=s, task=bdf.name.split('_task-')[1].split('_')[0], start=info['meas_date']))
    so = pd.DataFrame(rows)
    so['time'] = so.groupby('subject').start.rank(method='first').astype(int) - 1
    assert (so.groupby('subject').time.apply(sorted).map(tuple) == (0, 1, 2, 3)).all()
    so['second_pair'] = (so.time >= 2).astype(int)
    so['second_in_pair'] = (so.time % 2).astype(int)
    assert (so.second_in_pair == so.task.isin(['med2', 'think2'])).all()
    return so[['subject', 'task', 'second_pair', 'second_in_pair']]


def load(suffix, order=None):
    df = load_all(criticality_dir_for(suffix, epoch_len))
    df = df[~df.bad].merge(participants[['subject', 'group']], on='subject', how='left', validate='many_to_one')
    df = df[df.group != 'tm']
    order = session_order(sorted(df.subject.unique())) if order is None else order
    df = df.merge(order, on=['subject', 'task'], how='left', validate='many_to_one')
    assert df.second_pair.notna().all()
    df['med_task'] = df.task.isin(['med1breath', 'med2']).astype(int)
    keys = ['task', 'subject', 'group', 'med_task', 'second_pair', 'second_in_pair', 'band']
    return df.groupby(keys)[['dfa', 'fei']].mean().reset_index(), order


def fit(formula, d):
    return ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': pd.factorize(d.subject)[0]})


rows, order = [], None
for run, suffix in RUNS.items():
    s, order = load(suffix, order)
    for band in BANDS:
        b = s[s.band == band]
        h1 = b[b.task.isin(['med2', 'think2'])]
        h2 = b[b.group != 'ctr']
        models = {('H1', 'dfa'): (h1, 'dfa ~ C(group) + second_pair'),
                  ('H1', 'fei'): (h1, 'fei ~ C(group) + second_pair'),
                  ('H2', 'dfa'): (h2, 'dfa ~ C(med_task) * C(second_in_pair) + second_pair')}
        for (hyp, metric), (data, formula) in models.items():
            data = data.dropna(subset=[metric])  # fEI is only computed on channels with DFA > 0.6
            m = fit(formula, data)
            n = dict(run=run, band=band, hypothesis=hyp, metric=metric, n_obs=int(m.nobs),
                     n_subjects=data.subject.nunique())
            for term, _ in TERMS[hyp]:
                if term == 'med2 - med1breath':
                    names = list(m.params.index)
                    c = np.zeros(len(names))
                    c[names.index('C(second_in_pair)[T.1]')] = 1
                    c[names.index('C(med_task)[T.1]:C(second_in_pair)[T.1]')] = 1
                    t = m.t_test(c)
                    est, se, p = float(t.effect[0]), float(t.sd[0, 0]), float(t.pvalue)
                else:
                    est, se, p = m.params[term], m.bse[term], m.pvalues[term]
                rows.append(dict(**n, term=term, est=est, se=se, p=p))
            if hyp == 'H1':
                w = m.wald_test(' = '.join(['C(group)[T.htr]', 'C(group)[T.sny]', 'C(group)[T.vip]', '0']),
                                scalar=True)
                rows.append(dict(**n, term='joint group test', est=float(w.statistic), se=np.nan,
                                 p=float(w.pvalue)))

r = pd.DataFrame(rows)
r.to_csv(f'results/matched/criticality_final{TAG}.csv', index=False)
pd.set_option('display.width', 220)
print(r.round(4).to_string(index=False))
if TAG:  # robustness check at another epoch length: results CSV only
    sys.exit()


# --- tables (capped ICA, the baseline run)
def pval(p):
    s = f'{p:.3f}'.lstrip('0') if p >= 0.001 else '$<$.001'
    return rf'\textbf{{{s}}}' if p < 0.05 else s


def num(x):
    return f'{x:+.3f}'.replace('-', '$-$')


METRIC_NAME = {'dfa': 'DFA exponent', 'fei': 'fEI'}


def term_rows(base, hyp):
    lines = []
    for term, label in TERMS[hyp]:
        cells = []
        for band in BANDS:
            x = base[(base.band == band) & (base.term == term)].iloc[0]
            cells += [f'{num(x.est)} ({x.se:.3f})', pval(x.p)]
        lines.append(f'{label} & ' + ' & '.join(cells) + r' \\')
    if hyp == 'H1':
        cells = []
        for band in BANDS:
            x = base[(base.band == band) & (base.term == 'joint group test')].iloc[0]
            cells += [rf'$\chi^2_3 = {x.est:.2f}$', pval(x.p)]
        lines.append('All groups equal & ' + ' & '.join(cells) + r' \\')
    nn = base.drop_duplicates('band').set_index('band')
    lines.append('Observations (subjects) & ' + ' & '.join(
        rf'\multicolumn{{2}}{{c}}{{{nn.n_obs[b]} ({nn.n_subjects[b]})}}' for b in BANDS) + r' \\')
    return lines


for hyp, metrics in [('H1', ['dfa', 'fei']), ('H2', ['dfa'])]:
    lines = [r'\begin{tabular}{@{}l' + 'rr' * len(BANDS) + '@{}}', r'\toprule',
             ' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{b.capitalize()}}}' for b in BANDS) + r' \\',
             ''.join(rf'\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}' for i in range(len(BANDS))),
             'Term & ' + ' & '.join(['Estimate (SE)', '$p$'] * len(BANDS)) + r' \\']
    for metric in metrics:
        lines.append(r'\midrule')
        if len(metrics) > 1:
            lines.append(rf'\multicolumn{{{1 + 2 * len(BANDS)}}}{{@{{}}l}}{{\textit{{{METRIC_NAME[metric]}}}}} \\')
        lines += term_rows(r[(r.run == 'capped ICA') & (r.hypothesis == hyp) & (r.metric == metric)], hyp)
    lines += [r'\bottomrule', r'\end{tabular}']
    Path(f'results/tables/criticality_{hyp.lower()}.tex').write_text('\n'.join(lines) + '\n')

# --- figures: coefficient plots; main = capped ICA, appendix = both runs
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
COL = {'capped ICA': '#eb6834', 'no ICA': '#2a78d6'}
H1_TERMS = [t for t in TERMS['H1'] if t[0] != 'second_pair']
PANELS = [('H1', 'dfa', H1_TERMS, '(a) $H_1$: DFA', 'difference in DFA exponent'),
          ('H1', 'fei', H1_TERMS, '(b) $H_1$: fEI', 'difference in fEI'),
          ('H2', 'dfa', [TERMS['H2'][2], TERMS['H2'][3]], '(c) $H_2$: DFA, meditators', 'difference in DFA exponent')]


def coef_figure(runs, path):
    fig = plt.figure(figsize=(7.4, 6.4), layout='constrained')
    m = fig.subplot_mosaic([['a', 'b'], ['c', 'c']], height_ratios=[9, 6], width_ratios=[1.25, 1])
    axs = [m['a'], m['b'], m['c']]
    for ax, (hyp, metric, terms, title, xlabel) in zip(axs, PANELS):
        rows_ = [(band, term, label) for band in BANDS for term, label in terms]
        for i, (band, term, label) in enumerate(rows_):
            for j, run in enumerate(runs):
                x = r[(r.run == run) & (r.hypothesis == hyp) & (r.metric == metric) & (r.band == band)
                      & (r.term == term)].iloc[0]
                off = 0 if len(runs) == 1 else (j - 0.5) * 0.3
                ax.errorbar(x.est, i + off, xerr=1.96 * x.se, fmt='o', ms=4.5, color=COL[run], lw=1.6, mec='white',
                            mew=0.8, label=run if i == 0 else None)
        ax.axvline(0, color='#9a9993', lw=0.8, zorder=0)
        ax.set_yticks(range(len(rows_)))
        # the fEI panel shares the DFA panel's rows, so it only needs the band labels
        ax.set_yticklabels([f'{band}: {label}' for band, _, label in rows_] if metric == 'dfa' or hyp == 'H2'
                           else [''] * len(rows_), fontsize=8)
        ax.set_ylim(len(rows_) - 0.5, -0.5)
        for k in range(1, len(BANDS)):
            ax.axhline(k * len(terms) - 0.5, color='#e6e5e0', lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.grid(axis='x', color='#e6e5e0', lw=0.6)
        ax.set_axisbelow(True)
    if len(runs) > 1:
        fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=len(runs), frameon=False)
    for ext in ['pdf', 'png']:
        fig.savefig(f'{path}.{ext}', dpi=110)
    plt.close(fig)


coef_figure(['capped ICA'], 'figures/criticality_final')
coef_figure(['capped ICA', 'no ICA'], 'figures/criticality_final_noica')
