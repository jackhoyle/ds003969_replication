"""LaTeX tables for the results section, from the CSVs in results/matched/ (written by matched_sample.py,
cluster_tests.py and experience.py; the criticality tables come from criticality_final.py).

Usage (from writeup/): python scripts/results_tables.py
Writes results/tables/{roi,clusters,experience}.tex, \\input by replication_writeup.tex.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path('results/matched')
OUT = Path('results/tables')
OUT.mkdir(exist_ok=True)
SOURCES = ['paper (Zenodo)', 'matched, no ICA', 'matched, capped ICA']
RUNS = ['matched, no ICA', 'matched, capped ICA']


def pval(p):
    s = f'{p:.3f}'.lstrip('0') if p >= 0.001 else '$<$.001'
    return rf'\textbf{{{s}}}' if p < 0.05 else s


def write(name, lines):
    text = re.sub(r'(?<![\w$-])-(?=\d)', '$-$', '\n'.join(lines))  # typeset minus signs
    (OUT / f'{name}.tex').write_text(text + '\n')
    print(f'wrote {OUT / name}.tex')


# --- parieto-occipital cluster (ROI) group differences
t = pd.read_csv(R / 'matched_group_tests.csv')
rows = [('gamma', c, k) for c in ['MED', 'MW', 'trait']
        for k in ['HYT - CTR', 'ISY - CTR', 'VIP - CTR', 'all meditators - CTR']]
rows += [('alpha', c, k) for c in ['MED', 'MW'] for k in ['VIP - CTR', 'VIP vs HYT+ISY']]
label = {'all meditators - CTR': r'All meditators $-$ CTR', 'VIP vs HYT+ISY': r'VIP $-$ (HYT+ISY)'}
lines = [r'\begin{tabular}{@{}lll' + 'r' * 3 + '@{}}', r'\toprule',
         r'Band & Condition & Contrast & Paper\textquotesingle s values & No ICA & Capped ICA \\', r'\midrule']
prev = None
for band, cond, k in rows:
    if prev and band != prev[0]:
        lines.append(r'\midrule')
    cells = []
    for src in SOURCES:
        r = t[(t.source == src) & (t.band == band) & (t.cond == cond) & (t.contrast == k)].iloc[0]
        cells.append(f'{r.diff_db:+.2f} ({pval(r.p)})')
    show_band = band.capitalize() + (r' 60--110\,Hz' if band == 'gamma' else r' 7--11\,Hz') \
        if prev is None or band != prev[0] else ''
    show_cond = ('IMW' if cond == 'MW' else cond) if prev is None or (band, cond) != prev else ''
    lines.append(f"{show_band} & {show_cond} & {label.get(k, k.replace(' - ', r' $-$ '))} & "
                 + ' & '.join(cells) + r' \\')
    prev = (band, cond)
lines += [r'\bottomrule', r'\end{tabular}']
write('roi', lines)

# --- cluster permutation tests: smallest cluster p (number of electrodes)
c = pd.read_csv(R / 'cluster_tests.csv')
best = c.sort_values('p').groupby(['run', 'band', 'cond', 'comparison'], sort=False).first().reset_index()
rows = [('gamma', k) for k in ['HYT - CTR', 'ISY - CTR', 'VIP - CTR', 'all meditators - CTR']]
rows += [('alpha', k) for k in ['VIP - CTR', 'VIP - HYT', 'VIP - ISY', 'all meditators - CTR']]
conds = ['MED', 'IMW', 'trait']
lines = [r'\begin{tabular}{@{}ll' + 'c' * 6 + '@{}}', r'\toprule',
         r' & & \multicolumn{3}{c}{No ICA} & \multicolumn{3}{c}{Capped ICA} \\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
         r'Band & Contrast & ' + ' & '.join(conds * 2) + r' \\', r'\midrule']
prev = None
for band, k in rows:
    if prev and band != prev:
        lines.append(r'\midrule')
    cells = []
    for run in RUNS:
        for cond in conds:
            r = best[(best.run == run) & (best.band == band) & (best.cond == cond) & (best.comparison == k)]
            cells.append('--' if r.empty or np.isnan(r.p.iloc[0]) else
                         f'{pval(r.p.iloc[0])} ({int(r.n_channels.iloc[0])})')
    show_band = (band.capitalize() if band != prev else '')
    lines.append(f"{show_band} & {label.get(k, k.replace(' - ', r' $-$ '))} & " + ' & '.join(cells) + r' \\')
    prev = band
lines += [r'\bottomrule', r'\end{tabular}']
write('clusters', lines)

# --- gamma vs experience
e = pd.read_csv(R / 'experience_gamma.csv')
names = {"paper's values": r"Paper\textquotesingle s values",
         "paper's values, 47 in ds003969": r"\quad without HYT26",
         'no ICA': 'No ICA', 'capped ICA': 'Capped ICA'}
lines = [r'\begin{tabular}{@{}lrrrlcc@{}}', r'\toprule',
         r' & & & & & \multicolumn{2}{c}{95\% CI, outliers\ldots} \\', r'\cmidrule(l){6-7}',
         r'Gamma values & $n$ & Pearson $r$ & Skipped $r$ & Outliers & removed once '
         r'& redetected \\', r'\midrule']
for _, r in e.iterrows():
    lines.append(f'{names[r.source]} & {r.n} & {r.pearson_r:.2f} & {r.skipped_r:.2f} & '
                 f"{r.outliers if isinstance(r.outliers, str) else 'none'} & "
                 f'[{r.ci_fixed_low:.2f}, {r.ci_fixed_high:.2f}] & '
                 f'[{r.ci_redetected_low:.2f}, {r.ci_redetected_high:.2f}]' + r' \\')
    if r.source == "paper's values, 47 in ds003969":
        lines.append(r'\midrule')
lines += [r'\midrule', r'Paper (Fig.~4D) & 48 & & 0.33 & & [0.06, 0.53] & \\', r'\bottomrule', r'\end{tabular}']
write('experience', lines)
