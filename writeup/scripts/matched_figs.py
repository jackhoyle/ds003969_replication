"""Figures for the matched-sample section, from results/matched/*.csv (written by matched_sample.py).

Usage (from writeup/): python scripts/matched_figs.py
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

R = 'results/matched'
SOURCES = [('paper (Zenodo)', "Paper's own values", '#4a3aa7'), ('matched, no ICA', 'No ICA', '#2a78d6'),
           ('matched, capped ICA', 'ICA, capped; EOG $z{>}3$', '#eb6834')]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})

t = pd.read_csv(f'{R}/matched_group_tests.csv')
t = t[t.contrast != 'Welch ANOVA (4 groups)']


def panel(ax, band, cond, contrasts, title):
    for i, (label, contrast) in enumerate(contrasts):
        for j, (src, name, col) in enumerate(SOURCES):
            r = t[(t.source == src) & (t.band == band) & (t.cond == cond) & (t.contrast == contrast)].iloc[0]
            se = abs(r.diff_db / r.stat)
            ax.errorbar(r.diff_db, i + (j - 1) * 0.22, xerr=1.96 * se, fmt='o', ms=4.5, color=col, lw=1.6,
                        mec='white', mew=0.8, label=name if i == 0 else None)
    ax.axvline(0, color='#9a9993', lw=0.8, zorder=0)
    ax.set_yticks(range(len(contrasts)))
    ax.set_yticklabels([c for c, _ in contrasts])
    ax.set_ylim(len(contrasts) - 0.5, -0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel('difference (dB)')
    ax.set_title(title, loc='left', fontsize=9.5)
    ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)


G = [(f'{g} $-$ CTR', f'{g} - CTR') for g in ['HYT', 'ISY', 'VIP']] + [('All med. $-$ CTR', 'all meditators - CTR')]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.0), constrained_layout=True, width_ratios=[1, 1, 1.05])
panel(axs[0], 'gamma', 'trait', G, '(a) Gamma 60–110 Hz, trait')
panel(axs[1], 'gamma', 'MED', G, '(b) Gamma 60–110 Hz, MED')
panel(axs[2], 'alpha', 'MED', [('VIP $-$ CTR', 'VIP - CTR'), ('VIP $-$\n(HYT+ISY)', 'VIP vs HYT+ISY')],
      '(c) Alpha 7–11 Hz, MED'.replace(' 7–11 Hz', ''))
fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
fig.savefig('figures/matched_headline.pdf')
fig.savefig('figures/matched_headline.png', dpi=110)

# per-subject gamma in MED vs the paper's values
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True, sharex=True, sharey=True)
gcol = {'CTR': '#2a78d6', 'HYT': '#eb6834', 'ISY': '#1baf7a', 'VIP': '#eda100'}
for ax, (sfx, title) in zip(axs, [('_matchsample_noica_v0', '(a) No ICA'),
                                  ('_matchsample_v0', '(b) ICA, capped; EOG $z{>}3$')]):
    d = pd.read_csv(f'{R}/per_subject_vs_paper_{sfx}.csv')
    d = d[(d.band == 'gamma') & (d.cond == 'MED')]
    offset = (d.db_paper - d.db).mean()
    lo, hi = d.db_paper.min() - 1, d.db_paper.max() + 1
    ax.plot([lo, hi], [lo, hi], color='#9a9993', lw=0.8, zorder=0)
    for g, gg in d.groupby('group'):
        ax.scatter(gg.db + offset, gg.db_paper, s=22, color=gcol[g], edgecolor='white', linewidth=0.6, label=g)
    r = np.corrcoef(d.db, d.db_paper)[0, 1]
    ax.set_title(f'{title}: $r={r:.2f}$', loc='left', fontsize=9.5)
    ax.set_xlabel('this replication (dB, shifted by mean offset)')
    ax.grid(color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
axs[0].set_ylabel("paper's value (dB)")
fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
fig.savefig('figures/matched_per_subject.pdf')
fig.savefig('figures/matched_per_subject.png', dpi=110)
