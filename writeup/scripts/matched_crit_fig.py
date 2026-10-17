"""DFA contrast figure for the criticality section, from results/matched/matched_criticality.csv
(written by matched_criticality.py).

Usage (from writeup/): python scripts/matched_crit_fig.py
"""
import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

RUNS = [('matched, no ICA', 'No ICA', '#2a78d6'), ('matched, capped ICA', 'ICA, capped; EOG $z{>}3$', '#eb6834')]
TESTS = [('MED2 - MED1 (meditators)', 'MED2 $-$ MED1, meditators'),
         ('MED2 - MED1 (controls)', 'MED2 $-$ MED1, controls'),
         ('MED - IMW (meditators)', 'MED $-$ IMW, meditators'),
         ('meditators - CTR, IMW', 'Meditators $-$ CTR, IMW'),
         ('meditators - CTR, MED', 'Meditators $-$ CTR, MED')]
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})

c = pd.read_csv('results/matched/matched_criticality.csv')
c = c[c.metric == 'dfa']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True, sharey=True)
for ax, band, title in zip(axs, ['alpha', 'beta'], ['(a) alpha (8–13 Hz)', '(b) beta (13–30 Hz)']):
    for i, (test, _) in enumerate(TESTS):
        for j, (run, label, col) in enumerate(RUNS):
            r = c[(c.band == band) & (c.test == test) & (c.run == run)].iloc[0]
            ax.errorbar(r.est, i + (j - 0.5) * 0.24, xerr=1.96 * r.se, fmt='o', ms=4.5, color=col, lw=1.6,
                        mec='white', mew=0.8, label=label if i == 0 else None)
    ax.axvline(0, color='#9a9993', lw=0.8, zorder=0)
    ax.set_yticks(range(len(TESTS)))
    ax.set_yticklabels([lab for _, lab in TESTS])
    ax.set_ylim(len(TESTS) - 0.5, -0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
    ax.set_xlabel('difference in DFA exponent')
    ax.set_title(title, loc='left', fontsize=9.5)
    ax.grid(axis='x', color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
fig.savefig('figures/matched_dfa.pdf')
fig.savefig('figures/matched_dfa.png', dpi=110)
