"""Gamma vs meditation experience (the paper's Fig. 4D) on the paper's values and this replication's.

Skipped correlation (Pernet, Wilcox and Rousselet 2013): bivariate outliers found with the projection method
(MCD centre, boxplot rule with ideal fourths) are removed and Pearson's r computed on the rest. Two percentile
bootstrap CIs (2000 resamples):
  - 'outliers fixed': the outliers are removed once and Pearson's r is bootstrapped on the remaining subjects;
  - 'outliers redetected': the outlier detection is redone in every resample.
Gamma is the trait value (mean of MED and IMW) over the parieto-occipital cluster; hours from the paper's Zenodo file.
See exploration/261019_gamma_experience_age.ipynb for the age analyses.

Usage (from writeup/): python scripts/experience.py <zenodo_dir>
Needs results/matched/per_subject_vs_paper_{suffix}.csv from matched_sample.py. Writes
results/matched/experience_gamma.csv and figures/experience_gamma.pdf (+ .png).
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

zen_dir = Path(sys.argv[1])
R = Path('results/matched')
COLOURS = {'HYT': '#eb6834', 'ISY': '#1baf7a', 'VIP': '#eda100'}
N_BOOT = 2000


def _ideal_fourths(x):
    n = len(x)
    j = int(np.floor(n / 4 + 5 / 12))
    y = np.sort(x)
    g = n / 4 - j + 5 / 12
    low = (1 - g) * y[j - 1] + g * y[j]
    k = n - j + 1
    up = (1 - g) * y[k - 1] + g * y[k - 2]
    return up - low


def skipped_outliers(x, y, seed=42):
    """Boolean mask of bivariate outliers (projection method, Wilcox 2012; as in Pernet et al. 2013)."""
    X = np.column_stack([x, y])
    B = X - MinCovDet(random_state=seed).fit(X).location_
    bot = (B ** 2).sum(axis=1)
    dis = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        if bot[i] != 0:
            dis[i] = np.linalg.norm(np.outer(B @ B[i], B[i]) / bot[i], axis=1)
    gval = np.sqrt(stats.chi2.ppf(0.975, 2))
    out = np.zeros(len(X), bool)
    for i in range(len(X)):
        out |= dis[i] > np.median(dis[i]) + gval * _ideal_fourths(dis[i])
    return out


def skipped_corr(x, y, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = skipped_outliers(x, y)
    r = stats.pearsonr(x[~out], y[~out])[0]
    rng = np.random.default_rng(seed)
    redetected, fixed = [], []
    xk, yk = x[~out], y[~out]
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(x), len(x))
        try:
            o = skipped_outliers(x[idx], y[idx])
            redetected.append(stats.pearsonr(x[idx][~o], y[idx][~o])[0])
        except Exception:  # degenerate resample
            pass
        idx = rng.integers(0, len(xk), len(xk))
        fixed.append(stats.pearsonr(xk[idx], yk[idx])[0])
    return dict(r=r, outliers=out, ci_fixed=np.percentile(fixed, [2.5, 97.5]),
                ci_redetected=np.percentile(redetected, [2.5, 97.5]))


zen = pd.read_csv(zen_dir / 'mw_medit_combined_parieto_occip_Gamma_correl_exp.csv').rename(columns={
    'Sujet': 'lab_id', 'Type': 'group', 'Meditation Expe': 'hours', 'Median Gamma': 'gamma'})
zen = zen[zen.group != 'CTR'].assign(log_hours=lambda d: np.log10(d.hours))
mapping = pd.read_csv('results/paper_included_subjects.csv')
zen = zen.merge(mapping[['lab_id', 'participant_id']], on='lab_id', how='left')
zen['subject'] = zen.participant_id.str.removeprefix('sub-')


def replication(suffix):
    d = pd.read_csv(R / f'per_subject_vs_paper_{suffix}.csv', dtype={'subject': str})
    g = d[(d.band == 'gamma') & (d.cond == 'trait')].set_index('subject').db
    return zen.dropna(subset=['subject']).assign(gamma=lambda z: z.subject.map(g)).dropna(subset=['gamma'])


sources = {
    "paper's values": zen,
    "paper's values, 47 in ds003969": zen.dropna(subset=['subject']),
    'no ICA': replication('_matchsample_noica_v1'),
    'capped ICA': replication('_matchsample_v1'),
}
rows, fits = [], {}
for name, d in sources.items():
    sk = skipped_corr(d.log_hours, d.gamma)
    fits[name] = (d, sk)
    rows.append(dict(source=name, n=len(d), pearson_r=stats.pearsonr(d.log_hours, d.gamma)[0],
                     pearson_p=stats.pearsonr(d.log_hours, d.gamma)[1],
                     spearman_rho=stats.spearmanr(d.log_hours, d.gamma)[0], skipped_r=sk['r'],
                     n_outliers=int(sk['outliers'].sum()), outliers=' '.join(d.lab_id[sk['outliers']]),
                     ci_fixed_low=sk['ci_fixed'][0], ci_fixed_high=sk['ci_fixed'][1],
                     ci_redetected_low=sk['ci_redetected'][0], ci_redetected_high=sk['ci_redetected'][1]))
table = pd.DataFrame(rows)
table.to_csv(R / 'experience_gamma.csv', index=False)
pd.set_option('display.width', 250)
print(table.round(3).to_string(index=False))
print('paper: skipped r = 0.33, 95% CI [0.06, 0.53]')

plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
panels = [("paper's values", "(a) Paper's values"), ('no ICA', '(b) This replication, no ICA'),
          ('capped ICA', '(c) This replication, capped ICA')]
fig, axs = plt.subplots(1, 3, figsize=(7.6, 2.8), constrained_layout=True, sharex=True)
for ax, (name, title) in zip(axs, panels):
    d, sk = fits[name]
    out = sk['outliers']
    for g, gg in d.groupby('group'):
        o = out[(d.group == g).values]
        ax.scatter(gg.log_hours[~o], gg.gamma[~o], s=20, color=COLOURS[g], edgecolor='white', lw=0.6, label=g)
        ax.scatter(gg.log_hours[o], gg.gamma[o], s=20, facecolor='none', edgecolor=COLOURS[g], lw=1.1)
    b = np.polyfit(d.log_hours[~out], d.gamma[~out], 1)
    xs = np.linspace(d.log_hours.min(), d.log_hours.max(), 20)
    ax.plot(xs, np.polyval(b, xs), color='#52514e', lw=1)
    ax.set_title(f"{title}\nskipped $r={sk['r']:.2f}$", loc='left', fontsize=9)
    ax.set_xlabel('log$_{10}$ lifetime practice hours')
    ax.grid(color='#e6e5e0', lw=0.6)
    ax.set_axisbelow(True)
axs[0].set_ylabel('gamma, MED+IMW (dB)')
fig.legend(*axs[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
for ext in ['pdf', 'png']:
    fig.savefig(f'figures/experience_gamma.{ext}', dpi=110)
