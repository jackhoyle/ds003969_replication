"""Cluster-based permutation tests over electrodes on the paper's matched sample, as in Braboszcz et al. (2017).

Tests (two-tailed, 2000 permutations, cluster-forming threshold p_threshold), for each matched run, on median dB
power per electrode in gamma (60-110 Hz) and alpha (7-11 Hz):
  - each meditator group, and all meditators pooled, against controls, in MED (med2), IMW (think2) and trait
    (each subject's mean of MED and IMW);
  - for alpha, VIP against HYT and against ISY;
  - within each group, MED against IMW (paired).

Needs PSDs with interpolated bad channels for both matched runs, i.e. runs made with --interpolate-bads (from code/):
    python run_pipeline.py --suffix _matchsample_v1 --paper-subjects --interpolate-bads
    python run_pipeline.py --suffix _matchsample_noica_v1 --paper-subjects --interpolate-bads --no-ica

Correction across tests: each test's p-value is its smallest cluster p (already corrected across electrodes; 1 if
no cluster formed). Benjamini-Hochberg FDR at 5% is applied to these within each pipeline, over two families:
  - all tests run here;
  - the paper's own tests: gamma, each tradition vs CTR in MED, IMW and trait; alpha, VIP vs each other group in
    MED and IMW; MED vs IMW within each group, both bands.

Usage (from code/):
    python ../writeup/scripts/cluster_tests.py [--p-threshold 0.05] [--n-permutations 2000] [--n-jobs 8]
    python ../writeup/scripts/cluster_tests.py --fdr-only   # FDR from an existing cluster_tests{tag}.csv
Writes results/matched/cluster_tests{tag}.csv (one row per cluster), results/matched/cluster_tests_fdr{tag}.csv (one
row per test) and figures/cluster_{gamma,alpha}{tag}.pdf (+ .png), where tag is empty for p_threshold 0.05 and e.g.
'_p01' otherwise.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from cluster_stats import adjacency, band_power, channel_info, independent_test, paired_test, plot_cluster_topomap  # noqa: E402
from compute_psd_epochs import bids_root, paper_subjects, psd_dir_for  # noqa: E402

WRITEUP = Path(__file__).resolve().parents[1]
RUNS = {'matched, no ICA': '_matchsample_noica_v1', 'matched, capped ICA': '_matchsample_v1'}
BANDS = {'gamma': (60, 110), 'alpha': (7, 11)}
CONDS = {'MED': 'med2', 'IMW': 'think2'}
GROUP = {'ctr': 'CTR', 'htr': 'HYT', 'sny': 'ISY', 'vip': 'VIP'}
MEDITATORS = ['HYT', 'ISY', 'VIP']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--p-threshold', type=float, default=0.05, help='cluster-forming threshold (default: 0.05)')
    parser.add_argument('--n-permutations', type=int, default=2000)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--fdr-only', action='store_true',
                        help='skip the permutation tests and figures; apply FDR to an existing cluster_tests CSV')
    return parser.parse_args()


def load(run_suffix, band, subjects, group_of):
    """{cond: DataFrame (subjects x channels)} for MED, IMW and trait, and the subjects' groups."""
    psd_dir = psd_dir_for(run_suffix, interpolated=True)
    if not psd_dir.exists():
        sys.exit(f'{psd_dir} not found: compute interpolated PSDs first (see this script\'s docstring)')
    data = {cond: band_power(psd_dir, band, task, subjects) for cond, task in CONDS.items()}
    both = data['MED'].index.intersection(data['IMW'].index)
    data['trait'] = (data['MED'].loc[both] + data['IMW'].loc[both]) / 2
    return {cond: d.assign(group=d.index.map(group_of)) for cond, d in data.items()}


def comparisons(band):
    """(name, kind, a, b): kind 'between' compares group a with group b in each condition; 'within' compares
    MED with IMW in group a."""
    out = [(f'{g} - CTR', 'between', [g], ['CTR']) for g in MEDITATORS]
    out.append(('all meditators - CTR', 'between', MEDITATORS, ['CTR']))
    if band == 'alpha':
        out += [('VIP - HYT', 'between', ['VIP'], ['HYT']), ('VIP - ISY', 'between', ['VIP'], ['ISY'])]
    out += [(f'MED - IMW, {g}', 'within', [g], None) for g in ['CTR'] + MEDITATORS]
    return out


def in_paper_family(band, cond, comparison):
    """Whether a test is one of the paper's own electrode-level tests."""
    if cond == 'MED - IMW':
        return True
    if band == 'gamma':
        return cond in ('MED', 'IMW', 'trait') and comparison in [f'{g} - CTR' for g in MEDITATORS]
    return cond in ('MED', 'IMW') and comparison in ('VIP - CTR', 'VIP - HYT', 'VIP - ISY')


def fdr(table, alpha=0.05):
    """One row per test: smallest cluster p (1 if no cluster formed), and BH q-values within each pipeline, over all
    tests and over the paper's tests only."""
    tests = (table.groupby(['run', 'band', 'cond', 'comparison'], sort=False).p.min().fillna(1.0)
             .rename('p').reset_index())
    tests['paper_test'] = [in_paper_family(*r) for r in tests[['band', 'cond', 'comparison']].itertuples(index=False)]
    for family, sel in [('all', lambda d: np.ones(len(d), bool)), ('paper', lambda d: d.paper_test.values)]:
        tests[f'q_{family}'] = np.nan
        for run, d in tests.groupby('run'):
            idx = d.index[sel(d)]
            tests.loc[idx, f'q_{family}'] = multipletests(tests.loc[idx, 'p'], alpha=alpha, method='fdr_bh')[1]
    return tests


def report_fdr(table, tag):
    tests = fdr(table)
    out_csv = WRITEUP / 'results' / 'matched' / f'cluster_tests_fdr{tag}.csv'
    tests.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}')
    for run, d in tests.groupby('run', sort=False):
        print(f'{run}: {len(d)} tests ({d.paper_test.sum()} of the paper\'s); p < .05: {(d.p < .05).sum()}; '
              f'q < .05 over all tests: {(d.q_all < .05).sum()}; '
              f'over the paper\'s tests: {(d.q_paper < .05).sum()}')
    print('\nTests with p < .05 (q = BH-adjusted p; q_paper is blank for tests outside the paper\'s family):')
    print(tests[tests.p < .05].round(3).to_string(index=False))


def main():
    args = parse_args()
    tag = '' if args.p_threshold == 0.05 else f"_p{str(args.p_threshold).split('.')[1]}"
    pd.set_option('display.width', 250)
    pd.set_option('display.max_colwidth', 60)
    if args.fdr_only:
        report_fdr(pd.read_csv(WRITEUP / 'results' / 'matched' / f'cluster_tests{tag}.csv'), tag)
        return
    participants = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
    group_of = dict(zip(participants.participant_id.str.removeprefix('sub-'), participants.group.map(GROUP)))
    subjects = paper_subjects()
    kw = dict(n_permutations=args.n_permutations, p_threshold=args.p_threshold, n_jobs=args.n_jobs)

    rows, results = [], {}
    for run, suffix in RUNS.items():
        for band, edges in BANDS.items():
            data = load(suffix, edges, subjects, group_of)
            ch_names = [c for c in data['MED'].columns if c != 'group']
            info = channel_info(ch_names)
            adj = adjacency(info)
            for name, kind, a, b in comparisons(band):
                if kind == 'between':
                    for cond, d in data.items():
                        x, y = d[d.group.isin(a)][ch_names], d[d.group.isin(b)][ch_names]
                        results[run, band, cond, name] = res = independent_test(x, y, adj, **kw)
                        rows += [dict(run=run, band=band, cond=cond, comparison=name, n_a=len(x), n_b=len(y), **r)
                                 for r in res.summary().to_dict('records')]
                else:
                    x, y = (data[c][data[c].group.isin(a)][ch_names] for c in ['MED', 'IMW'])
                    results[run, band, 'MED - IMW', name] = res = paired_test(x, y, adj, **kw)
                    rows += [dict(run=run, band=band, cond='MED - IMW', comparison=name, n_a=res.n[0], n_b=res.n[0],
                                  **r) for r in res.summary().to_dict('records')]
                print(f'{run} | {band} | {name}: done')

    table = pd.DataFrame(rows)
    out_csv = WRITEUP / 'results' / 'matched' / f'cluster_tests{tag}.csv'
    table.to_csv(out_csv, index=False)
    print(f'\nwrote {out_csv}\nClusters with p < .05:')
    print(table[table.p < 0.05][['run', 'band', 'cond', 'comparison', 'sign', 'p', 'n_channels', 'channels']]
          .round(3).to_string(index=False))
    report_fdr(table, tag)

    # figures in the style of the paper's Figs 4A (gamma) and 6A (alpha): mean difference, black dots = channels in
    # a cluster with p < .05; one colour scale per figure
    plt.rcParams.update({'font.size': 8})
    panels = {
        'gamma': [(cond, name) for cond in ['MED', 'IMW', 'trait']
                  for name in [f'{g} - CTR' for g in MEDITATORS] + ['all meditators - CTR']],
        'alpha': [(cond, name) for cond in ['MED', 'IMW']
                  for name in ['VIP - CTR', 'VIP - HYT', 'VIP - ISY', 'all meditators - CTR']],
    }
    for band, cells in panels.items():
        n_cols = 4
        n_rows = len(cells) // n_cols * len(RUNS)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(2.3 * n_cols + 0.6, 2.25 * n_rows), squeeze=False)
        lim = max(np.abs(results[run, band, cond, name].diff).max() for run in RUNS for cond, name in cells)
        info = None
        for r, run in enumerate(RUNS):
            for k, (cond, name) in enumerate(cells):
                res = results[run, band, cond, name]
                info = info or channel_info(res.ch_names)
                ax = axs[r * len(cells) // n_cols + k // n_cols, k % n_cols]
                im = plot_cluster_topomap(res, info, ax, vlim=lim)
                p = min(res.p_values, default=np.nan)
                ax.set_title(f'{name}, {cond}\n{run.removeprefix("matched, ")}'
                             + (f'; min p = {p:.3f}' if np.isfinite(p) else '; no cluster'), fontsize=7)
        cbar = fig.colorbar(im, ax=axs, shrink=0.5, pad=0.02)
        cbar.set_label('difference (dB)')
        for ext in ['pdf', 'png']:
            fig.savefig(WRITEUP / 'figures' / f'cluster_{band}{tag}.{ext}', dpi=110, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
