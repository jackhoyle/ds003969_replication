"""Summarise the epoch length sweep from 261012_epoch_len_sensitivity.py into one JSON file for plotting.

Usage (from the code/ directory):
    python ../exploration/261012_epoch_len_analysis.py --suffix _v4 --out summary.json
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from compute_psd_epochs import bids_root, freq_bands  # noqa: E402

REFERENCE_LEN = 180  # current compute_criticality.EPOCH_LEN
METRICS = ['dfa', 'fei']


def summarise(suffix):
    d = bids_root / f'outputs{suffix}' / 'epoch_len_sensitivity'
    ep = pd.read_parquet(d / 'epochs.parquet')
    rec = pd.read_parquet(d / 'recordings.parquet')
    lens = sorted(rec.epoch_len.unique())
    bands = list(freq_bands)
    n_rec = rec[['subject', 'task']].drop_duplicates().shape[0]

    groups = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
    group_of = dict(zip(groups.participant_id.str.removeprefix('sub-'), groups.group))

    # 1. data retention
    rec['analysed_s'] = rec.n_epochs * rec.epoch_len
    retention = rec.groupby('epoch_len').agg(
        pct_kept=('n_epochs', lambda n: 100 * (n > 0).mean()),
        mean_analysed_s=('analysed_s', 'mean'),
        mean_duration_s=('duration', 'mean'),
        pct_single_epoch=('n_epochs', lambda n: 100 * (n == 1).mean()),
    ).reset_index()

    # channel-averaged value per recording/epoch (bad channels are absent from the epoch rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        per_epoch = ep.groupby(['subject', 'task', 'epoch_len', 'band', 'epoch'])[METRICS].mean().reset_index()
        per_rec = per_epoch.groupby(['subject', 'task', 'epoch_len', 'band'])[METRICS].mean().reset_index()

    # clean recordings: no BAD spans, so every epoch length sees the same data (up to the discarded tail)
    clean = rec[rec.bad_s == 0][['subject', 'task']].drop_duplicates()
    per_rec_clean = per_rec.merge(clean, on=['subject', 'task'])

    def mean_se(df, keys):
        g = df.groupby(keys)[METRICS]
        m, s, n = g.mean(), g.std(), g.count()
        out = m.add_suffix('_mean').join((s / np.sqrt(n)).add_suffix('_se')).join(n[['dfa']].rename(columns={'dfa': 'n'}))
        return out.reset_index()

    # 2. bias: mean over recordings vs epoch length
    bias_all = mean_se(per_rec, ['band', 'epoch_len'])
    bias_clean = mean_se(per_rec_clean, ['band', 'epoch_len'])

    # 3. noise: SD between epochs within a recording (recordings with >= 2 epochs), and the implied SE of a
    # recording's estimate given how many epochs it actually gets
    sd = per_epoch.groupby(['subject', 'task', 'epoch_len', 'band'])[METRICS].agg(['std', 'count'])
    noise_rows = []
    for (L, band), g in sd.groupby(level=['epoch_len', 'band']):
        row = dict(epoch_len=int(L), band=band)
        for m in METRICS:
            multi = g[(m, 'count')] >= 2
            row[f'{m}_epoch_sd'] = float(g.loc[multi, (m, 'std')].median()) if multi.any() else None
        noise_rows.append(row)
    noise = pd.DataFrame(noise_rows)

    # 4. reliability: Spearman correlation across subjects between two sessions of the same condition
    def session_corr(df, a, b):
        rows = []
        for (L, band), g in df.groupby(['epoch_len', 'band']):
            w = g.pivot(index='subject', columns='task', values=METRICS)
            row = dict(epoch_len=int(L), band=band)
            for m in METRICS:
                pair = w[m][[a, b]].dropna() if {a, b} <= set(w[m].columns) else pd.DataFrame()
                row[f'{m}_rho'] = float(pair[a].corr(pair[b], method='spearman')) if len(pair) >= 10 else None
                row[f'{m}_n'] = int(len(pair))
            rows.append(row)
        return pd.DataFrame(rows)
    retest_think = session_corr(per_rec, 'think1', 'think2')
    retest_med = session_corr(per_rec, 'med1breath', 'med2')

    # 5. agreement with the current setting: Spearman across recordings vs REFERENCE_LEN
    ref = per_rec[per_rec.epoch_len == REFERENCE_LEN].drop(columns='epoch_len')
    agree_rows = []
    for (L, band), g in per_rec.groupby(['epoch_len', 'band']):
        j = g.merge(ref[ref.band == band], on=['subject', 'task', 'band'], suffixes=('', '_ref'))
        row = dict(epoch_len=int(L), band=band, n=int(len(j)))
        for m in METRICS:
            jj = j[[m, f'{m}_ref']].dropna()
            row[f'{m}_rho'] = float(jj[m].corr(jj[f'{m}_ref'], method='spearman')) if len(jj) >= 10 else None
        agree_rows.append(row)
    agree = pd.DataFrame(agree_rows)

    # 6. conclusions: each group minus control on the subject mean over tasks, z = diff / SE
    subj = per_rec.groupby(['subject', 'epoch_len', 'band'])[METRICS].mean().reset_index()
    subj['group'] = subj.subject.map(group_of)
    z_rows = []
    for (L, band), g in subj.groupby(['epoch_len', 'band']):
        ctr = g[g.group == 'ctr']
        for group, gg in g[g.group != 'ctr'].groupby('group'):
            row = dict(epoch_len=int(L), band=band, group=group, n=int(len(gg)), n_ctr=int(len(ctr)))
            for m in METRICS:
                a, c = gg[m].dropna(), ctr[m].dropna()
                diff = a.mean() - c.mean()
                se = np.sqrt(a.var() / len(a) + c.var() / len(c)) if len(a) > 1 and len(c) > 1 else np.nan
                row[f'{m}_diff'] = float(diff)
                row[f'{m}_z'] = float(diff / se) if np.isfinite(se) and se > 0 else None
            z_rows.append(row)
    zs = pd.DataFrame(z_rows)

    def records(df):
        return json.loads(df.to_json(orient='records'))

    return dict(
        suffix=suffix, epoch_lens=[int(x) for x in lens], bands=bands,
        band_edges={b: list(e) for b, e in freq_bands.items()},
        reference_len=REFERENCE_LEN, n_recordings=int(n_rec), n_clean=int(len(clean)),
        n_subjects=int(rec.subject.nunique()),
        retention=records(retention), bias_all=records(bias_all), bias_clean=records(bias_clean),
        noise=records(noise), retest_think=records(retest_think), retest_med=records(retest_med),
        agree=records(agree), group_z=records(zs),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='_v4')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    Path(args.out).write_text(json.dumps(summarise(args.suffix)))
