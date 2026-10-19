"""Cluster-based permutation tests over electrodes, following Braboszcz et al. (2017).

The paper tested log spectral power at each electrode with FieldTrip: 2000 permutations, two-tailed t-tests
(unpaired between groups, paired between conditions), channel neighbours by triangulation, the default maxsum
cluster statistic and a cluster-forming threshold of p = 0.05. This module does the same with MNE:
  - per-subject values are the median dB power over a band's frequency bins at each electrode (band_power);
  - neighbours come from a Delaunay triangulation of the electrode positions (adjacency), which is what
    mne.channels.find_ch_adjacency falls back to for these channel names;
  - the cluster-forming threshold is the two-tailed t value for p_threshold with the test's degrees of freedom.

Every electrode needs a value, so the PSDs must be computed with interpolated bad channels
(compute_psd_epochs.main(..., interpolate_bads=True), or run_pipeline.py --interpolate-bads).

Note that a cluster's p-value applies to the cluster as a whole: it does not show that each of its electrodes
differs (Sassenhagen & Draschkow, 2019).
"""
from dataclasses import dataclass, field
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test, permutation_cluster_test, ttest_ind_no_p
from scipy import stats

from compute_psd_epochs import load_all

MONTAGE = 'colin27_1020'  # as set in process_inspect_loop.py


def channel_info(ch_names, montage=MONTAGE) -> mne.Info:
    """EEG info with electrode positions, for adjacency and topomaps."""
    info = mne.create_info([str(c) for c in ch_names], sfreq=256, ch_types='eeg')
    info.set_montage(montage)
    return info


def adjacency(info: mne.Info):
    """Sparse channel adjacency (Delaunay triangulation of the electrode positions), in info's channel order."""
    with mne.utils.use_log_level('error'):
        adj, names = mne.channels.find_ch_adjacency(info, ch_type='eeg')
    assert list(names) == list(info.ch_names)
    return adj


def band_power(psd_dir: Path, band, task, subjects=None) -> pd.DataFrame:
    """Median dB power over the bins lo <= f <= hi at each electrode: DataFrame (subjects x channels)."""
    by_task, freqs, ch_names, subjects_of = load_all(psd_dir, key='psd_mean', only_subjects=subjects)
    lo, hi = band
    in_band = (freqs >= lo) & (freqs <= hi)
    db = np.median(10 * np.log10(by_task[task][:, :, in_band]), axis=-1)
    if np.isnan(db).any():
        raise ValueError(f'{psd_dir} has NaN (bad) channels; compute PSDs with interpolate_bads=True')
    return pd.DataFrame(db, index=pd.Index(subjects_of[task], name='subject'), columns=[str(c) for c in ch_names])


@dataclass
class ClusterResult:
    """Output of a cluster test. t_obs: t value per channel; clusters: boolean channel masks; p_values: one per
    cluster; diff: mean difference per channel (a - b, or mean paired difference); n: subjects per sample."""
    ch_names: list
    t_obs: np.ndarray
    clusters: list
    p_values: np.ndarray
    diff: np.ndarray
    n: tuple
    threshold: float
    extra: dict = field(default_factory=dict)

    def mask(self, alpha=0.05):
        """Channels belonging to any cluster with p < alpha."""
        out = np.zeros(len(self.ch_names), bool)
        for c, p in zip(self.clusters, self.p_values):
            if p < alpha:
                out |= c
        return out

    def summary(self) -> pd.DataFrame:
        """One row per cluster, most significant first; a single row with p = NaN if no cluster formed."""
        rows = [dict(sign='+' if self.t_obs[c].sum() > 0 else '-', p=p, n_channels=int(c.sum()),
                     t_sum=float(self.t_obs[c].sum()), channels=' '.join(np.array(self.ch_names)[c]))
                for c, p in zip(self.clusters, self.p_values)]
        if not rows:
            rows = [dict(sign='', p=np.nan, n_channels=0, t_sum=0.0, channels='')]
        return pd.DataFrame(rows).sort_values('p', na_position='last').reset_index(drop=True)


def _two_sided_t(p_threshold, dof):
    return stats.t.ppf(1 - p_threshold / 2, dof)


def independent_test(a, b, adj, n_permutations=2000, p_threshold=0.05, seed=0, n_jobs=1) -> ClusterResult:
    """Unpaired (Student's) t-test of a - b at each channel, clustered over channels, two-tailed.
    a, b: DataFrames (subjects x channels) with the same columns."""
    assert list(a.columns) == list(b.columns)
    threshold = _two_sided_t(p_threshold, len(a) + len(b) - 2)
    t_obs, clusters, p_values, _ = permutation_cluster_test(
        [a.values, b.values], threshold=threshold, n_permutations=n_permutations, tail=0,
        stat_fun=lambda x, y: ttest_ind_no_p(x, y, equal_var=True), adjacency=adj, out_type='mask',
        rng=seed, n_jobs=n_jobs, verbose='error')
    return ClusterResult(list(a.columns), t_obs, list(clusters), np.asarray(p_values),
                         a.values.mean(0) - b.values.mean(0), (len(a), len(b)), threshold)


def paired_test(x, y, adj, n_permutations=2000, p_threshold=0.05, seed=0, n_jobs=1) -> ClusterResult:
    """Paired t-test of x - y at each channel (subjects present in both), clustered over channels, two-tailed."""
    common = x.index.intersection(y.index)
    d = x.loc[common] - y.loc[common]
    threshold = _two_sided_t(p_threshold, len(d) - 1)
    t_obs, clusters, p_values, _ = permutation_cluster_1samp_test(
        d.values, threshold=threshold, n_permutations=n_permutations, tail=0, adjacency=adj, out_type='mask',
        rng=seed, n_jobs=n_jobs, verbose='error')
    return ClusterResult(list(d.columns), t_obs, list(clusters), np.asarray(p_values), d.values.mean(0),
                         (len(d),), threshold)


def plot_cluster_topomap(result: ClusterResult, info: mne.Info, ax, alpha=0.05, vlim=None, cmap='RdBu_r'):
    """Topomap of the mean difference (dB) with channels in significant clusters marked by black dots,
    as in the paper's Figs 4A and 6A."""
    lim = vlim if vlim is not None else np.abs(result.diff).max()
    im, _ = mne.viz.plot_topomap(
        result.diff, info, axes=ax, show=False, cmap=cmap, vlim=(-lim, lim), contours=0, sensors=False,
        mask=result.mask(alpha), mask_params=dict(marker='o', markerfacecolor='k', markeredgecolor='k',
                                                   markersize=3))
    return im
