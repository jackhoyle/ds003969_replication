"""Bad channel detection with Local Outlier Factor on log power spectra.

mne.preprocessing.find_bad_channels_lof runs LOF on the raw waveforms, so it
mostly measures amplitude / waveform similarity. Here each channel is instead a
point described by its log-PSD (dB), so a channel is an outlier if its spectral
*shape or level* differs from its neighbours' (line noise, excess EMG, flat
channels, drifting channels, ...).
"""
import mne
import numpy as np


def _pool_log_bands(logpsd, freqs, fmin, fmax, n_bands):
    """Mean dB per log-spaced band, shape (n_channels, n_nonempty_bands)."""
    edges = np.geomspace(fmin, fmax, n_bands + 1)
    pooled = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        in_band = (freqs >= lo) & ((freqs < hi) | (i == n_bands - 1 and freqs <= hi))
        if in_band.any():
            pooled.append(logpsd[:, in_band].mean(axis=1))
    return np.stack(pooled, axis=1)


def find_bad_channels_psd_lof(raw: mne.io.BaseRaw, n_neighbors=20, threshold=1.5,
                              fmin=1.0, fmax=100.0, picks="eeg", return_scores=False,
                              log_bands=False, n_bands=10, **psd_kwargs):
    """Return names of channels whose log-PSD is a LOF outlier.

    n_neighbors : neighbourhood size; must be < number of picked channels.
    threshold   : |LOF score| at or above which a channel is flagged (1 = inlier).
    fmin, fmax  : frequency range (Hz) of the spectrum used.
    picks       : channels evaluated (bads already marked are excluded).
    log_bands   : if True, average the dB spectrum within n_bands log-spaced
                  bands between fmin and fmax before LOF. Without it every
                  linearly spaced bin is a feature, so high frequencies (most
                  of the bins) dominate the distance; pooling gives each
                  octave comparable weight. Bands containing no bin are dropped.
    n_bands     : number of log-spaced bands when log_bands is True.
    psd_kwargs  : passed to raw.compute_psd (e.g. method, n_fft). Segments
                  annotated BAD_* are skipped by default (reject_by_annotation).
    """
    from sklearn.neighbors import LocalOutlierFactor

    picks = mne.pick_types(raw.info, eeg=True, exclude="bads") if picks == "eeg" else picks
    psd = raw.compute_psd(picks=picks, fmin=fmin, fmax=fmax, **psd_kwargs)
    with np.errstate(divide="ignore"):
        logpsd = 10 * np.log10(psd.get_data())               # (n_channels, n_freqs)
    if log_bands:
        logpsd = _pool_log_bands(logpsd, psd.freqs, fmin, fmax, n_bands)
    flat = ~np.isfinite(logpsd).all(axis=1)                  # zero power -> -inf dB
    ch_names = np.array(psd.ch_names)

    scores = np.full(len(ch_names), np.inf)
    ok = ~flat
    k = min(n_neighbors, ok.sum() - 1)
    clf = LocalOutlierFactor(n_neighbors=k, metric="euclidean")
    clf.fit_predict(logpsd[ok])
    scores[ok] = -clf.negative_outlier_factor_

    bads = [name for name in psd.ch_names if scores[psd.ch_names.index(name)] >= threshold]
    return (bads, dict(zip(ch_names, scores))) if return_scores else bads
