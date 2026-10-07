"""Port of EEGLAB's pop_rejcont (Arnaud Delorme) for MNE Raw objects.

Follows pop_rejcont.m line by line, including its quirks:
  * the spectrum is averaged over channels (in dB) before thresholding, so
    one decision is made per epoch and rejected regions apply to all channels;
  * each channel/frequency dB spectrum is baselined by subtracting its mean
    across epochs, so thresholds are "dB above the recording's average";
  * a band is flagged if ANY frequency bin inside it exceeds the threshold
    (eegthresh behaviour), not if the band mean does, with the window edges
    mapped to bins as eegthresh does (see _eegthresh_window);
  * 'contiguous' is not a minimum run length: it only sets the merge distance
    grouplen = contiguous / 2 * epochlength seconds (see _merge_regions).
"""
import mne
import numpy as np


def _eegthresh_window(freqs, fmin, fmax):
    """Slice of frequency bins eegthresh() inspects for [fmin, fmax].

    Literal port of its time-to-frame mapping, with the frequency axis as the
    'time' axis and timerange = [freqs[0], freqs[-1]]. Note this is not a plain
    freqs >= fmin mask: e.g. 35-128 Hz at N=256 selects bins 34..127 and
    0-10 Hz selects bins 1..9 (1-based).
    """
    t1, t2, n = freqs[0], freqs[-1], len(freqs)
    fmin = max(fmin, t1)
    fmax = min(fmax, t2)
    lo = max(1, int(np.floor((fmin - t1) / (t2 - t1) * (n - 1))) + 1)
    hi = int(np.floor((fmax - t1) / (t2 - t1) * (n - 1))) + 1
    return slice(lo - 1, hi)


def _merge_regions(win, grouplen):
    """Literal port of the 'remove isolated regions and merge others' loop.

    win: list of [start, end] in 1-based inclusive samples (MATLAB convention).
    """
    win = [list(w) for w in win]
    for i in range(len(win) - 1, -1, -1):
        n = len(win)
        if win[i][1] - win[i][0] > grouplen or i == 0 and n > 1 and win[1][0] - win[0][1] > grouplen or i == n - 1 and n > 1 and win[i][0] - win[i - 1][1] > grouplen or (0 < i < n - 1 and win[i + 1][0] - win[i][1] > grouplen
              and win[i][0] - win[i - 1][1] > grouplen):
            del win[i]
        elif i < n - 1 and win[i + 1][0] - win[i][1] <= grouplen:
            win[i][1] = win[i + 1][1]
            del win[i + 1]
    return win


def rejcont(raw: mne.io.Raw, bands=((0, 10, 17), (35, 128, 14)), epochlength=1.0,
            recurrence=0.5, contiguous=8, addlength=0.25, picks="eeg",
            taper=False, annotate=True, description="BAD_rejcont"):
    """Detect artifact segments like pop_rejcont.

    bands       : (fmin, fmax, threshold_dB) triples; epochs are flagged if
                  any bin in [fmin, fmax] exceeds threshold_dB (union over bands).
    epochlength : epoch length in s.
    recurrence  : step between epoch starts in s. This is what EEGLAB's
                  'overlap' option really is (passed to eeg_regepochs as the
                  recurrence), so 1 s epochs with 'overlap' 0.5 = 50 % overlap.
    contiguous  : sets merge distance contiguous/2*epochlength (s).
    addlength   : seconds added to each side of every region (EEGLAB default 0.25).
    picks       : channels averaged over (EEGLAB 'elecrange').
    Returns regions as (onset_s, duration_s) relative to the first sample.
    """
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=picks)
    pnts = data.shape[1]
    N = int(round(epochlength * sfreq))
    step = int(round(recurrence * sfreq))
    starts = np.arange(0, pnts - N + 1, step)          # 0-based, full epochs only

    # channel-mean dB spectrum per epoch (DC and Nyquist dropped, as in
    # tmp(:,2:N/2,:)). Averaging over channels commutes with the baseline
    # subtraction over epochs, so the latter is done afterwards on 1 row.
    win = np.hamming(N) if taper else np.ones(N)
    spec = np.empty((len(starts), N // 2 - 1))
    with np.errstate(divide="ignore"):
        for k, s in enumerate(starts):
            f = np.fft.rfft(data[:, s:s + N] * win, axis=1)[:, 1:N // 2]
            spec[k] = (10 * np.log10(np.abs(f) ** 2)).mean(axis=0)
    spec -= spec.mean(axis=0, keepdims=True)

    # EEGLAB's (slightly stretched) frequency axis
    freqs = np.linspace(0, sfreq / 2, N // 2)[1:]

    rej = np.zeros(len(starts), bool)
    for fmin, fmax, thr in bands:
        sub = spec[:, _eegthresh_window(freqs, fmin, fmax)]
        rej |= (sub > thr).any(axis=1) | (sub < -100).any(axis=1)  # eegthresh: [-100 thr]

    # one window per flagged epoch, 1-based inclusive samples as in winrej
    win_ = [[s + 1, s + N] for s in starts[rej]]
    grouplen = contiguous / 2 * epochlength * sfreq + 1
    win_ = _merge_regions(win_, grouplen)

    regions = []
    for a, b in win_:
        a = max(1, np.floor(a - addlength * sfreq + 0.5))
        b = min(pnts, np.floor(b + addlength * sfreq + 0.5))
        regions.append(((a - 1) / sfreq, (b - a + 1) / sfreq))

    if annotate and regions:
        ann = raw.annotations.copy()
        off = raw.first_time if ann.orig_time is not None else 0.0
        ann.append([r[0] + off for r in regions], [r[1] for r in regions], description)
        raw.set_annotations(ann)
    return regions
