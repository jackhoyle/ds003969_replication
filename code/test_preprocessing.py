"""Tests for preprocessing.rejcont (port of EEGLAB's pop_rejcont).

Run from the repo root:  pytest code
"""
import numpy as np
import mne
import pytest

from preprocessing import _eegthresh_window, _merge_regions, rejcont

SFREQ = 256
DURATION = 120  # s
N_CH = 8
NOISE = 1e-5
BURST = 3e-4    # ~30x the noise amplitude, broadband

mne.set_log_level("ERROR")


def make_raw(bursts=(), channels=slice(None), seed=0, meas_date=False):
    """White noise Raw with broadband bursts given as (start_s, stop_s)."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_CH, SFREQ * DURATION)) * NOISE
    for start, stop in bursts:
        a, b = int(start * SFREQ), int(stop * SFREQ)
        data[channels, a:b] += rng.standard_normal((data[channels].shape[0], b - a)) * BURST
    raw = mne.io.RawArray(data, mne.create_info(N_CH, SFREQ, "eeg"))
    if meas_date:
        import datetime
        raw.set_meas_date(datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc))
    return raw


# --------------------------------------------------------------------------
# _merge_regions: literal port of the pop_rejcont merge / isolation loop
# --------------------------------------------------------------------------
GROUPLEN = 4 * SFREQ + 1  # contiguous=8, epochlength=1 s


def epoch(start_s):
    """1-based inclusive sample window of a 1 s epoch starting at start_s."""
    s = int(start_s * SFREQ) + 1
    return [s, s + SFREQ - 1]


def test_merge_adjacent_overlapping_epochs():
    out = _merge_regions([epoch(10), epoch(10.5), epoch(11)], GROUPLEN)
    assert out == [[epoch(10)[0], epoch(11)[1]]]


def test_merge_bridges_gaps_up_to_grouplen():
    # 3 s gap between end of the first and start of the second: merged
    out = _merge_regions([epoch(10), epoch(14), epoch(14.5)], GROUPLEN)
    assert len(out) == 1
    # 10 s gap: kept separate
    out = _merge_regions([epoch(10), epoch(10.5), epoch(30), epoch(30.5)], GROUPLEN)
    assert len(out) == 2


def test_merge_removes_isolated_epoch_between_clusters():
    win = [epoch(10), epoch(10.5), epoch(30), epoch(50), epoch(50.5)]
    out = _merge_regions(win, GROUPLEN)
    assert out == [[epoch(10)[0], epoch(10.5)[1]], [epoch(50)[0], epoch(50.5)[1]]]


def test_merge_removes_isolated_trailing_epoch():
    out = _merge_regions([epoch(10), epoch(10.5), epoch(60)], GROUPLEN)
    assert out == [[epoch(10)[0], epoch(10.5)[1]]]


def test_merge_contiguous_is_not_a_minimum_run_length():
    """EEGLAB quirk: a lone flagged epoch (or pair) survives; 'contiguous' only
    sets the merge distance."""
    assert _merge_regions([epoch(10)], GROUPLEN) == [epoch(10)]
    assert len(_merge_regions([epoch(10), epoch(10.5)], GROUPLEN)) == 1


def test_merge_empty():
    assert _merge_regions([], GROUPLEN) == []


# --------------------------------------------------------------------------
# rejcont
# --------------------------------------------------------------------------
def test_clean_noise_is_not_flagged():
    raw = make_raw()
    assert rejcont(raw) == []
    assert len(raw.annotations) == 0


def test_broadband_burst_is_flagged_with_padding():
    raw = make_raw(bursts=[(40, 46)])
    regions = rejcont(raw, addlength=0.25)
    assert len(regions) == 1
    onset, dur = regions[0]
    assert onset <= 40 and onset + dur >= 46          # burst fully covered
    assert onset > 38 and onset + dur < 48            # but not much more


def test_addlength_zero_shrinks_region_by_padding():
    padded = rejcont(make_raw(bursts=[(40, 46)]), addlength=0.25)[0]
    bare = rejcont(make_raw(bursts=[(40, 46)]), addlength=0.0)[0]
    assert bare[1] == pytest.approx(padded[1] - 0.5, abs=1 / SFREQ)
    assert bare[0] == pytest.approx(padded[0] + 0.25, abs=1 / SFREQ)


def test_two_bursts_give_two_regions():
    regions = rejcont(make_raw(bursts=[(20, 26), (80, 86)]))
    assert len(regions) == 2
    assert regions[0][0] < regions[1][0]


def test_amplitude_scale_invariant():
    """Spectra are baselined against the recording's own mean, so scaling the
    data must not change the result."""
    a = rejcont(make_raw(bursts=[(40, 46)]))
    raw = make_raw(bursts=[(40, 46)])
    raw.apply_function(lambda x: x * 1000)
    assert rejcont(raw) == pytest.approx(a)


def test_burst_on_one_channel_is_diluted_by_channel_average():
    """EEGLAB averages dB over channels, so one bad channel of 8 (~30 dB / 8)
    stays under the 14 dB threshold."""
    raw = make_raw(bursts=[(40, 46)], channels=[0])
    assert rejcont(raw) == []


def test_burst_on_all_channels_but_picks_restricts_averaging():
    raw = make_raw(bursts=[(40, 46)], channels=[0, 1, 2, 3])
    assert rejcont(raw, picks=[0, 1, 2, 3]) != []      # half the picks bad: 15 dB
    assert rejcont(raw, picks=[4, 5, 6, 7]) == []      # clean channels only


def test_narrowband_burst_uses_max_over_bins_not_band_mean():
    """A strong 60 Hz sinusoid raises a single bin far above 14 dB while the
    35-128 Hz band mean barely moves; EEGLAB flags on any bin."""
    raw = make_raw()
    t = np.arange(SFREQ * 6) / SFREQ
    a = 40 * SFREQ
    raw._data[:, a:a + len(t)] += 10 * NOISE * np.sin(2 * np.pi * 60 * t)
    regions = rejcont(raw, bands=((35, 128, 14),))
    assert len(regions) == 1
    assert regions[0][0] <= 40 and regions[0][0] + regions[0][1] >= 46


def eeglab_freqs(n=SFREQ):
    return np.linspace(0, SFREQ / 2, n // 2)[1:]


def test_eegthresh_window_matches_eeglab_indices():
    """1-based bins: 35-128 Hz -> 34..127, 0-10 Hz -> 1..9 (N=256)."""
    f = eeglab_freqs()
    assert len(f) == 127
    hi = _eegthresh_window(f, 35, 128)
    lo = _eegthresh_window(f, 0, 10)
    assert (hi.start + 1, hi.stop) == (34, 127)
    assert (lo.start + 1, lo.stop) == (1, 9)


def add_tone(raw, freq, start=40, dur=6, amp=10 * NOISE):
    t = np.arange(SFREQ * dur) / SFREQ
    a = start * SFREQ
    raw._data[:, a:a + len(t)] += amp * np.sin(2 * np.pi * freq * t)


def flagged_seconds(raw, band):
    regions = rejcont(raw, bands=(band,), annotate=False)
    return sum(d for _, d in regions)


def test_band_edges_follow_eegthresh_bin_mapping():
    """Integer-frequency tones sit exactly on FFT bin f (no leakage inside the
    burst); only the two edge epochs are broadband and always flagged. So the
    tone is 'in band' iff the whole ~7.5 s burst is flagged, 'out of band' iff
    only ~1.5 s (edge) is.
    EEGLAB window: 35-128 Hz -> bins 34..127, 0-10 Hz -> bins 1..9."""
    def dur(freq, band):
        raw = make_raw()
        add_tone(raw, freq)
        return flagged_seconds(raw, band)

    high, low = (35, 128, 14), (0, 10, 17)
    assert dur(34, high) > 6      # bin 34 is inside the 35-128 Hz window
    assert dur(33, high) < 3      # bin 33 is not
    assert dur(9, low) > 6        # bin 9 is inside the 0-10 Hz window
    assert dur(10, low) < 3       # bin 10 is not


def test_threshold_is_per_band():
    raw = make_raw(bursts=[(40, 46)])
    assert rejcont(raw, bands=((0, 10, 200),), annotate=False) == []
    assert rejcont(raw, bands=((0, 10, 17),), annotate=False) != []


def test_annotations_added_and_existing_kept():
    raw = make_raw(bursts=[(40, 46)])
    raw.set_annotations(mne.Annotations([5.0], [1.0], "BAD_previous"))
    rejcont(raw, description="BAD_rejcont")
    assert set(raw.annotations.description) == {"BAD_previous", "BAD_rejcont"}


def test_annotate_false_leaves_raw_untouched():
    raw = make_raw(bursts=[(40, 46)])
    regions = rejcont(raw, annotate=False)
    assert regions and len(raw.annotations) == 0


@pytest.mark.parametrize("meas_date", [False, True])
def test_annotations_line_up_with_burst(meas_date):
    """Works whether or not the Raw has a measurement date (orig_time)."""
    raw = make_raw(bursts=[(40, 46)], meas_date=meas_date)
    (onset, dur), = rejcont(raw)
    kept = raw.get_data(reject_by_annotation="omit").shape[1]
    removed = SFREQ * DURATION - kept
    assert removed == pytest.approx(dur * SFREQ, abs=2)
    ann = raw.annotations
    assert ann.onset[0] - (raw.first_time if meas_date else 0) == pytest.approx(onset)


def test_nonzero_first_samp():
    raw = make_raw(bursts=[(40, 46)])
    raw.set_meas_date(0)
    raw._first_samps = np.array([SFREQ * 100])
    raw._last_samps = raw._first_samps + SFREQ * DURATION - 1
    (onset, dur), = rejcont(raw)
    kept = raw.get_data(reject_by_annotation="omit").shape[1]
    assert SFREQ * DURATION - kept == pytest.approx(dur * SFREQ, abs=2)


def test_regions_clamped_to_recording_edges():
    raw = make_raw(bursts=[(0, 5)])
    (onset, dur), = rejcont(raw)
    assert onset == 0
    assert onset + dur <= DURATION
