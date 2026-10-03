"""Tests for psd_lof.find_bad_channels_psd_lof.

Run from the repo root:  pytest code
"""
import numpy as np
import mne
import pytest

from psd_lof import find_bad_channels_psd_lof

SFREQ = 256
DURATION = 60  # s
N_CH = 30
NOISE = 1e-5

mne.set_log_level("ERROR")


def make_raw(seed=0):
    """White noise Raw; individual channels are corrupted by the tests."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((N_CH, SFREQ * DURATION)) * NOISE
    info = mne.create_info([f"C{i}" for i in range(N_CH)], SFREQ, "eeg")
    return mne.io.RawArray(data, info)


def test_clean_data_flags_nothing():
    assert find_bad_channels_psd_lof(make_raw()) == []


def test_noisy_channel_flagged():
    raw = make_raw()
    raw._data[7] *= 10
    assert find_bad_channels_psd_lof(raw) == ["C7"]


def test_flat_channel_flagged_without_corrupting_others():
    raw = make_raw()
    raw._data[3] = 0
    bads, scores = find_bad_channels_psd_lof(raw, return_scores=True)
    assert bads == ["C3"]
    assert scores["C3"] == np.inf
    assert all(np.isfinite(v) for k, v in scores.items() if k != "C3")


def test_spectral_shape_outlier_flagged_at_matched_amplitude():
    # same broadband level as the rest, but a strong 50 Hz line on top of it
    raw = make_raw()
    t = np.arange(raw.n_times) / SFREQ
    raw._data[11] += 5 * NOISE * np.sin(2 * np.pi * 50 * t)
    assert find_bad_channels_psd_lof(raw, threshold=1.2) == ["C11"]


def test_return_scores_names_all_picked_channels():
    raw = make_raw()
    bads, scores = find_bad_channels_psd_lof(raw, return_scores=True)
    assert list(scores) == raw.ch_names
    assert bads == [c for c, s in scores.items() if s >= 1.5]


def test_threshold_controls_sensitivity():
    raw = make_raw()
    raw._data[7] *= 10
    assert find_bad_channels_psd_lof(raw, threshold=1e6) == []


def test_existing_bads_are_excluded():
    raw = make_raw()
    raw._data[7] *= 10
    raw.info["bads"] = ["C7"]
    bads, scores = find_bad_channels_psd_lof(raw, return_scores=True)
    assert bads == []
    assert "C7" not in scores


def test_picks_restricts_channels_and_ignores_non_eeg():
    raw = make_raw()
    raw.set_channel_types({"C0": "eog"})
    raw._data[0] *= 100                       # would be an outlier, but is not EEG
    _, scores = find_bad_channels_psd_lof(raw, return_scores=True)
    assert "C0" not in scores


def test_bad_annotated_segments_are_skipped():
    # channel 5 is only corrupted inside a BAD segment, so it should look clean
    raw = make_raw()
    a, b = 10 * SFREQ, 20 * SFREQ
    raw._data[5, a:b] *= 100
    raw.set_annotations(mne.Annotations(10, 10, "BAD_rejcont"))
    assert find_bad_channels_psd_lof(raw) == []
    assert find_bad_channels_psd_lof(raw, reject_by_annotation=False) == ["C5"]


def test_n_neighbors_is_capped_for_few_channels():
    raw = make_raw().pick(["C0", "C1", "C2", "C3", "C4"])
    find_bad_channels_psd_lof(raw, n_neighbors=20)      # must not raise


def test_log_bands_clean_data_flags_nothing():
    assert find_bad_channels_psd_lof(make_raw(), log_bands=True) == []


def test_log_bands_still_flags_flat_and_noisy():
    raw = make_raw()
    raw._data[3] = 0
    raw._data[7] *= 10
    assert find_bad_channels_psd_lof(raw, log_bands=True) == ["C3", "C7"]


def test_log_bands_detects_low_frequency_only_outlier():
    # a 6 Hz-only excess: ~4 of ~100 linear bins, but a whole band when pooled
    raw = make_raw()
    t = np.arange(raw.n_times) / SFREQ
    raw._data[9] += 5 * NOISE * np.sin(2 * np.pi * 6 * t)
    bads, scores = find_bad_channels_psd_lof(raw, log_bands=True, return_scores=True)
    _, linear = find_bad_channels_psd_lof(raw, return_scores=True)
    assert scores["C9"] > linear["C9"]


def test_log_bands_pooling_shape_and_empty_bands_dropped():
    from psd_lof import _pool_log_bands
    freqs = np.arange(1.0, 101.0)
    x = np.zeros((4, freqs.size))
    assert _pool_log_bands(x, freqs, 1.0, 100.0, 10).shape == (4, 10)
    # 40 bands over 1-100 Hz at 1 Hz resolution: the lowest ones hold no bin
    assert _pool_log_bands(x, freqs, 1.0, 100.0, 40).shape[1] < 40


def test_log_bands_pooled_value_is_band_mean():
    from psd_lof import _pool_log_bands
    freqs = np.arange(1.0, 101.0)
    x = np.tile(freqs, (2, 1))                       # value == frequency
    out = _pool_log_bands(x, freqs, 1.0, 100.0, 2)   # edges 1, 10, 100
    assert out[0, 0] == pytest.approx(np.mean(np.arange(1, 10)))
    assert out[0, 1] == pytest.approx(np.mean(np.arange(10, 101)))
