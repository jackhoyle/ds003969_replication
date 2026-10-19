from process_inspect_loop import SPECTRAL_LOF_THRESHOLD as S
from process_inspect_loop import WAVEFORM_LOF_THRESHOLD as W
from process_inspect_loop import cap_bad_channels


def test_under_cap_keeps_all_suggestions():
    kept, dropped = cap_bad_channels({'A': 2 * W, 'B': 0.5 * W}, {'C': 1.1 * S, 'D': 0.9 * S}, max_bad_channels=20)
    assert sorted(kept) == ['A', 'C'] and dropped == []


def test_spectral_only_dropped_first_weakest_first():
    waveform = {'w_strong': 3 * W, 'w_weak': 1.01 * W, 'both': 1.2 * W}
    spectral = {'s_strong': 5 * S, 's_weak': 1.1 * S, 'both': 1.5 * S}
    kept, dropped = cap_bad_channels(waveform, spectral, max_bad_channels=3)
    # both > waveform-only (strongest first) > spectral-only (strongest first)
    assert kept == ['both', 'w_strong', 'w_weak']
    assert dropped == ['s_strong', 's_weak']
    kept, dropped = cap_bad_channels(waveform, spectral, max_bad_channels=4)
    assert dropped == ['s_weak']


def test_existing_bads_count_towards_cap():
    kept, dropped = cap_bad_channels({'A': 2 * W, 'B': 3 * W}, {}, max_bad_channels=3, n_existing=2)
    assert kept == ['B'] and dropped == ['A']


def test_zero_or_none_means_no_cap():
    scores = {f'c{i}': 2 * W for i in range(30)}
    for cap in (0, None):
        kept, dropped = cap_bad_channels(scores, {}, max_bad_channels=cap)
        assert len(kept) == 30 and dropped == []


def test_flat_channel_infinite_spectral_score_is_strongest_spectral():
    kept, dropped = cap_bad_channels({}, {'flat': float('inf'), 'x': 1.2 * S}, max_bad_channels=1)
    assert kept == ['flat'] and dropped == ['x']
