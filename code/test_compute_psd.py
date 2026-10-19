import mne
import numpy as np

from compute_psd_epochs import compute_subject_task_psd


def _raw_with_bads(tmp_path, bads):
    """10 s of noise on standard 10-20 channels, each channel at a different amplitude so channels are
    distinguishable by power; saved to a .fif like the pipeline's processed files."""
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz', 'Oz']
    info = mne.create_info(ch_names, 256, 'eeg')
    rng = np.random.default_rng(0)
    scales = np.arange(1, len(ch_names) + 1)[:, None] * 1e-6
    raw = mne.io.RawArray(rng.normal(size=(len(ch_names), 2560)) * scales, info, verbose='error')
    raw.set_montage('standard_1020')
    raw.info['bads'] = bads
    path = tmp_path / 'sub-01_task-x_processed_raw.fif'
    raw.save(path, verbose='error')
    return path, ch_names, scales[:, 0]


def test_bad_channels_are_nan_and_good_channels_keep_their_own_spectrum(tmp_path):
    path, ch_names, scales = _raw_with_bads(tmp_path, ['Fp2', 'C3'])
    res = compute_subject_task_psd(path)
    nan_rows = [c for c, row in zip(res['ch_names'], res['psd_mean']) if np.isnan(row).all()]
    assert nan_rows == ['Fp2', 'C3']
    # white noise: mean power is proportional to the channel's variance, so the ratio to the first channel
    # identifies which channel's data ended up in each row
    power = np.nanmean(res['psd_mean'], axis=1)
    good = [i for i, c in enumerate(ch_names) if c not in ('Fp2', 'C3')]
    ratio = power[good] / power[0]
    expected = (scales[good] / scales[0]) ** 2
    np.testing.assert_allclose(ratio, expected, rtol=0.25)


def test_interpolation_fills_bads_without_changing_good_channels(tmp_path):
    path, ch_names, _ = _raw_with_bads(tmp_path, ['Fp2', 'C3'])
    plain, interp = compute_subject_task_psd(path), compute_subject_task_psd(path, interpolate_bads=True)
    assert not np.isnan(interp['psd_mean']).any()
    good = ~np.isnan(plain['psd_mean']).any(axis=1)
    np.testing.assert_allclose(interp['psd_mean'][good], plain['psd_mean'][good])
    assert list(interp['bad_channels']) == ['Fp2', 'C3']
