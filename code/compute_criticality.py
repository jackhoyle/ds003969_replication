"""Per-subject/task DFA exponent and fEI (crosci) per frequency band, excluding bad channels and bad time spans.

Ported from exploration/261007_crosci.ipynb. For each processed fif file:
  1. split into 180 s fixed-length epochs; epochs overlapping a BAD_* annotation are dropped
  2. per frequency band: band-pass filter each epoch and take the Hilbert amplitude envelope
  3. DFA exponent and fEI per channel and epoch, then average across epochs (ignoring NaNs:
     fEI is NaN where the DFA exponent is too low for it to be meaningful)
  4. save to an .npz with every EEG channel present (bad channels are NaN), as compute_psd_epochs.py does

load_all() collects the saved results into one long-format DataFrame (one row per subject/task/band/channel),
and main() also writes that table to criticality.csv next to the .npz folder.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import warnings

import mne
import numpy as np
import pandas as pd
from crosci.biomarkers import DFA, fEI

from compute_psd_epochs import bids_root, processed_dir_for, tasks

run_suffix = '_v4'

freq_bands = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'broadband': (1.0, 30.0),
}
EPOCH_LEN = 180.0              # s
DFA_WINDOW = (5.0, 30.0)       # s, window sizes the DFA is computed and fitted over
FEI_WINDOW = 5.0               # s
FEI_OVERLAP = 0.8              # fraction of overlap between fEI windows


def criticality_dir_for(run_suffix):
    return bids_root / f'outputs{run_suffix}' / 'criticality'


def criticality_csv_for(run_suffix):
    return bids_root / f'outputs{run_suffix}' / 'criticality.csv'


def compute_dfa_fei(epochs: mne.Epochs, l_freq: float, h_freq: float):
    """DFA exponent and fEI per channel for one frequency band, averaged over epochs.
    Returns (dfa, fei), each (n_channels,) in the channel order of epochs."""
    sfreq = epochs.info['sfreq']
    envelope = (
        epochs.copy()
        .filter(l_freq=l_freq, h_freq=h_freq, verbose='error')
        .apply_hilbert(envelope=True)
        .get_data()
    )
    dfa, fei = [], []
    for epoch in envelope:  # (n_channels, n_times)
        dfa_array, *_ = DFA(epoch, sfreq, DFA_WINDOW, DFA_WINDOW, overlap=True)
        fei_outliers_removed, *_ = fEI(epoch, sfreq, FEI_WINDOW, FEI_OVERLAP, dfa_array)
        dfa.append(dfa_array)
        fei.append(np.ravel(fei_outliers_removed))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # fEI NaN in every epoch for a channel
        return np.nanmean(dfa, axis=0), np.nanmean(fei, axis=0)


def compute_subject_task_criticality(fif_path: Path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose='error')
    all_eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    bads = list(raw.info['bads'])
    raw.pick('eeg', exclude='bads')

    # reject_by_annotation drops any epoch overlapping a BAD_ annotation
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_LEN, preload=True, reject_by_annotation=True, verbose='error'
    )
    if len(epochs) == 0:
        return None

    # (n_bands, n_channels) over the full EEG channel list, NaN where the channel was bad
    ch_ix = [all_eeg.index(ch) for ch in epochs.ch_names]
    dfa = np.full((len(freq_bands), len(all_eeg)), np.nan)
    fei = np.full((len(freq_bands), len(all_eeg)), np.nan)
    for i, (l_freq, h_freq) in enumerate(freq_bands.values()):
        dfa[i, ch_ix], fei[i, ch_ix] = compute_dfa_fei(epochs, l_freq, h_freq)

    return dict(
        bands=np.array(list(freq_bands)),
        band_edges=np.array(list(freq_bands.values())),
        ch_names=np.array(all_eeg),
        bad_channels=np.array(bads),
        n_epochs=len(epochs),
        dfa=dfa,
        fei=fei,
    )


def _process(fif: Path, out_path: Path):
    result = compute_subject_task_criticality(fif)
    if result is None:
        return f"{fif.name}: no good {EPOCH_LEN:g} s epochs, skipped"
    np.savez(out_path, **result)
    return f"{fif.name}: {result['n_epochs']} epochs, bads={list(result['bad_channels'])}"


def main(run_suffix=run_suffix, n_jobs=1):
    processed_dir, out_dir = processed_dir_for(run_suffix), criticality_dir_for(run_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for fif in sorted(processed_dir.glob('sub-*_task-*_processed_raw.fif')):
        task = fif.name.split('_task-')[1].split('_')[0]
        out_path = out_dir / fif.name.replace('_processed_raw.fif', '_criticality.npz')
        if task in tasks and not out_path.exists():
            todo.append((fif, out_path))

    if n_jobs == 1:
        for args in todo:
            print(_process(*args))
    else:
        with ProcessPoolExecutor(n_jobs) as pool:
            for message in pool.map(_process, *zip(*todo)):
                print(message)

    if any(out_dir.glob('sub-*_criticality.npz')):
        load_all(out_dir).to_csv(criticality_csv_for(run_suffix), index=False)


def load_all(criticality_dir: Path = criticality_dir_for(run_suffix)) -> pd.DataFrame:
    """Load saved results -> long-format DataFrame with columns
    subject, task, band, channel, dfa, fei, n_epochs, bad (one row per subject/task/band/channel).
    Bad channels are included with NaN dfa/fei, so drop them with df[~df.bad] or dropna()."""
    frames = []
    for f in sorted(criticality_dir.glob('sub-*_criticality.npz')):
        d = np.load(f)
        n_bands, n_channels = d['dfa'].shape
        frames.append(pd.DataFrame(dict(
            subject=f.name.split('_')[0].removeprefix('sub-'),
            task=f.name.split('_task-')[1].split('_')[0],
            band=np.repeat(d['bands'], n_channels),
            channel=np.tile(d['ch_names'], n_bands),
            dfa=d['dfa'].ravel(),
            fei=d['fei'].ravel(),
            n_epochs=int(d['n_epochs']),
            bad=np.tile(np.isin(d['ch_names'], d['bad_channels']), n_bands),
        )))
    return pd.concat(frames, ignore_index=True)
    # e.g. per-subject mean over channels: df.groupby(['subject', 'task', 'band'])[['dfa', 'fei']].mean()


if __name__ == '__main__':
    main()
