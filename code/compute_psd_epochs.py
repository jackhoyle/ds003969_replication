
"""Per-subject/task PSD from 1 s epochs, excluding bad channels and bad time spans.

For each processed fif file:
  1. split into 1 s fixed-length epochs; epochs overlapping a BAD_* annotation are dropped
  2. compute a Welch PSD per epoch (one Hann-windowed segment per epoch), bad channels excluded
  3. average across epochs
  4. save to an .npz with every EEG channel present (bad channels are NaN) so that
     results can be aggregated across subjects/tasks with np.nanmean.
"""
from pathlib import Path
import numpy as np
import mne

dataset = "ds003969"
tasks = ['think1', 'think2', 'med1breath', 'med2']
run_suffix = '_v2'

bids_root = Path('/home/jackhoyle') / 'mne_data' / dataset


def processed_dir_for(run_suffix):
    return bids_root / f'processed{run_suffix}'


def psd_dir_for(run_suffix):
    return bids_root / f'outputs{run_suffix}' / 'psd_epoch_mean'


processed_dir = processed_dir_for(run_suffix)
out_dir = psd_dir_for(run_suffix)

FMIN, FMAX = 2, 110
EPOCH_LEN = 1.0


def compute_subject_task_psd(fif_path: Path):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose='error')
    all_eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]

    # reject_by_annotation (default True) drops any epoch overlapping a BAD_ annotation
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_LEN, preload=True, reject_by_annotation=True, verbose='error'
    )
    n_epochs = len(epochs)
    if n_epochs == 0:
        return None

    # picks='eeg' excludes info['bads'] by default
    psd = epochs.compute_psd(
        method='welch', fmin=FMIN, fmax=FMAX,
        n_fft=epochs.get_data().shape[-1],  # one segment per epoch -> 1 Hz resolution
        window='hann', picks='eeg', verbose='error',
    )
    data = psd.get_data()  # (n_epochs, n_good_channels, n_freqs), V^2/Hz

    def to_full(arr):
        # put into the full EEG channel list, NaN where the channel was bad
        full = np.full((len(all_eeg), arr.shape[-1]), np.nan)
        for row, ch in zip(arr, psd.ch_names):
            full[all_eeg.index(ch)] = row
        return full

    return dict(
        freqs=psd.freqs,
        ch_names=np.array(all_eeg),
        bad_channels=np.array(raw.info['bads']),
        n_epochs=n_epochs,
        psd_mean=to_full(data.mean(axis=0)),                  # linear-power mean
        psd_median=to_full(np.median(data, axis=0)),          # robust to outlier epochs
        log_psd_mean=to_full(np.log10(data).mean(axis=0)),    # mean of log10 power
    )


def main(run_suffix=run_suffix):
    processed_dir, out_dir = processed_dir_for(run_suffix), psd_dir_for(run_suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(processed_dir.glob('sub-*_task-*_processed_raw.fif'))
    for fif in files:
        task = fif.name.split('_task-')[1].split('_')[0]
        if task not in tasks:
            continue
        out_path = out_dir / fif.name.replace('_processed_raw.fif', '_psd.npz')
        if out_path.exists():
            continue
        result = compute_subject_task_psd(fif)
        if result is None:
            print(f"{fif.name}: no good epochs, skipped")
            continue
        np.savez(out_path, **result)
        print(f"{fif.name}: {result['n_epochs']} epochs, bads={list(result['bad_channels'])}")


def load_all(psd_dir: Path = out_dir, key: str = 'psd_mean'):
    """Load saved results -> {task: array (n_subjects, n_channels, n_freqs)}, freqs, ch_names,
    {task: [subject ids]} (ids in the same order as axis 0 of each task's array)."""
    by_task = {t: [] for t in tasks}
    subjects = {t: [] for t in tasks}
    freqs = ch_names = None
    for f in sorted(psd_dir.glob('sub-*_psd.npz')):
        task = f.name.split('_task-')[1].split('_')[0]
        d = np.load(f)
        by_task[task].append(d[key])
        subjects[task].append(f.name.split('_')[0].removeprefix('sub-'))
        freqs, ch_names = d['freqs'], d['ch_names']
    present = [t for t, v in by_task.items() if v]
    return ({t: np.stack(by_task[t]) for t in present}, freqs, ch_names,
            {t: subjects[t] for t in present})
    # e.g. grand mean over subjects: np.nanmean(by_task['med2'], axis=0)


if __name__ == '__main__':
    main()
