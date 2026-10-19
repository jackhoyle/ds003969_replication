
"""Per-subject/task PSD from 1 s epochs, excluding bad channels and bad time spans.

For each processed fif file:
  1. split into 1 s fixed-length epochs; epochs overlapping a BAD_* annotation are dropped
  2. compute a Welch PSD per epoch (one Hann-windowed segment per epoch), bad channels excluded
  3. average across epochs
  4. save to an .npz with every EEG channel present (bad channels are NaN) so that
     results can be aggregated across subjects/tasks with np.nanmean.

With interpolate_bads=True, bad channels are instead interpolated (spherical splines) before step 1, so every
channel has a value; results go to a separate folder (psd_dir_for(..., interpolated=True)). Cluster statistics over
electrodes (cluster_stats.py) need this.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np
import pandas as pd

dataset = "ds003969"
tasks = ['think1', 'think2', 'med1breath', 'med2']
run_suffix = '_v2'

bids_root = Path('/home/jackhoyle') / 'mne_data' / dataset

# Hz, used for DFA/fEI (compute_criticality.py) and band power plots (pipeline_plots.py)
freq_bands = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'broadband': (1.0, 30.0),
    'gamma': (60.0, 110.0)
}


paper_subjects_csv = Path(__file__).parent / 'paper_subjects.csv'


def paper_subjects() -> set[str]:
    """Subject ids (e.g. '034') of the 64 age/gender-matched participants analysed by Braboszcz et al. (2017),
    from the subject list in their Zenodo deposit (doi 10.5281/zenodo.57911) mapped to ds003969 ids by recording
    date; 63 are in ds003969 (HYT26 is missing). See writeup/results/paper_included_subjects.csv."""
    df = pd.read_csv(paper_subjects_csv)
    return set(df.loc[df.in_dataset, 'participant_id'].str.removeprefix('sub-'))


def subject_of(path: Path) -> str:
    return path.name.split('_')[0].removeprefix('sub-')


def processed_dir_for(run_suffix):
    return bids_root / f'processed{run_suffix}'


def psd_dir_for(run_suffix, interpolated=False):
    return bids_root / f'outputs{run_suffix}' / ('psd_epoch_mean_interp' if interpolated else 'psd_epoch_mean')


processed_dir = processed_dir_for(run_suffix)
out_dir = psd_dir_for(run_suffix)

FMIN, FMAX = 2, 110
EPOCH_LEN = 1.0


def compute_subject_task_psd(fif_path: Path, interpolate_bads=False):
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose='error')
    all_eeg = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    bads = list(raw.info['bads'])
    if interpolate_bads and bads:
        raw.interpolate_bads(reset_bads=True, verbose='error')

    # reject_by_annotation (default True) drops any epoch overlapping a BAD_ annotation
    epochs = mne.make_fixed_length_epochs(
        raw, duration=EPOCH_LEN, preload=True, reject_by_annotation=True, verbose='error'
    )
    n_epochs = len(epochs)
    if n_epochs == 0:
        return None

    # explicit good-channel picks: with picks='eeg', the Spectrum's ch_names still list the bad channels while its
    # data rows exclude them, so names and rows would not line up
    good = [epochs.ch_names[i] for i in mne.pick_types(epochs.info, eeg=True, exclude='bads')]
    psd = epochs.compute_psd(
        method='welch', fmin=FMIN, fmax=FMAX,
        n_fft=epochs.get_data().shape[-1],  # one segment per epoch -> 1 Hz resolution
        window='hann', picks=good, verbose='error',
    )
    data = psd.get_data()  # (n_epochs, n_good_channels, n_freqs), V^2/Hz
    assert list(psd.ch_names) == good and data.shape[1] == len(good)

    def to_full(arr):
        # put into the full EEG channel list, NaN where the channel was bad
        full = np.full((len(all_eeg), arr.shape[-1]), np.nan)
        for row, ch in zip(arr, good, strict=True):
            full[all_eeg.index(ch)] = row
        return full

    return dict(
        freqs=psd.freqs,
        ch_names=np.array(all_eeg),
        bad_channels=np.array(bads),
        interpolated=interpolate_bads,
        n_epochs=n_epochs,
        psd_mean=to_full(data.mean(axis=0)),                  # linear-power mean
        psd_median=to_full(np.median(data, axis=0)),          # robust to outlier epochs
        log_psd_mean=to_full(np.log10(data).mean(axis=0)),    # mean of log10 power
    )


def _process(fif: Path, out_path: Path, interpolate_bads=False):
    result = compute_subject_task_psd(fif, interpolate_bads)
    if result is None:
        return f"{fif.name}: no good epochs, skipped"
    np.savez(out_path, **result)
    return f"{fif.name}: {result['n_epochs']} epochs, bads={list(result['bad_channels'])}"


def main(run_suffix=run_suffix, n_jobs=1, subjects=None, interpolate_bads=False):
    """subjects: optional set of subject ids (e.g. paper_subjects()) to restrict to; None means all.
    interpolate_bads: interpolate bad channels instead of leaving them NaN; writes to a separate folder."""
    processed_dir, out_dir = processed_dir_for(run_suffix), psd_dir_for(run_suffix, interpolate_bads)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for fif in sorted(processed_dir.glob('sub-*_task-*_processed_raw.fif')):
        task = fif.name.split('_task-')[1].split('_')[0]
        out_path = out_dir / fif.name.replace('_processed_raw.fif', '_psd.npz')
        if subjects is not None and subject_of(fif) not in subjects:
            continue
        if task in tasks and not out_path.exists():
            todo.append((fif, out_path, interpolate_bads))

    if n_jobs == 1:
        for args in todo:
            print(_process(*args))
    elif todo:
        with ProcessPoolExecutor(n_jobs) as pool:
            for message in pool.map(_process, *zip(*todo)):
                print(message)


def load_all(psd_dir: Path = out_dir, key: str = 'psd_mean', only_subjects=None):
    """Load saved results -> {task: array (n_subjects, n_channels, n_freqs)}, freqs, ch_names,
    {task: [subject ids]} (ids in the same order as axis 0 of each task's array).
    only_subjects: optional set of subject ids to restrict to; None loads all."""
    by_task = {t: [] for t in tasks}
    subjects = {t: [] for t in tasks}
    freqs = ch_names = None
    for f in sorted(psd_dir.glob('sub-*_psd.npz')):
        if only_subjects is not None and subject_of(f) not in only_subjects:
            continue
        task = f.name.split('_task-')[1].split('_')[0]
        d = np.load(f)
        by_task[task].append(d[key])
        subjects[task].append(subject_of(f))
        freqs, ch_names = d['freqs'], d['ch_names']
    present = [t for t, v in by_task.items() if v]
    return ({t: np.stack(by_task[t]) for t in present}, freqs, ch_names,
            {t: subjects[t] for t in present})
    # e.g. grand mean over subjects: np.nanmean(by_task['med2'], axis=0)


if __name__ == '__main__':
    main()
