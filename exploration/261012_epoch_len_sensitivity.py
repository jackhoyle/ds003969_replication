"""How much do DFA / fEI estimates depend on compute_criticality.EPOCH_LEN?

For every processed recording of a run, recompute DFA and fEI per band for a range of epoch lengths, keeping
the per-epoch, per-channel values so that data loss, between-epoch noise and bias can be separated afterwards.

Usage (from the code/ directory, which holds the modules imported below):
    python ../exploration/261012_epoch_len_sensitivity.py --suffix _v4 --n-jobs 18
Writes outputs{suffix}/epoch_len_sensitivity/{epochs,recordings}.parquet.
"""
import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')  # see compute_criticality.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))

import mne  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from compute_criticality import DFA_WINDOW, FEI_OVERLAP, FEI_WINDOW  # noqa: E402
from compute_psd_epochs import bids_root, freq_bands, processed_dir_for, tasks  # noqa: E402
from crosci.biomarkers import DFA, fEI  # noqa: E402

EPOCH_LENS = [40, 60, 90, 120, 180, 240, 300]  # s; must exceed the largest DFA window (30 s)


def run_one(fif: Path):
    raw = mne.io.read_raw_fif(fif, preload=True, verbose='error')
    raw.pick('eeg', exclude='bads')
    subject = fif.name.split('_')[0].removeprefix('sub-')
    task = fif.name.split('_task-')[1].split('_')[0]
    sfreq = raw.info['sfreq']
    bad_s = sum(d for d, desc in zip(raw.annotations.duration, raw.annotations.description) if desc.startswith('BAD'))

    epoch_rows, rec_rows = [], []
    for epoch_len in EPOCH_LENS:
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_len, preload=True,
                                              reject_by_annotation=True, verbose='error')
        rec_rows.append(dict(subject=subject, task=task, epoch_len=epoch_len, n_epochs=len(epochs),
                             duration=raw.times[-1], bad_s=bad_s))
        if len(epochs) == 0:
            continue
        for band, (l_freq, h_freq) in freq_bands.items():
            envelope = (epochs.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose='error')
                        .apply_hilbert(envelope=True).get_data())
            for e, epoch in enumerate(envelope):
                dfa, *_ = DFA(epoch, sfreq, DFA_WINDOW, DFA_WINDOW, overlap=True)
                fei, *_ = fEI(epoch, sfreq, FEI_WINDOW, FEI_OVERLAP, dfa)
                epoch_rows.append(pd.DataFrame(dict(
                    subject=subject, task=task, epoch_len=epoch_len, band=band, epoch=e,
                    epoch_start=epochs.events[e, 0] / sfreq - raw.first_time,
                    channel=epochs.ch_names, dfa=dfa, fei=np.ravel(fei),
                )))
    return (pd.concat(epoch_rows) if epoch_rows else None), pd.DataFrame(rec_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='_v4')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None, help='only the first N recordings (for testing)')
    args = parser.parse_args()

    fifs = [f for f in sorted(processed_dir_for(args.suffix).glob('sub-*_task-*_processed_raw.fif'))
            if f.name.split('_task-')[1].split('_')[0] in tasks][:args.limit]
    out_dir = bids_root / f'outputs{args.suffix}' / 'epoch_len_sensitivity'
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs, recordings = [], []
    with warnings.catch_warnings(), ProcessPoolExecutor(args.n_jobs) as pool:
        warnings.simplefilter('ignore', RuntimeWarning)
        for i, (ep, rec) in enumerate(pool.map(run_one, fifs, chunksize=1)):
            epochs.append(ep)
            recordings.append(rec)
            print(f'{i + 1}/{len(fifs)}', flush=True)
    pd.concat(epochs).to_parquet(out_dir / 'epochs.parquet', index=False)
    pd.concat(recordings).to_parquet(out_dir / 'recordings.parquet', index=False)


if __name__ == '__main__':
    main()
