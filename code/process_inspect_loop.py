from pathlib import Path
import numpy as np
import plotly.express as px
import os

import openneuro
import mne
from mne.preprocessing import ICA
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    get_entity_vals,
)

import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')
plt.ion()

import logging
logging.basicConfig(level=logging.INFO)

from preprocessing import preprocess
from psd_lof import find_bad_channels_psd_lof

dataset = "ds003969"

root = Path('/home/jackhoyle')

# Create a BIDS root directory in the MNE dataset folder
bids_root = root / 'mne_data' / dataset
bids_root.mkdir(parents=True, exist_ok=True)

# detect all subjects/tasks present in the dataset; combinations whose
# recording is missing (e.g. not downloaded) are skipped in the loop below
subjects = get_entity_vals(bids_root, 'subject')
tasks = get_entity_vals(bids_root, 'task')
logging.info(f"Found subjects {subjects} and tasks {tasks} in {dataset}")

processed_root = bids_root / 'processed'
processed_root.mkdir(parents=True, exist_ok=True)

already_done = set(f.name for f in processed_root.glob("sub-*_task-*_processed_raw.fif"))

for subject in subjects:
    for task in tasks:
        if f"sub-{subject}_task-{task}_processed_raw.fif" in already_done:
            logging.info(f"Skipping already processed subject {subject}, task {task}")
            continue
        if not BIDSPath(subject=subject, root=bids_root, task=task, datatype="eeg",
                        suffix="eeg", extension=".bdf").fpath.exists():
            logging.info(f"No data for subject {subject}, task {task}, skipping")
            continue
        # load data
        logging.info(f"Loading data for subject {subject}, task {task}")
        raw: mne.io.edf.edf.RawBDF = read_raw_bids(
            BIDSPath(subject=subject, root=bids_root, task=task, datatype="eeg"),
            verbose=False
        )
        raw.load_data()
        raw.set_montage("colin27_1020", on_missing="warn")
        assert type(raw) == mne.io.edf.edf.RawBDF

        preprocess(
            raw,
            reference='EXG6',
            subsample_freq=256,
            high_thresh=None,
            low_thresh=1,
            pop_rejcont=True,
            bad_channel_variance_threshold=None
        )

        # now inspect and mark bad channels if needed
        # automated bad channel suggestions
        bad_waveform_dist = mne.preprocessing.find_bad_channels_lof(raw, picks='eeg', threshold=1.5)
        # for spectral dist suggestion restrict to less than 45Hz to avoid line noise
        # for this replication line noise is just excluded for analysis
        bad_spectral_dist = find_bad_channels_psd_lof(raw, fmax=45, threshold=2, verbose=False)
        
        psd = raw.compute_psd()
        psd.plot(picks="eeg", average=False)

        print(f'Suggested bad channels from waveform dist: {bad_waveform_dist}')
        print(f'Suggested bad channels from spectral dist: {bad_spectral_dist}')

        # manually mark bad channels here
        raw.plot(duration=10, scalings=dict(eeg=40e-6), block=True)

        # Now ICA

        ica = ICA(n_components=20, method='infomax', max_iter="auto", rng=97)
        ica.fit(raw, decim = 5)

        muscle_bads, muscle_bad_scores = ica.find_bads_muscle(raw)
        # ica.plot_scores(muscle_bad_scores)
        # ica.plot_properties(raw, picks=muscle_bads)
        # input("Press the Enter key to continue: ")

        raw.set_channel_types({'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog'})

        eog_bads, eog_bad_scores = ica.find_bads_eog(raw)
        # ica.plot_scores(eog_bad_scores)
        # ica.plot_properties(raw, picks=eog_bads)
        # input("Press the Enter key to continue: ")

        # now reconstruct
        ica.exclude = muscle_bads + eog_bads
        print(f'Excluding ICA components: {ica.exclude}')

        ica.apply(raw)

        # record excluded ICA components in the raw object (persisted in the .fif)
        raw.info['description'] = (
            f"ICA excluded components: muscle={muscle_bads}, eog={eog_bads}"
        )

        # save processed data
        raw.save(processed_root / f"sub-{subject}_task-{task}_processed_raw.fif", overwrite=True)