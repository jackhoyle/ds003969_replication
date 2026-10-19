import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
from mne.preprocessing import ICA
from mne_bids import (
    BIDSPath,
    get_entity_vals,
    read_raw_bids,
)
from preprocessing import preprocess
from psd_lof import find_bad_channels_psd_lof

dataset = "ds003969"
root = Path('/home/jackhoyle')
run_suffix = '_v4'

auto_detect_bad_channels = True

bids_root = root / 'mne_data' / dataset

# bad channel detection: LOF score at or above which a channel is suggested as bad
WAVEFORM_LOF_THRESHOLD = 1.5
SPECTRAL_LOF_THRESHOLD = 2
MAX_BAD_CHANNELS = 20


def cap_bad_channels(waveform_scores, spectral_scores, max_bad_channels=MAX_BAD_CHANNELS, n_existing=0):
    """Combine the two detectors' suggestions, keeping at most max_bad_channels bad channels in total.

    waveform_scores / spectral_scores: {channel: LOF score}. A channel is suggested by a detector when its score is
    at or above that detector's threshold. The strength of a suggestion is score / threshold (>= 1).
    n_existing: channels already marked bad, which always stay bad and count towards the cap.
    max_bad_channels: None or 0 means no cap.

    Over the cap, suggestions are dropped in this order:
      1. channels suggested only by the spectral detector, weakest first;
      2. channels suggested only by the waveform detector, weakest first;
      3. channels suggested by both, weakest (by their larger strength) first.
    Returns (kept, dropped): channel name lists, each strongest first.
    """
    waveform = {ch: s / WAVEFORM_LOF_THRESHOLD for ch, s in waveform_scores.items() if s >= WAVEFORM_LOF_THRESHOLD}
    spectral = {ch: s / SPECTRAL_LOF_THRESHOLD for ch, s in spectral_scores.items() if s >= SPECTRAL_LOF_THRESHOLD}

    def priority(ch):  # larger = kept first
        tier = 2 if (ch in waveform and ch in spectral) else 1 if ch in waveform else 0
        return tier, max(waveform.get(ch, 0), spectral.get(ch, 0))

    ranked = sorted(set(waveform) | set(spectral), key=priority, reverse=True)
    if not max_bad_channels:
        return ranked, []
    n_keep = max(max_bad_channels - n_existing, 0)
    return ranked[:n_keep], ranked[n_keep:]


def process_all(run_suffix=run_suffix, auto_detect_bad_channels=auto_detect_bad_channels, n_jobs=1, run_ica=True,
                only_subjects=None, max_bad_channels=MAX_BAD_CHANNELS):
    """Preprocess, mark bad channels, run ICA (unless run_ica=False) and save every subject/task to
    processed{run_suffix}/. only_subjects: optional set of subject ids (e.g. '034') to restrict to.
    max_bad_channels: cap on automatically marked bad channels (see cap_bad_channels); None or 0 for no cap."""
    if not auto_detect_bad_channels:
        # interactive plots are only needed for manual bad channel marking
        import matplotlib.pyplot as plt
        plt.switch_backend('Qt5Agg')
        plt.ion()

    bids_root.mkdir(parents=True, exist_ok=True)

    # detect all subjects/tasks present in the dataset; combinations whose
    # recording is missing (e.g. not downloaded) are skipped in the loop below
    subjects = get_entity_vals(bids_root, 'subject')
    if only_subjects is not None:
        subjects = [s for s in subjects if s in only_subjects]
    tasks = get_entity_vals(bids_root, 'task')
    logging.info(f"Found subjects {subjects} and tasks {tasks} in {dataset}")

    processed_root = bids_root / f'processed{run_suffix}'
    processed_root.mkdir(parents=True, exist_ok=True)

    already_done = set(f.name for f in processed_root.glob("sub-*_task-*_processed_raw.fif"))

    todo = []
    for subject in subjects:
        for task in tasks:
            if f"sub-{subject}_task-{task}{run_suffix}_processed_raw.fif" in already_done:
                logging.info(f"Skipping already processed subject {subject}, task {task}")
                continue
            if not BIDSPath(subject=subject, root=bids_root, task=task, datatype="eeg",
                            suffix="eeg", extension=".bdf").fpath.exists():
                logging.info(f"No data for subject {subject}, task {task}, skipping")
                continue
            todo.append((subject, task))

    # manual bad channel marking needs interactive plots, so only run in parallel in auto mode
    if n_jobs == 1 or not auto_detect_bad_channels:
        for subject, task in todo:
            print(process_one(subject, task, processed_root, run_suffix, auto_detect_bad_channels, run_ica,
                              max_bad_channels))
    elif todo:
        with ProcessPoolExecutor(n_jobs) as pool:
            futures = [pool.submit(process_one, subject, task, processed_root, run_suffix, True, run_ica,
                                   max_bad_channels)
                       for subject, task in todo]
            for future in futures:
                print(future.result())


def process_one(subject, task, processed_root: Path, run_suffix, auto_detect_bad_channels, run_ica=True,
                max_bad_channels=MAX_BAD_CHANNELS):
    """Preprocess, mark bad channels, run ICA (unless run_ica=False) and save one subject/task."""
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
    good_eeg = mne.pick_types(raw.info, eeg=True, exclude='bads')
    bad_waveform_dist, waveform_scores = mne.preprocessing.find_bad_channels_lof(
        raw, picks=good_eeg, threshold=WAVEFORM_LOF_THRESHOLD, return_scores=True)
    # MNE returns LOF scores negated (flags |score| >= threshold)
    waveform_scores = dict(zip([str(raw.ch_names[i]) for i in good_eeg], abs(waveform_scores)))
    # for spectral dist suggestion restrict to less than 45Hz to avoid line noise
    # for this replication line noise is just excluded for analysis
    bad_spectral_dist, spectral_scores = find_bad_channels_psd_lof(
        raw, fmax=45, threshold=SPECTRAL_LOF_THRESHOLD, return_scores=True, verbose=False)
    spectral_scores = {str(ch): score for ch, score in spectral_scores.items()}

    print(f'sub-{subject} task-{task}: suggested bad channels from waveform dist: {bad_waveform_dist}')
    print(f'sub-{subject} task-{task}: suggested bad channels from spectral dist: {bad_spectral_dist}')

    if auto_detect_bad_channels:
        suggested, dropped = cap_bad_channels(waveform_scores, spectral_scores, max_bad_channels,
                                              n_existing=len(raw.info['bads']))
        if dropped:
            print(f'sub-{subject} task-{task}: {len(suggested) + len(dropped)} bad channels suggested, over the cap '
                  f'of {max_bad_channels}; not marking the weakest: {dropped}')
        raw.info['bads'] = list(dict.fromkeys(raw.info['bads'] + suggested))
    else:
        # manual bad channel marking
        psd = raw.compute_psd(verbose=False)
        psd.plot(picks="eeg", average=False)

        # manually mark bad channels here
        raw.plot(duration=10, scalings=dict(eeg=40e-6), block=True)

    raw.set_channel_types({'EXG1': 'eog', 'EXG2': 'eog', 'EXG3': 'eog', 'EXG4': 'eog'})
    out_path = processed_root / f"sub-{subject}_task-{task}{run_suffix}_processed_raw.fif"

    if not run_ica:
        raw.info['description'] = "ICA not run"
        raw.save(out_path, overwrite=True)
        return f"sub-{subject} task-{task}: bads={raw.info['bads']}, ICA not run"

    # Now ICA
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
    ica = ICA(n_components=20, method='infomax', max_iter="auto", rng=97)
    ica.fit(raw, picks=picks, decim = 5)

    eog_bads, eog_bad_scores = ica.find_bads_eog(raw, threshold=3.0, verbose=False)
    eog_bads = [int(i) for i in eog_bads]
    # if len(eog_bads) > 0:
    #     ica.plot_scores(eog_bad_scores)
    #     ica.plot_properties(raw, picks=eog_bads)
    #     print(eog_bads)
    #     input("Press the Enter key to continue: ")

    # at most 4 here
    eog_bads = eog_bads[:min(4, len(eog_bads))]

    muscle_bads, muscle_bad_scores = ica.find_bads_muscle(raw, threshold=0.75, verbose=False)
    # if len(muscle_bads)>0:
    #     ica.plot_scores(muscle_bad_scores)
    #     ica.plot_properties(raw, picks=muscle_bads)

    # always remove at least 1 ICA component, and at most 5 in total
    max_muscle = 5 - len(eog_bads)
    if len(muscle_bads) == 0:
        # always remove one
        muscle_bads = [max(range(len(muscle_bad_scores)), key=muscle_bad_scores.__getitem__)]
    elif len(muscle_bads) > max_muscle:
        # take the max_muscle components with the highest scores
        muscle_bads = sorted(muscle_bads, key=lambda x: muscle_bad_scores[x], reverse=True)[:max_muscle]
    # now reconstruct
    ica.exclude = sorted(set(muscle_bads + eog_bads))
    print(f'sub-{subject} task-{task}: suggested ICA components: muscle={muscle_bads}, eog={eog_bads}')

    if not auto_detect_bad_channels:
        select_ica_components(ica, raw, eog_bads, eog_bad_scores, muscle_bads, muscle_bad_scores)

    print(f'sub-{subject} task-{task}: excluding ICA components: {ica.exclude}')

    ica.apply(raw)

    # record excluded ICA components in the raw object (persisted in the .fif)
    raw.info['description'] = (
        f"ICA excluded components: {ica.exclude} "
        f"(suggested muscle={muscle_bads}, eog={eog_bads})"
    )

    # save processed data
    raw.save(out_path, overwrite=True)

    return f"sub-{subject} task-{task}: bads={raw.info['bads']}, ICA excluded={ica.exclude}"


def select_ica_components(ica, raw, eog_bads, eog_bad_scores, muscle_bads, muscle_bad_scores):
    """Interactively choose which ICA components to exclude, starting from the suggestions.

    Updates ica.exclude in place. Two stages, one after the other (both plots edit
    ica.exclude, so they must not be open at the same time):
      1. Topomaps of all components, alongside the scores and properties of the
         suggested ones. Click a component's title to toggle it bad (grey) or good
         (black); click a topomap to open its properties. Close the topomap window
         to continue.
      2. Component time courses. Click a trace to toggle it bad (grey) or good;
         click a component name to open its properties. Close the window to accept.
    """
    import matplotlib.pyplot as plt

    # context for the suggestions (left open while choosing)
    # ica.plot_scores(eog_bad_scores, exclude=eog_bads, title='EOG scores')
    # ica.plot_scores(muscle_bad_scores, exclude=muscle_bads, title='Muscle scores')
    # if ica.exclude:
    #     ica.plot_properties(raw, picks=ica.exclude, show=True)

    # stage 1: topomaps, wait until closed
    print('Click component titles to mark bad/good, click topomaps for properties. '
          'Close the topomap window to continue.')
    figs = ica.plot_components(inst=raw, show=True)
    figs = figs if isinstance(figs, list) else [figs]
    while any(plt.fignum_exists(fig.number) for fig in figs):
        plt.pause(0.1)

    # stage 2: time courses, blocks until closed and sets ica.exclude on close
    print(f'Currently excluding {sorted(ica.exclude)}. Click traces to mark bad/good. '
          'Close the sources window to accept.')
    ica.plot_sources(raw, block=True)

    ica.exclude = sorted(int(i) for i in ica.exclude)
    plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    process_all()
