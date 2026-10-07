
import mne
import numpy as np
from pop_rejcont import rejcont


def preprocess(
        raw: mne.io.edf.edf.RawEDF,
        reference: str | list[str] | None = None,
        subsample_freq: float | None = None,
        high_thresh: float | None = None,
        low_thresh: float | None = None,
        pop_rejcont: bool = True,
        bad_channel_variance_threshold: float | None = None,
        filter_order: int = 6
    ):
    # 0: FP1 electrode is not used and recorded with Ext8 instead 
    # Find the indices of the channels
    idx_bad = raw.ch_names.index('Fp1')
    idx_good = raw.ch_names.index('EXG8')

    # Replace the data in place
    raw._data[idx_bad] = raw._data[idx_good]

    #1: set reference
    if reference is not None:
        raw.set_eeg_reference(reference if isinstance(reference, list) else [reference]) # EXG6 is right mastoid

    #2: subsample
    if subsample_freq is not None:
        raw.resample(subsample_freq)

    #3: apply a high pass filter 
    if high_thresh is not None or low_thresh is not None:
        iir_params = dict(order=filter_order, ftype='butter')
        raw.filter(l_freq=low_thresh, h_freq=high_thresh, l_trans_bandwidth=0.3, h_trans_bandwidth=0.3, method='iir', iir_params=iir_params)

    #4: "pop_rejcont" automated rejection of artifact segments (port of EEGLAB's pop_rejcont, see code/preprocessing.py)
    # 1 s epochs, 0.5 s step (EEGLAB 'overlap'=0.5), 0-10 Hz > 17 dB and 35-128 Hz > 14 dB, 
    # contiguous=8 (-> merge distance 4 s), thresholds relative to the recording's mean spectrum.
    if pop_rejcont:
        regions = rejcont(raw, bands=((0, 10, 17), (35, 128, 14)), epochlength=1.0,
                        recurrence=0.5, contiguous=8, addlength=0.25, picks='eeg')
        
        print(f"{len(regions)} regions, {sum(d for _, d in regions):.1f} s of {raw.times[-1]:.0f} s rejected")

    # 5. Visual inspection and potential adjustment of rejection thresholds?
    #  "The data was then checked visually for
    # any potential remaining artifact—leading in some rare cases to adjust decibels thresholds from
    # 1 to 5 dB and rerunning the rejection procedure."

    # for now I'm going to skip this as it's super unclear what they did

    # 6. automated bad channel identification
    if bad_channel_variance_threshold is not None:
        bad_channels = identify_bad_channels(raw, threshold_min_variance=bad_channel_variance_threshold)

def identify_bad_channels(
        raw: mne.io.edf.edf.RawEDF, 
        threshold_min_variance: bool = True,
        min_var_threshold: float = 0.01,
        verbose: bool = True,
    ):
    bad_channels = []

    eeg_ixs = [i for i in mne.pick_types(raw.info, eeg=True, exclude=[])]
    eeg_ch_names = [raw.ch_names[i] for i in eeg_ixs]
    data = raw.get_data(picks='eeg')

    variances = np.var(data, axis=1)
    mean_variance = np.mean(variances)

    if threshold_min_variance:
        bad_bool = variances < min_var_threshold * mean_variance
        for i, is_bad in enumerate(bad_bool):
            if is_bad:
                if verbose:
                    print(f"Channel {eeg_ch_names[i]} fails min variance threshold ({variances[i] / mean_variance:.4f})")
                if eeg_ch_names[i] not in bad_channels:
                    bad_channels.append(eeg_ch_names[i])

    for bad_ch in bad_channels:
        if bad_ch not in raw.info['bads']:
            raw.info['bads'].append(bad_ch)
    
    return bad_channels