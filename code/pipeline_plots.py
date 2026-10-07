"""Summary plots for a pipeline run, ported from the exploration notebooks.

  - bad channels per group/task          (exploration/261004_bad_channels.ipynb)
  - ICA components excluded per group/task (exploration/261004_ica_components.ipynb)
  - group mean ± SE power spectra          (exploration/261005_analysis.ipynb)
  - difference spectra: each group minus control, per task; and within-group task contrasts

Only subject/task combinations with a saved PSD (i.e. those used in the analysis) are included.
"""
import re
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from compute_psd_epochs import load_all, tasks
from plotly.colors import hex_to_rgb


def _fif_files(processed_root: Path, run_suffix: str):
    """Yield (subject, task, path) for each processed .fif of a run."""
    pattern = re.compile(r'sub-(.+)_task-(.+)' + re.escape(run_suffix) + '_processed_raw')
    for fif in sorted(processed_root.glob(f'sub-*_task-*{run_suffix}_processed_raw.fif')):
        subject, task = pattern.match(fif.stem).groups()
        yield subject, task, fif


def load_bad_channels(processed_root: Path, run_suffix: str) -> pd.DataFrame:
    # channels marked bad by process_inspect_loop.py are saved in info['bads'] of each .fif,
    # which can be read without loading the data
    rows = []
    for subject, task, fif in _fif_files(processed_root, run_suffix):
        info = mne.io.read_info(fif, verbose='error')
        rows.append(dict(
            subject=subject, task=task,
            n_bad_channels=len(info['bads']),
            n_channels=len(mne.pick_types(info, eeg=True, exclude=[])),
            bad_channels=info['bads'],
        ))
    df = pd.DataFrame(rows)
    df['pct_bad_channels'] = 100 * df.n_bad_channels / df.n_channels
    return df


def parse_ica_description(description):
    # process_inspect_loop.py stores e.g. "ICA excluded components: muscle=[11, 15], eog=[np.int64(2)]"
    muscle, eog = re.search(r'muscle=\[(.*?)\], eog=\[(.*?)\]', description).groups()
    return [int(i) for i in re.findall(r'\d+', muscle)], [int(i) for i in re.findall(r'\d+', eog)]


def load_ica_components(processed_root: Path, run_suffix: str) -> pd.DataFrame:
    rows = []
    for subject, task, fif in _fif_files(processed_root, run_suffix):
        muscle, eog = parse_ica_description(mne.io.read_info(fif, verbose='error')['description'])
        rows.append(dict(
            subject=subject, task=task,
            n_muscle=len(muscle), n_eog=len(eog),
            n_total=len(set(muscle) | set(eog)),  # a component flagged as both is only excluded once
        ))
    return pd.DataFrame(rows)


def group_colours(groups):
    """Fixed colour per group, so a group has the same colour in every plot."""
    return dict(zip(sorted(groups), px.colors.qualitative.Plotly))


def plot_bad_channels(df, colours):
    box = px.box(
        df, x='group', y='n_bad_channels', color='group', points='all', facet_col='task',
        category_orders={'task': tasks, 'group': sorted(colours)}, color_discrete_map=colours,
        hover_data=['subject', 'bad_channels'],
        labels={'n_bad_channels': 'bad channels'},
        title='Bad channels by group',
        height=450,
    )
    # mean ± SE of the number of bad channels, per group (all tasks pooled)
    summary = df.groupby('group').n_bad_channels.agg(['mean', 'sem']).reset_index()
    bar = px.bar(
        summary, x='group', y='mean', error_y='sem', color='group', color_discrete_map=colours,
        labels={'mean': 'mean bad channels'},
        title='Mean bad channels by group (all tasks pooled, ± SE)',
    )
    return box, bar


def plot_ica_components(df, colours):
    long = df.melt(
        id_vars=['subject', 'task', 'group'], value_vars=['n_muscle', 'n_eog'],
        var_name='component_type', value_name='n_components',
    )
    long['component_type'] = long['component_type'].map({'n_muscle': 'muscle', 'n_eog': 'eye (EOG)'})

    box = px.box(
        long, x='group', y='n_components', color='group', points='all',
        facet_col='task', facet_row='component_type',
        category_orders={'task': tasks, 'group': sorted(colours)}, color_discrete_map=colours,
        hover_data=['subject'],
        labels={'n_components': 'ICA components excluded'},
        title='ICA components excluded by group',
        height=700,
    )
    # mean ± SE of the number removed, per group (all tasks pooled)
    summary = long.groupby(['component_type', 'group']).n_components.agg(['mean', 'sem']).reset_index()
    bar = px.bar(
        summary, x='group', y='mean', error_y='sem', color='component_type', barmode='group',
        labels={'mean': 'mean ICA components excluded', 'component_type': ''},
        title='Mean components excluded by group (all tasks pooled, ± SE)',
    )
    return box, bar


# within-subject task contrasts: name -> (tasks averaged on the + side, tasks averaged on the - side)
task_contrasts = {
    'med1breath-think1': (['med1breath'], ['think1']),
    'med2-think2': (['med2'], ['think2']),
    'med-think': (['med1breath', 'med2'], ['think1', 'think2']),
}


def _mean_se(arr):
    """Mean and SE over subjects (axis 0), ignoring NaNs (bad channels)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slices (channel bad for every subject)
        n = np.sum(~np.isnan(arr), axis=0)
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n)


def group_spectra(by_task, subjects, group_of, scale=1e12):  # V²/Hz -> µV²/Hz
    """Mean and SE spectrum per (task, group), averaging over subjects and ignoring NaNs (bad channels).
    Returns {(task, group): (mean, se)}, each (n_channels, n_freqs) in µV²/Hz."""
    out = {}
    for task, arr in by_task.items():                 # arr: (n_subjects, n_channels, n_freqs)
        task_groups = np.array([group_of[s] for s in subjects[task]])
        for group in np.unique(task_groups):
            out[(task, group)] = _mean_se(scale * arr[task_groups == group])
    return out


def subject_task_means(by_task, subjects, name='all_tasks'):
    """Each subject's spectrum averaged over the tasks they have (a channel bad in some tasks is averaged over
    the others), in the same {condition: array} / {condition: [subjects]} form as load_all."""
    rows = {}
    for task, arr in by_task.items():
        for s, spectrum in zip(subjects[task], arr):
            rows.setdefault(s, []).append(spectrum)
    subs = sorted(rows)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # channel bad in every task for a subject
        arr = np.stack([np.nanmean(rows[s], axis=0) for s in subs])
    return {name: arr}, {name: subs}


def group_differences(spectra, control='ctr'):
    """Each group's mean spectrum minus the control group's, per task.
    The groups are independent, so SE of the difference = sqrt(se_group² + se_control²).
    Returns {(task, group): (mean_diff, se)} for the non-control groups."""
    out = {}
    for (task, group), (mean, se) in spectra.items():
        if group == control or (task, control) not in spectra:
            continue
        control_mean, control_se = spectra[(task, control)]
        out[(task, group)] = (mean - control_mean, np.sqrt(se**2 + control_se**2))
    return out


def task_differences(by_task, subjects, group_of, scale=1e12):  # V²/Hz -> µV²/Hz
    """Within-subject task contrasts (see task_contrasts), then mean and SE of the paired difference per group.
    Only subjects with every task in a contrast are included; a channel bad in any of those tasks is NaN.
    Returns {(contrast, group): (mean_diff, se)}, each (n_channels, n_freqs) in µV²/Hz."""
    index = {t: {s: i for i, s in enumerate(subs)} for t, subs in subjects.items()}
    out = {}
    for name, (plus, minus) in task_contrasts.items():
        if not all(t in by_task for t in plus + minus):
            continue
        subs = sorted(set.intersection(*(set(index[t]) for t in plus + minus)))

        def task_mean(ts):  # (n_subjects, n_channels, n_freqs), averaged over the tasks ts
            return np.mean([by_task[t][[index[t][s] for s in subs]] for t in ts], axis=0)

        diff = scale * (task_mean(plus) - task_mean(minus))
        groups = np.array([group_of[s] for s in subs])
        for group in np.unique(groups):
            out[(name, group)] = _mean_se(diff[groups == group])
    return out


def plot_group_spectrum(spectra, freqs, ch_names, ch, condition, colours, title=None, zscore=False,
                        height=500):
    """Group mean spectrum for one channel and condition (a task, or a contrast), with a shaded mean ± SE band.
    zscore=True is for difference spectra: plots z = mean / SE per frequency (no band, the SE is already in z),
    on a linear axis with reference lines at 0 and ±1.96 (uncorrected two-sided p = 0.05)."""
    ch_ix = list(ch_names).index(ch)
    fig = go.Figure()
    for group in sorted(colours):
        if (condition, group) not in spectra:
            continue
        mean, se = (a[ch_ix] for a in spectra[(condition, group)])
        if zscore:
            mean = mean / se
        else:
            lower = np.maximum(mean - se, mean * 1e-3)  # keep positive for the log axis
            upper = mean + se
            r, g, b = hex_to_rgb(colours[group])
            fig.add_trace(go.Scatter(
                x=np.concatenate([freqs, freqs[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill='toself', fillcolor=f'rgba({r},{g},{b},0.3)', line=dict(width=0),
                hoverinfo='skip', showlegend=False, legendgroup=group,
            ))
        fig.add_trace(go.Scatter(
            x=freqs, y=mean, mode='lines', name=group, legendgroup=group,
            line=dict(color=colours[group], width=2),
            hovertemplate=f'{group}<br>%{{x}} Hz: %{{y:.3g}}<extra></extra>',
        ))
    if zscore:
        fig.add_hline(y=0, line=dict(color='grey', width=1))
        for y in (-1.96, 1.96):
            fig.add_hline(y=y, line=dict(color='grey', width=1, dash='dash'))
    fig.update_layout(
        title=title or f'{ch} - {condition} (group mean ± SE)', height=height,
        xaxis_title='Frequency (Hz)', yaxis_title='z (difference / SE)' if zscore else 'PSD (µV²/Hz)',
        yaxis_type='linear' if zscore else 'log',
        legend_title_text='group', hovermode='x unified',
    )
    return fig


def make_all_plots(bids_root: Path, run_suffix: str, psd_dir: Path, plots_dir: Path, channels, control='ctr'):
    """Write all summary plots (HTML) and their underlying tables (CSV) to plots_dir."""
    processed_root = bids_root / f'processed{run_suffix}'
    plots_dir.mkdir(parents=True, exist_ok=True)

    by_task, freqs, ch_names, subjects = load_all(psd_dir, key='psd_mean')
    used = pd.DataFrame([(s, t) for t, subs in subjects.items() for s in subs], columns=['subject', 'task'])

    subject_info = pd.read_csv(bids_root / 'participants.tsv', sep='\t')
    subject_info['subject'] = subject_info['participant_id'].str.removeprefix('sub-')
    group_of = dict(zip(subject_info.subject, subject_info.group))
    colours = group_colours(used.subject.map(group_of).unique())

    def with_groups(df, required):
        df = (
            used
            .merge(df, on=['subject', 'task'], how='left', validate='one_to_one')
            .merge(subject_info[['subject', 'group']], on='subject', how='left', validate='many_to_one')
        )
        assert df[required + ['group']].notna().all().all(), 'missing values for some subject/task'
        return df

    bads = with_groups(load_bad_channels(processed_root, run_suffix), ['n_bad_channels'])
    bads.to_csv(plots_dir / 'bad_channels.csv', index=False)
    for name, fig in zip(['bad_channels_box', 'bad_channels_mean_se'], plot_bad_channels(bads, colours)):
        fig.write_html(plots_dir / f'{name}.html', include_plotlyjs='cdn')

    ica = with_groups(load_ica_components(processed_root, run_suffix), ['n_muscle', 'n_eog'])
    ica.to_csv(plots_dir / 'ica_components.csv', index=False)
    for name, fig in zip(['ica_components_box', 'ica_components_mean_se'], plot_ica_components(ica, colours)):
        fig.write_html(plots_dir / f'{name}.html', include_plotlyjs='cdn')

    spectra = group_spectra(by_task, subjects, group_of)
    spectra_dir = plots_dir / 'spectra'
    spectra_dir.mkdir(exist_ok=True)
    for ch in channels:
        for task in by_task:
            plot_group_spectrum(spectra, freqs, ch_names, ch, task, colours) \
                .write_html(spectra_dir / f'{ch}_{task}.html', include_plotlyjs='cdn')

    # each non-control group minus control, per task and for each subject's mean over tasks (trait difference)
    vs_control = group_differences(spectra, control)
    vs_control |= group_differences(group_spectra(*subject_task_means(by_task, subjects), group_of), control)
    vs_control_dir = plots_dir / f'spectra_diff_vs_{control}'
    vs_control_dir.mkdir(exist_ok=True)
    for ch in channels:
        for task in [*by_task, 'all_tasks']:
            label = 'all tasks (subject mean)' if task == 'all_tasks' else task
            plot_group_spectrum(vs_control, freqs, ch_names, ch, task, colours, zscore=True,
                                title=f'{ch} - {label}: group − {control} (z of mean difference)') \
                .write_html(vs_control_dir / f'{ch}_{task}.html', include_plotlyjs='cdn')

    # within-group task contrasts (paired, per subject)
    contrasts = task_differences(by_task, subjects, group_of)
    contrasts_dir = plots_dir / 'spectra_diff_tasks'
    contrasts_dir.mkdir(exist_ok=True)
    for ch in channels:
        for name, (plus, minus) in task_contrasts.items():
            title = f'{ch} - avg({", ".join(plus)}) − avg({", ".join(minus)})' if len(plus) > 1 \
                else f'{ch} - {plus[0]} − {minus[0]}'
            plot_group_spectrum(contrasts, freqs, ch_names, ch, name, colours, zscore=True,
                                title=f'{title} within group (z of mean paired difference)') \
                .write_html(contrasts_dir / f'{ch}_{name}.html', include_plotlyjs='cdn')
