# ds003969 EEG meditation replication

A from-scratch MNE-Python replication of [Braboszcz et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5261734/), which amongst other things reports higher gamma (60–110 Hz) power in meditators than in controls. The project uses the public dataset [OpenNeuro ds003969](https://openneuro.org/datasets/ds003969/versions/1.0.0) and extends the analysis to criticality measures: long-range temporal correlations (DFA) and functional E/I balance (fEI).

**Findings** (full details in [`writeup/replication_writeup.pdf`](writeup/replication_writeup.pdf)):
- **The pipeline reproduces the paper's processing:** the analysis uses the paper's own matched sample, recovered from the authors' [Zenodo deposit](https://doi.org/10.5281/zenodo.57911) (63 of 64 participants; HYT26 is missing from the public release). Without ICA, per-subject parieto-occipital gamma matches the paper's published values at r = 0.97, and group differences agree to within 1 dB. Remaining discrepancies come from the paper's manual steps (ICA component choice, bad channels, segment rejection), which were replaced by automated equivalents.
- **Gamma replicates:** each meditator group exceeds controls by 3.5–4.8 dB over parieto-occipital electrodes, during meditation and as a trait. Automated ICA shrinks this by about 1 dB, mostly through a few recordings; the Vipassana difference is then no longer significant, but the pooled meditators still are.
- **Alpha replicates:** the Vipassana group has more 7–11 Hz power than controls and the other traditions during meditation, with or without ICA.
- **The electrode-level results don't survive correction across tests:** the paper's cluster permutation tests reproduce test by test, but none survive a 5% false discovery rate correction across the paper's 23 tests (smallest q = 0.097).
- **The gamma–experience correlation is fragile:** the paper's r = 0.33 with log lifetime practice hours reproduces exactly, but only after removing two outliers. Redoing the outlier detection within the bootstrap gives a confidence interval that includes zero, and this replication's own gamma values show no relationship (r = 0.16 without ICA, −0.05 with).
- **Criticality shows no meditator–control trait differences in DFA**, in line with [Lewis-Healey et al. (2026)](https://www.biorxiv.org/content/10.64898/2026.08.07.743470v1). The exception is higher alpha fEI in the Vipassana group, which closely tracks that group's higher alpha power and cannot be separated from it.
- **Beta DFA rises during meditation, but this can't be attributed to meditation:** it is higher in the second meditation block than the first, but the rise is not significantly larger than the equivalent rise across the two mind-wandering blocks, so it cannot be separated from a generic time-on-task effect.
- The criticality results are robust to removing ICA and to using 90 s instead of 180 s epochs.

## Layout

| Path | Contents |
|---|---|
| `code/` | The pipeline (see below), plus a Python port of EEGLAB's `pop_rejcont` and a spectral bad-channel detector, both with tests |
| `exploration/` | Dated notebooks and scripts (`YYMMDD_*`) from developing the pipeline and analyses |
| `notes/log.md` | Day-by-day project log: decisions, dead ends, open questions |
| `writeup/` | LaTeX write-up (`replication_writeup.tex`), the scripts that produce its numbers and figures (`scripts/`), and their outputs (`results/`, `figures/`) |

## Running the pipeline

Install the dependencies, then download ds003969 (e.g. with `openneuro-py`):

```bash
pip install -r requirements.txt
```

The data location is hard-coded as `bids_root` (`~/mne_data/ds003969`) in `code/compute_psd_epochs.py` and `code/process_inspect_loop.py`. Edit both if your data is elsewhere.

Run the pipeline from `code/`:

```bash
python run_pipeline.py --suffix _matchsample_v1 --paper-subjects --interpolate-bads           # preprocess + ICA -> PSDs -> DFA/fEI -> plots
python run_pipeline.py --suffix _matchsample_noica_v1 --paper-subjects --interpolate-bads --no-ica  # same without ICA
python run_pipeline.py --help                                                                 # all options (manual ICA review, skipping steps, ...)
```

These are the two runs behind the write-up. `--paper-subjects` restricts every step to the paper's matched sample (listed in `code/paper_subjects.csv`); without it, the pipeline processes all 98 participants. Automatically detected bad channels are capped at 20 per recording (`--max-bad-channels`), dropping the weakest suggestions first, starting with those made only by the spectral detector. `--interpolate-bads` interpolates bad channels before computing spectra (written to `psd_epoch_mean_interp/`), which the cluster statistics over electrodes need; the write-up's parieto-occipital means still leave each recording's bad channels out.

Each run writes to two folders next to the data: `processed{suffix}/` holds the cleaned recordings, and `outputs{suffix}/` holds per-recording PSDs, `criticality.csv` and HTML plots. Results that already exist are skipped, so re-runs are cheap.

The pipeline has four steps:
1. **Preprocessing:** right-mastoid reference, 256 Hz, 1 Hz high-pass, `pop_rejcont` artefact rejection, automatic bad-channel detection, then Infomax ICA with automatic EOG/muscle component removal (1–5 components per recording, to mirror the paper's manual choice).
2. **Spectra:** PSDs from 1 s epochs, with bad channels left empty or (with `--interpolate-bads`) interpolated.
3. **Criticality:** DFA and fEI per frequency band via `crosci`, on 180 s epochs (`--criticality-epoch-len` for other lengths).
4. **Plots:** group plots.

## Reproducing the write-up

After the two pipeline runs, and with the paper's Zenodo deposit ([doi 10.5281/zenodo.57911](https://doi.org/10.5281/zenodo.57911)) downloaded to `<zenodo_dir>`:

```bash
cd code
python ../writeup/scripts/matched_sample.py <zenodo_dir> ../writeup/results/matched  # cluster-mean group tests, agreement with the paper's values
python ../writeup/scripts/cluster_tests.py --n-jobs 8                                # cluster permutation tests (code/cluster_stats.py) + FDR
python -c "import compute_criticality as c, compute_psd_epochs as p; [c.main(s, n_jobs=8, subjects=p.paper_subjects(), epoch_len=90) for s in ('_matchsample_v1', '_matchsample_noica_v1')]"  # 90 s robustness check
cd ../writeup
python scripts/matched_figs.py                         # Figs 1-2
python scripts/experience.py <zenodo_dir>              # gamma vs practice hours (Fig 3)
python scripts/criticality_final.py                    # H1/H2 regressions, Fig 4 and Fig 7
python scripts/criticality_final.py --epoch-len 90     # 90 s robustness check (results CSV only)
python scripts/results_tables.py                       # LaTeX tables
latexmk -pdf replication_writeup.tex
```

Supporting checks referred to in the write-up: the epoch-length sweep is `exploration/261012_epoch_len_*.py`, the hypothesis models were developed in `exploration/261020_criticality_final.ipynb`, and the gamma/experience/age exploration is `exploration/261019_gamma_experience_age.ipynb`.
