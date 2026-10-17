# EEG meditation replication

A from-scratch MNE-Python replication of [Braboszcz et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5261734/), which reports higher gamma (60–110 Hz) power in meditators than in controls. The project uses the public dataset [OpenNeuro ds003969](https://openneuro.org/datasets/ds003969/versions/1.0.0) and extends the analysis to criticality measures: long-range temporal correlations (DFA) and functional E/I balance (fEI).

**Findings so far** (details in [writeup/](writeup/)). All results use the paper's own age- and gender-matched sample (63 of its 64 participants), recovered from the authors' Zenodo deposit because the dataset doesn't record which participants the paper analysed. Earlier full-sample runs gave a misleading picture and are superseded.
- **The pipeline reproduces the paper:** without ICA, per-subject parieto-occipital gamma matches the paper's published values at r = 0.97, and group differences agree to within 0.6 dB.
- **Gamma replicates:** meditators exceed controls by about 4 dB during meditation. Automated ICA shrinks this to about 2.5 dB (still significant when the three traditions are pooled), mostly through a few recordings.
- **Alpha replicates:** the Vipassana group has more 7–11 Hz power than controls and the other traditions.
- **Criticality shows no meditator–control differences.** The one within-meditator effect (beta DFA higher in the second meditation block) appears equally in controls, so it looks like time-on-task.

## Layout

| Path | Contents |
|---|---|
| `code/` | The pipeline (see below), plus a Python port of EEGLAB's `pop_rejcont` and a spectral bad-channel detector, both with tests |
| `exploration/` | Dated notebooks and scripts (`YYMMDD_*`) from developing the pipeline and analyses |
| `notes/log.md` | Day-by-day project log: decisions, dead ends, open questions |
| `writeup/` | LaTeX write-up, its figures, and the scripts and CSVs behind its numbers |

## Running the pipeline

Install the dependencies, then download ds003969 (e.g. with `openneuro-py`):

```bash
pip install -r requirements.txt
```

The data location is hard-coded as `bids_root` (`~/mne_data/ds003969`) in `code/compute_psd_epochs.py` and `code/process_inspect_loop.py`. Edit both if your data is elsewhere.

Run the pipeline from `code/`:

```bash
python run_pipeline.py --suffix _matchsample_v0 --paper-subjects                 # preprocess + ICA -> PSDs -> DFA/fEI -> plots
python run_pipeline.py --suffix _matchsample_noica_v0 --paper-subjects --no-ica  # same without ICA
python run_pipeline.py --help                                                    # all options (manual ICA review, skipping steps, ...)
```

These are the two runs behind the write-up. `--paper-subjects` restricts every step to the paper's matched sample (listed in `code/paper_subjects.csv`); without it, the pipeline processes all 98 participants.

Each run writes to two folders next to the data: `processed{suffix}/` holds the cleaned recordings, and `outputs{suffix}/` holds per-recording PSDs, `criticality.csv` and HTML plots. Results that already exist are skipped, so re-runs are cheap.

The pipeline has four steps:
1. **Preprocessing:** right-mastoid reference, 256 Hz, 1 Hz high-pass, `pop_rejcont` artefact rejection, automatic bad-channel detection, then Infomax ICA with automatic EOG/muscle component removal (1–5 components per recording, to mirror the paper's manual choice).
2. **Spectra:** PSDs from 1 s epochs.
3. **Criticality:** DFA and fEI per frequency band via `crosci`, on 180 s epochs.
4. **Plots:** group plots.

## Write-up

The write-up is built with LaTeX:

```bash
cd writeup && latexmk -pdf replication_writeup.tex
```

The compiled `writeup/replication_writeup.pdf` is tracked in git; all other PDFs are ignored.

The numbers and figures come from the scripts in `writeup/scripts/`, run after the two pipeline runs above. The `matched_*` scripts produce everything the write-up reports. `matched_sample.py` also needs the paper's per-subject CSVs from its Zenodo deposit ([doi 10.5281/zenodo.57911](https://doi.org/10.5281/zenodo.57911)). Their CSVs are in `writeup/results/matched/`, and `writeup/results/paper_included_subjects.csv` maps the paper's lab IDs to dataset subjects. The other scripts (`crit.py`, `figs.py`, `headline.py`, `matched_selection.py`) are from the superseded full-sample analysis.
