# EEG meditation replication

A from-scratch MNE-Python replication of [Braboszcz et al. (2017)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5261734/), which amongst other things reports higher gamma (60–110 Hz) power in meditators than in controls. The project uses the public dataset [OpenNeuro ds003969](https://openneuro.org/datasets/ds003969/versions/1.0.0) and extends the analysis to criticality measures: long-range temporal correlations (DFA) and functional E/I balance (fEI).

**Findings so far** (details in [writeup/](writeup/)):
- **Alpha replicates:** the Vipassana group has more 7–11 Hz power than the other traditions, under every pipeline variant tried.
- **Gamma replicates in direction but not in strength:** meditators exceed controls, but the difference is clearly significant only without ICA. Automated muscle-component removal roughly halves it.
- **Criticality shows no meditator–control differences.** The one within-meditator effect looks like time-on-task.

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
python run_pipeline.py --suffix _v4                    # full run: preprocess + ICA -> PSDs -> DFA/fEI -> plots
python run_pipeline.py --suffix _noica --no-ica        # same without ICA
python run_pipeline.py --suffix _v4 --paper-subjects   # only the paper's 63 age/gender-matched subjects
python run_pipeline.py --help                          # all options (manual ICA review, skipping steps, ...)
```

Each run writes to two folders next to the data: `processed{suffix}/` holds the cleaned recordings, and `outputs{suffix}/` holds per-recording PSDs, `criticality.csv` and HTML plots. Results that already exist are skipped, so re-runs are cheap.

The pipeline has four steps:
1. **Preprocessing:** right-mastoid reference, 256 Hz, 1 Hz high-pass, `pop_rejcont` artefact rejection, automatic bad-channel detection, then Infomax ICA with automatic EOG/muscle component removal.
2. **Spectra:** PSDs from 1 s epochs.
3. **Criticality:** DFA and fEI per frequency band via `crosci`.
4. **Plots:** group plots.

## Write-up

The write-up is built with LaTeX:

```bash
cd writeup && latexmk -pdf replication_writeup.tex
```

The compiled `writeup/replication_writeup.pdf` is tracked in git; all other PDFs are ignored.
