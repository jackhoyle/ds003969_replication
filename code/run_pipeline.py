"""Run the full pipeline for one run suffix, then write summary plots.

  1. process_inspect_loop.process_all  raw BIDS data -> processed{suffix}/*.fif
  2. compute_psd_epochs.main           processed .fif -> outputs{suffix}/psd_epoch_mean/*.npz
  3. compute_criticality.main          processed .fif -> outputs{suffix}/criticality/*.npz
                                                         + outputs{suffix}/criticality.csv (DFA, fEI)
  4. pipeline_plots.make_all_plots     -> outputs{suffix}/plots/

Already processed files / existing PSDs / existing DFA+fEI results are skipped, so re-running is cheap.
Steps 1-3 each process subject/task files in parallel over --n-jobs processes.
With --manual, step 1 runs serially and opens interactive plots to mark bad channels
and choose ICA components for each subject/task.

Usage (from the code/ directory):
    python run_pipeline.py --suffix _matchsample_v1 --interpolate-bads --paper-subjects
    python run_pipeline.py --suffix _matchsample_noica_v1 --interpolate-bads --paper-subjects --no-ica
"""

import argparse
import logging

import compute_psd_epochs
import pipeline_plots
from compute_psd_epochs import bids_root, paper_subjects, psd_dir_for


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suffix', default='',
                        help="run suffix for the processed/outputs folders, e.g. '_v3' (default: %(default)s)")
    parser.add_argument('--channels', nargs='+', default=['Oz', 'Pz', 'Fz', 'Cz'],
                        help='channels to plot group mean ± SE spectra for (default: %(default)s)')
    parser.add_argument('--skip-processing', action='store_true',
                        help='skip preprocessing/ICA (step 1)')
    parser.add_argument('--manual', action='store_true',
                        help='manually mark bad channels and choose ICA components in step 1 '
                             '(interactive, runs serially)')
    parser.add_argument('--no-ica', action='store_true',
                        help='skip ICA in step 1, so no components are removed')
    parser.add_argument('--max-bad-channels', type=int, default=20,
                        help='step 1: at most this many automatically marked bad channels per recording; over it, the '
                             'weakest suggestions are dropped, spectral-only ones first. 0 means no cap '
                             '(default: %(default)s)')
    parser.add_argument('--interpolate-bads', action='store_true',
                        help='step 2: interpolate bad channels before computing PSDs, writing to psd_epoch_mean_interp/ '
                             '(and plots to plots*_interp/) instead of leaving them NaN; needed for cluster statistics')
    parser.add_argument('--paper-subjects', action='store_true',
                        help="restrict every step to the paper's age/gender-matched subjects (paper_subjects.csv); "
                             'plots are written to plots_paper/ instead of plots/')
    parser.add_argument('--criticality-epoch-len', type=float, default=180.0,
                        help='step 3: epoch length in s for DFA/fEI (default 180); other lengths write to '
                             'criticality_{len}s/ so the default results are kept')
    parser.add_argument('--skip-criticality', action='store_true',
                        help='skip DFA/fEI (step 3), which is the slowest step after ICA')
    parser.add_argument('--plots-only', action='store_true',
                        help='skip steps 1-3 and only regenerate plots')
    parser.add_argument('--n-jobs', type=int, default=4,
                        help='parallel processes for preprocessing/ICA, PSDs and DFA/fEI (default: %(default)s)')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    subjects = paper_subjects() if args.paper_subjects else None
    if subjects is not None:
        logging.info(f"Restricting to the paper's {len(subjects)} matched subjects")

    if not (args.skip_processing or args.plots_only):
        # imported here so --plots-only doesn't need the processing dependencies
        from process_inspect_loop import process_all
        logging.info(f'Step 1/4: processing raw data -> processed{args.suffix}')
        process_all(run_suffix=args.suffix, auto_detect_bad_channels=not args.manual, n_jobs=args.n_jobs,
                    run_ica=not args.no_ica, only_subjects=subjects, max_bad_channels=args.max_bad_channels)

    psd_dir = psd_dir_for(args.suffix, interpolated=args.interpolate_bads)
    if not args.plots_only:
        logging.info(f'Step 2/4: computing epoch PSDs -> {psd_dir}')
        compute_psd_epochs.main(run_suffix=args.suffix, n_jobs=args.n_jobs, subjects=subjects,
                                interpolate_bads=args.interpolate_bads)

    if not (args.skip_criticality or args.plots_only):
        import compute_criticality  # needs crosci
        out = compute_criticality.criticality_dir_for(args.suffix, args.criticality_epoch_len)
        logging.info(f'Step 3/4: computing DFA/fEI -> {out}')
        compute_criticality.main(run_suffix=args.suffix, n_jobs=args.n_jobs, subjects=subjects,
                                 epoch_len=args.criticality_epoch_len)

    plots_dir = (bids_root / f'outputs{args.suffix}' /
                 (('plots_paper' if args.paper_subjects else 'plots') + ('_interp' if args.interpolate_bads else '')))
    logging.info(f'Step 4/4: writing plots -> {plots_dir}')
    pipeline_plots.make_all_plots(bids_root, args.suffix, psd_dir, plots_dir, args.channels,
                                  only_subjects=subjects)
    logging.info('Done')


if __name__ == '__main__':
    main()
