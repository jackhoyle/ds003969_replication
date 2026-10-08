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
    python run_pipeline.py --suffix _v3
    python run_pipeline.py --suffix _v2 --plots-only --channels Oz Pz Fz
    python run_pipeline.py --suffix _v4 --skip-processing --n-jobs 8
    python run_pipeline.py --suffix _v5 --manual
"""
import argparse
import logging

import compute_psd_epochs
import pipeline_plots
from compute_psd_epochs import bids_root, psd_dir_for


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

    if not (args.skip_processing or args.plots_only):
        # imported here so --plots-only doesn't need the processing dependencies
        from process_inspect_loop import process_all
        logging.info(f'Step 1/4: processing raw data -> processed{args.suffix}')
        process_all(run_suffix=args.suffix, auto_detect_bad_channels=not args.manual, n_jobs=args.n_jobs)

    if not args.plots_only:
        logging.info(f'Step 2/4: computing epoch PSDs -> {psd_dir_for(args.suffix)}')
        compute_psd_epochs.main(run_suffix=args.suffix, n_jobs=args.n_jobs)

    if not (args.skip_criticality or args.plots_only):
        import compute_criticality  # needs crosci
        logging.info(f'Step 3/4: computing DFA/fEI -> {compute_criticality.criticality_dir_for(args.suffix)}')
        compute_criticality.main(run_suffix=args.suffix, n_jobs=args.n_jobs)

    plots_dir = bids_root / f'outputs{args.suffix}' / 'plots'
    logging.info(f'Step 4/4: writing plots -> {plots_dir}')
    pipeline_plots.make_all_plots(bids_root, args.suffix, psd_dir_for(args.suffix), plots_dir, args.channels)
    logging.info('Done')


if __name__ == '__main__':
    main()
