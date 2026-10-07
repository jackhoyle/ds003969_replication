"""Run the full pipeline for one run suffix, then write summary plots.

  1. process_inspect_loop.process_all  raw BIDS data -> processed{suffix}/*.fif
  2. compute_psd_epochs.main           processed .fif -> outputs{suffix}/psd_epoch_mean/*.npz
  3. pipeline_plots.make_all_plots     -> outputs{suffix}/plots/

Already processed files / existing PSDs are skipped, so re-running is cheap.

Usage (from the code/ directory):
    python run_pipeline.py --suffix _v3
    python run_pipeline.py --suffix _v2 --plots-only --channels Oz Pz Fz
"""
import argparse
import logging

import compute_psd_epochs
import pipeline_plots
from compute_psd_epochs import bids_root, psd_dir_for


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suffix', default='_v3',
                        help="run suffix for the processed/outputs folders, e.g. '_v3' (default: %(default)s)")
    parser.add_argument('--channels', nargs='+', default=['Oz', 'Pz', 'Fz', 'Cz'],
                        help='channels to plot group mean ± SE spectra for (default: %(default)s)')
    parser.add_argument('--skip-processing', action='store_true',
                        help='skip preprocessing/ICA (step 1)')
    parser.add_argument('--plots-only', action='store_true',
                        help='skip steps 1 and 2 and only regenerate plots')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if not (args.skip_processing or args.plots_only):
        # imported here so --plots-only doesn't need the processing dependencies
        from process_inspect_loop import process_all
        logging.info(f'Step 1/3: processing raw data -> processed{args.suffix}')
        process_all(run_suffix=args.suffix, auto_detect_bad_channels=True)

    if not args.plots_only:
        logging.info(f'Step 2/3: computing epoch PSDs -> {psd_dir_for(args.suffix)}')
        compute_psd_epochs.main(run_suffix=args.suffix)

    plots_dir = bids_root / f'outputs{args.suffix}' / 'plots'
    logging.info(f'Step 3/3: writing plots -> {plots_dir}')
    pipeline_plots.make_all_plots(bids_root, args.suffix, psd_dir_for(args.suffix), plots_dir, args.channels)
    logging.info('Done')


if __name__ == '__main__':
    main()
