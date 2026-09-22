"""Kilosort wrapper.
    python -m pipeline.sort <probe_path> [options]
    sort <probe_path> [options]
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Kilosort spike sorting (subprocess-safe)")
    parser.add_argument('probe_path', type=str, help='Path to concatenated recording directory')
    parser.add_argument('--per-shank', action='store_true', help='Sort each shank independently')
    parser.add_argument('--verbose', action='store_true', help='Enable kilosort console output')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results')
    parser.add_argument('--openblas-threads', type=int, default=1,
                        help='OPENBLAS/MKL thread count (default: 1)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Write log output to file instead of stderr')
    args = parser.parse_args()

    # Set threading env vars BEFORE importing numpy/torch/kilosort
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.openblas_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.openblas_threads)

    import matplotlib
    matplotlib.use('Agg')

    from pathlib import Path

    import numpy as np
    import spikeinterface as si
    from loguru import logger

    from .probe import probe_to_kilosort
    from .utils import log_rec

    try:
        from kilosort import run_kilosort
    except ImportError:
        print("ERROR: kilosort is not installed. Install with: uv sync", file=sys.stderr)
        sys.exit(1)

    # Logging setup
    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        "<cyan>kilosort</cyan> | "
        "<level>{message}</level>"
    )
    sink = args.log_file if args.log_file else sys.stderr
    logger.add(sink, format=fmt, level="DEBUG", colorize=(args.log_file is None))

    probe_path = Path(args.probe_path)
    results_dir = probe_path.parent / 'kilosort'
    results_dir.mkdir(parents=True, exist_ok=True)

    if (results_dir / 'spike_times.npy').exists() and not args.overwrite:
        logger.info(f"Results already exist, skipping: {results_dir}")
        sys.exit(0)

    rec = si.load(probe_path)
    logger.info(f"Loaded concatenated recording: {probe_path}")
    log_rec(rec)

    settings = {
        'n_chan_bin': rec.get_num_channels(),
        'fs': rec.get_sampling_frequency(),
    }
    ks_probe = probe_to_kilosort(rec.get_probe())
    filename = probe_path / 'traces_cached_seg0.raw'

    logger.info(f"Saving Kilosort results to: {results_dir}")

    if args.per_shank:
        for shank_id in np.unique(ks_probe['kcoords']):
            logger.info(f"Sorting shank {int(shank_id)}")
            run_kilosort(
                settings=settings,
                probe=ks_probe,
                filename=filename,
                results_dir=results_dir,
                device='cuda',
                verbose_console=args.verbose,
                shank_idx=int(shank_id),
            )
    else:
        run_kilosort(
            settings=settings,
            probe=ks_probe,
            filename=filename,
            results_dir=results_dir,
            device='cuda',
            verbose_console=args.verbose,
        )

    logger.info("Sorting complete")


if __name__ == '__main__':
    main()
