"""Export ten-panel PNG and SVG figures for every unit in a saved analysis."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from EBB.spatial_cell_plotting import plot_results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, help="Unit PNG/SVG directory; defaults to Probe*/all_units")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hd-tc", type=Path, help="Saved HD TC; defaults to this session's data/tuning_curves/Probe*/tuning_curves.tc")
    args = parser.parse_args(argv)
    plot_results(args.results, output=args.output, workers=args.workers, hd_tc_path=args.hd_tc)


if __name__ == "__main__":
    main()
