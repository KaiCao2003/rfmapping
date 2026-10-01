"""Analyze one MATLAB RF output with the same function as locate_rf.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

from Utils.rf_analysis import analyze_rf_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="RF file written by MATLAB")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--rf-type", choices=("excitatory", "inhibitory", "both"), default="excitatory",
                        help="Use 'both' to save excitatory and inhibitory results together")
    parser.add_argument("--time-range", nargs=2, type=float, default=(0.0, 0.2), metavar=("START", "STOP"))
    parser.add_argument("--max-zero-bins", type=int, default=2)
    parser.add_argument("--cluster-forming-z-2d", type=float,
                        help="Default: 1.5 for inhibitory, 1.8 for excitatory")
    parser.add_argument("--cluster-forming-z-1d", type=float,
                        help="Default: 0.75 for inhibitory, 1.0 for excitatory")
    parser.add_argument("--drop-bins", type=int, default=2)
    parser.add_argument("--collapse-from-2d", action="store_true")
    parser.add_argument("--no-wrap-x", action="store_true")
    args = parser.parse_args()
    analyses = analyze_rf_file(
        args.source, probe=args.probe, rf_type=args.rf_type, time_range_s=tuple(args.time_range),
        max_zero_bins=args.max_zero_bins,
        cluster_forming_z_2d=args.cluster_forming_z_2d,
        cluster_forming_z_1d=args.cluster_forming_z_1d, drop_bins=args.drop_bins,
        wrap_x=not args.no_wrap_x, collapse_from_2d=args.collapse_from_2d,
    )
    if args.rf_type != "both":
        analyses = {args.rf_type: analyses}
    for rf_type, result in analyses.items():
        qc = result["bin_qc"]
        print(
            f"Probe{args.probe} {rf_type}: kept {result['summed'].n_units}/{qc['unit_ids'].size} units; "
            f"2D RF {len(result['units_with_rf'])}; 1D RF {len(result['units_with_rf_1d'])}"
        )
        for name, path in result["output_paths"].items():
            print(f"{name}: {path}")


if __name__ == "__main__":
    main()
