"""Analyze one MATLAB RF output with the same function as locate_rf.ipynb."""

from __future__ import annotations

import argparse
from pathlib import Path

from Utils.rflocate.workflow import analyze_rf_file
from Utils.tc_preparation import prepare_rf_comparison


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
    parser.add_argument("--unit-prefix", help="Comparison unit identity, e.g. mouse:date:probe")
    parser.add_argument("--comparison-output-dir", type=Path,
                        help="Write comparison CSVs here and native projections in its parent directory")
    parser.add_argument("--all-rf-rows", action="store_true",
                        help="Sum all spatial rows in comparison CSVs instead of only detected RF rows")
    args = parser.parse_args()
    if (args.unit_prefix is None) != (args.comparison_output_dir is None):
        parser.error("--unit-prefix and --comparison-output-dir must be supplied together")
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
        if args.comparison_output_dir is not None:
            rf_only = not args.all_rf_rows
            rf_suffix = "_rfonly" if rf_only else ""
            projection_name = f"{args.source.stem}_Probe{args.probe}"
            if rf_only and rf_type == "inhibitory":
                projection_name += "_inhibitory"
            projection_path = args.comparison_output_dir.parent / f"{projection_name}_1d{rf_suffix}.csv"
            comparison_path = args.comparison_output_dir / f"rf_{rf_type}_x_2d{rf_suffix}_Probe{args.probe}.csv"
            prepare_rf_comparison(
                args.source, comparison_path, projection_path=projection_path,
                detected_rf_path=result["output_paths"]["result_2d"],
                time_range_s=tuple(args.time_range), rf_only=rf_only,
                unit_prefix=args.unit_prefix,
            )
            print(f"Comparison CSV: {comparison_path}")


if __name__ == "__main__":
    main()
