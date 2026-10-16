"""Analyze and plot one MATLAB RF output using the locate_rf.ipynb helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

from RF.rflocate.workflow import analyze_rf_file
from RF.tc_generation import generate_rf_comparison, generate_rf_tc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="RF file written by MATLAB")
    parser.add_argument("--probe", required=True)
    parser.add_argument("--reference-2d", choices=("global", "bottom_row"), required=True,
                        help="2-D baseline mean: whole map or last spatial row; both use whole-map SD")
    parser.add_argument("--rf-type", choices=("excitatory", "inhibitory", "both"), default="both",
                        help="Use 'both' to save excitatory and inhibitory results together")
    parser.add_argument("--time-range", nargs=2, type=float, default=(0.0, 0.2), metavar=("START", "STOP"))
    parser.add_argument("--max-zero-bins", type=int, default=2)
    parser.add_argument("--sigma-2d", nargs=2, type=float, default=(1.8, -1.0),
                        metavar=("EXCITATORY", "INHIBITORY"),
                        help="Signed 2-D thresholds (default: 1.8 -1)")
    parser.add_argument("--sigma-1d", nargs=2, type=float, default=(1.0, -0.75),
                        metavar=("EXCITATORY", "INHIBITORY"),
                        help="Signed 1-D thresholds (default: 1 -0.75)")
    parser.add_argument("--cluster-forming-z-2d", type=float,
                        help="Legacy unsigned threshold override for both response directions")
    parser.add_argument("--cluster-forming-z-1d", type=float,
                        help="Legacy unsigned threshold override for both response directions")
    parser.add_argument("--drop-bins", type=int, required=True,
                        help="Choose explicitly: 0 keeps all threshold-passing bins; N removes components of N bins or fewer")
    parser.add_argument("--collapse-from-2d", action="store_true")
    parser.add_argument("--no-wrap-x", action="store_true")
    parser.add_argument("--unit-prefix", help="Comparison unit identity, e.g. mouse:date:probe")
    parser.add_argument("--comparison-output-dir", type=Path,
                        help="Write comparison CSVs here and native projections in its parent directory")
    parser.add_argument("--overwrite-comparison", action="store_true",
                        help="Replace existing comparison CSVs when generating them")
    parser.add_argument("--all-rf-rows", action="store_true",
                        help="Sum all spatial rows in comparison CSVs instead of only detected RF rows")
    parser.add_argument("--plot-output-dir", type=Path,
                        help="Figure directory (default: <source_stem>_figures beside the RF file)")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG/SVG figure exports")
    parser.add_argument("--plot-workers", type=int, default=8,
                        help="Parallel unit-figure processes (default: 8; 1 for serial export)")
    args = parser.parse_args()
    if args.plot_workers < 1:
        parser.error("--plot-workers must be at least 1")
    if (args.unit_prefix is None) != (args.comparison_output_dir is None):
        parser.error("--unit-prefix and --comparison-output-dir must be supplied together")
    analyses = analyze_rf_file(
        args.source, probe=args.probe, rf_type=args.rf_type, time_range_s=tuple(args.time_range),
        reference_2d=args.reference_2d,
        max_zero_bins=args.max_zero_bins,
        rf_sigmas={
            label: (args.sigma_2d[index], args.sigma_1d[index])
            for index, label in enumerate(("excitatory", "inhibitory"))
        },
        cluster_forming_z_2d=args.cluster_forming_z_2d,
        cluster_forming_z_1d=args.cluster_forming_z_1d, drop_bins=args.drop_bins,
        wrap_x=not args.no_wrap_x, collapse_from_2d=args.collapse_from_2d,
    )
    if args.rf_type != "both":
        analyses = {args.rf_type: analyses}
    rf_only = not args.all_rf_rows
    generated_projections = set()
    if not args.no_plots:
        matplotlib.use("Agg")
        from RF.rflocate.plotting import export_rf_units, plot_rf_population, rf_population_counts

        plot_output_dir = args.plot_output_dir or args.source.with_name(f"{args.source.stem}_figures")
        analyses_by_probe = {args.probe: analyses}
    for rf_type, result in analyses.items():
        qc = result["bin_qc"]
        print(
            f"Probe{args.probe} {rf_type}: kept {result['summed'].n_units}/{qc['unit_ids'].size} units; "
            f"2D RF {len(result['units_with_rf'])}; 1D RF {len(result['units_with_rf_1d'])}"
        )
        for name, path in result["output_paths"].items():
            print(f"{name}: {path}")
        if args.comparison_output_dir is not None:
            rf_suffix = "_rfonly" if rf_only else ""
            projection_name = f"{args.source.stem}_Probe{args.probe}"
            if rf_only and rf_type == "inhibitory":
                projection_name += "_inhibitory"
            projection_path = args.comparison_output_dir.parent / f"{projection_name}_1d{rf_suffix}.csv"
            comparison_path = args.comparison_output_dir / f"rf_{rf_type}_x_2d{rf_suffix}_Probe{args.probe}.csv"
            if projection_path in generated_projections:
                generate_rf_tc(
                    projection_path, comparison_path,
                    select_results=(result["output_paths"]["result_2d"],),
                    unit_prefix=args.unit_prefix, overwrite=args.overwrite_comparison,
                )
            else:
                generate_rf_comparison(
                    args.source, comparison_path, projection_path=projection_path,
                    detected_rf_path=result["output_paths"]["result_2d"],
                    time_range_s=tuple(args.time_range), rf_only=rf_only,
                    unit_prefix=args.unit_prefix, overwrite=args.overwrite_comparison,
                )
                generated_projections.add(projection_path)
            print(f"Comparison CSV: {comparison_path}")
        if not args.no_plots:
            counts = rf_population_counts(analyses_by_probe, rf_type=rf_type)
            # Each MATLAB invocation contains one probe, not a pooled session.
            plot_rf_population(
                {f"Probe{args.probe}": counts[f"Probe{args.probe}"]},
                rf_type=rf_type, output_dir=plot_output_dir, show=False,
            )
            exported_units = export_rf_units(
                analyses_by_probe, plot_output_dir, rf_type=rf_type,
                workers=args.plot_workers,
            )
            print(f"{rf_type} figures: {plot_output_dir} ({exported_units} units; PNG/SVG)")


if __name__ == "__main__":
    main()
