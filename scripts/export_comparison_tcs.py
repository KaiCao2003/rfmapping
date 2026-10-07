"""Save one explicitly chosen HD or RF table for a comparison notebook.

Run on hhw9l84 with the rfmapping venv, from the repository root::

    python -m scripts.export_comparison_tcs hd source.tc chosen_hd.csv \
        --bins 30 --hd-class 3 --unit-prefix m14:260609:A
    python -m scripts.export_comparison_tcs rf native_y.csv chosen_rf.csv \
        --select-results detected_2d.npz detected_1d.npz --unit-prefix m14:260609:A

HD rates are rebinned without smoothing. RF input is an already prepared
projection CSV: its values and coordinates are copied without resampling.
Use the same explicit unit prefix for recordings of the same biological units.
"""

import argparse
from pathlib import Path

from Utils.tc_preparation import prepare_hd_tc, prepare_rf_tc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    hd = commands.add_parser("hd", help="Rebin a saved HD .tc file to unsmoothed rates")
    hd.add_argument("--bins", required=True, type=int, help="Number of HD bins")
    hd.add_argument("--hd-class", type=int, choices=range(4), help="Keep this HD class only")
    rf = commands.add_parser("rf", help="Copy an existing native RF projection CSV")
    rf.add_argument("--select-results", nargs="+", type=Path, default=(),
                    help="Keep IDs detected in any of these saved RF results")
    for command in (hd, rf):
        command.add_argument("source", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--unit-prefix", required=True,
                             help="Explicit identity shared by paired recordings, e.g. mouse:date:probe")
        command.add_argument("--overwrite", action="store_true",
                             help="Replace the specified output CSV")
    args = parser.parse_args(argv)
    inputs = [args.source, *getattr(args, "select_results", ())]
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("output must differ from every input path")
    exists = args.output.exists()
    if args.kind == "hd":
        prepare_hd_tc(args.source, args.output, bins=args.bins, hd_class=args.hd_class,
                      unit_prefix=args.unit_prefix, overwrite=args.overwrite)
    else:
        prepare_rf_tc(args.source, args.output, select_results=args.select_results,
                      unit_prefix=args.unit_prefix, overwrite=args.overwrite)
    action = "Skipped existing" if exists and not args.overwrite else "Wrote"
    print(f"{action} {args.output}", flush=True)


if __name__ == "__main__":
    main()
