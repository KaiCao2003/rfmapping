"""Redetect existing regular RF maps and export the locate_rf notebook figures.

Run from the repository root on hhw9l84 with the rfmapping environment::

    python -m scripts.redetect_rf_sessions /path/to/mouse/date/session --reference-2d global --drop-bins 2 --plot-workers 8

Only active Probe directories under good/-100_400_1ms are read. Raw RF maps
are retained; current detections, comparison CSVs, and figures are replaced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import matplotlib
from scipy.ndimage import gaussian_filter

matplotlib.use("Agg")

from RF.rflocate.workflow import analyze_rf_file, save_rf_analysis, save_rf_unit_lists
from RF.rflocate.plotting import export_rf_units, plot_rf_population, rf_population_counts
from RF.tc_generation import generate_rf_comparison


def file_record(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def comparison_paths(session, source, probe, rf_type):
    suffix = "_inhibitory" if rf_type == "inhibitory" else ""
    projection = session / "data" / f"{source.stem}_Probe{probe}{suffix}_1d_rfonly.csv"
    comparison = session / "data/tc_comparison" / f"rf_{rf_type}_x_2d_rfonly_Probe{probe}.csv"
    return projection, comparison


def output_paths(session, analyses_by_probe, *, plots):
    rf_dir = session / "data/rfmapping/good/-100_400_1ms"
    probes = list(analyses_by_probe)
    list_dir = rf_dir / f"Probe{probes[0]}" if len(probes) == 1 else rf_dir
    paths = []
    for rf_type in ("excitatory", "inhibitory"):
        suffix = "_inhibitory" if rf_type == "inhibitory" else ""
        paths.extend(list_dir / f"{name}{suffix}.npy" for name in ("units_with_rf", "units_with_rf_1d"))
        figure_dir = session / "data/rfmapping" / ("inhibitory/rfmap" if suffix else "rfmap")
        for probe, analyses in analyses_by_probe.items():
            result = analyses[rf_type]
            paths.extend(result["output_paths"].values())
            paths.extend(comparison_paths(session, result["source_path"], probe, rf_type))
            if plots:
                paths.extend(
                    figure_dir / "units" / probe / f"{unit}{extension}"
                    for unit in result["summed"].unit_ids for extension in (".png", ".svg")
                )
        if plots:
            paths.extend(
                figure_dir / panel / f"rf_summary{extension}"
                for panel in [*(f"Probe{probe}" for probe in probes), "all", "all_smoothed"]
                for extension in (".png", ".svg")
            )
    return paths, list_dir


def redetect_session(session, *, reference_2d, drop_bins, plot_workers, plots, backup_dir):
    started = time.monotonic()
    rf_dir = session / "data/rfmapping/good/-100_400_1ms"
    sources = sorted(rf_dir.glob(f"Probe?/regular_unitsSpikeCounts_{session.name}.rfmap"))
    if not sources:
        raise FileNotFoundError(f"No regular RF maps in {rf_dir}")
    before = [file_record(source) for source in sources]
    analyses_by_probe = {}
    for source in sources:
        probe = source.parent.name.removeprefix("Probe")
        analyses_by_probe[probe] = analyze_rf_file(
            source, probe=probe, drop_bins=drop_bins, rf_type="both", save_results=False,
            reference_2d=reference_2d,
        )
        for rf_type, result in analyses_by_probe[probe].items():
            print(
                f"{session.parent.parent.name}/{session.name} Probe{probe} {rf_type}: "
                f"QC {result['summed'].n_units}/{result['bin_qc']['unit_ids'].size}; "
                f"2D {len(result['units_with_rf'])}; 1D {len(result['units_with_rf_1d'])}",
                flush=True,
            )
    outputs, list_dir = output_paths(session, analyses_by_probe, plots=plots)
    preserved = []
    if backup_dir is not None:
        session_backup = backup_dir / session.parent.parent.name / session.parent.name / session.name
        for path in outputs:
            if path.exists():
                target = session_backup / path.relative_to(session)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                preserved.append({**file_record(path), "backup": str(target)})
        session_backup.mkdir(parents=True, exist_ok=True)
        (session_backup / "manifest.json").write_text(json.dumps(preserved, indent=2) + "\n")
    counts = {}
    for probe, analyses in analyses_by_probe.items():
        counts[probe] = {}
        for rf_type, result in analyses.items():
            save_rf_analysis(result)
            projection, comparison = comparison_paths(session, result["source_path"], probe, rf_type)
            generate_rf_comparison(
                result["source_path"], comparison, projection_path=projection,
                detected_rf_path=result["output_paths"]["result_2d"],
                time_range_s=tuple(result["parameters"]["time_range_s"]), rf_only=True,
                unit_prefix=f"{session.parent.parent.name}:{session.parent.name}:{probe}",
                overwrite=True,
            )
            counts[probe][rf_type] = {
                "total_units": int(result["bin_qc"]["unit_ids"].size),
                "qc_units": result["summed"].n_units,
                "rf_2d": len(result["units_with_rf"]),
                "rf_1d": len(result["units_with_rf_1d"]),
                "parameters": result["parameters"],
            }
    save_rf_unit_lists(analyses_by_probe, list_dir)
    if plots:
        for rf_type in ("excitatory", "inhibitory"):
            population = rf_population_counts(analyses_by_probe, rf_type=rf_type)
            population["all_smoothed"] = {
                key: gaussian_filter(values.astype(float), sigma=1.0, mode="nearest")
                for key, values in population["all"].items()
            }
            plot_rf_population(population, rf_type=rf_type, output_dir=session / "data/rfmapping", show=False)
            export_rf_units(
                analyses_by_probe, session / "data/rfmapping", rf_type=rf_type,
                rf_only=False, workers=plot_workers,
            )
    after = [file_record(source) for source in sources]
    if before != after:
        raise RuntimeError(f"RF source changed during detection: {session}")
    return {
        "session": str(session), "sources": before, "sources_unchanged": True,
        "counts": counts, "preserved_outputs": preserved,
        "outputs": [file_record(path) for path in outputs],
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+", type=Path)
    parser.add_argument("--reference-2d", choices=("global", "bottom_row"), required=True,
                        help="2-D baseline mean: whole map or last spatial row; both use whole-map SD")
    parser.add_argument("--drop-bins", type=int, required=True,
                        help="0 keeps all threshold-passing bins; N removes components of N bins or fewer")
    parser.add_argument("--plot-workers", type=int, default=8)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--backup-dir", type=Path, help="Preserve replaced outputs here before writing")
    parser.add_argument("--report", type=Path, help="JSON output manifest; default: .codex_tmp/rf-redetection-TIMESTAMP.json")
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report or Path(".codex_tmp") / f"rf-redetection-{stamp}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": stamp, "reference_2d": args.reference_2d, "drop_bins": args.drop_bins,
              "plot_workers": args.plot_workers, "sessions": []}
    for session in args.sessions:
        report["sessions"].append(redetect_session(
            session.resolve(), reference_2d=args.reference_2d, drop_bins=args.drop_bins,
            plot_workers=args.plot_workers, plots=not args.no_plots,
            backup_dir=args.backup_dir.resolve() if args.backup_dir is not None else None,
        ))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Completed {session}; report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
