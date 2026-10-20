"""Stage and replace existing 2-D RF detections while preserving independent 1-D files.

The explicit inventory is frozen before processing. Preparation writes only to
the report directory; use --apply after validating those staged results.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from RF.rflocate import RFMapList, detect_rf, load_rf, load_rfmap, save_rf, save_rf_tc
from RF.rflocate.workflow import rf_analysis_paths, rf_bin_qc
from RF.tc_generation import calculate_rf_projection, calculate_rf_tc
from Utils.direction_comparison import save_tc
from scripts.rf_redetection_csvs import stage_existing_csvs


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_report(report, path):
    path.write_text(json.dumps(report, indent=2) + "\n")


def protect_files(source, rf_root):
    """Protect the raw container and all existing independent 1-D sidecars/lists."""
    paths = {source}
    for parent in (source.parent, source.parent.parent):
        paths.update(parent.glob("*_1d.npz"))
        paths.update(parent.glob("*units_with_rf_1d*.npy"))
    # Indexed NPZ sources are self-contained; legacy JSON can refer to sidecars.
    paths.update(source.parent.glob(source.stem + "*.npy"))
    paths = {p for p in paths if "units_with_rf" not in p.name or "_1d" in p.name}
    return {str(p): sha256(p) for p in sorted(paths) if p.is_file()}


def prepare_source(item, stage, settings):
    source = Path(item["path"])
    probe = item["probe"].removeprefix("Probe")
    counts = load_rfmap(source)
    presentations = item.get("presentation_counts_path")
    rates = (counts.to_firing_rate(presentation_counts=np.load(presentations, allow_pickle=False))
             if presentations else counts.to_firing_rate(reconstruct_presentations=True))
    summed_all = rates.mean_rate(0, .2, show_progress=False)
    qc = rf_bin_qc(summed_all, max_zero_bins=2)
    if not qc["keep"].any():
        raise ValueError(f"No units pass the existing bin QC: {source}")
    summed = RFMapList([m for m, keep in zip(summed_all, qc["keep"], strict=True) if keep], source)
    stage.mkdir(parents=True, exist_ok=True)
    row = {
        "source_path": str(source), "probe": probe, "rf_root": item["rf_root"],
        "source_kind": item["source_kind"], "session_path": item["session_path"],
        "raw_units": len(summed_all), "qc_units": len(summed), "outputs": {},
        "targets": {}, "replacements": [], "counts": {},
    }
    if presentations:
        row["presentation_counts_path"] = presentations
    plot_data = {
        "unit_ids": np.asarray(summed.unit_ids),
        "response": np.stack([m.to_2d_array() for m in summed]),
        "response_1d": np.stack([m.to_1d_array("x") for m in summed.sum_to_1d("x", missing_as_zero=True)]),
        "x_positions": summed[0].x_positions, "y_positions": summed[0].y_positions,
    }
    for polarity, options in settings.items():
        target = rf_analysis_paths(source, polarity)
        result = detect_rf(summed, reference="global", sigma=options["sigma"],
                           drop_bins=options["drop_bins"], is_shuffle=False,
                           exclude_zero_bins=False, wrap_x=True, show_progress=False)
        paths = {key: stage / p.name for key, p in target.items() if key in ("result_2d", "units_2d", "summary")}
        save_rf(result, paths["result_2d"])
        selected = [(probe, int(uid)) for uid, mask in zip(result.unit_ids, result.mask_2d, strict=True) if mask.any()]
        np.save(paths["units_2d"], np.asarray(selected, dtype=str).reshape(-1, 2))
        summary = json.loads(target["summary"].read_text()) if target["summary"].exists() else {}
        old_parameters = summary.get("parameters", {}).copy()
        parameters = dict(old_parameters, rf_type=polarity, reference_2d="global",
                          time_range_s=[0., .2], max_zero_bins=2, sigma_2d=options["sigma"],
                          cluster_forming_z_2d=abs(options["sigma"]),
                          drop_bins_2d=options["drop_bins"], wrap_x=True,
                          is_shuffle=False, exclude_zero_bins=False)
        # A shared legacy drop_bins value must not imply that independent 1-D changed.
        parameters["drop_bins_1d"] = old_parameters.get("drop_bins_1d", old_parameters.get("drop_bins"))
        parameters["drop_bins"] = old_parameters.get("drop_bins")
        summary.update(source_path=str(source), probe=probe, parameters=parameters,
                       bin_qc={key: values.tolist() for key, values in qc.items()},
                       kept_unit_ids=list(summed.unit_ids), units_with_rf=selected,
                       output_paths={key: str(p) for key, p in target.items()}, saved_dimensions=["2d"],
                       independent_1d_status="preserved" if target["result_1d"].exists() else "not_previously_saved")
        write_report(summary, paths["summary"])
        row["outputs"][polarity] = {key: str(p) for key, p in paths.items()}
        row["targets"][polarity] = {key: str(target[key]) for key in paths}
        row["replacements"].extend({"staged": str(p), "target": str(target[key])} for key, p in paths.items())
        row["counts"][polarity] = {"units": len(selected), "bins": int(result.mask_2d.sum()),
                                   "regions": int(result.center_2d.sum()), **options}
        for kind in ("mask", "center"):
            plot_data[f"{polarity}_{kind}_2d"] = getattr(result, f"{kind}_2d")
            plot_data[f"{polarity}_{kind}_1d"] = np.full((len(summed), summed[0].n_x), np.nan)
        if target["result_1d"].exists():
            saved_1d = load_rf(target["result_1d"])
            by_id = {int(uid): i for i, uid in enumerate(saved_1d.unit_ids)}
            for i, uid in enumerate(summed.unit_ids):
                if uid in by_id:
                    for kind in ("mask", "center"):
                        plot_data[f"{polarity}_{kind}_1d"][i] = getattr(saved_1d, f"{kind}_2d")[by_id[uid], 0]
        # Source-specific CSVs cannot collide when ON, OFF and rotation coexist.
        data_dir = Path(item["session_path"]) / "data"
        suffix = "_inhibitory" if polarity == "inhibitory" else ""
        projection_name = f"{source.stem}_Probe{probe}{suffix}_1d_rfonly.csv"
        projection = calculate_rf_projection(counts, result, time_range_s=(0., .2), rf_only=True)
        prefix = f"{item['mouse']}:{item['date']}:{probe}"
        comparison = calculate_rf_tc(projection, detected_results=[result], unit_prefix=prefix)
        native_stage = stage / projection_name
        save_rf_tc(projection, native_stage)
        comparison_name = f"{source.stem}_{polarity}_x_2d_rfonly_Probe{probe}.csv"
        comparison_stage = stage / comparison_name
        save_tc(comparison, comparison_stage)
        for staged, destination in ((native_stage, data_dir / projection_name),
                                    (comparison_stage, data_dir / "tc_comparison" / comparison_name)):
            row["replacements"].append({"staged": str(staged), "target": str(destination)})
        row.setdefault("csv_outputs", {})[polarity] = {
            "projection": str(native_stage), "comparison": str(comparison_stage),
            "selected_units": len(comparison), "unit_prefix": prefix,
        }
        # The unsuffixed historical comparison namespace denotes regular ON.
        if item["root_kind"] == "rfmapping" and item["source_kind"] == "regular" and "vertical_bar" not in source.stem:
            old_comparison = data_dir / "tc_comparison" / f"rf_{polarity}_x_2d_rfonly_Probe{probe}.csv"
            row["replacements"].append({"staged": str(comparison_stage), "target": str(old_comparison)})
            full_comparison = data_dir / "tc_comparison" / f"rf_{polarity}_x_2d_Probe{probe}.csv"
            if full_comparison.exists():
                full_projection = calculate_rf_projection(counts, result, time_range_s=(0., .2), rf_only=False)
                full_table = calculate_rf_tc(full_projection, detected_results=[result], unit_prefix=prefix)
                full_stage = stage / full_comparison.name
                save_tc(full_table, full_stage)
                row["replacements"].append({"staged": str(full_stage), "target": str(full_comparison)})
        row["replacements"].extend(stage_existing_csvs(item, counts, result, polarity, stage))
    row["replacements"] = list({r["target"]: r for r in row["replacements"]}.values())
    row["plot_data"] = str(stage / "plot_data.npz")
    np.savez_compressed(row["plot_data"], **plot_data)
    return row


def apply_report(report, report_path):
    if report["errors"] or len(report["sources"]) != report["expected_sources"]:
        raise ValueError("All inventoried sources must prepare successfully before replacement")
    for path, digest in report["protected_sha256"].items():
        if sha256(path) != digest:
            raise RuntimeError(f"Protected source or 1-D file changed: {path}")
    for row in report["sources"]:
        for replacement in row["replacements"]:
            source, target = map(Path, (replacement["staged"], replacement["target"]))
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".redetect-tmp")
            shutil.copyfile(source, temporary)
            temporary.replace(target)
            replacement["sha256"] = sha256(target)
        row["outputs"] = row["targets"]
    save_aggregate_lists(report)
    report["applied_utc"] = datetime.now(timezone.utc).isoformat()
    write_report(report, report_path)


def save_aggregate_lists(report):
    """Regular maps own shared aliases; rotation owns them only when alone."""
    groups = {}
    for row in report["sources"]:
        groups.setdefault(str(Path(row["source_path"]).parent.parent), []).append(row)
    report["aggregate_lists"] = []
    for root, rows in groups.items():
        regular = [row for row in rows if row["source_kind"] == "regular"]
        rows = regular or rows
        if len({row["probe"] for row in rows}) != len(rows):
            raise ValueError(f"Multiple source variants compete for shared unit lists: {root}")
        for polarity in ("excitatory", "inhibitory"):
            suffix = "_inhibitory" if polarity == "inhibitory" else ""
            combined = []
            for row in rows:
                values = np.load(row["outputs"][polarity]["units_2d"], allow_pickle=False)
                combined.extend(values.tolist())
                target = Path(row["source_path"]).parent / f"units_with_rf{suffix}.npy"
                np.save(target, values)
                report["aggregate_lists"].append(str(target))
            target = Path(root) / f"units_with_rf{suffix}.npy"
            np.save(target, np.asarray(combined, dtype=str).reshape(-1, 2))
            report["aggregate_lists"].append(str(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sigma-ext", type=float, required=True)
    parser.add_argument("--sigma-inh", type=float, required=True)
    parser.add_argument("--drop-ext", type=int, required=True)
    parser.add_argument("--drop-inh", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.apply:
        apply_report(json.loads(args.report.read_text()), args.report)
        return
    inventory = json.loads(args.inventory.read_text())
    selected = [s for s in inventory.get("active_saved_sources", inventory["sources"]) if s["has_saved_2d_detection"]
                and not s["in_backup"] and s["probe"] and "_bk" not in Path(s["path"]).parts[-2]
                and not Path(s["path"]).name.startswith("._")]
    settings = {"excitatory": {"sigma": args.sigma_ext, "drop_bins": args.drop_ext},
                "inhibitory": {"sigma": args.sigma_inh, "drop_bins": args.drop_inh}}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "settings": settings,
              "reference_2d": "global", "time_range_s": [0., .2], "sources": [],
              "protected_sha256": {}, "errors": [], "expected_sources": len(selected)}
    for item in selected:
        report["protected_sha256"].update(protect_files(Path(item["path"]), Path(item["rf_root"])))
        if item.get("presentation_counts_path"):
            path = item["presentation_counts_path"]
            report["protected_sha256"][path] = sha256(path)
    write_report(report, args.report)
    for index, item in enumerate(selected):
        try:
            row = prepare_source(item, args.report.parent / f"source_{index:03d}", settings)
            report["sources"].append(row)
            print(f"{index+1}/{len(selected)} prepared {item['path']}: {row['counts']}", flush=True)
        except Exception as exc:
            report["errors"].append({"source_path": item["path"], "error": str(exc)})
            print(f"ERROR {item['path']}: {exc}", flush=True)
        write_report(report, args.report)
    if report["errors"]:
        raise SystemExit(f"Preparation failed for {len(report['errors'])} sources; originals not replaced")


if __name__ == "__main__":
    main()
