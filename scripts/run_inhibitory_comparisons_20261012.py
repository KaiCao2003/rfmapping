"""Export the six requested inhibitory RF / HD comparisons on hhw9l84.

Saved classifications and detections are inputs. All regenerated projections,
tables, plots, and provenance stay below the task output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from Utils.direction_comparison import align_profiles, concat, minmax_profiles, normalize_tc, paired_peak_angles, profile_table, resample_profiles, save_tc, tc_loader, tcRange
from RF.profiles import rf_pick
from Utils.plotting import LIGHT_PLOT_STYLE, plot_direction_comparison
from RF.rflocate import load_rf, rf_result_path
from Utils.statistic_utils import compare_direction_angles
from RF.tc_generation import generate_rf_comparison
from Utils.statistic_utils import rayleigh_test


ROOT = Path("/mnt/senzailab/Kai/#Recording")
OUT = Path("output/session_comparisons_20261012/inhibitory")
SESSIONS = [
    ("m14", "260609", 3, 1, "A"),
    ("m15", "260630", 3, 1, "AB"),
    ("m19", "260827", 2, 9, "A"),
    ("m20", "260921", 2, 9, "A"),
    ("m21", "261006", 3, 9, "A"),
    ("m22", "261008", 2, 9, "A"),
]


def file_record(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def saved_hd(source, prefix):
    """Rebin saved counts/occupancy, retaining the saved class exactly."""
    data = json.loads(source.read_text())
    ids = np.asarray(data["unit_id"])
    selected = np.asarray(data["unit_data"]["hd_class"]) == 3
    counts = np.asarray(data["spike_counts"], dtype=float)
    occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
    factor = len(occupancy) // 30
    counts = counts.reshape(len(ids), 30, factor).sum(axis=2)
    occupancy = occupancy.reshape(30, factor).sum(axis=1)
    rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan), where=occupancy > 0)
    edges = np.asarray(data["angle_bin_edges_deg"])[::factor]
    table = profile_table(rates[selected], [f"{prefix}:{unit}" for unit in ids[selected]],
                          (edges[:-1] + edges[1:]) / 2)
    table.attrs = {"range": tcRange(False), "response_units": "Hz", "label": "HD"}
    return table, data["metadata"]


def compare_cache(table, path):
    if not path.exists():
        return {"exists": False}
    cached = tc_loader(path)
    same_ids = list(cached.index) == list(table.index)
    same_angles = list(cached.columns) == list(table.columns)
    same = same_ids and same_angles and np.allclose(
        table.to_numpy(), cached.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True,
    )
    return {**file_record(path), "exists": True, "same_unit_order": same_ids,
            "same_angles": same_angles, "matches_task_table": bool(same),
            "cached_units": len(cached), "task_units": len(table)}


def load_pair(mouse, date, rfid, hdsid, probe, *, generate_rf_inputs=False, overwrite_rf_inputs=False):
    key = f"{mouse}_{date}_RF{rfid}_HD{hdsid}_{probe}"
    directory = OUT / "inputs" / key
    directory.mkdir(parents=True, exist_ok=True)
    prefix = f"{mouse}:{date}:{probe}"
    rfdir = ROOT / mouse / date / f"{date}_{rfid}" / "data"
    hddir = ROOT / mouse / date / f"{date}_{hdsid}" / "data"
    hdsource = hddir / f"tuning_curves/Probe{probe}/tuning_curves.tc"
    rfsource = rfdir / f"rfmapping/good/-100_400_1ms/Probe{probe}/regular_unitsSpikeCounts_{date}_{rfid}.rfmap"
    detection = rf_result_path(rfsource, rf_type="inhibitory")
    before = [file_record(path) for path in (hdsource, rfsource, detection)]
    hd, metadata = saved_hd(hdsource, prefix)
    save_tc(hd, directory / "hd_class3.csv")
    hd_cache = compare_cache(hd, hddir / f"tc_comparison/hd_class3_Probe{probe}.csv")
    rf_output = directory / "rf_inhibitory.csv"
    projection_output = directory / "rf_native_projection.csv"
    if generate_rf_inputs and (overwrite_rf_inputs or not (rf_output.exists() and projection_output.exists())):
        generate_rf_comparison(
            rfsource, rf_output, projection_path=projection_output, detected_rf_path=detection,
            time_range_s=(0.0, 0.2), rf_only=True, unit_prefix=prefix,
            overwrite=overwrite_rf_inputs,
        )
    elif not rf_output.exists():
        raise FileNotFoundError(
            f"Missing saved RF comparison input {rf_output}; rerun with --generate-rf-inputs explicitly."
        )
    rf_range = [-150, 150] if mouse == "m14" else tcRange(True)
    rf = tc_loader(rf_output, range=rf_range,
                   label="Inhibitory RF", response_units="spike_count")
    rf_cache = compare_cache(rf, rfdir / f"tc_comparison/rf_inhibitory_x_2d_rfonly_Probe{probe}.csv")
    n_rf_before = len(rf)
    rf = rf_pick(rf, max_zero_bins=2)
    units = hd.index.intersection(rf.index)
    hd_pair, rf_pair = normalize_tc(hd.loc[units]), normalize_tc(rf.loc[units])
    with np.load(detection, allow_pickle=False) as archive:
        detection_manifest = json.loads(str(archive["manifest_json"].item()))
    after = [file_record(path) for path in (hdsource, rfsource, detection)]
    assert before == after
    info = {
        "key": key, "mouse": mouse, "date": date, "probe": probe,
        "rf_session": f"{date}_{rfid}", "hd_session": f"{date}_{hdsid}",
        "hd_class3_units": len(hd), "rf_2d_units": n_rf_before,
        "rf_after_zero_filter": len(rf), "shared_units": len(units),
        "sources": before, "sources_unchanged": True, "hd_metadata": metadata,
        "rf_detection_manifest": detection_manifest,
        "existing_hd_cache": hd_cache, "existing_rf_cache": rf_cache,
        "native_rf_centers_deg": list(rf.columns),
    }
    print(key, "HD", len(hd), "RF", len(rf), "shared", len(units),
          "cache match", hd_cache.get("matches_task_table"), rf_cache.get("matches_task_table"), flush=True)
    return (hd_pair, rf_pair), info


def save_figure(fig, directory, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=260, facecolor="white", transparent=False)
    plt.close(fig)
    return str(directory / f"{name}.png")


def transformed_tables(parts, peaks, mode, reference):
    ref_col = "reference_peak_deg" if reference == "hd" else "matched_peak_deg"
    order = peaks.sort_values(ref_col, kind="stable").index
    hd_tables, rf_tables = [], []
    for hd, rf in parts:
        offsets = peaks.loc[hd.index, ref_col].to_numpy()
        if mode == "native":
            hd_out = hd
            rf_out = (resample_profiles(rf, range=tcRange(True), bins=30, fill_value=np.nan)
                      if len(parts) > 1 else rf)
        else:
            hd_offset = -offsets if reference == "hd" or mode == "aligned" else offsets
            rf_offset = -offsets if reference == "rf" or mode == "aligned" else offsets
            hd_out = align_profiles(hd, hd.index, hd_offset, bins=30, fill_value=np.nan)
            rf_out = align_profiles(rf, rf.index, rf_offset, bins=30, fill_value=np.nan)
        hd_tables.append(hd_out)
        rf_tables.append(rf_out)
    return concat(*hd_tables).loc[order], concat(*rf_tables).loc[order]


def plot_heatmaps(hd, rf, mode, reference, *, all_labels=False):
    n = len(hd)
    height = max(5.4, 0.13 * n + 1.8) if all_labels else 5.4
    fig, axes = plt.subplots(1, 2, figsize=(11.4, height), layout="constrained", sharey=True)
    for ax, table, label in zip(axes, (hd, rf), ("HD", "Inhibitory RF"), strict=True):
        columns = np.asarray(table.columns, dtype=float)
        columns = (columns + 180) % 360 - 180
        col_order = np.argsort(columns)
        centers = columns[col_order]
        values = minmax_profiles(table).to_numpy()[:, col_order]
        if mode != "native" and centers[0] == -180:
            # The seam bin spans both ends of the circular display.
            centers = np.r_[centers, 180.0]
            values = np.c_[values, values[:, 0]]
        spacing = centers[1] - centers[0]
        extent = (centers[0] - spacing / 2, centers[-1] + spacing / 2, n - 0.5, -0.5)
        image = ax.imshow(values, cmap="Greys" if label == "Inhibitory RF" else "viridis", vmin=0, vmax=1,
                          aspect="auto", interpolation="nearest", extent=extent)
        ticks = (-180, -90, 0, 90, 180)
        ax.set(xlim=(-180, 180), xticks=ticks, title=label)
        if mode == "native":
            xlabel = f"{label} direction (°)"
            if label == "HD":
                ax.set_xticklabels((180, 270, 0, 90, 180))
        elif mode == "aligned":
            xlabel = f"Angle − {'HD peak' if reference == 'hd' else 'RF minimum'} (°)"
        else:
            is_reference = (label == "HD") == (reference == "hd")
            operator = "−" if is_reference else "+"
            xlabel = f"Angle {operator} {'HD peak' if reference == 'hd' else 'RF minimum'} (°)"
        ax.set_xlabel(xlabel)
        if all_labels or n <= 35:
            ax.set_yticks(np.arange(n), labels=hd.index)
            ax.tick_params(axis="y", labelsize=7 if all_labels else 8, length=0)
        else:
            indices = np.unique(np.linspace(0, n - 1, 6).astype(int))
            ax.set_yticks(indices, labels=indices + 1)
        fig.colorbar(image, ax=ax, pad=0.025, fraction=0.045, label="Response (unit min–max)")
    axes[0].set_ylabel("Unit ID" if all_labels or n <= 35 else "Ordered units")
    order_name = "HD peak" if reference == "hd" else "RF minimum"
    fig.suptitle(f"{mode.capitalize()} · sorted by {order_name} · n = {n}", fontsize=16)
    return fig


def rerender_heatmaps():
    """Rerender display scaling from saved curves without changing statistics."""
    backup = Path(".codex_tmp/session_comparisons_20261012/inhibitory_minmax_before")
    protected = {
        str(path): file_record(path)
        for path in OUT.rglob("*") if path.is_file()
        and path.name not in ("heatmap_palette_rerender.json", "heatmap_minmax_rerender.json")
        and not any(path.name.startswith(f"{mode}_sorted_by_") and path.suffix in (".png", ".pdf", ".svg")
                    for mode in ("native", "aligned", "sum"))
    }
    previous, updated = [], []
    for summary_path in sorted(OUT.glob("*/summary.json")):
        directory = summary_path.parent
        for mode in ("native", "aligned", "sum"):
            for reference in ("hd", "rf"):
                basename = f"{mode}_sorted_by_{reference}"
                hd = tc_loader(directory / f"{basename}_hd.csv")
                rf = tc_loader(directory / f"{basename}_rf.csv")
                for all_labels in ([False, True] if mode == "native" else [False]):
                    name = basename + ("_all_unit_labels" if all_labels else "")
                    for suffix in ("png", "pdf", "svg"):
                        path = directory / f"{name}.{suffix}"
                        if path.exists():
                            target = backup / path.relative_to(OUT)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if not target.exists():
                                shutil.copy2(path, target)
                            previous.append(file_record(target))
                    save_figure(plot_heatmaps(hd, rf, mode, reference, all_labels=all_labels), directory, name)
                    updated.extend(file_record(directory / f"{name}.{suffix}") for suffix in ("png", "pdf", "svg"))
        print("Updated per-unit min–max display", directory.name, flush=True)
    assert protected == {path: file_record(Path(path)) for path in protected}
    receipt = dict(code=file_record(Path(__file__)), input_tables_and_statistics_unchanged=True,
                   statistics_recomputed=False, rf_colormap="Greys: row minimum white, row maximum black",
                   display_normalization="Both HD and RF: (row - finite min) / (finite max - finite min); constant rows 0; NaNs preserved",
                   hd_colormap="viridis", previous_figures=previous, updated_figures=updated,
                   protected_files=list(protected.values()), backup_directory=str(backup))
    (OUT / "heatmap_minmax_rerender.json").write_text(json.dumps(receipt, indent=2) + "\n")


def export_group(name, label, parts, pair_info, *, sensitivity=False):
    directory = OUT / name
    directory.mkdir(parents=True, exist_ok=True)
    # Pool native per-session directions; display interpolation never picks minima.
    peak_tables = [paired_peak_angles(hd, rf, matched_min=True) for hd, rf in parts]
    peaks = pd.concat(peak_tables, verify_integrity=True).sort_values("reference_peak_deg", kind="stable")
    peaks.to_csv(directory / "paired_native_directions.csv", index_label="unit_id")
    figures = {}
    for mode in ("native", "aligned", "sum"):
        for reference in ("hd", "rf"):
            hd, rf = transformed_tables(parts, peaks, mode, reference)
            basename = f"{mode}_sorted_by_{reference}"
            save_tc(hd, directory / f"{basename}_hd.csv")
            save_tc(rf, directory / f"{basename}_rf.csv")
            figures[basename] = save_figure(plot_heatmaps(hd, rf, mode, reference), directory, basename)
            if mode == "native":
                save_figure(plot_heatmaps(hd, rf, mode, reference, all_labels=True),
                            directory, f"{basename}_all_unit_labels")
    hd_angles = peaks.reference_peak_deg.to_numpy()
    rf_angles = peaks.matched_peak_deg.to_numpy()
    statistics = compare_direction_angles(hd_angles, rf_angles, n_permutations=10_000, random_seed=1)
    plots = plot_direction_comparison(
        hd_angles, rf_angles, statistics, reference_label="HD", matched_label="Inhibitory RF",
        reference_ticklabels=(180, 270, 0, 90, 180), matched_ticklabels=(-180, -90, 0, 90, 180),
        density_bin_count=15, show=False,
    )
    for plot_name, (fig, ax) in zip(("scatter", "mantel"), plots, strict=True):
        ax.set_title("HD peak vs. inhibitory RF minimum" if plot_name == "scatter" else "Pairwise angular distances")
        ax.set_ylabel("Inhibitory RF minimum direction (°)" if plot_name == "scatter" else "RF minimum pair distance (°)")
        ax.set_xlabel("HD peak direction (°)" if plot_name == "scatter" else "HD peak pair distance (°)")
        fig.tight_layout()
        figures[plot_name] = save_figure(fig, directory, plot_name)
    counts, edges = np.histogram(np.deg2rad((hd_angles + rf_angles) % 360), bins=15, range=(0, 2 * np.pi))
    score, pvalue = rayleigh_test(counts, np.ones_like(counts), np.rad2deg((edges[:-1] + edges[1:]) / 2))
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"}, layout="constrained")
    ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges), color="#3475A8", edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids((0, 90, 180, 270))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"HD peak + RF minimum\nn = {len(peaks)} · Rayleigh p = {pvalue:.4g}", pad=20)
    ax.set_xlabel("Units (radial axis)")
    figures["sum_polar"] = save_figure(fig, directory, "sum_polar")
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ax.hist(peaks.peak_sum_deg, bins=np.linspace(-180, 180, 16), color="#3475A8", edgecolor="white")
    ax.set(xlim=(-180, 180), xticks=(-180, -90, 0, 90, 180), xlabel="HD peak + RF minimum (°)",
           ylabel="Units", title=f"Angle sums · n = {len(peaks)}\nRayleigh score = {score:.3g}, p = {pvalue:.4g}")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    figures["sum_histogram"] = save_figure(fig, directory, "sum_histogram")
    entry = {"name": name, "label": label, "paired_units": len(peaks), "input_pairs": pair_info,
             "statistics": statistics, "sum_histogram": {"counts": counts.tolist(),
             "edges_rad": edges.tolist(), "rayleigh_score": score, "rayleigh_p": pvalue},
             "figures": figures, "sensitivity_only": sensitivity,
             "native_directions_csv": str(directory / "paired_native_directions.csv")}
    (directory / "summary.json").write_text(json.dumps(entry, indent=2) + "\n")
    print("DONE", name, "n", len(peaks), "circular p", statistics["same_unit"]["permutation_p"],
          "Mantel p", statistics["pairwise"]["permutation_p"], flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatmaps-only", action="store_true", help="Rerender saved ordered heatmaps without scientific calculations")
    parser.add_argument("--generate-rf-inputs", action="store_true",
                        help="Explicitly generate RF projection and comparison inputs")
    parser.add_argument("--overwrite-rf-inputs", action="store_true",
                        help="Replace RF inputs when used with --generate-rf-inputs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(LIGHT_PLOT_STYLE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.titlesize": 15, "axes.labelsize": 13,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    if args.heatmaps_only:
        rerender_heatmaps()
        return
    manifest = {
        "parameters": {"hd_class": 3, "hd_classification": "saved unit_data.hd_class; no reclassification",
            "rf_detection": "saved inhibitory 2D mask", "response_window_s": [0, 0.2],
            "rf_projection": "raw counts summed across entire y rows touched by saved RF",
            "native_zero_bin_limit": 2, "normalization": "Saved curves: response divided by finite row maximum; display: per-unit min–max for HD and RF; never inverted",
            "directions": "HD argmax; inhibitory RF argmin on native selected curves",
            "comparison_bins": 30, "statistics_bins": 15, "added_smoothing": 0,
            "n_permutations": 10000, "random_seed": 1,
            "pooled_direction_rule": "concatenate exact per-session native directions; no padded zero minima",
            "missing_angles": "NaN outside m14 measured -150..150 field; retained in heatmaps",
            "m22_hd_pairing": "HD9 Motive main, consistent with established HD9 pairs; HD11 sensitivity",
            "mantel": "test unique off-diagonal pairs; density shows ordered pairs including diagonal",
            "inference": "unit-level exploratory p-values; pooled permutations not clustered by mouse"},
        "code": [file_record(Path(__file__)), file_record(Path("Utils/direction_comparison.py")),
                 file_record(Path("Utils/tc_preparation.py")), file_record(Path("Utils/statistic_utils.py"))],
        "input_pairs": [], "groups": [],
    }
    pooled, pooled_keys = [], []
    for mouse, date, rfid, hdsid, probes in SESSIONS:
        parts, keys = [], []
        for probe in probes:
            part, info = load_pair(mouse, date, rfid, hdsid, probe,
                                   generate_rf_inputs=args.generate_rf_inputs,
                                   overwrite_rf_inputs=args.overwrite_rf_inputs)
            parts.append(part)
            keys.append(info["key"])
            manifest["input_pairs"].append(info)
        name = f"{mouse}_{date}"
        manifest["groups"].append(export_group(name, f"{mouse} {date} RF{rfid} / HD{hdsid}", parts, keys))
        if len(probes) > 1:
            for probe, part, key in zip(probes, parts, keys, strict=True):
                manifest["groups"].append(export_group(f"{name}_{probe}", f"{mouse} {date} Probe{probe}", [part], [key]))
        pooled.extend(parts)
        pooled_keys.extend(keys)
        (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["groups"].append(export_group("pooled", "All six RF sessions", pooled, pooled_keys))
    part, info = load_pair("m22", "261008", 2, 11, "A",
                           generate_rf_inputs=args.generate_rf_inputs,
                           overwrite_rf_inputs=args.overwrite_rf_inputs)
    manifest["input_pairs"].append(info)
    manifest["groups"].append(export_group("m22_261008_HD11_sensitivity", "m22 RF2 / HD11 sensitivity",
                                            [part], [info["key"]], sensitivity=True))
    for entry in manifest["input_pairs"]:
        for source in entry["sources"]:
            assert file_record(Path(source["path"])) == source
    manifest["all_sources_unchanged"] = True
    manifest["all_existing_caches_unchanged"] = True
    for entry in manifest["input_pairs"]:
        for cache in (entry["existing_hd_cache"], entry["existing_rf_cache"]):
            if cache["exists"]:
                current = file_record(Path(cache["path"]))
                assert all(current[field] == cache[field] for field in ("bytes", "sha256"))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
