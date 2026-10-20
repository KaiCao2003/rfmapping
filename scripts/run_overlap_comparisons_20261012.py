"""Compare HD with the native-bin intersection of independent 2-D/1-D RFs.

Run only on hhw9l84, from ~/Developer/rfmapping with the rfmapping Python.
All outputs are isolated from recording-side detection results and TC caches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from Utils.direction_comparison import profile_table, prepare_comparison
from Utils.json_tools import read_formatted_json
from Utils.plotting import LIGHT_PLOT_STYLE
from RF.rflocate import load_rfmap, save_rf
from RF.rflocate.workflow import analyze_rf_file
from Utils.statistic_utils import (compare_direction_angles, fisher_lee_statistics,
                                   rayleigh_uniformity, circular_distance_matrix_deg)

SESSIONS = [("m14", "260609", 3, 1), ("m15", "260630", 3, 1),
            ("m19", "260827", 2, 9), ("m20", "260921", 2, 9),
            ("m21", "261006", 3, 9), ("m22", "261008", 2, 9)]
METHOD = {
    "reference_2d": "global",
    "rf_time_range_s": [0, 0.2], "sigma_2d": 1.8, "sigma_1d": 1.0,
    "drop_bins": 0, "max_zero_bins": 2, "wrap_x": True,
    "is_shuffle": False, "collapse_from_2d": False,
    "intersection": "mask_2d & mask_1d[:, None, :] on identical native x positions",
    "projection": "Sum presentation-normalized 0-200 ms mean rates over only intersecting y,x bins at each native x; excluded bins contribute zero; masks and raw counts saved separately",
    "hd": "Saved HD class 3; unsmoothed 30-bin count/occupancy rates; no saved data changed",
    "peak": "First native maximum in source bin order, no smoothing or angular resampling",
    "normalization": "For heatmaps only, divide each full native curve by its finite maximum",
    "pooling": "Retain native RF grid separately for each unit, including m14 partial -150..150 degree field; pooled statistics descriptive",
    "statistics": "Session circular-circular JS, Fisher-Lee, and Mantel pairing permutations (10000, seed 1); one-angle-per-unit Rayleigh for wrapped HD+RF peak sum; exploratory uncorrected p values",
    "statistics_bins": 15,
    "aligned_heatmaps": "Display-only linear interpolation after subtracting HD peak; partial fields retain NaN outside measured extent",
    "scope": "Exploratory threshold RF regions, not shuffle-significance RF selection",
}


def record(path):
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")


def load_saved_hd(path, prefix):
    data = read_formatted_json(path)
    ids = np.asarray(data["unit_id"], dtype=int)
    selected = np.asarray(data["unit_data"]["hd_class"]) == 3
    counts = np.asarray(data["spike_counts"], dtype=float)[selected]
    occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
    group = len(occupancy) // 30
    counts = counts.reshape(len(counts), 30, group).sum(axis=2)
    occupancy = occupancy.reshape(30, group).sum(axis=1)
    rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan), where=occupancy > 0)
    edges = np.asarray(data["angle_bin_edges_deg"], dtype=float)[::group]
    table = profile_table(rates, [f"{prefix}:{u}" for u in ids[selected]],
                          (edges[:-1] + edges[1:]) / 2)
    table.attrs.update(label="HD", range=[-180, 180])
    return table, data["metadata"]["classification"]


def save_figure(fig, path):
    fig.savefig(path, dpi=190, facecolor="white", transparent=False)
    fig.savefig(path.with_suffix(".svg"), facecolor="white", transparent=False)
    plt.close(fig)
    return str(path)


def edges_for(centers):
    centers = np.asarray(centers)
    return np.r_[centers[0] - (centers[1] - centers[0]) / 2,
                 (centers[:-1] + centers[1:]) / 2,
                 centers[-1] + (centers[-1] - centers[-2]) / 2]


def normalized(values):
    maximum = np.nanmax(values)
    return values / maximum if maximum > 0 else values.copy()


def heatmap_axis(ax, rows, key, *, title, xlabel, wrap_seam=False):
    # Drawing rows on their own native degree grids avoids pooling interpolation.
    for j, row in enumerate(rows):
        angles, values = row[key + "_angles"], row[key + "_display"]
        order = np.argsort(angles)
        angles, values = angles[order], values[order]
        if wrap_seam:
            # The seam bin spans both ends of the circular display.
            angles = np.r_[angles, 180.0]
            values = np.r_[values, values[0]]
        ax.pcolormesh(edges_for(angles), [j, j + 1], values[None, :],
                      vmin=0, vmax=1, cmap="viridis", shading="flat", rasterized=True)
    ax.set(xlim=(-180, 180), ylim=(len(rows), 0), xticks=[-180, -90, 0, 90, 180],
           title=title, xlabel=xlabel, ylabel="Paired unit (rank)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))


def native_figure(rows, out, label, sort_by):
    rows = sorted(rows, key=lambda row: row[sort_by + "_peak_deg"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), layout="constrained")
    heatmap_axis(axes[0], rows, "hd", title=f"HD | sorted by {sort_by.upper()}", xlabel="HD peak coordinate (°)")
    heatmap_axis(axes[1], rows, "rf", title=f"2-D ∩ 1-D excitatory RF | sorted by {sort_by.upper()}", xlabel="Native RF azimuth (°)")
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), ax=axes,
                 label="Normalized response", fraction=0.025, pad=0.015)
    fig.suptitle(f"{label}  |  n = {len(rows)}", fontsize=17)
    pd.DataFrame({"unit_key": [r["key"] for r in rows]}).to_csv(out / f"native_{sort_by}_row_order.csv", index=False)
    return save_figure(fig, out / f"native_sorted_by_{sort_by}.png")


def aligned_figure(rows, out, label):
    aligned = []
    for row in rows:
        hd = pd.DataFrame([row["hd_display"]], index=[row["key"]], columns=row["hd_angles"])
        rf = pd.DataFrame([row["rf_display"]], index=[row["key"]], columns=row["rf_angles"])
        hd.attrs["range"], rf.attrs["range"] = [-180, 180], row["rf_range"]
        comparison = prepare_comparison(hd, rf, mode="aligned", bins=60)
        aligned.append({**row,
            "hd_angles": comparison["reference"].columns.to_numpy(),
            "hd_display": comparison["reference"].iloc[0].to_numpy(),
            "rf_angles": comparison["matched"].columns.to_numpy(),
            "rf_display": comparison["matched"].iloc[0].to_numpy()})
    aligned.sort(key=lambda row: row["hd_peak_deg"])
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), layout="constrained")
    heatmap_axis(axes[0], aligned, "hd", title="HD | HD peak aligned", xlabel="Angle from HD peak (°)", wrap_seam=True)
    heatmap_axis(axes[1], aligned, "rf", title="2-D ∩ 1-D RF | HD peak aligned", xlabel="RF angle − HD peak (°)", wrap_seam=True)
    fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap="viridis"), ax=axes,
                 label="Normalized response", fraction=0.025, pad=0.015)
    fig.suptitle(f"{label}  |  n = {len(rows)}  |  display interpolation only", fontsize=16)
    return save_figure(fig, out / "hd_peak_aligned.png")


def statistics_and_figures(rows, out, label, pooled=False, *, statistics=None):
    hd = np.asarray([r["hd_peak_deg"] for r in rows])
    rf = np.asarray([r["rf_peak_deg"] for r in rows])
    sums = (hd + rf + 180) % 360 - 180
    if statistics is None:
        stats = {"n_units": len(rows),
                 "fisher_lee": fisher_lee_statistics(hd, rf, n_permutations=10000, random_seed=1, test=not pooled),
                 "sum_rayleigh": rayleigh_uniformity(sums),
                 "direction": None if pooled else compare_direction_angles(hd, rf, n_permutations=10000, random_seed=1)}
    else:
        stats = statistics
    rayleigh, fl = stats["sum_rayleigh"], stats["fisher_lee"]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.5), layout="constrained")
    axes[0].scatter(hd, rf, s=30, c="#225c91", alpha=.8)
    axes[0].set(xlabel="HD peak (°)", ylabel="Overlap RF peak (°)",
                xlim=(-180, 180), ylim=(-180, 180), xticks=[-180, -90, 0, 90, 180], yticks=[-180, -90, 0, 90, 180])
    axes[0].grid(color=".87")
    test_text = f"Fisher–Lee ρ = {fl['rho']:.3f}" if fl['rho'] is not None else "Fisher–Lee ρ undefined"
    if fl['permutation_p'] is not None:
        test_text += f"\np = {fl['permutation_p']:.4g} (10,000 permutations)"
    if stats["direction"] is not None:
        same = stats["direction"]["same_unit"]
        test_text += f"\nJS ρ = {same['circular_correlation_rho']:.3f}; p = {same['permutation_p']:.4g}"
    axes[0].set_title(test_text, fontsize=13)
    axes[1].hist(sums, bins=np.linspace(-180, 180, 16), color="#225c91", edgecolor="white")
    axes[1].set(xlabel="Wrapped HD peak + overlap RF peak (°)", ylabel="Unit count",
                xlim=(-180, 180), xticks=[-180, -90, 0, 90, 180],
                title=f"Resultant R = {rayleigh['r']:.3f}" + (" | pooled descriptive" if pooled else f"; Rayleigh p = {rayleigh['p']:.4g}"))
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(f"{label}  |  n = {len(rows)}", fontsize=17)
    paths = [save_figure(fig, out / "peak_statistics.png")]
    if not pooled:
        dx, dy = circular_distance_matrix_deg(hd), circular_distance_matrix_deg(rf)
        upper = np.triu_indices(len(rows), k=1)
        fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
        image = ax.hist2d(dx[upper], dy[upper], bins=np.linspace(0, 180, 16), cmap="viridis")[3]
        fig.colorbar(image, ax=ax, label="Unique unit-pair count")
        pair = stats["direction"]["pairwise"]
        ax.set(xlabel="Pairwise HD peak distance (°)", ylabel="Pairwise overlap RF peak distance (°)",
               title=f"{label}\nMantel ρ = {pair['spearman_mantel_rho']:.3f}; p = {pair['permutation_p']:.4g}",
               xticks=[0, 45, 90, 135, 180], yticks=[0, 45, 90, 135, 180])
        paths.append(save_figure(fig, out / "pairwise_distances.png"))
    if pooled:
        stats["sum_rayleigh"]["p"] = None
        stats["inference_note"] = "Pooled descriptive only; cells nested within sessions. Refer to six session-specific permutation results."
    return stats, paths


def analyze_session(base, output, mouse, date, rf_session, hd_session):
    session = base / mouse / date / f"{date}_{rf_session}"
    hd_dir = base / mouse / date / f"{date}_{hd_session}"
    key = f"{mouse}_{date}_rf{rf_session}_hd{hd_session}"
    out = output / key
    out.mkdir(parents=True, exist_ok=True)
    sources = sorted((session / "data/rfmapping/good/-100_400_1ms").glob("Probe?/regular*.rfmap"))
    rows, probe_reports, all_before = [], [], []
    for source in sources:
        probe = source.parent.name.removeprefix("Probe")
        prefix = f"{mouse}:{date}:{probe}"
        tc_path = hd_dir / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
        protected = [source, tc_path, *source.parent.glob("*.npz"), *source.parent.glob("*_analysis.json")]
        before = [record(p) for p in protected]
        hd, classification = load_saved_hd(tc_path, prefix)
        result = analyze_rf_file(
            source, probe=probe, reference_2d="global", drop_bins=0, save_results=False,
        )
        save_rf(result["result_2d"], out / f"Probe{probe}_drop0_2d.npz")
        save_rf(result["result_1d"], out / f"Probe{probe}_drop0_1d.npz")
        mask2 = result["mask_2d"].astype(bool)
        mask1 = result["mask_1d"].astype(bool)
        overlap = mask2 & mask1[:, None, :]
        rates = np.stack([m.to_2d_array() for m in result["summed"]])
        overlap_rates = np.where(overlap, rates, 0).sum(axis=1)
        raw = load_rfmap(source).sum(0, .2, show_progress=False)
        counts = np.stack([raw.by_unit_id(int(u)).to_2d_array() for u in result["summed"].unit_ids])
        overlap_counts = np.where(overlap, counts, 0).sum(axis=1)
        x, y = result["summed"][0].x_positions, result["summed"][0].y_positions
        np.savez_compressed(out / f"Probe{probe}_native_overlap.npz",
            unit_ids=result["summed"].unit_ids, x_positions=x, y_positions=y,
            mask_2d=mask2, mask_1d=mask1, overlap_mask=overlap,
            raw_counts=counts, mean_rate_hz=rates, projected_overlap_counts=overlap_counts,
            projected_overlap_rate_sum=overlap_rates,
            original_missing_mask=~np.isfinite(rates))
        ids = [f"{prefix}:{u}" for u in result["summed"].unit_ids]
        pd.DataFrame(overlap_rates, index=pd.Index(ids, name="unit_key"), columns=x).to_csv(out / f"Probe{probe}_overlap_rates_native.csv")
        hd.to_csv(out / f"Probe{probe}_hd30_class3.csv")
        valid = overlap.any(axis=(1, 2))
        for i in np.flatnonzero(valid):
            unit = ids[i]
            if unit not in hd.index:
                continue
            hv, rv = hd.loc[unit].to_numpy(), overlap_rates[i]
            rows.append(dict(key=unit, session=key, hd_peak_deg=float(hd.columns[np.nanargmax(hv)]),
                             rf_peak_deg=float(x[np.nanargmax(rv)]),
                             hd_angles=hd.columns.to_numpy(), hd_values=hv,
                             rf_angles=x, rf_values=rv, rf_range=edges_for(x)[[0,-1]].tolist(),
                             hd_display=normalized(hv), rf_display=normalized(rv),
                             overlap_bins=int(overlap[i].sum())))
        after = [record(p) for p in protected]
        assert before == after, "Recording sources or existing RF outputs changed"
        assert np.array_equal(overlap, mask2 & mask1[:, None, :])
        assert np.array_equal(np.where(overlap, rates, 0).sum(axis=1), overlap_rates, equal_nan=True)
        extra = result["summed"][0].metadata.get("presentationCountSource", {})
        reconstruction_sources = [record(Path(extra[k])) for k in ("trials_mat", "onsets_npy") if k in extra]
        probe_reports.append({"probe": probe, "source": record(source), "hd_source": record(tc_path),
            "classification": classification, "reconstruction_sources": reconstruction_sources,
            "n_raw": len(result["bin_qc"]["unit_ids"]), "n_qc": len(ids),
            "n_2d": int(mask2.any(axis=(1,2)).sum()), "n_1d": int(mask1.any(axis=1).sum()),
            "n_overlap": int(valid.sum()), "n_hd3": len(hd),
            "n_paired": sum(r["key"].startswith(prefix + ":") for r in rows),
            "grid_x": x.tolist(), "grid_y": y.tolist(), "detection_parameters": result["parameters"],
            "source_artifacts_unchanged": before == after})
        all_before.extend(before)
    label = f"{mouse} {date} | RF{rf_session} × HD{hd_session} | drop_bins = 0"
    plots = [native_figure(rows, out, label, "hd"), native_figure(rows, out, label, "rf"), aligned_figure(rows, out, label)]
    statistics, stat_plots = statistics_and_figures(rows, out, label)
    plots.extend(stat_plots)
    pd.DataFrame([{k: row[k] for k in ("key", "session", "hd_peak_deg", "rf_peak_deg", "overlap_bins")} for row in rows]).to_csv(out / "paired_peaks.csv", index=False)
    report = {"key": key, "label": label, "rf_session": str(session), "hd_session": str(hd_dir),
              "probes": probe_reports, "statistics": statistics, "figures": plots,
              "protected_artifacts": all_before, "sources_unchanged": True}
    write_json(out / "manifest.json", report)
    print(f"{key}: paired n={len(rows)}; {len(plots)} figures", flush=True)
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/mnt/senzailab/Kai/#Recording"))
    parser.add_argument("--output", type=Path, default=Path("output/session_comparisons_20261012/overlap"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({**LIGHT_PLOT_STYLE, "font.size": 12, "axes.titlesize": 13,
                         "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11})
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "method": METHOD,
              "script": record(Path(__file__)), "sessions": []}
    pooled = []
    for session in SESSIONS:
        item, rows = analyze_session(args.base, args.output, *session)
        report["sessions"].append(item)
        pooled.extend(rows)
        write_json(args.output / "manifest.json", report)
    out = args.output / "pooled"
    out.mkdir(exist_ok=True)
    plots = [native_figure(pooled, out, "All six sessions | native grids", "hd"),
             native_figure(pooled, out, "All six sessions | native grids", "rf"),
             aligned_figure(pooled, out, "All six sessions")]
    stats, paths = statistics_and_figures(pooled, out, "All six sessions | descriptive", pooled=True)
    pd.DataFrame([{k: row[k] for k in ("key", "session", "hd_peak_deg", "rf_peak_deg", "overlap_bins")} for row in pooled]).to_csv(out / "paired_peaks.csv", index=False)
    report["pooled"] = {"statistics": stats, "figures": plots + paths}
    report["output_files"] = [record(p) for p in sorted(args.output.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    write_json(args.output / "manifest.json", report)
    print(f"COMPLETE: {args.output}/manifest.json", flush=True)


if __name__ == "__main__":
    main()
