"""Refresh figures from a completed 2-D redetection report and preserved 1-D masks.

Each source owns a figure directory. Legacy session directories prefer the
primary regular-stimulus grid; a session without regular input displays its
primary other stimulus/grid. Different stimulus types or grids are never pooled.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from RF.rflocate.plotting import _finish_figure, _heatmap
from Utils.plotting import LIGHT_PLOT_STYLE


POLARITIES = ("excitatory", "inhibitory")


def figure_root(root, polarity):
    return Path(root) / ("inhibitory/rfmap" if polarity == "inhibitory" else "rfmap")


def native_ticks(ax, x, y=None):
    """Label stored bin centers without resampling or reversing array rows."""
    columns = np.unique(np.linspace(0, len(x) - 1, min(7, len(x))).round().astype(int))
    ax.set_xticks(columns, [f"{x[i]:g}" for i in columns])
    ax.set_xlabel("Horizontal position (deg)")
    if y is not None:
        rows = np.unique(np.linspace(0, len(y) - 1, min(7, len(y))).round().astype(int))
        ax.set_yticks(rows, [f"{y[i]:g}" for i in rows])
        ax.set_ylabel("Vertical position (deg)")


def missing_panel(ax):
    ax.text(.5, .5, "Not previously saved", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Independent 1-D RF")
    ax.set_axis_off()


def export_unit(data, title, output, aliases):
    x, y = data["x_positions"], data["y_positions"]
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.suptitle(title)
        _heatmap(fig, axes[0, 0], data["response"], "Mean firing rate, 0–200 ms")
        axes[0, 0].images[0].colorbar.set_label("Firing rate (Hz)")
        _heatmap(fig, axes[0, 1], data["mask_2d"], "2-D RF and centers", binary=True)
        rows, columns = np.nonzero(data["center_2d"])
        axes[0, 1].scatter(columns, rows, marker="x", color="#b91c1c", s=60)
        for ax in axes[0]:
            native_ticks(ax, x, y)
        axes[1, 0].plot(data["response_1d"], color="black")
        axes[1, 0].set(title="Horizontal response (sum across rows)", ylabel="Summed firing rate (Hz)")
        native_ticks(axes[1, 0], x)
        if np.isfinite(data["mask_1d"]).any():
            columns = np.flatnonzero(data["mask_1d"] == 1)
            axes[1, 0].scatter(columns, data["response_1d"][columns], color="#b91c1c", label="Previously saved 1-D RF")
            axes[1, 0].legend(fontsize=8)
            _heatmap(fig, axes[1, 1], data["mask_1d"][None, :], "Independent 1-D RF (preserved)", binary=True)
            columns = np.flatnonzero(data["center_1d"] == 1)
            axes[1, 1].scatter(columns, np.zeros(len(columns)), marker="x", color="#b91c1c", s=60)
            native_ticks(axes[1, 1], x)
            axes[1, 1].set_yticks([])
            axes[1, 1].set_ylabel("")
        else:
            missing_panel(axes[1, 1])
        _finish_figure(fig, output, False)
    created = []
    for suffix in (".png", ".svg"):
        source = Path(output).with_suffix(suffix)
        created.append(str(source))
        for alias in aliases:
            target = Path(alias).with_suffix(suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            created.append(str(target))
    return created


def population_counts(sources, polarity):
    keys = ("mask_2d", "center_2d", "mask_1d", "center_1d")
    counts = {key: sum(np.nansum(data[f"{polarity}_{key}"], axis=0)
                       for _, data in sources) for key in keys}
    counts["units"] = sum(len(data["unit_ids"]) for _, data in sources)
    counts["saved_1d_units"] = sum(int(np.isfinite(data[f"{polarity}_mask_1d"]).any(axis=1).sum())
                                   for _, data in sources)
    return counts


def export_population(sources, polarity, title, output, *, smooth=False):
    counts = population_counts(sources, polarity)
    if smooth:
        for key in ("mask_2d", "center_2d", "mask_1d", "center_1d"):
            counts[key] = gaussian_filter(counts[key].astype(float), sigma=1., mode="nearest")
        title += "\nDisplay counts smoothed: Gaussian sigma = 1 bin (masks unchanged)"
    x, y = (sources[0][1][key] for key in ("x_positions", "y_positions"))
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.suptitle(f"{title} · {polarity} RF · {counts['units']} units")
        for column, key, label in ((0, "mask_2d", "2-D RF area"), (1, "center_2d", "2-D RF centers")):
            _heatmap(fig, axes[0, column], counts[key], label)
            axes[0, column].images[0].set_clim(0, max(1, counts[key].max()))
            axes[0, column].images[0].colorbar.set_label("Smoothed unit count" if smooth else "Unit count")
            native_ticks(axes[0, column], x, y)
        for column, key, label in ((0, "mask_1d", "1-D RF area"), (1, "center_1d", "1-D RF centers")):
            if counts["saved_1d_units"]:
                axes[1, column].plot(counts[key], color="black")
                axes[1, column].set(title=f"Preserved {label} ({counts['saved_1d_units']} saved units)",
                                    ylabel="Smoothed unit count" if smooth else "Unit count")
                native_ticks(axes[1, column], x)
            else:
                missing_panel(axes[1, column])
        _finish_figure(fig, output, False)
    return [str(Path(output).with_suffix(suffix)) for suffix in (".png", ".svg")]


def grid_key(data):
    return tuple(data["x_positions"].tolist()), tuple(data["y_positions"].tolist())


def grid_name(key):
    digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:8]
    return f"grid_{len(key[1])}x{len(key[0])}_{digest}"


def write_manifest(manifest, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def export_report(report, manifest_path, *, workers=16):
    if report["errors"] or len(report["sources"]) != report["expected_sources"]:
        raise ValueError("A complete successful redetection report is required")
    sources = []
    for row in report["sources"]:
        with np.load(row["plot_data"], allow_pickle=False) as arrays:
            sources.append((row, {key: arrays[key] for key in arrays.files}))
    groups = {}
    for item in sources:
        row, data = item
        groups.setdefault((row["rf_root"], row["source_kind"], grid_key(data)), []).append(item)
    primary = {}
    for (root, kind, key), items in groups.items():
        score = (kind == "regular", len(key[1]) > 1, sum(len(data["unit_ids"]) for _, data in items))
        if root not in primary or score > primary[root][0]:
            primary[root] = score, kind, key
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(), "workers": workers,
        "source_count": len(sources), "created_files": [], "deleted_stale_unit_files": [],
        "grid_groups": [], "sources": [], "unit_figures_completed": 0,
    }
    for (root, kind, key), items in groups.items():
        manifest["grid_groups"].append({
            "rf_root": root, "source_kind": kind, "grid": grid_name(key),
            "x_positions": list(key[0]), "y_positions": list(key[1]),
            "source_paths": [row["source_path"] for row, _ in items],
            "legacy_primary": primary[root][1:] == (kind, key),
        })
    jobs, expected_units = [], {}
    for row, data in sources:
        source = Path(row["source_path"])
        canonical = source.with_name(source.stem + "_figures")
        legacy = primary[row["rf_root"]][1:] == (row["source_kind"], grid_key(data))
        manifest["sources"].append({"source_path": str(source), "figure_root": str(canonical),
                                    "legacy_root": row["rf_root"] if legacy else None,
                                    "units": len(data["unit_ids"])})
        for polarity in POLARITIES:
            unit_root = figure_root(canonical, polarity) / "units" / row["probe"]
            legacy_root = figure_root(row["rf_root"], polarity) / "units" / row["probe"]
            for root in ([unit_root, legacy_root] if legacy else [unit_root]):
                expected_units.setdefault(root, set()).update(str(int(uid)) for uid in data["unit_ids"])
            for index, uid in enumerate(data["unit_ids"]):
                prepared = {key: data[key][index] for key in ("response", "response_1d")}
                prepared.update({key: data[f"{polarity}_{key}"][index]
                                 for key in ("mask_2d", "center_2d", "mask_1d", "center_1d")})
                prepared.update({key: data[key] for key in ("x_positions", "y_positions")})
                label = f"{source.stem}\nProbe{row['probe']} · Unit {int(uid)} · {polarity}"
                aliases = [legacy_root / str(int(uid))] if legacy else []
                jobs.append((prepared, label, unit_root / str(int(uid)), aliases))
            for panel in (f"Probe{row['probe']}", "all", "all_smoothed"):
                output = figure_root(canonical, polarity) / panel / "rf_summary"
                manifest["created_files"].extend(export_population([(row, data)], polarity, f"{source.stem} · {panel}", output,
                                                                    smooth=panel == "all_smoothed"))
    for (root, kind, key), items in groups.items():
        if primary[root][1:] != (kind, key):
            continue
        panels = {"all": items, "all_smoothed": items}
        for item in items:
            panels.setdefault(f"Probe{item[0]['probe']}", []).append(item)
        for polarity in POLARITIES:
            for panel, panel_items in panels.items():
                output = figure_root(root, polarity) / panel / "rf_summary"
                manifest["created_files"].extend(export_population(panel_items, polarity, f"{panel} · {kind} · {grid_name(key)}", output,
                                                                    smooth=panel == "all_smoothed"))
    write_manifest(manifest, manifest_path)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(export_unit, *job) for job in jobs]
        for future in as_completed(futures):
            manifest["created_files"].extend(future.result())
            manifest["unit_figures_completed"] += 1
            if manifest["unit_figures_completed"] % 100 == 0:
                print(f"Rendered {manifest['unit_figures_completed']}/{len(jobs)} unit figures", flush=True)
                write_manifest(manifest, manifest_path)
    # Delete only stale numbered unit figures in directories actually refreshed.
    # Other plots, independent 1-D files and source/backup directories are untouched.
    for root, expected in expected_units.items():
        for path in root.iterdir():
            if path.is_file() and path.suffix in (".png", ".svg") and path.stem.isdecimal() and path.stem not in expected:
                path.unlink()
                manifest["deleted_stale_unit_files"].append(str(path))
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_manifest(manifest, manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    export_report(json.loads(args.report.read_text()), args.manifest, workers=args.workers)


if __name__ == "__main__":
    main()
