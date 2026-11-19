#!/usr/bin/env python3
"""Inspect and export M22 261008 HD / ON / OFF examples for slides_v2.

Run on hhw9l84 from the repository with its rfmapping virtualenv. The screening
cohort is saved HD class 3 intersected with saved excitatory ON RF detections.
No RF detection, rate masking, or change to saved classification is performed.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from HD.profiles import load_hd_tc, hd_profiles
from RF.paths import rf_source_path
from RF.rflocate import load_rf, load_rfmap, rf_result_path
from Utils.plotting import LIGHT_PLOT_STYLE
from Utils.statistic_utils import smooth_tuning_curve


class RFGrayNorm(Normalize):
    """Use the same 8-bit gray palette quantization as slides_v2 and the GUI."""

    def __call__(self, value, clip=None):
        return np.ma.round(super().__call__(value, clip=clip) * 232) / 232


def cell_edges(positions):
    mid = (positions[:-1] + positions[1:]) / 2
    return np.r_[2 * positions[0] - mid[0], mid, 2 * positions[-1] - mid[-1]]


def plot_hd(ax, curve):
    angles = curve.index.to_numpy(dtype=float) % 360
    order = np.argsort(angles)
    angles, values = angles[order], curve.to_numpy()[order]
    theta = np.deg2rad(np.r_[angles, angles[0] + 360])
    values = np.r_[values, values[0]]
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.plot(theta, values, color="#202020", lw=1.8)
    ax.fill_between(theta, 0, values, color="0.85")
    ax.set_thetagrids([0, 90, 180, 270], ["0°", "90°", "180°", "270°"])
    ax.set_ylim(bottom=0)
    ax.grid(color="#d2d7df", lw=.8)
    ax.set_title("HD tuning (Hz)", pad=22)


def plot_annular(ax, response, title, *, vmin=None):
    values = response.to_2d_array()
    edges = cell_edges(response.x_positions)
    rows, columns = values.shape
    palette = ListedColormap(np.repeat(np.arange(18, 251)[:, None] / 255, 3, axis=1))
    norm = RFGrayNorm(float(np.nanmin(values)) if vmin is None else vmin,
                      float(np.nanmax(values)))
    # Source Y increases downward. Higher elevation is on the outer ring;
    # angular X is clockwise from the top, with four blank inner rows.
    for row in range(rows):
        for col in range(columns):
            value = values[row, col]
            color = palette(norm(value)) if np.isfinite(value) else "white"
            wedge = Wedge((0, 0), 4 + rows - row, 90 - edges[col + 1],
                          90 - edges[col], width=1, facecolor=color,
                          edgecolor=color, linewidth=.3, antialiased=False)
            if not np.isfinite(value):
                wedge.set(hatch="///", edgecolor=".7")
            ax.add_patch(wedge)
    radius = 4 + rows
    for angle, label in ((-180, "±180°"), (-90, "−90°"), (0, "0°"), (90, "90°")):
        theta = np.deg2rad(90 - angle)
        ax.text((radius + 1.1) * np.cos(theta), (radius + 1.1) * np.sin(theta),
                label, ha="center", va="center", fontsize=10)
    ax.set(xlim=(-radius - 2, radius + 2), ylim=(-radius - 2, radius + 2), aspect="equal")
    ax.axis("off")
    ax.set_title(title, pad=10)
    colorbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette), ax=ax,
                                 label="Firing rate (Hz)", fraction=.035, pad=.025, shrink=.73)
    colorbar.solids.set_rasterized(False)


def save_figure(fig, path):
    for extension in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=180,
                    facecolor="white", transparent=False, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", nargs="*", type=int, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("slides_v2_output"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({**LIGHT_PLOT_STYLE, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white",
                         "savefig.transparent": False, "text.color": "#202020",
                         "axes.labelcolor": "#202020", "xtick.color": "#202020",
                         "ytick.color": "#202020", "svg.fonttype": "none", "font.size": 11})
    base = Path("/mnt/senzailab/Kai/#Recording/m22/261008")
    hd_path = base / "261008_9/data/tuning_curves/ProbeA/tuning_curves.tc"
    tc = load_hd_tc(hd_path)
    profiles = hd_profiles(tc, probe="A", bins=30, hd_class=3)
    info = pd.DataFrame(tc["unit_data"], index=tc["unit_id"])
    sources = {pol: rf_source_path(base / "261008_2", "A", root_kind=root)
               for pol, root in (("ON", "rfmapping"), ("OFF", "rfmapping_off"))}
    results = {pol: load_rf(rf_result_path(source, rf_type="excitatory"))
               for pol, source in sources.items()}
    on_result = results["ON"]
    units = [int(unit) for _, unit in profiles.index
             if unit in on_result.unit_ids and on_result.for_unit(unit).any()]
    curves = {unit: smooth_tuning_curve(profiles.loc[("A", unit)], sigma=1) for unit in units}
    responses = {unit: {} for unit in units}
    for polarity, source in sources.items():
        maps = load_rfmap(source)
        presentations = maps.by_unit_id(units[0]).to_firing_rate(
            reconstruct_presentations=True).presentation_counts
        for unit in units:
            rates = maps.by_unit_id(unit).to_firing_rate(presentation_counts=presentations)
            responses[unit][polarity] = rates.mean_rate(*results[polarity].manifest["time_range_s"])
        del maps, rates
    rows = []
    for unit in units:
        row = {"unit_id": unit, "hd_class": 3, "hd_kappa": float(info.loc[unit, "von_mises_kappa"]),
               "hd_mvl": float(info.loc[unit, "rate_mvl"]),
               "hd_peak_deg": float(curves[unit].idxmax()) % 360, "hd_peak_hz": float(curves[unit].max())}
        for pol in sources:
            values = responses[unit][pol].to_2d_array()
            mask = results[pol].for_unit(unit).astype(bool)
            center = results[pol].for_unit(unit, center_only=True).astype(bool)
            row.update({f"{pol.lower()}_peak_hz": float(np.nanmax(values)),
                        f"{pol.lower()}_rf_bins": int(mask.sum()),
                        f"{pol.lower()}_rf_centers_deg": json.dumps(responses[unit][pol].x_positions[np.nonzero(center)[1]].tolist()),
                        f"{pol.lower()}_rf_mean_hz": float(np.nanmean(values[mask])) if mask.any() else None,
                        f"{pol.lower()}_outside_rf_mean_hz": float(np.nanmean(values[~mask]))})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("unit_id")
    table.to_csv(args.output_dir / "m22_261008_f3_candidate_metrics.csv")
    # Compact screening sheets retain every eligible unit, not only chosen examples.
    for page in range((len(units) + 5) // 6):
        cohort = units[page * 6:(page + 1) * 6]
        fig = plt.figure(figsize=(13, 2.2 * len(cohort)), layout="constrained")
        grid = fig.add_gridspec(len(cohort), 3, width_ratios=(1, 2.2, 2.2))
        for i, unit in enumerate(cohort):
            ax = fig.add_subplot(grid[i, 0], projection="polar")
            plot_hd(ax, curves[unit])
            ax.set_title(f"Unit {unit} · κ={table.loc[unit, 'hd_kappa']:.2f}", fontsize=10, pad=12)
            ax.tick_params(labelsize=7)
            for j, pol in enumerate(sources, 1):
                ax = fig.add_subplot(grid[i, j])
                response = responses[unit][pol]
                image = ax.pcolormesh(cell_edges(response.x_positions), cell_edges(response.y_positions),
                                      response.to_2d_array(), cmap="gray", shading="flat")
                ax.invert_yaxis()
                ax.contour(response.x_positions, response.y_positions,
                           results[pol].for_unit(unit), levels=[.5], colors=["#d62728"], linewidths=.8)
                ax.set_title(f"{pol} · unsmoothed 0–200 ms", fontsize=10)
                ax.set(xlabel="Native X (deg)", ylabel="Source Y (deg)")
                fig.colorbar(image, ax=ax, label="Hz", shrink=.65)
        fig.savefig(args.output_dir / f"m22_261008_f3_screen_{page + 1}.png", dpi=120,
                    facecolor="white", transparent=False)
        plt.close(fig)
    for unit in args.units:
        assert unit in units, f"Unit {unit} is outside the saved HD3 / excitatory ON cohort"
        prefix = args.output_dir / f"m22_261008_f3_unit{unit}"
        fig = plt.figure(figsize=(14, 8), facecolor="white")
        hd_ax = fig.add_axes((.06, .17, .39, .66), projection="polar")
        plot_hd(hd_ax, curves[unit])
        for polarity, bottom in (("ON", .51), ("OFF", .04)):
            ax = fig.add_axes((.56, bottom, .35, .40))
            plot_annular(ax, responses[unit][polarity], f"{polarity} Receptive Field Map")
        fig.suptitle(f"M22 · 261008 · Unit {unit}", x=.05, ha="left", fontsize=22, weight="bold")
        fig.text(.05, .025, "Probe A  |  HD session 9  |  RF session 2, 0–200 ms, unsmoothed", fontsize=10)
        save_figure(fig, prefix)
        fig, ax = plt.subplots(figsize=(5.1, 5.1), subplot_kw={"projection": "polar"}, layout="constrained")
        plot_hd(ax, curves[unit])
        save_figure(fig, prefix.with_name(prefix.name + "_hd_polar"))
        for pol in sources:
            fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
            plot_annular(ax, responses[unit][pol], f"{pol} Receptive Field Map")
            save_figure(fig, prefix.with_name(prefix.name + f"_{pol.lower()}_annular"))
    source_files = {"HD": hd_path, **sources,
                    **{f"{pol}_excitatory_detection": rf_result_path(source) for pol, source in sources.items()}}
    manifest = {"mouse": "m22", "date": "261008", "probe": "A", "hd_session": 9, "rf_session": 2,
                "cohort": "Saved HD class 3 and saved excitatory ON RF; all raw rate bins retained",
                "candidate_unit_ids": units, "selected_unit_ids": args.units,
                "selection_notes": {
                    "263": "Clear right ON RF, center +78 deg; HD peak 330 deg. Closest single right-field example to requested layout.",
                    "269": "Clear left ON RF, center -78 deg; sharply tuned HD peak 102 deg.",
                    "256": "Strong forward/right ON RF, center +18 deg; HD peak 18 deg and high peak firing rate.",
                },
                "hd_display": "30 angular bins; circular Gaussian smoothing sigma=1 bin",
                "rf_display": "Unsmoothed mean firing rate in saved 0–200 ms detector window; independent ON/OFF color scales",
                "annular_coordinates": "Native X clockwise from top; source row 0 outermost, last row innermost; radius is row index, not angle or distance",
                "files": {name: {"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
                          for name, path in source_files.items()},
                "selected_metrics": table.loc[args.units].reset_index().to_dict(orient="records")}
    (args.output_dir / "m22_261008_f3_selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
