"""Export continuous RF fits to a new directory without changing detections.

Run on hhw9l84 from the repository root with the rfmapping environment.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from RF.rflocate import fit_rf_gaussians, load_rf, load_rfmap


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(value):
    if isinstance(value, Mapping):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def draw_map(ax, values, x, y, *, vmin, vmax, title):
    dx, dy = x[1] - x[0], y[1] - y[0]
    picture = ax.imshow(
        values, cmap="gray", origin="upper", interpolation="nearest",
        extent=(x[0] - dx / 2, x[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2),
        vmin=vmin, vmax=vmax, aspect="equal",
    )
    ax.set(title=title, xlabel="x position (deg)", ylabel="y position (deg)")
    ax.set_xticks(x[np.unique(np.linspace(0, x.size - 1, 5).astype(int))])
    ax.set_yticks(y[::2])
    return picture


def export_plots(rows, output):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "#202020", "axes.labelcolor": "#202020",
        "xtick.color": "#202020", "ytick.color": "#202020", "font.size": 10,
    })
    fig, axes = plt.subplots(len(rows), 3, figsize=(19, 3.3 * len(rows)), squeeze=False,
                             layout="constrained")
    for row_axes, (values, fit) in zip(axes, rows):
        x, y = fit["x_positions"], fit["y_positions"]
        fitted = fit["fitted_response_hz"]
        vmin, vmax = min(np.nanmin(values), np.nanmin(fitted)), max(np.nanmax(values), np.nanmax(fitted))
        counts = fit["component_counts"]
        titles = [f"Cluster {fit['unit_id']} | original response",
                  f"Saved regions | {counts['excitatory']} EXT + {counts['inhibitory']} INH",
                  f"Continuous fit | RMSE {fit['training_rmse_hz']:.2f} Hz, R² {fit['training_r_squared']:.2f}"]
        for ax, data, title in zip(row_axes, (values, values, fitted), titles):
            picture = draw_map(ax, data, x, y, vmin=vmin, vmax=vmax, title=title)
        for polarity, color in (("excitatory", "#159c49"), ("inhibitory", "#a346cf")):
            for r, c in np.argwhere(fit["detection_masks"][polarity]):
                row_axes[1].add_patch(Rectangle(
                    (x[c] - (x[1] - x[0]) / 2, y[r] - (y[1] - y[0]) / 2),
                    x[1] - x[0], y[1] - y[0], fill=False, edgecolor=color, linewidth=1.3,
                    linestyle="-" if polarity == "excitatory" else "--",
                ))
        fig.colorbar(picture, ax=list(row_axes), shrink=.65, label="Hz (shared within row)")
    window = rows[0][1]["time_range_s"]
    fig.suptitle(f"Detected-region Gaussian fits · native bins · {window[0] * 1000:g}–{window[1] * 1000:g} ms · no final threshold", fontsize=16)
    fig.savefig(output / "original_regions_fit.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 3.3 * len(rows)), squeeze=False,
                             layout="constrained")
    for row_axes, (_, fit) in zip(axes, rows):
        maximum = max(float(np.nanmax(fit[key])) for key in ("net_excitatory_hz", "net_inhibitory_hz"))
        for ax, key, label in zip(row_axes, ("net_excitatory_hz", "net_inhibitory_hz"), ("EXT", "INH")):
            picture = draw_map(ax, fit[key], fit["x_positions"], fit["y_positions"],
                               vmin=0, vmax=maximum if maximum > 0 else 1,
                               title=f"Cluster {fit['unit_id']} | net {label} magnitude")
        fig.colorbar(picture, ax=list(row_axes), shrink=.65, label="Hz relative to fitted offset")
    fig.suptitle("Continuous EXT / INH outputs · user chooses the final threshold", fontsize=15)
    fig.savefig(output / "continuous_ext_inh.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--excitatory", required=True, type=Path)
    parser.add_argument("--inhibitory", required=True, type=Path)
    parser.add_argument("--units", required=True, type=int, nargs="+")
    parser.add_argument("--window", required=True, type=float, nargs=2, metavar=("START_S", "END_S"))
    parser.add_argument("--wrap-x", required=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = (args.source, args.excitatory, args.inhibitory)
    before = {str(path): file_hash(path) for path in inputs}
    args.output.mkdir(parents=True, exist_ok=False)
    raw, excitatory, inhibitory = load_rfmap(args.source), load_rf(args.excitatory), load_rf(args.inhibitory)
    rows, reports = [], {}
    for unit_id in args.units:
        print(f"Fitting cluster {unit_id} from its saved EXT/INH regions", flush=True)
        response = raw.by_unit_id(unit_id).to_firing_rate(reconstruct_presentations=True).mean_rate(*args.window)
        values = response.to_2d_array()
        fit = fit_rf_gaussians(response, excitatory=excitatory, inhibitory=inhibitory, wrap_x=args.wrap_x)
        if not fit["success"]:
            raise RuntimeError(f"Cluster {unit_id} did not converge: {fit['message']}")
        arrays = {key: value for key, value in fit.items() if isinstance(value, np.ndarray)}
        arrays["observed_response_hz"] = values
        for polarity in ("excitatory", "inhibitory"):
            arrays[f"{polarity}_detection_mask"] = fit["detection_masks"][polarity]
            arrays[f"{polarity}_detection_labels"] = fit["detection_labels"][polarity]
        np.savez_compressed(args.output / f"cluster{unit_id}_continuous_fit.npz", **arrays)
        reports[str(unit_id)] = {key: value for key, value in fit.items()
                                if key not in arrays and key not in {"detection_masks", "detection_labels"}}
        rows.append((values, fit))
        print(f"  components={fit['component_counts']}, RMSE={fit['training_rmse_hz']:.4f} Hz", flush=True)
    export_plots(rows, args.output)
    after = {str(path): file_hash(path) for path in inputs}
    if before != after:
        raise RuntimeError("An input changed during fitting; inspect provenance before using the export")
    summary = {
        "model": "one signed axis-aligned Gaussian per detected connected region plus fitted offset",
        "final_response_threshold": None, "input_sha256_before": before, "input_sha256_after": after,
        "input_manifests": {"excitatory": excitatory.manifest, "inhibitory": inhibitory.manifest},
        "code_sha256": {str(path): file_hash(path) for path in (Path(__file__), Path("RF/rflocate/fitting.py"))},
        "units": reports,
    }
    (args.output / "summary.json").write_text(json.dumps(json_value(summary), indent=2, allow_nan=False) + "\n")
    print(f"New results: {args.output}; original source and detections unchanged", flush=True)


if __name__ == "__main__":
    main()
