"""Render RF response maps, detected fields, and prepared population arrays."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from RF.paths import organized_rf_source
from Utils.plotting import LIGHT_PLOT_STYLE

__all__ = [
    "plot_2d_rfmap", "plot_1d_rfmap", "rf_unit_plot_data", "rf_population_counts",
    "plot_rf_unit", "plot_rf_population_map", "plot_rf_population", "smooth_rf_population", "export_rf_units",
    "plot_rf_center_correlograms",
]


def _finish_figure(fig, output_path, show):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight",
            facecolor="white", transparent=False,
        )
    if show:
        plt.show()
    plt.close(fig)


def _figure_dir(output_dir, rf_type):
    return Path(output_dir) / "rfmap" / ("inh" if rf_type == "inhibitory" else "ext")


def plot_rf_center_correlograms(histograms, *, curves=None, title=None, output_path=None, show=True):
    """Draw raw angular pair counts or explicitly prepared display curves.

    ``histograms`` comes from ``rf_center_histograms``; optional ``curves`` uses
    the same degree index and columns. Loading, pairing and smoothing are external.
    """
    display = histograms if curves is None else curves
    theta = np.deg2rad(display.index.to_numpy(dtype=float))
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={"projection": "polar"}, layout="constrained")
        if title is not None:
            fig.suptitle(title)
        for ax, label, color in zip(axes, ("E–E", "E→I"), ("#248f55", "#8055a5")):
            values = display[label].to_numpy()
            ax.plot(np.r_[theta, theta[0] + 2 * np.pi], np.r_[values, values[0]], color=color, linewidth=2)
            ax.set_ylim(bottom=0)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetagrids([0, 60, 120, 180, 240, 300], ["0°", "+60°", "+120°", "180°", "−120°", "−60°"])
            n_pairs = histograms.attrs["pair_counts"][label]
            n_units = histograms.attrs["unit_counts"][label]
            ax.set_title(f"{label} | {n_units} unit records | {n_pairs} ordered pairs", pad=20)
            if n_pairs == 0:
                ax.set_ylim(0, 1)
                ax.text(0.5, 0.5, "No pairs", transform=ax.transAxes, ha="center", va="center")
        _finish_figure(fig, output_path, show)
    return fig, axes


def _heatmap(fig, ax, values, title, *, binary=False):
    n_rows, n_columns = values.shape
    aspect = n_columns * 7.0 / 30.0 if n_rows == 1 else "equal"
    limits = dict(vmin=0, vmax=1) if binary else {}
    image = ax.imshow(values, cmap="Greys", interpolation="nearest", aspect=aspect, **limits)
    ax.set(title=title, xlabel="Column", ylabel="Row")
    fig.colorbar(image, ax=ax, label="RF mask" if binary else "Response")


def rf_unit_plot_data(analysis, unit_id, *, rf_only=False):
    """Use the detector's horizontal input, or an explicit RF-row projection."""
    maps = analysis["summed"]
    index = maps.unit_ids.index(unit_id)
    projected = (
        maps[index].sum_to_1d("x", rf_only=True, detected_rf=analysis["result_2d"])
        if rf_only else analysis["response_1d"][index]
    )
    return {
        "unit_id": unit_id, "response": maps[index].to_2d_array(),
        "response_1d": projected.to_1d_array("x"),
        **{key: analysis[key][index] for key in ("mask_2d", "mask_1d", "center_2d", "center_1d")},
    }


def plot_rf_unit(data, *, title=None, output_path=None, show=True):
    """Render supplied unit responses, RF masks, and centers."""
    response, response_1d = data["response"], data["response_1d"]
    mask_2d, mask_1d = data["mask_2d"], data["mask_1d"]
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.suptitle(title or f"Unit {data['unit_id']}")
        _heatmap(fig, axes[0, 0], response, "Response in detection window")
        _heatmap(fig, axes[0, 1], mask_2d, "2-D RF and center", binary=True)
        rows, columns = np.nonzero(data["center_2d"])
        axes[0, 1].scatter(columns, rows, marker="x", color="#b91c1c", s=60)
        axes[1, 0].plot(response_1d, color="black")
        columns = np.flatnonzero(mask_1d)
        axes[1, 0].scatter(columns, response_1d[columns], color="#b91c1c", label="1-D RF")
        axes[1, 0].set(title="Horizontal response", xlabel="Column", ylabel="Response")
        axes[1, 0].legend()
        _heatmap(fig, axes[1, 1], mask_1d[None, :], "1-D RF and center", binary=True)
        columns = np.flatnonzero(data["center_1d"])
        axes[1, 1].scatter(columns, np.zeros(columns.size), marker="x", color="#b91c1c", s=60)
        _finish_figure(fig, output_path, show)
    return fig, axes


def plot_rf_population_map(values, *, title="RF population", output_path=None, show=True):
    """Plot one prepared 2-D count map with its horizontal and vertical sums."""
    n_rows, n_columns = values.shape
    # Keep singleton-row maps at the same 30:7 display footprint as unit plots.
    map_ratio = n_rows / n_columns if n_rows > 1 else 7.0 / 30.0
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig = plt.figure(figsize=(11, 3.4 + 7.5 * map_ratio), layout="constrained")
        grid = fig.add_gridspec(
            2, 3, width_ratios=(1, 0.22, 0.025), height_ratios=(0.22, map_ratio),
        )
        ax = fig.add_subplot(grid[1, 0])
        top = fig.add_subplot(grid[0, 0], sharex=ax)
        right = fig.add_subplot(grid[1, 1], sharey=ax)
        colorbar = fig.add_subplot(grid[1, 2])
        axes = {"map": ax, "top": top, "right": right}
        fig.suptitle(title)
        image = ax.imshow(
            values, cmap="Greys", interpolation="nearest", aspect="auto", origin="upper",
            vmin=0, vmax=max(1, values.max()),
        )
        fig.colorbar(image, cax=colorbar, label="Unit count")
        ax.set(xlabel="Column", ylabel="Row")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # These sums count occupied unit-bins, not distinct units or 1-D detections.
        x_counts, y_counts = values.sum(axis=0), values.sum(axis=1)
        columns, rows = np.arange(n_columns), np.arange(n_rows)
        top.fill_between(columns, x_counts, color="0.85")
        top.plot(columns, x_counts, color="black", marker="o", markersize=3)
        top.set(ylabel="Sum over rows", ylim=(0, None))
        top.tick_params(axis="x", bottom=False, labelbottom=False)
        top.spines[["top", "right"]].set_visible(False)
        right.fill_betweenx(rows, y_counts, color="0.85")
        right.plot(y_counts, rows, color="black", marker="o", markersize=3)
        right.set(xlabel="Sum over columns", xlim=(0, None))
        right.tick_params(axis="y", left=False, labelleft=False)
        right.spines[["top", "right"]].set_visible(False)
        ax.set(xlim=(-0.5, n_columns - 0.5), ylim=(n_rows - 0.5, -0.5))
        _finish_figure(fig, output_path, show)
    return fig, axes


def rf_population_counts(analyses_by_probe, *, rf_type="excitatory"):
    """Count masks and centers for each probe and their pooled population."""
    keys = ("mask_2d", "center_2d", "mask_1d", "center_1d")
    counts_by_probe = {
        probe: {key: analyses[rf_type][key].sum(axis=0, dtype=np.int64) for key in keys}
        for probe, analyses in analyses_by_probe.items()
    }
    counts_all = {
        key: np.sum([counts[key] for counts in counts_by_probe.values()], axis=0)
        for key in keys
    }
    return {**{f"Probe{probe}": counts for probe, counts in counts_by_probe.items()}, "all": counts_all}


def smooth_rf_population(
    counts_by_panel, *, x_positions, y_positions, sigma_deg=6.0, center_sigma_deg=12.0,
    wrap_x=True,
):
    """Prepare Gaussian-smoothed 2-D display counts; preserve the supplied raw arrays.

    RF area and centers use separate sigmas in degrees on each native,
    uniformly spaced axis (6 and 12 degrees by default). Reflect at the
    vertical boundaries and wrap horizontal angles when requested. Both boundary
    modes preserve the total count. Singleton axes have no smoothing dimension.
    """
    sigmas = {"mask_2d": sigma_deg, "center_2d": center_sigma_deg}
    if any(not np.isfinite(value) or value < 0 for value in sigmas.values()):
        raise ValueError("Smoothing sigmas must be finite and nonnegative")
    bins_per_degree = []
    for positions in (y_positions, x_positions):
        positions = np.asarray(positions, dtype=float)
        if positions.size == 1:
            bins_per_degree.append(0.0)
            continue
        spacing = np.diff(positions)
        if spacing[0] == 0 or not np.allclose(spacing, spacing[0]):
            raise ValueError("Population smoothing requires uniformly spaced native angles")
        bins_per_degree.append(1.0 / abs(spacing[0]))
    return {
        name: {
            **counts,
            **{
                key: gaussian_filter(
                    counts[key].astype(float), sigma=[sigmas[key] * scale for scale in bins_per_degree],
                    mode=("reflect", "wrap" if wrap_x else "reflect"),
                )
                for key in ("mask_2d", "center_2d")
            },
            "smoothing_sigma_deg": sigmas,
        }
        for name, counts in counts_by_panel.items()
    }


def plot_rf_population(counts_by_panel, *, rf_type="excitatory", output_dir=None, show=True):
    """Render RF and RF center maps with marginals summed from each supplied 2-D array.

    Return ``figures[panel][label] = (figure, axes)`` with map/top/right axes.
    Explicitly smoothed panels retain their supplied values.
    """
    figures = {}
    for name, counts in counts_by_panel.items():
        figures[name] = {}
        # A single panel belongs directly to the caller's probe/kind directory.
        panel_dir = output_dir
        if output_dir is not None and len(counts_by_panel) > 1:
            panel_dir = Path(output_dir) / name
        for key, label, filename in (("mask_2d", "RF", "population"), ("center_2d", "RF center", "centers")):
            output_path = (
                _figure_dir(panel_dir, rf_type) / filename
                if output_dir is not None else None
            )
            title = f"{name} · {rf_type} · {label}"
            sigma_deg = counts.get("smoothing_sigma_deg", {}).get(key, 0)
            if sigma_deg:
                title += f" · Gaussian σ = {sigma_deg:g}°"
            figures[name][label] = plot_rf_population_map(
                counts[key], title=title, output_path=output_path, show=show,
            )
    return figures


def _initialize_plot_worker():
    matplotlib.use("Agg")


def _export_rf_unit(data, title, output_path):
    plot_rf_unit(data, title=title, output_path=output_path, show=False)


def export_rf_units(
    analyses_by_probe, output_dir=None, *, rf_type="excitatory", rf_only=False,
    workers=1, show_progress=True,
):
    """Export PDFs beside each organized source, or under an explicit custom root."""
    if workers < 1:
        raise ValueError("workers must be at least 1")
    jobs = []
    for probe, analyses in analyses_by_probe.items():
        analysis = analyses[rf_type]
        root = (
            organized_rf_source(analysis["source_path"]).parent
            if output_dir is None else Path(output_dir)
        )
        if output_dir is not None and len(analyses_by_probe) > 1:
            root = root / f"Probe{probe}"
        figure_dir = _figure_dir(root, rf_type)
        for unit_id in analysis["summed"].unit_ids:
            data = rf_unit_plot_data(analysis, unit_id, rf_only=rf_only)
            jobs.append((
                data, f"Probe{probe} · Unit {unit_id} · {rf_type} RF",
                figure_dir / f"unit_{unit_id}",
            ))
    with tqdm(total=len(jobs), desc=f"RF plots ({rf_type})", unit="unit",
              disable=not show_progress) as progress:
        if workers == 1:
            for job in jobs:
                _export_rf_unit(*job)
                progress.update()
        else:
            # Each process owns its Matplotlib state; send only prepared unit arrays.
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize_plot_worker,
            ) as pool:
                futures = [pool.submit(_export_rf_unit, *job) for job in jobs]
                for future in as_completed(futures):
                    future.result()
                    progress.update()
    return len(jobs)


def plot_2d_rfmap(
    data: np.ndarray,
    *,
    cmap: str = "viridis",
    is_save: bool = False,
    save_path: str = "rfmap.png",
):
    """Plot a 2D array as a heatmap without changing its orientation.

    The first array dimension is shown vertically (rows), and the second
    dimension is shown horizontally (columns). A singleton y row keeps the
    GUI's 30:7 spatial-map footprint instead of rendering as a thin strip.
    """
    with plt.rc_context(LIGHT_PLOT_STYLE):
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be a 2D array.")
        if 0 in data.shape:
            raise ValueError("data must not have an empty dimension.")

        n_rows, n_columns = data.shape
        fig, ax = plt.subplots()
        image_aspect: str | float = "equal"
        if n_rows == 1:
            image_aspect = n_columns * 7.0 / 30.0
        image = ax.imshow(
            data,
            aspect=image_aspect,
            cmap=cmap,
            interpolation="nearest",
        )

        ax.set_xticks(np.arange(n_columns))
        ax.set_yticks(np.arange(n_rows))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        if is_save:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        return fig, ax


def plot_1d_rfmap(unitsSpikeCounts: np.ndarray, label_list, *, isLineplot: bool = False,
                  isHeatmap: bool = False, offset: float = 1.0, xinDeg: bool = False):
    """Render supplied RF curves with the shared tuning-curve plotter."""
    from Utils.plotting import plot_tuning_curves_for_cluster

    return plot_tuning_curves_for_cluster(
        unitsSpikeCounts, label_list, isLineplot=isLineplot, isHeatmap=isHeatmap,
        offset=offset, xinDeg=xinDeg,
    )
