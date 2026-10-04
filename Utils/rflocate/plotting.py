"""Render RF response maps, detected fields, and prepared population arrays."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from Utils.plotting import LIGHT_PLOT_STYLE

__all__ = [
    "plot_2d_rfmap", "plot_1d_rfmap", "rf_unit_plot_data", "rf_population_counts",
    "plot_rf_unit", "plot_rf_population", "export_rf_units",
]


def _finish_figure(fig, output_path, show):
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".svg"):
            fig.savefig(
                output_path.with_suffix(suffix), dpi=300, bbox_inches="tight",
                facecolor="white", transparent=False,
            )
    if show:
        plt.show()
    plt.close(fig)


def _figure_dir(output_dir, rf_type):
    return Path(output_dir) / ("inhibitory/rfmap" if rf_type == "inhibitory" else "rfmap")


def _heatmap(fig, ax, values, title, *, binary=False):
    n_rows, n_columns = values.shape
    aspect = n_columns * 7.0 / 30.0 if n_rows == 1 else "equal"
    limits = dict(vmin=0, vmax=1) if binary else {}
    image = ax.imshow(values, cmap="Greys", interpolation="nearest", aspect=aspect, **limits)
    ax.set(title=title, xlabel="Column", ylabel="Row")
    fig.colorbar(image, ax=ax, label="RF mask" if binary else "Response")


def rf_unit_plot_data(analysis, unit_id, *, rf_only=False):
    """Prepare the horizontal response, optionally using only detected RF rows."""
    maps = analysis["summed"]
    index = maps.unit_ids.index(unit_id)
    projected = maps[index].sum_to_1d(
        "x", rf_only=rf_only,
        detected_rf=analysis["result_2d"] if rf_only else None,
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


def _population_figure(counts, title, output_path, show):
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.suptitle(title)
        for column, key, label in ((0, "mask_2d", "2-D RF area"), (1, "center_2d", "2-D RF centers")):
            _heatmap(fig, axes[0, column], counts[key], label)
            axes[0, column].images[0].set_clim(0, max(1, counts[key].max()))
            axes[0, column].images[0].colorbar.set_label("Unit count")
        for column, key, label in ((0, "mask_1d", "1-D RF area"), (1, "center_1d", "1-D RF centers")):
            axes[1, column].plot(counts[key], color="black")
            axes[1, column].set(title=label, xlabel="Column", ylabel="Unit count")
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


def plot_rf_population(counts_by_panel, *, rf_type="excitatory", output_dir=None, show=True):
    """Render supplied population arrays, including any explicitly smoothed panel."""
    figures = {}
    for name, counts in counts_by_panel.items():
        output_path = (
            _figure_dir(output_dir, rf_type) / name / "rf_summary"
            if output_dir is not None else None
        )
        figures[name] = _population_figure(counts, f"{name} · {rf_type} RF", output_path, show)
    return figures


def export_rf_units(analyses_by_probe, output_dir, *, rf_type="excitatory", rf_only=False):
    """Export one combined 2-D/1-D figure per unit without inline display."""
    count = 0
    output_dir = _figure_dir(output_dir, rf_type)
    for probe, analyses in analyses_by_probe.items():
        analysis = analyses[rf_type]
        for unit_id in analysis["summed"].unit_ids:
            data = rf_unit_plot_data(analysis, unit_id, rf_only=rf_only)
            plot_rf_unit(
                data, title=f"Probe{probe} · Unit {unit_id} · {rf_type} RF",
                output_path=Path(output_dir) / "units" / probe / str(unit_id), show=False,
            )
            count += 1
    return count


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
