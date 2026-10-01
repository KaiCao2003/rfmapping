"""Unit previews and population figures for locate_rf.ipynb."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from Utils.plotting import LIGHT_PLOT_STYLE

__all__ = ["plot_rf_unit", "plot_rf_population", "export_rf_units"]


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


def plot_rf_unit(analysis, unit_id, *, title=None, output_path=None, show=True):
    """Show the response, 2-D RF and center, and independent 1-D RF together."""
    maps = analysis["summed"]
    index = maps.unit_ids.index(unit_id)
    response = maps[index].to_2d_array()
    response_1d = maps[index].to_1d_array("x")
    mask_2d = analysis["mask_2d"][index]
    mask_1d = analysis["mask_1d"][index]
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        fig.suptitle(title or f"Unit {unit_id}")
        _heatmap(fig, axes[0, 0], response, "Response in detection window")
        _heatmap(fig, axes[0, 1], mask_2d, "2-D RF and center", binary=True)
        rows, columns = np.nonzero(analysis["center_2d"][index])
        axes[0, 1].scatter(columns, rows, marker="x", color="#b91c1c", s=60)
        axes[1, 0].plot(response_1d, color="black")
        columns = np.flatnonzero(mask_1d)
        axes[1, 0].scatter(columns, response_1d[columns], color="#b91c1c", label="1-D RF")
        axes[1, 0].set(title="Horizontal response", xlabel="Column", ylabel="Response")
        axes[1, 0].legend()
        _heatmap(fig, axes[1, 1], mask_1d[None, :], "1-D RF and center", binary=True)
        columns = np.flatnonzero(analysis["center_1d"][index])
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


def plot_rf_population(analyses_by_probe, *, rf_type="excitatory", output_dir=None, show=True):
    """Plot a selected RF type from the notebook's dual-type analysis results."""
    keys = ("mask_2d", "center_2d", "mask_1d", "center_1d")
    counts_by_probe = {
        probe: {key: analyses[rf_type][key].sum(axis=0, dtype=np.int64) for key in keys}
        for probe, analyses in analyses_by_probe.items()
    }
    counts_all = {
        key: np.sum([counts[key] for counts in counts_by_probe.values()], axis=0)
        for key in keys
    }
    counts_smoothed = {
        key: gaussian_filter(values.astype(float), sigma=1.0, mode="nearest")
        for key, values in counts_all.items()
    }
    panels = [(f"Probe{probe}", counts) for probe, counts in counts_by_probe.items()]
    panels.extend((("all", counts_all), ("all_smoothed", counts_smoothed)))
    figures = {}
    for name, counts in panels:
        output_path = (
            _figure_dir(output_dir, rf_type) / name / "rf_summary"
            if output_dir is not None else None
        )
        figures[name] = _population_figure(counts, f"{name} · {rf_type} RF", output_path, show)
    return figures


def export_rf_units(analyses_by_probe, output_dir, *, rf_type="excitatory"):
    """Export one combined 2-D/1-D figure per unit without inline display."""
    count = 0
    output_dir = _figure_dir(output_dir, rf_type)
    for probe, analyses in analyses_by_probe.items():
        analysis = analyses[rf_type]
        for unit_id in analysis["summed"].unit_ids:
            plot_rf_unit(
                analysis, unit_id, title=f"Probe{probe} · Unit {unit_id} · {rf_type} RF",
                output_path=Path(output_dir) / "units" / probe / str(unit_id), show=False,
            )
            count += 1
    return count
