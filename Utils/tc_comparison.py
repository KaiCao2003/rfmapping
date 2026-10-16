"""Prepare normalized HD/RF curves and calculate and plot paired comparisons."""

from textwrap import fill
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from Utils.direction_comparison import normalize_tc, prepare_comparison, plot_comparison_heatmaps, profile_coordinates
from Utils.plotting import LIGHT_PLOT_STYLE, plot_direction_comparison
from Utils.statistic_utils import compare_direction_angles
from Utils.statistic_utils import rayleigh_test


def calculate_tc_statistics(
    hd: pd.DataFrame, rf: pd.DataFrame, *, bins: int = 30, is_wrap: bool = True,
    n_permutations: int = 10_000, random_seed: int | None = 1,
) -> dict[str, Any]:
    """Keep the input TC bin counts; bins controls the statistical histograms."""
    units = hd.index.intersection(rf.index)
    hd, rf = hd.loc[units], rf.loc[units]
    comparisons = []
    for mode in ("native", "aligned", "sum"):
        comparisons.append(prepare_comparison(
            hd, rf, mode=mode, bins=hd.shape[1], is_wrap=is_wrap, matched_fill_value=0,
        ))
        comparisons.append(prepare_comparison(
            rf, hd, mode=mode, bins=rf.shape[1], is_wrap=is_wrap, reference_fill_value=0,
        ))
    peaks = comparisons[0]["peaks"].dropna(subset=["reference_peak_deg", "matched_peak_deg"])
    direction = None
    if len(peaks) >= 3:
        direction = compare_direction_angles(
            peaks.reference_peak_deg.to_numpy(), peaks.matched_peak_deg.to_numpy(),
            n_permutations=n_permutations, random_seed=random_seed,
        )
    peak_sum_rad = np.deg2rad((peaks.reference_peak_deg + peaks.matched_peak_deg) % 360)
    counts, edges = np.histogram(peak_sum_rad, bins=bins, range=(0, 2 * np.pi))
    centers_deg = np.rad2deg((edges[:-1] + edges[1:]) / 2)
    # Keep empty histogram bins and equal occupancy, as in the individual cells.
    rayleigh_score, rayleigh_p = rayleigh_test(counts, np.ones_like(counts), centers_deg)
    return {
        "comparisons": comparisons, "peaks": peaks, "direction": direction,
        "counts": counts, "edges": edges,
        "rayleigh_score": rayleigh_score, "rayleigh_p": rayleigh_p,
    }


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_tc_comparison(
    hd: pd.DataFrame, rf: pd.DataFrame, stats: dict[str, Any], *,
    figsize: tuple[float, float] = (10, 10), show: bool = True,
) -> list[tuple[Figure, Axes]]:
    """Draw prepared heatmaps, the peak-sum histogram, and direction statistics."""
    figures = []
    for comparison in stats["comparisons"]:
        figures.extend(plot_comparison_heatmaps(
            comparison, figsize=figsize, vmin=0, vmax=1,
            colorbar_label="Normalized response", show=False,
        ))

    counts, edges = stats["counts"], stats["edges"]
    figure, axis = plt.subplots(
        figsize=(6, 6), subplot_kw={"projection": "polar"}, layout="constrained",
    )
    axis.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges), edgecolor="white")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.set_xlabel("Unit count (radial axis)")
    axis.set_title(
        fill(f"{hd.attrs['label']} + {rf.attrs['label']} peak angles", width=60)
        + f"\nn = {int(counts.sum())}\n"
        + f"Rayleigh score = {stats['rayleigh_score']:.3f}, p = {stats['rayleigh_p']:.3g}",
        pad=20,
    )
    figures.append((figure, axis))

    if stats["direction"] is not None:
        peaks = stats["peaks"]
        direction_figures = plot_direction_comparison(
            peaks.reference_peak_deg.to_numpy(), peaks.matched_peak_deg.to_numpy(),
            stats["direction"],
            reference_label=hd.attrs["label"], matched_label=rf.attrs["label"],
            reference_ticklabels=profile_coordinates(hd)["ticklabels"],
            matched_ticklabels=profile_coordinates(rf)["ticklabels"],
            density_bin_count=len(counts), show=False,
        )
        for figure, axis in direction_figures:
            axis.set_title(fill(axis.get_title(), width=60))
            figure.tight_layout()
        figures.extend(direction_figures)

    if show:
        for figure, axis in figures:
            display(figure)
            plt.close(figure)
    return figures
