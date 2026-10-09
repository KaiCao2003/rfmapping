"""Prepare normalized HD/RF curves and calculate and plot paired comparisons."""

from collections.abc import Sequence
from pathlib import Path
from textwrap import fill
from typing import Any

import numpy as np
import pandas as pd
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from Utils import tc_preparation
from Utils.direction_comparison import (
    normalize_tc, prepare_comparison, plot_comparison_heatmaps,
    profile_coordinates, resample_profiles, rf_pick, tc_loader, tcRange,
)
from Utils.plotting import LIGHT_PLOT_STYLE, plot_direction_comparison
from Utils.rflocate import rf_result_path
from Utils.statistic_utils import compare_direction_angles
from Utils.tuning_curve_utils import rayleigh_test


def prep_hd_tc(
    mouseid: str, base_dir: str | Path, probe: str, date: str | int, sessionID: int, *,
    bins: int = 30, hd_class: int | None = 3, overwrite: bool = False,
) -> pd.DataFrame:
    """Return normalized HD curves; reuse the prepared CSV unless overwrite=True."""
    session_dir = Path(base_dir) / mouseid / str(date) / f"{date}_{sessionID}"
    output = session_dir / "data/tc_comparison" / f"hd_class{hd_class}_Probe{probe}.csv"
    tc_preparation.prepare_hd_tc(
        session_dir / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc",
        output, bins=bins, hd_class=hd_class,
        unit_prefix=f"{mouseid}:{date}:{probe}", overwrite=overwrite,
    )
    hd = tc_loader(
        output, label=f"{mouseid} {date} Probe{probe} HD",
        range=tcRange(False), response_units="Hz",
    )
    return normalize_tc(hd)


def prep_rf_tc(
    mouseid: str, base_dir: str | Path, probe: str, date: str | int, sessionID: int, *,
    bins: int = 30, rf_range: Sequence[float] = (-180, 180),
    time_range_s: tuple[float, float] = (0.0, 0.2), rf_only: bool = True,
    max_zero_bins: int | None = 2,
    rf_map_dir: str | Path = "rfmapping/good/-100_400_1ms", overwrite: bool = False,
) -> pd.DataFrame:
    """Return normalized excitatory RF x curves, filtering before resampling.

    Existing CSVs retain their saved settings unless overwrite=True.
    """
    session_dir = Path(base_dir) / mouseid / str(date) / f"{date}_{sessionID}"
    source = (session_dir / "data" / rf_map_dir / f"Probe{probe}"
              / f"regular_unitsSpikeCounts_{date}_{sessionID}.rfmap")
    suffix = "_rfonly" if rf_only else ""
    output = session_dir / "data/tc_comparison" / f"rf_excitatory_x_2d{suffix}_Probe{probe}.csv"
    tc_preparation.prepare_rf_comparison(
        source, output,
        projection_path=session_dir / "data" / f"{source.stem}_Probe{probe}_1d{suffix}.csv",
        detected_rf_path=rf_result_path(source, rf_type="excitatory"),
        time_range_s=time_range_s, rf_only=rf_only,
        unit_prefix=f"{mouseid}:{date}:{probe}", overwrite=overwrite,
    )
    rf = tc_loader(
        output, label=f"{mouseid} {date} Probe{probe} RF{sessionID}",
        response_units="spike_count",
    )
    rf = rf_pick(rf, max_zero_bins=max_zero_bins)
    rf = resample_profiles(rf, range=rf_range, bins=bins, fill_value=0)
    return normalize_tc(rf)


def calculate_tc_statistics(
    hd: pd.DataFrame, rf: pd.DataFrame, *, bins: int = 30, is_wrap: bool = True,
    n_permutations: int = 10_000, random_seed: int | None = 1,
) -> dict[str, Any]:
    """Match units and prepare both reference orders, peak statistics, and counts."""
    units = hd.index.intersection(rf.index)
    hd, rf = hd.loc[units], rf.loc[units]
    comparisons = []
    for mode in ("native", "aligned", "sum"):
        comparisons.append(prepare_comparison(
            hd, rf, mode=mode, is_wrap=is_wrap, matched_fill_value=0,
        ))
        comparisons.append(prepare_comparison(
            rf, hd, mode=mode, is_wrap=is_wrap, reference_fill_value=0,
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
            matched_ticklabels=profile_coordinates(rf)["ticklabels"], show=False,
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
