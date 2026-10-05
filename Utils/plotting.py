from collections.abc import Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import PowerNorm

from Utils.statistic_utils import circular_distance_matrix_deg


LIGHT_PLOT_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.transparent": False,
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "#d1d5db",
    "legend.facecolor": "white",
    "legend.edgecolor": "black",
    "legend.framealpha": 1.0,
}


def apply_light_plot_style() -> None:
    plt.rcParams.update(LIGHT_PLOT_STYLE)


apply_light_plot_style()


def angular_ticks(ticklabels):
    """Unwrap circular degree labels into one continuous display range."""
    labels = np.asarray(ticklabels, dtype=float)
    ticks = np.unwrap(labels, period=360)
    return ticks - 360 * np.round(ticks[len(ticks) // 2] / 360)


def plot_keyed_heatmap(
    row_by_key, unit_key_sequence, *, column_order, xticks, xticklabels,
    xlabel="Direction (°)", title=None,
    show=True, empty_message="No units to plot.", angle_centers=None, figsize=(10, 6),
    cmap="viridis", vmin=None, vmax=None, colorbar_label="Response",
):
    """Plot keyed angular profiles in the supplied row and column order.

    Keys may be unit IDs or tuples identifying the recording, probe, and unit.
    Values are displayed as supplied, with missing bins kept blank. Normalization,
    filtering, and peak sorting belong to the caller.
    Tick endpoints define the displayed angular extent;
    labels may show a different angular convention at those same positions.
    Every row is labeled with its unit key; figsize controls the figure size.
    """
    n_units = len(unit_key_sequence)
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        if n_units:
            data = np.stack([row_by_key[key] for key in unit_key_sequence])
            data = data[:, column_order].astype(float)
            extent = [xticks[0], xticks[-1], n_units - 0.5, -0.5]
            if angle_centers is not None:
                step = angle_centers[1] - angle_centers[0]
                extent[:2] = [angle_centers[0] - step / 2, angle_centers[-1] + step / 2]
            image = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,
                interpolation="nearest",
                origin="upper",
                extent=extent,
                vmin=vmin, vmax=vmax,
            )
            ax.set_yticks(range(n_units))
            ax.set_yticklabels([
                ":".join(map(str, key)) if isinstance(key, tuple) else str(key)
                for key in unit_key_sequence
            ])
            ax.set_xticks(xticks, labels=xticklabels)
            ax.set_xlim(xticks[0], xticks[-1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Unit ID")
            fig.colorbar(image, ax=ax, label=colorbar_label)
        else:
            ax.text(0.5, 0.5, empty_message, ha="center", va="center")
            ax.set_axis_off()
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        if show:
            plt.show()
    return fig, ax


def plot_direction_comparison(
    reference_deg, matched_deg, statistics, *, reference_label, matched_label,
    reference_ticklabels, matched_ticklabels, angle_ticks=None,
    reference_ticks=None, matched_ticks=None, density_bin_count=16, show=True,
):
    """Plot source directions, deriving each axis from its degree labels."""
    if reference_ticks is None:
        reference_ticks = angular_ticks(reference_ticklabels) if angle_ticks is None else angle_ticks
    if matched_ticks is None:
        matched_ticks = angular_ticks(matched_ticklabels) if angle_ticks is None else angle_ticks
    reference_start = min(reference_ticks[0], reference_ticks[-1])
    matched_start = min(matched_ticks[0], matched_ticks[-1])
    reference_display = (np.asarray(reference_deg) - reference_start) % 360 + reference_start
    matched_display = (np.asarray(matched_deg) - matched_start) % 360 + matched_start
    same_unit = statistics["same_unit"]
    pairwise = statistics["pairwise"]
    annotation = {
        "ha": "left", "va": "top", "color": "black",
        "bbox": {"boxstyle": "round", "facecolor": "white", "alpha": 1.0},
    }
    with plt.rc_context(LIGHT_PLOT_STYLE):
        scatter_figure, scatter_axis = plt.subplots(figsize=(6, 6))
        scatter_axis.scatter(reference_display, matched_display, s=28, alpha=0.8)
        scatter_axis.set(
            xlabel=f"{reference_label} peak direction (deg)",
            ylabel=f"{matched_label} peak direction (deg)",
            title=f"Same-unit {reference_label} and {matched_label} peak directions",
            xlim=(reference_ticks[0], reference_ticks[-1]),
            ylim=(matched_ticks[0], matched_ticks[-1]),
            xticks=reference_ticks, yticks=matched_ticks,
        )
        scatter_axis.set_xticklabels(reference_ticklabels)
        scatter_axis.set_yticklabels(matched_ticklabels)
        scatter_axis.text(
            0.03, 0.97,
            f"Circular-circular rho = {same_unit['circular_correlation_rho']:.3f}\n"
            f"p = {same_unit['permutation_p']:.4g}",
            transform=scatter_axis.transAxes, **annotation,
        )
        scatter_axis.set_aspect("equal", adjustable="box")
        scatter_axis.grid(color="0.85", linewidth=0.8)
        scatter_figure.tight_layout()
        if show:
            plt.show()

        # Retain the original density plot's ordered pairs, including its diagonal.
        reference_distances = circular_distance_matrix_deg(reference_deg).ravel()
        matched_distances = circular_distance_matrix_deg(matched_deg).ravel()
        # Sixteen bins center on 0, 12, ..., 180 degrees, matching 30-bin tuning curves.
        bin_width = 180.0 / (density_bin_count - 1)
        bin_edges = np.linspace(-bin_width / 2, 180.0 + bin_width / 2, density_bin_count + 1)
        density_figure, density_axis = plt.subplots(figsize=(8, 8))
        image = density_axis.hist2d(
            reference_distances, matched_distances, bins=(bin_edges, bin_edges),
            cmap="viridis", norm=PowerNorm(gamma=0.5, vmin=0),
        )[3]
        density_figure.colorbar(image, ax=density_axis, pad=0.02, label="Pair count")
        density_axis.plot(
            [0, 180], [0, 180], linestyle="--", linewidth=1.2,
            color="white", label="Equal pair distance",
        )
        density_axis.legend(loc="lower right")
        density_axis.set(
            xlabel=f"Pairwise {reference_label} peak-direction distance (deg)",
            ylabel=f"Pairwise {matched_label} peak-direction distance (deg)",
            title=f"{reference_label} × {matched_label}: pair distances",
            xlim=(-5, 185), ylim=(-5, 185),
            xticks=np.arange(0, 181, 45), yticks=np.arange(0, 181, 45),
        )
        density_axis.text(
            0.03, 0.97,
            f"Mantel ρ = {pairwise['spearman_mantel_rho']:.3f}\n"
            f"p = {pairwise['permutation_p']:.4g}\n"
            f"n = {pairwise['unit_count']}",
            transform=density_axis.transAxes, **annotation,
        )
        density_axis.set_aspect("equal", adjustable="box")
        density_figure.tight_layout()
        if show:
            plt.show()
    return (scatter_figure, scatter_axis), (density_figure, density_axis)


def get_tuning_curve_for_cluster(
        tuning_curves,
        cluster_id: int,
) -> pd.Series:
    """
    Return a single tuning curve as a pandas Series indexed by angle in degrees.

    Supports both:
    - xarray.DataArray with dims ('unit', angle_dim) or (angle_dim, 'unit')
    - pandas DataFrame with angle index and unit columns
    """
    cluster_id = int(cluster_id)

    if hasattr(tuning_curves, "sel") and hasattr(tuning_curves, "dims") and hasattr(tuning_curves, "coords"):
        if "unit" not in tuning_curves.dims:
            raise ValueError("xarray tuning_curves must contain a 'unit' dimension.")

        angle_dims = [dim for dim in tuning_curves.dims if dim != "unit"]
        if len(angle_dims) != 1:
            raise ValueError(
                "xarray tuning_curves must have exactly one non-'unit' dimension."
            )

        angle_dim = angle_dims[0]
        curve = tuning_curves.sel(unit=cluster_id)
        angles_deg = np.asarray(curve.coords[angle_dim].values, dtype=float)
        rates = np.asarray(curve.values, dtype=float)
        return pd.Series(rates, index=angles_deg, name=cluster_id, dtype=float)

    if isinstance(tuning_curves, pd.Series):
        curve = tuning_curves.copy()
        curve.name = cluster_id if curve.name is None else curve.name
        return curve.astype(float)

    if isinstance(tuning_curves, pd.DataFrame):
        candidate_keys = [cluster_id, str(cluster_id)]
        for key in candidate_keys:
            if key in tuning_curves.columns:
                curve = tuning_curves[key]
                return pd.Series(
                    np.asarray(curve.values, dtype=float),
                    index=np.asarray(tuning_curves.index.values, dtype=float),
                    name=cluster_id,
                    dtype=float,
                )
        raise KeyError(f"cluster_id {cluster_id} not found in tuning_curves columns.")

    raise TypeError(
        "tuning_curves must be an xarray DataArray, pandas DataFrame, or pandas Series."
    )


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_hd_tuning_curve(
        tuning_curves,
        cluster_id: int,
        *,
        is_return_line_plot: bool = True,
        is_return_polar_plot: bool = False,
        clockwise: bool = True,
        line_figsize: tuple[float, float] = (7.0, 4.0),
        polar_figsize: tuple[float, float] = (6.0, 6.0),
        color: str = "C0",
        fill_polar: bool = True,
        ring_radii: Sequence[int | float] = (),
        line_label: str | None = None,
        title_prefix: str = "HD tuning curve",
        is_save: bool = False,
):
    """
    Plot one cluster's tuning curve as a line plot, polar plot, or both.

    Supply already prepared curves and optional reference-ring radii. Missing
    bins stay in place. Returns the extracted curve and figure/axes objects.
    """
    if not is_return_line_plot and not is_return_polar_plot:
        raise ValueError(
            "At least one of is_return_line_plot or is_return_polar_plot must be True."
        )

    curve = get_tuning_curve_for_cluster(tuning_curves, cluster_id)

    angles_deg = curve.index.to_numpy(dtype=float)
    rates = curve.to_numpy(dtype=float)

    result = {
        "curve": curve,
        "angles_deg": angles_deg,
        "rates": rates,
        "line_fig": None,
        "line_ax": None,
        "polar_fig": None,
        "polar_ax": None,
    }

    if is_return_line_plot:
        line_fig, line_ax = plt.subplots(figsize=line_figsize)
        line_ax.plot(angles_deg, rates, color=color, label=line_label or f"cluster {cluster_id}")
        line_ax.set_xlim(0.0, 360.0)
        line_ax.set_xlabel("Head direction (deg)")
        line_ax.set_ylabel("Firing rate (Hz)")
        line_ax.set_title(f"{title_prefix} - cluster {cluster_id}")
        if line_label is not None:
            line_ax.legend()
        line_fig.tight_layout()
        result["line_fig"] = line_fig
        result["line_ax"] = line_ax

    if is_return_polar_plot:
        polar_fig = plt.figure(figsize=polar_figsize)
        polar_ax = polar_fig.add_subplot(111, polar=True)

        angles_rad = np.deg2rad(angles_deg)
        if angles_rad.size > 0:
            plot_angles_rad = np.concatenate([angles_rad, angles_rad[:1]])
            plot_rates = np.concatenate([rates, rates[:1]])
        else:
            plot_angles_rad = angles_rad
            plot_rates = rates

        polar_ax.plot(plot_angles_rad, plot_rates, color=color, linewidth=2)
        if fill_polar:
            polar_ax.fill(plot_angles_rad, plot_rates, alpha=0.3, color=color)
        polar_ax.set_rticks([])

        theta_full = np.linspace(0.0, 2.0 * np.pi, 361)
        label_theta = np.deg2rad(6.0)

        seen_radii: list[float] = []
        for radius in ring_radii:
            if not np.isfinite(radius):
                continue
            if any(np.isclose(radius, seen) for seen in seen_radii):
                continue
            seen_radii.append(radius)
            polar_ax.plot(
                theta_full,
                np.full_like(theta_full, radius),
                linestyle="--",
                linewidth=1,
                color="0.5",
            )
            polar_ax.text(
                label_theta,
                radius,
                f"{radius:.2f}",
                ha="left",
                va="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.5),
            )

        polar_ax.set_theta_zero_location("N")

        polar_ax.set_theta_direction(-1) if clockwise else polar_ax.set_theta_direction(1)

        polar_ax.set_title(f"{title_prefix} - cluster {cluster_id}", va="bottom")

        if is_save:
            polar_fig.savefig(f"{title_prefix} - cluster {cluster_id}.png")

        result["polar_fig"] = polar_fig
        result["polar_ax"] = polar_ax


    return result


def plot_tuning_curves_for_cluster(
    unitsSpikeCounts: np.ndarray, targetList: list[int], *, isLineplot: bool = False,
    isHeatmap: bool = False, offset: float = 1.0, xinDeg: bool = False,
    plotSize: tuple[int, int] = (10, 8),
):
    """Render supplied tuning curves; normalize or transform before calling."""
    with plt.rc_context(LIGHT_PLOT_STYLE):
        if isLineplot == isHeatmap:
            raise ValueError("Exactly one of isLineplot or isHeatmap must be True.")

        if unitsSpikeCounts.ndim == 1:
            unitsSpikeCounts = unitsSpikeCounts[np.newaxis, :]

        n_units, n_x = unitsSpikeCounts.shape

        if len(targetList) != n_units:
            raise ValueError("targetList must have one label per row in unitsSpikeCounts.")

        x_values = np.linspace(0, 360, n_x, endpoint=False) if xinDeg else np.arange(n_x)
        x_label = "Angle (deg)" if xinDeg else "x"

        fig, ax = plt.subplots(figsize=plotSize)

        if isLineplot:
            yticks_height = []

            for unit_idx, spikeCounts in enumerate(unitsSpikeCounts):
                y = spikeCounts + (n_units - 1 - unit_idx) * offset
                yticks_height.append(np.average(y))
                ax.plot(x_values, y, linewidth=1)

            ax.set_yticks(yticks_height)
            ax.set_yticklabels(targetList)

        if isHeatmap:
            imshow_kwargs = dict(
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )
            if xinDeg:
                imshow_kwargs["extent"] = [0, 360, n_units - 0.5, -0.5]

            im = ax.imshow(unitsSpikeCounts, **imshow_kwargs)

            ax.set_yticks(np.arange(n_units))
            ax.set_yticklabels(targetList)
            fig.colorbar(im, ax=ax, label="Response")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Unit ID")
        if xinDeg:
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))

        plt.tight_layout()
        plt.show()
        return fig, ax


def _get_phase_colors(labels: list[str], color_list: list[str] | None) -> list[str]:
    if color_list is not None and len(color_list) != len(labels):
        raise ValueError("color_list must match the number of labels.")

    default_cycle = plt.rcParams.get("axes.prop_cycle", None)
    default_colors = (
        default_cycle.by_key().get("color", [])
        if default_cycle is not None else []
    )
    if not default_colors:
        default_colors = [f"C{i}" for i in range(max(len(labels), 1))]

    if color_list is None:
        if len(default_colors) < len(labels):
            default_colors.extend(
                f"C{i}" for i in range(len(default_colors), len(labels))
            )
        return default_colors[:len(labels)]

    return list(color_list)


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_phase_bar(
        summary: dict[str, object],
        *,
        is_show_up_error_bar: bool = True,
        is_show_down_error_bar: bool = True,
        is_show_per_trial_data: bool = True,
        color_list: list[str] | None = None,
        legend: bool = False,
        title: str | None = None,
        ylabel: str | None = None,
        is_show_grid: bool = True,
        is_save: str | None = None,
) -> tuple:
    """Render prepared trial values, means, and standard deviations unchanged."""
    labels = summary["labels"]
    phase_trial_values = summary["phase_trial_values"]

    if len(labels) != len(phase_trial_values):
        raise ValueError("labels and phase_trial_values must have the same length.")
    phase_colors = _get_phase_colors(labels, color_list)
    phase_arrays = [np.asarray(values, dtype=float) for values in phase_trial_values]

    means = summary["phase_means"]
    stds = np.asarray(summary["phase_stds"], dtype=float)
    x_positions = np.arange(len(labels)) * 0.5

    fig_width = max(5.0, len(labels) * 1.25 + 1.5)
    figure, axis = plt.subplots(figsize=(fig_width, 5.0))

    lower = stds if is_show_down_error_bar else np.zeros_like(stds)
    upper = stds if is_show_up_error_bar else np.zeros_like(stds)
    yerr = np.vstack([lower, upper])

    axis.bar(
        x_positions,
        means,
        width=0.4,
        yerr=yerr,
        capsize=0,
        color=phase_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.35,
        zorder=1,
    )

    cap_halfwidth = 0.04
    finite_means = np.asarray(means, dtype=float)
    valid_mask = np.isfinite(finite_means)
    if is_show_up_error_bar and np.any(valid_mask):
        axis.hlines(
            finite_means[valid_mask] + upper[valid_mask],
            x_positions[valid_mask] - cap_halfwidth,
            x_positions[valid_mask] + cap_halfwidth,
            color="black",
            linewidth=0.8,
            zorder=4,
        )
    if is_show_down_error_bar and np.any(valid_mask):
        axis.hlines(
            finite_means[valid_mask] - lower[valid_mask],
            x_positions[valid_mask] - cap_halfwidth,
            x_positions[valid_mask] + cap_halfwidth,
            color="black",
            linewidth=0.8,
            zorder=4,
        )

    if is_show_per_trial_data:
        for center, phase_values, color in zip(x_positions, phase_arrays, phase_colors):
            if phase_values.size == 0:
                continue
            jitter_width = 0.08
            jitter = (
                np.linspace(-jitter_width, jitter_width, phase_values.size)
                if phase_values.size > 1 else np.array([0.0])
            )
            axis.scatter(
                np.full(phase_values.size, center) + jitter,
                phase_values,
                color=color,
                edgecolor="black",
                linewidth=0.7,
                s=42,
                alpha=1.0,
                zorder=3,
            )

    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    if title:
        axis.set_title(title)
    if ylabel:
        axis.set_ylabel(ylabel)
    if is_show_grid:
        axis.grid(alpha=0.25, axis="y")
    else:
        axis.grid(False)
    axis.margins(x=0.08)

    if legend:
        legend_handles = [
            plt.Line2D([0], [0], color=color, marker="s", linestyle="", markersize=9)
            for color in phase_colors
        ]
        axis.legend(legend_handles, labels, loc="upper left")

    plt.tight_layout()
    if is_save is not None:
        plt.savefig(is_save)
    plt.show()

    return figure, axis


def plot_head_turn_bar(summary, *, ylabel=None, **plot_options):
    """Render a head-turn summary without reading events or changing trial values."""
    if ylabel is None:
        if summary["has_target_data"]:
            ylabel = "Mean target value during head turns"
        else:
            ylabel = "Head turn count" if summary["value_mode"] == "count" else "Head turn count/s"
    return plot_phase_bar(summary, ylabel=ylabel, **plot_options)


def _plot_heatmap(
    axis,
    horizontal_edges,
    vertical_edges,
    values,
    title,
    colorbar_label,
    *,
    cmap="viridis",
):
    image = axis.pcolormesh(
        horizontal_edges,
        vertical_edges,
        values,
        shading="auto",
        cmap=cmap,
    )
    axis.set_title(title)
    axis.figure.colorbar(image, ax=axis, label=colorbar_label)
    return image


def plot_egocentric_heatmap(
    axis,
    distance_edges,
    theta_edges,
    values,
    title,
    colorbar_label,
    *,
    cmap="viridis",
):
    image = _plot_heatmap(
        axis,
        distance_edges,
        theta_edges,
        values,
        title,
        colorbar_label,
        cmap=cmap,
    )
    axis.set_xlabel("Distance to boundary (cm)")
    axis.set_ylabel("Egocentric theta")
    axis.set_yticks([0, 90, 180, 270, 360])
    axis.set_yticklabels(
        ["0°", "90°", "180°", "270°", "360°"]
    )
    return image


def plot_allocentric_heatmap(
    axis,
    x_edges,
    y_edges,
    values,
    title,
    colorbar_label,
    *,
    cmap="viridis",
):
    image = _plot_heatmap(
        axis,
        x_edges,
        y_edges,
        values.T,
        title,
        colorbar_label,
        cmap=cmap,
    )
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    return image


def plot_trajectory_spikes(
    axis,
    trajectory_x_cm,
    trajectory_y_cm,
    spike_x_cm,
    spike_y_cm,
    *,
    x_limits,
    y_limits,
):
    axis.plot(
        trajectory_x_cm,
        trajectory_y_cm,
        color="0.65",
        linewidth=0.35,
        label="Trajectory",
    )
    axis.scatter(
        spike_x_cm,
        spike_y_cm,
        s=2,
        color="red",
        alpha=0.3,
        linewidths=0,
        label="Spikes",
    )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_xlabel("x (cm)")
    axis.set_ylabel("y (cm)")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_title("Trajectory and spike positions")
    axis.legend(loc="upper right")


def plot_egocentric_polar(
    axis,
    theta_edges,
    distance_edges,
    rate_map,
    *,
    hole_fraction=None,
    colorbar_label="Hz",
):
    """Plot angle-by-distance rates with 0 degrees forward and 90 degrees left.

    ``hole_fraction`` sets the donut hole radius as a fraction of the outer
    radius. It changes only the radial origin; bin edges and cm labels keep
    their saved values. None uses the physical distance origin at zero.
    """
    if hole_fraction is not None and not 0 <= hole_fraction < 1:
        raise ValueError("hole_fraction must be between 0 (inclusive) and 1 (exclusive).")

    axis.grid(False)
    image = axis.pcolormesh(
        np.deg2rad(theta_edges),
        distance_edges,
        rate_map.T,
        shading="auto",
        cmap="viridis",
    )
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(1)
    axis.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axis.set_xticklabels(["0°", "90°", "180°", "270°"])
    inner, outer = distance_edges[0], distance_edges[-1]
    axis.set_ylim(0, outer)
    axis.set_rorigin(0)
    if outer > inner:
        if hole_fraction is not None:
            # (inner - origin) / (outer - origin) is the visible hole fraction.
            axis.set_ylim(inner, outer)
            axis.set_rorigin((inner - hole_fraction * outer) / (1 - hole_fraction))
        ticks = np.linspace(inner, outer, 5)
        axis.set_yticks(ticks, labels=[f"{value:.1f}".rstrip("0").rstrip(".") for value in ticks])
    else:
        axis.text(0.5, 0.5, "No distance bins in this band.", ha="center", transform=axis.transAxes)
    axis.set_rlabel_position(135)
    for label in axis.get_yticklabels():
        label.set_bbox(dict(facecolor="white", edgecolor="none", pad=1))
    axis.grid(True, linewidth=0.5)
    axis.set_xlabel("Distance to boundary (cm)", labelpad=18)
    axis.set_title("Egocentric firing-rate map (polar)")
    axis.figure.colorbar(image, ax=axis, label=colorbar_label, pad=0.12)
    return image


def plot_tuning_curve(axis, theta_edges, firing_rate, title):
    theta_centers = np.deg2rad(
        (theta_edges[:-1] + theta_edges[1:]) / 2
    )
    theta = np.r_[theta_centers, theta_centers[0] + 2 * np.pi]
    rate = np.r_[firing_rate, firing_rate[0]]

    line = axis.plot(theta, rate, linewidth=2)[0]
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(1)
    axis.set_xticks(np.deg2rad([0, 90, 180, 270]))
    axis.set_xticklabels(["0", "90", "180", "270"])
    axis.set_ylim(bottom=0)
    axis.set_title(title)
    return line
