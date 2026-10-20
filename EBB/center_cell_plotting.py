"""Plot prepared center-cell tuning, classification, and behavior results."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from Utils.plotting import LIGHT_PLOT_STYLE, plot_egocentric_polar, plot_tuning_curve


_COLORS = {
    "raw": "#667085",
    "main": "#167D9A",
    "first": "#007F5F",
    "second": "#D47718",
    "fit": "#9F3B66",
    "missing": "#E9EDF2",
}


_PLOT_STYLE = {
    **LIGHT_PLOT_STYLE,
    "figure.edgecolor": "white",
    "savefig.edgecolor": "white",
    "legend.labelcolor": "black",
    "axes.prop_cycle": plt.cycler(
        color=[_COLORS[key] for key in ("main", "second", "fit", "first", "raw")],
    ),
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "medium",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def _finish(fig, title):
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.)
    for axis in fig.axes:
        axis.set_facecolor("white")
        axis.patch.set_alpha(1.)
    fig.suptitle(title, fontsize=14, fontweight="medium")
    return fig


def _angular_axis(axis, feature):
    axis.set_xlim(0, 360)
    if feature == "cb":
        labels = ["0\nfront", "90\nright", "180\nback", "270\nleft", "360"]
        axis.set_xlabel("Center bearing (°)")
    else:
        labels = ["0\nup", "90\nright", "180\ndown", "270\nleft", "360"]
        axis.set_xlabel("Head direction (°)")
    axis.set_xticks([0, 90, 180, 270, 360], labels)


def _format(value):
    return f"{value:.3g}" if np.isfinite(value) else "NA"


def _classification_state(row, feature):
    if row[f"{feature}_status"] == "classified":
        return "positive" if bool(row[f"{feature}_is_cell"]) else "negative"
    return "unassessed"


def _classification_label(row, feature):
    strength = "cd_r2" if feature == "cd" else f"{feature}_mvl"
    strength_name = "R²" if feature == "cd" else "MVL"
    label = (
        f"{feature.upper()}: {_classification_state(row, feature)}"
        f" · {strength_name} = {_format(row[strength])}"
        f" (threshold {_format(row[f'{feature}_strength_threshold'])})"
        f"\nSplit-half r = {_format(row[f'{feature}_stability'])}"
        f" (threshold {_format(row[f'{feature}_stability_threshold'])})"
    )
    if row[f"{feature}_status"] != "classified":
        label += f" · {row[f'{feature}_status']}"
    return label


def _rate_cmap():
    return plt.get_cmap("viridis").with_extremes(bad=_COLORS["missing"])


def _rate_colorbar(fig, axis, image, values):
    finite = np.asarray(values)[np.isfinite(values)]
    options = {}
    if finite.size and not np.any(finite):
        image.set_clim(0., 1.)
        options["ticks"] = [0.]
    fig.colorbar(image, ax=axis, label="Firing rate (Hz); gray = unvisited", **options)


@plt.rc_context(_PLOT_STYLE)
def plot_center_tuning(result, unit_id, *, feature="cb", plot_style=None, title=None):
    """Display one saved CB/CD curve and its saved classification thresholds.

    CB supports line, polar, heatmap, and donut styles; CD supports line and
    heatmap on its native centimetre axis. The donut's radius is decorative,
    not a distance coordinate. Missing bins stay missing in every style.
    """
    if feature not in ("cb", "cd"):
        raise ValueError("feature must be 'cb' or 'cd'.")
    plot_style = plot_style or ("polar" if feature == "cb" else "line")
    styles = ("line", "heatmap", "polar", "donut") if feature == "cb" else ("line", "heatmap")
    if plot_style not in styles:
        raise ValueError(f"{feature.upper()} plot_style must be one of {styles}.")
    curve = result["tuning"][unit_id][feature]
    row = result["summary"].loc[unit_id]
    edges, centers = np.asarray(curve["edges"]), np.asarray(curve["centers"])
    rates = np.asarray(curve["smoothed_rate_hz"])
    polar = plot_style in ("polar", "donut")
    fig, axis = plt.subplots(
        figsize=(8, 7) if polar else (10, 4), layout="constrained",
        subplot_kw={"projection": "polar"} if polar else {},
    )
    name = "Center bearing" if feature == "cb" else "Center distance"
    if plot_style == "donut":
        # Use the spatial angular renderer for a single decorative radial band.
        image = plot_egocentric_polar(
            axis, edges, np.array([0., 1.]), rates[:, None], hole_fraction=.5,
            colorbar_label="Firing rate (Hz); gray = unvisited",
        )
        image.set_cmap(_rate_cmap())
        axis.set_yticks([])
        axis.set_xlabel("Center bearing (°; 0 = front, clockwise)")
    elif plot_style == "heatmap":
        image = axis.pcolormesh(
            edges, [0., 1.], rates[None, :], shading="flat", cmap=_rate_cmap(), vmin=0.,
        )
        axis.set_yticks([])
        _rate_colorbar(fig, axis, image, rates)
    else:
        for key, color, label, width in (
            ("rate_hz", _COLORS["raw"], "Unsmoothed", .8),
            ("smoothed_rate_hz", _COLORS["main"], "Smoothed", 1.8),
        ):
            if polar:
                line = plot_tuning_curve(axis, edges, curve[key], name)
                line.set(color=color, label=label, linewidth=width)
            else:
                axis.plot(centers, curve[key], color=color, lw=width, label=label)
        if feature == "cd":
            slope, intercept = row["cd_slope_hz_per_cm"], row["cd_intercept_hz"]
            if np.isfinite([slope, intercept]).all():
                fitted = slope * centers + intercept
                fitted[np.asarray(curve["occupancy_s"]) <= 0] = np.nan
                axis.plot(centers, fitted, color=_COLORS["fit"], ls="--", label="Linear fit")
        if polar:
            axis.set_xlabel("Radius: firing rate (Hz)", labelpad=24)
        else:
            axis.set_ylabel("Firing rate (Hz)")
        axis.legend(fontsize=8)
    if not polar:
        if feature == "cd":
            axis.set(xlabel="Center distance (cm)", xlim=edges[[0, -1]])
        else:
            _angular_axis(axis, feature)
        if plot_style == "line":
            axis.grid(axis="y", color="#E9EDF2", lw=.6)
    else:
        axis.set_xticks(np.deg2rad([0, 90, 180, 270]), ["0°\nfront", "90°\nright", "180°\nback", "270°\nleft"])
    axis.set_title(_classification_label(row, feature), fontsize=10, pad=18)
    return _finish(fig, title or f"Unit {unit_id} | {name} tuning")


@plt.rc_context(_PLOT_STYLE)
def plot_center_population(result, *, feature="cb", unit_ids=None,
                           rate_key="smoothed_rate_hz", title=None):
    """Display supplied unit order and rates on native CB degrees or CD cm.

    No normalization, peak sorting, angle wrapping, or classification is run.
    Units must share the same saved bin edges; absent bins remain gray.
    """
    if feature not in ("cb", "cd"):
        raise ValueError("feature must be 'cb' or 'cd'.")
    if rate_key not in ("rate_hz", "smoothed_rate_hz"):
        raise ValueError("rate_key must be 'rate_hz' or 'smoothed_rate_hz'.")
    units = list(result["summary"].index if unit_ids is None else unit_ids)
    fig, axis = plt.subplots(figsize=(10, min(20, max(4, len(units) * .24))), layout="constrained")
    name = "Center bearing" if feature == "cb" else "Center distance"
    if units:
        curves = [result["tuning"][unit][feature] for unit in units]
        edges = np.asarray(curves[0]["edges"])
        if any(not np.array_equal(curve["edges"], edges) for curve in curves[1:]):
            raise ValueError("Population tuning curves must share the same saved bin edges.")
        values = np.stack([curve[rate_key] for curve in curves])
        image = axis.pcolormesh(
            edges, np.arange(len(units) + 1) - .5, values,
            shading="flat", cmap=_rate_cmap(), vmin=0.,
        )
        _rate_colorbar(fig, axis, image, values)
        axis.set_ylim(len(units) - .5, -.5)
        labels = [f"{unit}: {_classification_state(result['summary'].loc[unit], feature)}" for unit in units]
        axis.set_yticks(np.arange(len(units)), labels)
        axis.set_ylabel("Unit ID: classification")
        if feature == "cb":
            _angular_axis(axis, feature)
        else:
            axis.set(xlabel="Center distance (cm)", xlim=edges[[0, -1]])
    else:
        axis.text(.5, .5, "No units selected", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
    return _finish(fig, title or f"{name} population tuning")


def _feature_occupancy(behavior, feature):
    """Bin movement exposure for the behavior overview."""
    if feature == "cd":
        width = behavior["config"]["distance_bin_cm"]
        maximum = behavior["max_center_distance_cm"]
        edges = np.arange(int(np.ceil(maximum / width)) + 1) * width
        values = behavior["cd_cm"]
    else:
        width = behavior["config"]["angle_bin_deg"]
        edges = np.linspace(0, 360, int(round(360 / width)) + 1)
        values = behavior[f"{feature}_deg"]
    selected = behavior["moving_mask"] & np.isfinite(values)
    occupancy = np.histogram(
        values[selected], edges, weights=behavior["exposure_s"][selected],
    )[0]
    return edges, occupancy


@plt.rc_context(_PLOT_STYLE)
def plot_behavior(behavior, title="Behavior and occupancy"):
    """Show trajectory, speed, and movement occupancy in acquisition coordinates."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    trajectory_axis, spatial_axis, speed_axis = axes[0]
    xy = np.array(behavior["smoothed_xy_cm_y_down"], dtype=float, copy=True)
    times = np.asarray(behavior["times_s"])
    valid = np.asarray(behavior["valid_position_mask"])
    max_gap_s = behavior["median_frame_interval_s"] * behavior["config"]["max_gap_factor"]
    gaps = np.r_[False, np.diff(times) > max_gap_s]
    # Break the displayed path at excluded frames and tracking gaps.
    xy[~valid | gaps] = np.nan
    arena = np.asarray(behavior["arena_size_cm"])
    threshold = behavior["config"]["speed_threshold_cm_s"]
    trajectory_axis.plot(xy[:, 0], xy[:, 1], color="#BFC6CF", lw=.55, label="Tracked path")
    moving_xy = xy.copy()
    moving_xy[~behavior["moving_mask"]] = np.nan
    trajectory_axis.plot(
        moving_xy[:, 0], moving_xy[:, 1], color=_COLORS["main"], lw=.7,
        label=f"Speed >{threshold:g} cm/s",
    )
    center = np.asarray(behavior["center_xy_cm"])
    if behavior["arena_shape"] == "circle":
        trajectory_axis.add_patch(Circle(center, arena[0] / 2, fill=False, color="#344054", lw=1))
    trajectory_axis.scatter(
        *center, s=70, marker="+", linewidths=1.8, color=_COLORS["fit"], label="Arena center",
    )
    trajectory_axis.set(
        xlim=(0, arena[0]), ylim=(arena[1], 0), aspect="equal",
        xlabel="x (cm)", ylabel="y (cm)", title="Trajectory",
    )
    trajectory_axis.legend(
        fontsize=8, loc="upper right", frameon=True,
        facecolor="white", framealpha=1., edgecolor="#D0D5DD",
    )

    occupancy = np.asarray(behavior["spatial_occupancy_s"])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(_COLORS["missing"])
    mesh = spatial_axis.pcolormesh(
        behavior["spatial_x_edges_cm"], behavior["spatial_y_edges_cm"],
        np.ma.masked_where(occupancy.T == 0, occupancy.T), cmap=cmap, shading="flat", vmin=0,
    )
    spatial_axis.set(
        xlim=(0, arena[0]), ylim=(arena[1], 0), aspect="equal",
        xlabel="x (cm)", ylabel="y (cm)",
        title=f"Spatial occupancy · coverage {behavior['spatial_coverage']:.1%}",
    )
    fig.colorbar(mesh, ax=spatial_axis, label="Occupancy (s); gray = unvisited", shrink=.8)

    speed = np.array(behavior["speed_cm_s"], dtype=float, copy=True)
    speed[gaps] = np.nan
    speed_axis.plot((times - times[0]) / 60, speed, color=_COLORS["main"], lw=.65)
    speed_axis.axhline(
        threshold, color=_COLORS["fit"], ls="--", lw=1.1, label=f"Threshold: {threshold:g} cm/s",
    )
    speed_axis.set(xlabel="Elapsed time (min)", ylabel="Speed (cm/s)", title="Speed")
    speed_axis.legend(fontsize=8)

    features = (("hd", "Head direction"), ("cb", "Center bearing"), ("cd", "Center distance"))
    for axis, (feature, name) in zip(axes[1], features):
        edges, exposure = _feature_occupancy(behavior, feature)
        axis.stairs(exposure, edges, color=_COLORS["main"], fill=True, alpha=.8)
        axis.set(ylabel="Movement occupancy (s)", title=f"{name} occupancy")
        if feature == "cd":
            axis.set_xlabel("Center distance (cm)")
        else:
            _angular_axis(axis, feature)
    return _finish(fig, title)


@plt.rc_context(_PLOT_STYLE)
def plot_center_unit(result, unit_id, title=None):
    """Plot tuning, split-half stability, occupancy, and supplied classifications.

    The CD fit uses the saved coefficients from unsmoothed rates. Unvisited
    bins remain gaps, and observed zero rates remain zero.
    """
    curves = result["tuning"][unit_id]
    row = result["summary"].loc[unit_id]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), layout="constrained")
    states = []
    features = (("cb", "Center bearing"), ("hd", "Head direction"), ("cd", "Center distance"))
    for col, (feature, name) in enumerate(features):
        curve = curves[feature]
        x = np.asarray(curve["centers"])
        full, halves, occupancy = axes[:, col]
        full.plot(
            x, curve["rate_hz"], color=_COLORS["raw"], lw=.8,
            marker=".", ms=3, label="Unsmoothed",
        )
        full.plot(x, curve["smoothed_rate_hz"], color=_COLORS["main"], lw=1.8, label="Smoothed")
        strength_key = "cd_r2" if feature == "cd" else f"{feature}_mvl"
        strength_name = "R²" if feature == "cd" else "MVL"
        full.set(
            title=f"{name} · {strength_name} = {_format(row[strength_key])}",
            ylabel="Firing rate (Hz)",
        )
        if feature == "cd":
            slope, intercept = row["cd_slope_hz_per_cm"], row["cd_intercept_hz"]
            if np.isfinite([slope, intercept]).all():
                fitted = slope * x + intercept
                # Preserve the fit's domain: unvisited bins are not observations.
                fitted[np.asarray(curve["occupancy_s"]) <= 0] = np.nan
                full.plot(x, fitted, color=_COLORS["fit"], ls="--", lw=1.5, label="Linear fit")
        full.legend(fontsize=8)
        half_styles = ((_COLORS["first"], "First half"), (_COLORS["second"], "Second half"))
        for index, (color, label) in enumerate(half_styles):
            halves.plot(x, curve["half_smoothed_rate_hz"][index], color=color, lw=1.5, label=label)
            occupancy.stairs(
                curve["half_occupancy_s"][index], curve["edges"], color=color, lw=1.1, label=label,
            )
        halves.set(
            title=f"Split-half stability · r = {_format(row[f'{feature}_stability'])}",
            ylabel="Firing rate (Hz)",
        )
        halves.legend(fontsize=8)
        occupancy.stairs(
            curve["occupancy_s"], curve["edges"], color=_COLORS["raw"],
            lw=1.3, label="Full recording",
        )
        occupancy.set(title="Occupancy", ylabel="Movement occupancy (s)")
        occupancy.legend(fontsize=8)
        for axis in axes[:, col]:
            if feature == "cd":
                axis.set_xlabel("Center distance (cm)")
            else:
                _angular_axis(axis, feature)
            axis.grid(axis="y", color="#E9EDF2", lw=.6)
        status = row[f"{feature}_status"]
        state = "unassessed"
        if status == "classified":
            state = "positive" if bool(row[f"{feature}_is_cell"]) else "negative"
        states.append(f"{feature.upper()}: {state}")
    heading = title or f"Unit {unit_id} | Center and head-direction tuning"
    subtitle = " · ".join(states) + f" | Movement spikes: {row['n_speed_spikes']:,}"
    return _finish(fig, f"{heading}\n{subtitle}")


@plt.rc_context(_PLOT_STYLE)
def plot_center_classification(counts):
    """Show supplied positive, negative, and unassessed unit counts by session."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    x = np.arange(len(counts))
    for axis, feature in zip(axes, ("cb", "cd", "hd")):
        positives = counts[f"{feature}_positive"]
        negatives = counts[f"{feature}_assessed"] - positives
        missing = counts[f"{feature}_unassessed"]
        axis.bar(x, positives, label="Positive")
        axis.bar(x, negatives, bottom=positives, label="Negative")
        axis.bar(x, missing, bottom=positives + negatives, label="Unassessed")
        axis.set(xticks=x, xticklabels=counts.index, ylabel="Units", title=feature.upper())
        axis.tick_params(axis="x", rotation=60)
    axes[0].legend()
    return _finish(fig, "Center-cell classification")


# Preserve existing imports while theta notebooks use their own plotting module.
from Utils.theta_plotting import plot_lfp_reference, plot_theta_unit
