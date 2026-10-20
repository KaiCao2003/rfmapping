"""Plot prepared center-cell, behavior, and theta results."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

from Utils.plotting import LIGHT_PLOT_STYLE


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


def _unavailable(axis, message):
    axis.text(
        .5, .5, message, ha="center", va="center", transform=axis.transAxes,
        color="#667085", wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


@plt.rc_context(_PLOT_STYLE)
def plot_theta_unit(unit_results, title="Theta analysis"):
    """Plot spike rhythmicity, phase counts, occupancy, and the precession test."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    rhythmicity = unit_results.get("rhythmicity")
    if rhythmicity is not None:
        axes[0, 0].plot(
            rhythmicity["acg_lags_s"], rhythmicity["acg_counts"], color=_COLORS["main"], lw=1,
        )
        axes[0, 0].set(
            xlabel="Time lag (s)", ylabel="Spike-pair count", title="Spike autocorrelogram",
        )
        axes[0, 1].plot(
            rhythmicity["frequency_hz"], rhythmicity["spectrum"], color=_COLORS["main"], lw=1,
        )
        if np.any(np.asarray(rhythmicity["spectrum"]) > 0):
            # Retain DC in the calculation/display without hiding theta detail.
            axes[0, 1].set_yscale("log")
        params = rhythmicity["parameters"]
        axes[0, 1].axvspan(
            *params["theta_band_hz"], color="#E4EFF3", zorder=0, label="Theta band",
        )
        peak = rhythmicity["peak_frequency_hz"]
        if np.isfinite(peak):
            axes[0, 1].axvline(
                peak, color=_COLORS["fit"], ls="--", lw=1, label=f"Peak: {peak:.2f} Hz",
            )
        axes[0, 1].set(
            xlim=(0, params["reference_band_hz"][1]), xlabel="Frequency (Hz)",
            ylabel=f"ACG spectral {params['spectrum_convention']}",
            title=f"Theta rhythmicity · TRI = {_format(rhythmicity['theta_index'])}",
        )
        axes[0, 1].legend(fontsize=8)
    else:
        for axis in axes[0]:
            _unavailable(axis, "Rhythmicity not assessed")

    locking = unit_results.get("phase_locking")
    if locking is not None:
        edges = np.degrees(locking["phase_bin_edges_rad"])
        axes[1, 0].stairs(
            locking["phase_counts"], edges, fill=True, color=_COLORS["main"], alpha=.8,
        )
        axes[1, 0].set(
            xlabel="LFP phase (°; 0 = positive peak)", ylabel="Spike count",
            title=(f"Phase locking · MVL = {_format(locking['mean_vector_length'])}"
                   f" · p = {_format(locking['rayleigh_p'])}"),
        )
        preferred = locking["preferred_phase_rad"]
        if np.isfinite(preferred):
            axes[1, 0].axvline(
                np.degrees(preferred), color=_COLORS["fit"], ls="--", label="Preferred phase",
            )
            axes[1, 0].legend(fontsize=8)
        axes[1, 1].stairs(
            locking["phase_occupancy_s"], edges, color=_COLORS["raw"], fill=True, alpha=.7,
        )
        axes[1, 1].set(
            xlabel="LFP phase (°)", ylabel="Occupancy (s)",
            title="Phase occupancy",
        )
        for axis in axes[1]:
            axis.set_xlim(-180, 180)
            axis.set_xticks([-180, -90, 0, 90, 180])
    else:
        for axis in axes[1]:
            _unavailable(axis, "LFP phase locking not assessed")

    precession = unit_results.get("phase_precession")
    if precession is not None:
        axes[2, 0].plot(
            precession["acg_lags_cycles"], precession["acg_counts"], color=_COLORS["main"], lw=1,
        )
        axes[2, 0].set(
            xlabel="LFP phase lag (cycles)", ylabel="Spike-pair count",
            title="Phase autocorrelogram",
        )
        axes[2, 1].plot(
            precession["frequency_per_cycle"], precession["spectrum"], color=_COLORS["main"], lw=1,
        )
        for index, band in enumerate(precession["parameters"]["reference_bands"]):
            axes[2, 1].axvspan(
                *band, color="#E9EDF2", zorder=0, label="Reference bands" if index == 0 else None,
            )
        axes[2, 1].axvline(1., color="#344054", ls=":", label="LFP frequency")
        peak = precession["peak_frequency_per_cycle"]
        if np.isfinite(peak):
            axes[2, 1].axvline(peak, color=_COLORS["fit"], ls="--", label=f"Peak {peak:.3f}")
        candidate = "candidate" if precession["precession_candidate"] else "not a candidate"
        if not precession["eligible"]:
            candidate = "unassessed"
        axes[2, 1].set(
            xlim=(.4, 1.9), xlabel="Frequency / LFP frequency",
            ylabel=f"ACG spectral {precession['parameters']['spectrum_convention']}",
            title=(f"Precession test · {candidate}"
                   f"\nSpectral ratio = {_format(precession['phase_precession_index'])}"),
        )
        frequency = np.asarray(precession["frequency_per_cycle"])
        visible = (frequency >= .4) & (frequency <= 1.9)
        visible_peak = np.max(np.asarray(precession["spectrum"])[visible], initial=0.)
        if visible_peak > 0:
            # Autoscaling on the entire spectrum includes out-of-view DC.
            axes[2, 1].set_ylim(0, visible_peak * 1.12)
        axes[2, 1].legend(fontsize=8)
    else:
        for axis in axes[2]:
            _unavailable(axis, "Phase precession not assessed")
    return _finish(fig, title)


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


@plt.rc_context(_PLOT_STYLE)
def plot_lfp_reference(references, *, channel):
    """Plot prepared LFP spectra and movement excerpts for each recording."""
    fig, axes = plt.subplots(
        len(references), 2, figsize=(14, 3.5 * len(references)),
        layout="constrained", squeeze=False,
    )
    for (key, reference), (left, right) in zip(references.items(), axes):
        spectrum, trace = reference["spectrum"], reference["trace"]
        frequency = spectrum["frequency_hz"]
        selected = (frequency >= 1) & (frequency <= 40)
        fraction = reference["summary"]["power_4_11_over_1_40"]
        left.semilogy(frequency[selected], spectrum["psd_uv2_per_hz"][selected])
        left.axvspan(4, 11, alpha=.15)
        left.set(
            xlabel="Frequency (Hz)", ylabel="Power spectral density (µV²/Hz)",
            title=f"{key} | 4–11 Hz / 1–40 Hz power = {fraction:.1%}",
        )
        times = trace["time_from_midpoint_s"]
        right.plot(times, trace["voltage_uv"], label="LFP")
        right.plot(times, trace["filtered_uv"], label="5–10 Hz")
        right.plot(times, trace["amplitude_uv"], label="Amplitude envelope")
        right.set(
            xlabel="Time from movement midpoint (s)", ylabel="Voltage (µV)",
            title=f"Channel {channel} | Midpoint: {trace['midpoint_s']:.1f} s",
        )
        right.legend()
    return _finish(fig, "LFP reference")
