"""Display prepared theta and LFP results without loading or recalculating data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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


def _format(value):
    return f"{value:.3g}" if np.isfinite(value) else "NA"


def _saved_state(summary, flag, *, positive="positive", negative="negative"):
    value = summary[flag]
    return "unassessed" if pd.isna(value) else positive if bool(value) else negative


def _unavailable(axis, message):
    axis.text(
        .5, .5, message, ha="center", va="center", transform=axis.transAxes,
        color="#667085", wrap=True,
    )
    axis.set_xticks([])
    axis.set_yticks([])


@plt.rc_context(_PLOT_STYLE)
def plot_theta_unit(unit_results, title="Theta analysis", *, summary=None):
    """Plot saved theta measurements and, when supplied, final summary labels.

    Summary classifications include the analysis eligibility gates. They take
    precedence over raw candidate flags, whose descriptive values may remain
    available even when the final classification is unassessed.
    """
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
        if summary is not None:
            axes[0, 1].set_title(
                axes[0, 1].get_title() + f"\n{_saved_state(summary, 'is_theta_rhythmic')}"
                f" · {summary['rhythmicity_status'].replace('_', ' ')}",
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
        if summary is not None:
            axes[1, 0].set_title(
                axes[1, 0].get_title() + f"\n{_saved_state(summary, 'is_phase_locked')}"
                f" · {summary['phase_locking_status'].replace('_', ' ')}",
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
        if summary is not None:
            candidate = _saved_state(
                summary, "precession_candidate", positive="candidate", negative="not a candidate",
            )
            candidate += f" · {summary['precession_status'].replace('_', ' ')}"
        else:
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
    if summary is not None and summary["theta_status"] != "computed":
        title += f"\n{summary['theta_status'].replace('_', ' ')}"
    return _finish(fig, title)


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
        if trace.get("status") == "no_movement":
            right.set_title(f"Channel {channel}")
            _unavailable(right, "No movement exposure")
            continue
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
