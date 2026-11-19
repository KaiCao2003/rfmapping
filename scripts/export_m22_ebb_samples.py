"""Draw M22 EBB examples from the saved clockwise inner-boundary rate maps.

Loading and saving stay in slides_m22.ipynb. Maps retain native angle/distance
bins; tuning curves use the same finite rate-cell means as the population panels.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrow
import numpy as np

from Utils.plotting import (
    LIGHT_PLOT_STYLE, plot_egocentric_polar, plot_tuning_curve,
    signed_egocentric_map,
)


EBB_BANDS = (("0-8", "≤ 8 cm"), ("8-16", "8–16 cm"), ("16-inf", "> 16 cm"))


def band_slices(distance_edges):
    centers = (distance_edges[:-1] + distance_edges[1:]) / 2
    first, second = np.searchsorted(centers, [8., 16.], side="right")
    return {"0-8": (0, first), "8-16": (first, second),
            "16-inf": (second, len(centers))}


def mean_band_profiles(rates, distance_edges):
    # Equal-weight finite rate cells, not summed rates or occupancy-pooled rates.
    paired_angles = rates.reshape(len(rates), 30, rates.shape[1] // 30, -1)
    return {band: np.nanmean(paired_angles[:, :, :, start:stop], axis=(2, 3))
            for band, (start, stop) in band_slices(distance_edges).items()}


def plot_ebb_map(ax, rate_map, theta_edges, distance_edges, band, title):
    start, stop = band_slices(distance_edges)[band]
    image = plot_egocentric_polar(
        ax, theta_edges, distance_edges[start:stop + 1], rate_map[:, start:stop],
        hole_fraction=.45 if band == "0-8" else None,
        colorbar_label="Firing rate (Hz)",
    )
    # All three maps of a unit share its full-map rate scale.
    image.set_clim(np.nanmin(rate_map), np.nanmax(rate_map))
    image.colorbar.solids.set_rasterized(False)
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_title(title, pad=22)
    return image


def plot_ebb_tuning(ax, profile, theta_edges, title):
    line = plot_tuning_curve(ax, theta_edges[::2], profile, title, signed=True)
    ax.set_facecolor("white")
    ax.set_xlabel("Mean firing rate (Hz)", labelpad=20)
    ax.set_title(title, pad=22)
    return line


def sample_figure(unit_id, rate_map, theta_edges, distance_edges):
    profiles = mean_band_profiles(rate_map[None], distance_edges)
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10.5),
                                 subplot_kw={"projection": "polar"},
                                 layout="constrained", facecolor="white")
        for column, (band, label) in enumerate(EBB_BANDS):
            plot_ebb_map(axes[0, column], rate_map, theta_edges, distance_edges,
                         band, f"EBB map · {label}")
            plot_ebb_tuning(axes[1, column], profiles[band][0], theta_edges,
                            f"EBB angular tuning · {label}")
        fig.suptitle(f"M22 261008 S9 · Probe A · unit {unit_id}", fontsize=17)
    return fig


def sample_panel_figure(unit_id, rate_map, theta_edges, distance_edges, band, kind):
    label = dict(EBB_BANDS)[band]
    title = f"M22 261008 S9 · unit {unit_id}\nEBB · {label}"
    with plt.rc_context(LIGHT_PLOT_STYLE):
        size = 6.2 if kind == "map" else 5.2
        fig, ax = plt.subplots(figsize=(size, size),
                               subplot_kw={"projection": "polar"},
                               layout="constrained", facecolor="white")
        if kind == "map":
            plot_ebb_map(ax, rate_map, theta_edges, distance_edges, band, title)
        elif kind == "tuning":
            profile = mean_band_profiles(rate_map[None], distance_edges)[band][0]
            plot_ebb_tuning(ax, profile, theta_edges, title)
        else:
            raise ValueError("kind must be 'map' or 'tuning'")
    return fig


def rate_map_conversion_figure(unit_id, rate_map, theta_edges, distance_edges):
    """Show the same native rate cells in rectangular and physical polar views."""
    angles, values = signed_egocentric_map(theta_edges, rate_map)
    norm = Normalize(np.nanmin(rate_map), np.nanmax(rate_map))
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("white")
    mesh_style = dict(cmap=color_map, norm=norm, shading="flat",
                      edgecolors="face", linewidth=0, antialiased=False)
    guide_style = dict(color="#cf381c", linewidth=3, linestyle=(0, (1, 1.5)))
    outer = distance_edges[-1]

    with plt.rc_context({**LIGHT_PLOT_STYLE, "svg.fonttype": "none", "font.size": 16}):
        fig = plt.figure(figsize=(16, 9), facecolor="white")
        fig.text(.04, .925, f"Unit {unit_id}: egocentric boundary tuning",
                 fontsize=28, weight="bold")
        fig.text(.5, .845, f"M22 261008 S9 · Probe A · unit {unit_id}",
                 fontsize=16, ha="center")
        fig.text(.305, .73, "Occupancy-normalized firing", fontsize=22, ha="center")
        fig.text(.76, .73, "Egocentric response map", fontsize=22, ha="center")

        rectangular = fig.add_axes([.135, .20, .34, .465], facecolor="white")
        rectangular.pcolormesh(distance_edges, angles, values, **mesh_style)
        rectangular.set(xlim=(distance_edges[0], outer), ylim=(-180, 180),
                        xlabel="Boundary distance, d (cm)", ylabel="Relative bearing, θ")
        rectangular.set_xticks([0, 8, 16, outer], ["0", "8", "16", f"{outer:g}"])
        rectangular.set_yticks([-180, -90, 0, 90, 180],
                              ["Back", "Left", "Front", "Right", "Back"])
        rectangular.tick_params(labelsize=16, pad=6)
        rectangular.spines[["top", "right"]].set_visible(False)

        polar = fig.add_axes([.62, .155, .28, .50], facecolor="white")
        theta, radius = np.meshgrid(np.deg2rad(angles), distance_edges, indexing="ij")
        image = polar.pcolormesh(radius * np.sin(theta), radius * np.cos(theta),
                                values, **mesh_style)
        # Exact 8/16 cm guides do not rebin the native 1.5 cm distance cells.
        for distance in (8, 16):
            rectangular.axvline(distance, **guide_style)
            polar.add_patch(Circle((0, 0), distance, fill=False, **guide_style))
        polar.add_patch(Circle((0, 0), outer * .007, facecolor=".45", edgecolor="none"))
        polar.set(aspect="equal", xlim=(-1.02 * outer, 1.02 * outer),
                  ylim=(-1.02 * outer, 1.02 * outer))
        polar.axis("off")
        polar.text(.5, 1.05, "Front", transform=polar.transAxes,
                   ha="center", fontsize=17)
        fig.add_artist(FancyArrow(.49, .435, .113, 0, width=.026,
                                  head_width=.055, head_length=.016,
                                  length_includes_head=True, transform=fig.transFigure,
                                  facecolor="black", edgecolor="none"))

        cax = fig.add_axes([.92, .20, .018, .41], facecolor="white")
        colorbar = fig.colorbar(image, cax=cax, ticks=[norm.vmin, norm.vmax])
        colorbar.ax.set_yticklabels([f"{norm.vmin:.1f}", f"{norm.vmax:.1f}"])
        colorbar.ax.tick_params(labelsize=16)
        colorbar.set_label("Hz", fontsize=17, labelpad=10)
        colorbar.solids.set_rasterized(False)
        fig.text(.135, .09, "White bins: unvisited", fontsize=11, color=".3")
    return fig
