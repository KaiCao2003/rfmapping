"""Explain the M22 all-elevation projection using loaded counts and masks."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from RF.profiles import prep_rf_tc
from scripts.export_m22_f3_examples import cell_edges
from scripts.export_m22_rf_sample_variants import gray_palette, rectangle_axes
from Utils.plotting import LIGHT_PLOT_STYLE


def projection_data(response, detected, rates):
    """Prepare the same count projection and peak normalization as sorting."""
    mask = detected.for_unit(response.unit_id).astype(bool)
    projected = response.sum_to_1d(axis="x", rf_only=False)
    counts = projected.to_1d_array()
    native = pd.DataFrame([counts], index=[response.unit_id], columns=response.x_positions)
    normalized = prep_rf_tc(native, bins=30, max_zero_bins=None)
    return {
        "unit_id": response.unit_id,
        "x": response.x_positions,
        "y": response.y_positions,
        "counts_2d": response.to_2d_array(),
        "rates_2d": rates.to_2d_array(),
        "rf_mask": mask,
        "selected_rows": np.arange(len(response.y_positions)),
        "rf_rows": np.flatnonzero(mask.any(axis=1)),
        "x_edges": cell_edges(response.x_positions),
        "y_edges": cell_edges(response.y_positions),
        "native_counts": counts,
        "angles": normalized.columns.to_numpy(dtype=float),
        "normalized": normalized.iloc[0].to_numpy(),
    }


def projection_export(data, sources, *, cmap="viridis"):
    """Serialize scientific values and matching palettes for slide assembly."""
    exported = {key: value.tolist() if isinstance(value, np.ndarray) else int(value)
                for key, value in data.items()}
    counts_palette, counts_norm = gray_palette(data["counts_2d"])
    rate_palette, rate_norm = gray_palette(data["rates_2d"])
    heatmap_palette = plt.colormaps[cmap]
    exported.update(
        count_color_min=float(np.nanmin(data["counts_2d"])),
        count_color_max=float(np.nanmax(data["counts_2d"])),
        rate_color_min=float(np.nanmin(data["rates_2d"])),
        rate_color_max=float(np.nanmax(data["rates_2d"])),
        colors_2d=[[to_hex(counts_palette(counts_norm(value))) for value in row]
                   for row in data["counts_2d"]],
        colors_rates_2d=[[to_hex(rate_palette(rate_norm(value))) for value in row]
                         for row in data["rates_2d"]],
        colors_1d=[to_hex(heatmap_palette(value)) for value in data["normalized"]],
        colorbar=[to_hex(heatmap_palette(value)) for value in np.linspace(0, 1, 101)],
        count_colorbar=[to_hex(counts_palette(value)) for value in np.linspace(0, 1, 101)],
        time_range_s=[0, .2], response_units="spike_count", map_response_units="Hz",
        aggregation="sum all seven elevation rows at each native azimuth",
        normalization="divide by curve maximum", smoothing=False,
        elevation_display="negative native Y; cell edges inferred from native centers",
        elevation_edges=(-data["y_edges"]).tolist(),
        exact_saved_count_match=True, sorting_normalized_match=True, sources=sources,
    )
    return exported


def projection_figure(data, *, cmap="viridis", line_color="#253d5b"):
    """Show the native rate map, all-row count sum, and peak-normalized values."""
    style = {
        **LIGHT_PLOT_STYLE, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "#202020", "axes.labelcolor": "#202020",
        "xtick.color": "#202020", "ytick.color": "#202020", "font.size": 11,
        "svg.fonttype": "none",
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
        ax_map = fig.add_axes((.09, .62, .50, .18))
        ax_counts = fig.add_axes((.09, .415, .50, .115))
        ax_curve = fig.add_axes((.09, .235, .50, .115))
        ax_row = fig.add_axes((.09, .125, .50, .04))
        x_edges, y_edges = data["x_edges"], data["y_edges"]
        rate_palette, norm = gray_palette(data["rates_2d"])
        image = ax_map.pcolormesh(x_edges, y_edges, data["rates_2d"], cmap=rate_palette,
                                 norm=norm, shading="flat", rasterized=False)
        for row, column in np.argwhere(data["rf_mask"]):
            ax_map.add_patch(Rectangle((x_edges[column], y_edges[row]),
                                      np.diff(x_edges)[column], np.diff(y_edges)[row],
                                      fill=False, edgecolor="#e33d35", linewidth=1.4))
        for row in data["selected_rows"]:
            ax_map.plot([x_edges[-1] + 4, x_edges[-1] + 4], y_edges[row:row + 2],
                        color="#d47700", linewidth=4, clip_on=False)
        rectangle_axes(ax_map, x_edges, y_edges)
        ax_map.set(ylabel="Elevation (°)", xlabel="", aspect="auto")
        ax_map.tick_params(axis="x", labelbottom=False)
        cax = fig.add_axes((.615, .62, .012, .18))
        bar = fig.colorbar(image, cax=cax, label="Firing rate (Hz)")
        bar.solids.set_rasterized(False)
        ax_counts.plot(data["x"], data["native_counts"], "o-", color=line_color,
                       linewidth=1.8, markersize=3.5)
        ax_counts.set(ylim=(0, data["native_counts"].max() * 1.08), ylabel="Spike count")
        ax_counts.tick_params(axis="x", labelbottom=False)
        ax_counts.spines[["top", "right"]].set_visible(False)
        ax_curve.plot(data["angles"], data["normalized"], "o-", color=line_color,
                      linewidth=1.8, markersize=3.5)
        ax_curve.set(ylim=(0, 1.08), ylabel="Normalized\nresponse")
        ax_curve.set_yticks([0, .5, 1])
        ax_curve.tick_params(axis="x", labelbottom=False)
        ax_curve.spines[["top", "right"]].set_visible(False)
        angle_edges = cell_edges(data["angles"])
        ax_row.pcolormesh(angle_edges, [0, 1], data["normalized"][None, :],
                          cmap=cmap, vmin=0, vmax=1, shading="flat", rasterized=False)
        ax_row.set(yticks=[], xlabel="Visual RF direction (°)")
        for axis in (ax_map, ax_counts, ax_curve, ax_row):
            axis.set_xlim(x_edges[[0, -1]])
            axis.set_xticks(np.arange(-180, 181, 60))
        fig.text(.05, .925, "2D RF map to 1D directional tuning", fontsize=25, weight="bold")
        fig.text(.05, .875, f"M22 261008, unit {data['unit_id']}   ON RF, 0–200 ms", fontsize=13)
        fig.text(.09, .825, "2D response map", fontsize=14, weight="bold")
        fig.text(.09, .555, "All 7 elevations → one count per azimuth", fontsize=13, weight="bold")
        fig.text(.09, .375, "Peak-normalized 1D tuning curve", fontsize=13, weight="bold")
        fig.text(.09, .188, "Same values as one heatmap row", fontsize=12, weight="bold")
        fig.text(.70, .76, "1  Keep every elevation", fontsize=15, weight="bold")
        fig.text(.70, .675, "All 7 rows contribute.\nRed outlines identify the saved RF.", fontsize=11.5)
        fig.text(.70, .56, "2  Sum each column", fontsize=15, weight="bold")
        fig.text(.70, .49, "Sum raw counts over all 7 rows.\nKeep all 30 azimuth bins.", fontsize=11.5)
        fig.text(.70, .425, r"$T(x)=\sum_{y=1}^{7} C(y,x)$", fontsize=18)
        fig.text(.70, .32, "3  Normalize and color", fontsize=15, weight="bold")
        fig.text(.70, .25, r"$T_{\rm norm}(x)=T(x)/\max_x T(x)$", fontsize=18)
        fig.text(.70, .17, "The curve and heatmap use\nthe same 30 normalized values.", fontsize=11.5)
        fig.text(.05, .035, "Map: mean firing rate. Projection: raw counts pooled across trials, "
                 "0–200 ms. No smoothing or baseline subtraction.", fontsize=10)
    return fig
