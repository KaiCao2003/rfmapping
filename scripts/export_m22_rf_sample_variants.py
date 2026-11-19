"""Draw slide-ready M22 sample variants from loaded responses and saved RF masks.

Keep source geometry and detector membership intact. Loading, selection and saving
belong to slides_m22.ipynb; red outlines always denote saved excitatory regions.
"""

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath
import numpy as np

from RF.rflocate._detector import _label_sections
from scripts.export_m22_f3_examples import RFGrayNorm, cell_edges, plot_annular, plot_hd


RF_COLOR = "#D62728"


def measured_bins(response):
    shape = response.to_2d_array().shape
    return (np.asarray(response.metadata['occupancyTimeSec']).reshape(shape) > 0
            ) & (response.presentation_counts.reshape(shape) > 0)


def gray_palette(values, *, vmin=None):
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if vmin is None else float(vmin)
    high = float(finite.max())
    if high == low:
        high = low + 1
    palette = ListedColormap(np.repeat(np.arange(18, 251)[:, None] / 255, 3, axis=1))
    palette.set_bad("#e6e8eb")
    return palette, RFGrayNorm(low, high)


def rectangle_axes(ax, x_edges, y_edges):
    # Source Y increases down the stimulus texture; elevation is its negative.
    ax.set(xlim=(x_edges[0], x_edges[-1]), ylim=(y_edges[-1], y_edges[0]),
           xlabel="Stimulus X (deg)", ylabel="Elevation (deg)", aspect="equal",
           facecolor="white")
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks([y_edges[0], 0, y_edges[-1]],
                  [f"{-y_edges[0]:+g}", "0", f"{-y_edges[-1]:+g}"])


def draw_rectangle(ax, values, x_edges, y_edges, cmap, norm):
    image = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(values),
                         shading="flat", cmap=cmap, norm=norm,
                         antialiased=False, rasterized=False)
    for row, column in np.argwhere(~np.isfinite(values)):
        ax.add_patch(Rectangle((x_edges[column], y_edges[row]),
                               x_edges[column + 1] - x_edges[column],
                               y_edges[row + 1] - y_edges[row], facecolor="#e6e8eb",
                               edgecolor="#a9abb1", hatch="///", linewidth=0))
    rectangle_axes(ax, x_edges, y_edges)
    return image


def annular_paths(mask, x_edges):
    """Original slides_v2 native-cell perimeters, including holes and the seam."""
    mask = np.asarray(mask, dtype=bool)
    rows, columns = mask.shape
    wrap = bool(np.isclose(x_edges[-1] - x_edges[0], 360))
    outgoing = {}

    def key(vertex):
        column, row = vertex
        return (column % columns if wrap else column, row)

    def edge(start, stop):
        outgoing.setdefault(key(start), []).append((start, stop))

    for row, column in np.argwhere(mask):
        if row == 0 or not mask[row - 1, column]:
            edge((column, row), (column + 1, row))
        if (column == columns - 1 and not wrap) or not mask[row, (column + 1) % columns]:
            edge((column + 1, row), (column + 1, row + 1))
        if row == rows - 1 or not mask[row + 1, column]:
            edge((column + 1, row + 1), (column, row + 1))
        if (column == 0 and not wrap) or not mask[row, (column - 1) % columns]:
            edge((column, row + 1), (column, row))
    paths = []
    while outgoing:
        first = current = next(iter(outgoing))
        pieces = []
        while True:
            start, stop = outgoing[current].pop()
            if not outgoing[current]:
                del outgoing[current]
            if start[1] == stop[1]:
                theta = np.deg2rad(np.linspace(90 - x_edges[start[0]], 90 - x_edges[stop[0]], 16))
                radius = 4 + rows - start[1]
                points = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
            else:
                theta = np.deg2rad(90 - x_edges[start[0]])
                radii = 4 + rows - np.array([start[1], stop[1]])
                points = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))
            pieces.append(points[:-1])
            current = key(stop)
            if current == first:
                break
        vertices = np.concatenate(pieces)
        vertices = np.vstack((vertices, vertices[0]))
        codes = np.full(len(vertices), MplPath.LINETO, dtype=np.uint8)
        codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
        paths.append(MplPath(vertices, codes))
    return paths


def add_annular_outline(ax, response, mask):
    for path in annular_paths(mask, cell_edges(response.x_positions)):
        ax.add_patch(PathPatch(path, fill=False, edgecolor=RF_COLOR,
                               linewidth=1.6, zorder=3, joinstyle="round"))


def add_rectangle_outline(ax, response, mask):
    """Draw only detected-region boundaries, including holes and display edges."""
    mask = np.asarray(mask, dtype=bool)
    xe, ye = cell_edges(response.x_positions), cell_edges(response.y_positions)
    vertical = np.diff(np.pad(mask, ((0, 0), (1, 1))), axis=1)
    horizontal = np.diff(np.pad(mask, ((1, 1), (0, 0))), axis=0)
    segments = [[(xe[col], ye[row]), (xe[col], ye[row + 1])]
                for row, col in np.argwhere(vertical)]
    segments += [[(xe[col], ye[row]), (xe[col + 1], ye[row])]
                 for row, col in np.argwhere(horizontal)]
    ax.add_collection(LineCollection(segments, colors=RF_COLOR, linewidths=1.6, zorder=3))


def annular_axes(ax, y_positions, *, elevation_origin=0):
    rows = len(y_positions)
    radius = 4 + rows
    for angle, label in ((-180, "±180°"), (-90, "−90°"), (0, "0°"), (90, "90°")):
        theta = np.deg2rad(90 - angle)
        ax.text((radius + 1) * np.cos(theta), (radius + 1) * np.sin(theta),
                label, ha="center", va="center")
    for row in (0, rows - 1):
        ring_center = 4 + rows - row - .5
        ax.plot([ring_center, ring_center], [-.15, .15], color="black", lw=.8)
        ax.text(ring_center, .35, f"{elevation_origin - y_positions[row]:+g}°", ha="center", va="bottom",
                fontsize=8, bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.text(.02, .02, "Rings: elevation", transform=ax.transAxes, fontsize=8)
    ax.set(xlim=(-radius - 2, radius + 2), ylim=(-radius - 2, radius + 2),
           aspect="equal", facecolor="white")
    ax.axis("off")


def mask_legend(ax):
    ax.legend(handles=[Line2D([], [], color=RF_COLOR, label="Saved excitatory RF")],
              loc="lower center", bbox_to_anchor=(.5, -.1), frameon=False)


def rectangle_figure(response, mask, unit_id, polarity="ON", outline=True, *, vmin=None):
    values = response.to_2d_array()
    xe, ye = cell_edges(response.x_positions), cell_edges(response.y_positions)
    cmap, norm = gray_palette(values, vmin=vmin)
    fig, ax = plt.subplots(figsize=(8.8, 3.8), layout="constrained", facecolor="white")
    image = draw_rectangle(ax, values, xe, ye, cmap, norm)
    if outline and mask.any():
        for row, column in np.argwhere(mask):
            ax.add_patch(Rectangle((xe[column], ye[row]), xe[column + 1] - xe[column],
                                   ye[row + 1] - ye[row], fill=False,
                                   edgecolor=RF_COLOR, linewidth=.9))
        ax.legend(handles=[Line2D([], [], color=RF_COLOR, label="Saved excitatory RF")],
                  loc="upper center", bbox_to_anchor=(.5, -.3), frameon=False, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, label="Firing rate (Hz)", shrink=.7)
    colorbar.solids.set_rasterized(False)
    ax.set_title(f"M22 · 261008 · unit {unit_id} · {polarity} RF\n0–200 ms · unsmoothed")
    return fig


def annular_figure(response, mask, unit_id, polarity="ON", *, elevation_origin=0, vmin=None):
    values = response.to_2d_array()
    xe = cell_edges(response.x_positions)
    cmap, norm = gray_palette(values, vmin=vmin)
    rows, columns = values.shape
    fig, ax = plt.subplots(figsize=(6.5, 6.8), layout="constrained", facecolor="white")
    for row in range(rows):
        for column in range(columns):
            value = values[row, column]
            color = cmap(norm(value)) if np.isfinite(value) else "white"
            wedge = Wedge((0, 0), 4 + rows - row, 90 - xe[column + 1], 90 - xe[column],
                          width=1, facecolor=color, edgecolor=color, linewidth=.3, antialiased=False)
            if not np.isfinite(value):
                wedge.set(hatch="///", edgecolor=".7")
            ax.add_patch(wedge)
    add_annular_outline(ax, response, mask)
    annular_axes(ax, response.y_positions, elevation_origin=elevation_origin)
    ax.set_title(f"M22 · 261008 · unit {unit_id} · {polarity} RF\n0–200 ms · unsmoothed")
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                           label="Firing rate (Hz)", shrink=.65)
    colorbar.solids.set_rasterized(False)
    if mask.any():
        mask_legend(ax)
    return fig


def binary_figure(response, mask, unit_id, annular=False):
    measured = measured_bins(response)
    xe, ye = cell_edges(response.x_positions), cell_edges(response.y_positions)
    if annular:
        fig, ax = plt.subplots(figsize=(6.1, 6.1), layout="constrained", facecolor="white")
        rows, columns = mask.shape
        for row in range(rows):
            for column in range(columns):
                color = "white" if mask[row, column] else "black"
                wedge = Wedge((0, 0), 4 + rows - row, 90 - xe[column + 1], 90 - xe[column],
                              width=1, facecolor=color, edgecolor=color, linewidth=.3, antialiased=False)
                if not measured[row, column]:
                    wedge.set(facecolor="white", edgecolor=".7", hatch="///")
                ax.add_patch(wedge)
        annular_axes(ax, response.y_positions)
        ax.set_title(f"M22 · 261008 · unit {unit_id}\nSaved excitatory ON mask · white = RF")
    else:
        values = mask.astype(float)
        values[~measured] = np.nan
        fig, ax = plt.subplots(figsize=(8.8, 3.6), layout="constrained", facecolor="white")
        draw_rectangle(ax, values, xe, ye, "gray", Normalize(0, 1))
        ax.set_title(f"M22 · 261008 · unit {unit_id}\nSaved excitatory ON mask")
        handles = [Patch(facecolor="white", edgecolor="black", label="Saved RF"),
                   Patch(facecolor="black", label="Measured non-RF")]
        if not measured.all():
            handles.append(Patch(facecolor="#e6e8eb", edgecolor=".8", hatch="///", label="Not measured"))
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.3),
                  ncol=len(handles), frameon=False)
    return fig


def highlight_figure(unit_id, hd_curve, responses, masks):
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    plot_hd(fig.add_axes((.06, .17, .39, .66), projection="polar"), hd_curve)
    for polarity, bottom in (("ON", .51), ("OFF", .04)):
        ax = fig.add_axes((.56, bottom, .35, .40))
        plot_annular(ax, responses[polarity], f"{polarity} Receptive Field Map")
        add_annular_outline(ax, responses[polarity], masks[polarity])
    fig.suptitle(f"M22 · 261008 · Unit {unit_id}", x=.05, ha="left", fontsize=22, weight="bold")
    fig.text(.05, .025, "Probe A | HD S9 | RF S2, 0–200 ms | Red: saved excitatory RF", fontsize=10)
    return fig


def region_psth_data(rates, mask, wrap_x=True):
    mask = np.asarray(mask, dtype=bool)
    labels, sizes = _label_sections(np.ones(mask.shape), mask, wrap_x=wrap_x)
    # Retain the full saved -100..400 ms range; use the original slide's 10-ms bins.
    edges = rates.time_bin_edges_s
    display_edges = np.linspace(-.1, .4, 51)
    indices = np.array([np.argmin(abs(edges - value)) for value in display_edges])
    np.testing.assert_allclose(edges[indices], display_edges, atol=1e-9, rtol=0)
    np.testing.assert_allclose(edges[[0, -1]], display_edges[[0, -1]], atol=1e-9, rtol=0)
    time_ms = (display_edges[:-1] + display_edges[1:]) * 500
    regions = []
    for index in range(1, len(sizes) + 1):
        native_values = rates.spike_counts[labels == index]
        values = np.stack([(native_values[:, left:right] * np.diff(edges)[left:right]).sum(axis=1)
                           / (edges[right] - edges[left])
                           for left, right in zip(indices[:-1], indices[1:])], axis=1)
        mean = np.nanmean(values, axis=0)
        n = np.isfinite(values).sum(axis=0)
        sem = np.nanstd(values, axis=0, ddof=1) / np.sqrt(n)
        regions.append(dict(region=index, positions=int((labels == index).sum()), mean=mean, sem=sem))
    return time_ms, regions


def region_psth_figure(rates, mask, unit_id, wrap_x=True):
    time_ms, regions = region_psth_data(rates, mask, wrap_x)
    fig, ax = plt.subplots(figsize=(7.8, 4.4), layout="constrained", facecolor="white")
    ax.set_facecolor("white")
    # Original slide's reference window, not an inferred stimulus duration.
    ax.axvspan(0, 100, color=".92", zorder=0)
    ax.axvline(0, color=".45", linewidth=.8)
    ax.set(xlim=tuple(rates.time_bin_edges_s[[0, -1]] * 1000),
           xlabel="Time from stimulus onset (ms)", ylabel="Firing rate (Hz)")
    for region in regions:
        color = plt.get_cmap("tab10")(region['region'] - 1)
        mean, sem = region['mean'], region['sem']
        ax.fill_between(time_ms, mean - sem, mean + sem, color=color, alpha=.18, linewidth=0)
        ax.plot(time_ms, mean, color=color, linewidth=1.8,
                label=f"RF {region['region']}: {region['positions']} positions")
    ax.legend(title="Mean ± SEM across RF positions", fontsize=9, title_fontsize=9,
              facecolor="white", framealpha=1)
    ax.set_title(f"M22 · 261008 · unit {unit_id} · saved excitatory ON regions")
    return fig


def cumulative_figure(response, counts, units):
    values = np.asarray(counts, dtype=float).copy()
    values[~measured_bins(response)] = np.nan
    maximum = int(np.nanmax(values))
    shades = np.linspace(0, 1, maximum + 1)
    cmap = ListedColormap(np.repeat(shades[:, None], 3, axis=1))
    norm = BoundaryNorm(np.arange(maximum + 2) - .5, cmap.N)
    fig, ax = plt.subplots(figsize=(8.8, 3.4), layout="constrained", facecolor="white")
    image = draw_rectangle(ax, values, cell_edges(response.x_positions),
                           cell_edges(response.y_positions), cmap, norm)
    colorbar = fig.colorbar(image, ax=ax, ticks=np.arange(maximum + 1), label="RF count", shrink=.75)
    colorbar.solids.set_rasterized(False)
    ax.set_title("M22 · 261008 · cumulative saved ON masks\nUnits " + ", ".join(map(str, units)))
    return fig
