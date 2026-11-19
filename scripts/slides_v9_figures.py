"""Pure figure/data helpers for the figures used by WorkInProgress v9.

The slides notebook owns loading and saving. Native RF counts, saved masks and
angular coordinates remain separate from display normalization and smoothing.
"""

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Affine2D
import numpy as np
from scipy.ndimage import gaussian_filter

from Utils.direction_comparison import paired_peak_angles, profile_coordinates
from Utils.statistic_utils import rayleigh_test
from scripts.export_m22_f3_examples import cell_edges, plot_annular, plot_hd
from scripts.export_m22_rf_sample_variants import (
    add_annular_outline, annular_figure, draw_rectangle, gray_palette, rectangle_axes,
)


POLAR_BLUE = "#266c9b"


def example_figure(unit_id, hd_curve, responses, mask):
    """Show the same ON/OFF rate cells in rectangular and annular coordinates."""
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
    hd_axis = fig.add_axes((.055, .23, .27, .52), projection="polar")
    plot_hd(hd_axis, hd_curve)
    hd_axis.set_title("HD tuning (Hz)", pad=20)
    for polarity, bottom in (("ON", .55), ("OFF", .12)):
        response = responses[polarity]
        values = response.to_2d_array()
        palette, norm = gray_palette(values)
        rect = fig.add_axes((.39, bottom + .025, .35, .19))
        draw_rectangle(rect, values, cell_edges(response.x_positions),
                       cell_edges(response.y_positions), palette, norm)
        rect.set_title(f"{polarity} RF", pad=8)
        rect.set(xlabel="Azimuth (°)", ylabel="Elevation (°)")
        annular = fig.add_axes((.77, bottom - .065, .19, .37))
        plot_annular(annular, response, "")
        for label in annular.texts:
            label.set_fontsize(9)
            label.set_position(np.asarray(label.get_position()) * 1.15)
        annular.set(xlim=(-14.5, 14.5), ylim=(-14.5, 14.5))
        if polarity == "ON":
            for row, column in np.argwhere(mask):
                xe, ye = cell_edges(response.x_positions), cell_edges(response.y_positions)
                rect.add_patch(plt.Rectangle((xe[column], ye[row]), np.diff(xe)[column],
                                            np.diff(ye)[row], fill=False,
                                            edgecolor="#D62728", linewidth=1.1))
            add_annular_outline(annular, response, mask)
    fig.text(.04, .94, f"Unit {unit_id}", fontsize=22, weight="bold")
    return fig


def rectangle_panel_figure(response):
    """Export the rectangular rate view beside the existing annular slide view."""
    values = response.to_2d_array()
    palette, norm = gray_palette(values)
    fig, axis = plt.subplots(figsize=(6.5, 2.6), layout="constrained", facecolor="white")
    draw_rectangle(axis, values, cell_edges(response.x_positions),
                   cell_edges(response.y_positions), palette, norm)
    axis.set(xlabel="Azimuth (°)", ylabel="Elevation (°)")
    return fig


def cumulative_mask_figure(response, mask, cumulative, unit_id):
    """Keep individual masks binary and give the combined count map its scale."""
    fig = plt.figure(figsize=(13.33, 7.5), facecolor="white")
    xe, ye = cell_edges(response.x_positions), cell_edges(response.y_positions)
    maximum = int(cumulative.max())
    for values, bottom, label, norm in (
        (mask, .55, f"Unit {unit_id}", Normalize(0, 1)),
        (cumulative, .13, "Combined masks", Normalize(0, maximum)),
    ):
        axis = fig.add_axes((.15, bottom, .70, .23))
        image = draw_rectangle(axis, values, xe, ye, "gray", norm)
        axis.set(xlabel="Azimuth (°)", ylabel="Elevation (°)")
        axis.text(-.13, .5, label, transform=axis.transAxes, ha="right", va="center",
                  fontsize=13, weight="bold")
    bar = fig.colorbar(image, cax=fig.add_axes((.88, .13, .015, .23)),
                       ticks=np.arange(maximum + 1), label="RF count")
    bar.solids.set_rasterized(False)
    return fig


def prepare_count_coverage(result, measured, *, sigma_deg=6., rows=None, unit_ids=None):
    """Smooth the map and count distinct RF-positive units per column and row."""
    x = np.asarray(result.manifest["x_positions"], dtype=float)
    y = np.asarray(result.manifest["y_positions"], dtype=float)
    masks = result.mask_2d
    if unit_ids is not None:
        masks = masks[np.isin(result.unit_ids, unit_ids)]
    if rows is not None:
        y, measured, masks = y[rows], measured[rows], masks[:, rows]
    n_rf = int(masks.any(axis=(1, 2)).sum())
    xe, ye = cell_edges(x), cell_edges(y)
    raw = masks.sum(axis=0).astype(float)
    raw[~measured] = np.nan
    sigma = (sigma_deg / np.diff(y)[0], sigma_deg / np.diff(x)[0])
    mode = ("reflect", "wrap" if np.isclose(xe[-1] - xe[0], 360) else "reflect")
    weights = gaussian_filter(measured.astype(float), sigma, mode=mode)
    numerator = gaussian_filter(np.where(measured, raw, 0), sigma, mode=mode)
    values = np.divide(numerator, weights, out=np.full_like(raw, np.nan),
                       where=measured & (weights > 0))
    # A unit spanning several heights/azimuths contributes only once per column/row.
    masks = masks & measured[None, :, :]
    azimuth_count = masks.any(axis=1).sum(axis=0).astype(float)
    elevation_count = masks.any(axis=2).sum(axis=0).astype(float)
    azimuth_count[~measured.any(axis=0)] = np.nan
    elevation_count[~measured.any(axis=1)] = np.nan
    return dict(x=x, y=y, x_edges=xe, y_edges=ye, raw_count=raw, display_count=values,
                azimuth_count=azimuth_count, elevation_count=elevation_count,
                n_rf=n_rf,
                n_assessed=len(masks))


def pool_count_coverage(panels):
    """Sum piecewise-constant native counts on the exact union of their edges."""
    xe = np.unique(np.concatenate([panel["x_edges"] for panel in panels]))
    ye = np.unique(np.concatenate([panel["y_edges"] for panel in panels]))
    x, y = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2
    pooled = {key: np.zeros((len(y), len(x))) for key in ("raw_count", "display_count")}
    support = np.zeros_like(pooled["raw_count"])
    covered = np.zeros_like(support, dtype=bool)
    azimuth_count, elevation_count = np.zeros(len(x)), np.zeros(len(y))
    for panel in panels:
        ix = np.searchsorted(panel["x_edges"], x, side="right") - 1
        iy = np.searchsorted(panel["y_edges"], y, side="right") - 1
        inside = ((iy[:, None] >= 0) & (iy[:, None] < len(panel["y"]))
                  & (ix[None, :] >= 0) & (ix[None, :] < len(panel["x"])))
        rows = np.clip(iy, 0, len(panel["y"]) - 1)[:, None]
        columns = np.clip(ix, 0, len(panel["x"]) - 1)[None, :]
        valid = inside & np.isfinite(panel["raw_count"][rows, columns])
        support += valid * panel["n_rf"]
        covered |= valid
        for key in pooled:
            pooled[key] += np.where(valid, panel[key][rows, columns], 0)
        azimuth_count += np.where(valid.any(axis=0), panel["azimuth_count"][columns[0]], 0)
        elevation_count += np.where(valid.any(axis=1), panel["elevation_count"][rows[:, 0]], 0)
    for values in pooled.values():
        values[~covered] = np.nan
    azimuth_count[~covered.any(axis=0)] = np.nan
    elevation_count[~covered.any(axis=1)] = np.nan
    return dict(x=x, y=y, x_edges=xe, y_edges=ye, **pooled, support=support,
                azimuth_count=azimuth_count, elevation_count=elevation_count,
                n_rf=sum(panel["n_rf"] for panel in panels),
                n_assessed=sum(panel["n_assessed"] for panel in panels))


def count_coverage_figure(data):
    """Show the smoothed coverage map with distinct-unit count marginals."""
    x, y, xe, ye = (data[key] for key in ("x", "y", "x_edges", "y_edges"))
    values = data["display_count"]
    horizontal, vertical = data["azimuth_count"], data["elevation_count"]
    fig = plt.figure(figsize=(11.4, 4.2), facecolor="white")
    height = 8 * (ye[-1] - ye[0]) / (xe[-1] - xe[0])

    def axes(left, bottom, width, height, **kwargs):
        return fig.add_axes(np.array([left, bottom, width, height]) / [11.4, 4.2, 11.4, 4.2],
                            **kwargs)

    axis = axes(.65, .5, 8, height)
    top = axes(.65, .5 + height + .12, 8, 1.35, sharex=axis)
    right = axes(8.85, .5, 1.45, height, sharey=axis)
    cax = axes(10.52, .5, .16, height)
    image = axis.pcolormesh(xe, ye, np.ma.masked_invalid(values), cmap="Greys", vmin=0,
                            shading="flat", antialiased=False, rasterized=False)
    rectangle_axes(axis, xe, ye)
    axis.set(xlabel="Visual field azimuth (°)", ylabel="Elevation (°)")
    axis.text(.985, .06, f"n = {data['n_rf']}", transform=axis.transAxes, ha="right")
    top.fill_between(x, horizontal, color=".85")
    top.plot(x, horizontal, "o-", color="black", markersize=2.5, lw=1.3)
    top.set(ylabel="RF count", ylim=(0, None))
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    right.fill_betweenx(y, vertical, color=".85")
    right.plot(vertical, y, "o-", color="black", markersize=2.5, lw=1.3)
    right.set(xlabel="RF count", xlim=(0, None))
    right.tick_params(axis="y", left=False, labelleft=False)
    for current in (axis, top, right):
        current.spines[["top", "right"]].set_visible(False)
    bar = fig.colorbar(image, cax=cax, label="RF count")
    bar.solids.set_rasterized(False)
    return fig


def sorting_figure(profiles, *, kind):
    """Draw the prepared row order directly, with black representing zero."""
    width = 7.4 if kind == "HD" else 6
    fig = plt.figure(figsize=(width, 7.5), facecolor="white")
    left = .17 if kind == "HD" else .035
    # Keep the heatmap's physical size while making room for the HD colorbar.
    axis = fig.add_axes((left * 6 / width, .08, .765 * 6 / width, .88))
    axis.set_facecolor("white")
    angles = profiles.columns.to_numpy(dtype=float)
    if kind == "HD":
        coordinates = profile_coordinates(profiles)
        start = coordinates["ticks"][0]
        angles = (angles - start) % 360 + start
    order = np.argsort(angles, kind="stable")
    edges = cell_edges(angles[order])
    image = axis.pcolormesh(edges, np.arange(len(profiles) + 1),
                            profiles.to_numpy()[:, order], cmap="gray", vmin=0, vmax=1,
                            shading="flat", antialiased=False, rasterized=False)
    axis.set(xlim=(edges[0], edges[-1]), ylim=(len(profiles), 0),
             xlabel="HD direction (°)" if kind == "HD" else "Visual RF direction (°)")
    if kind == "HD":
        axis.set_xticks(coordinates["ticks"], labels=coordinates["ticklabels"])
        # This panel contains one recording/probe; show its saved numeric unit IDs.
        unit_ids = [str(key).rsplit(":", 1)[-1] for key in profiles.index]
        axis.set_yticks(np.arange(len(profiles)) + .5, labels=unit_ids)
        axis.set_ylabel("Unit ID")
        axis.tick_params(axis="x", labelsize=30)
        axis.tick_params(axis="y", labelsize=26)
        axis.xaxis.label.set_fontsize(32)
        axis.yaxis.label.set_fontsize(32)
    else:
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.set_yticks([])
    bar_left = 5.85 / width if kind == "HD" else .825
    bar = fig.colorbar(image, cax=fig.add_axes((bar_left, .08, .15 / width, .88)),
                       ticks=[0, 1], label=("Peak-normalized\nfiring rate" if kind == "HD"
                                           else "Peak-normalized response"))
    bar.ax.set_yticklabels(["0", "max"])
    if kind == "HD":
        bar.ax.tick_params(labelsize=30)
        bar.ax.yaxis.label.set_fontsize(30)
    bar.solids.set_rasterized(False)
    return fig


def _direction_xy(degrees):
    """Cartesian unit vector for the plots' clockwise, zero-at-north angles."""
    angle = np.deg2rad(degrees)
    return np.array([np.sin(angle), np.cos(angle)])


def _direction_circle(*, title=None):
    fig, axis = plt.subplots(figsize=(4.8, 4.8), facecolor="white")
    axis.set_facecolor("white")
    axis.set(aspect="equal", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    axis.set_axis_off()
    axis.add_patch(Circle((0, 0), 1, fill=False, edgecolor="#999999", linewidth=1.1))
    for angle in (0, 90, 180, 270):
        point = 1.5 * _direction_xy(angle)
        axis.text(*point, f"{angle}°", ha="center", va="center", fontsize=15,
                  color="#202020")
    if title:
        axis.set_title(title, fontsize=14, color="#202020", pad=12)
    return fig, axis


def _direction_arrow(axis, origin, degrees, *, color="#202020"):
    arrow = FancyArrowPatch(origin, origin + .38 * _direction_xy(degrees),
                            arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
                            color=color, shrinkA=0, shrinkB=0, zorder=3)
    axis.add_patch(arrow)
    return arrow


def _schematic_mouse(axis, heading, *, scale=1.):
    """Draw the schematic mouse with vector paths, without an image asset."""
    transform = Affine2D().scale(scale).rotate_deg(-heading) + axis.transData
    body = Path([(0, .12), (-.05, .10), (-.13, -.04), (-.06, -.07),
                 (-.03, -.09), (.08, -.09), (.08, -.03),
                 (.08, .02), (.03, .10), (0, .12), (0, .12)],
                [Path.MOVETO] + [Path.CURVE4] * 9 + [Path.CLOSEPOLY])
    tail = Path([(.01, -.08), (.03, -.12), (-.07, -.17), (.02, -.20)],
                [Path.MOVETO] + [Path.CURVE4] * 3)
    for path, fill in ((body, "white"), (tail, "none")):
        axis.add_patch(PathPatch(path, transform=transform, facecolor=fill,
                                 edgecolor="#202020", linewidth=2.2,
                                 capstyle="round", joinstyle="round", zorder=4))
    for side in (-1, 1):
        axis.plot([side * .02, side * .065], [.065, .11], transform=transform,
                  color="#202020", linewidth=2.2, solid_capstyle="round", zorder=4)


def sample_rf_polar_figure(response, mask, unit_id):
    """Keep native RF cells and place a forward-facing mouse in the empty center."""
    fig = annular_figure(response, mask, unit_id)
    _schematic_mouse(fig.axes[0], heading=0, scale=14.)
    return fig


def schematic_figure(kind):
    """Illustrative HD/RF geometry; fixed example angles are not measurements."""
    fig, axis = _direction_circle()
    # These examples reproduce the north and west positions of the slide sketch.
    point = _direction_xy(-6)
    axis.plot([0, point[0]], [0, point[1]], color="#b8b8b8", linewidth=.9)
    if kind in ("rf", "combined"):
        axis.plot(*point, "o", color="#202020", markersize=5)
        _direction_arrow(axis, point, 48)
    if kind == "hd":
        axis.text(-.02, .46, "HD", fontsize=15, color="#202020")
    elif kind == "rf":
        axis.text(.16, 1.01, "RF", fontsize=15, color="#202020")
    elif kind == "combined":
        west = _direction_xy(270)
        axis.plot([0, west[0]], [0, west[1]], color="#b8b8b8", linewidth=.9)
        axis.plot(*west, "o", color="#202020", markersize=5)
        for angle in (336, 36):
            _direction_arrow(axis, west, angle)
    else:
        raise ValueError(f"Unknown schematic: {kind}")
    _schematic_mouse(axis, 270 if kind == "combined" else 0)
    return fig


def region_positions_figure(region_pairs, *, highlighted=False):
    """Place each region at its HD peak and point its arrow along HD + RF X."""
    fig, axis = _direction_circle(title="Position: HD peak")
    for row in region_pairs.itertuples(index=False):
        origin = _direction_xy(row.hd_peak_deg)
        axis.plot([0, origin[0]], [0, origin[1]], color="#b8b8b8", linewidth=.8)
        axis.plot(*origin, "o", color="#202020", markersize=5, zorder=2)
        delta = (row.direction_deg - 48 + 180) % 360 - 180
        color = "#D62728" if highlighted and abs(delta) <= 30 else "#202020"
        arrow = _direction_arrow(axis, origin, row.direction_deg, color=color)
        arrow.set_gid(f"region-arrow-{row.probe}-{row.unit_id}-{row.rf_region}")
    return fig


def plot_region_counts(axis, stats):
    """Draw saved-center counts with the same 24-degree bins on every mouse."""
    axis.set_facecolor("white")
    edges = np.deg2rad(stats["edges_deg"])
    counts = np.asarray(stats["counts"])
    axis.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges),
             color=POLAR_BLUE, edgecolor="white", linewidth=.7)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_thetagrids([0, 90, 180, 270], ["0°", "90°", "180°", "270°"])
    step = max(1, int(np.ceil(counts.max() / 3)))
    axis.set_yticks(np.arange(1, 4) * step)
    axis.set_ylim(0, step * 3 * 1.05)
    axis.set_xlabel("RF region count", labelpad=18, color="#202020")
    axis.tick_params(colors="#202020")
    axis.spines["polar"].set_color("#202020")
    axis.grid(color="#d2d7df", linewidth=.8)


def region_histogram_figure(stats):
    """Plot the prepared region counts and nominal region-level Rayleigh result."""
    fig, axis = plt.subplots(figsize=(4.8, 5.4), subplot_kw={"projection": "polar"},
                              layout="constrained", facecolor="white")
    plot_region_counts(axis, stats)
    axis.set_title(f"Rayleigh score = {stats['rayleigh_score']:.2f}\n"
                   f"p = {stats['rayleigh_p_nominal']:.5f}", pad=24, color="#202020")
    return fig


def seven_mice_region_figure(statistics):
    """Show one direction per RF region, retaining each mouse's count scale."""
    fig = plt.figure(figsize=(16, 7.8), facecolor="white")
    for index, (mouse, stats) in enumerate(statistics.items()):
        row, column = divmod(index, 4)
        axis = fig.add_axes((.035 + column * .25, .57 - row * .48, .18, .31),
                            projection="polar")
        plot_region_counts(axis, stats)
        axis.set_title(f"{mouse.upper()}\n{stats['n_regions']} RF regions / "
                       f"{stats['n_cells']} cells", pad=18, color="#202020")
        axis.set_xlabel("")
    return fig


def peak_sum_data(hd, rf, *, bins=15):
    """One native maximum per paired cell, as in tc_comparison_pairs.ipynb."""
    peaks = paired_peak_angles(hd, rf)
    peaks = peaks.dropna(subset=["reference_peak_deg", "matched_peak_deg"]).copy()
    peaks["sum_deg"] = (peaks.reference_peak_deg + peaks.matched_peak_deg) % 360
    counts, edges = np.histogram(np.deg2rad(peaks.sum_deg), bins=bins, range=(0, 2 * np.pi))
    centers = np.rad2deg((edges[:-1] + edges[1:]) / 2)
    score, p = rayleigh_test(counts, np.ones_like(counts), centers)
    return dict(peaks=peaks, counts=counts, edges=edges, rayleigh_score=score, rayleigh_p=p)


def plot_peak_sum(axis, data, title):
    """Draw prepared cell counts without treating multiple RF regions as cells."""
    edges = data["edges"]
    axis.bar((edges[:-1] + edges[1:]) / 2, data["counts"], width=np.diff(edges),
             color=POLAR_BLUE, edgecolor="white", linewidth=.7)
    axis.set_facecolor("white")
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.set_thetagrids([0, 90, 180, 270], ["0°", "90°", "180°", "270°"])
    axis.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    axis.set_title(f"{title}\nn = {int(data['counts'].sum())} cells", pad=18)
    axis.set_xlabel("Cell count", labelpad=14)
    axis.tick_params(colors="#202020")
    axis.spines["polar"].set_color("#202020")
    axis.grid(color="#d2d7df", linewidth=.8)


def peak_sum_figure(data, *, title="HD + RF peak-bin directions"):
    fig, axis = plt.subplots(figsize=(5.2, 5.6), subplot_kw={"projection": "polar"},
                              layout="constrained", facecolor="white")
    plot_peak_sum(axis, data, title)
    return fig


def seven_mice_figure(data):
    """Keep the requested mouse order, with an independent integer count scale."""
    fig = plt.figure(figsize=(16, 7.2), facecolor="white")
    for index, (mouse, values) in enumerate(data.items()):
        row, column = divmod(index, 4)
        axis = fig.add_axes((.035 + column * .25, .57 - row * .48, .18, .31),
                            projection="polar")
        plot_peak_sum(axis, values, mouse.upper())
        axis.set_xlabel("")
    return fig
