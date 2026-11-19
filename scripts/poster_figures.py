"""Pure plotting helpers for poster.ipynb; input loading and saving stay there."""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np

from scripts.export_m22_f3_examples import cell_edges
from scripts.export_m22_rf_sample_variants import add_rectangle_outline, annular_figure, draw_rectangle, gray_palette
from scripts.export_m22_ebb_samples import EBB_BANDS
from scripts.slides_v9_figures import count_coverage_figure
from Utils.plotting import plot_egocentric_polar


def centered_elevation(y_positions):
    """Upward-positive degrees relative to the middle stimulus row's center."""
    y = np.asarray(y_positions, dtype=float)
    return y[len(y) // 2] - y


def center_rectangle_ticks(axis, y_positions):
    y = np.asarray(y_positions)
    rows = [0, len(y) // 2, len(y) - 1]
    elevation = centered_elevation(y)
    axis.set_yticks(y[rows], [f"{value:+g}" if value else "0" for value in elevation[rows]])
    axis.set_ylabel("Elevation (°)")


def horizontal_rate_colorbar(fig, image, box, *, labelsize=14):
    bar = fig.colorbar(image, cax=fig.add_axes(box), orientation='horizontal')
    bar.set_ticks([0, image.norm.vmax], labels=['0', f'{image.norm.vmax:.1f}'])
    bar.ax.tick_params(labelsize=labelsize, length=3, pad=3)
    bar.ax.text(1.10, .5, 'Hz', transform=bar.ax.transAxes,
                ha='left', va='center', fontsize=labelsize + 2)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    return bar


def rf_rate_figure(response, mask, unit_id, polarity, *, cohort='M22 · 261008'):
    width, height = 9.5, 3.35
    fig = plt.figure(figsize=(width, height), facecolor='white')
    axis = fig.add_axes(np.array([1., .85, 8., 8. * 84 / 360]) / [width, height, width, height])
    axis.set_gid('rf_rectangle_8in')
    values = response.to_2d_array()
    palette, norm = gray_palette(values, vmin=0)
    image = draw_rectangle(axis, values, cell_edges(response.x_positions),
                           cell_edges(response.y_positions), palette, norm)
    add_rectangle_outline(axis, response, mask)
    center_rectangle_ticks(axis, response.y_positions)
    axis.set_xticks([-180, 0, 180])
    axis.set_title(f'Unit {unit_id} · {polarity}', fontsize=18, pad=12)
    axis.set_xlabel('')
    fig.text(1. / width, .28 / height, 'Azimuth (°)', fontsize=15)
    horizontal_rate_colorbar(fig, image, np.array([5.9, .32, 2.6, .10]) / [width, height, width, height])
    return fig


def rf_polar_figure(response, mask, unit_id, polarity, *, cohort='M22 · 261008'):
    y = response.y_positions
    fig = annular_figure(response, mask, unit_id, polarity,
                         elevation_origin=y[len(y) // 2], vmin=0)
    axis = fig.axes[0]
    fig.axes[-1].remove()
    fig.set_layout_engine(None)
    axis.set_position([.08, .22, .84, .68])
    if axis.get_legend() is not None:
        axis.get_legend().remove()
    axis.set_title(f'Unit {unit_id} · {polarity}', fontsize=18, pad=16)
    radius = 4 + len(y) - len(y) // 2 - .5
    axis.text(radius, .35, "0°", ha="center", va="bottom", fontsize=8,
              bbox=dict(facecolor="white", edgecolor="none", pad=1))
    palette, norm = gray_palette(response.to_2d_array(), vmin=0)
    horizontal_rate_colorbar(fig, plt.cm.ScalarMappable(norm=norm, cmap=palette), [.56, .065, .30, .016])
    return fig


def ebb_polar_figure(rate_map, theta_edges, distance_edges, unit_id, *, guides=False, mouse='M22'):
    fig, axis = plt.subplots(figsize=(6.8, 6.3), subplot_kw={"projection": "polar"},
                              layout="constrained", facecolor="white")
    image = plot_egocentric_polar(axis, theta_edges, distance_edges, rate_map,
                                 colorbar_label="Firing rate (Hz)")
    image.set_clim(float(np.nanmin(rate_map)), float(np.nanmax(rate_map)))
    image.cmap = image.cmap.copy()
    image.cmap.set_bad("#e6e8eb")
    image.set_rasterized(False)
    image.set_edgecolor('face')
    image.set_linewidth(.25)
    image.set_antialiased(False)
    image.colorbar.solids.set_rasterized(False)
    axis.set_facecolor("white")
    axis.grid(False)
    axis.yaxis.set_zorder(6)
    outer = float(distance_edges[-1])
    axis.set_yticks([8, 16, outer], labels=["8", "16", f'{outer:g}' if outer.is_integer() else f'{outer:.1f}'])
    axis.set_title(f"{mouse} · unit {unit_id} · EBB", pad=24)
    if guides:
        theta = np.linspace(0, 2 * np.pi, 721)
        for distance in (8, 16):
            # Physical cm; these guides do not change the native bin boundaries.
            axis.plot(theta, np.full_like(theta, distance), color="#cf381c",
                       linewidth=2, linestyle=(0, (1, 1.7)), zorder=4)
    for label in axis.get_yticklabels():
        label.set_zorder(5)
    return fig


def comparison_order(table):
    peaks = table.idxmax(axis=1).to_numpy(float)
    signed = (peaks + 180) % 360 - 180
    return table.index[np.argsort(signed, kind="stable")]


def comparison_figure(profiles, order, *, kind):
    """One shared unit order across the reference and three EBB distances."""
    tables = [(kind, "Visual RF" if kind == "RF" else "HD")]
    tables.extend((band, f"EBB {label}") for band, label in EBB_BANDS)
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 6.7), sharey=True,
                             layout="constrained", facecolor="white")
    for column, (axis, (key, title)) in enumerate(zip(axes, tables)):
        table = profiles[key].loc[order]
        angles = (table.columns.to_numpy(float) + 180) % 360 - 180
        columns = np.argsort(angles, kind="stable")
        image = axis.pcolormesh(cell_edges(angles[columns]), np.arange(len(order) + 1),
                                table.to_numpy()[:, columns], cmap="viridis",
                                vmin=0, vmax=1, shading="flat", rasterized=False,
                                antialiased=False)
        axis.set(xlim=(-180, 180), ylim=(len(order), 0), title=title,
                 xlabel="HD direction (°)" if key == "HD" else "Bearing (°)",
                 facecolor="white")
        ticks = [-180, -90, 0, 90, 180]
        axis.set_xticks(ticks, ["180", "270", "0", "90", "180"] if key == "HD"
                         else ["−180", "−90", "0", "90", "180"])
        axis.tick_params(axis="x", labelsize=10)
        if column == 0:
            rows = np.unique([0, (len(order) - 1) // 2, len(order) - 1])
            axis.set_yticks(rows + .5, [str(row + 1) for row in rows])
            axis.set_ylabel("Cell rank")
        else:
            axis.tick_params(axis="y", left=False)
    bar = fig.colorbar(image, ax=axes, fraction=.025, pad=.025,
                       ticks=[0, 1], label="Peak-normalized response")
    bar.ax.set_yticklabels(['0', 'max'])
    bar.solids.set_rasterized(False)
    fig.suptitle(f"M22 · {len(order)} cells · sorted by {kind} peak", fontsize=16)
    return fig


def example_zscore_figure(curve, x, y, on_z, off_z, on_mask):
    fig = plt.figure(figsize=(14.5, 6.2), facecolor="white")
    axis = fig.add_axes([.075, .21, .24, .60], projection="polar")
    angles = curve.index.to_numpy(float) % 360
    order = np.argsort(angles)
    theta = np.deg2rad(np.r_[angles[order], angles[order][0] + 360])
    rates = np.r_[curve.to_numpy()[order], curve.to_numpy()[order][0]]
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    axis.plot(theta, rates, color="black", lw=2)
    axis.fill_between(theta, 0, rates, color=".85")
    axis.set_thetagrids([0, 90, 180, 270], ["0°", "90°", "180°", "270°"])
    axis.tick_params(pad=8, labelsize=20)
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.set_ylim(0, np.ceil(rates.max() / 10) * 10)
    axis.grid(color=".8", lw=.9)
    axis.set_title("HD tuning (Hz)", pad=24)
    xe, ye = cell_edges(x), cell_edges(centered_elevation(y)[::-1])
    for label, values, bottom in (("ON", on_z, .61), ("OFF", off_z, .14)):
        axis = fig.add_axes([.45, bottom, .42, .235])
        limit = float(np.ceil(np.nanmax(abs(values))))
        image = axis.pcolormesh(xe, ye, values[::-1], cmap="RdBu_r", vmin=-limit,
                                vmax=limit, shading="flat", antialiased=False)
        if label == "ON":
            for row, col in np.argwhere(on_mask[::-1]):
                axis.add_patch(Rectangle((xe[col], ye[row]), np.diff(xe)[col], np.diff(ye)[row],
                                         fill=False, edgecolor="black", linewidth=1.4))
        axis.set(xlim=(xe[0], xe[-1]), ylim=(ye[0], ye[-1]), ylabel="Elevation (°)", title=label + " RF")
        axis.set_xticks([-180, 0, 180])
        axis.set_yticks([-36, 0, 36], ["−36", "0", "+36"])
        if label == "OFF":
            axis.set_xlabel("Visual azimuth (°)")
        else:
            axis.tick_params(labelbottom=False)
        bar = fig.colorbar(image, cax=fig.add_axes([.895, bottom, .016, .235]), label="z-score")
        bar.set_ticks([-limit, 0, limit])
        bar.solids.set_rasterized(False)
    return fig


def rotation_figure(data):
    width, height = 14.3, 4.2
    fig = plt.figure(figsize=(width, height), facecolor='white')
    def box(values):
        return np.asarray(values) / [width, height, width, height]
    x_starts, map_width = [.7, 5.2, 9.7], 3.75
    map_height = map_width * 84 / 360
    headers = [('Head-fixed', 'Screen coordinates'), ('Rotating', 'Screen coordinates'),
               ('Rotating', 'Animal coordinates')]
    for left, (title, subtitle) in zip(x_starts, headers):
        fig.text((left + map_width / 2) / width, 3.92 / height, title, ha='center', fontsize=20)
        fig.text((left + map_width / 2) / width, 3.60 / height, subtitle, ha='center', fontsize=16)
    units = list(dict.fromkeys(unit for unit, _ in data))
    if len(units) != 2:
        raise ValueError('The rotation panel has two example rows')
    for unit, bottom in zip(units, (2.25, .70)):
        fig.text(x_starts[0] / width, (bottom + map_height + .14) / height,
                 f'Unit {unit}', fontsize=16)
        for col, key in enumerate(('head_fixed', 'screen', 'animal')):
            response = data[unit, key]
            values = response.to_2d_array()
            axis = fig.add_axes(box([x_starts[col], bottom, map_width, map_height]))
            axis.set_gid(f'rotation_{unit}_{key}')
            xe = cell_edges(response.x_positions)
            ye = cell_edges(centered_elevation(response.y_positions)[::-1])
            image = axis.pcolormesh(xe, ye, values[::-1], cmap='gray', vmin=0,
                                    vmax=np.nanmax(values), shading='flat')
            axis.set(xlim=(-180, 180), ylim=(ye[0], ye[-1]), aspect='equal')
            axis.set_xticks([-180, 0, 180])
            axis.set_yticks([-36, 0, 36], ['−36', '0', '+36'])
            axis.tick_params(labelsize=14, length=3, pad=3)
            if col:
                axis.set_yticklabels([])
            horizontal_rate_colorbar(fig, image,
                box([x_starts[col] + 2., bottom - .40, 1.35, .07]), labelsize=11)
    fig.text(.20 / width, .5, 'Elevation (°)', rotation=90, ha='center', va='center', fontsize=16)
    fig.text(.7 / width, .12 / height, 'Azimuth (°)', fontsize=15)
    return fig


def coverage_figure(data):
    """Use the slides' distinct-cell marginals with the poster's centered ticks."""
    fig = count_coverage_figure(data)
    axis, top, right, colorbar_axis = fig.axes
    axis.set_yticks([-36, 0, 36], ["+36", "0", "−36"])
    axis.get_xticklabels()[-1].set_horizontalalignment('right')
    top.set_ylabel('Cell count')
    right.set_xlabel('Cell count')
    colorbar_axis.set_ylabel('Cell count')
    axis.texts[0].set_fontsize(14)
    top.yaxis.set_major_locator(MaxNLocator(3, integer=True))
    right.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    for current in (axis, top, right):
        current.tick_params(labelsize=14)
        current.xaxis.label.set_fontsize(16)
        current.yaxis.label.set_fontsize(16)
    colorbar_axis.tick_params(labelsize=14)
    colorbar_axis.yaxis.label.set_fontsize(16)
    return fig
