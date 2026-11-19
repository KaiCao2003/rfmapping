"""Pure scientific panels arranged for the user's three-column poster sketch."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
from matplotlib.ticker import MaxNLocator
import numpy as np

from scripts.export_m22_f3_examples import cell_edges, plot_annular, plot_hd
from scripts.export_m22_rf_sample_variants import add_annular_outline, add_rectangle_outline, draw_rectangle, gray_palette
from scripts.poster_figures import center_rectangle_ticks, horizontal_rate_colorbar
from scripts.slides_v9_figures import _direction_xy, _schematic_mouse, region_positions_figure
from Utils.plotting import signed_egocentric_map


PSTH_EDGES = np.linspace(-.1, .4, 51)


def rf_position_rates(rates, mask):
    edges = rates.time_bin_edges_s
    indices = np.array([np.argmin(abs(edges - value)) for value in PSTH_EDGES])
    np.testing.assert_allclose(edges[indices], PSTH_EDGES, atol=1e-9, rtol=0)
    values = rates.spike_counts[np.asarray(mask, dtype=bool)]
    return np.stack([(values[:, left:right] * np.diff(edges)[left:right]).sum(axis=1)
                     / (edges[right] - edges[left])
                     for left, right in zip(indices[:-1], indices[1:])], axis=1)


def response_moments(values):
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0)
    return dict(n=finite.sum(axis=0), total=clean.sum(axis=0), squares=(clean * clean).sum(axis=0))


def response_mean_sem(moments):
    n, total, squares = (moments[key] for key in ('n', 'total', 'squares'))
    mean = total / n
    variance = (squares - total * mean) / (n - 1)
    return mean, np.sqrt(np.maximum(variance, 0) / n)


def draw_rf_pair(fig, on, off, on_mask, off_mask, *, unit_id, left=0):
    """ON/OFF rectangles and one ON annulus, following the supplied poster sketch."""
    width, height = fig.get_size_inches()
    def box(values):
        x, y, w, h = values
        return np.asarray([x + left, y, w, h]) / [width, height, width, height]
    def text(x, y, value, **kwargs):
        return fig.text((x + left) / width, y / height, value, **kwargs)

    map_height = 8. * 84 / 360
    text(.60, 6.0, f'Unit {unit_id}', fontsize=17)
    for response, mask, label, bottom in ((on, on_mask, 'ON', 3.6), (off, off_mask, 'OFF', .85)):
        axis = fig.add_axes(box([.95, bottom, 8., map_height]))
        axis.set_gid(f'rf_pair_{label.lower()}_rectangle')
        palette, norm = gray_palette(response.to_2d_array(), vmin=0)
        draw_rectangle(axis, response.to_2d_array(), cell_edges(response.x_positions),
                       cell_edges(response.y_positions), palette, norm)
        add_rectangle_outline(axis, response, mask)
        center_rectangle_ticks(axis, response.y_positions)
        axis.set(xlabel='', ylabel='', xticks=[-180, 0, 180])
        axis.tick_params(labelsize=17, pad=4)
        if label == 'ON':
            axis.tick_params(axis='x', bottom=False, labelbottom=False)
        title_gap = .29 if label == 'ON' else .13
        text(6.5, bottom + map_height + title_gap, label, fontsize=22, ha='center')
        bar_left = 1.25 if label == 'ON' else 6.15
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=palette),
                           cax=fig.add_axes(box([bar_left, bottom - .47, 2.6, .08])), orientation='horizontal')
        bar.set_ticks([0, norm.vmax], labels=['0', f'{norm.vmax:.1f}'])
        bar.ax.tick_params(labelsize=12, length=3, pad=3)
        bar.ax.text(1.09, .5, 'Hz', transform=bar.ax.transAxes, ha='left', va='center', fontsize=14)
        bar.solids.set_rasterized(False)
        bar.solids.set_edgecolor('face')
    text(.95, .23, 'Azimuth (°)', fontsize=18)
    text(.15, 3.15, 'Elevation (°)', rotation=90, ha='center', va='center', fontsize=16)

    polar_size = 3.8 * 26 / 22
    polar_box = box([9.21, 1.13, polar_size, polar_size])
    polar = fig.add_axes(polar_box)
    polar.set_gid('rf_pair_on_annulus')
    plot_annular(polar, on, '', vmin=0)
    fig.axes[-1].remove()
    polar.set_position(polar_box)
    add_annular_outline(polar, on, on_mask)
    _schematic_mouse(polar, heading=0, scale=14.)
    for label in polar.texts:
        label.set_fontsize(13)
        if label.get_text() == '±180°':
            label.set_text('+180°')

    # Rounded, stepped outline groups the ON rectangle with its annular view.
    from matplotlib.path import Path as PlotPath
    from matplotlib.patches import PathPatch
    l, r, top, bottom, step, middle, radius = .30, 13.76, 5.62, 1.02, 9.08, 3.36, .13
    vertices = [(l+radius, top), (r-radius, top), (r, top), (r, top-radius),
                (r, bottom+radius), (r, bottom), (r-radius, bottom),
                (step+radius, bottom), (step, bottom), (step, bottom+radius),
                (step, middle-radius), (step, middle), (step-radius, middle),
                (l+radius, middle), (l, middle), (l, middle+radius),
                (l, top-radius), (l, top), (l+radius, top), (l+radius, top)]
    codes = [PlotPath.MOVETO, PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3,
             PlotPath.LINETO, PlotPath.CURVE3, PlotPath.CURVE3, PlotPath.CLOSEPOLY]
    vertices = (np.asarray(vertices) + [left, 0]) / [width, height]
    fig.add_artist(PathPatch(PlotPath(vertices, codes), transform=fig.transFigure,
                             fill=False, edgecolor='#202020', linewidth=1.4, zorder=5))


def rf_map_figure(on, off, on_mask, off_mask, *, unit_id):
    fig = plt.figure(figsize=(14., 6.25), facecolor='white')
    draw_rf_pair(fig, on, off, on_mask, off_mask, unit_id=unit_id)
    return fig


def sample_overview_figure(curve, on, off, on_mask, off_mask, *, unit_id):
    width, height = 18.3, 6.25
    fig = plt.figure(figsize=(width, height), facecolor='white')
    axis = fig.add_axes(np.asarray([.6, 1.85, 3., 3.]) / [width, height, width, height], projection='polar')
    axis.set_gid('sample_hd_diameter_3in')
    plot_hd(axis, curve)
    axis.set_title('HD', pad=24, fontsize=24)
    axis.set_xlabel('')
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.tick_params(labelsize=16, pad=5)
    fig.text(2.1 / width, 1.3 / height, 'Hz', ha='center', fontsize=17)
    draw_rf_pair(fig, on, off, on_mask, off_mask, unit_id=unit_id, left=4.3)
    return fig

def timing_figure(sample, pooled, n_units, n_positions, *, unit_id=307):
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 3.1), layout='constrained', facecolor='white')
    time_ms = (PSTH_EDGES[:-1] + PSTH_EDGES[1:]) * 500
    for axis, moments, title in zip(axes, (sample, pooled),
                                    (f'Unit {unit_id} · RF positions', 'HD + ON RF · seven mice')):
        mean, sem = response_mean_sem(moments)
        axis.axvspan(0, 100, color='.92', zorder=0)
        axis.axvline(0, color='.5', linewidth=.8)
        axis.fill_between(time_ms, mean - sem, mean + sem, color='.8', linewidth=0)
        axis.plot(time_ms, mean, color='black', linewidth=2)
        axis.set(xlim=(-100, 400), xticks=[-100, 0, 200, 400],
                 xlabel='Time from onset (ms)', ylabel='Firing rate (Hz)', title=title)
        axis.yaxis.set_major_locator(MaxNLocator(3))
        axis.tick_params(labelsize=14)
        axis.xaxis.label.set_fontsize(15)
        axis.yaxis.label.set_fontsize(15)
        axis.title.set_fontsize(19)
        axis.spines[['top', 'right']].set_visible(False)
    axes[1].text(.97, .97, f'{n_positions:,} RF bins\n{n_units} cells', transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=16)
    return fig


def fraction_figure(counts, *, n_unassessed=0, denominator='assessed'):
    counts = counts.copy()
    if denominator == 'assessed':
        counts = counts.drop('RF unassessed', errors='ignore')
    elif denominator == 'all_hd_missing_as_zero':
        counts['0 ON RF'] += counts.pop('RF unassessed')
    total = int(counts.sum())
    fig = plt.figure(figsize=(7.5, 5.5), facecolor='white')
    axis = fig.add_axes([.04, .05, .92, .90])
    categories = [('>1 ON RF', '#205a91'), ('1 ON RF', '#9ccae5'), ('0 ON RF', '#e6e6e6')]
    if denominator == 'all_hd':
        categories.append(('RF unassessed', '#ffffff'))
    sizes = [int(counts[category]) for category, _ in categories]
    wedges, _ = axis.pie(sizes, colors=[color for _, color in categories],
        startangle=90, counterclock=False, wedgeprops=dict(edgecolor='white', linewidth=1.6))
    angle = np.deg2rad((wedges[0].theta1 + wedges[0].theta2) / 2)
    # Anchor the elbow inside the small multiple-RF sector.
    anchor = .84 * np.array([np.cos(angle), np.sin(angle)])
    elbow = np.array([1.30 * np.cos(angle), 1.30])
    axis.plot([anchor[0], elbow[0], .69], [anchor[1], elbow[1], elbow[1]],
              color='#202020', linewidth=3, solid_capstyle='butt', solid_joinstyle='miter')
    axis.text(.77, 1.30, f'2+ RFs: {100 * sizes[0] / total:.1f}%',
              ha='left', va='center', fontsize=26, color='#202020')
    axis.set(xlim=(-1.12, 2.65), ylim=(-1.12, 1.62), aspect='equal')
    axis.axis('off')
    return fig


def draw_free_moving_rf(axis, x, rates):
    # One measured azimuth row represents full-height bars, not seven elevation samples.
    palette, norm = gray_palette(rates, vmin=0)
    image = axis.pcolormesh(cell_edges(x), [0, 1], rates[None, :],
                            cmap=palette, norm=norm, shading='flat', antialiased=False)
    axis.set(xlim=(-180, 180), ylim=(0, 1), yticks=[], xticks=[-180, 0, 180],
             xlabel='Azimuth (°)', title='Visual RF')
    return image


def free_moving_figure(x, rates):
    width, height = 9.5, 3.35
    fig = plt.figure(figsize=(width, height), facecolor='white')
    axis = fig.add_axes(np.array([1., .85, 8., 8. * 84 / 360]) / [width, height, width, height])
    axis.set_gid('free_moving_rectangle_8in')
    image = draw_free_moving_rf(axis, x, rates)
    axis.set_xlabel('')
    fig.text(1. / width, .28 / height, 'Azimuth (°)', fontsize=15)
    horizontal_rate_colorbar(fig, image, np.array([5.9, .32, 2.6, .10]) / [width, height, width, height])
    return fig


def draw_ebb_polar(axis, rates, theta_edges, distance_edges, *, guides=True):
    axis.grid(False)
    palette = plt.get_cmap('viridis').copy()
    palette.set_bad('#e6e8eb')
    image = axis.pcolormesh(np.deg2rad(theta_edges), distance_edges, rates.T,
                            cmap=palette, vmin=np.nanmin(rates), vmax=np.nanmax(rates), shading='flat',
                            edgecolors='face', linewidth=.25, antialiased=False)
    axis.set_theta_zero_location('N')
    axis.set_theta_direction(-1)
    axis.set_xticks(np.deg2rad([0, 90, 180, 270]), ['0°', '90°', '180°', '−90°'])
    axis.set(ylim=(0, distance_edges[-1]), yticks=[8, 16])
    axis.set_yticklabels(['8', '16'])
    axis.set_rlabel_position(135)
    axis.yaxis.set_zorder(6)
    for label in axis.get_yticklabels():
        label.set_bbox(dict(facecolor='white', edgecolor='none', pad=.6))
        label.set_zorder(5)
    if guides:
        theta = np.linspace(0, 2 * np.pi, 721)
        for radius in (8, 16):
            axis.plot(theta, np.full_like(theta, radius), color='#cf381c', linestyle=(0, (1, 1.7)), lw=2, zorder=4)
    return image


def draw_ebb_rectangle(axis, rates, theta_edges, distance_edges):
    signed_edges, values = signed_egocentric_map(theta_edges, rates)
    palette = plt.get_cmap('viridis').copy()
    palette.set_bad('#e6e8eb')
    image = axis.pcolormesh(distance_edges, signed_edges, values, cmap=palette,
                            vmin=np.nanmin(rates), vmax=np.nanmax(rates), shading='flat')
    for distance in (8, 16):
        axis.axvline(distance, color='#cf381c', linestyle=(0, (1, 1.7)), linewidth=2)
    axis.set(xlim=(0, distance_edges[-1]), ylim=(-180, 180), xticks=[0, 8, 16, 28],
             yticks=[-180, 0, 180], xlabel='Distance (cm)', ylabel='Bearing (°)')
    return image


def free_movement_overview_figure(x, fm_rates, ebb_rates, theta_edges, distance_edges, *, unit_id):
    width, height = 15.3, 5.3
    fig = plt.figure(figsize=(width, height), facecolor='white')
    def box(values):
        return np.asarray(values) / [width, height, width, height]
    # Full-height bars keep their single measured elevation row and the 8-in RF aspect.
    bottom = 3. - 8. * 84 / 360 / 2
    axis = fig.add_axes(box([.55, bottom, 8., 8. * 84 / 360]))
    axis.set_gid('free_moving_rectangle_8in')
    image = draw_free_moving_rf(axis, x, fm_rates)
    axis.set(title='', xlabel='')
    axis.tick_params(labelsize=18)
    fig.text(.55 / width, (bottom - .55) / height, 'Azimuth (°)', fontsize=18)
    horizontal_rate_colorbar(fig, image, box([5.25, bottom - .47, 2.6, .10]))
    polar = fig.add_axes(box([9., 1.8, 2.4, 2.4]), projection='polar')
    image = draw_ebb_polar(polar, ebb_rates, theta_edges, distance_edges)
    polar.tick_params(labelsize=11, pad=2)
    axis = fig.add_axes(box([12.45, 1.3, 1.8, 3.4]))
    draw_ebb_rectangle(axis, ebb_rates, theta_edges, distance_edges)
    axis.tick_params(labelsize=12, pad=3)
    axis.xaxis.label.set_fontsize(15)
    axis.yaxis.label.set_fontsize(15)
    axis.yaxis.labelpad = -4
    bar = fig.colorbar(image, cax=fig.add_axes(box([14.55, 1.3, .10, 3.4])),
                       ticks=[image.norm.vmin, image.norm.vmax])
    bar.ax.set_yticklabels([f'{image.norm.vmin:.1f}', f'{image.norm.vmax:.1f}'])
    bar.ax.set_title('Hz', fontsize=14, pad=10)
    bar.ax.tick_params(labelsize=12)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    for x_pos, title in ((4.55, 'Visual RF'), (10.2, 'EBB'), (13.35, 'EBB')):
        fig.text(x_pos / width, 4.96 / height, title, ha='center', fontsize=22)
    return fig


def ebb_sample_figure(rate_map, theta_edges, distance_edges, *, unit_id=307, source_label=None):
    fig = plt.figure(figsize=(12, 5.5), facecolor='white')
    polar = fig.add_axes([.04, .16, .36, .67], projection='polar')
    image = draw_ebb_polar(polar, rate_map, theta_edges, distance_edges)
    polar.set_title(f'EBB · unit {unit_id}', pad=24)
    axis = fig.add_axes([.57, .19, .30, .64])
    draw_ebb_rectangle(axis, rate_map, theta_edges, distance_edges)
    bar = fig.colorbar(image, cax=fig.add_axes([.93, .19, .018, .64]), label='Hz')
    bar.solids.set_rasterized(False)
    if source_label is not None:
        fig.text(.5, .95, source_label, ha='center', fontsize=16)
    return fig


def _hd_ebb_surfaces(surfaces, hd_edges, theta_edges, *, unit_id, source_label, title, colorbar_label, footer):
    fig = plt.figure(figsize=(13.8, 5.), facecolor='white')
    heading = (hd_edges[:-1] + hd_edges[1:]) / 2
    heading = heading % 360
    hd_order = np.argsort(heading)
    heading_edges = cell_edges(heading[hd_order])
    maximum = max(float(np.nanmax(values)) for values in surfaces.values())
    palette = plt.get_cmap('viridis').copy()
    palette.set_bad('white')
    for column, (name, band_title) in enumerate((('0-8', '0–8 cm'), ('8-16', '8–16 cm'), ('16-inf', '>16 cm'))):
        axis = fig.add_axes([.07 + column * .285, .20, .235, .59])
        # signed_egocentric_map expects bearing on its first axis.
        bearing_edges, values = signed_egocentric_map(theta_edges, surfaces[name].T)
        image = axis.pcolormesh(bearing_edges, heading_edges, values.T[hd_order], cmap=palette,
                                vmin=0, vmax=maximum, shading='flat', rasterized=False)
        axis.set(xlim=(-180, 180), ylim=(360, 0), title=band_title,
                 xlabel='Boundary bearing (°)', xticks=[-180, 0, 180], yticks=[0, 90, 180, 270, 360])
        axis.set_yticklabels(['0', '90', '180', '270', '0'] if column == 0 else [])
        axis.tick_params(labelsize=15)
        axis.title.set_fontsize(20)
        axis.xaxis.label.set_fontsize(17)
        if column == 0:
            axis.set_ylabel('Head direction (°)', fontsize=17)
    bar = fig.colorbar(image, cax=fig.add_axes([.92, .20, .014, .59]), label=colorbar_label)
    bar.ax.tick_params(labelsize=14)
    bar.solids.set_rasterized(False)
    fig.text(.07, .92, f'Unit {unit_id} · {title}', fontsize=23)
    fig.text(.07, .035, f'{source_label} · {footer}', fontsize=13)
    return fig


def hd_ebb_joint_figure(joint, *, unit_id, source_label='Same-session S11'):
    """The same cell's conditional Hz at each HD/bearing pair, by distance band."""
    return _hd_ebb_surfaces({key: value['hz'] for key, value in joint['bands'].items()},
        joint['hd_edges'], joint['theta_edges'], unit_id=unit_id, source_label=source_label,
        title='Measured HD × boundary bearing', colorbar_label='Hz',
        footer=f"{joint['smoothing_deg']:g}° smoothing · unvisited bins blank")


def hd_ebb_product_figure(example, *, unit_id, source_label):
    """Display a separable marginal-curve product, explicitly distinct from joint Hz."""
    return _hd_ebb_surfaces(example['products'], example['joint']['hd_edges'], example['theta_edges'],
        unit_id=unit_id, source_label=source_label, title='HD × EBB curve product',
        colorbar_label='Normalized product', footer='Separate marginal curves; not measured joint firing rate')


def annotated_region_arrows_figure(region_pairs, *, example_unit):
    """Mark the two measured angles on one region without changing its geometry."""
    fig = region_positions_figure(region_pairs, highlighted=False)
    fig.set_size_inches(5.5, 5.5)
    axis = fig.axes[0]
    axis.set_position([.03, .06, .94, .88])
    axis.set_title('M22', fontsize=16, pad=8)
    row = region_pairs.loc[region_pairs.unit_id.eq(example_unit)].iloc[0]
    heading, bearing = float(row.hd_peak_deg) % 360, float(row.rf_x_deg)
    point = _direction_xy(heading)
    blue, orange = '#1769a6', '#ca4f19'
    axis.plot([0, 0], [0, 1], color=blue, linestyle=(0, (4, 3)), linewidth=1.3, zorder=4)
    axis.plot([0, point[0]], [0, point[1]], color=blue, linewidth=1.8, zorder=4)
    reference = point + .52 * _direction_xy(heading)
    axis.plot([point[0], reference[0]], [point[1], reference[1]],
              color=blue, linestyle=(0, (3, 2)), linewidth=1.3, zorder=4)
    axis.plot(*point, 'o', color=blue, markersize=5, zorder=5)

    def angle_arc(origin, radius, start, span, color):
        angles = np.deg2rad(np.linspace(start, start + span, 80))
        vertices = origin + radius * np.column_stack((np.sin(angles), np.cos(angles)))
        axis.add_patch(FancyArrowPatch(path=Path(vertices), arrowstyle='-|>',
            mutation_scale=9, linewidth=1.8, color=color, zorder=5))

    angle_arc(np.zeros(2), .43, 0, heading, blue)
    axis.text(*(.61 * _direction_xy(heading / 2)), 'HD', color=blue,
              ha='center', va='center', fontsize=17, weight='bold')
    angle_arc(point, .31, heading, bearing, orange)
    axis.text(*(point + .52 * _direction_xy(heading + bearing / 2)), 'RF', color=orange,
              ha='center', va='center', fontsize=17, weight='bold')
    for arrow in axis.patches:
        if arrow.get_gid() == f'region-arrow-{row.probe}-{row.unit_id}-{row.rf_region}':
            arrow.set_color(orange)
    fig.text(.5, .02, f'{region_pairs.unit_id.nunique()} cells · {len(region_pairs)} RF regions',
             ha='center', fontsize=12, color='#202020')
    return fig


def rank_ticks(axis, n):
    rows = np.unique([0, (n - 1) // 2, n - 1])
    axis.set_yticks(rows + .5, [str(row + 1) for row in rows])
    axis.set_ylabel('Cell rank')


def plot_profile(axis, table, order, title, *, hd=False):
    table = table.loc[order]
    angles = (table.columns.to_numpy(float) + 180) % 360 - 180
    columns = np.argsort(angles)
    image = axis.pcolormesh(cell_edges(angles[columns]), np.arange(len(order) + 1),
                            table.to_numpy()[:, columns], cmap='viridis', vmin=0, vmax=1, shading='flat')
    axis.set(xlim=(-180, 180), ylim=(len(order), 0), title=title,
             xlabel='Direction (°)' if hd else 'Bearing (°)')
    axis.set_xticks([-180, -90, 0, 90, 180],
                   ['180', '270', '0', '90', '180'] if hd else ['−180', '−90', '0', '90', '180'])
    return image


def poster_comparison_figure(profiles, order, kind, *, include_reference=True, source_label=None):
    panels = [(kind, 'Visual RF' if kind == 'RF' else 'HD')] if include_reference else []
    panels += [('0-8', '0–8 cm'), ('8-16', '8–16 cm'), ('16-inf', '>16 cm')]
    fig, axes = plt.subplots(1, len(panels), figsize=(13.4, 6.6), sharey=True,
                             layout='constrained', facecolor='white')
    for axis, (key, title) in zip(axes, panels):
        image = plot_profile(axis, profiles[key], order, title, hd=key == 'HD')
        axis.tick_params(labelsize=14)
        axis.xaxis.label.set_fontsize(19)
        if key == 'RF':
            axis.set_xlabel('Azimuth (°)')
    rank_ticks(axes[0], len(order))
    bar = fig.colorbar(image, ax=axes, fraction=.022, pad=.025, ticks=[0, 1], label='Normalized response')
    bar.ax.set_yticklabels(['0', 'max'])
    bar.solids.set_rasterized(False)
    if source_label is not None:
        fig.suptitle(source_label, fontsize=18)
    return fig


def hd_rf_figure(profiles, order, *, panel_aspect=1.6):
    """Paired heatmaps with an explicit physical height-to-width ratio."""
    panel_width = 4.9
    panel_height = panel_width * panel_aspect
    width, height = 13.2, panel_height + 1.4
    fig = plt.figure(figsize=(width, height), facecolor='white')
    def box(values):
        return np.asarray(values) / [width, height, width, height]
    axes = [fig.add_axes(box([left, .85, panel_width, panel_height])) for left in (.8, 6.45)]
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    for axis, kind in zip(axes, ('HD', 'RF')):
        image = plot_profile(axis, profiles[kind], order, 'HD' if kind == 'HD' else 'Visual RF', hd=kind == 'HD')
        if kind == 'RF':
            axis.set_xlabel('Azimuth (°)')
        axis.tick_params(labelsize=19)
    rank_ticks(axes[0], len(order))
    bar = fig.colorbar(image, cax=fig.add_axes(box([11.85, .85, .16, panel_height])),
                       ticks=[0, 1], label='Normalized response')
    bar.ax.set_yticklabels(['0', 'max'])
    bar.solids.set_rasterized(False)
    return fig


def hd_rf_ebb_overview_figure(profiles_rf, rf_order, profiles_wall, ebb_order):
    width, height = 14.3333333333, 14.5
    fig = plt.figure(figsize=(width, height), facecolor='white')
    def axes(values):
        return fig.add_axes(np.asarray(values) / [width, height, width, height])
    for index, (left, key) in enumerate(((.75, 'HD'), (6.7, 'RF'))):
        axis = axes([left, 8.6, 5.15, 5.2])
        image = plot_profile(axis, profiles_rf[key], rf_order,
                             'HD' if key == 'HD' else 'Visual RF', hd=key == 'HD')
        if key == 'RF':
            axis.set_xlabel('Azimuth (°)')
        axis.tick_params(labelsize=17)
        axis.xaxis.label.set_fontsize(18)
        axis.title.set_fontsize(22)
        if index == 0:
            rank_ticks(axis, len(rf_order))
            axis.yaxis.label.set_fontsize(18)
            axis.yaxis.labelpad = 2
        else:
            axis.set_yticks([])
    fig.text(.75 / width, 7.05 / height, 'Boundary bearing', fontsize=24, fontweight='bold')
    for index, (left, key, title) in enumerate(((.75, '0-8', '0–8 cm'),
                                               (4.55, '8-16', '8–16 cm'),
                                               (8.35, '16-inf', '>16 cm'))):
        axis = axes([left, 1., 3.15, 4.9])
        plot_profile(axis, profiles_wall[key], ebb_order, title)
        axis.tick_params(labelsize=14)
        axis.xaxis.label.set_fontsize(18)
        axis.title.set_fontsize(21)
        if index == 0:
            rank_ticks(axis, len(ebb_order))
            axis.yaxis.label.set_fontsize(18)
            axis.yaxis.labelpad = 2
        else:
            axis.set_yticks([])
    bar = fig.colorbar(image, cax=axes([12.55, 3.4, .18, 7.9]), ticks=[0, 1])
    bar.ax.set_yticklabels(['0', 'max'])
    bar.set_label('Normalized response', fontsize=18, labelpad=16)
    bar.ax.tick_params(labelsize=16)
    bar.solids.set_rasterized(False)
    bar.solids.set_edgecolor('face')
    return fig
