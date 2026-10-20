"""Prepare retinotopography data, calculate depth fits, and draw prepared figures."""
from HD.profiles import load_hd_tc

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import patheffects, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PathCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text
from probeinterface import read_openephys
from scipy.stats import linregress

from HD.profiles import hd_profiles
from Utils.direction_comparison import peak_angles
from Utils.plotting import LIGHT_PLOT_STYLE
from RF.rflocate import load_rfmap, load_rf, rf_result_path
from Utils.si_utils import WaveformArtifactStore

# Probe plotting sets rcParams on import; keep its style local to its own figures.
with plt.rc_context():
    from Utils.probe_plotting import WaveformUnitPlotCollection


UNIT_IDENTITY = ["mouse", "date", "probe", "unit_id"]


@dataclass
class HDOverview:
    title: str
    values: np.ndarray


@dataclass
class ProbeLocationSession:
    mouse: str
    date: str
    rf_session: int
    units: pd.DataFrame
    probes: list[tuple[str, pd.DataFrame, WaveformUnitPlotCollection]]
    ylim: tuple[float, float]


@dataclass
class DepthSession:
    mouse: str
    date: str
    rf_session: int
    probes: str
    hd: pd.DataFrame
    rf: pd.DataFrame
    colors: dict[str, np.ndarray]


def prepare_sources(
    root: Path, recordings: list[dict[str, str | int]], *, hd_class: int = 3,
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], pd.DataFrame]]:
    """Read HD curves and recording paths using the established 180-bin HD grid."""
    profiles_by_recording = {}
    source_rows = []
    for recording in recordings:
        mouse, date, probe = (recording[key] for key in ("mouse", "date", "probe"))
        session_dir = root / mouse / date / f"{date}_{recording['rf_session']}"
        data_dir = session_dir / "data"
        position_data_dir = root / mouse / date / f"{date}_{recording['position_session']}" / "data"
        tc_file = (root / mouse / date / f"{date}_{recording['hd_session']}" /
                   "data" / "tuning_curves" / f"Probe{probe}" / "tuning_curves.tc")
        rf_file = (data_dir / "rfmapping/good/-100_400_1ms" / f"Probe{probe}" /
                   f"regular_unitsSpikeCounts_{date}_{recording['rf_session']}.rfmap")
        key = (mouse, date, probe)
        profiles_by_recording[key] = hd_profiles(load_hd_tc(tc_file), probe=probe, bins=180,
                                            smoothing_deg=0, hd_class=hd_class)
        source_rows.append({**recording, "hd_file": str(tc_file), "rf_file": str(rf_file),
                           "waveform_dir": str(position_data_dir / "waveform" / f"Probe{probe}"),
                           "position_dir": str(position_data_dir / "spike_position" / f"Probe{probe}"),
                           "settings_file": str(session_dir / date / "Record Node 102/settings.xml"),
                           "hd_count": len(profiles_by_recording[key])})
    sources = pd.DataFrame(source_rows)
    return sources, profiles_by_recording


def prepare_geometry(
    root: Path, sources: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load unit positions and verify the per-shank geometry against source recordings."""
    geometry_rows = []
    position_rows = []
    for source in sources.itertuples():
        store = WaveformArtifactStore(source.waveform_dir)
        if source.position_session != source.rf_session:
            rf_store = WaveformArtifactStore(root / source.mouse / source.date /
                                             f"{source.date}_{source.rf_session}" /
                                             "data" / "waveform" / f"Probe{source.probe}")
            assert store.manifest["canonical_source"] == rf_store.manifest["canonical_source"]
            for unit_id in set(store.unit_summaries) & set(rf_store.unit_summaries):
                unit, rf_unit = store.unit_summaries[unit_id], rf_store.unit_summaries[unit_id]
                np.testing.assert_allclose([unit.unit_x_um, unit.unit_y_um],
                                           [rf_unit.unit_x_um, rf_unit.unit_y_um], rtol=0, atol=0)
            np.testing.assert_array_equal(store.channel_locations, rf_store.channel_locations)
            np.testing.assert_array_equal(store.channel_shank_ids, rf_store.channel_shank_ids)
        setup = read_openephys(source.settings_file, stream_name=f"Probe{source.probe}")
        channel_order = np.argsort(setup.device_channel_indices)
        coordinates = setup.contact_positions[channel_order]
        shanks = setup.shank_ids[channel_order].astype(int)
        kilosort_dir = Path(store.manifest["session"]["kilosort_dir"])
        kilosort_coordinates = np.load(kilosort_dir / "channel_positions.npy")
        kilosort_shanks = np.load(kilosort_dir / "ops.npy", allow_pickle=True).item()["probe"]["kcoords"]
        np.testing.assert_allclose(coordinates, store.channel_locations, rtol=0, atol=1e-6)
        np.testing.assert_allclose(coordinates, kilosort_coordinates, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(shanks, store.channel_shank_ids)
        np.testing.assert_array_equal(shanks, np.asarray(kilosort_shanks).astype(int))
        tips = np.asarray(setup.annotations["shank_tips"])
        position_manifest = json.loads((Path(source.position_dir) / "manifest.json").read_text())
        position_method = (position_manifest["spike_position"]["method"]
                           if "spike_position" in position_manifest
                           else position_manifest["canonical_source"]["method"])
        for shank in np.unique(shanks):
            channel_y = coordinates[shanks == shank, 1]
            tip_y = tips[shank, 1]
            assert tip_y < channel_y.min()
            geometry_rows.append(dict(mouse=source.mouse, date=source.date, probe=source.probe,
                                      shank=int(shank), part_number=setup.annotations["part_number"],
                                      min_channel_y_um=channel_y.min(), max_channel_y_um=channel_y.max(),
                                      tip_y_um=tip_y, deeper_is="smaller_y",
                                      max_coordinate_error_um=float(np.max(abs(coordinates - store.channel_locations))),
                                      position_method=position_method))
        for unit_id, unit in store.unit_summaries.items():
            position_rows.append(dict(mouse=source.mouse, date=source.date, probe=source.probe,
                                      unit_id=unit_id, shank=int(store.channel_shank_ids[unit.best_channel_index]),
                                      x_um=unit.unit_x_um, y_um=unit.unit_y_um,
                                      best_channel_id=unit.best_channel_id,
                                      best_channel_y_um=unit.best_channel_y_um))
    geometry = pd.DataFrame(geometry_rows)
    positions = pd.DataFrame(position_rows)
    return geometry, positions


def prepare_unit_depths(
    sources: pd.DataFrame, hd_profiles: dict[tuple[str, str, str], pd.DataFrame],
    positions: pd.DataFrame, *, hd_class: int = 3,
    response_window_s: tuple[float, float] = (0.0, 0.2), require_rf_detection: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Join native HD/RF peak angles and measure micrometres below each shank's topmost HD unit."""
    unit_rows = []
    identity = UNIT_IDENTITY
    for source in sources.itertuples():
        table = hd_profiles[(source.mouse, source.date, source.probe)]
        preferred = pd.Series(peak_angles(table, table.index) % 360, index=table.index)
        maps = load_rfmap(source.rf_file).to_firing_rate(reconstruct_presentations=True).mean_rate(
            *response_window_s, show_progress=False)
        maps_by_id = {rf_map.unit_id: rf_map for rf_map in maps}
        detected = load_rf(rf_result_path(source.rf_file))
        detected_ids = set(detected.unit_ids[np.any(detected.mask_2d, axis=(1, 2))].tolist())
        for (_, unit_id), hd_angle in preferred.items():
            row = dict(mouse=source.mouse, date=source.date, probe=source.probe, unit_id=int(unit_id),
                       hd_class=hd_class, hd_preferred_deg=hd_angle, rf_max_angle_deg=np.nan,
                       rf_max_y_deg=np.nan, rf_max_response=np.nan, rf_peak_ties=0,
                       rf_detected_2d=unit_id in detected_ids, rf_peak_status="unit_not_in_rf")
            if unit_id in maps_by_id:
                rf_map = maps_by_id[unit_id]
                response = rf_map.to_2d_array()
                finite_values = response[np.isfinite(response)]
                row["rf_peak_status"] = "flat_or_unmeasured"
                if finite_values.size and np.ptp(finite_values) > 0:
                    y, x = np.unravel_index(np.nanargmax(response), response.shape)
                    row.update(rf_max_angle_deg=float(rf_map.x_positions[x]),
                               rf_max_y_deg=float(rf_map.y_positions[y]),
                               rf_max_response=float(response[y, x]),
                               rf_peak_ties=int(np.sum(response == response[y, x])),
                               rf_peak_status="measured_peak")
            unit_rows.append(row)
    units = pd.DataFrame(unit_rows).merge(positions, on=identity, how="left", validate="one_to_one")
    units["depth_eligible"] = np.isfinite(units.y_um) & np.isfinite(units.hd_preferred_deg)
    units["shank_label"] = [
        f"{m} {d}/{p} S{int(s)}" if pd.notna(s) else None
        for m, d, p, s in units[["mouse", "date", "probe", "shank"]].itertuples(index=False, name=None)
    ]
    hd_depth = units.loc[units.depth_eligible]
    origins = hd_depth.groupby("shank_label").y_um.max()
    units["topmost_hd_y_um"] = units.shank_label.map(origins)
    units["normalized_depth_um"] = units.topmost_hd_y_um - units.y_um
    units["rf_depth_eligible"] = units.depth_eligible & units.rf_max_angle_deg.notna()
    if require_rf_detection:
        units["rf_depth_eligible"] &= units.rf_detected_2d
    hd_depth = units.loc[units.depth_eligible].copy()
    rf_depth = units.loc[units.rf_depth_eligible].copy()
    np.testing.assert_allclose(hd_depth.groupby("shank_label").normalized_depth_um.min(), 0)
    assert (hd_depth.normalized_depth_um >= 0).all()
    assert not units.duplicated(identity).any()
    return units, hd_depth, rf_depth


def summarize_depths(
    units: pd.DataFrame, hd_depth: pd.DataFrame, rf_depth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count eligible units and preserve the physical depth span of every shank."""
    summary = units.groupby(["mouse", "date", "probe"], sort=False).agg(
        hd_cells=("unit_id", "size"), hd_with_depth=("depth_eligible", "sum"),
        hd_with_rf_peak_and_depth=("rf_depth_eligible", "sum"),
        saved_significant_rf=("rf_detected_2d", "sum"))
    shank_summary = hd_depth.groupby("shank_label", sort=True).agg(
        n_hd=("unit_id", "size"), topmost_hd_y_um=("y_um", "max"),
        normalized_span_um=("normalized_depth_um", "max"))
    shank_summary["n_rf"] = rf_depth.groupby("shank_label").size().reindex(
        shank_summary.index, fill_value=0)
    return summary, shank_summary


def prepare_hd_overview(
    sources: pd.DataFrame, hd_profiles: dict[tuple[str, str, str], pd.DataFrame],
) -> list[HDOverview]:
    """Sort each session's native HD curves and scale their display rows to their maximum."""
    panels = []
    for (mouse, date, rf_session), session_sources in sources.groupby(
            ["mouse", "date", "rf_session"], sort=False):
        curves = []
        for source in session_sources.itertuples():
            table = hd_profiles[(source.mouse, source.date, source.probe)].copy()
            table.attrs = {}
            curves.append(table)
        table = pd.concat(curves)
        angles = np.asarray(table.columns) % 360
        preferred = peak_angles(table, table.index) % 360
        values = table.to_numpy()[np.argsort(preferred, kind="stable")]
        values = values / np.nanmax(values, axis=1, keepdims=True)
        probes = "+".join(session_sources.probe)
        panels.append(HDOverview(
            f"{mouse} {date}_{rf_session} / {probes} · {len(table)} HD cells",
            values[:, np.argsort(angles)]))
    return panels


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_hd_overview(panels: list[HDOverview]) -> Figure:
    """Draw prepared HD overview matrices."""
    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), squeeze=False,
        layout="constrained", facecolor="white")
    for ax in axes.flat[len(panels):]:
        ax.remove()
    axes = axes.ravel()[:len(panels)]
    for ax, panel in zip(axes, panels, strict=True):
        image = ax.imshow(panel.values, origin="lower", aspect="auto",
                          interpolation="nearest", extent=(0, 360, 0, len(panel.values)),
                          cmap="viridis", vmin=0, vmax=1)
        ax.set(title=panel.title, xlabel="HD direction (°; clockwise)",
               ylabel="Units sorted by preferred direction", xticks=[0, 90, 180, 270, 360])
    figure.colorbar(image, ax=axes, label="Firing rate / each unit's maximum", shrink=0.85)
    return figure


def place_unit_labels(ax: Axes, renderer: RendererBase) -> None:
    # Search the rendered axes for nearby labels without moving the probe points.
    points = ax.transData.transform([text.xy for text in ax.texts])
    bounds = ax.get_window_extent()
    pixel_scale = renderer.points_to_pixels(1)
    boxes = [Text.get_window_extent(text, renderer) for text in ax.texts]
    step_x = max(box.width for box in boxes) / pixel_scale + 2
    step_y = max(box.height for box in boxes) / pixel_scale + 2
    offsets = sorted(
        ((sx * dx, sy * dy)
         for dx in np.arange(4, bounds.width / pixel_scale, step_x)
         for dy in np.arange(4, bounds.height / pixel_scale, step_y)
         for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))),
        key=lambda offset: offset[0] ** 2 + offset[1] ** 2)
    placed = []
    for text in ax.texts:
        for dx, dy in offsets:
            text.set_position((dx, dy))
            text.set_horizontalalignment("left" if dx > 0 else "right")
            text.set_verticalalignment("bottom" if dy > 0 else "top")
            text.update_positions(renderer)
            box = Text.get_window_extent(text, renderer).padded(pixel_scale)
            inside = bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
            covers_point = ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1) &
                            (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)).any()
            if inside and not covers_point and not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
        else:
            raise ValueError(f"No non-overlapping position for unit label {text.get_text()}")


def prepare_probe_locations(
    units: pd.DataFrame, sources: pd.DataFrame,
) -> list[ProbeLocationSession]:
    """Load probe backgrounds, select session units, and verify their physical positions."""
    sessions = []
    for (mouse, date, rf_session), session_sources in sources.groupby(
            ["mouse", "date", "rf_session"], sort=False):
        session_units = units.loc[units.mouse.eq(mouse) & units.date.eq(date)]
        assert len(session_units) == session_sources.hd_count.sum()
        assert np.isfinite(session_units[["x_um", "y_um", "hd_preferred_deg"]]).all().all()
        probe_groups = []
        lower_y, upper_y = [], []
        for source in session_sources.itertuples():
            group = session_units.loc[session_units.probe.eq(source.probe)].copy()
            plot_collection = WaveformUnitPlotCollection(source.waveform_dir, [])
            plot_collection.probe = read_openephys(
                source.settings_file, stream_name=f"Probe{source.probe}")
            tips = np.asarray(plot_collection.probe.annotations["shank_tips"])
            contact_y = plot_collection.store.channel_locations[:, 1]
            lower_y.extend([tips[:, 1].min(), contact_y.min(), group.y_um.min()])
            upper_y.extend([contact_y.max(), group.y_um.max()])
            for row in group.itertuples():
                summary = plot_collection.store.unit_summaries[row.unit_id]
                np.testing.assert_allclose([summary.unit_x_um, summary.unit_y_um],
                                           [row.x_um, row.y_um], rtol=0, atol=1e-6)
            probe_groups.append((source.probe, group, plot_collection))
        ylim = (min(lower_y) - 50, max(upper_y) + 100)
        sessions.append(ProbeLocationSession(
            mouse, date, int(rf_session), session_units, probe_groups, ylim))

    return sessions


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_probe_locations(session: ProbeLocationSession, *, cohort_label: str = "all") -> Figure:
    """Render already-loaded probe positions; the caller saves and verifies the figure."""
    mouse, date, rf_session = session.mouse, session.date, session.rf_session
    session_units, probe_groups, ylim = session.units, session.probes, session.ylim
    position_color_bins = 12
    hd_position_cmap = plt.colormaps["rainbow"].resampled(position_color_bins)
    hd_position_norm = BoundaryNorm(np.linspace(0, 360, position_color_bins + 1),
                                    position_color_bins, clip=True)
    # Keep the same colors for equal angles on the signed RF and unsigned HD bars.
    rf_position_cmap = ListedColormap(
        np.roll(hd_position_cmap(np.arange(position_color_bins)), -position_color_bins // 2, axis=0),
        name="rainbow_12_signed_rf")
    rf_position_norm = BoundaryNorm(np.linspace(-180, 180, position_color_bins + 1),
                                    position_color_bins, clip=True)
    missing_rf_color = "#9ca3af"
    figure_width = 12 if len(probe_groups) == 1 else 20
    probe_width = max(np.ptp(collection.store.channel_locations[:, 0]) + 140
                      for _, _, collection in probe_groups)
    panel_width = (figure_width / 2 - 1.2) / len(probe_groups)
    figure_height = (13 if len(probe_groups) > 1 else
                     min(13, 2.4 + panel_width * (ylim[1] - ylim[0]) / probe_width))
    figure = plt.figure(figsize=(figure_width, figure_height), dpi=220,
                         layout="constrained", facecolor="white")
    figure.suptitle(f"{mouse} · {date}_{rf_session} · {cohort_label} {len(session_units)} HD cells",
                   fontsize=16)
    subfigures = figure.subfigures(1, 2, wspace=0.07)
    position_axes = []
    panel_specs = [
        ("HD preferred direction", "hd_preferred_deg", hd_position_cmap,
         hd_position_norm, np.arange(0, 361, 60), "HD preferred direction (°; 30° bins)"),
        ("RF maximum-bin horizontal angle", "rf_max_angle_deg", rf_position_cmap,
         rf_position_norm, np.arange(-180, 181, 60), "RF maximum-bin angle (°; 30° bins)"),
    ]
    for subfigure, (title, angle_column, cmap, norm, ticks, colorbar_label) in zip(
            subfigures, panel_specs, strict=True):
        subfigure.set_facecolor("white")
        subfigure.suptitle(title, fontsize=13)
        axes = subfigure.subplots(1, len(probe_groups), squeeze=False, sharey=True)[0]
        for column, (ax, (probe, group, plot_collection)) in enumerate(
                zip(axes, probe_groups, strict=True)):
            angles = group[angle_column].to_numpy()
            angles = angles % 360 if angle_column == "hd_preferred_deg" else (angles + 180) % 360 - 180
            unit_values = dict(zip(group.unit_id.astype(int), angles, strict=True))
            plot_collection.plot_unit_locations(
                ax=ax, unit_values=unit_values, cmap=cmap, norm=norm,
                missing_color=missing_rf_color)
            for unit in group.itertuples():
                text = ax.annotate(str(int(unit.unit_id)), (unit.x_um, unit.y_um),
                                   xytext=(4, 4), textcoords="offset points", fontsize=7,
                                   color="#222222", ha="left", va="bottom",
                                   arrowprops=dict(arrowstyle="-", color="#6b7280", lw=0.4,
                                                   shrinkA=2, shrinkB=4),
                                   path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
                text.set_in_layout(False)
            position_axes.append(ax)
            contact_x = plot_collection.store.channel_locations[:, 0]
            ax.set(xlim=(min(contact_x.min(), group.x_um.min()) - 70,
                         max(contact_x.max(), group.x_um.max()) + 70),
                   ylim=ylim, title=f"Probe {probe} · {len(group)} HD cells")
            ax.set_aspect("equal", adjustable="box")
            ax.tick_params(colors="#222222")
            ax.set_ylabel("Probe y (µm; larger = shallower)" if column == 0 else "")
            if angle_column == "rf_max_angle_deg":
                missing = int(group[angle_column].isna().sum())
                if missing:
                    ax.legend(handles=[Line2D([], [], linestyle="none", marker="o",
                                             markerfacecolor=missing_rf_color, markeredgecolor="white",
                                             markersize=7, label=f"No RF peak (n = {missing})")],
                              loc="upper right", fontsize=8, framealpha=1,
                              facecolor="white", edgecolor="#d1d5db")
        colorbar = subfigure.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                                     ax=list(axes), orientation="horizontal",
                                     ticks=ticks, shrink=0.85, pad=0.06, fraction=0.04, drawedges=True)
        colorbar.set_label(colorbar_label)
        colorbar.dividers.set_color("white")
        colorbar.dividers.set_linewidth(0.8)
    figure.supxlabel("Recorded probe region · native x/y in µm · identical HD cells in both panels",
                     fontsize=10)
    figure.canvas.draw()
    figure.set_layout_engine("none")  # Keep display-coordinate label placement stable on save.
    for ax in position_axes:
        place_unit_labels(ax, figure.canvas.get_renderer())
    return figure


def verify_probe_locations(
    figure: Figure, session: ProbeLocationSession,
) -> tuple[dict[str, Any], list[float]]:
    """Check plotted positions, one unit label per point, and non-overlapping labels."""
    position_axes = [ax for ax in figure.axes if ax.texts]
    assert len(position_axes) == 2 * len(session.probes)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    position_offset_errors_um = []
    label_count, label_overlaps = 0, 0
    for ax, (_, group, _) in zip(position_axes, session.probes * 2, strict=True):
        offsets = np.concatenate([
            np.asarray(artist.get_offsets()) for artist in ax.collections
            if isinstance(artist, PathCollection)])
        expected = group[["x_um", "y_um"]].to_numpy()
        offsets = offsets[np.lexsort((offsets[:, 1], offsets[:, 0]))]
        expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
        np.testing.assert_allclose(offsets, expected, rtol=0, atol=1e-6)
        position_offset_errors_um.append(float(np.max(abs(offsets - expected))))
        assert sorted(int(text.get_text()) for text in ax.texts) == sorted(group.unit_id.astype(int))
        boxes = [Text.get_window_extent(text, renderer) for text in ax.texts]
        label_count += len(boxes)
        label_overlaps += sum(box.overlaps(other)
                              for i, box in enumerate(boxes) for other in boxes[i + 1:])
    assert label_count == 2 * len(session.units)
    assert label_overlaps == 0
    counts = dict(
        unit_id_labels=label_count, unit_id_label_overlaps=int(label_overlaps),
        mouse=session.mouse, date=session.date, rf_session=session.rf_session, panels=2,
        probe_axes_per_panel=len(session.probes), hd_cells_per_panel=len(session.units),
        rf_colored_cells=int(session.units.rf_max_angle_deg.notna().sum()),
        rf_gray_cells=int(session.units.rf_max_angle_deg.isna().sum()),
        plotted_units_by_probe={probe: group.unit_id.astype(int).tolist()
                               for probe, group, _ in session.probes},
        position_units="um", y_direction="larger_y_is_shallower",
        colormap="rainbow", color_bins=12, color_bin_width_deg=30)
    return counts, position_offset_errors_um


def fit_relationship(
    data: pd.DataFrame, angle_column: str, relationship: str,
) -> pd.DataFrame:
    """Calculate native-angle OLS fits per shank and for the supplied session."""
    rows = []
    groups = [("pooled", data), *data.groupby("shank_label", sort=True)]
    for label, group in groups:
        row = dict(relationship=relationship, shank=label, n=len(group),
                   slope_deg_per_um=np.nan, intercept_deg=np.nan, r_squared=np.nan,
                   p_value=np.nan, slope_stderr=np.nan, status="insufficient_distinct_depths_or_units")
        if len(group) >= 3 and group.normalized_depth_um.nunique() >= 2:
            result = linregress(group.normalized_depth_um, group[angle_column])
            row.update(slope_deg_per_um=result.slope, intercept_deg=result.intercept,
                       r_squared=result.rvalue**2, p_value=result.pvalue,
                       slope_stderr=result.stderr, status="fit")
        rows.append(row)
    return pd.DataFrame(rows)


def plot_depth(
    ax: Axes, data: pd.DataFrame, angle_column: str, fits: pd.DataFrame,
    colors: dict[str, np.ndarray], title: str, angle_label: str, ylim: tuple[float, float],
) -> Axes:
    """Draw unit points and already-calculated depth fits on one axis."""
    by_shank = fits.set_index("shank")
    for label, group in data.groupby("shank_label", sort=True):
        color = colors[label]
        probe_shank = f"{group.probe.iloc[0]} S{int(group.shank.iloc[0])}"
        ax.scatter(group.normalized_depth_um, group[angle_column], s=32, color=color,
                   edgecolor="white", linewidth=0.35, alpha=0.9,
                   label=f"{probe_shank} (n={len(group)})")
        fit = by_shank.loc[label]
        if fit.status == "fit":
            x = np.array([group.normalized_depth_um.min(), group.normalized_depth_um.max()])
            ax.plot(x, fit.intercept_deg + fit.slope_deg_per_um * x, color=color, linewidth=1.1)
    fit = by_shank.loc["pooled"]
    if fit.status == "fit":
        x = np.array([data.normalized_depth_um.min(), data.normalized_depth_um.max()])
        ax.plot(x, fit.intercept_deg + fit.slope_deg_per_um * x, "k--", linewidth=1.8,
                label="Session OLS")
        title += f"\nSession OLS: slope={fit.slope_deg_per_um:.3f}°/µm · R²={fit.r_squared:.3f} · p={fit.p_value:.3g}"
    ax.set(title=title, xlabel="Normalized depth (µm below topmost HD unit)",
           ylabel=angle_label, ylim=ylim)
    ax.grid(alpha=0.45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9,
              frameon=True, borderaxespad=0)

    return ax


def prepare_depth_sessions(
    sources: pd.DataFrame, hd_depth: pd.DataFrame, rf_depth: pd.DataFrame, *,
    color_units: pd.DataFrame | None = None,
) -> list[DepthSession]:
    """Select session data and stable shank colors before statistics or plotting."""
    sessions = []
    for (mouse, date, rf_session), session_sources in sources.groupby(
            ["mouse", "date", "rf_session"], sort=False):
        hd = hd_depth.loc[hd_depth.mouse.eq(mouse) & hd_depth.date.eq(date)]
        rf = rf_depth.loc[rf_depth.mouse.eq(mouse) & rf_depth.date.eq(date)]
        color_data = hd if color_units is None else color_units.loc[
            color_units.mouse.eq(mouse) & color_units.date.eq(date)]
        labels = sorted(color_data.shank_label.dropna().unique())
        colors = dict(zip(labels, plt.colormaps["tab10"](np.arange(len(labels))), strict=True))
        sessions.append(DepthSession(
            mouse, date, int(rf_session), "+".join(session_sources.probe), hd, rf, colors))
    return sessions


def calculate_depth_statistics(sessions: list[DepthSession]) -> pd.DataFrame:
    """Calculate all per-session and per-shank descriptive fits."""
    tables = []
    for session in sessions:
        hd_fits = fit_relationship(session.hd, "hd_preferred_deg", "HD")
        rf_fits = fit_relationship(session.rf, "rf_max_angle_deg", "RF")
        tables.append(pd.concat([hd_fits, rf_fits], ignore_index=True).assign(
            mouse=session.mouse, date=session.date, rf_session=session.rf_session))
    return pd.concat(tables, ignore_index=True)


@plt.rc_context(LIGHT_PLOT_STYLE)
def plot_depth_session(
    session: DepthSession, fits: pd.DataFrame, *, selected: bool = False,
) -> Figure:
    """Draw one session's prepared depth points and fits without fitting or saving."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True,
                                layout="constrained", facecolor="white")
    prefix = "ADN · " if selected else ""
    unit_label = "selected HD cells" if selected else "HD cells"
    figure.suptitle(
        f"{prefix}{session.mouse} · {session.date}_{session.rf_session} · Probe {session.probes}",
        fontsize=16)
    plot_depth(axes[0], session.hd, "hd_preferred_deg", fits.loc[fits.relationship.eq("HD")],
               session.colors, f"HD preferred direction · {len(session.hd)} {unit_label}",
               "HD preferred direction (°; clockwise)", (-5, 365))
    plot_depth(axes[1], session.rf, "rf_max_angle_deg", fits.loc[fits.relationship.eq("RF")],
               session.colors, f"RF maximum-bin angle · {len(session.rf)} {unit_label}",
               "RF maximum spatial bin: horizontal angle (°)", (-185, 185))
    axes[0].set_yticks([0, 90, 180, 270, 360])
    axes[1].set_yticks([-180, -90, 0, 90, 180])
    if selected:
        for ax in axes:
            ax.set_xlabel("Depth (µm below topmost selected HD unit on each shank)")
    return figure


def verify_depth_plot(figure: Figure, session: DepthSession) -> dict[str, Any]:
    """Check that the two depth panels contain all prepared points."""
    point_counts = [sum(len(artist.get_offsets()) for artist in ax.collections
                        if isinstance(artist, PathCollection)) for ax in figure.axes]
    assert point_counts == [len(session.hd), len(session.rf)]
    return dict(mouse=session.mouse, date=session.date, rf_session=session.rf_session,
                hd_cells=point_counts[0], rf_cells=point_counts[1],
                hd_shanks=int(session.hd.shank_label.nunique()),
                rf_shanks=int(session.rf.shank_label.nunique()), panels=len(figure.axes))


def load_adn_archive(
    config_path: Path, source_recordings_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the annotated historical ADN cohort and verify its archived source hashes."""
    adn_config = json.loads(config_path.read_text())
    for source_key in ("source_unit_table", "source_pdf", "replacement_image"):
        digest = hashlib.sha256(Path(adn_config[source_key]).read_bytes()).hexdigest()
        assert digest == adn_config[f"{source_key}_sha256"]
    adn_source_units = pd.read_csv(adn_config["source_unit_table"], dtype={"date": str})
    adn_sources = pd.read_csv(source_recordings_path, dtype={"date": str})
    adn_identity = UNIT_IDENTITY
    adn_selection = pd.DataFrame([
        dict(mouse=record["mouse"], date=record["date"], probe=record["probe"], unit_id=unit_id,
             selection_source=record.get("replacement_image", "scan113026.pdf"),
             boundary_included=unit_id in record["boundary_unit_ids"])
        for record in adn_config["recordings"] for unit_id in record["unit_ids"]
    ])
    assert not adn_selection.duplicated(adn_identity).any()
    assert not adn_source_units.duplicated(adn_identity).any()
    # Later anatomical annotations do not change the cohort in this historical archive.
    adn_selection = adn_selection.merge(
        adn_sources[["mouse", "date", "probe"]].drop_duplicates(),
        on=["mouse", "date", "probe"], how="inner", sort=False, validate="many_to_one",
    )
    assert set(map(tuple, adn_selection[adn_identity].to_numpy())) <= set(
        map(tuple, adn_source_units[adn_identity].to_numpy()))
    adn_audit = adn_source_units.merge(adn_selection, on=adn_identity, how="left",
                                        sort=False, validate="one_to_one")
    adn_audit["adn_selected"] = adn_audit.selection_source.notna()
    adn_units = adn_audit.loc[adn_audit.adn_selected].copy()
    assert len(adn_units) == len(adn_selection)
    return adn_config, adn_source_units, adn_sources, adn_audit, adn_units


def prepare_selected_depths(
    units: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Rebase selected units to their topmost selected HD unit per shank, retaining old depths."""
    units = units.copy()
    # Retain the original origin so both depth conventions remain auditable.
    units["all_hd_topmost_y_um"] = units.topmost_hd_y_um
    units["all_hd_normalized_depth_um"] = units.normalized_depth_um
    origins = units.loc[units.depth_eligible].groupby("shank_label").y_um.max()
    units["topmost_hd_y_um"] = units.shank_label.map(origins)
    units["normalized_depth_um"] = units.topmost_hd_y_um - units.y_um
    hd_depth = units.loc[units.depth_eligible].copy()
    rf_depth = units.loc[units.rf_depth_eligible].copy()
    return units, hd_depth, rf_depth, origins
