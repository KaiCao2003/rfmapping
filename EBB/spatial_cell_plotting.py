"""Render the restored ten spatial-cell views from saved analysis arrays."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import re
from dataclasses import dataclass
from multiprocessing import get_all_start_methods, get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from HD.profiles import hd_profiles, load_hd_tc
from EBB.spatial_cell_results import collapse_distance_band
from Utils.plotting import LIGHT_PLOT_STYLE, plot_allocentric_heatmap, plot_egocentric_heatmap, plot_egocentric_polar, plot_trajectory_spikes, plot_tuning_curve


@dataclass(frozen=True)
class MapPlot:
    filename: str
    kind: str
    map_key: str = ""
    title: str = ""
    colorbar_label: str = ""
    cmap: str = "viridis"


MAP_PLOTS = (
    MapPlot(
        "egocentric_time_map", "egocentric", "egocentric_occupancy",
        "Egocentric time map", "Seconds", "magma",
    ),
    MapPlot(
        "egocentric_spike_map", "egocentric", "egocentric_spike_map",
        "Egocentric spike map", "Spikes", "magma",
    ),
    MapPlot(
        "egocentric_rate_map", "egocentric", "egocentric_rate_map",
        "Egocentric firing-rate map", "Hz",
    ),
    MapPlot("egocentric_rate_map_polar", "polar"),
    MapPlot(
        "egocentric_tuning_curve", "ego_tuning",
        title="EBC tuning: distance sums",
    ),
    MapPlot(
        "allocentric_occupancy", "allocentric", "allocentric_occupancy",
        "Allocentric occupancy", "Seconds", "magma",
    ),
    MapPlot(
        "allocentric_spike_map", "allocentric", "allocentric_spike_map",
        "Allocentric spike map", "Spikes", "magma",
    ),
    MapPlot(
        "allocentric_rate_map", "allocentric", "allocentric_rate_map",
        "Allocentric firing-rate map", "Hz",
    ),
    MapPlot("trajectory_spike_positions", "trajectory"),
    MapPlot(
        "allocentric_tuning_curve", "hd_tuning", "allocentric_tuning_curve",
        "HD tuning (saved TC)",
    ),
)
PLOT_LAYOUTS = {
    "egocentric": ((7, 5.5), None),
    "allocentric": ((6.5, 6), None),
    "polar": ((7, 6), "polar"),
    "tuning": ((6, 6), "polar"),
    "ego_tuning": ((6, 6), "polar"),
    "hd_tuning": ((6, 6), "polar"),
    "trajectory": ((6.5, 6), None),
}
FIGURE_FILE_TYPES = ("png", "svg")
EBC_PROFILE_BANDS = (
    ("Full map", (None, None), "black"),
    ("≤8 cm", (None, 8), "#0072B2"),
    ("8–16 cm", (8, 16), "#D55E00"),
    (">16 cm", (16, None), "#009E73"),
)


def prepare_spatial_profiles(session, hd_tc):
    """Prepare curves from loaded EBC maps and saved HD counts; never classify."""
    hd = hd_profiles(hd_tc, probe=None, bins=30, smoothing_deg=0, hd_class=None)
    hd_units = hd.reindex(session["unit_ids"])
    hd_edges = np.asarray(hd_tc["angle_bin_edges_deg"])[::len(hd_tc["occupancy_time_s"]) // 30]
    profiles = np.stack([
        collapse_distance_band(session["egocentric_rate_maps"], session["distance_edges"], band)[0][..., 0]
        for _, band, _ in EBC_PROFILE_BANDS
    ], axis=1)
    return {
        **session,
        "egocentric_tuning_profiles": profiles,
        "hd_theta_edges": hd_edges,
        "allocentric_tuning_curves": hd_units.to_numpy(),
        "hd_tc_present": np.isin(session["unit_ids"], hd.index),
    }


def _init_unit_export(metadata, session, counts, output_directory):
    global _UNIT_EXPORT_DATA
    _UNIT_EXPORT_DATA = metadata, session, counts, output_directory


def _export_unit(unit_id):
    metadata, session, counts, output_directory = _UNIT_EXPORT_DATA
    with plt.ioff():
        figure = plot_maps(restored_unit(session, counts, unit_id), int(unit_id), metadata)
        paths = {}
        try:
            for file_type in FIGURE_FILE_TYPES:
                destination = output_directory / f"unit_{unit_id}.{file_type}"
                save_spatial_figure(figure, destination)
                paths[file_type] = str(destination)
        finally:
            plt.close(figure)
    return dict(unit=int(unit_id), path=paths["png"], svg_path=paths["svg"])


def save_spatial_figure(figure, destination, *, dpi=100):
    """Save an opaque figure, keeping SVG heatmap cells as seamless vectors."""
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white", transparent=False)
    if destination.suffix == ".svg":
        # SVG viewers otherwise antialias adjacent cells against the white canvas.
        svg = re.sub(r'(<g id="QuadMesh_\d+")', r'\1 shape-rendering="crispEdges"', destination.read_text())
        destination.write_text(svg)


def export_all_units(metadata, session, counts, output_directory, *, workers=4, units=None):
    """Save PNG and SVG ten-panel figures for every unit in loaded analysis data."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data = metadata, session, counts, output_directory
    unit_ids = session["unit_ids"] if units is None else units
    if workers == 1:
        _init_unit_export(*data)
        return [_export_unit(unit_id) for unit_id in tqdm(
            unit_ids, desc="Spatial-cell figures", unit="unit",
        )]
    # Share the session arrays once per worker instead of sending them per unit.
    start_method = "fork" if "fork" in get_all_start_methods() else "spawn"
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context(start_method),
        initializer=_init_unit_export, initargs=data,
    ) as pool:
        return list(tqdm(
            pool.map(_export_unit, unit_ids), total=len(unit_ids),
            desc="Spatial-cell figures", unit="unit",
        ))


def draw_map(axis, maps, plot):
    if plot.kind in ("egocentric", "allocentric"):
        if plot.kind == "egocentric":
            plot_heatmap = plot_egocentric_heatmap
            horizontal_edges = maps["distance_edges"]
            vertical_edges = maps["theta_edges"]
        else:
            plot_heatmap = plot_allocentric_heatmap
            horizontal_edges = maps["x_edges"]
            vertical_edges = maps["y_edges"]
        plot_heatmap(
            axis, horizontal_edges, vertical_edges, maps[plot.map_key],
            plot.title, plot.colorbar_label, cmap=plot.cmap,
        )
    elif plot.kind == "polar":
        plot_egocentric_polar(
            axis, maps["theta_edges"], maps["distance_edges"],
            maps["egocentric_rate_map"],
        )
    elif plot.kind == "ego_tuning":
        for profile, (label, _, color) in zip(maps["egocentric_tuning_profiles"], EBC_PROFILE_BANDS, strict=True):
            line = plot_tuning_curve(axis, maps["theta_edges"], profile, plot.title, signed=True)
            line.set(label=label, color=color, linewidth=2 if label == "Full map" else 1.2)
        edges = maps["distance_edges"]
        cuts = np.searchsorted((edges[:-1] + edges[1:]) / 2, [8., 16.], side="right")
        support = ", ".join(f"{edge:.2f}" for edge in np.r_[edges[0], edges[cuts], edges[-1]])
        axis.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncol=2, fontsize=9,
                    title=f"Bin edges (cm): {support}", title_fontsize=8)
    elif plot.kind == "hd_tuning":
        if not maps["hd_tc_present"]:
            axis.set_axis_off()
            axis.set_title(plot.title)
            axis.text(.5, .5, "No saved HD TC for this unit", ha="center", va="center", transform=axis.transAxes)
            return
        plot_tuning_curve(
            axis, maps["hd_theta_edges"], maps[plot.map_key], plot.title,
        )
    elif plot.kind == "trajectory":
        plot_trajectory_spikes(
            axis, maps["trajectory_x_cm"], maps["trajectory_y_cm"],
            maps["spike_x_cm"], maps["spike_y_cm"],
            x_limits=maps["x_edges"][[0, -1]],
            y_limits=maps["y_edges"][[0, -1]],
            frame_times=maps.get("frame_times"),
            frame_ids=maps.get("frame_ids"),
        )


def plot_maps(maps, selected_unit_id, metadata):
    with plt.rc_context(LIGHT_PLOT_STYLE):
        figure = plt.figure(figsize=(30, 11), layout="constrained")
        grid = figure.add_gridspec(2, 5)
        for index, plot in enumerate(MAP_PLOTS):
            _, projection = PLOT_LAYOUTS[plot.kind]
            axis = figure.add_subplot(grid[divmod(index, 5)], projection=projection)
            draw_map(axis, maps, plot)
        figure.suptitle(
            f"{metadata['date']}, rec {metadata['recording_number']}, "
            f"Probe{metadata['probe_name']}, {metadata['phase_key']}, "
            f"unit {selected_unit_id}",
            fontsize=16,
        )
    return figure


def prepare_output_directories(output_directory):
    plot_types = ("spatial_maps", *(plot.filename for plot in MAP_PLOTS))
    for plot_type in plot_types:
        for file_type in FIGURE_FILE_TYPES:
            (output_directory / plot_type / file_type).mkdir(
                parents=True, exist_ok=True,
            )


def save_figure(figure, plot_type, unit_id, output_directory):
    for file_type in FIGURE_FILE_TYPES:
        plot_directory = output_directory / plot_type / file_type
        save_spatial_figure(figure, plot_directory / f"{unit_id}.{file_type}", dpi=200)


def save_individual_plots(maps, unit_id, output_directory):
    with plt.rc_context(LIGHT_PLOT_STYLE):
        for plot in MAP_PLOTS:
            figsize, projection = PLOT_LAYOUTS[plot.kind]
            figure, axis = plt.subplots(
                figsize=figsize,
                layout="constrained",
                subplot_kw={"projection": projection},
            )
            draw_map(axis, maps, plot)
            try:
                save_figure(figure, plot.filename, unit_id, output_directory)
            finally:
                plt.close(figure)


def load_results(result_directory):
    result_directory = Path(result_directory)
    manifest_path = result_directory / "metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No completed spatial analysis in {result_directory}: metadata.json is missing"
        )
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (metadata["schema_name"], metadata["schema_version"]) != (
        "rfmapping-spatial-cells", 1,
    ):
        raise ValueError("Unsupported spatial-cell result schema")
    with np.load(result_directory / "session.npz", allow_pickle=False) as archive:
        session = dict(archive)
    return metadata, session


def load_unit(result_directory, session, unit_id):
    unit_path = Path(result_directory) / "units" / f"{unit_id}.npz"
    with np.load(unit_path, allow_pickle=False) as archive:
        maps = {**session, **dict(archive)}
    counts = maps["spike_frame_counts"]
    maps["spike_x_cm"] = np.repeat(maps["trajectory_x_cm"], counts)
    maps["spike_y_cm"] = np.repeat(maps["trajectory_y_cm"], counts)
    return maps


def plot_results(result_directory, *, output=None, units=None, hd_tc_path=None, workers=4):
    """Export the selected saved units without running analysis."""
    result_directory = Path(result_directory).expanduser().resolve()
    metadata, session, counts = load_restored_results(result_directory)
    hd_tc_path = (Path(hd_tc_path) if hd_tc_path is not None else
                  result_directory.parents[2] / "tuning_curves" / f"Probe{metadata['probe_name']}" / "tuning_curves.tc")
    hd_tc = load_hd_tc(hd_tc_path)
    session = prepare_spatial_profiles(session, hd_tc)
    output_directory = (
        Path(output).expanduser().resolve() if output else result_directory.parent / "all_units"
    )
    plots = export_all_units(metadata, session, counts, output_directory, workers=workers, units=units)
    return save_plot_manifest(output_directory, metadata, session, plots, hd_tc_path)


def save_plot_manifest(output_directory, metadata, session, plots, hd_tc_path):
    """Save the figure index beside its images, leaving analysis inputs untouched."""
    plotted_ids = {plot["unit"] for plot in plots}
    missing_ids = session["unit_ids"][~session["hd_tc_present"]]
    metadata = {**metadata, "hd_tc_path": str(hd_tc_path), "hd_plot_bins": 30,
                "hd_missing_unit_ids": [int(unit) for unit in missing_ids if unit in plotted_ids],
                "ebc_tuning_definition": "sum over distance bins selected by their centers; all, <=8 cm, 8-16 cm, >16 cm",
                "ebc_distance_bin_edges_cm": session["distance_edges"].tolist()}
    report = dict(unit_count=len(plots), panels_per_unit=10, plots=plots, metadata=metadata)
    (Path(output_directory) / "spatial_plot_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--units", type=int, nargs="+")
    parser.add_argument("--hd-tc", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        plot_results(args.results, output=args.output, units=args.units, hd_tc_path=args.hd_tc, workers=args.workers)
    except ValueError as error:
        parser.error(str(error))


def load_restored_results(result_directory):
    """Read a verified batch cache without opening raw recordings."""
    from scipy.sparse import load_npz
    result_directory = Path(result_directory)
    audit = json.loads((result_directory / 'audit.json').read_text())
    if audit['status'] not in ('verified', 'complete'):
        raise ValueError('Spatial result audit is incomplete.')
    with np.load(result_directory / 'restored_maps.npz', allow_pickle=False) as archive:
        session = dict(archive)
    counts = load_npz(result_directory / 'frame_spike_counts.npz')
    metadata = dict(audit, recording_number=audit['ebc_session'],
                    probe_name=audit.get('probe_name', 'A'), phase_key=audit.get('phase_key', 'baseline'))
    return metadata, session, counts


def restored_unit(session, counts, unit_id):
    """Select one unit from batch arrays, retaining raw occupancy and spike counts."""
    index = int(np.flatnonzero(session['unit_ids'] == unit_id)[0])
    maps = {key: session[key] for key in (
        'theta_edges', 'distance_edges', 'x_edges', 'y_edges',
        'trajectory_x_cm', 'trajectory_y_cm', 'egocentric_occupancy', 'allocentric_occupancy',
    )}
    maps['frame_times'] = session['frame_times']
    if 'frame_ids' in session:
        maps['frame_ids'] = session['frame_ids']
    for single, plural in (
        ('egocentric_spike_map', 'egocentric_spike_maps'),
        ('egocentric_rate_map', 'egocentric_rate_maps'),
        ('allocentric_spike_map', 'allocentric_spike_maps'),
        ('allocentric_rate_map', 'allocentric_rate_maps'),
        ('allocentric_tuning_curve', 'allocentric_tuning_curves'),
    ):
        maps[single] = session[plural][index]
    maps['egocentric_tuning_profiles'] = session['egocentric_tuning_profiles'][index]
    maps['hd_theta_edges'] = session['hd_theta_edges']
    maps['hd_tc_present'] = session['hd_tc_present'][index]
    frame_counts = counts[index].toarray().ravel()
    maps['spike_x_cm'] = np.repeat(maps['trajectory_x_cm'], frame_counts)
    maps['spike_y_cm'] = np.repeat(maps['trajectory_y_cm'], frame_counts)
    return maps


if __name__ == "__main__":
    main()
