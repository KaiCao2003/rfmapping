"""Render spatial-cell results without loading raw recording data."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Utils.plotting import (
    LIGHT_PLOT_STYLE, plot_allocentric_heatmap, plot_egocentric_heatmap,
    plot_egocentric_polar, plot_trajectory_spikes, plot_tuning_curve,
)


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
        "egocentric_tuning_curve", "tuning", "egocentric_tuning_curve",
        "Egocentric tuning ({preferred_distance_cm:.1f} cm)",
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
        "allocentric_tuning_curve", "tuning", "allocentric_tuning_curve",
        "Allocentric HD tuning",
    ),
)
PLOT_LAYOUTS = {
    "egocentric": ((7, 5.5), None),
    "allocentric": ((6.5, 6), None),
    "polar": ((7, 6), "polar"),
    "tuning": ((6, 6), "polar"),
    "trajectory": ((6.5, 6), None),
}
FIGURE_FILE_TYPES = ("png", "svg")


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
    elif plot.kind == "tuning":
        plot_tuning_curve(
            axis, maps["theta_edges"], maps[plot.map_key],
            plot.title.format(**maps),
        )
    elif plot.kind == "trajectory":
        plot_trajectory_spikes(
            axis, maps["trajectory_x_cm"], maps["trajectory_y_cm"],
            maps["spike_x_cm"], maps["spike_y_cm"],
            x_limits=maps["x_edges"][[0, -1]],
            y_limits=maps["y_edges"][[0, -1]],
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
        figure.savefig(
            plot_directory / f"{unit_id}.{file_type}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--units", type=int, nargs="+")
    args = parser.parse_args(argv)
    result_directory = args.results.expanduser().resolve()
    metadata, session = load_results(result_directory)
    unit_ids = (
        metadata["unit_ids"] if args.units is None else list(dict.fromkeys(args.units))
    )
    unknown = set(unit_ids) - set(metadata["unit_ids"])
    if unknown:
        parser.error(f"Units absent from results: {sorted(unknown)}")
    output_directory = (
        args.output.expanduser().resolve() if args.output else result_directory / "plots"
    )
    prepare_output_directories(output_directory)
    for unit_id in unit_ids:
        maps = load_unit(result_directory, session, unit_id)
        figure = plot_maps(maps, unit_id, metadata)
        try:
            save_figure(figure, "spatial_maps", unit_id, output_directory)
        finally:
            plt.close(figure)
        save_individual_plots(maps, unit_id, output_directory)
        print(f"plotted unit {unit_id}: {output_directory}")


if __name__ == "__main__":
    main()
