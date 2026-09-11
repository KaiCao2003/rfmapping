from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from Utils.json_tools import read_formatted_json
from Utils.plotting import (
    LIGHT_PLOT_STYLE,
    plot_allocentric_heatmap,
    plot_egocentric_heatmap,
    plot_egocentric_polar,
    plot_trajectory_spikes,
    plot_tuning_curve,
)
from Utils.tuning_curve_utils import get_exposure_timestamps

recording_root = Path("/mnt/senzailab/Kai/#Recording/m19")
date = "260831"
recording_number = 2
probe_name = "A"
phase_key = "baseline"

camera_input_channel = 1
camera_ttl_threshold = 14000
camera_ttl_active_high = True

# rig pixel lim
x_min, x_max = 370, 920
y_min, y_max = 210, 760
rig_size_cm = 41
cm_per_px = rig_size_cm / (x_max - x_min)

theta_bin_deg = 6
number_of_distance_bins = 20
number_of_spatial_bins = 40
egocentric_smoothing_sigma = 5
allocentric_smoothing_sigma = 1.5

save_root_directory = (
    recording_root
    / date
    / f"{date}_{recording_number}"
    / "data"
    / "spatial_cells"
)

worker_data = None


def d(theta_deg, center_x, center_y, head_direction_deg):
    """Return ray distances to the arena boundary in pixels."""
    absolute_angle_rad = np.deg2rad(
        head_direction_deg[:, None] + theta_deg[None, :]
    )
    direction_x = -np.sin(absolute_angle_rad)
    direction_y = -np.cos(absolute_angle_rad)

    # Cardinal angles have tiny nonzero components from trigonometric roundoff.
    distance_x = np.divide(
        np.where(
            direction_x > 0,
            x_max - center_x[:, None],
            x_min - center_x[:, None],
        ),
        direction_x,
        out=np.full_like(direction_x, np.inf),
        where=np.abs(direction_x) > 1e-12,
    )
    distance_y = np.divide(
        np.where(
            direction_y > 0,
            y_max - center_y[:, None],
            y_min - center_y[:, None],
        ),
        direction_y,
        out=np.full_like(direction_y, np.inf),
        where=np.abs(direction_y) > 1e-12,
    )

    return np.minimum(distance_x, distance_y)


def compute_2d_map(
        coordinate_1,
        coordinate_2,
        coordinate_1_edges,
        coordinate_2_edges,
        frame_weights,
):
    coordinate_1, coordinate_2 = np.broadcast_arrays(
        coordinate_1,
        coordinate_2,
    )
    weight_shape = (len(frame_weights),) + (1,) * (
            coordinate_1.ndim - 1
    )
    weights = np.broadcast_to(
        np.asarray(frame_weights).reshape(weight_shape),
        coordinate_1.shape,
    )

    return np.histogram2d(
        coordinate_1.ravel(),
        coordinate_2.ravel(),
        bins=[coordinate_1_edges, coordinate_2_edges],
        weights=weights.ravel(),
    )[0]


def compute_rate_map(
        spike_map,
        smoothed_occupancy,
        smoothing_sigma,
        smoothing_mode,
):
    smoothed_spikes = gaussian_filter(
        spike_map,
        sigma=smoothing_sigma,
        mode=smoothing_mode,
        output=float,
    )
    rate_map = np.full_like(smoothed_spikes, np.nan, dtype=float)
    np.divide(
        smoothed_spikes,
        smoothed_occupancy,
        out=rate_map,
        where=smoothed_occupancy > 0,
    )
    return rate_map


def count_spikes_by_frame(spike_times, frame_times):
    """Count spikes within the sorted frame-time interval loaded by load_data."""
    right = np.searchsorted(frame_times, spike_times)
    left = np.maximum(right - 1, 0)
    use_right = (
            np.abs(frame_times[right] - spike_times)
            < np.abs(frame_times[left] - spike_times)
    )
    nearest_frame = np.where(use_right, right, left)
    return np.bincount(
        nearest_frame,
        minlength=len(frame_times),
    )


def read_good_unit_ids(kilosort_dir: Path) -> list[int]:
    cluster_labels = pd.read_csv(
        kilosort_dir / "cluster_KSLabel.tsv",
        sep="\t",
    )
    return (
        cluster_labels.loc[
            cluster_labels["KSLabel"] == "good",
            "cluster_id",
        ]
        .astype(int)
        .tolist()
    )


def load_data():
    session_dir = (
            recording_root / date / f"{date}_{recording_number}"
    )
    data_dir = session_dir / "data"
    kilosort_dir = next(
        (session_dir / "kilosort" / f"Probe{probe_name}").glob(
            "kilosort_*"
        )
    )

    pose = pd.read_csv(session_dir / f"{date}.csv")
    pose = pose.dropna(
        subset=["frame", "center_x", "center_y", "hd_deg"]
    )

    session_info = read_formatted_json(
        data_dir / "session_info.json"
    )["session_info"]
    exposure_timestamps, adc_time_origin_s, _ = (
        get_exposure_timestamps(
            session_info=session_info,
            camera_input_channel=camera_input_channel,
            camera_ttl_threshold=camera_ttl_threshold,
            camera_ttl_active_high=camera_ttl_active_high,
        )
    )
    pose_times = exposure_timestamps[
        pose["frame"].to_numpy(dtype=int)
    ]

    interval_table = pd.read_csv(data_dir / "interval_table.csv")
    interval = interval_table.loc[
        interval_table["interval_type"] == phase_key,
        ["start", "end"],
    ].iloc[0]
    start = float(interval["start"])
    end = float(interval["end"])

    in_interval = (pose_times >= start) & (pose_times <= end)
    pose = pose.loc[in_interval].reset_index(drop=True)
    pose_times = pose_times[in_interval]

    good_unit_ids = read_good_unit_ids(kilosort_dir)
    spike_clusters = np.asarray(
        np.load(
            kilosort_dir / "spike_clusters.npy",
            mmap_mode="r",
        ).reshape(-1),
        dtype=int,
    )
    spike_samples = np.asarray(
        np.load(
            kilosort_dir / "spike_times.npy",
            mmap_mode="r",
        ).reshape(-1),
        dtype=np.int64,
    )

    probe_timestamps_path = (
            Path(session_info["base_path"])
            / session_info["record_nodes"]
            / session_info["experiment_id"]
            / session_info["recording_name"]
            / "continuous"
            / session_info[f"continuous_probe_{probe_name}_folder"]
            / "timestamps.npy"
    )
    probe_timestamps = np.load(
        probe_timestamps_path,
        mmap_mode="r",
    )
    spike_times = (
            np.asarray(probe_timestamps[spike_samples], dtype=float)
            - adc_time_origin_s
    )
    spikes_in_interval = (
            (spike_times >= pose_times[0])
            & (spike_times <= pose_times[-1])
    )

    return (
        pose,
        pose_times,
        spike_times[spikes_in_interval],
        spike_clusters[spikes_in_interval],
        good_unit_ids,
    )


def prepare_session_maps(pose, pose_times):
    """Compute geometry and occupancy once, before forking unit workers."""
    center_x_px = pose["center_x"].to_numpy()
    center_y_px = pose["center_y"].to_numpy()
    center_x_cm = (center_x_px - x_min) * cm_per_px
    center_y_cm = (center_y_px - y_min) * cm_per_px
    head_direction_deg = pose["hd_deg"].to_numpy()

    theta_edges = np.arange(0, 360 + theta_bin_deg, theta_bin_deg)
    theta_deg = theta_edges[:-1]
    theta_d_cm = d(
        theta_deg,
        center_x_px,
        center_y_px,
        head_direction_deg,
    ) * cm_per_px
    theta_grid = np.broadcast_to(theta_deg[None, :], theta_d_cm.shape)

    maximum_distance_cm = rig_size_cm / 2 * np.sqrt(2)
    distance_edges = np.linspace(
        0,
        maximum_distance_cm,
        number_of_distance_bins + 1,
    )
    x_edges = np.linspace(
        0,
        rig_size_cm,
        number_of_spatial_bins + 1,
    )
    y_edges = np.linspace(
        0,
        rig_size_cm,
        number_of_spatial_bins + 1,
    )

    frame_dt_s = float(np.median(np.diff(pose_times)))
    frame_time_weights = np.full(len(pose_times), frame_dt_s)
    egocentric_occupancy = compute_2d_map(
        theta_grid,
        theta_d_cm,
        theta_edges,
        distance_edges,
        frame_time_weights,
    )
    head_direction_deg = head_direction_deg % 360
    allocentric_direction_occupancy = np.histogram(
        head_direction_deg,
        bins=theta_edges,
        weights=frame_time_weights,
    )[0]
    allocentric_occupancy = compute_2d_map(
        center_x_cm,
        center_y_cm,
        x_edges,
        y_edges,
        frame_time_weights,
    )

    return {
        "frame_times": pose_times,
        "head_direction_deg": head_direction_deg,
        "theta_grid": theta_grid,
        "theta_d_cm": theta_d_cm,
        "theta_edges": theta_edges,
        "distance_edges": distance_edges,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "trajectory_x_cm": center_x_cm,
        "trajectory_y_cm": center_y_cm,
        "egocentric_occupancy": egocentric_occupancy,
        "allocentric_occupancy": allocentric_occupancy,
        "smoothed_egocentric_occupancy": gaussian_filter(
            egocentric_occupancy,
            sigma=egocentric_smoothing_sigma,
            mode=("wrap", "nearest"),
        ),
        "smoothed_allocentric_occupancy": gaussian_filter(
            allocentric_occupancy,
            sigma=allocentric_smoothing_sigma,
            mode="nearest",
        ),
        "smoothed_direction_occupancy": gaussian_filter(
            allocentric_direction_occupancy,
            sigma=allocentric_smoothing_sigma,
            mode="wrap",
        ),
    }


def compute_maps(spike_times, session_maps):
    center_x_cm = session_maps["trajectory_x_cm"]
    center_y_cm = session_maps["trajectory_y_cm"]
    theta_edges = session_maps["theta_edges"]
    distance_edges = session_maps["distance_edges"]
    x_edges = session_maps["x_edges"]
    y_edges = session_maps["y_edges"]
    spike_frame_counts = count_spikes_by_frame(
        spike_times,
        session_maps["frame_times"],
    )

    egocentric_spike_map = compute_2d_map(
        session_maps["theta_grid"],
        session_maps["theta_d_cm"],
        theta_edges,
        distance_edges,
        spike_frame_counts,
    )
    egocentric_rate_map = compute_rate_map(
        egocentric_spike_map,
        session_maps["smoothed_egocentric_occupancy"],
        smoothing_sigma=egocentric_smoothing_sigma,
        smoothing_mode=("wrap", "nearest"),
    )
    preferred_distance_index = np.unravel_index(
        np.nanargmax(egocentric_rate_map),
        egocentric_rate_map.shape,
    )[1]
    egocentric_tuning_curve = egocentric_rate_map[
        :, preferred_distance_index
    ]
    preferred_distance_cm = np.mean(
        distance_edges[
            preferred_distance_index: preferred_distance_index + 2
        ]
    )

    allocentric_direction_spikes = np.histogram(
        session_maps["head_direction_deg"],
        bins=theta_edges,
        weights=spike_frame_counts,
    )[0]
    allocentric_tuning_curve = compute_rate_map(
        allocentric_direction_spikes,
        session_maps["smoothed_direction_occupancy"],
        smoothing_sigma=allocentric_smoothing_sigma,
        smoothing_mode="wrap",
    )
    allocentric_spike_map = compute_2d_map(
        center_x_cm,
        center_y_cm,
        x_edges,
        y_edges,
        spike_frame_counts,
    )
    allocentric_rate_map = compute_rate_map(
        allocentric_spike_map,
        session_maps["smoothed_allocentric_occupancy"],
        smoothing_sigma=allocentric_smoothing_sigma,
        smoothing_mode="nearest",
    )

    return {
        "theta_edges": theta_edges,
        "distance_edges": distance_edges,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "egocentric_occupancy": session_maps["egocentric_occupancy"],
        "egocentric_spike_map": egocentric_spike_map,
        "egocentric_rate_map": egocentric_rate_map,
        "egocentric_tuning_curve": egocentric_tuning_curve,
        "preferred_distance_cm": preferred_distance_cm,
        "allocentric_occupancy": session_maps["allocentric_occupancy"],
        "allocentric_spike_map": allocentric_spike_map,
        "allocentric_rate_map": allocentric_rate_map,
        "allocentric_tuning_curve": allocentric_tuning_curve,
        "trajectory_x_cm": center_x_cm,
        "trajectory_y_cm": center_y_cm,
        "spike_x_cm": np.repeat(center_x_cm, spike_frame_counts),
        "spike_y_cm": np.repeat(center_y_cm, spike_frame_counts),
    }


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


def plot_maps(maps, selected_unit_id):
    with plt.rc_context(LIGHT_PLOT_STYLE):
        figure = plt.figure(figsize=(30, 11), layout="constrained")
        grid = figure.add_gridspec(2, 5)
        for index, plot in enumerate(MAP_PLOTS):
            _, projection = PLOT_LAYOUTS[plot.kind]
            axis = figure.add_subplot(grid[divmod(index, 5)], projection=projection)
            draw_map(axis, maps, plot)
        figure.suptitle(
            f"rec {recording_number}, Probe{probe_name}, unit {selected_unit_id}",
            fontsize=16,
        )
    return figure


def prepare_output_directories():
    plot_types = ("spatial_maps", *(plot.filename for plot in MAP_PLOTS))
    for plot_type in plot_types:
        for file_type in FIGURE_FILE_TYPES:
            (save_root_directory / plot_type / file_type).mkdir(
                parents=True, exist_ok=True,
            )


def save_figure(figure, plot_type, unit_id):
    for file_type in FIGURE_FILE_TYPES:
        plot_directory = save_root_directory / plot_type / file_type
        figure.savefig(
            plot_directory / f"{unit_id}.{file_type}",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )


def save_individual_plots(maps, unit_id):
    with plt.rc_context(LIGHT_PLOT_STYLE):
        for plot in MAP_PLOTS:
            figsize, projection = PLOT_LAYOUTS[plot.kind]
            figure, axis = plt.subplots(
                figsize=figsize,
                layout="constrained",
                subplot_kw={"projection": projection},
            )
            draw_map(axis, maps, plot)
            save_figure(figure, plot.filename, unit_id)
            plt.close(figure)


def process_unit(selected_unit_id):
    session_maps, spike_times, spike_clusters = worker_data
    maps = compute_maps(
        spike_times[spike_clusters == selected_unit_id],
        session_maps,
    )
    figure = plot_maps(maps, selected_unit_id)
    save_figure(figure, "spatial_maps", selected_unit_id)
    plt.close(figure)
    save_individual_plots(maps, selected_unit_id)
    return selected_unit_id


def main():
    global worker_data

    (
        pose,
        pose_times,
        spike_times,
        spike_clusters,
        good_unit_ids,
    ) = load_data()
    print("good unit ids:", good_unit_ids)

    session_maps = prepare_session_maps(pose, pose_times)
    prepare_output_directories()
    worker_data = session_maps, spike_times, spike_clusters
    with ProcessPoolExecutor(
        mp_context=get_context("fork"),
    ) as executor:
        for selected_unit_id in executor.map(
            process_unit,
            good_unit_ids,
        ):
            print(
                f"saved unit {selected_unit_id}: {save_root_directory}"
            )


if __name__ == "__main__":
    main()
