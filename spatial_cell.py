from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from Utils.json_tools import read_formatted_json
from Utils.plotting import (
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
rig_size_px = 550
rig_size_cm = 41
cm_per_px = rig_size_cm / rig_size_px

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


def px_to_cm(px: int) -> int:
    return int(px * cm_per_px + 0.5)


def d(theta_deg, center_x, center_y, head_direction_deg):
    absolute_angle_rad = np.deg2rad(
        head_direction_deg[:, None] + theta_deg[None, :]
    )
    direction_x = -np.sin(absolute_angle_rad)
    direction_y = -np.cos(absolute_angle_rad)

    with np.errstate(divide="ignore", invalid="ignore"):
        distance_x = np.where(
            direction_x > 0,
            (x_max - center_x[:, None]) / direction_x,
            (x_min - center_x[:, None]) / direction_x,
        )
        distance_y = np.where(
            direction_y > 0,
            (y_max - center_y[:, None]) / direction_y,
            (y_min - center_y[:, None]) / direction_y,
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
        occupancy_map,
        smoothing_sigma,
        smoothing_mode,
):
    smoothed_spikes = gaussian_filter(
        spike_map,
        sigma=smoothing_sigma,
        mode=smoothing_mode,
    )
    smoothed_occupancy = gaussian_filter(
        occupancy_map,
        sigma=smoothing_sigma,
        mode=smoothing_mode,
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
    insertion = np.searchsorted(frame_times, spike_times)
    right = np.clip(insertion, 0, len(frame_times) - 1)
    left = np.clip(insertion - 1, 0, len(frame_times) - 1)
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


def compute_maps(unit_info):
    center_x_px = unit_info["x"]
    center_y_px = unit_info["y"]
    center_x_cm = (center_x_px - x_min) * cm_per_px
    center_y_cm = (center_y_px - y_min) * cm_per_px
    head_direction_deg = unit_info["direction"]
    pose_times = unit_info["frame_times"]
    spike_times = unit_info["spike_times"]

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
    spike_frame_counts = count_spikes_by_frame(
        spike_times,
        pose_times,
    )
    spike_x_cm = np.repeat(center_x_cm, spike_frame_counts)
    spike_y_cm = np.repeat(center_y_cm, spike_frame_counts)

    egocentric_occupancy = compute_2d_map(
        theta_grid,
        theta_d_cm,
        theta_edges,
        distance_edges,
        frame_time_weights,
    )
    egocentric_spike_map = compute_2d_map(
        theta_grid,
        theta_d_cm,
        theta_edges,
        distance_edges,
        spike_frame_counts,
    )
    egocentric_rate_map = compute_rate_map(
        egocentric_spike_map,
        egocentric_occupancy,
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

    allocentric_direction_occupancy = np.histogram(
        head_direction_deg % 360,
        bins=theta_edges,
        weights=frame_time_weights,
    )[0]
    allocentric_direction_spikes = np.histogram(
        head_direction_deg % 360,
        bins=theta_edges,
        weights=spike_frame_counts,
    )[0]
    allocentric_tuning_curve = compute_rate_map(
        allocentric_direction_spikes,
        allocentric_direction_occupancy,
        smoothing_sigma=allocentric_smoothing_sigma,
        smoothing_mode="wrap",
    )

    allocentric_occupancy = compute_2d_map(
        center_x_cm,
        center_y_cm,
        x_edges,
        y_edges,
        frame_time_weights,
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
        allocentric_occupancy,
        smoothing_sigma=allocentric_smoothing_sigma,
        smoothing_mode="nearest",
    )

    return {
        "theta_edges": theta_edges,
        "distance_edges": distance_edges,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "egocentric_occupancy": egocentric_occupancy,
        "egocentric_spike_map": egocentric_spike_map,
        "egocentric_rate_map": egocentric_rate_map,
        "egocentric_tuning_curve": egocentric_tuning_curve,
        "preferred_distance_cm": preferred_distance_cm,
        "allocentric_occupancy": allocentric_occupancy,
        "allocentric_spike_map": allocentric_spike_map,
        "allocentric_rate_map": allocentric_rate_map,
        "allocentric_tuning_curve": allocentric_tuning_curve,
        "trajectory_x_cm": center_x_cm,
        "trajectory_y_cm": center_y_cm,
        "spike_x_cm": spike_x_cm,
        "spike_y_cm": spike_y_cm,
    }


def plot_maps(maps, selected_unit_id):
    figure = plt.figure(figsize=(30, 11), layout="constrained")
    grid = figure.add_gridspec(2, 5)

    egocentric_time_axis = figure.add_subplot(grid[0, 0])
    egocentric_spike_axis = figure.add_subplot(grid[0, 1])
    egocentric_rate_axis = figure.add_subplot(grid[0, 2])
    egocentric_polar_axis = figure.add_subplot(
        grid[0, 3],
        projection="polar",
    )
    egocentric_tuning_axis = figure.add_subplot(
        grid[0, 4],
        projection="polar",
    )
    allocentric_time_axis = figure.add_subplot(grid[1, 0])
    allocentric_spike_axis = figure.add_subplot(grid[1, 1])
    allocentric_rate_axis = figure.add_subplot(grid[1, 2])
    trajectory_axis = figure.add_subplot(grid[1, 3])
    allocentric_tuning_axis = figure.add_subplot(
        grid[1, 4],
        projection="polar",
    )

    plot_egocentric_heatmap(
        egocentric_time_axis,
        maps["distance_edges"],
        maps["theta_edges"],
        maps["egocentric_occupancy"],
        "Egocentric time map",
        "Seconds",
        cmap="magma",
    )
    plot_egocentric_heatmap(
        egocentric_spike_axis,
        maps["distance_edges"],
        maps["theta_edges"],
        maps["egocentric_spike_map"],
        "Egocentric spike map",
        "Spikes",
        cmap="magma",
    )
    plot_egocentric_heatmap(
        egocentric_rate_axis,
        maps["distance_edges"],
        maps["theta_edges"],
        maps["egocentric_rate_map"],
        "Egocentric firing-rate map",
        "Hz",
    )
    plot_egocentric_polar(
        egocentric_polar_axis,
        maps["theta_edges"],
        maps["distance_edges"],
        maps["egocentric_rate_map"],
    )
    plot_tuning_curve(
        egocentric_tuning_axis,
        maps["theta_edges"],
        maps["egocentric_tuning_curve"],
        f"Egocentric tuning ({maps['preferred_distance_cm']:.1f} cm)",
    )

    plot_allocentric_heatmap(
        allocentric_time_axis,
        maps["x_edges"],
        maps["y_edges"],
        maps["allocentric_occupancy"],
        "Allocentric occupancy",
        "Seconds",
        cmap="magma",
    )
    plot_allocentric_heatmap(
        allocentric_spike_axis,
        maps["x_edges"],
        maps["y_edges"],
        maps["allocentric_spike_map"],
        "Allocentric spike map",
        "Spikes",
        cmap="magma",
    )
    plot_allocentric_heatmap(
        allocentric_rate_axis,
        maps["x_edges"],
        maps["y_edges"],
        maps["allocentric_rate_map"],
        "Allocentric firing-rate map",
        "Hz",
    )
    plot_trajectory_spikes(
        trajectory_axis,
        maps["trajectory_x_cm"],
        maps["trajectory_y_cm"],
        maps["spike_x_cm"],
        maps["spike_y_cm"],
    )
    plot_tuning_curve(
        allocentric_tuning_axis,
        maps["theta_edges"],
        maps["allocentric_tuning_curve"],
        "Allocentric HD tuning",
    )

    figure.suptitle(
        f"rec {recording_number}, Probe{probe_name}, unit {selected_unit_id}",
        fontsize=16,
    )
    return figure


def save_figure(figure, plot_type, unit_id):
    for file_type in ["png", "svg"]:
        plot_directory = save_root_directory / plot_type / file_type
        plot_directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            plot_directory / f"{unit_id}.{file_type}",
            dpi=200,
            bbox_inches="tight",
        )


def save_individual_plots(maps, unit_id):
    egocentric_plots = [
        (
            "egocentric_time_map",
            "egocentric_occupancy",
            "Egocentric time map",
            "Seconds",
            "magma",
        ),
        (
            "egocentric_spike_map",
            "egocentric_spike_map",
            "Egocentric spike map",
            "Spikes",
            "magma",
        ),
        (
            "egocentric_rate_map",
            "egocentric_rate_map",
            "Egocentric firing-rate map",
            "Hz",
            "viridis",
        ),
    ]
    for filename, map_key, title, colorbar_label, cmap in egocentric_plots:
        figure, axis = plt.subplots(
            figsize=(7, 5.5),
            layout="constrained",
        )
        plot_egocentric_heatmap(
            axis,
            maps["distance_edges"],
            maps["theta_edges"],
            maps[map_key],
            title,
            colorbar_label,
            cmap=cmap,
        )
        save_figure(figure, filename, unit_id)
        plt.close(figure)

    polar_figure, polar_axis = plt.subplots(
        figsize=(7, 6),
        layout="constrained",
        subplot_kw={"projection": "polar"},
    )
    plot_egocentric_polar(
        polar_axis,
        maps["theta_edges"],
        maps["distance_edges"],
        maps["egocentric_rate_map"],
    )
    save_figure(
        polar_figure,
        "egocentric_rate_map_polar",
        unit_id,
    )
    plt.close(polar_figure)

    tuning_curve_plots = [
        (
            "egocentric_tuning_curve",
            "egocentric_tuning_curve",
            f"Egocentric tuning ({maps['preferred_distance_cm']:.1f} cm)",
        ),
        (
            "allocentric_tuning_curve",
            "allocentric_tuning_curve",
            "Allocentric HD tuning",
        ),
    ]
    for filename, map_key, title in tuning_curve_plots:
        figure, axis = plt.subplots(
            figsize=(6, 6),
            layout="constrained",
            subplot_kw={"projection": "polar"},
        )
        plot_tuning_curve(
            axis,
            maps["theta_edges"],
            maps[map_key],
            title,
        )
        save_figure(figure, filename, unit_id)
        plt.close(figure)

    allocentric_plots = [
        (
            "allocentric_occupancy",
            "allocentric_occupancy",
            "Allocentric occupancy",
            "Seconds",
            "magma",
        ),
        (
            "allocentric_spike_map",
            "allocentric_spike_map",
            "Allocentric spike map",
            "Spikes",
            "magma",
        ),
        (
            "allocentric_rate_map",
            "allocentric_rate_map",
            "Allocentric firing-rate map",
            "Hz",
            "viridis",
        ),
    ]
    for filename, map_key, title, colorbar_label, cmap in allocentric_plots:
        figure, axis = plt.subplots(
            figsize=(6.5, 6),
            layout="constrained",
        )
        plot_allocentric_heatmap(
            axis,
            maps["x_edges"],
            maps["y_edges"],
            maps[map_key],
            title,
            colorbar_label,
            cmap=cmap,
        )
        save_figure(figure, filename, unit_id)
        plt.close(figure)

    trajectory_figure, trajectory_axis = plt.subplots(
        figsize=(6.5, 6),
        layout="constrained",
    )
    plot_trajectory_spikes(
        trajectory_axis,
        maps["trajectory_x_cm"],
        maps["trajectory_y_cm"],
        maps["spike_x_cm"],
        maps["spike_y_cm"],
    )
    save_figure(
        trajectory_figure,
        "trajectory_spike_positions",
        unit_id,
    )
    plt.close(trajectory_figure)


def process_unit(selected_unit_id):
    pose, pose_times, spike_times, spike_clusters = worker_data
    unit_info = {
        "unit_id": selected_unit_id,
        "x": pose["center_x"].to_numpy(),
        "y": pose["center_y"].to_numpy(),
        "direction": pose["hd_deg"].to_numpy(),
        "frame_times": pose_times,
        "spike_times": spike_times[
            spike_clusters == selected_unit_id
        ],
    }

    maps = compute_maps(unit_info)
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

    worker_data = pose, pose_times, spike_times, spike_clusters
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
