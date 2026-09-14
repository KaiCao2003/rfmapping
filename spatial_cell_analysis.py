"""Analyze spatial cells once and save results for independent plotting."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from Utils.json_tools import read_formatted_json
from Utils.tuning_curve_utils import get_exposure_timestamps

recording_root = Path("/mnt/senzailab/Kai/#Recording/m19")
date = "260831"
recording_number = 2
probe_name = "A"
phase_key = "baseline"

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
    / f"Probe{probe_name}"
    / phase_key
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

    pose = pd.read_csv(
        session_dir / f"{date}.csv",
        usecols=["frame", "center_x", "center_y", "hd_deg"],
    ).dropna()

    session_info = read_formatted_json(
        data_dir / "session_info.json"
    )["session_info"]
    exposure_timestamps, adc_time_origin_s, timing_metadata = (
        get_exposure_timestamps(
            session_info=session_info,
            data_dir=data_dir,
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
    spike_times = np.load(
        data_dir / f"probe{probe_name}" / "adc_spike_time.npy",
        mmap_mode="r",
    ).reshape(-1) - adc_time_origin_s
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
        {
            "pose_path": str(session_dir / f"{date}.csv"),
            "kilosort_dir": str(kilosort_dir),
            "spike_times_path": str(
                data_dir / f"probe{probe_name}" / "adc_spike_time.npy"
            ),
            "interval_table_path": str(data_dir / "interval_table.csv"),
            "selected_interval_s": [start, end],
            "adc_time_origin_s": float(adc_time_origin_s),
            "camera_timing": timing_metadata,
        },
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
        "spike_frame_counts": spike_frame_counts,
        "allocentric_occupancy": session_maps["allocentric_occupancy"],
        "allocentric_spike_map": allocentric_spike_map,
        "allocentric_rate_map": allocentric_rate_map,
        "allocentric_tuning_curve": allocentric_tuning_curve,
        "trajectory_x_cm": center_x_cm,
        "trajectory_y_cm": center_y_cm,
    }


# Only these shared arrays are needed to reproduce the figures.
SESSION_KEYS = (
    "frame_times", "theta_edges", "distance_edges", "x_edges", "y_edges",
    "trajectory_x_cm", "trajectory_y_cm", "egocentric_occupancy",
    "allocentric_occupancy",
)


def save_session(session_maps, unit_ids, source_metadata):
    save_root_directory.mkdir(parents=True, exist_ok=True)
    (save_root_directory / "units").mkdir(exist_ok=True)
    np.savez_compressed(
        save_root_directory / "session.npz",
        **{key: session_maps[key] for key in SESSION_KEYS},
    )
    metadata = {
        "schema_name": "rfmapping-spatial-cells",
        "schema_version": 1,
        "recording_root": str(recording_root),
        "date": date,
        "recording_number": recording_number,
        "probe_name": probe_name,
        "phase_key": phase_key,
        "unit_ids": unit_ids,
        "source": source_metadata,
        "arena_bounds_px": [x_min, x_max, y_min, y_max],
        "rig_size_cm": rig_size_cm, "cm_per_px": cm_per_px,
        "theta_bin_deg": theta_bin_deg,
        "number_of_distance_bins": number_of_distance_bins,
        "number_of_spatial_bins": number_of_spatial_bins,
        "egocentric_smoothing_sigma": egocentric_smoothing_sigma,
        "allocentric_smoothing_sigma": allocentric_smoothing_sigma,
        "egocentric_axis_order": ["angle_deg", "distance_cm"],
        "allocentric_axis_order": ["x_cm", "y_cm"],
        "angle_convention": "0 forward; positive toward left in image coordinates",
        "trajectory_coordinates": "arena origin at top left; y increases down",
        "rate_unit": "Hz", "occupancy_unit": "seconds",
        "smoothing_sigma_unit": "bins",
        "egocentric_smoothing_mode": ["wrap", "nearest"],
        "allocentric_smoothing_mode": "nearest",
        "direction_smoothing_mode": "wrap",
        "frame_time_reference": "seconds relative to ADC origin",
        "frame_dt_s": float(np.median(np.diff(session_maps["frame_times"]))),
        "occupancy_weighting": "median interval between retained pose frames",
        "spike_frame_assignment": "nearest retained frame; ties go to earlier frame",
        "egocentric_ray_angles": "theta_edges[:-1]",
    }
    return metadata


def save_unit(maps, unit_id):
    np.savez_compressed(
        save_root_directory / "units" / f"{unit_id}.npz",
        **{key: value for key, value in maps.items() if key not in SESSION_KEYS},
    )


def process_unit(selected_unit_id):
    session_maps, spike_times, spike_clusters = worker_data
    maps = compute_maps(
        spike_times[spike_clusters == selected_unit_id], session_maps,
    )
    save_unit(maps, selected_unit_id)
    return selected_unit_id


def main(argv=None):
    global worker_data, save_root_directory
    global recording_root, date, recording_number, probe_name, phase_key

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=recording_root)
    parser.add_argument("--date", default=date)
    parser.add_argument("--recording-number", type=int, default=recording_number)
    parser.add_argument("--probe", choices=("A", "B"), default=probe_name)
    parser.add_argument("--phase", default=phase_key)
    parser.add_argument("--output", type=Path, help="New result directory")
    parser.add_argument("--units", type=int, nargs="+", help="Subset of good unit IDs")
    parser.add_argument("--workers", type=int, help="Number of analysis processes")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    recording_root = args.recording_root.expanduser().resolve()
    date = args.date
    recording_number = args.recording_number
    probe_name = args.probe
    phase_key = args.phase
    save_root_directory = (
        args.output.expanduser().resolve() if args.output else
        recording_root / date / f"{date}_{recording_number}"
        / "data" / "spatial_cells" / f"Probe{probe_name}" / phase_key
    )
    # A new directory prevents mixing units or metadata from different runs.
    save_root_directory.mkdir(parents=True, exist_ok=False)
    (
        pose, pose_times, spike_times, spike_clusters, good_unit_ids, source_metadata,
    ) = load_data()
    unit_ids = good_unit_ids if args.units is None else list(dict.fromkeys(args.units))
    unknown = set(unit_ids) - set(good_unit_ids)
    if unknown:
        parser.error(f"Units absent from good-unit labels: {sorted(unknown)}")
    session_maps = prepare_session_maps(pose, pose_times)
    metadata = save_session(session_maps, unit_ids, source_metadata)
    worker_data = session_maps, spike_times, spike_clusters
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("fork"),
    ) as executor:
        for unit_id in executor.map(process_unit, unit_ids):
            print(f"saved unit {unit_id}: {save_root_directory}")
    # Publish the manifest last, so interrupted analysis cannot look complete.
    manifest_path = save_root_directory / "metadata.json.tmp"
    manifest_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    manifest_path.replace(save_root_directory / "metadata.json")


if __name__ == "__main__":
    main()
