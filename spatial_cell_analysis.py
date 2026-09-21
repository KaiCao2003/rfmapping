"""Save full egocentric tuning matrices and three distance-band bearing RFMaps."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
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
egocentric_smoothing_sigma = 5
distance_bands_cm = (
    ("0-8", None, 8),
    ("8-16", 8, 16),
    ("16-", 16, None),
)

worker_data = None


def d(theta_deg, center_x, center_y, head_direction_deg, *, bounds=None):
    """Return ray distances to the arena boundary in pixels."""
    left, right, top, bottom = (x_min, x_max, y_min, y_max) if bounds is None else bounds
    absolute_angle_rad = np.deg2rad(
        head_direction_deg[:, None] + theta_deg[None, :]
    )
    direction_x = -np.sin(absolute_angle_rad)
    direction_y = -np.cos(absolute_angle_rad)

    # Cardinal angles have tiny nonzero components from trigonometric roundoff.
    distance_x = np.divide(
        np.where(
            direction_x > 0,
            right - center_x[:, None],
            left - center_x[:, None],
        ),
        direction_x,
        out=np.full_like(direction_x, np.inf),
        where=np.abs(direction_x) > 1e-12,
    )
    distance_y = np.divide(
        np.where(
            direction_y > 0,
            bottom - center_y[:, None],
            top - center_y[:, None],
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


def nearest_frame_indices(spike_times, frame_times):
    """Assign in-range spikes to their nearest frame, breaking ties to the left."""
    right = np.searchsorted(frame_times, spike_times)
    left = np.maximum(right - 1, 0)
    use_right = (
            np.abs(frame_times[right] - spike_times)
            < np.abs(frame_times[left] - spike_times)
    )
    return np.where(use_right, right, left)


def count_spikes_by_frame(spike_times, frame_times):
    """Count spikes within the sorted frame-time interval loaded by load_data."""
    return np.bincount(
        nearest_frame_indices(spike_times, frame_times),
        minlength=len(frame_times),
    )


def prepare_2d_projection(coordinate_1, coordinate_2, edges_1, edges_2):
    """Bin shared frame geometry once, preserving histogram2d's edge rules."""
    first, second = np.broadcast_arrays(coordinate_1, coordinate_2)
    bins_1 = np.searchsorted(edges_1, first, side="right") - 1
    bins_2 = np.searchsorted(edges_2, second, side="right") - 1
    # Histogram bins are left-closed, except the final bin includes its right edge.
    bins_1[first == edges_1[-1]] -= 1
    bins_2[second == edges_2[-1]] -= 1
    shape = (len(edges_1) - 1, len(edges_2) - 1)
    valid = (bins_1 >= 0) & (bins_1 < shape[0]) & (bins_2 >= 0) & (bins_2 < shape[1])
    frames = np.broadcast_to(
        np.arange(len(first)).reshape((-1,) + (1,) * (first.ndim - 1)), first.shape,
    )
    rows = bins_1[valid] * shape[1] + bins_2[valid]
    return csr_matrix(
        (np.ones(len(rows)), (rows, frames[valid])),
        shape=(shape[0] * shape[1], len(first)),
    )


def group_spike_times(spike_times, spike_clusters, unit_ids):
    """Group selected spikes once so workers do not rescan every cluster ID."""
    selected = np.isin(spike_clusters, unit_ids)
    clusters = spike_clusters[selected]
    order = np.argsort(clusters, kind="stable")
    times = spike_times[selected][order]
    ids, starts, counts = np.unique(clusters[order], return_index=True, return_counts=True)
    groups = {int(unit): times[start:start + count]
              for unit, start, count in zip(ids, starts, counts)}
    return {int(unit): groups.get(int(unit), times[:0]) for unit in unit_ids}


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


def load_data(
    *, basler_output=True, optihub2_output=False,
    camera_input_channel=1, camera_ttl_threshold=14000,
    session_dir=None, probe=None, phase=None,
):
    if basler_output == optihub2_output:
        raise ValueError("Set exactly one of basler_output and optihub2_output to True.")
    session_dir = Path(session_dir) if session_dir is not None else (
        recording_root / date / f"{date}_{recording_number}"
    )
    selected_probe = probe_name if probe is None else probe
    selected_phase = phase_key if phase is None else phase
    session_date = session_dir.name.split("_")[0]
    data_dir = session_dir / "data"
    kilosort_dir = next(
        (session_dir / "kilosort" / f"Probe{selected_probe}").glob(
            "kilosort_*"
        )
    )

    pose = pd.read_csv(
        session_dir / f"{session_date}.csv",
        usecols=["frame", "center_x", "center_y", "hd_deg"],
    )

    session_info = read_formatted_json(
        data_dir / "session_info.json"
    )["session_info"]
    exposure_timestamps, adc_time_origin_s, timing_metadata = (
        get_exposure_timestamps(
            session_info=session_info,
            data_dir=data_dir,
            camera_input_channel=camera_input_channel,
            camera_ttl_threshold=camera_ttl_threshold,
            # Basler opto-coupled ExposureActive is electrically active-low.
            camera_ttl_active_high=optihub2_output,
        )
    )
    timing_metadata["camera_output"] = "basler" if basler_output else "optihub2"
    # Match tuning_curves.ipynb: Motive may export one final frame without a TTL.
    if optihub2_output and np.array_equal(
        pose["frame"].to_numpy(), np.arange(len(exposure_timestamps) + 1),
    ):
        pose = pose.iloc[:-1]
        timing_metadata["dropped_trailing_motive_frame"] = True
    pose = pose.dropna()
    frame_ids = pose["frame"].to_numpy(dtype=int)
    if np.any((frame_ids < 0) | (frame_ids >= len(exposure_timestamps))):
        raise ValueError(
            f"Pose frame IDs exceed the {len(exposure_timestamps)} camera timestamps "
            f"for {timing_metadata['camera_output']} output."
        )
    pose_times = exposure_timestamps[
        frame_ids
    ]

    interval_table = pd.read_csv(data_dir / "interval_table.csv")
    interval = interval_table.loc[
        interval_table["interval_type"] == selected_phase,
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
        data_dir / f"probe{selected_probe}" / "adc_spike_time.npy",
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
            "pose_path": str(session_dir / f"{session_date}.csv"),
            "kilosort_dir": str(kilosort_dir),
            "spike_times_path": str(
                data_dir / f"probe{selected_probe}" / "adc_spike_time.npy"
            ),
            "interval_table_path": str(data_dir / "interval_table.csv"),
            "selected_interval_s": [start, end],
            "adc_time_origin_s": float(adc_time_origin_s),
            "camera_timing": timing_metadata,
        },
    )


def prepare_session_maps(pose, pose_times):
    """Compute the shared geometry and occupancy needed for egocentric tuning."""
    theta_edges = np.arange(0, 360 + theta_bin_deg, theta_bin_deg)
    theta_deg = theta_edges[:-1]
    theta_d_cm = d(
        theta_deg,
        pose["center_x"].to_numpy(),
        pose["center_y"].to_numpy(),
        pose["hd_deg"].to_numpy(),
    ) * cm_per_px
    theta_grid = np.broadcast_to(theta_deg[None, :], theta_d_cm.shape)
    distance_edges = np.linspace(
        0, rig_size_cm / 2 * np.sqrt(2), number_of_distance_bins + 1,
    )
    frame_dt_s = float(np.median(np.diff(pose_times)))
    projection = prepare_2d_projection(theta_grid, theta_d_cm, theta_edges, distance_edges)
    occupancy = (projection @ np.full(len(pose_times), frame_dt_s)).reshape(
        len(theta_edges) - 1, len(distance_edges) - 1,
    )
    return {
        "frame_times": pose_times,
        "theta_grid": theta_grid,
        "theta_d_cm": theta_d_cm,
        "theta_edges": theta_edges,
        "distance_edges": distance_edges,
        "spike_projection": projection,
        "smoothed_egocentric_occupancy": gaussian_filter(
            occupancy, sigma=egocentric_smoothing_sigma, mode=("wrap", "nearest"),
        ),
    }


def compute_tuning_matrix(spike_times, session_maps):
    spike_frame_counts = count_spikes_by_frame(spike_times, session_maps["frame_times"])
    spike_map = (session_maps["spike_projection"] @ spike_frame_counts).reshape(
        session_maps["smoothed_egocentric_occupancy"].shape,
    )
    return compute_rate_map(
        spike_map, session_maps["smoothed_egocentric_occupancy"],
        smoothing_sigma=egocentric_smoothing_sigma, smoothing_mode=("wrap", "nearest"),
    )


def save_egocentric_rfmap(
    result_path, rate_maps, session_maps, unit_ids, interval_s,
    *, distance_band_cm=None, metadata=None,
):
    """Save full matrices or distance-band sums as a version-2 indexed RFMap."""
    values = np.asarray(rate_maps, dtype=np.float64)[..., None]
    distance_edges = session_maps["distance_edges"]
    theta_edges = session_maps["theta_edges"]
    band_metadata = {}
    if distance_band_cm is not None:
        lower_cm, upper_cm = distance_band_cm
        distance_centers = (distance_edges[:-1] + distance_edges[1:]) / 2
        selected = np.ones(len(distance_centers), dtype=bool)
        if lower_cm is not None:
            selected &= distance_centers > lower_cm
        if upper_cm is not None:
            selected &= distance_centers <= upper_cm
        # Sum the saved rates by bin center, matching the plotting projection.
        selected_values = values[:, :, selected, :]
        values = np.nansum(selected_values, axis=2, keepdims=True)
        values[~np.isfinite(selected_values).any(axis=2, keepdims=True)] = np.nan
        band_start_cm = distance_edges[0] if lower_cm is None else lower_cm
        band_end_cm = distance_edges[-1] if upper_cm is None else upper_cm
        # An open-ended band beyond the available grid has an empty extent.
        distance_edges = np.array([band_start_cm, max(band_start_cm, band_end_cm)])
        band_metadata = {
            "distanceSelection": "bin_centers",
            "distanceBinCentersCm": distance_centers[selected].tolist(),
            "responseAggregation": "sum_over_distance",
        }
    payload = {
        "unitsSpikeCountsSize": list(values.shape),
        "xBinEdges": distance_edges.tolist(),
        "yBinEdges": theta_edges.tolist(),
        "xUnits": "cm",
        "yUnits": "deg",
        "responseUnits": "Hz",
        "responseNormalization": "already_normalized",
        **band_metadata,
        **({} if metadata is None else metadata),
        "format": "rfmap",
        "formatVersion": 2,
        "storage": "indexed_npz",
        "unitArrayKeyPattern": "unit_{unit_id}",
        "unitArrayAxes": ["y", "x", "time"],
        "occupancyTimeSecDefinition": "normalization_identity_for_precomputed_rates",
    }
    arrays = {
        "metadata": np.frombuffer(json.dumps(payload, allow_nan=False).encode("utf-8"), dtype=np.uint8),
        "unitPool": np.asarray(unit_ids, dtype=np.int64),
        "xPositions": np.asarray((distance_edges[:-1] + distance_edges[1:]) / 2, dtype=np.float64),
        "yPositions": np.asarray((theta_edges[:-1] + theta_edges[1:]) / 2, dtype=np.float64),
        "timeBinEdges": np.asarray(interval_s, dtype=np.float64),
        # Required by the indexed schema; stored Hz values need no further division.
        # This identity array is not measured occupancy.
        "occupancyTimeSec": np.ones(values.shape[1:3], dtype=np.float64, order="F"),
        **{f"unit_{unit_id}": np.asfortranarray(matrix)
           for unit_id, matrix in zip(unit_ids, values, strict=True)},
    }
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Passing a stream preserves the .rfmap suffix instead of appending .npz.
    temporary_path = result_path.with_name(result_path.name + ".tmp")
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary_path.replace(result_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_egocentric_rfmaps(result_path, rate_maps, session_maps, unit_ids, interval_s):
    """Save the full map, then <=8 cm, (8, 16] cm, and >16 cm bearing maps."""
    result_path = Path(result_path)
    rate_maps = np.asarray(rate_maps, dtype=np.float64)
    save_egocentric_rfmap(result_path, rate_maps, session_maps, unit_ids, interval_s)
    result_paths = [result_path]
    for suffix, lower_cm, upper_cm in distance_bands_cm:
        band_path = result_path.with_name(f"{result_path.stem}_{suffix}{result_path.suffix}")
        save_egocentric_rfmap(
            band_path, rate_maps, session_maps, unit_ids, interval_s,
            distance_band_cm=(lower_cm, upper_cm),
        )
        result_paths.append(band_path)
    return result_paths


def process_unit(selected_unit_id):
    session_maps, spikes_by_unit = worker_data
    return compute_tuning_matrix(
        spikes_by_unit[selected_unit_id], session_maps,
    )


def run_analysis(
    *, output=None, units=None, workers=None,
    basler_output=True, optihub2_output=False,
    camera_input_channel=1, camera_ttl_threshold=14000,
):
    """Analyze the recording and return all four saved RFMap paths, full map first."""
    global worker_data
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    result_path = (
        Path(output).expanduser().resolve() if output else
        recording_root / date / f"{date}_{recording_number}"
        / "data" / "spatial_cells" / f"Probe{probe_name}" / phase_key
        / "egocentric_rate_map.rfmap"
    )
    (
        pose, pose_times, spike_times, spike_clusters, good_unit_ids, source_metadata,
    ) = load_data(
        basler_output=basler_output, optihub2_output=optihub2_output,
        camera_input_channel=camera_input_channel,
        camera_ttl_threshold=camera_ttl_threshold,
    )
    timing = source_metadata["camera_timing"]
    print(
        f"{timing['camera_output']}: {timing['ttl_pulse_count']} camera frames, "
        f"{timing['measured_rate_hz']:.3f} Hz ({timing['timestamp_reference']}); "
        f"{len(pose)} valid pose frames in {phase_key}"
    )
    unit_ids = good_unit_ids if units is None else list(dict.fromkeys(units))
    if not unit_ids:
        raise ValueError("No good units selected.")
    unknown = set(unit_ids) - set(good_unit_ids)
    if unknown:
        raise ValueError(f"Units absent from good-unit labels: {sorted(unknown)}")
    session_maps = prepare_session_maps(pose, pose_times)
    worker_data = session_maps, group_spike_times(spike_times, spike_clusters, unit_ids)
    del spike_times, spike_clusters
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=get_context("fork"),
        ) as executor:
            rate_maps = list(executor.map(process_unit, unit_ids))
    finally:
        worker_data = None
    result_paths = save_egocentric_rfmaps(
        result_path, rate_maps, session_maps, unit_ids,
        source_metadata["selected_interval_s"],
    )
    for path in result_paths:
        print(f"Saved {len(unit_ids)} tuning matrices: {path}")
    return result_paths


def main(argv=None):
    global recording_root, date, recording_number, probe_name, phase_key

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=recording_root)
    parser.add_argument("--date", default=date)
    parser.add_argument("--recording-number", type=int, default=recording_number)
    parser.add_argument("--probe", choices=("A", "B"), default=probe_name)
    parser.add_argument("--phase", default=phase_key)
    parser.add_argument(
        "--output", type=Path,
        help="Full .rfmap filename; three distance-band files use the same stem (all replaced on rerun)",
    )
    parser.add_argument("--units", type=int, nargs="+", help="Subset of good unit IDs")
    parser.add_argument("--workers", type=int, help="Number of analysis processes")
    parser.add_argument("--basler-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optihub2-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-input-channel", type=int, default=1)
    parser.add_argument("--camera-ttl-threshold", type=float, default=14000)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    recording_root = args.recording_root.expanduser().resolve()
    date = args.date
    recording_number = args.recording_number
    probe_name = args.probe
    phase_key = args.phase
    return run_analysis(
        output=args.output, units=args.units, workers=args.workers,
        basler_output=args.basler_output, optihub2_output=args.optihub2_output,
        camera_input_channel=args.camera_input_channel,
        camera_ttl_threshold=args.camera_ttl_threshold,
    )


if __name__ == "__main__":
    main()
