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
from EBB.ebc_geometry import BASLER_SIZE_CM, pixel_heading_to_world_deg, rectangle_distances_cm
from EBB.ebc_pose import POSITION_CORRECTION_POLICY, hold_rectangle_positions
from Utils.json_tools import read_formatted_json
from EBB.spatial_cell_results import collapse_distance_band, spatial_result_path
from Utils.camera_timing import get_exposure_timestamps

recording_root = Path("/mnt/senzailab/Kai/#Recording/m19")
date = "260831"
recording_number = 2
probe_name = "A"
phase_key = "baseline"
legacy_basler_channel_one = {"m19_260831_2", "m20_260918_11", "m22_261009_4"}
session_bounds_px = {
    "m19_260831_2": (370., 920., 210., 760.),
    "m20_260918_11": (370., 920., 210., 760.),
    "m20_260921_11": (225., 720., 280., 785.),
    "m21_261006_11": (225., 720., 280., 785.),
    "m22_261008_11": (225., 720., 280., 785.),
    "m22_261009_3": (230., 740., 335., 835.),
    "m22_261009_4": (364.4598693847656, 920.0496215820312,
                      218.121337890625, 763.5052490234375),
}

# Basler bounds in pixels: left, right, top, bottom.
boundary_old = (370., 920., 210., 760.)
boundary_new = (225., 720., 280., 785.)
x_min, x_max, y_min, y_max = boundary_new
rig_size_cm = BASLER_SIZE_CM
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


def basler_session_defaults(session_key, boundary="new"):
    """Return the saved session boundary and camera channel used by every entry."""
    bounds = session_bounds_px.get(
        session_key, boundary_old if boundary == "old" else boundary_new,
    )
    channel = 1 if session_key in legacy_basler_channel_one else 6
    return bounds, channel


def d(theta_deg, center_x, center_y, head_direction_deg, *, bounds=None):
    """Return pixel boundary distances for north-zero CW HD and CW ego bearings."""
    left, right, top, bottom = (x_min, x_max, y_min, y_max) if bounds is None else bounds
    absolute_angle_rad = np.deg2rad(
        head_direction_deg[:, None] + theta_deg[None, :]
    )
    direction_x = np.sin(absolute_angle_rad)
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
    camera_input_channel=None, camera_ttl_threshold=14000,
    session_dir=None, probe=None, phase=None, bounds_px=None,
):
    if basler_output == optihub2_output:
        raise ValueError("Set exactly one of basler_output and optihub2_output to True.")
    session_dir = Path(session_dir) if session_dir is not None else (
        recording_root / date / f"{date}_{recording_number}"
    )
    session_key = f"{session_dir.parent.parent.name}_{session_dir.name}"
    if camera_input_channel is None:
        camera_input_channel = (
            1 if optihub2_output or session_key in legacy_basler_channel_one else 6
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

    pose_name = f"{session_date}.csv"
    if basler_output and any(session_dir.glob("*.tak")):
        pose_name = f"{session_date}_yolo.csv"
    pose_path = session_dir / pose_name
    pose = pd.read_csv(
        pose_path,
        usecols=["frame", "center_x", "center_y", "hd_deg"],
    )
    bounds_px = (
        session_bounds_px.get(session_key, (x_min, x_max, y_min, y_max))
        if bounds_px is None else tuple(bounds_px)
    )
    raw_positions = pose[["center_x", "center_y"]].to_numpy()
    corrected, held, initial_outside = hold_rectangle_positions(raw_positions, bounds_px)
    pose[["raw_center_x", "raw_center_y"]] = raw_positions
    pose[["center_x", "center_y"]] = corrected
    pose["position_held"] = held
    pose["initial_outside"] = initial_outside

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
    # Keep the original frame selection, including leading outside frames whose
    # corrected location is missing; heading and spike assignment remain intact.
    pose = pose.dropna(subset=["frame", "raw_center_x", "raw_center_y", "hd_deg"])
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
    # Preserve the complete camera clock for video/audio, including missing pose
    # rows and frames outside this phase; the result saver writes it to NPZ.
    pose.attrs["exposure_times"] = exposure_timestamps

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
            "pose_path": str(pose_path),
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
    if "position_held" in pose:
        corrected = pose[["center_x", "center_y"]].to_numpy()
        raw_positions = pose[["raw_center_x", "raw_center_y"]].to_numpy()
        held = pose.position_held.to_numpy()
        initial_outside = pose.initial_outside.to_numpy()
    else:
        raw_positions = pose[["center_x", "center_y"]].to_numpy()
        corrected, held, initial_outside = hold_rectangle_positions(
            raw_positions, (x_min, x_max, y_min, y_max),
        )
    theta_edges = np.arange(0, 360 + theta_bin_deg, theta_bin_deg)
    theta_deg = (theta_edges[:-1] + theta_edges[1:]) / 2
    theta_d_cm = rectangle_distances_cm(
        corrected[:, 0],
        corrected[:, 1],
        pose["hd_deg"].to_numpy(),
        theta_deg, bounds_px=(x_min, x_max, y_min, y_max), size_cm=rig_size_cm,
    )
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
        "raw_positions_px": raw_positions,
        "corrected_positions_px": corrected,
        "position_held": held,
        "initial_outside": initial_outside,
        "ebc_hd_deg": (90. - pixel_heading_to_world_deg(
            pose["hd_deg"].to_numpy(), bounds_px=(x_min, x_max, y_min, y_max),
            size_cm=rig_size_cm,
        )) % 360.,
        "geometry_metadata": {
            "boundaryPx": [x_min, x_max, y_min, y_max],
            "arenaSizeCm": [rig_size_cm, rig_size_cm],
            "cmPerPixelXY": [rig_size_cm / (x_max - x_min), rig_size_cm / (y_max - y_min)],
            "headingConvention": "physical north-zero clockwise heading; clockwise egocentric bearings",
            "rawHeadingConvention": "pixel-plane north-zero clockwise",
            "headingCalibration": "image_unit_vector_scaled_by_cm_per_pixel_xy",
            "outsideBoundaryPolicy": POSITION_CORRECTION_POLICY,
            "positionCorrectionPolicy": POSITION_CORRECTION_POLICY,
            "heldPositionCount": int(held.sum()),
            "initialOutsideCount": int(initial_outside.sum()),
        },
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
    values = np.asarray(rate_maps, dtype=np.float64)
    distance_edges = session_maps["distance_edges"]
    theta_edges = session_maps["theta_edges"]
    band_metadata = {}
    if distance_band_cm is not None:
        full_distance_edges = distance_edges
        values, distance_edges, distance_centers = collapse_distance_band(
            values, distance_edges, distance_band_cm,
        )
        selected_bins = np.searchsorted(full_distance_edges, distance_centers, side="right") - 1
        band_metadata = {
            "distanceSelection": "bin_centers",
            "distanceBinCentersCm": distance_centers.tolist(),
            "requestedDistanceBandCm": list(distance_band_cm),
            "distanceBinSupportCm": [
                float(full_distance_edges[selected_bins[0]]),
                float(full_distance_edges[selected_bins[-1] + 1]),
            ] if len(distance_centers) else [],
            "responseAggregation": "sum_over_distance",
        }
    values = values[..., None]
    payload = {
        "unitsSpikeCountsSize": list(values.shape),
        "xBinEdges": distance_edges.tolist(),
        "yBinEdges": theta_edges.tolist(),
        "xUnits": "cm",
        "yUnits": "deg",
        "responseUnits": "Hz",
        "responseNormalization": "already_normalized",
        "angleConvention": "egocentric clockwise: 0 forward, 90 right",
        "angleSampling": "bin_centers",
        **session_maps.get("geometry_metadata", {}),
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
    camera_input_channel=None, camera_ttl_threshold=14000, bounds_px=None,
):
    """Analyze the recording and return all four saved RFMap paths, full map first."""
    global worker_data, x_min, x_max, y_min, y_max, cm_per_px
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    session_dir = recording_root / date / f"{date}_{recording_number}"
    result_path = (
        Path(output).expanduser().resolve() if output else
        spatial_result_path(session_dir, probe_name=probe_name, phase_key=phase_key)
    )
    session_key = f"{session_dir.parent.parent.name}_{session_dir.name}"
    if camera_input_channel is None:
        camera_input_channel = (
            1 if optihub2_output or session_key in legacy_basler_channel_one else 6
        )
    bounds_px = (
        session_bounds_px.get(session_key, (x_min, x_max, y_min, y_max))
        if bounds_px is None else tuple(bounds_px)
    )
    x_min, x_max, y_min, y_max = bounds_px
    cm_per_px = rig_size_cm / (x_max - x_min)
    (
        pose, pose_times, spike_times, spike_clusters, good_unit_ids, source_metadata,
    ) = load_data(
        basler_output=basler_output, optihub2_output=optihub2_output,
        camera_input_channel=camera_input_channel,
        camera_ttl_threshold=camera_ttl_threshold, bounds_px=bounds_px,
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
    spikes_by_unit = group_spike_times(spike_times, spike_clusters, unit_ids)
    worker_data = session_maps, spikes_by_unit
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
    from EBB.spatial_cell_results import save_spatial_plot_inputs
    save_spatial_plot_inputs(
        result_path.parent, pose, pose_times, spikes_by_unit, unit_ids, session_maps, rate_maps,
        dict(date=date, ebc_session=recording_number, probe_name=probe_name, phase_key=phase_key,
             source=source_metadata),
    )
    for path in result_paths:
        print(f"Saved {len(unit_ids)} tuning matrices: {path}")
    return result_paths


def main(argv=None):
    global recording_root, date, recording_number, probe_name, phase_key
    global x_min, x_max, y_min, y_max, cm_per_px

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=recording_root)
    parser.add_argument("--date", default=date)
    parser.add_argument("--recording-number", type=int, default=recording_number)
    parser.add_argument("--probe", choices=("A", "B"), default=probe_name)
    parser.add_argument("--phase", default=phase_key)
    parser.add_argument("--bounds-px", type=float, nargs=4, metavar=("LEFT", "RIGHT", "TOP", "BOTTOM"))
    parser.add_argument("--boundary", choices=("old", "new"), default="new")
    parser.add_argument(
        "--output", type=Path,
        help="Full .rfmap filename; three distance-band files use the same stem (all replaced on rerun)",
    )
    parser.add_argument("--units", type=int, nargs="+", help="Subset of good unit IDs")
    parser.add_argument("--workers", type=int, help="Number of analysis processes")
    parser.add_argument("--basler-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optihub2-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--camera-input-channel", type=int)
    parser.add_argument("--camera-ttl-threshold", type=float, default=14000)
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")
    recording_root = args.recording_root.expanduser().resolve()
    date = args.date
    recording_number = args.recording_number
    probe_name = args.probe
    phase_key = args.phase
    session_key = f"{recording_root.name}_{date}_{recording_number}"
    default_bounds, default_channel = basler_session_defaults(session_key, args.boundary)
    bounds = (
        tuple(args.bounds_px) if args.bounds_px is not None
        else default_bounds
    )
    x_min, x_max, y_min, y_max = bounds
    cm_per_px = rig_size_cm / (x_max - x_min)
    camera_input_channel = args.camera_input_channel
    if camera_input_channel is None:
        camera_input_channel = (
            default_channel if args.basler_output else 1
        )
    return run_analysis(
        output=args.output, units=args.units, workers=args.workers,
        basler_output=args.basler_output, optihub2_output=args.optihub2_output,
        camera_input_channel=camera_input_channel,
        camera_ttl_threshold=args.camera_ttl_threshold, bounds_px=bounds,
    )


if __name__ == "__main__":
    main()
