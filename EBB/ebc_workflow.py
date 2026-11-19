"""Saved-analysis Basler overlays and historical camera replay adapters."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from EBB.ebc_geometry import (EBCInfo, PositionInfo, compute_ebc_rays,
                              cylinder_boundary, rectangle_boundary)
from EBB.ebc_overlay import prepare_overlay_info
from EBB.ebc_pose import load_motive_position
from EBB.ebc_video import export_ebc_overlay, export_good_unit_videos


def read_video_config(path):
    """Read settings; file paths are relative to this config, never the shell cwd."""
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ("session", "video_path", "pose_path", "kilosort_dir", "output_dir",
                "calibration_path", "registration_path", "clock_dir", "hd_reference_path",
                "stimulus_mat_path", "stimulus_onsets_path", "analysis_dir"):
        if key in config:
            config[key] = (path.parent / Path(config[key]).expanduser()).resolve()
    config["config_path"] = str(path)
    return config


def select_position_interval(position, interval_s):
    """Mask geometry outside the chosen epoch without dropping video clock rows."""
    start, stop = interval_s
    selected = (position.times_s >= start) & (position.times_s <= stop)
    return replace(position, valid=position.valid & selected)


def load_video_clock(directory, session):
    """Read an audited full-video clock, preserving every declared unknown frame."""
    directory = Path(directory)
    qc = json.loads((directory / "video_clock_qc.json").read_text())
    if Path(qc["session"]).resolve() != Path(session).resolve():
        raise ValueError("The audited video clock belongs to a different session.")
    times = np.load(directory / "video_adc_times.npy")
    rows = np.load(directory / "video_motive_rows.npy")
    valid = np.ones(len(times), dtype=bool)
    for first, last in qc.get("uncertain_video_frame_ranges_inclusive", []):
        valid[first:last + 1] = False
    if (len(times) != qc["decoded_video_frame_count"] or rows.shape != times.shape
            or not np.array_equal(valid, np.isfinite(times))
            or not np.array_equal(valid, np.isfinite(rows))
            or np.any(np.diff(times[valid]) <= 0)):
        raise ValueError("The measured clock/mapping must cover every decoded frame or an audited gap.")
    return times, rows, float(qc["adc_time_origin_s"]), qc


def load_stimulus_trials(mat_path, onset_path, adc_origin_s):
    """Read actual stimulus trials and their N+1 ADC-relative time boundaries."""
    trials = np.atleast_1d(loadmat(mat_path, simplify_cells=True)["trials"])
    edges = np.load(onset_path) - adc_origin_s
    if len(edges) != len(trials) + 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("MAT trials must match increasing N+1 stimulus boundaries.")
    return dict(edges=edges,
                screen_deg=np.array([trial["Square_PositionX"] for trial in trials]),
                luminance=np.array([trial["Square_Luminance"] for trial in trials]))


def calculate_stimulus_targets(position, trials, *, center_xy_cm, radius_cm, vs_zero_deg, background_luminance):
    """Convert visible cylinder stimuli to world-cm targets before rendering."""
    index = np.searchsorted(trials["edges"], position.times_s, side="right") - 1
    valid = position.valid & (index >= 0) & (index < len(trials["screen_deg"]))
    selected = np.clip(index, 0, len(trials["screen_deg"]) - 1)
    valid &= trials["luminance"][selected] != background_luminance
    angles = np.deg2rad(vs_zero_deg - trials["screen_deg"][selected])
    target = position.xyz_cm.copy()
    target[:, :2] = center_xy_cm + radius_cm * np.c_[np.cos(angles), np.sin(angles)]
    target[~valid] = np.nan
    return target


def prepare_video_data(config, position, boundary, exposure_times, adc_origin_s, timing, *, targets=None):
    """Compose the shared workflow, with calculation and display projection explicit."""
    if position.raw_xyz_cm is None:
        position = replace(position, raw_xyz_cm=position.xyz_cm.copy(),
                           position_held=np.zeros(len(position.frame_ids), dtype=bool),
                           initial_outside=np.zeros(len(position.frame_ids), dtype=bool))
    if "interval_s" in config:
        position = select_position_interval(position, config["interval_s"])
    ebc = compute_ebc_rays(position, boundary)
    overlay = prepare_overlay_info(position, boundary, ebc, target_xyz_cm=targets)
    return dict(
        session=config["session"], probe=config["probe"], phase=config.get("phase", "recording"),
        frame_ids=position.frame_ids, times=position.times_s, xy=position.xyz_cm[:, :2], hd=position.hd_deg,
        raw_xyz_cm=position.raw_xyz_cm, position_held=position.position_held,
        initial_outside=position.initial_outside,
        exposure_times=exposure_times, adc_time_origin_s=adc_origin_s, camera_timing=timing,
        overlay_info=overlay, ebc_info=ebc, geometry_metadata=dict(boundary.metadata,
            angle_convention="HD CCW from world +X; EBC bearing clockwise from HD (0 front, 90 right); distances in cm"),
        source=dict(kilosort_dir=str(config["kilosort_dir"]),
                    selected_interval_s=config.get("interval_s"),
                    spike_times_path=str(config["session"] / "data" / f"probe{config['probe']}" / "adc_spike_time.npy")),
        provenance=dict(config_path=config.get("config_path"),
                        config={key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
                        position=position.source, timing=timing),
    )


def load_saved_ebc_frames(directory):
    """Read the analyzed frame values and provenance; never generate missing data."""
    directory = Path(directory)
    audit = json.loads((directory / "audit.json").read_text())
    with np.load(directory / "position_correction.npz", allow_pickle=False) as saved:
        frames = {name: saved[name] for name in (
            "frame_ids", "frame_times", "raw_xy_px", "corrected_xy_px",
            "position_held", "initial_outside", "hd_deg", "ebc_hd_deg",
            "theta_deg", "theta_d_cm", "exposure_times",
        )}
    return frames, audit


def select_saved_ebc_rays(frames, audit):
    """Select eight native samples; reconstruct endpoints only for projection."""
    left, right, top, bottom = audit["bounds_px"]
    width, height = audit["arena_size_cm"]
    scale = np.array([width / (right - left), height / (bottom - top)])
    xyz = np.zeros((len(frames["frame_ids"]), 3))
    xyz[:, :2] = (frames["corrected_xy_px"] - [left, bottom]) * scale * [1, -1]
    raw_xyz = xyz.copy()
    raw_xyz[:, :2] = (frames["raw_xy_px"] - [left, bottom]) * scale * [1, -1]
    heading = (90. - frames["ebc_hd_deg"]) % 360.
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(heading)
    position = PositionInfo(frames["frame_ids"], frames["frame_times"], xyz, heading,
                            valid, audit["source"], raw_xyz, frames["position_held"],
                            frames["initial_outside"])
    # Native 6-degree bin centers have no exact 0/90-degree sample.
    indices = np.arange(8) * len(frames["theta_deg"]) // 8
    bearings = (frames["theta_deg"][indices] + 180.) % 360. - 180.
    distances = frames["theta_d_cm"][:, indices]
    radians = np.deg2rad(heading[:, None] - bearings)
    directions = np.stack((np.cos(radians), np.sin(radians), np.zeros_like(radians)), axis=-1)
    endpoints = xyz[:, None, :] + distances[..., None] * directions
    ebc = EBCInfo(bearings, xyz, endpoints, distances,
                  valid & np.isfinite(distances).all(axis=1))
    boundary = rectangle_boundary(audit["bounds_px"], width)
    return position, boundary, ebc, indices


def prepare_rectangle_video(config):
    """Replay eight samples from saved analysis; no tracking or ray recalculation."""
    frames, audit = load_saved_ebc_frames(config["analysis_dir"])
    position, boundary, ebc, indices = select_saved_ebc_rays(frames, audit)
    source = audit["source"]
    return dict(
        session=config["session"], probe=audit["probe_name"], phase=audit["phase_key"],
        arena_type="rectangle", frame_ids=position.frame_ids, source_frame_ids=position.frame_ids,
        times=position.times_s, xy=position.xyz_cm[:, :2], hd=position.hd_deg,
        raw_xyz_cm=position.raw_xyz_cm, position_held=position.position_held,
        initial_outside=position.initial_outside, exposure_times=frames["exposure_times"],
        adc_time_origin_s=source["adc_time_origin_s"], camera_timing=source["camera_timing"],
        overlay_info=prepare_overlay_info(position, boundary, ebc), ebc_info=ebc,
        geometry_metadata=dict(boundary.metadata,
            angle_convention="EBC clockwise from forward; native saved bin-center samples",
            display_heading_convention="HD north zero, positive CW; EBC forward zero, positive CW",
            selected_angle_indices=indices.tolist(), geometry_source="saved_analysis"),
        source=source,
        provenance=dict(config_path=config.get("config_path"),
                        analysis_dir=str(config["analysis_dir"]),
                        geometry_file=str(config["analysis_dir"] / "position_correction.npz"),
                        selected_angle_indices=indices.tolist(), analysis=audit),
    )


def prepare_cylinder_video(config):
    """Load the cylinder recording, then calculate the same overlay payload."""
    calibration = json.loads(config["calibration_path"].read_text())
    registration = json.loads(config["registration_path"].read_text())
    boundary = cylinder_boundary(calibration, registration)
    times, rows, origin, timing = load_video_clock(config["clock_dir"], config["session"])
    reference = json.loads(config["hd_reference_path"].read_text())
    position = load_motive_position(
        config["pose_path"], times, rows, units_per_cm=boundary.metadata["raw_units_per_cm"],
        reference_quaternion_xyzw=reference["reference_quaternion_xyzw"],
        hd_world_zero_deg=config["hd_world_zero_deg"], headplate=config["headplate"],
        center_xy_cm=np.asarray(boundary.metadata["center_xyz_cm"][:2]),
        radius_cm=boundary.metadata["radius_cm"],
    )
    targets = None
    if "stimulus_mat_path" in config:
        trials = load_stimulus_trials(config["stimulus_mat_path"], config["stimulus_onsets_path"], origin)
        targets = calculate_stimulus_targets(
            position, trials, center_xy_cm=np.asarray(boundary.metadata["center_xyz_cm"][:2]),
            radius_cm=boundary.metadata["radius_cm"], vs_zero_deg=config["vs_zero_deg"],
            background_luminance=config["background_luminance"],
        )
    data = prepare_video_data(config, position, boundary, times, origin, timing, targets=targets)
    data["arena_type"] = "cylinder"
    data["video_frame_count"] = timing["decoded_video_frame_count"]
    data["source_frame_ids"] = rows
    return data


def save_geometry_audit(data, result, path):
    """Save calculated frame values separately from frame drawing and encoding."""
    info, ebc = data["overlay_info"], data["ebc_info"]
    lookup = dict(zip(data["frame_ids"], range(len(data["frame_ids"]))))
    columns = ["video_frame", "source_frame", "adc_time_s", "sync_valid", "pose_valid", "geometry_valid",
               "x_cm", "y_cm", "z_cm", "hd_world_deg", "vs_bearing_deg",
               "raw_x_cm", "raw_y_cm", "raw_z_cm", "position_held", "initial_outside"]
    for bearing in info.bearings_deg:
        columns.extend(f"ray_{bearing:g}_{field}" for field in ("distance_cm", "x_cm", "y_cm", "z_cm"))
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for frame in range(result["source_first_frame"], result["source_last_frame"] + 1):
            row = lookup.get(frame)
            values = [frame] + [None] * (len(columns) - 1)
            if row is not None:
                bearing = None if info.target_bearing_deg is None else info.target_bearing_deg[row]
                values = [frame, data["source_frame_ids"][row], info.times_s[row], info.synchronized[row],
                          info.pose_valid[row], info.valid[row], *info.xyz_cm[row], info.hd_deg[row], bearing,
                          *data["raw_xyz_cm"][row], data["position_held"][row], data["initial_outside"][row]]
                for distance, endpoint in zip(ebc.distances_cm[row], ebc.endpoints_cm[row]):
                    values.extend((distance, *endpoint))
            writer.writerow(["" if value is None or not np.isfinite(value) else value for value in values])


def measured_video_rate(times):
    """Estimate frame duration only across adjacent, synchronized exposures."""
    times = np.asarray(times)
    adjacent = np.isfinite(times[:-1]) & np.isfinite(times[1:])
    periods = np.diff(times)[adjacent]
    if not len(periods) or np.any(periods <= 0):
        raise ValueError("At least two adjacent increasing camera times are required.")
    return float(1. / np.median(periods))


def export_prepared_video(data, config, *, start_s=0., duration_s=None, silent=False):
    """Export either source using the identical renderer and audio pipeline."""
    # Basler playback retains the AVI rate; ADC quantization is not a new FPS.
    fps = None if data.get("arena_type") == "rectangle" else measured_video_rate(data["exposure_times"])
    first = None if fps is None else int(round(start_s * fps))
    stop = data.get("video_frame_count")
    if stop is not None and duration_s is not None and fps is not None:
        stop = min(stop, first + int(round(duration_s * fps)))
    settings = dict(start_s=start_s, duration_s=duration_s, frame_rate=fps,
                    first_frame=first, stop_frame=stop,
                    workers=config.get("video_workers", 1), video_encoder=config.get("video_encoder", "auto"))
    if silent:
        result = export_ebc_overlay(data, config["video_path"], config["output_dir"] / "overlay.mp4", **settings)
    else:
        result = export_good_unit_videos(
            data, config["video_path"], config["output_dir"], **settings,
            audio_workers=config.get("audio_workers", 4), **config.get("audio", {}),
        )
    geometry_path = config["output_dir"] / "frame_geometry.csv"
    save_geometry_audit(data, result, geometry_path)
    result["frame_geometry"] = str(geometry_path)
    summary_path = config["output_dir"] / ("overlay.json" if silent else "manifest.json")
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    if not silent:
        overlay_path = config["output_dir"] / "overlay.json"
        overlay = json.loads(overlay_path.read_text())
        overlay["frame_geometry"] = str(geometry_path)
        overlay_path.write_text(json.dumps(overlay, indent=2) + "\n")
    return result


def video_main(prepare, argv=None):
    """Shared CLI orchestration; each entry selects only its source adapter."""
    parser = argparse.ArgumentParser(description="Render eight EBC rays using an explicit recording config.")
    parser.add_argument("config", type=Path, help="JSON config; relative paths resolve beside this file")
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--duration", type=float, help="Seconds; omitted renders the full measured video clock")
    parser.add_argument("--silent", action="store_true", help="Export the shared overlay without per-unit audio")
    args = parser.parse_args(argv)
    config = read_video_config(args.config)
    data = prepare(config)
    return export_prepared_video(data, config, start_s=args.start, duration_s=args.duration, silent=args.silent)
