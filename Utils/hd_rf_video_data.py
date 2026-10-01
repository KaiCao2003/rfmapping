"""Read generated JSON headings on the verified Basler video frame clock."""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.ebc_analysis import BASLER_SIZE_CM, boundary_new, boundary_old
from Utils.ebc_camera import load_motive_head
from Utils.json_tools import read_formatted_json
from Utils.tuning_curve_utils import _read_exposure_timestamps, get_exposure_timestamps


def _probe_video(path):
    # Some Basler AVI headers count an empty initial timestamp as a frame.
    # Read the decoded count once in the parent process before worker launch.
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-threads", "4", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return dict(width=int(stream["width"]), height=int(stream["height"]),
                fps=float(Fraction(stream["avg_frame_rate"])),
                declared_frame_count=int(stream["nb_frames"]),
                decoded_frame_count=int(stream["nb_read_frames"]))


def load_scheme_video_data(session, *, probe="A", phase="baseline", wall_config="old"):
    """Load CW JSON HD and front-head positions indexed by decoded AVI ordinal.

    The saved TC contributes only its camera-clock metadata. Its HD curves,
    classifications and historical angle convention are not used here.
    JSON frame origin comes from explicit metadata, never a guessed join.
    """
    session = Path(session)
    directory = session / "data"
    tc_directory = directory / "tuning_curves" / f"Probe{probe}"
    tc_path = tc_directory / "tuning_curves.tc"
    if not tc_path.is_file():
        tc_path = tc_directory / "tuning_curves.json"
    metadata = read_formatted_json(tc_path)["metadata"]
    timing_source = metadata["ttl_qc"]
    policy = metadata.get("frame_mapping") or timing_source.get("frame_timestamp_mapping")
    if policy == "pose_frame_id_minus_one_to_exposure_midpoint":
        json_offset = -1
    elif policy == "zero_based_csv_frame":
        json_offset = 0
    else:
        raise ValueError(f"Explicit JSON-to-video frame mapping is required; found {policy!r} in {tc_path}")

    hd_path = directory / "processed/head_direction.json"
    heading = read_formatted_json(hd_path)[metadata.get("headplate", "hp4")]
    hd_key = "hd" if "hd" in heading else "head_direction_deg"
    json_frames = np.asarray(heading["frames"], dtype=float)
    if not np.isfinite(json_frames).all() or np.any(json_frames != np.floor(json_frames)):
        raise ValueError("Generated JSON frame IDs must be finite integers.")
    frames = json_frames.astype(int) + json_offset
    if np.any(frames < 0) or np.any(np.diff(frames) <= 0):
        raise ValueError("Generated JSON must map to unique increasing zero-based video frame IDs.")
    hd = pd.Series(np.asarray(heading[hd_key], dtype=float) % 360., index=frames, name="hd")

    position_source = session / f"{session.name.split('_')[0]}.csv"
    if not position_source.is_file():
        position_source = directory / "dark_pose.csv"
    # The CSV heading column is deliberately not read: generated JSON is the
    # only behavioral HD source. Back points serve only the frame/sign audit.
    pose = pd.read_csv(position_source, usecols=["frame", "front_x", "front_y", "back_x", "back_y"])
    position_frame_count = len(pose)
    if pose.frame.duplicated().any() or np.any(np.diff(pose.frame) <= 0):
        raise ValueError("Pose CSV must have unique increasing zero-based frame IDs.")
    pose = pose.set_index("frame").reindex(frames)
    pose["hd"] = hd
    vector = pose[["front_x", "front_y"]].to_numpy() - pose[["back_x", "back_y"]].to_numpy()
    geometric_hd = np.rad2deg(np.arctan2(vector[:, 0], -vector[:, 1])) % 360.
    error = abs((pose.hd.to_numpy() - geometric_hd + 180.) % 360. - 180.)
    audited = np.isfinite(error) & np.any(vector != 0, axis=1)
    if not audited.any() or np.max(error[audited]) > .01:
        raise ValueError("Generated JSON HD does not match CW front/back geometry at the documented frame offset.")

    video_path = session / f"{session.name.split('_')[0]}.avi"
    video_probe = _probe_video(video_path)
    if (video_probe["width"], video_probe["height"]) != (1280, 1024):
        raise ValueError("Use the original uncropped 1280 × 1024 Basler AVI.")
    total = video_probe["decoded_frame_count"]
    exposures, origin, camera_timing = get_exposure_timestamps(
        read_formatted_json(directory / "session_info.json")["session_info"], directory,
        camera_input_channel=timing_source["camera_input_channel"],
        camera_ttl_threshold=timing_source["camera_ttl_threshold"],
        camera_ttl_active_high=timing_source["camera_ttl_active_high"],
    )
    synchronized = (frames < len(exposures)) & (frames < total)
    dropped = int((~synchronized).sum())
    pose, frames = pose.loc[synchronized], frames[synchronized]
    times = exposures[frames]
    intervals = pd.read_csv(directory / "interval_table.csv")
    interval = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0].to_numpy(float)
    available = np.isfinite(pose[["front_x", "front_y", "hd"]]).all(axis=1).to_numpy()
    selected = available & (times >= interval[0]) & (times <= interval[1])
    pose = pose.loc[selected]
    bounds = {"old": boundary_old, "new": boundary_new}[wall_config]
    left, right, top, bottom = bounds
    xy = np.c_[(pose.front_x - left) * BASLER_SIZE_CM / (right - left),
               (bottom - pose.front_y) * BASLER_SIZE_CM / (bottom - top)]
    frame_mapping = dict(
        policy=policy, json_frame_offset_to_video=json_offset, json_frame_count=len(json_frames),
        position_frame_count=position_frame_count,
        geometry_audit_frames=int(audited.sum()), geometry_max_error_deg=float(np.max(error[audited])),
        ttl_count=len(exposures), decoded_video_frame_count=total,
        trailing_ttl_without_video=max(0, len(exposures) - total),
        video_frames_without_ttl=max(0, total - len(exposures)),
        dropped_unsynchronized_json_frames=dropped, invalid_position_or_heading_rows=int((~available).sum()),
        selected_interval_s=interval.tolist(), selected_tracking_frames=int(selected.sum()),
        mapping="JSON frame + offset = pose CSV frame = decoded AVI ordinal = exposure array index",
    )
    return dict(
        session=session, probe=probe, phase=phase, xy=xy,
        hd=pose.hd.to_numpy(), frame_ids=frames[selected], times=times[selected],
        exposure_times=exposures[:total], adc_time_origin_s=origin, camera_timing=camera_timing,
        bounds_px=bounds, arena_size_cm=BASLER_SIZE_CM, hd_path=str(hd_path),
        hd_sha256=hashlib.sha256(hd_path.read_bytes()).hexdigest(), position_source=str(position_source),
        position_columns=["front_x", "front_y"], frame_mapping=frame_mapping,
        video_path=video_path, video_frame_count=total, video_probe=video_probe,
        dropped_unsynchronized_pose_frames=dropped,
    )


def build_video_clock(motive_times, basler_times, frame_count, *, stride=2, phase=0,
                      segments=None):
    """Pair measured pulses and apply an explicit AVI-to-Motive row mapping.

    Regular recordings require exactly one video frame per configured stride.
    Dropped-frame recordings supply inclusive (first, last, offset) segments;
    uncovered frames and ambiguous pulse pairs retain NaN in both clocks.
    """
    motive_times = np.asarray(motive_times, dtype=float)
    basler_times = np.asarray(basler_times, dtype=float)
    for times in (motive_times, basler_times):
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or not np.all(np.diff(times) > 0)):
            raise ValueError("Camera pulse timestamps must be finite and strictly increasing.")
    if (int(frame_count) != frame_count or frame_count < 1
            or int(stride) != stride or stride < 1 or int(phase) != phase or phase < 0):
        raise ValueError("Frame count and stride must be positive integers; phase must be nonnegative.")
    frame_count, stride, phase = int(frame_count), int(stride), int(phase)
    motive_rows = np.full(frame_count, np.nan)
    if segments is None:
        rows = np.arange(phase, len(motive_times), stride)
        if len(rows) != frame_count:
            raise ValueError(
                f"The explicit stride/phase gives {len(rows)} video frames, but the AVI has "
                f"{frame_count}; provide mapping segments for a dropped-frame recording."
            )
        motive_rows[:] = rows
        mapping_segments = [(0, frame_count - 1, phase)]
    else:
        mapping_segments = [tuple(segment) for segment in segments]
        for first, last, offset in mapping_segments:
            if (any(int(value) != value for value in (first, last, offset))
                    or not 0 <= first <= last < frame_count):
                raise ValueError("Mapping segments require integer, in-bounds inclusive video ranges.")
            first, last, offset = int(first), int(last), int(offset)
            if np.isfinite(motive_rows[first:last + 1]).any():
                raise ValueError("Video mapping segments must not overlap.")
            motive_rows[first:last + 1] = stride * np.arange(first, last + 1) + offset
    mapped = np.isfinite(motive_rows)
    rows = motive_rows[mapped].astype(int)
    if np.any((rows < 0) | (rows >= len(motive_times))):
        raise ValueError("Mapped Motive rows must lie within the measured Motive clock.")

    right = np.clip(np.searchsorted(basler_times, motive_times), 0, len(basler_times) - 1)
    left = np.maximum(right - 1, 0)
    nearest = np.where(abs(basler_times[left] - motive_times)
                       <= abs(basler_times[right] - motive_times), left, right)
    errors = abs(basler_times[nearest] - motive_times)
    tolerance = float(np.median(np.diff(motive_times)) / 4)
    paired = errors <= tolerance
    multiplicity = np.bincount(nearest[paired], minlength=len(basler_times))
    ambiguous = paired & (multiplicity[nearest] != 1)
    # Neither competing Motive row owns a reused pulse; do not choose a winner.
    paired &= ~ambiguous
    exposures = np.full(frame_count, np.nan)
    frame_ids = np.flatnonzero(mapped)
    valid_rows = paired[rows]
    exposures[frame_ids[valid_rows]] = basler_times[nearest[rows[valid_rows]]]
    synchronized = np.isfinite(exposures)
    motive_rows[~synchronized] = np.nan
    if not np.all(np.diff(exposures[synchronized]) > 0):
        raise ValueError("The explicit mapping must produce increasing video exposure times.")
    consecutive = synchronized[:-1] & synchronized[1:]
    if not consecutive.any():
        raise ValueError("The video clock needs two adjacent synchronized frames to measure its period.")
    frame_period = float(np.median(np.diff(exposures)[consecutive]))
    missing = np.flatnonzero(~synchronized)
    gaps = ([] if not len(missing) else
            [[int(part[0]), int(part[-1])] for part in
             np.split(missing, np.flatnonzero(np.diff(missing) != 1) + 1)])
    qc = dict(
        decoded_video_frame_count=frame_count, motive_frame_count=len(motive_times),
        basler_raw_pulse_count=len(basler_times), paired_motive_pulse_count=int(paired.sum()),
        unmatched_motive_pulse_count=int((~paired).sum()),
        basler_matched_pulse_count=int(paired.sum()),
        unpaired_basler_pulse_count=int(len(basler_times) - paired.sum()),
        ambiguous_motive_pair_count=int(ambiguous.sum()), pairing_tolerance_s=tolerance,
        pairing_max_error_s=float(errors[paired].max()) if paired.any() else None,
        finite_video_frame_count=int(synchronized.sum()), uncertain_video_frame_count=len(missing),
        uncertain_video_frame_ranges_inclusive=gaps, frame_period_s=frame_period,
        stride=stride, phase=phase,
        mapping_segments=[dict(first_video_frame=int(first), last_video_frame=int(last),
                              motive_row_slope=stride, motive_row_offset=int(offset))
                          for first, last, offset in mapping_segments],
        mapping_method="explicit_segments" if segments is not None else "explicit_regular_stride_phase",
    )
    return exposures, motive_rows, qc


def load_population_video_data(session, *, probe, phase, motive_clock,
                               basler_clock, video_stride, video_phase, video_segments=None,
                               camera_registration_path, cylinder_calibration_path):
    """Join saved HD and Motive XYZ to raw timing, using shared camera geometry."""
    session = Path(session)
    directory = session / "data"
    date = session.name.split("_")[0]
    video_path = session / f"{date}.avi"
    video_probe = _probe_video(video_path)
    total = video_probe["decoded_frame_count"]
    if (video_probe["width"], video_probe["height"]) != (1280, 1024):
        raise ValueError("Use the original uncropped 1280 × 1024 Basler AVI.")
    info = read_formatted_json(directory / "session_info.json")["session_info"]
    motive_times, origin, motive_source = _read_exposure_timestamps(info, **motive_clock)
    basler_times, _, basler_source = _read_exposure_timestamps(info, **basler_clock)
    exposures, motive_rows, qc = build_video_clock(
        motive_times, basler_times, total, stride=video_stride, phase=video_phase,
        segments=video_segments,
    )
    qc.update(session=str(session), adc_time_origin_s=origin,
              motive_source=str(motive_source), basler_source=str(basler_source),
              motive_clock=motive_clock, basler_clock=basler_clock,
              timestamp_reference="raw ADC exposure midpoints, seconds relative to ADC origin")
    frames = np.arange(total)
    synchronized = np.isfinite(exposures)
    frame_period_s = qc["frame_period_s"]
    hd_path = directory / "processed/head_direction.json"
    heading = read_formatted_json(hd_path)["hp4"]
    hd = pd.Series(heading["head_direction_deg"], index=heading["frames"]).reindex(motive_rows).to_numpy(float)
    position_source = directory / "processed/trimmed_input.csv"
    xyz = load_motive_head(position_source).reindex(motive_rows)[["X", "Y", "Z"]].to_numpy(float)
    available = np.isfinite(hd) & np.isfinite(xyz).all(axis=1)
    camera_registration_path = Path(camera_registration_path)
    cylinder_calibration_path = Path(cylinder_calibration_path)
    registration = json.loads(camera_registration_path.read_text())
    calibration = json.loads(cylinder_calibration_path.read_text())
    projection = np.asarray(registration["world_to_video_projection_raw"], dtype=float)
    cylinder_center = np.asarray(calibration["screen_bottom_center"][:2], dtype=float)
    cylinder_radius = float(calibration["screen_diamter"] / 2)
    raw_units_per_cm = float(10 * calibration["motive_units_per_real_mm"])
    intervals = pd.read_csv(directory / "interval_table.csv")
    interval = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0].to_numpy(float)
    selected = synchronized & available & (exposures >= interval[0]) & (exposures <= interval[1])
    if not selected.any():
        raise ValueError("The selected phase contains no synchronized HD frames.")
    selected_frames = frames[selected]
    selected_times = exposures[selected]
    # Each continuous block supplies support only through half a measured
    # frame period at either end; missing video-clock nodes remain gaps.
    breaks = np.flatnonzero((np.diff(selected_frames) != 1)
                           | (np.diff(selected_times) > 1.5 * frame_period_s)) + 1
    starts, ends = np.r_[0, breaks], np.r_[breaks, len(selected_frames)] - 1
    decoding_intervals = np.c_[
        np.maximum(selected_times[starts] - frame_period_s / 2, interval[0]),
        np.minimum(selected_times[ends] + frame_period_s / 2, interval[1]),
    ]
    dropped = int((~synchronized).sum())
    frame_mapping = dict(
        policy="explicit_video_to_motive_exposure_mapping",
        decoded_video_frame_count=total, synchronized_video_frames=int(synchronized.sum()),
        dropped_unsynchronized_pose_frames=dropped,
        invalid_position_or_heading_rows=int((~available).sum()),
        selected_interval_s=interval.tolist(), selected_tracking_frames=int(selected.sum()),
        mapping="decoded AVI ordinal -> configured Motive exposure ordinal -> raw Basler midpoint",
        hd_mapping="configured Motive exposure ordinal = saved HD JSON frame",
        hd_convention="Saved Motive head_direction_deg, unchanged",
    )
    return dict(
        session=session, probe=probe, phase=phase, arena_type="cylinder", xyz=xyz[selected],
        projection=projection, cylinder_center_raw=cylinder_center, cylinder_radius_raw=cylinder_radius,
        raw_units_per_cm=raw_units_per_cm,
        cylinder_bottom_z_raw=float(calibration["screen_bottom_center"][2]),
        cylinder_top_z_raw=float(calibration["screen_top_center"][2]),
        camera_registration_path=str(camera_registration_path),
        camera_registration_sha256=hashlib.sha256(camera_registration_path.read_bytes()).hexdigest(),
        cylinder_calibration_path=str(cylinder_calibration_path),
        cylinder_calibration_sha256=hashlib.sha256(cylinder_calibration_path.read_bytes()).hexdigest(),
        hd=hd[selected], frame_ids=selected_frames, times=selected_times,
        exposure_times=exposures, adc_time_origin_s=origin, camera_timing=qc,
        frame_period_s=frame_period_s, decoding_intervals_s=decoding_intervals, video_clock=qc,
        hd_path=str(hd_path),
        hd_source="Motive processed/head_direction.json",
        hd_sha256=hashlib.sha256(hd_path.read_bytes()).hexdigest(), position_source=str(position_source),
        position_columns=["hp4/Position/X", "hp4/Position/Y", "hp4/Position/Z"], frame_mapping=frame_mapping,
        video_path=video_path, video_frame_count=total, video_probe=video_probe,
        dropped_unsynchronized_pose_frames=dropped,
    )
