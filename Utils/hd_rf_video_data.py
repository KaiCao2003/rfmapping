"""Read generated JSON headings on the verified Basler video frame clock."""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.ebc_analysis import BASLER_SIZE_CM, boundary_new, boundary_old
from Utils.json_tools import read_formatted_json
from Utils.tuning_curve_utils import get_exposure_timestamps


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
