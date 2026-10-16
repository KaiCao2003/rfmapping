"""Helpers for Motive rigid-body poses."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def matched_motive_frame_count(frames, exposure_timestamps, timing, *, ignore_hd_ttl=False) -> int:
    """Match a continuous CSV prefix to original TTL times without resampling.

    Opt-in mismatches retain the shorter prefix. Otherwise only the established
    single extra trailing Motive frame is accepted. Original TTL QC is preserved.
    """
    frame_count, pulse_count = len(frames), len(exposure_timestamps)
    if not np.array_equal(frames, np.arange(frame_count)):
        raise ValueError("Motive frame IDs must be zero-based and continuous.")
    if frame_count == pulse_count:
        count, alignment = frame_count, "exact"
    elif ignore_hd_ttl:
        count, alignment = min(frame_count, pulse_count), "front_aligned_truncate_to_shorter"
        print(f"Ignoring Motive frame/TTL count mismatch: {frame_count} frames vs "
              f"{pulse_count} exposure pulses; retaining the first {count} pairs "
              "with original TTL timestamps.")
    elif frame_count == pulse_count + 1:
        count, alignment = pulse_count, "drop_single_trailing_motive_frame"
        print(f"Dropping trailing Motive frame: {frames[-1]}")
    else:
        raise ValueError(
            f"Motive frame/TTL count mismatch: {frame_count} frames vs "
            f"{pulse_count} exposure pulses."
        )
    timing.update({
        "motive_frame_count_original": int(frame_count),
        "motive_frame_count": int(count),
        "motive_frame_alignment": alignment,
        "frame_timestamp_mapping": "zero_based_motive_frame_id_to_original_ttl_timestamp",
        "ignore_hd_ttl": bool(ignore_hd_ttl),
    })
    return count


def motive_z_yaw_degrees(rotation: np.ndarray) -> np.ndarray:
    """Return Motive intrinsic-XYZ Z yaw, CCW from world +X.

    Input rows are Euler XYZ degrees or Quaternion XYZW. Missing quaternion
    poses retain NaN; this planar heading is independent of fused HD exports.
    """
    if rotation.shape[1] == 3:
        return rotation[:, 2] % 360.
    valid = np.isfinite(rotation).all(axis=1) & (np.linalg.norm(rotation, axis=1) > 0)
    heading = np.full(len(rotation), np.nan)
    if valid.any():
        heading[valid] = Rotation.from_quat(rotation[valid]).as_euler("XYZ", degrees=True)[:, 2] % 360.
    return heading


def eye_positions_at_times(
    frame_times: np.ndarray,
    position_xyz: np.ndarray,
    rotation_xyz_deg: np.ndarray,
    eye_offset_xyz: np.ndarray,
    query_times: np.ndarray,
) -> np.ndarray:
    """Return world XYZ using Motive's intrinsic XYZ (Rx @ Ry @ Rz) poses.

    Positions and the rigid-local eye offset must use the same length units.
    Times share one clock, poses are finite, and queries lie within frame_times.
    The body rotation is independent of the RF head-direction zero offset.
    """
    rotations = Rotation.from_euler("XYZ", rotation_xyz_deg, degrees=True)
    query_rotations = Slerp(frame_times, rotations)(query_times)
    query_positions = np.column_stack([
        np.interp(query_times, frame_times, position_xyz[:, axis])
        for axis in range(3)
    ])
    return query_positions + query_rotations.apply(eye_offset_xyz)
