"""Helpers for Motive rigid-body poses."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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
