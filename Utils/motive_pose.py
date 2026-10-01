"""Interpolate Motive rigid-body poses for an eye point at trial onset."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


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
