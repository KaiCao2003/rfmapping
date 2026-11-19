"""Read synchronized Basler or Motive poses into the same physical coordinates."""

from pathlib import Path

import numpy as np
import pandas as pd

from EBB.ebc_geometry import PositionInfo, pixel_heading_to_world_deg


POSITION_CORRECTION_POLICY = "hold_last_in_bounds; inclusive boundary; leading outside missing; missing unchanged"


def hold_last_in_bounds(points, inside):
    """Hold each finite outside point at the last complete in-bound position.

    Missing rows neither receive a held position nor erase the last legal point.
    Process the full temporal sequence before selecting a phase or video range.
    """
    points = np.asarray(points, dtype=float)
    finite = np.isfinite(points).all(axis=1)
    legal = np.asarray(inside, dtype=bool) & finite
    last = np.maximum.accumulate(np.where(legal, np.arange(len(points)), -1))
    outside = finite & ~legal
    held = outside & (last >= 0)
    initial_outside = outside & (last < 0)
    corrected = points.copy()
    corrected[held] = points[last[held]]
    corrected[initial_outside] = np.nan
    return corrected, held, initial_outside


def hold_rectangle_positions(xy_px, bounds_px):
    """Apply whole-point holding within inclusive left/right/top/bottom bounds."""
    xy_px = np.asarray(xy_px, dtype=float)
    left, right, top, bottom = bounds_px
    inside = ((xy_px[:, 0] >= left) & (xy_px[:, 0] <= right)
              & (xy_px[:, 1] >= top) & (xy_px[:, 1] <= bottom))
    return hold_last_in_bounds(xy_px, inside)


def load_basler_position(pose_path, exposure_times, *, bounds_px, size_cm,
                         position_columns=("center_x", "center_y"), heading_column="hd_deg",
                         heading_clockwise=False, heading_zero_deg=90., frame_offset=0):
    """Read explicit CSV frame IDs and convert calibrated pixels into world cm.

    Exposure times must already be seconds relative to the ADC origin. Missing
    CSV rows retain their known clock time with invalid pose; CSV frames without
    a finite exposure remain present but invalid.
    """
    pose_path = Path(pose_path)
    pose = pd.read_csv(pose_path, usecols=["frame", *position_columns, heading_column])
    frames = pose["frame"].to_numpy(dtype=float) + frame_offset
    if not np.isfinite(frames).all() or np.any(frames != np.floor(frames)) or np.any(frames < 0):
        raise ValueError("Basler frame IDs must map to nonnegative integer exposure indices.")
    frames = frames.astype(int)
    if len(np.unique(frames)) != len(frames):
        raise ValueError("Basler frame IDs must be unique.")
    exposures = np.asarray(exposure_times, dtype=float)
    pose.index = frames
    frames = np.arange(max(len(exposures), int(frames.max(initial=-1)) + 1))
    pose = pose.reindex(frames)
    synchronized = frames < len(exposures)
    times = np.full(len(frames), np.nan)
    times[synchronized] = exposures[frames[synchronized]]
    left, right, top, bottom = np.asarray(bounds_px, dtype=float)
    if not np.isfinite([left, right, top, bottom, size_cm]).all() or not (
        right > left and bottom > top and size_cm > 0
    ):
        raise ValueError("Basler calibration requires increasing pixel bounds and a positive size in cm.")
    raw_pixels = pose[list(position_columns)].to_numpy(dtype=float)
    pixels, held, initial_outside = hold_rectangle_positions(raw_pixels, bounds_px)
    raw_xyz = np.column_stack(((raw_pixels[:, 0] - left) * size_cm / (right - left),
                              (bottom - raw_pixels[:, 1]) * size_cm / (bottom - top),
                              np.where(pose["frame"].notna(), 0., np.nan)))
    xyz = np.column_stack(((pixels[:, 0] - left) * size_cm / (right - left),
                           (bottom - pixels[:, 1]) * size_cm / (bottom - top),
                           np.where(pose["frame"].notna(), 0., np.nan)))
    xyz[initial_outside] = np.nan
    hd = pixel_heading_to_world_deg(
        pose[heading_column].to_numpy(dtype=float), bounds_px=bounds_px, size_cm=size_cm,
        heading_clockwise=heading_clockwise, heading_zero_deg=heading_zero_deg,
    )
    valid = np.isfinite(times) & np.isfinite(xyz).all(axis=1) & np.isfinite(hd)
    xyz[~np.isfinite(times)] = np.nan
    hd[~np.isfinite(times)] = np.nan
    return PositionInfo(frames, times, xyz, hd, valid, dict(
        pose_path=str(pose_path), position_columns=list(position_columns),
        heading_column=heading_column, heading_clockwise=bool(heading_clockwise),
        heading_zero_deg=float(heading_zero_deg), frame_offset=int(frame_offset),
        bounds_px=[left, right, top, bottom], size_cm=float(size_cm),
        timestamp_reference="provided ADC-relative exposure times",
        angle_convention="CCW from world +X",
        heading_calibration="pixel heading vector scaled by independent X/Y cm per pixel",
        position_correction_policy=POSITION_CORRECTION_POLICY,
        held_position_count=int(held.sum()), initial_outside_count=int(initial_outside.sum()),
    ), raw_xyz_cm=raw_xyz, position_held=held, initial_outside=initial_outside)


def load_motive_position(pose_path, video_times, video_motive_rows, *, units_per_cm,
                         reference_quaternion_xyzw, hd_world_zero_deg, headplate="hp4",
                         center_xy_cm=None, radius_cm=None):
    """Read an explicit video-to-Motive frame mapping and calibrated fused yaw.

    World XYZ retains the original Motive origin, scaled into centimetres.
    Missing clock/mapping/pose entries remain invalid full-video ordinals.
    """
    from EBB.ebc_camera import fused_yaw, load_motive_head

    pose_path = Path(pose_path)
    times = np.asarray(video_times, dtype=float).copy()
    rows = np.asarray(video_motive_rows, dtype=float)
    if times.ndim != 1 or rows.shape != times.shape:
        raise ValueError("Video times and Motive frame mapping must have the same one-dimensional shape.")
    if not np.isfinite(units_per_cm) or units_per_cm <= 0:
        raise ValueError("Motive calibration requires positive finite raw units per cm.")
    mapped = np.isfinite(rows) & (rows >= 0)
    if np.any(rows[mapped] != np.floor(rows[mapped])):
        raise ValueError("Mapped Motive frame IDs must be integers.")
    frame_ids = np.arange(len(times))
    lookup = np.full(len(times), -1, dtype=int)
    lookup[mapped] = rows[mapped].astype(int)
    full_pose = load_motive_head(pose_path, headplate=headplate)
    raw_positions = full_pose[["X", "Y", "Z"]].to_numpy(dtype=float) / units_per_cm
    corrected = raw_positions.copy()
    held = initial_outside = np.zeros(len(full_pose), dtype=bool)
    if radius_cm is not None:
        inside = np.linalg.norm(raw_positions[:, :2] - center_xy_cm, axis=1) <= radius_cm
        corrected, held, initial_outside = hold_last_in_bounds(raw_positions, inside)
    positions = pd.DataFrame(corrected, index=full_pose.index).reindex(lookup)
    raw_xyz = pd.DataFrame(raw_positions, index=full_pose.index).reindex(lookup).to_numpy()
    held = pd.Series(held, index=full_pose.index).reindex(lookup, fill_value=False).to_numpy()
    initial_outside = pd.Series(initial_outside, index=full_pose.index).reindex(lookup, fill_value=False).to_numpy()
    pose = full_pose.reindex(lookup)
    xyz = positions.to_numpy(copy=True)
    rotations = pose[[f"rotation_{axis}_deg" for axis in "xyz"]].to_numpy(dtype=float)
    rotation_valid = np.isfinite(rotations).all(axis=1)
    hd = np.full(len(times), np.nan)
    if rotation_valid.any():
        hd[rotation_valid] = (fused_yaw(rotations[rotation_valid], reference_quaternion_xyzw)
                              + hd_world_zero_deg) % 360.
    synchronized = mapped & np.isfinite(times)
    times[~synchronized] = np.nan
    xyz[~synchronized] = np.nan
    hd[~synchronized] = np.nan
    valid = synchronized & np.isfinite(xyz).all(axis=1) & np.isfinite(hd)
    return PositionInfo(frame_ids, times, xyz, hd, valid, dict(
        pose_path=str(pose_path), headplate=headplate, raw_units_per_cm=float(units_per_cm),
        reference_quaternion_xyzw=np.asarray(reference_quaternion_xyzw).tolist(),
        hd_world_zero_deg=float(hd_world_zero_deg),
        frame_mapping="explicit video ordinal to Motive CSV frame ID",
        timestamp_reference="provided ADC-relative video times",
        angle_convention="CCW from world +X; reference-relative fused yaw plus world zero",
        position_correction_policy=POSITION_CORRECTION_POLICY if radius_cm is not None else "none; no boundary supplied",
        held_position_count=int(held.sum()), initial_outside_count=int(initial_outside.sum()),
    ), raw_xyz_cm=raw_xyz, position_held=held, initial_outside=initial_outside)
