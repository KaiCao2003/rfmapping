"""Boundary adapters and EBC rays in world centimetres and CCW world angles."""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np


@dataclass
class PositionInfo:
    frame_ids: np.ndarray
    times_s: np.ndarray
    xyz_cm: np.ndarray
    hd_deg: np.ndarray
    valid: np.ndarray
    source: dict = field(default_factory=dict)


@dataclass
class BoundaryInfo:
    intersect: Callable
    projection: np.ndarray
    outline_xyz_cm: tuple
    metadata: dict = field(default_factory=dict)


@dataclass
class EBCInfo:
    bearings_deg: np.ndarray
    positions_cm: np.ndarray
    endpoints_cm: np.ndarray
    distances_cm: np.ndarray
    valid: np.ndarray


def _rectangle_intersections(origins, directions, *, size_cm):
    xy = np.asarray(origins)[:, None, :2]
    step = np.asarray(directions)[..., :2]
    walls = np.where(step > 0, size_cm, 0.)
    distances = np.min(np.divide(
        walls - xy, step, out=np.full_like(step, np.inf),
        where=np.abs(step) > 1e-12,
    ), axis=-1)
    inside = np.isfinite(xy).all(axis=(1, 2)) & ((xy >= 0) & (xy <= size_cm)).all(axis=(1, 2))
    return np.where(inside[:, None], np.maximum(distances, 0.), np.nan)


def _cylinder_intersections(origins, directions, *, center_xy_cm, radius_cm):
    relative = np.asarray(origins)[:, None, :2] - center_xy_cm
    step = np.asarray(directions)[..., :2]
    projection = np.sum(relative * step, axis=-1)
    norm2 = np.sum(relative * relative, axis=-1)
    distances = -projection + np.sqrt(np.maximum(projection**2 + radius_cm**2 - norm2, 0.))
    inside = np.isfinite(norm2) & (norm2 <= radius_cm**2 + 1e-9)
    return np.where(inside, np.maximum(distances, 0.), np.nan)


def rectangle_boundary(bounds_px, size_cm):
    """Adapt a calibrated image rectangle to lower-left-origin world XY in cm."""
    left, right, top, bottom = np.asarray(bounds_px, dtype=float)
    size_cm = float(size_cm)
    if not np.isfinite([left, right, top, bottom, size_cm]).all() or not (
        right > left and bottom > top and size_cm > 0
    ):
        raise ValueError("Rectangle calibration requires increasing pixel bounds and a positive size in cm.")
    projection = np.array([
        [(right - left) / size_cm, 0., 0., left],
        [0., -(bottom - top) / size_cm, 0., bottom],
        [0., 0., 0., 1.],
    ])
    outline = np.array([[0., 0., 0.], [size_cm, 0., 0.], [size_cm, size_cm, 0.],
                        [0., size_cm, 0.], [0., 0., 0.]])
    return BoundaryInfo(
        intersect=partial(_rectangle_intersections, size_cm=size_cm),
        projection=projection, outline_xyz_cm=(outline,),
        metadata=dict(bounds_px=[left, right, top, bottom], size_cm=size_cm,
                      coordinate_system="world cm; XY origin at rectangle lower left"),
    )


def cylinder_boundary(calibration, registration):
    """Adapt the physical cylinder and raw-world camera matrix without refitting."""
    units_per_cm = 10. * float(calibration["motive_units_per_real_mm"])
    center = np.asarray(calibration["screen_bottom_center"], dtype=float)
    diameter = float(calibration["screen_diamter"])
    if not np.isfinite(units_per_cm) or units_per_cm <= 0 or not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Cylinder calibration requires positive finite physical dimensions.")
    center = center / units_per_cm
    radius_cm = diameter / (2. * units_per_cm)
    projection = np.asarray(registration["world_to_video_projection_raw"], dtype=float).copy()
    if center.shape != (3,) or not np.isfinite(center).all() or projection.shape != (3, 4):
        raise ValueError("Cylinder calibration requires an XYZ center and a 3 by 4 camera projection.")
    projection[:, :3] *= units_per_cm
    # The sampled circle is only a drawing outline; ray intersections stay analytic.
    angles = np.linspace(0., 2. * np.pi, 361)
    outline = center + radius_cm * np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    return BoundaryInfo(
        intersect=partial(_cylinder_intersections, center_xy_cm=center[:2], radius_cm=radius_cm),
        projection=projection, outline_xyz_cm=(outline,),
        metadata=dict(center_xyz_cm=center.tolist(), radius_cm=radius_cm,
                      raw_units_per_cm=units_per_cm,
                      coordinate_system="absolute Motive world coordinates divided by raw_units_per_cm"),
    )


def compute_ebc_rays(position, boundary, *, bearings_deg=(0., 45., 90., 135., 180., 225., 270., 315.)):
    """Compute horizontal rays; positive egocentric bearing turns left from HD.

    Both pose loaders supply CCW world HD with +X at zero. Boundary adapters
    own their exact intersection rule, keeping geometry dispatch out of here.
    """
    bearings = np.asarray(bearings_deg, dtype=float)
    origins = np.asarray(position.xyz_cm, dtype=float)
    angles = np.deg2rad(np.asarray(position.hd_deg)[:, None] + bearings)
    directions = np.stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1)
    distances = boundary.intersect(origins, directions)
    valid = (np.asarray(position.valid, dtype=bool) & np.isfinite(origins).all(axis=1)
             & np.isfinite(position.hd_deg) & np.isfinite(distances).all(axis=1))
    distances = np.where(valid[:, None], distances, np.nan)
    endpoints = origins[:, None, :] + directions * distances[..., None]
    return EBCInfo(bearings, origins.copy(), endpoints, distances, valid)


def project_points(points, projection):
    """Project world-cm points into camera pixels without changing geometry."""
    points = np.asarray(points, dtype=float)
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    projected = homogeneous @ np.asarray(projection).T
    return np.divide(projected[..., :2], projected[..., 2:3],
                     out=np.full(projected.shape[:-1] + (2,), np.nan),
                     where=projected[..., 2:3] != 0)
