"""Egocentric bearings of circular-screen edges."""

import numpy as np


def bearing(x, y, r, hd, startedgeindeg, endedgeindeg):
    """Return ``(start_ego_deg, end_ego_deg)`` for a counterclockwise VS arc.

    ``x, y`` are the observer's position relative to the circle center
    ``(0, 0)``; ``r`` is its positive radius. All three use the same unit and
    scale, and the observer must be inside the circle.

    ``hd``, ``startedgeindeg`` and ``endedgeindeg`` are allocentric degrees
    in the same XY frame: 0 along +X, 90 along +Y, positive counterclockwise.
    The edges delimit the arc traversed counterclockwise from start to end.
    As in FM bar geometry, ego bearing is ``atan2(edge_y-y, edge_x-x) - hd``.
    No calibration, eye-position, or angular offsets are applied.

    Results are degrees in [-180, 180), with 0 straight ahead and positive
    counterclockwise. Edge order is preserved across the wrap: (170, -170)
    represents the counterclockwise arc through 180, so do not sort it.
    Scalar inputs return two floats; broadcastable array inputs return two
    NumPy arrays of the broadcast shape. Inputs must be finite.

    Examples
    --------
    >>> bearing(0, 0, 1, 0, -10, 10)
    (-10.0, 10.0)
    >>> bearing(0.5, 0.5, 1, 0, 0, 90)
    (-45.0, 135.0)
    """
    values = np.broadcast_arrays(
        *[np.asarray(value, dtype=float) for value in
          (x, y, r, hd, startedgeindeg, endedgeindeg)]
    )
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("All bearing inputs must be finite.")
    x, y, r, hd, start, end = values
    if np.any(r <= 0):
        raise ValueError("r must be positive.")
    if np.any(np.hypot(x, y) >= r):
        raise ValueError("The observer must be inside the circle: hypot(x, y) < r.")

    # Work on the unit circle so the result is independent of length units.
    x, y = x / r, y / r
    hd = hd % 360

    def edge_bearing(edge):
        angle = np.deg2rad(edge % 360)
        allocentric = np.rad2deg(np.arctan2(np.sin(angle) - y, np.cos(angle) - x))
        ego = (allocentric - hd + 180) % 360 - 180
        return ego.item() if ego.ndim == 0 else ego

    return edge_bearing(start), edge_bearing(end)
