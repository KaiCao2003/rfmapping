import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pytest

from EBB import spatial_cell_analysis as spatial
from EBB.ebc_analysis import THETA_DEG, THETA_EDGES, circle_distances
from EBB.ebc_geometry import PositionInfo, compute_ebc_rays, rectangle_boundary, rectangle_distances_cm
from Utils.plotting import plot_egocentric_polar, plot_tuning_curve


def test_pixel_rays_turn_right_for_clockwise_bearings():
    distances = spatial.d(np.array([0., 90., 180., 270.]), np.array([30., 30.]),
                          np.array([80., 80.]), np.array([0., 90.]), bounds=(0, 100, 0, 200))
    np.testing.assert_allclose(distances, [[80, 70, 120, 30], [70, 120, 30, 80]])


def test_physical_rectangle_matches_video_at_non_square_pixel_bounds(monkeypatch):
    bounds = (230., 740., 335., 835.)
    for key, value in zip(("x_min", "x_max", "y_min", "y_max"), bounds):
        monkeypatch.setattr(spatial, key, value)
    pose = pd.DataFrame(dict(center_x=[383., 383.], center_y=[515., 515.], hd_deg=[0., 90.]))
    maps = spatial.prepare_session_maps(pose, np.array([0., .02]))
    centers = (maps["theta_edges"][:-1] + maps["theta_edges"][1:]) / 2
    np.testing.assert_array_equal(centers, np.arange(3., 360., 6.))
    np.testing.assert_array_equal(maps["theta_grid"], np.broadcast_to(centers, (2, 60)))
    xyz = np.array([[12., 25.6, 0.], [12., 25.6, 0.]])
    position = PositionInfo(np.arange(2), np.array([0., .02]), xyz,
                            np.array([90., 0.]), np.ones(2, dtype=bool))
    video = compute_ebc_rays(position, rectangle_boundary(bounds, 40.), bearings_deg=centers)
    np.testing.assert_allclose(maps["theta_d_cm"], video.distances_cm, atol=1e-12)
    cardinals = rectangle_distances_cm(pose.center_x, pose.center_y, pose.hd_deg,
                                       [0., 90., 180., 270.], bounds_px=bounds, size_cm=40.)
    np.testing.assert_allclose(cardinals, [[14.4, 28., 25.6, 12.], [28., 25.6, 12., 14.4]])
    np.testing.assert_allclose(maps["theta_d_cm"][:, 0], np.array([14.4, 28.]) / np.cos(np.deg2rad(3.)))


def test_invalid_rectangle_position_has_no_distance():
    distances = rectangle_distances_cm([-1., np.nan], [50., 50.], [0., 0.],
                                       [0., 90.], bounds_px=(0, 100, 0, 100), size_cm=40.)
    assert np.isnan(distances).all()


def test_cylinder_raw_ccw_yaw_has_clockwise_egocentric_bearings():
    xy = np.array([[3., 4.], [3., 4.]])
    distances = circle_distances(xy, np.array([0., 90.]), 10., theta_deg=np.array([0., 90., 180., 270.]))
    east, south, west, north = np.sqrt(84.) - 3., np.sqrt(91.) + 4., np.sqrt(84.) + 3., np.sqrt(91.) - 4.
    np.testing.assert_allclose(distances, [[east, south, west, north], [north, east, south, west]])
    np.testing.assert_array_equal(THETA_DEG, (THETA_EDGES[:-1] + THETA_EDGES[1:]) / 2)


def test_exported_angles_are_exact_geometry_samples(tmp_path, monkeypatch):
    pose = pd.DataFrame(dict(center_x=[500., 510.], center_y=[400., 410.], hd_deg=[0., 90.]))
    maps = spatial.prepare_session_maps(pose, np.array([0., .02]))
    path = tmp_path / "new.rfmap"
    spatial.save_egocentric_rfmap(path, np.zeros((1, 60, 20)), maps, [1], [0., .02])
    with np.load(path) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        np.testing.assert_array_equal(archive["yPositions"], maps["theta_grid"][0])
        assert metadata["angleSampling"] == "bin_centers"
        assert metadata["angleConvention"] == "egocentric clockwise: 0 forward, 90 right"
        assert metadata["arenaSizeCm"] == [40, 40]


@pytest.mark.parametrize("kind", ["map", "tuning"])
def test_polar_positive_90_is_physically_right(kind):
    figure, axis = plt.subplots(subplot_kw={"projection": "polar"})
    if kind == "map":
        plot_egocentric_polar(axis, np.arange(0., 361., 90.), np.array([0., 1.]), np.ones((4, 1)))
    else:
        plot_tuning_curve(axis, np.arange(0., 361., 90.), np.ones(4), "EBC")
    figure.canvas.draw()
    center = axis.transData.transform((0., 0.))
    right = axis.transData.transform((np.pi / 2, .5))
    assert right[0] > center[0]
    assert right[1] == pytest.approx(center[1])
    assert axis.get_theta_direction() == -1
    plt.close(figure)
