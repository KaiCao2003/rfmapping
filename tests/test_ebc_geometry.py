import pickle

import numpy as np
import pandas as pd
import pytest

from EBB.ebc_geometry import (
    BASLER_SIZE_CM, PositionInfo, compute_ebc_rays, cylinder_boundary,
    pixel_heading_to_world_deg, project_points, rectangle_boundary,
    rectangle_distances_cm,
)
from EBB.ebc_pose import load_basler_position, load_motive_position


def position(xyz, headings, *, valid=None):
    xyz = np.asarray(xyz, dtype=float)
    frames = np.arange(len(xyz))
    return PositionInfo(frames, frames / 30., xyz, np.asarray(headings, dtype=float),
                        np.ones(len(xyz), dtype=bool) if valid is None else np.asarray(valid))


def cylinder():
    calibration = dict(screen_bottom_center=[100., 200., 30.], screen_diamter=200.,
                       motive_units_per_real_mm=1.)
    registration = dict(world_to_video_projection_raw=[
        [2., 0., .5, 100.], [0., -3., .25, 700.], [0., 0., .001, 1.],
    ])
    return cylinder_boundary(calibration, registration), registration


def test_rectangle_eight_rays_are_cm_and_project_to_calibrated_pixels():
    boundary = rectangle_boundary((100., 500., 200., 800.), 20.)
    data = position([[10., 10., 0.]], [90.])
    result = compute_ebc_rays(data, boundary)
    np.testing.assert_array_equal(result.bearings_deg, [0., 45., 90., 135., -180., -135., -90., -45.])
    np.testing.assert_allclose(result.distances_cm[0, ::2], 10.)
    np.testing.assert_allclose(result.distances_cm[0, 1::2], np.sqrt(200.))
    pixels = project_points(result.endpoints_cm, boundary.projection)
    np.testing.assert_allclose(pixels[0, ::2], [[300., 200.], [500., 500.],
                                             [300., 800.], [100., 500.]], atol=1e-10)
    assert result.valid.tolist() == [True]


def test_rectangle_at_wall_uses_opposite_wall_for_inward_rays():
    boundary = rectangle_boundary((0., 200., 0., 200.), 20.)
    data = position([[0., 10., 0.], [0., 0., 0.]], [0., 0.])
    result = compute_ebc_rays(data, boundary)
    np.testing.assert_allclose(result.distances_cm[0, [0, 2, 4, 6]], [20., 10., 0., 10.])
    np.testing.assert_allclose(result.distances_cm[1, [0, 7, 6, 4, 2]],
                               [20., np.sqrt(800.), 20., 0., 0.], atol=1e-10)


def test_cylinder_intersection_is_exact_and_preserves_pose_height():
    boundary, registration = cylinder()
    data = position([[10., 20., 8.], [15., 20., 8.], [20., 20., 8.]], [0., 0., 0.])
    result = compute_ebc_rays(data, boundary)
    np.testing.assert_allclose(result.distances_cm[0], 10.)
    np.testing.assert_allclose(result.distances_cm[1, [0, 2, 4]], [5., np.sqrt(75.), 15.])
    np.testing.assert_allclose(result.distances_cm[2, [0, 4]], [0., 20.])
    np.testing.assert_allclose(result.endpoints_cm[..., 2], 8.)
    # Projecting cm must equal projecting the original absolute Motive world units.
    np.testing.assert_allclose(project_points(data.xyz_cm, boundary.projection),
                               project_points(data.xyz_cm * 10., registration["world_to_video_projection_raw"]))
    assert boundary.metadata["raw_units_per_cm"] == 10.


@pytest.mark.parametrize("shape", ["rectangle", "cylinder"])
def test_invalid_pose_and_outside_origins_never_get_rays(shape):
    boundary = rectangle_boundary((0., 200., 0., 200.), 20.) if shape == "rectangle" else cylinder()[0]
    data = position([[100., 100., 0.], [np.nan, 10., 0.], [10., 20., 0.]],
                    [0., 0., 0.], valid=[True, True, False])
    result = compute_ebc_rays(data, boundary)
    assert not result.valid.any()
    assert np.isnan(result.distances_cm).all()
    assert np.isnan(result.endpoints_cm).all()


@pytest.mark.parametrize("shape", ["rectangle", "cylinder"])
def test_boundary_can_cross_spawn_worker_boundary(shape):
    boundary = rectangle_boundary((0., 200., 0., 200.), 20.) if shape == "rectangle" else cylinder()[0]
    restored = pickle.loads(pickle.dumps(boundary))
    data = position([[10., 15., 2.]], [23.])
    np.testing.assert_allclose(compute_ebc_rays(data, restored).distances_cm,
                               compute_ebc_rays(data, boundary).distances_cm)


@pytest.mark.parametrize("clockwise, expected_heading", [(False, 180.), (True, 0.)])
def test_basler_loader_keeps_frame_ids_and_explicit_heading_convention(tmp_path, clockwise, expected_heading):
    path = tmp_path / "pose.csv"
    pd.DataFrame(dict(frame=[0, 2, 3], center_x=[300., 300., 300.],
                      center_y=[500., 500., 500.], hd_deg=[90., 0., 0.])).to_csv(path, index=False)
    loaded = load_basler_position(path, [2., 2.1, np.nan], bounds_px=(100., 500., 200., 800.),
                                  size_cm=20., heading_clockwise=clockwise)
    np.testing.assert_array_equal(loaded.frame_ids, [0, 1, 2, 3])
    np.testing.assert_allclose(loaded.xyz_cm[0], [10., 10., 0.])
    assert loaded.hd_deg[0] == expected_heading
    np.testing.assert_allclose(loaded.times_s, [2., 2.1, np.nan, np.nan], equal_nan=True)
    assert loaded.valid.tolist() == [True, False, False, False]
    assert np.isnan(loaded.xyz_cm[1:]).all()


def test_basler_missing_pose_retains_known_exposure_and_full_video_ordinals(tmp_path):
    path = tmp_path / "pose.csv"
    pd.DataFrame(dict(frame=[0, 2], center_x=[300., 300.],
                      center_y=[500., 500.], hd_deg=[0., 90.])).to_csv(path, index=False)
    loaded = load_basler_position(path, [10., 10.1, 10.2],
                                  bounds_px=(100., 500., 200., 800.), size_cm=20.)
    np.testing.assert_array_equal(loaded.frame_ids, [0, 1, 2])
    np.testing.assert_allclose(loaded.times_s, [10., 10.1, 10.2])
    np.testing.assert_allclose(loaded.xyz_cm[[0, 2]], [[10., 10., 0.], [10., 10., 0.]])
    assert loaded.valid.tolist() == [True, False, True]
    assert np.isnan(loaded.xyz_cm[1]).all()
    assert np.isnan(loaded.hd_deg[1])


def test_basler_loader_uses_requested_columns_and_frame_offset(tmp_path):
    path = tmp_path / "pose.csv"
    pd.DataFrame(dict(frame=[1], front_x=[100.], front_y=[200.], angle=[0.])).to_csv(path, index=False)
    loaded = load_basler_position(path, [3.], bounds_px=(100., 500., 200., 800.), size_cm=20.,
                                  position_columns=("front_x", "front_y"), heading_column="angle",
                                  heading_zero_deg=45., frame_offset=-1)
    np.testing.assert_array_equal(loaded.frame_ids, [0])
    np.testing.assert_allclose(loaded.xyz_cm, [[0., 20., 0.]])
    np.testing.assert_allclose(loaded.hd_deg, [np.rad2deg(np.arctan2(20. / 600., 20. / 400.))])


@pytest.mark.parametrize("clockwise, zero_deg", [(True, 90.), (False, 90.), (False, 0.), (True, 45.)])
def test_pixel_heading_matches_affine_front_back_vector(tmp_path, clockwise, zero_deg):
    bounds = (100., 900., 200., 800.)
    back = np.array([[400., 450.], [500., 450.], [500., 500.], [400., 500.]])
    front = back + np.array([[80., -120.], [-120., -80.], [-80., 120.], [120., 80.]])
    pixel_vector = front - back
    pixel_world_angle = np.rad2deg(np.arctan2(-pixel_vector[:, 1], pixel_vector[:, 0]))
    sign = -1. if clockwise else 1.
    source_heading = (pixel_world_angle - zero_deg) / sign
    # Independent oracle: affine-transform the measured front and back points.
    affine = np.array([[40. / 800., 0.], [0., -40. / 600.]])
    physical_vector = front @ affine.T - back @ affine.T
    expected = np.rad2deg(np.arctan2(physical_vector[:, 1], physical_vector[:, 0])) % 360.
    actual = pixel_heading_to_world_deg(source_heading, bounds_px=bounds, size_cm=BASLER_SIZE_CM,
                                        heading_clockwise=clockwise, heading_zero_deg=zero_deg)
    np.testing.assert_allclose(actual, expected, atol=1e-12)

    center = (front + back) / 2.
    path = tmp_path / "pose.csv"
    pd.DataFrame(dict(frame=np.arange(len(center)), center_x=center[:, 0],
                      center_y=center[:, 1], hd_deg=source_heading)).to_csv(path, index=False)
    loaded = load_basler_position(path, np.arange(len(center)) / 30., bounds_px=bounds,
                                  size_cm=BASLER_SIZE_CM, heading_clockwise=clockwise,
                                  heading_zero_deg=zero_deg)
    np.testing.assert_allclose(loaded.hd_deg, expected, atol=1e-12)
    rays = compute_ebc_rays(loaded, rectangle_boundary(bounds, BASLER_SIZE_CM), bearings_deg=[0.])
    forward = rays.endpoints_cm[:, 0, :2] - loaded.xyz_cm[:, :2]
    unit_vector = physical_vector / np.linalg.norm(physical_vector, axis=1)[:, None]
    np.testing.assert_allclose(forward / np.linalg.norm(forward, axis=1)[:, None], unit_vector, atol=1e-12)
    pixel_north_cw = np.rad2deg(np.arctan2(pixel_vector[:, 0], -pixel_vector[:, 1]))
    distances = rectangle_distances_cm(center[:, 0], center[:, 1], pixel_north_cw, [0.],
                                       bounds_px=bounds, size_cm=BASLER_SIZE_CM)
    np.testing.assert_allclose(distances, rays.distances_cm, atol=1e-12)
    endpoint = rays.endpoints_cm[:, 0, :2]
    assert np.all(np.any(np.isclose(endpoint, 0.) | np.isclose(endpoint, BASLER_SIZE_CM), axis=1))


@pytest.mark.parametrize("clockwise, zero_deg", [(True, 90.), (False, 90.), (False, 0.), (True, 45.)])
def test_square_pixel_calibration_preserves_heading(clockwise, zero_deg):
    headings = np.arange(-720., 721., 13.)
    actual = pixel_heading_to_world_deg(headings, bounds_px=(100., 600., 200., 700.),
                                        size_cm=BASLER_SIZE_CM, heading_clockwise=clockwise,
                                        heading_zero_deg=zero_deg)
    sign = -1. if clockwise else 1.
    expected = (zero_deg + sign * headings) % 360.
    error = (actual - expected + 180.) % 360. - 180.
    np.testing.assert_allclose(error, 0., atol=1e-12)


def test_heading_calibration_preserves_missing_values():
    actual = pixel_heading_to_world_deg([0., np.nan, 90.], bounds_px=(0., 600., 0., 400.), size_cm=40.)
    np.testing.assert_allclose(actual, [90., np.nan, 0.], equal_nan=True)


def test_motive_loader_uses_csv_frame_ids_and_keeps_full_video_gaps(tmp_path):
    path = tmp_path / "motive.csv"
    columns = pd.MultiIndex.from_tuples(
        [("meta", "id", "meta", "Frame")]
        + [("chosen", "rigid-id", kind, axis) for kind in ("Position", "Rotation") for axis in "XYZ"]
    )
    pd.DataFrame([[12., 100., 200., 80., 0., 0., 90.],
                  [15., 150., 200., 100., 0., 0., 0.]], columns=columns).to_csv(path, index=False)
    loaded = load_motive_position(path, [1., np.nan, 1.2, 1.3], [12., 15., 15., np.nan],
                                  units_per_cm=10., reference_quaternion_xyzw=[0., 0., 0., 1.],
                                  hd_world_zero_deg=30., headplate="chosen")
    np.testing.assert_array_equal(loaded.frame_ids, [0, 1, 2, 3])
    np.testing.assert_allclose(loaded.xyz_cm[[0, 2]], [[10., 20., 8.], [15., 20., 10.]])
    np.testing.assert_allclose(loaded.hd_deg[[0, 2]], [120., 30.])
    assert loaded.valid.tolist() == [True, False, True, False]
    assert np.isnan(loaded.xyz_cm[[1, 3]]).all()
    assert np.isnan(loaded.times_s[[1, 3]]).all()
