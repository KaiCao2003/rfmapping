import pickle

import numpy as np
import pandas as pd
import pytest

from Utils.ebc_geometry import (PositionInfo, compute_ebc_rays, cylinder_boundary,
                                project_points, rectangle_boundary)
from Utils.ebc_pose import load_basler_position, load_motive_position


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
    np.testing.assert_array_equal(result.bearings_deg, np.arange(0., 360., 45.))
    np.testing.assert_allclose(result.distances_cm[0, ::2], 10.)
    np.testing.assert_allclose(result.distances_cm[0, 1::2], np.sqrt(200.))
    pixels = project_points(result.endpoints_cm, boundary.projection)
    np.testing.assert_allclose(pixels[0, ::2], [[300., 200.], [100., 500.],
                                             [300., 800.], [500., 500.]], atol=1e-10)
    assert result.valid.tolist() == [True]


def test_rectangle_at_wall_uses_opposite_wall_for_inward_rays():
    boundary = rectangle_boundary((0., 200., 0., 200.), 20.)
    data = position([[0., 10., 0.], [0., 0., 0.]], [0., 0.])
    result = compute_ebc_rays(data, boundary)
    np.testing.assert_allclose(result.distances_cm[0, [0, 2, 4, 6]], [20., 10., 0., 10.])
    np.testing.assert_allclose(result.distances_cm[1, [0, 1, 2, 4, 6]],
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
    np.testing.assert_allclose(loaded.hd_deg, [45.])


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
