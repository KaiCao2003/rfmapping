import numpy as np

from Utils.ebc_replay import legacy_eye_xy, nearest_rows, ray_circle_distance, saved_boundary_in_world, signed_ebc_display


def test_saved_geometry_keeps_original_ebc_distances_after_screen_transform():
    calibration = dict(motive_units_per_real_mm=4., screen_bottom_center=[80., 20., 0.])
    metadata = dict(boundaryCenterRaw=[-40., 50.], cmPerPositionUnit=.02, boundaryDiameterCm=30.)
    boundary = saved_boundary_in_world(metadata, calibration)
    raw_body = np.array([30., -50.])
    world_body = (raw_body - np.array([80., 20.])) / 40
    world_distance = ray_circle_distance(world_body, np.arange(0, 360, 6), boundary["center_cm"], boundary["radius_cm"])
    model_body = (raw_body - metadata["boundaryCenterRaw"]) * metadata["cmPerPositionUnit"]
    expected = ray_circle_distance(model_body, np.arange(0, 360, 6), np.zeros(2), 15.)
    np.testing.assert_allclose(world_distance * boundary["model_cm_per_world_cm"], expected, atol=1e-12)


def test_forward_intersection_handles_offcenter_tangent_and_outside():
    np.testing.assert_allclose(ray_circle_distance([3., 0.], [0., 90., 180.], np.zeros(2), 5.), [2., 4., 8.])
    assert ray_circle_distance([5., 0.], 90., np.zeros(2), 5.) < 1e-7
    assert np.isnan(ray_circle_distance([6., 0.], 0., np.zeros(2), 5.))


def test_eye_model_applies_calibration_zero_once_and_keeps_units():
    calibration = dict(fm_eye_position_model="xy_offset_rotated_by_calibrated_hd", screen_center=20., hp_to_eye_center=[2., 0., 8.])
    actual = legacy_eye_xy(np.array([[10., 30.], [10., 30.]]), np.array([20., 110.]), calibration)
    np.testing.assert_allclose(actual, [[12., 30.], [10., 32.]])


def test_nearest_pose_does_not_shift_tie_or_outside_target():
    np.testing.assert_array_equal(nearest_rows(np.array([1., 2., 4.]), np.array([.9, 1.5, 2.2, 3.9, 5.])), [0, 0, 1, 2, 2])


def test_heatmap_cells_center_on_sampled_rays_and_wrap_the_seam():
    original = np.arange(60.)[:, None]
    edges, values = signed_ebc_display(original)
    centers = (edges[:-1] + edges[1:]) / 2
    np.testing.assert_array_equal(centers, np.arange(-180, 181, 6))
    np.testing.assert_array_equal(values[:, 0] * 6 % 360, centers % 360)
    assert values[0, 0] == values[-1, 0]
