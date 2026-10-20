import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from Utils.ebc_camera import (
    boundary_endpoints, fused_yaw, load_motive_head, motive_vs_geometry, project,
)


def test_boundary_rays_rotate_with_head_and_stay_on_calibrated_screen():
    body = np.array([10., 20., 30.])
    center = np.array([0., 0.])
    endpoints, distances = boundary_endpoints(body, 90., center, 50.)
    np.testing.assert_allclose(np.linalg.norm(endpoints[:, :2]-center, axis=1), 50.)
    np.testing.assert_allclose(endpoints[:, 2], body[2])
    assert endpoints[0, 1] > body[1]  # Forward is +Y at 90 degrees.
    assert endpoints[1, 0] < body[0]  # Left is -X.
    assert np.all(distances > 0)


def test_projection_preserves_head_height_and_perspective_division():
    projection = np.array([[2., 0, 0, 5.], [0, 3., 0, -4.], [0, 0, .1, 1.]])
    np.testing.assert_allclose(project(np.array([10., 20., 30.]), projection), [6.25, 14.])


def test_bearing_uses_head_position_and_corrected_hd_before_projection():
    # Head is off center, with a target three units right and three units up.
    head = np.array([12., 21., 30.])
    center = np.array([10., 20.])
    world_angle = np.rad2deg(np.arctan2(4., 5.))
    target, beta = motive_vs_geometry(head, 15., center, np.sqrt(41.), -world_angle)
    np.testing.assert_allclose(target, [15., 24., 30.])
    np.testing.assert_allclose(beta, 30.)
    corrected_target, corrected_beta = motive_vs_geometry(
        head, 15. + 31., center, np.sqrt(41.), -world_angle)
    np.testing.assert_allclose(corrected_target, target)
    np.testing.assert_allclose(corrected_beta, -1.)
    direction = np.deg2rad(46. + corrected_beta)
    reconstructed = head[:2] + np.linalg.norm(target[:2] - head[:2]) * np.array([np.cos(direction), np.sin(direction)])
    np.testing.assert_allclose(reconstructed, target[:2])


def test_bearing_keeps_rf_sign_and_wraps_at_rear():
    head = np.array([0., 0., 7.])
    target, beta = motive_vs_geometry(head, 190., np.zeros(2), 10., -10.)
    np.testing.assert_allclose(target[:2], 10 * np.array([np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))]))
    np.testing.assert_allclose(beta, -180.)


def test_bearing_is_invariant_to_common_motive_rotation_translation_and_scale():
    head = np.array([12., 21., 30.])
    center = np.array([10., 20.])
    target, beta = motive_vs_geometry(head, 73., center, 10., -45.)
    rotation_deg = 31.
    c, s = np.cos(np.deg2rad(rotation_deg)), np.sin(np.deg2rad(rotation_deg))
    rotation = np.array([[c, -s], [s, c]])
    translation = np.array([130., -62.])
    moved_head = np.r_[3 * rotation @ head[:2] + translation, head[2] * 3]
    moved_center = 3 * rotation @ center + translation
    moved_target, moved_beta = motive_vs_geometry(
        moved_head, 73. + rotation_deg, moved_center, 30., -45. - rotation_deg)
    np.testing.assert_allclose(moved_target[:2], 3 * rotation @ target[:2] + translation)
    np.testing.assert_allclose(moved_beta, beta)


def test_vs_cardinal_positions_reverse_direction_without_extra_zero_offset():
    head = np.array([10., 20., 30.])
    center = head[:2]
    for vs, xy, expected_beta in [(0., [60., 20.], 0.), (90., [10., -30.], -90.),
                                  (180., [-40., 20.], -180.), (270., [10., 70.], 90.)]:
        target, beta = motive_vs_geometry(head, 0., center, 50., vs)
        np.testing.assert_allclose(target, [*xy, 30.], atol=1e-12)
        np.testing.assert_allclose(beta, expected_beta)


def test_session_first_frame_uses_negative_vs_and_corrected_head_direction():
    target, beta = motive_vs_geometry(
        np.array([529.0401, -283.708527, 876.517944]), 294.060394 + 31.,
        np.array([88.75734929896345, 24.961410592847983]),
        2399.4772567503296 / 2, 114.)
    np.testing.assert_allclose(target[:2], [-399.2203129747159, -1071.0543637176484], atol=1e-9)
    np.testing.assert_allclose(beta, -104.75595107351535, atol=1e-9)


def test_screen_zero_changes_target_without_changing_head_coordinate_frame():
    head = np.array([3., 1., 2.])
    target, beta = motive_vs_geometry(head, 90., np.zeros(2), 5., 0., vs_zero_deg=90.)
    np.testing.assert_allclose(target, [0., 5., 2.], atol=1e-12)
    np.testing.assert_allclose(beta, np.rad2deg(np.arctan2(4., -3.)) - 90.)
    np.testing.assert_array_equal(head, [3., 1., 2.])


def test_preserved_motive_reader_pairs_world_pose_by_original_frame_id(tmp_path):
    columns = pd.MultiIndex.from_tuples([
        ('', '', '', 'Frame'),
        ('hp4', 'body-id', 'Position', 'X'), ('hp4', 'body-id', 'Position', 'Y'),
        ('hp4', 'body-id', 'Position', 'Z'), ('hp4', 'body-id', 'Rotation', 'Z'),
        ('hp4', 'body-id', 'Rotation', 'X'), ('hp4', 'body-id', 'Rotation', 'Y'),
        ('hp4:Marker 001', 'marker-id', 'Position', 'X'),
    ])
    path = tmp_path / 'trimmed_input.csv'
    pd.DataFrame([[9, 10, 20, 30, -20, 3, 4, 999], [13, 40, 50, 60, 370, 5, 6, 888]], columns=columns).to_csv(path, index=False)
    selected = load_motive_head(path).loc[[13, 9]]
    np.testing.assert_array_equal(selected[["X", "Y", "Z", "hd_deg"]].to_numpy(), [[40, 50, 60, 10], [10, 20, 30, 340]])
    np.testing.assert_array_equal(selected[[f"rotation_{axis}_deg" for axis in "xyz"]], [[5, 6, 370], [3, 4, -20]])


def test_fused_yaw_keeps_fixed_heading_through_ninety_degree_tilt():
    reference = Rotation.from_euler("XYZ", [27., -2., 3.], degrees=True)
    corrected = Rotation.from_euler("Z", [[70.], [70.], [70.]], degrees=True) * Rotation.from_euler("X", [[0.], [90.], [120.]], degrees=True)
    rotations = (corrected * reference).as_euler("XYZ", degrees=True)
    np.testing.assert_allclose(fused_yaw(rotations, reference.as_quat()), 70., atol=1e-12)


def test_fused_yaw_zero_is_reference_pose_and_inverted_pose_is_invalid():
    reference = Rotation.from_euler("XYZ", [27., -2., 3.], degrees=True)
    rotations = (Rotation.from_euler("X", [[0.], [180.]], degrees=True) * reference).as_euler("XYZ", degrees=True)
    heading = fused_yaw(rotations, reference.as_quat())
    np.testing.assert_allclose((heading[0] + 180) % 360 - 180, 0., atol=1e-12)
    assert np.isnan(heading[1])


def test_reference_facing_vs_zero_defines_world_heading_without_extra_hd_offset():
    # At the screen center, the level reference faces VS=0 regardless of the
    # independent Motive azimuth of that screen zero.
    for screen_zero in [0., 44.7956488331414, 270.]:
        _, beta = motive_vs_geometry(np.array([2., 3., 4.]), screen_zero,
                                     np.array([2., 3.]), 10., 0., screen_zero)
        np.testing.assert_allclose(beta, 0., atol=1e-12)
        _, beta = motive_vs_geometry(np.array([2., 3., 4.]), screen_zero + 30.,
                                     np.array([2., 3.]), 10., 20., screen_zero)
        np.testing.assert_allclose(beta, -50., atol=1e-12)
