import json
from pathlib import Path

import numpy as np
import pytest

from Utils.motive_pose import eye_positions_at_times


@pytest.mark.parametrize(
    "angles,expected",
    [([90, 0, 0], [11, 27, 33]), ([0, 90, 0], [6, 22, 33])],
)
def test_roll_and_pitch_move_the_vertical_eye_offset_into_xy(angles, expected):
    eye = eye_positions_at_times(
        np.array([0., 2.]),
        np.array([[10., 20., 30.], [12., 24., 36.]]),
        np.array([angles, angles], dtype=float),
        np.array([0., 0., -5.]),
        np.array([1.]),
    )
    np.testing.assert_allclose(eye[0], expected, atol=1e-12)


def test_orientation_interpolation_crosses_the_angle_wrap_by_the_short_arc():
    eye = eye_positions_at_times(
        np.array([0., 2.]), np.zeros((2, 3)),
        np.array([[0., 0., 179.], [0., 0., -179.]]),
        np.array([10., 0., 0.]), np.array([1.]),
    )
    np.testing.assert_allclose(eye[0], [-10., 0., 0.], atol=1e-12)


def test_noncommuting_rotations_follow_motive_rx_ry_rz_order():
    angles = np.array([30., 40., 50.])
    x, y, z = np.radians(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    offset = np.array([2., 3., 4.])
    eye = eye_positions_at_times(
        np.array([0., 1.]), np.zeros((2, 3)),
        np.stack([angles, angles]), offset, np.array([0., .5, 1.]),
    )
    np.testing.assert_allclose(eye, np.tile(rx @ ry @ rz @ offset, (3, 1)), atol=1e-12)


def test_real_260821_frame_matches_the_gui_eye_position():
    fixture = json.loads((Path(__file__).parent / 'fixtures/pose_260821_1_frame_90289.json').read_text())
    eye = eye_positions_at_times(
        np.asarray(fixture['frame_times']), np.asarray(fixture['position_xyz']),
        np.asarray(fixture['rotation_xyz_deg']), np.asarray(fixture['eye_offset_xyz']),
        np.asarray([fixture['query_time']]),
    )
    np.testing.assert_allclose(eye[0], fixture['gui_eye_xyz'], rtol=0, atol=1e-10)


def test_real_pose_has_the_same_result_in_motive_and_physical_units():
    fixture = json.loads((Path(__file__).parent / 'fixtures/pose_260821_1_frame_90289.json').read_text())
    eye_mm = eye_positions_at_times(
        np.asarray(fixture['frame_times']), np.asarray(fixture['position_xyz']) / 3.74,
        np.asarray(fixture['rotation_xyz_deg']), np.asarray(fixture['eye_offset_xyz']) / 3.74,
        np.asarray([fixture['query_time']]),
    )
    np.testing.assert_allclose(eye_mm[0] * 3.74, fixture['gui_eye_xyz'], rtol=0, atol=1e-10)
