import numpy as np
from scipy.spatial.transform import Rotation

from Utils.motive_pose import motive_z_yaw_degrees


def test_z_yaw_cardinal_directions_in_euler_and_quaternion():
    euler = np.array([[0., 0., angle] for angle in (0., 90., 180., -90.)])
    expected = [0., 90., 180., 270.]
    np.testing.assert_allclose(motive_z_yaw_degrees(euler), expected)
    quaternion = Rotation.from_euler("XYZ", euler, degrees=True).as_quat()
    np.testing.assert_allclose(motive_z_yaw_degrees(quaternion), expected, atol=1e-12)
    np.testing.assert_allclose(motive_z_yaw_degrees(-quaternion), expected, atol=1e-12)


def test_z_yaw_retains_motive_xyz_with_tilt():
    euler = np.array([[25., 35., 70.], [-40., 60., -120.]])
    quaternion = Rotation.from_euler("XYZ", euler, degrees=True).as_quat()
    np.testing.assert_allclose(motive_z_yaw_degrees(quaternion), [70., 240.], atol=1e-12)


def test_missing_quaternions_remain_missing():
    quaternion = np.array([[0., 0., 0., 0.], [np.nan, np.nan, np.nan, np.nan]])
    assert np.isnan(motive_z_yaw_degrees(quaternion)).all()
