import numpy as np
from scipy.sparse import csr_matrix

from EBB.joint import hd_boundary_joint_maps


def test_joint_map_conditions_frame_spikes_and_exposure_on_both_angles():
    projection = csr_matrix((np.ones(8), ([0, 3, 1, 2, 0, 2, 1, 3], [0, 0, 1, 1, 2, 2, 3, 3])),
                            shape=(4, 4))
    result = hd_boundary_joint_maps(projection, [2, 0, 4, 1], np.arange(4), [0, 0, 180, 360],
        [0, 180, 360], [0, 1, 2], {'near': (0, 1), 'far': (1, 2)}, hd_bins=2, smoothing_deg=0)
    np.testing.assert_array_equal(result['spike_maps'].sum(axis=0), [[6, 1], [4, 3]])
    np.testing.assert_array_equal(result['occupancy_maps'].sum(axis=0), np.full((2, 2), 2))
    np.testing.assert_array_equal(result['bands']['near']['raw_hz'], [[2, 0], [4, 4]])
    np.testing.assert_allclose(result['bands']['far']['raw_hz'], [[.5, 1.5], [np.nan, np.nan]])
    np.testing.assert_allclose(result['bands']['far']['hz'], result['bands']['far']['raw_hz'])


def test_joint_smoothing_keeps_unvisited_missing_and_constant_rate_constant():
    projection = csr_matrix((np.ones(4), ([0, 1, 0, 1], [0, 0, 1, 1])), shape=(2, 2))
    result = hd_boundary_joint_maps(projection, [2, 2], [0, 1], [0, 0],
        [0, 180, 360], [0, 1], {'all': (0, 1)}, hd_bins=2, smoothing_deg=90)
    np.testing.assert_allclose(result['bands']['all']['hz'], [[2, 2], [np.nan, np.nan]])
