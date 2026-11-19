import numpy as np
import pytest
from scipy.sparse import csr_matrix

from EBB.quality import boundary_map_quality
from RF.free_moving import free_moving_exposure


def test_boundary_quality_preserves_missing_and_uses_temporal_halves():
    bins = [0, 1, 2, 0, 0, 1, 2, 0]
    projection = csr_matrix((np.ones(8), (bins, np.arange(8))), shape=(4, 8))
    counts = csr_matrix([[0, 2, 4, 0, 0, 4, 8, 0], [0] * 8])
    held = np.array([False, False, True, False] * 2)
    result = boundary_map_quality(projection, counts, np.arange(8), held, (2, 2), sigma=0)
    np.testing.assert_array_equal(result['occupancy'], [[4, 2], [2, 0]])
    np.testing.assert_array_equal(result['spike_maps'][0], [[0, 6], [12, 0]])
    np.testing.assert_allclose(result['rates'][0], [[0, 3], [6, np.nan]])
    np.testing.assert_allclose(result['half_rates'][:, 0, 0, 1], [2, 4])
    assert np.isnan(result['unheld_rates'][0, 1, 0])
    assert result['metrics'][0]['half_r'] == pytest.approx(1)
    assert result['metrics'][0]['first_half_hz'] == pytest.approx(1.5)
    assert result['metrics'][0]['second_half_hz'] == pytest.approx(3)
    assert result['metrics'][0]['half_shared_bins'] == 3
    assert np.isnan(result['metrics'][1]['half_r'])
    assert np.isnan(result['metrics'][1]['descriptive_score'])


def test_boundary_quality_rejects_misaligned_frames():
    with pytest.raises(ValueError, match='frame axis'):
        boundary_map_quality(csr_matrix((4, 3)), csr_matrix((1, 2)),
                             np.arange(3), np.zeros(3, bool), (2, 2), sigma=0)


def test_free_moving_counts_are_independent_of_trial_durations_and_wrap():
    trials = [dict(Square_PositionX=x, Square_PositionY=0, Square_Size=20,
                   Square_Luminance=lum) for x, lum in [(15, 1), (15, 1), (-175, 1), (15, 0)]]
    result = free_moving_exposure(
        trials, [0, .1, .4, .6, 1], np.zeros((4, 2)), np.zeros(4),
        dict(screen_diamter=200, screen_center=0),
        n_x_bins=12, screen_width_px=360, screen_deg=360,
    )
    expected_counts, expected_seconds = np.zeros(12), np.zeros(12)
    expected_counts[[0, 6, 11]] = [1, 2, 1]
    expected_seconds[[0, 6, 11]] = [.2, .4, .2]
    np.testing.assert_array_equal(result['presentation_counts'], expected_counts)
    np.testing.assert_allclose(result['occupancy_s'], expected_seconds)
    assert not result['trial_bin_mask'][-1].any()


def test_free_moving_geometry_uses_xy_and_heading():
    trial = dict(Square_PositionX=15, Square_PositionY=0, Square_Size=20, Square_Luminance=1)
    result = free_moving_exposure(
        [trial, trial], [0, 1, 2], np.array([[50, 0], [50, 0]]), [0, 90],
        dict(screen_diamter=200, screen_center=0),
        n_x_bins=12, screen_width_px=360, screen_deg=360,
    )
    np.testing.assert_array_equal(np.flatnonzero(result['trial_bin_mask'][0]), [6, 7])
    np.testing.assert_array_equal(np.flatnonzero(result['trial_bin_mask'][1]), [9, 10])
