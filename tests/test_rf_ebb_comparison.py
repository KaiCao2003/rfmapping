import numpy as np
import pandas as pd

from EBB.profiles import boundary_band_profiles
from EBB.ebc_analysis import rates_from_counts
from Utils.statistic_utils import compare_direction_angles
from scripts.export_m22_ebb_samples import mean_band_profiles


def test_native_grids_and_missing_exposure():
    edges = np.array([0., 4., 8., 12., 16., 20.])
    rates = np.arange(60 * 5, dtype=float).reshape(1, 60, 5)
    occupancy = np.ones((60, 5))
    occupancy[:2] = 0
    rates[:, 2:4] = 0
    result = boundary_band_profiles(rates, occupancy, np.arange(0, 361, 6), edges, pd.Index(['u']))
    assert np.isnan(result['all'].iloc[0, 0])
    assert result['all'].iloc[0, 1] == 0
    assert result['0-8'].iloc[0, 2] == rates[0, 4:6, :2].mean()
    coarse = rates.reshape(1, 30, 2, 5).mean(axis=2)
    for band, values in mean_band_profiles(coarse, edges).items():
        np.testing.assert_allclose(values, mean_band_profiles(rates, edges)[band])


def test_permutations_do_not_move_units_between_recordings():
    # Each recording has one constant EBB preference, so within-recording
    # permutations cannot change a statistic even though the pooled map varies.
    result = compare_direction_angles([0, 20, 40, 180, 200, 220], [10, 10, 10, 190, 190, 190],
        permutation_groups=['a', 'a', 'a', 'b', 'b', 'b'], n_permutations=97)
    assert result['pairwise']['permutation_p'] == 1
    assert result['same_unit']['permutation_p'] == 1


def test_constant_pair_distances_have_no_mantel_p_value():
    for reference, matched in (([0, 20, 40, 60], [10] * 4), ([10] * 4, [0, 20, 40, 60])):
        result = compare_direction_angles(reference, matched, n_permutations=97)
        assert np.isnan(result['pairwise']['spearman_mantel_rho'])
        assert np.isnan(result['pairwise']['permutation_p'])


def test_anisotropic_smoothing_preserves_unit_and_angle_axes():
    counts = np.zeros((2, 6, 8))
    counts[0, 2, 4] = 10
    rates = rates_from_counts(counts, np.ones((6, 8)), sigma=(0, 1))
    assert np.all(rates[1] == 0)
    assert np.all(rates[0, [0, 1, 3, 4, 5]] == 0)
    assert 0 < rates[0, 2, 3] < rates[0, 2, 4] < 10
