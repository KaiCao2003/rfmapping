import numpy as np
import pytest
from scipy.stats import permutation_test, spearmanr

from Utils.statistic_utils import (
    circular_distance_matrix_deg, compare_direction_angles, rayleigh_uniformity,
)


def test_rayleigh_matches_circstat_reference_sample():
    # https://github.com/circstat/pycircstat/blob/master/tests/test_tests.py
    radians = np.array([
        -0.94904375, 0.26575165, -0.03226759, 1.98389239, 2.0084918,
        0.56965871, -0.19199522, -0.33080002, -0.03141245, 1.18560637,
        4.31138236, 4.84776075, 2.10336819, 3.45465972, 4.64572697,
        5.80976322, 6.27160926, 0.67979652, 1.33776004, 0.58059308,
    ])
    result = rayleigh_uniformity(np.rad2deg(radians))
    assert result["n"] == 20
    assert result["z"] == pytest.approx(3.1678, abs=0.001)
    assert result["p"] == pytest.approx(0.0400878, abs=0.001)


def test_rayleigh_distinguishes_uniform_from_concentrated_directions():
    uniform = rayleigh_uniformity(np.arange(24) * 15)
    concentrated = rayleigh_uniformity(np.full(20, 83.0))
    assert uniform["r"] == pytest.approx(0, abs=1e-15)
    assert uniform["p"] == pytest.approx(1)
    assert concentrated["r"] == pytest.approx(1)
    assert concentrated["z"] == pytest.approx(20)
    assert concentrated["p"] == pytest.approx(np.exp(-32), rel=1e-12, abs=0)


def test_rayleigh_does_not_detect_balanced_opposing_modes():
    result = rayleigh_uniformity([0] * 20 + [180] * 20)
    assert result["r"] == pytest.approx(0, abs=1e-15)
    assert result["p"] == pytest.approx(1)


def test_rayleigh_ignores_missing_angles_and_is_invariant_to_wrapping():
    angles = np.array([-150, -25, 5, 10, 10, 37, 58])
    expected = rayleigh_uniformity(angles)
    shifted = angles + np.array([-720, 360, 0, 1080, -360, 720, -1080])
    assert rayleigh_uniformity(shifted) == pytest.approx(expected)
    assert rayleigh_uniformity(angles + 71) == pytest.approx(expected)
    assert rayleigh_uniformity([*angles, np.nan, np.inf]) == pytest.approx(expected)


@pytest.mark.parametrize("angles", [[], [np.nan], [10], [10, 20, np.nan]])
def test_rayleigh_does_not_report_a_p_value_for_too_few_units(angles):
    result = rayleigh_uniformity(angles)
    assert result["n"] == np.count_nonzero(np.isfinite(angles))
    assert np.isnan(result["p"])


def test_circular_distances_use_shortest_path_across_zero():
    np.testing.assert_array_equal(
        circular_distance_matrix_deg([350, 10, 170]),
        [[0, 20, 180], [20, 0, 160], [180, 160, 0]],
    )


def test_direction_comparison_recovers_a_known_rotation():
    reference = np.array([7, 35, 89, 152, 223, 304])
    matched = (reference + 67) % 360
    result = compare_direction_angles(reference, matched, n_permutations=99, random_seed=4)
    same_unit = result["same_unit"]
    pairwise = result["pairwise"]
    assert same_unit["circular_correlation_rho"] == pytest.approx(1)
    assert same_unit["one_to_one_offset_deg"] == pytest.approx(67)
    assert same_unit["alignment_resultant"] == pytest.approx(1)
    assert "alignment_permutation_p" not in same_unit
    assert same_unit["permutation_p"] <= 0.05
    assert pairwise["spearman_mantel_rho"] == pytest.approx(1)
    assert pairwise["permutation_p"] <= 0.05
    assert pairwise["unit_count"] == 6
    assert pairwise["ordered_pair_count"] == 36


def test_direction_comparison_preserves_results_across_angle_wrapping():
    reference = np.array([7, 35, 89, 152, 223, 304])
    matched = np.array([180, 13, 273, 142, 320, 77])
    options = {"n_permutations": 99, "random_seed": 2}
    result = compare_direction_angles(reference, matched, **options)
    wrapped = compare_direction_angles(reference + 360, matched - 360, **options)
    for group in result:
        assert wrapped[group] == pytest.approx(result[group])


@pytest.mark.parametrize("n_permutations", [79, 1000])
@pytest.mark.parametrize("matched", [
    np.array([0., 12., 12., 36., 48., 60.]),
    np.random.default_rng(4).uniform(-720, 720, 6),
    np.arange(6) * 12. + np.random.default_rng(71).uniform(-1e-14, 1e-14, 6),
])
def test_cached_mantel_preserves_legacy_ranks_and_permutation_p(matched, n_permutations):
    reference = np.array([7., 35., 89., 152., 223., 304.])
    upper = np.triu_indices(len(reference), 1)
    reference_distances = circular_distance_matrix_deg(reference)[upper]

    def statistic(values):
        return float(spearmanr(
            reference_distances, circular_distance_matrix_deg(values)[upper],
        ).statistic)

    expected = permutation_test(
        (matched,), lambda values: abs(statistic(values)),
        permutation_type="pairings", vectorized=False,
        n_resamples=n_permutations, alternative="greater", rng=np.random.default_rng(17),
    )
    result = compare_direction_angles(
        reference, matched, n_permutations=n_permutations, random_seed=17,
    )["pairwise"]
    assert result["spearman_mantel_rho"] == statistic(matched)
    assert result["permutation_p"] == expected.pvalue
