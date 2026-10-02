import numpy as np
import pytest
from scipy.stats import circmean, pearsonr, permutation_test, spearmanr

from Utils import statistic_utils


def _scalar_comparison(reference, matched, n_permutations, seed):
    """Keep the pre-vectorization statistics as an independent reference."""
    reference_rad = np.deg2rad(reference)
    matched_rad = np.deg2rad(matched)

    def circular(shuffled):
        reference_centered = np.sin(reference_rad - circmean(reference_rad))
        matched_centered = np.sin(shuffled - circmean(shuffled))
        return float(pearsonr(reference_centered, matched_centered).statistic)

    triangle = np.triu_indices(len(reference), k=1)

    def distances(angles):
        delta = angles[:, None] - angles[None, :]
        return np.abs((delta + 180.0) % 360.0 - 180.0)

    reference_distances = distances(reference)[triangle]
    matched_distances = distances(matched)

    def mantel(order):
        selected = matched_distances[order[triangle[0]], order[triangle[1]]]
        return float(spearmanr(reference_distances, selected).statistic)

    def pvalue(values, statistic):
        return float(permutation_test(
            (values,), lambda shuffled: abs(statistic(shuffled)),
            permutation_type="pairings", vectorized=False,
            n_resamples=n_permutations, alternative="greater",
            rng=np.random.default_rng(seed),
        ).pvalue)

    order = np.arange(len(reference))
    return (
        circular(matched_rad), pvalue(matched_rad, circular),
        mantel(order), pvalue(order, mantel),
    )


@pytest.mark.parametrize(
    "reference,matched,n_permutations,seed",
    [
        pytest.param(
            [0, 45, 110], [10, 160, 280], 10_000, 1,
            id="exact-three",
        ),
        pytest.param(
            [0, 30, 80, 140, 220, 300], [20, 300, 200, 170, 40, 80],
            1_000, 1, id="exact-six",
        ),
        pytest.param(
            [0, 30, 60, 120, 180, 300],
            [0.1, 30.1, 60.1, 120.1, 180.1, 300.1],
            720, 7, id="directed-floating-near-ties",
        ),
        pytest.param(
            [0, 0, 45, 45, 90, 90, 180, 180, 270],
            [180, 45, 45, 0, 180, 0, 90, 270, 90],
            257, 17, id="integer-ties",
        ),
        pytest.param(
            np.random.default_rng(3).uniform(0, 360, 9),
            np.random.default_rng(4).uniform(0, 360, 9),
            257, 37, id="floating-monte-carlo",
        ),
        pytest.param(
            [15, 55, 100, 120, 150, 200, 300, 310],
            np.arange(8) * 45, 257, 1, id="balanced-matched-circular-mean",
        ),
        pytest.param(
            np.arange(8) * 45,
            [15, 55, 100, 120, 150, 200, 300, 310],
            257, 1, id="balanced-reference-circular-mean",
        ),
        pytest.param(
            np.random.default_rng(8).integers(0, 360, 97),
            np.random.default_rng(9).integers(0, 360, 97),
            257, 23, id="symmetric-mantel-across-batches",
        ),
        pytest.param(
            np.random.default_rng(10).uniform(0, 360, 65),
            np.random.default_rng(11).uniform(0, 360, 65),
            257, 31, id="directed-mantel-across-batches",
        ),
    ],
)
def test_direction_comparison_matches_scalar_statistics(
    reference, matched, n_permutations, seed,
):
    reference = np.asarray(reference, dtype=float)
    matched = np.asarray(matched, dtype=float)
    expected_circular, expected_circular_p, expected_mantel, expected_mantel_p = (
        _scalar_comparison(reference, matched, n_permutations, seed)
    )

    actual = statistic_utils.compare_direction_angles(
        reference, matched, n_permutations=n_permutations, random_seed=seed,
    )

    same_unit = actual["same_unit"]
    pairwise = actual["pairwise"]
    assert same_unit["circular_correlation_rho"] == expected_circular
    assert same_unit["permutation_p"] == expected_circular_p
    assert pairwise["spearman_mantel_rho"] == expected_mantel
    assert pairwise["permutation_p"] == expected_mantel_p
    assert same_unit["unit_count"] == pairwise["unit_count"] == len(reference)
    assert pairwise["ordered_pair_count"] == len(reference) ** 2


def test_direction_comparison_seed_is_reproducible():
    reference = np.random.default_rng(10).uniform(0, 360, 19)
    matched = np.random.default_rng(11).uniform(0, 360, 19)

    first = statistic_utils.compare_direction_angles(
        reference, matched, n_permutations=257, random_seed=73,
    )
    second = statistic_utils.compare_direction_angles(
        reference, matched, n_permutations=257, random_seed=73,
    )

    assert first == second


def test_fisher_lee_permutation_matches_independent_pairwise_definition():
    alpha = np.array([2., 23., 67., 102., 148., 213., 255., 307., 346.])
    beta = np.array([17., 309., 281., 212., 91., 72., 167., 48., 338.])

    def pairwise_rho(first, second):
        first = np.deg2rad(first)
        second = np.deg2rad(second)
        first_diff, second_diff = [], []
        for i in range(len(first) - 1):
            for j in range(i + 1, len(first)):
                first_diff.append(np.sin(first[i] - first[j]))
                second_diff.append(np.sin(second[i] - second[j]))
        numerator = sum(a * b for a, b in zip(first_diff, second_diff))
        denominator = np.sqrt(sum(a * a for a in first_diff) * sum(b * b for b in second_diff))
        return numerator / denominator

    observed = pairwise_rho(alpha, beta)
    rng = np.random.default_rng(17)
    shuffled = [pairwise_rho(alpha, rng.permutation(beta)) for _ in range(257)]
    expected_p = (np.sum(np.abs(shuffled) >= abs(observed) - 1e-14) + 1) / 258
    actual = statistic_utils.fisher_lee_statistics(
        alpha, beta, n_permutations=257, random_seed=17,
    )
    assert actual["rho"] == pytest.approx(observed, abs=1e-15)
    assert actual["permutation_p"] == expected_p
    assert actual["n_permutations"] == 257


@pytest.mark.parametrize("angles", [np.zeros(8), np.tile([0., 180.], 4)])
def test_fisher_lee_undefined_angles_have_no_inference(angles):
    actual = statistic_utils.fisher_lee_statistics(angles, np.arange(8) * 20.)
    assert actual == {"rho": None, "permutation_p": None, "n_permutations": 0}


def test_fisher_lee_descriptive_mode_never_draws_permutations(monkeypatch):
    def unexpected_rng(*args, **kwargs):
        raise AssertionError("Descriptive statistics must not draw permutations")

    monkeypatch.setattr(np.random, "default_rng", unexpected_rng)
    angles = np.arange(8) * 45.
    actual = statistic_utils.fisher_lee_statistics(angles, 30. - angles, test=False)
    assert actual == {"rho": -1., "permutation_p": None, "n_permutations": 0}
    with pytest.raises(ValueError, match="nonnegative integer"):
        statistic_utils.fisher_lee_statistics(angles, angles, n_permutations=-1, test=False)


def test_fisher_lee_preserves_small_but_defined_variation():
    angles = np.rad2deg(np.arange(8) * 1e-6)
    assert statistic_utils.fisher_lee_correlation(
        hd_deg=angles, rf_ego_deg=2. * angles,
    ) == pytest.approx(1.)
