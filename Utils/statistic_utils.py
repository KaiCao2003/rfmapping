import numpy as np
from collections.abc import Mapping
from scipy.stats import circmean, pearsonr, permutation_test, rankdata


def rayleigh_uniformity(angles_deg):
    """Test circular uniformity using one unweighted angle per unit.

    ``r`` is the mean resultant length; ``z = n * r**2``. The approximate
    p-value follows CircStat's ``circ_rtest`` (Zar, equation 27.4), which is
    sensitive to a single preferred direction. Nonfinite angles are omitted;
    fewer than three observations produce a NaN p-value.
    """
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    n = int(angles_deg.size)
    if n == 0:
        return {"n": 0, "r": np.nan, "z": np.nan, "p": np.nan}
    radians = np.deg2rad(angles_deg % 360)
    r = min(float(abs(np.mean(np.exp(1j * radians)))), 1.0)
    z = n * r ** 2
    p = float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * r) ** 2))
                     - (1 + 2 * n))) if n >= 3 else np.nan
    return {"n": n, "r": r, "z": z, "p": p}


def circular_distance_matrix_deg(angles_deg):
    """Return shortest angular distances, in degrees, between every pair."""
    angles_deg = np.asarray(angles_deg, dtype=float)
    delta = angles_deg[:, None] - angles_deg[None, :]
    return np.abs((delta + 180.0) % 360.0 - 180.0)


def circular_correlation(alpha, beta):
    """Correlate sine-centered angles supplied in radians (JS coefficient).

    Preserve SciPy's circular-mean convention for balanced samples. A zero
    resultant has no unique mean direction, so this coefficient is unstable
    there; Fisher–Lee correlation does not require a mean direction.
    """
    alpha_centered = np.sin(alpha - circmean(alpha))
    beta_centered = np.sin(beta - circmean(beta))
    return float(pearsonr(alpha_centered, beta_centered).statistic)


def _paired_radians(alpha_deg, beta_deg):
    alpha = np.asarray(alpha_deg, dtype=float)
    beta = np.asarray(beta_deg, dtype=float)
    if alpha.ndim != 1 or beta.shape != alpha.shape or alpha.size < 3:
        raise ValueError("At least three aligned one-dimensional angle pairs are required")
    if not np.isfinite(alpha).all() or not np.isfinite(beta).all():
        raise ValueError("Paired angles must be finite")
    return np.deg2rad(alpha % 360.), np.deg2rad(beta % 360.)


def _fisher_lee_components(alpha, beta):
    pairs = np.triu_indices(alpha.size, k=1)
    alpha_sine = np.sin(alpha[pairs[0]] - alpha[pairs[1]])
    beta_sine_matrix = np.sin(beta[:, None] - beta[None, :])
    beta_sine = beta_sine_matrix[pairs]
    alpha_norm, beta_norm = np.linalg.norm(alpha_sine), np.linalg.norm(beta_sine)
    # Identical or exactly antipodal angles have no defined T-linear correlation.
    tolerance = 8 * np.finfo(float).eps * np.sqrt(alpha_sine.size)
    if min(alpha_norm, beta_norm) <= tolerance:
        return None, pairs, alpha_sine, beta_sine_matrix, None
    denominator = alpha_norm * beta_norm
    rho = float(np.clip(np.dot(alpha_sine, beta_sine) / denominator, -1., 1.))
    return rho, pairs, alpha_sine, beta_sine_matrix, denominator


def fisher_lee_correlation(hd_deg, rf_ego_deg):
    """Return signed Fisher–Lee circular correlation, or None if undefined.

    Inputs are paired angles in degrees. Pairwise sine differences avoid an
    estimated mean direction; a rotated reflection has rho=-1 even for uniform
    angle samples. See https://doi.org/10.1093/biomet/70.2.327.
    """
    alpha, beta = _paired_radians(hd_deg, rf_ego_deg)
    return _fisher_lee_components(alpha, beta)[0]


def fisher_lee_statistics(
    alpha_deg, beta_deg, *, n_permutations=10_000, random_seed=1, test=True,
):
    """Return Fisher–Lee rho and an optional two-sided pairing-permutation p.

    The caller decides whether inference is appropriate for its selected
    cohort. ``test=False`` returns descriptive rho without drawing permutations.
    Undefined correlations return None for both rho and p. Pairwise sine
    differences are cached so each shuffle only reorders existing values.
    """
    alpha, beta = _paired_radians(alpha_deg, beta_deg)
    if n_permutations < 0 or int(n_permutations) != n_permutations:
        raise ValueError("n_permutations must be a nonnegative integer")
    rho, pairs, alpha_sine, beta_sine_matrix, denominator = _fisher_lee_components(alpha, beta)
    p_value = None
    performed = 0
    if test and rho is not None and n_permutations:
        rng = np.random.default_rng(random_seed)
        exceedances = 0
        for start in range(0, int(n_permutations), 256):
            count = min(256, int(n_permutations) - start)
            order = rng.permuted(np.broadcast_to(np.arange(alpha.size), (count, alpha.size)), axis=1)
            shuffled_sine = beta_sine_matrix[order[:, pairs[0]], order[:, pairs[1]]]
            shuffled_rho = (shuffled_sine @ alpha_sine) / denominator
            exceedances += int(np.sum(abs(shuffled_rho) >= abs(rho) - 1e-14))
        performed = int(n_permutations)
        p_value = (exceedances + 1.) / (performed + 1.)
    return {"rho": rho, "permutation_p": p_value, "n_permutations": performed}


def compare_direction_angles(
    reference_deg, matched_deg, *, n_permutations=10_000, random_seed=1,
):
    """Compare aligned unit directions with circular and Mantel tests.

    Permute unit pairings, not individual distance entries. Mantel statistics
    use unique off-diagonal pairs; plots may show the full ordered-pair matrix.
    The reported offset is matched minus reference, wrapped to [-180, 180].
    """
    reference_deg = np.asarray(reference_deg, dtype=float)
    matched_deg = np.asarray(matched_deg, dtype=float)
    if reference_deg.ndim != 1 or matched_deg.shape != reference_deg.shape:
        raise ValueError("direction arrays must be one-dimensional and aligned by unit")
    if reference_deg.size < 3:
        raise ValueError("direction comparisons require at least three shared units")
    if not np.all(np.isfinite(reference_deg)) or not np.all(np.isfinite(matched_deg)):
        raise ValueError("direction comparisons require finite peak angles")

    def permutation_p(values, statistic, *, batch):
        result = permutation_test(
            (values,), statistic, permutation_type="pairings", vectorized=True,
            n_resamples=n_permutations, alternative="greater",
            rng=np.random.default_rng(random_seed), batch=batch,
        )
        return float(result.pvalue)

    reference_rad = np.deg2rad(reference_deg)
    matched_rad = np.deg2rad(matched_deg)
    rho_circ = circular_correlation(reference_rad, matched_rad)
    reference_centered = np.sin(reference_rad - circmean(reference_rad))

    def circular_statistic(shuffled, axis=-1):
        # Recompute the mean per shuffle: nearly balanced angles are sensitive
        # to floating-point summation order when their resultant is near zero.
        centered = np.sin(shuffled - circmean(shuffled, axis=axis, keepdims=True))
        return np.abs(pearsonr(reference_centered, centered, axis=axis).statistic)

    p_circ = permutation_p(
        matched_rad, circular_statistic, batch=256,
    )
    residual_vector = np.mean(np.exp(1j * (matched_rad - reference_rad)))

    upper_triangle = np.triu_indices(reference_deg.size, k=1)
    reference_distances = circular_distance_matrix_deg(reference_deg)[upper_triangle]
    reference_ranks = rankdata(reference_distances)
    matched_distances = circular_distance_matrix_deg(matched_deg)
    observed_ranks = rankdata(matched_distances[upper_triangle])
    mantel_rho = float(np.corrcoef(reference_ranks, observed_ranks)[1, 0])
    reference_centered_ranks = reference_ranks - reference_ranks.mean()
    reference_norm = np.linalg.norm(reference_centered_ranks)
    matched_ranks = None
    if np.array_equal(matched_distances, matched_distances.T):
        matched_ranks = np.zeros_like(matched_distances)
        centered_ranks = observed_ranks - observed_ranks.mean()
        matched_ranks[upper_triangle] = centered_ranks
        matched_ranks[upper_triangle[::-1]] = matched_ranks[upper_triangle]
        matched_norm = np.linalg.norm(centered_ranks)

    def mantel_statistic(order, axis=-1):
        pairs = order[..., upper_triangle[0]], order[..., upper_triangle[1]]
        if matched_ranks is None:
            # Floating modulo can make reverse distances differ at near ties.
            # Preserve those directed values and re-rank each permuted sample.
            ranks = rankdata(matched_distances[pairs], axis=axis)
            ranks -= ranks.mean(axis=axis, keepdims=True)
            norm = np.linalg.norm(ranks, axis=axis)
        else:
            ranks = matched_ranks[pairs]
            norm = matched_norm
        rho = np.sum(reference_centered_ranks * ranks, axis=axis) / (reference_norm * norm)
        return np.abs(np.clip(rho, -1, 1))

    unit_order = np.arange(reference_deg.size)
    # Bound the temporary permutation-by-pair arrays as the unit count grows.
    batch = max(1, min(256, 1_000_000 // reference_ranks.size))
    mantel_p = permutation_p(unit_order, mantel_statistic, batch=batch)
    unit_count = int(reference_deg.size)
    return {
        "same_unit": {
            "circular_correlation_rho": rho_circ,
            "permutation_p": p_circ,
            "one_to_one_offset_deg": float(np.rad2deg(np.angle(residual_vector))),
            "alignment_resultant": float(abs(residual_vector)),
            "unit_count": unit_count,
        },
        "pairwise": {
            "spearman_mantel_rho": mantel_rho,
            "permutation_p": mantel_p,
            "unit_count": unit_count,
            "ordered_pair_count": unit_count ** 2,
        },
    }


def basic_statistics(target: list | np.ndarray) -> None:
    """Print min, max, mean, median and std of a list or numpy array."""

    if isinstance(target, list):
        target = np.array(target)

    print(f"min: {min(target)}")
    print(f"max: {max(target)}")
    print(f"mean: {np.mean(target)}")
    print(f"median: {np.median(target)}")
    print(f"std: {np.std(target)}")


def select_by_indices(index_dict: dict, data_dict: dict | list | np.ndarray, *, pre_range: int = 0,
                      post_range: int = 0) -> dict:
    """
    index_dict: the 'a' (index) dict (>=2 layers)
    data_dict:  the 'x' (data) dict (exactly one layer shallower than index_dict)

    Behavior:
      - For overlapping layers, keys must match.
      - At the leaf: data_dict has a list; index_dict has a dict of one-or-more keys,
        each mapping to a list of integer indices. Those leaf keys (e.g. 'c', 'c1', 'b2')
        are preserved in the output and each becomes the selected list from data_dict's list.
    """

    def _to_list(seq: list | np.ndarray) -> list:
        if isinstance(seq, np.ndarray):
            return seq.tolist()
        return list(seq)

    def pick_windows(indices_list: list, base_seq: list | np.ndarray) -> list:
        base_list = _to_list(base_seq)
        n = len(base_list)
        out: list = []
        for i in indices_list:
            if not isinstance(i, int):
                out.append(None)
                continue
            if i < 0 or i >= n:
                out.append(None)
                continue
            start = max(0, i - pre_range)
            end = min(n - 1, i + post_range)
            # end is inclusive; Python slice needs end+1
            out.append(base_list[start:end + 1])
        return out

    # Recurse through dict layers
    if isinstance(data_dict, dict):
        # For each shared key, recurse, passing along the ranges
        return {
            k: select_by_indices(index_dict[k], v, pre_range=pre_range, post_range=post_range)
            for k, v in data_dict.items()
        }

    # Leaf: index_dict is a mapping {leaf_name: [indices]}
    result_leaf: dict = {}
    for leaf_name, indices in index_dict.items():
        result_leaf[leaf_name] = pick_windows(indices, data_dict)

    return result_leaf


def _find_innermost_dicts(nested_dict):
    """
    Recursively traverse a nested dictionary and collect all 'innermost' dictionaries.
    An innermost dictionary is one whose values are not dictionaries themselves.
    """
    innermost_dicts = []

    def _traverse(current_value):
        if isinstance(current_value, Mapping):
            # If all values are non-dict, treat as a leaf
            if current_value and all(not isinstance(v, Mapping) for v in current_value.values()):
                innermost_dicts.append(current_value)
            else:
                # Otherwise, keep going deeper
                for sub_value in current_value.values():
                    _traverse(sub_value)

    _traverse(nested_dict)
    return innermost_dicts


def collect_innermost_fields(nested_data, *, fill_value=None):
    """
    Traverse an arbitrarily nested dictionary and aggregate all innermost (leaf-level)
    dictionaries into a combined structure where each key maps to a list of its values
    across all leaves.

    Parameters
    ----------
    nested_data : dict
        The input dictionary that may contain multiple layers of nested dictionaries.
    fill_value : any, optional
        A value to use if a specific key is missing in some innermost dictionaries.

    Returns
    -------
    dict
        A dictionary mapping each leaf key to a list of its collected values.
    """
    innermost_dicts = _find_innermost_dicts(nested_data)
    if not innermost_dicts:
        return {}

    # Collect all keys that appear in any innermost dict
    all_leaf_keys = set().union(*(leaf.keys() for leaf in innermost_dicts))

    # Aggregate values for each key
    aggregated_result = {
        key: [leaf.get(key, fill_value) for leaf in innermost_dicts]
        for key in all_leaf_keys
    }

    return aggregated_result
