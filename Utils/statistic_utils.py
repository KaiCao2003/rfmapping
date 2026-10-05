import numpy as np
import csv
from pathlib import Path
from collections.abc import Mapping, Sequence

import pandas as pd
from scipy.ndimage import gaussian_filter1d
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


def safe_std(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size <= 1:
        return 0.0
    return float(np.nanstd(array, ddof=0))


def safe_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    return float(np.nanmean(array))


def _validate_index_interval_groups(
        labels: list[str],
        data_list: list[list[tuple[int, int]]],
) -> list[list[tuple[int, int]]]:
    if len(labels) != len(data_list):
        raise ValueError("labels and data_list must have the same length.")

    validated_groups: list[list[tuple[int, int]]] = []
    for phase_label, intervals in zip(labels, data_list):
        validated_intervals: list[tuple[int, int]] = []
        for trial_index, (start, end) in enumerate(intervals, start=1):
            start_index = int(start)
            end_index = int(end)
            if start_index > end_index:
                raise ValueError(
                    f"Invalid interval for {phase_label} trial {trial_index}: "
                    f"start ({start_index}) is greater than end ({end_index})."
                )
            validated_intervals.append((start_index, end_index))
        validated_groups.append(validated_intervals)

    return validated_groups


def _validate_numeric_interval_groups(
        labels: list[str],
        data_list: list[list[tuple[int | float, int | float]]],
) -> list[list[tuple[float, float]]]:
    if len(labels) != len(data_list):
        raise ValueError("labels and data_list must have the same length.")

    validated_groups: list[list[tuple[float, float]]] = []
    for phase_label, intervals in zip(labels, data_list):
        validated_intervals: list[tuple[float, float]] = []
        for trial_index, interval in enumerate(intervals, start=1):
            if len(interval) != 2:
                raise ValueError(
                    f"Invalid interval for {phase_label} trial {trial_index}: "
                    f"expected a pair, got {interval!r}."
                )
            start_value = float(interval[0])
            end_value = float(interval[1])
            if not (np.isfinite(start_value) and np.isfinite(end_value)):
                raise ValueError(
                    f"Invalid interval for {phase_label} trial {trial_index}: "
                    f"({start_value}, {end_value}) contains a non-finite value."
                )
            if start_value > end_value:
                raise ValueError(
                    f"Invalid interval for {phase_label} trial {trial_index}: "
                    f"start ({start_value}) is greater than end ({end_value})."
                )
            validated_intervals.append((start_value, end_value))
        validated_groups.append(validated_intervals)

    return validated_groups


def _iter_event_rows(
        event_source: str | Path | Sequence[Mapping[str, object]] | object,
):
    if isinstance(event_source, (str, Path)):
        event_path = Path(event_source)
        with event_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Event CSV is missing a header row: {event_path}")
            yield from reader
        return

    if hasattr(event_source, "to_dict"):
        try:
            records = event_source.to_dict("records")
        except TypeError:
            records = event_source.to_dict(orient="records")
        for row in records:
            if not isinstance(row, Mapping):
                raise TypeError("event_source.to_dict('records') must return mapping rows.")
            yield dict(row)
        return

    if isinstance(event_source, Sequence) and not isinstance(event_source, (str, bytes, bytearray)):
        for row in event_source:
            if not isinstance(row, Mapping):
                raise TypeError("event_source sequences must contain mapping rows.")
            yield dict(row)
        return

    raise TypeError(
        "event_source must be a CSV path, a DataFrame-like object, or a sequence of mapping rows."
    )


def _is_truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and np.isnan(value):
            return False
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def summarize_head_turns_from_intervals(
        labels: list[str],
        data_list: list[list[tuple[int | float, int | float]]],
        event_source: str | Path | Sequence[Mapping[str, object]] | object,
        *,
        target_data: list | np.ndarray | None = None,
        event_start_column: str = "start_frame_actual",
        event_end_column: str = "end_frame",
        skip_column: str | None = "is_skipped",
        count_mode: str = "start",
        value_mode: str = "count",
        fps: float | None = None,
) -> dict[str, object]:
    """
    Count head-turn events inside phase/trial intervals and return plot-ready values.

    ``labels`` and ``data_list`` follow the same grouped-by-phase layout as
    ``summarize_phase_intervals``. Each event is taken from ``event_source``,
    which may be a CSV path or a DataFrame-like object.

    ``count_mode="start"`` counts events whose start timestamp falls inside each
    inclusive interval. ``count_mode="overlap"`` counts events that overlap the
    interval at all and therefore uses both ``event_start_column`` and
    ``event_end_column``.

    ``value_mode="count"`` returns raw counts per trial. ``value_mode="rate"``
    and ``value_mode="count_per_s"`` divide those counts by the interval duration
    in seconds and therefore require ``fps``.

    When ``target_data`` is provided, the plotted/statistical trial value is the
    mean of ``target_data`` restricted to head-turn event samples inside each
    trial interval. Raw event counts are still returned in ``phase_trial_counts``.
    """
    validated_groups = _validate_numeric_interval_groups(labels, data_list)

    normalized_count_mode = str(count_mode).strip().lower()
    if normalized_count_mode not in {"start", "overlap"}:
        raise ValueError("count_mode must be either 'start' or 'overlap'.")

    normalized_value_mode = str(value_mode).strip().lower()
    if normalized_value_mode not in {"count", "rate", "count_per_s"}:
        raise ValueError("value_mode must be 'count', 'rate', or 'count_per_s'.")
    if normalized_value_mode != "count":
        if fps is None:
            raise ValueError("fps is required when value_mode is 'rate' or 'count_per_s'.")
        if float(fps) <= 0:
            raise ValueError("fps must be positive when provided.")

    target_array: np.ndarray | None = None
    if target_data is not None:
        target_array = np.asarray(target_data, dtype=float)
        if target_array.ndim != 1:
            raise ValueError("target_data must be a 1D list or numpy array when provided.")

    event_starts: list[float] = []
    event_ends: list[float] = []
    event_intervals: list[tuple[float, float]] = []
    kept_event_count = 0
    for row in _iter_event_rows(event_source):
        if skip_column is not None and _is_truthy(row.get(skip_column)):
            continue
        if event_start_column not in row:
            raise KeyError(f"Missing event start column: {event_start_column}")

        start_value = float(row[event_start_column])
        if not np.isfinite(start_value):
            continue
        raw_end_value = row.get(event_end_column, start_value)
        end_value = float(raw_end_value)
        if not np.isfinite(end_value):
            end_value = start_value
        end_value = max(start_value, end_value)

        event_starts.append(start_value)
        event_ends.append(end_value)
        event_intervals.append((start_value, end_value))
        kept_event_count += 1

    event_start_array = np.sort(np.asarray(event_starts, dtype=float))
    event_end_array = np.sort(np.asarray(event_ends, dtype=float))
    event_intervals.sort(key=lambda pair: pair[0])

    phase_trial_counts: list[np.ndarray] = []
    phase_trial_values: list[np.ndarray] = []
    phase_trial_durations_s: list[np.ndarray] = []
    phase_trial_event_sample_counts: list[np.ndarray] = []
    phase_stats: list[dict[str, object]] = []

    for phase_label, intervals in zip(labels, validated_groups):
        count_values: list[int] = []
        phase_values: list[float] = []
        duration_values_s: list[float] = []
        event_sample_counts: list[int] = []

        for start_value, end_value in intervals:
            if normalized_count_mode == "start":
                left = np.searchsorted(event_start_array, start_value, side="left")
                right = np.searchsorted(event_start_array, end_value, side="right")
                event_count = int(right - left)
            else:
                started_by_end = np.searchsorted(event_start_array, end_value, side="right")
                ended_before_start = np.searchsorted(event_end_array, start_value, side="left")
                event_count = int(started_by_end - ended_before_start)

            count_values.append(event_count)

            if target_array is None:
                event_sample_counts.append(0)
                if normalized_value_mode == "count":
                    phase_values.append(float(event_count))
                    duration_values_s.append(np.nan)
                    continue

                duration_frames = end_value - start_value + 1.0
                duration_s = duration_frames / float(fps)
                duration_values_s.append(duration_s)
                phase_values.append(float(event_count / duration_s) if duration_s > 0 else np.nan)
                continue

            trial_start_index = int(np.ceil(start_value))
            trial_end_index = int(np.floor(end_value))
            if trial_start_index < 0 or trial_end_index >= target_array.size:
                raise IndexError(
                    f"Interval for {phase_label} is out of bounds for target_data: "
                    f"({trial_start_index}, {trial_end_index}) with target_data length {target_array.size}."
                )

            segment_bounds: list[tuple[int, int]] = []
            for event_start_value, event_end_value in event_intervals:
                if normalized_count_mode == "start":
                    if not (start_value <= event_start_value <= end_value):
                        continue
                else:
                    if event_end_value < start_value or event_start_value > end_value:
                        continue

                segment_start = max(trial_start_index, int(np.ceil(event_start_value)))
                segment_end = min(trial_end_index, int(np.floor(event_end_value)))
                if segment_start > segment_end:
                    continue

                segment_bounds.append((segment_start, segment_end))

            duration_values_s.append(np.nan)
            if not segment_bounds:
                event_sample_counts.append(0)
                phase_values.append(np.nan)
                continue

            merged_bounds: list[list[int]] = []
            for segment_start, segment_end in segment_bounds:
                if not merged_bounds or segment_start > merged_bounds[-1][1] + 1:
                    merged_bounds.append([segment_start, segment_end])
                else:
                    merged_bounds[-1][1] = max(merged_bounds[-1][1], segment_end)

            event_sample_counts.append(
                int(sum(segment_end - segment_start + 1 for segment_start, segment_end in merged_bounds))
            )
            event_values = np.concatenate(
                [target_array[segment_start:segment_end + 1] for segment_start, segment_end in merged_bounds]
            )
            finite_values = event_values[np.isfinite(event_values)]
            phase_values.append(float(np.nanmean(finite_values)) if finite_values.size else np.nan)

        count_array = np.asarray(count_values, dtype=int)
        value_array = np.asarray(phase_values, dtype=float)
        duration_array = np.asarray(duration_values_s, dtype=float)
        event_sample_count_array = np.asarray(event_sample_counts, dtype=int)

        phase_trial_counts.append(count_array)
        phase_trial_values.append(value_array)
        phase_trial_durations_s.append(duration_array)
        phase_trial_event_sample_counts.append(event_sample_count_array)
        phase_stats.append(
            {
                "label": phase_label,
                "n_trials": int(value_array.size),
                "mean": safe_mean(value_array),
                "std": safe_std(value_array),
                "total_count": int(count_array.sum()),
                "total_event_samples": int(event_sample_count_array.sum()),
            }
        )

    return {
        "labels": list(labels),
        "data_list": validated_groups,
        "phase_trial_counts": phase_trial_counts,
        "phase_trial_values": phase_trial_values,
        "phase_trial_durations_s": phase_trial_durations_s,
        "phase_trial_event_sample_counts": phase_trial_event_sample_counts,
        "phase_means": np.asarray([row["mean"] for row in phase_stats], dtype=float),
        "phase_stds": np.asarray([row["std"] for row in phase_stats], dtype=float),
        "phase_stats": phase_stats,
        "analysis_mode": "head_turn_target_mean" if target_array is not None else "head_turn_frequency",
        "count_mode": normalized_count_mode,
        "value_mode": normalized_value_mode,
        "event_start_column": event_start_column,
        "event_end_column": event_end_column,
        "kept_event_count": kept_event_count,
        "has_target_data": target_array is not None,
    }



def summarize_phase_intervals(labels, data_list, target_data):
    """Compute trial means on inclusive intervals, then phase means and SDs."""
    validated_groups = _validate_index_interval_groups(labels, data_list)
    target_array = np.asarray(target_data, dtype=float)
    if target_array.ndim != 1:
        raise ValueError("target_data must be a 1D list or numpy array.")
    phase_trial_values = []
    for phase_label, intervals in zip(labels, validated_groups):
        trial_values = []
        for trial_index, (start_index, end_index) in enumerate(intervals, start=1):
            if start_index < 0 or end_index >= target_array.size:
                raise IndexError(
                    f"Interval for {phase_label} trial {trial_index} is out of bounds: "
                    f"({start_index}, {end_index}) for target_data of length {target_array.size}."
                )
            interval_values = target_array[start_index:end_index + 1]
            finite_values = interval_values[np.isfinite(interval_values)]
            if finite_values.size:
                trial_values.append(float(np.mean(finite_values)))
        phase_trial_values.append(np.asarray(trial_values, dtype=float))
    return {
        "labels": list(labels), "phase_trial_values": phase_trial_values,
        "phase_means": np.asarray([safe_mean(values) for values in phase_trial_values]),
        "phase_stds": np.asarray([safe_std(values) for values in phase_trial_values]),
    }


def smooth_tuning_curve(curve, *, sigma=1.5):
    """Smooth a circular Series on its intact bin grid, retaining missing bins.

    Finite-weight convolution prevents missing observations from becoming zeros;
    a missing source bin stays missing in the result. Sigma is measured in bins.
    """
    values = curve.to_numpy(dtype=float)
    finite = np.isfinite(values)
    numerator = gaussian_filter1d(np.where(finite, values, 0.0), sigma=sigma, mode="wrap")
    weights = gaussian_filter1d(finite.astype(float), sigma=sigma, mode="wrap")
    smoothed = np.full(values.shape, np.nan)
    np.divide(numerator, weights, out=smoothed, where=finite & (weights > 0))
    return pd.Series(smoothed, index=curve.index, name=curve.name)


def tuning_curve_reference_radii(curve, *, levels=(1,)):
    """Prepare mean and mean +/- SD reference rings for a tuning-curve plot."""
    values = np.asarray(curve, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return []
    mean, std = float(np.mean(finite)), float(np.std(finite))
    radii = [mean]
    for level in levels:
        if level > 0:
            radii.append(mean + level * std)
            if mean - level * std > 0:
                radii.append(mean - level * std)
    return radii
