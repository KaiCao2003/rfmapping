"""Trial-label cluster-permutation detection for two-dimensional RF maps.

Responses remain fixed while the joint spatial labels are permuted within
``stratum_ids``.  The public API is intentionally one function operating on
plain arrays and returning a plain dictionary of read-only NumPy arrays.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

__all__ = ["detect_rf"]


Alternative = Literal["greater", "less"]
_AUTO_MAX_THREADS = 2

_SPATIAL_STRUCTURE = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=np.uint8,
)
_SPATIAL_STRUCTURE.setflags(write=False)

# The leading dimension indexes independent permutation maps.  It must never
# connect to the preceding or following map.
_BATCH_STRUCTURE = np.zeros((3, 3, 3), dtype=np.uint8)
_BATCH_STRUCTURE[1, :, :] = _SPATIAL_STRUCTURE
_BATCH_STRUCTURE.setflags(write=False)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be a finite number")
    return parsed


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return parsed


def _shape_pair(shape: Sequence[int]) -> tuple[int, int]:
    if isinstance(shape, (str, bytes)):
        raise ValueError("shape must be a (n_y, n_x) pair")
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must be a (n_y, n_x) pair") from exc
    if len(values) != 2:
        raise ValueError("shape must be a (n_y, n_x) pair")
    return (
        _integer(values[0], "shape[0]", minimum=1),
        _integer(values[1], "shape[1]", minimum=1),
    )


def _validate_responses(responses: Any) -> NDArray[np.float64]:
    raw = np.asarray(responses)
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError("responses must be a real numeric array")
    try:
        values = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"responses must be a real numeric array: {exc}") from exc
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("responses must have shape (unit, trial)")
    if not np.all(np.isfinite(values)):
        raise ValueError("responses must contain only finite values")
    return values


def _validate_trial_ids(
    values: Any,
    n_trials: int,
    name: str,
) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or raw.size != n_trials
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(
            f"{name} must be a one-dimensional integer array aligned to trials"
        )
    return np.array(raw, dtype=np.int64, copy=True)


def _validate_unit_ids(values: Any | None, n_units: int) -> NDArray[np.int64]:
    if values is None:
        return np.arange(n_units, dtype=np.int64)
    raw = np.asarray(values)
    if (
        raw.ndim != 1
        or raw.size != n_units
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(
            "unit_ids must be a one-dimensional integer array aligned to units"
        )
    unit_ids = np.array(raw, dtype=np.int64, copy=True)
    if np.unique(unit_ids).size != n_units:
        raise ValueError("unit_ids must be unique")
    return unit_ids


def _available_cpu_count() -> int:
    """Return the CPUs available to this process with portable fallbacks."""

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        try:
            count = process_cpu_count()
        except (OSError, NotImplementedError):
            count = None
        if count is not None and int(count) > 0:
            return int(count)

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            count = len(get_affinity(0))
        except (OSError, NotImplementedError):
            count = 0
        if count > 0:
            return count

    return max(1, int(os.cpu_count() or 1))


def _resolve_n_workers(
    n_jobs: int | None,
    n_items: int,
    *,
    is_shuffle: bool,
) -> int:
    if not is_shuffle or n_items <= 1:
        return 1
    available = _available_cpu_count()
    requested = min(available, _AUTO_MAX_THREADS) if n_jobs is None else n_jobs
    return max(1, min(int(requested), available, n_items))


def _running_in_notebook() -> bool:
    """Return whether tqdm should use its interactive notebook display."""

    if "ipykernel" not in sys.modules:
        return False
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and getattr(shell, "kernel", None) is not None


def _make_progress_bar(*, total: int, n_workers: int) -> Any:
    """Create the RF permutation bar; kept separate for deterministic tests."""

    from tqdm import tqdm

    worker_label = "worker" if n_workers == 1 else "workers"
    return tqdm(
        total=total,
        desc="Detecting RF",
        unit="permutation",
        postfix=f"{n_workers} {worker_label}",
        dynamic_ncols=True,
        # tqdm's automatic TTY check keeps redirected/test output clean. Its
        # notebook backend needs an explicit False to render in Jupyter.
        disable=False if _running_in_notebook() else None,
    )


def _neighbors(
    y: int,
    x: int,
    shape: tuple[int, int],
    *,
    wrap_x: bool,
) -> tuple[tuple[int, int], ...]:
    n_y, n_x = shape
    found: list[tuple[int, int]] = []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_y = y + dy
        if not 0 <= next_y < n_y:
            continue
        next_x = x + dx
        if dx and wrap_x:
            next_x %= n_x
        elif not 0 <= next_x < n_x:
            continue
        neighbor = (next_y, next_x)
        if neighbor != (y, x) and neighbor not in found:
            found.append(neighbor)
    return tuple(found)


def _union_seam_labels(
    labels: NDArray[np.int32],
    candidate: NDArray[np.bool_],
    number_of_labels: int,
) -> NDArray[np.int32]:
    """Return a lookup that merges components touching the horizontal seam."""

    lookup = np.arange(number_of_labels + 1, dtype=np.int32)
    if candidate.shape[-1] <= 1:
        return lookup
    seam = candidate[..., 0] & candidate[..., -1]
    if not np.any(seam):
        return lookup

    parents = np.arange(number_of_labels + 1, dtype=np.int32)

    def find(label: int) -> int:
        root = label
        while int(parents[root]) != root:
            root = int(parents[root])
        while int(parents[label]) != label:
            next_label = int(parents[label])
            parents[label] = root
            label = next_label
        return root

    left_labels = labels[..., 0][seam]
    right_labels = labels[..., -1][seam]
    for left, right in zip(left_labels, right_labels, strict=True):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for label in range(1, number_of_labels + 1):
        lookup[label] = find(label)
    return lookup


def _label_clusters(
    score_map: NDArray[np.float64],
    candidate_mask: NDArray[np.bool_],
    *,
    wrap_x: bool,
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    labels, number_of_labels = ndimage.label(
        candidate_mask,
        structure=_SPATIAL_STRUCTURE,
    )
    labels = np.asarray(labels, dtype=np.int32)
    if number_of_labels == 0:
        return labels, np.empty(0, dtype=np.float64)

    if wrap_x:
        roots = _union_seam_labels(labels, candidate_mask, int(number_of_labels))
        rooted_labels = roots[labels]
        active_roots = np.unique(rooted_labels[candidate_mask])
        contiguous = np.zeros(number_of_labels + 1, dtype=np.int32)
        contiguous[active_roots] = np.arange(1, active_roots.size + 1)
        labels = contiguous[rooted_labels]

    masses = np.bincount(
        labels.ravel(),
        weights=np.where(candidate_mask, score_map, 0.0).ravel(),
        minlength=int(labels.max()) + 1,
    )[1:]
    return labels, np.asarray(masses, dtype=np.float64)


def _max_cluster_masses_batch(
    score_maps: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    *,
    cluster_forming_z: float,
    wrap_x: bool,
) -> NDArray[np.float64]:
    """Compute one maximum cluster mass per map in a SciPy batch."""

    scores = np.asarray(score_maps, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if scores.ndim != 3 or scores.shape[1:] != valid.shape:
        raise ValueError("score_maps must have shape (batch, y, x)")
    n_maps = scores.shape[0]
    maximum = np.zeros(n_maps, dtype=np.float64)
    if n_maps == 0:
        return maximum

    candidate = valid[np.newaxis] & (scores >= cluster_forming_z)
    if not np.any(candidate):
        return maximum

    labels, number_of_labels = ndimage.label(
        candidate,
        structure=_BATCH_STRUCTURE,
    )
    labels = np.asarray(labels, dtype=np.int32)
    if wrap_x:
        roots = _union_seam_labels(labels, candidate, int(number_of_labels))
        component_labels = roots[labels]
    else:
        component_labels = labels

    component_masses = np.bincount(
        component_labels.ravel(),
        weights=np.where(candidate, scores, 0.0).ravel(),
        minlength=int(number_of_labels) + 1,
    )
    active_labels = np.unique(component_labels[candidate])
    map_grid = np.broadcast_to(
        np.arange(n_maps, dtype=np.int64)[:, np.newaxis, np.newaxis],
        candidate.shape,
    )
    component_maps = np.full(number_of_labels + 1, -1, dtype=np.int64)
    np.maximum.at(
        component_maps,
        component_labels[candidate],
        map_grid[candidate],
    )
    np.maximum.at(
        maximum,
        component_maps[active_labels],
        component_masses[active_labels],
    )
    return maximum


def _fill_single_holes(
    significant_labels: NDArray[np.int32],
    valid_mask: NDArray[np.bool_],
    *,
    wrap_x: bool,
    min_hole_neighbors: int,
) -> NDArray[np.bool_]:
    filled = np.zeros(valid_mask.shape, dtype=bool)
    for y, x in np.argwhere(valid_mask & (significant_labels == 0)):
        neighbors = _neighbors(
            int(y),
            int(x),
            valid_mask.shape,
            wrap_x=wrap_x,
        )
        if len(neighbors) < min_hole_neighbors:
            continue
        neighbor_labels = [int(significant_labels[item]) for item in neighbors]
        if neighbor_labels and neighbor_labels[0] > 0 and all(
            label == neighbor_labels[0] for label in neighbor_labels
        ):
            filled[y, x] = True
    return filled


def _permuted_label_batch(
    original: NDArray[np.int64],
    stratum_indices: tuple[NDArray[np.int64], ...],
    current_batch_size: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate labels sequentially so batch size cannot change the RNG stream."""

    batch = np.empty((current_batch_size, original.size), dtype=np.int64)
    for batch_index in range(current_batch_size):
        labels = original.copy()
        for indices in stratum_indices:
            if indices.size > 1:
                labels[indices] = original[indices][rng.permutation(indices.size)]
        batch[batch_index] = labels
    return batch


def _null_masses_for_unit(
    response: NDArray[np.float64],
    flat_bins: NDArray[np.int64],
    n_positions: int,
    counts: NDArray[np.float64],
    valid_flat: NDArray[np.bool_],
    null_mean_flat: NDArray[np.float64],
    null_sd_flat: NDArray[np.float64],
    testable_flat: NDArray[np.bool_],
    shape: tuple[int, int],
    cluster_forming_z: float,
    alternative: Alternative,
    wrap_x: bool,
) -> NDArray[np.float64]:
    current_batch_size = flat_bins.shape[0]
    sums = np.bincount(
        flat_bins.ravel(),
        weights=np.tile(response, current_batch_size),
        minlength=current_batch_size * n_positions,
    ).reshape(current_batch_size, n_positions)
    maps = np.zeros_like(sums)
    maps[:, valid_flat] = sums[:, valid_flat] / counts[valid_flat]
    scores = np.zeros_like(maps)
    scores[:, testable_flat] = (
        maps[:, testable_flat] - null_mean_flat[testable_flat]
    ) / null_sd_flat[testable_flat]
    if alternative == "less":
        scores *= -1.0
    return _max_cluster_masses_batch(
        scores.reshape((current_batch_size, *shape)),
        testable_flat.reshape(shape),
        cluster_forming_z=cluster_forming_z,
        wrap_x=wrap_x,
    )


def _cluster_pvalues(
    cluster_masses: NDArray[np.float64],
    null_max_masses: NDArray[np.float64],
) -> NDArray[np.float64]:
    denominator = null_max_masses.size + 1
    return np.asarray(
        [
            (1 + int(np.count_nonzero(null_max_masses >= mass))) / denominator
            for mass in cluster_masses
        ],
        dtype=np.float64,
    )


def _freeze(array: NDArray[Any]) -> NDArray[Any]:
    array.setflags(write=False)
    return array


def detect_rf(
    responses: Any,
    position_ids: Any,
    shape: Sequence[int],
    *,
    stratum_ids: Any | None = None,
    unit_ids: Any | None = None,
    is_shuffle: bool = True,
    cluster_forming_z: float = 1.5,
    alpha: float = 0.05,
    n_permutations: int = 10_000,
    alternative: Alternative = "greater",
    wrap_x: bool = True,
    fill_single_holes: bool = False,
    min_hole_neighbors: int = 3,
    random_seed: int | None = 0,
    batch_size: int = 64,
    n_jobs: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Detect trial-level two-dimensional RF clusters for one or more units.

    Parameters
    ----------
    responses
        Real finite array with shape ``(unit, trial)``.
    position_ids
        Zero-based row-major joint spatial labels, ``y * n_x + x``.
    shape
        RF grid shape ``(n_y, n_x)``.
    stratum_ids
        Optional trial-aligned integer labels. Position labels are shuffled
        only within each stratum.
    n_jobs
        Worker threads across units, or across permutation chunks for a single
        unit. ``None`` uses one worker for a single unit and at most two
        available CPUs across multiple units; ``1`` always forces serial work.
    show_progress
        Show a ``Detecting RF`` permutation progress bar in an interactive
        terminal or notebook. Redirected and test output stays silent. No bar
        is created when ``is_shuffle=False``.

    Returns
    -------
    dict
        Batch-shaped read-only arrays. Variable-length per-unit cluster masses
        and p-values are tuples of read-only arrays. ``is_shuffle=False`` is
        exploratory: ``final_mask`` equals ``candidate_mask`` and no cluster is
        represented as permutation-significant.
    """

    response_values = _validate_responses(responses)
    n_units, n_trials = response_values.shape
    grid_shape = _shape_pair(shape)
    n_positions = grid_shape[0] * grid_shape[1]

    positions = _validate_trial_ids(position_ids, n_trials, "position_ids")
    if np.any(positions < 0) or np.any(positions >= n_positions):
        raise ValueError(f"position_ids must be between 0 and {n_positions - 1}")

    if stratum_ids is None:
        strata = np.zeros(n_trials, dtype=np.int64)
    else:
        raw_strata = _validate_trial_ids(stratum_ids, n_trials, "stratum_ids")
        _stratum_values, strata = np.unique(raw_strata, return_inverse=True)
        strata = strata.astype(np.int64, copy=False)
    output_unit_ids = _validate_unit_ids(unit_ids, n_units)

    if not isinstance(is_shuffle, (bool, np.bool_)):
        raise ValueError("is_shuffle must be bool")
    run_shuffle = bool(is_shuffle)
    if not isinstance(show_progress, (bool, np.bool_)):
        raise ValueError("show_progress must be bool")
    display_progress = bool(show_progress)
    threshold = _finite_float(cluster_forming_z, "cluster_forming_z")
    if threshold <= 0:
        raise ValueError("cluster_forming_z must be greater than zero")
    parsed_alpha = _finite_float(alpha, "alpha")
    if not 0 < parsed_alpha < 1:
        raise ValueError("alpha must be between zero and one")
    permutation_count = _integer(n_permutations, "n_permutations", minimum=0)
    if run_shuffle and permutation_count < 1:
        raise ValueError("n_permutations must be at least 1 when is_shuffle=True")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less'")
    if not isinstance(wrap_x, (bool, np.bool_)):
        raise ValueError("wrap_x must be bool")
    if not isinstance(fill_single_holes, (bool, np.bool_)):
        raise ValueError("fill_single_holes must be bool")
    wrap = bool(wrap_x)
    fill_holes = bool(fill_single_holes)
    minimum_neighbors = _integer(
        min_hole_neighbors,
        "min_hole_neighbors",
        minimum=1,
    )
    if minimum_neighbors > 4:
        raise ValueError("min_hole_neighbors must not exceed four")
    parsed_batch_size = _integer(batch_size, "batch_size", minimum=1)
    if random_seed is not None:
        parsed_seed: int | None = _integer(random_seed, "random_seed", minimum=0)
    else:
        parsed_seed = None
    if n_jobs is not None:
        parsed_n_jobs: int | None = _integer(n_jobs, "n_jobs", minimum=1)
    else:
        parsed_n_jobs = None
    parallel_items = (
        n_units
        if n_units > 1
        else min(permutation_count, parsed_batch_size)
    )
    if parsed_n_jobs is None and n_units == 1:
        # On the real 7 x 30 workload, splitting one unit's small permutation
        # maps costs more than it saves. Explicit n_jobs > 1 remains available.
        n_workers = 1
    else:
        n_workers = _resolve_n_workers(
            parsed_n_jobs,
            parallel_items,
            is_shuffle=run_shuffle,
        )
    counts = np.bincount(positions, minlength=n_positions).astype(np.float64)
    valid_flat = counts > 0
    stratum_indices: list[NDArray[np.int64]] = []
    stratum_position_counts: list[NDArray[np.float64]] = []
    for stratum in range(int(strata.max()) + 1):
        indices = np.flatnonzero(strata == stratum).astype(np.int64)
        stratum_indices.append(indices)
        stratum_position_counts.append(
            np.bincount(positions[indices], minlength=n_positions).astype(np.float64)
        )
    stratum_indices_tuple = tuple(stratum_indices)

    expected_sums = np.zeros((n_units, n_positions), dtype=np.float64)
    variance_sums = np.zeros_like(expected_sums)
    for indices, position_counts in zip(
        stratum_indices,
        stratum_position_counts,
        strict=True,
    ):
        stratum_responses = response_values[:, indices]
        stratum_size = indices.size
        means = stratum_responses.mean(axis=1, dtype=np.float64)
        variances = stratum_responses.var(axis=1, dtype=np.float64, ddof=0)
        expected_sums += means[:, None] * position_counts[None]
        if stratum_size > 1:
            finite_population_factor = (
                position_counts
                * (stratum_size - position_counts)
                / (stratum_size - 1)
            )
            variance_sums += variances[:, None] * finite_population_factor[None]

    response_flat = np.full((n_units, n_positions), np.nan, dtype=np.float64)
    null_mean_flat = np.full_like(response_flat, np.nan)
    null_sd_flat = np.full_like(response_flat, np.nan)
    for unit_index in range(n_units):
        sums = np.bincount(
            positions,
            weights=response_values[unit_index],
            minlength=n_positions,
        )
        response_flat[unit_index, valid_flat] = sums[valid_flat] / counts[valid_flat]
    null_mean_flat[:, valid_flat] = (
        expected_sums[:, valid_flat] / counts[valid_flat]
    )
    variance_sums = np.maximum(variance_sums, 0.0)
    null_sd_flat[:, valid_flat] = (
        np.sqrt(variance_sums[:, valid_flat]) / counts[valid_flat]
    )
    testable_flat = (
        valid_flat[np.newaxis]
        & np.isfinite(response_flat)
        & np.isfinite(null_mean_flat)
        & np.isfinite(null_sd_flat)
        & (null_sd_flat > 0)
    )
    z_flat = np.zeros((n_units, n_positions), dtype=np.float64)
    z_flat[testable_flat] = (
        response_flat[testable_flat] - null_mean_flat[testable_flat]
    ) / null_sd_flat[testable_flat]
    score_flat = z_flat if alternative == "greater" else -z_flat
    candidate_flat = testable_flat & (score_flat >= threshold)

    response_map = response_flat.reshape((n_units, *grid_shape))
    null_mean_map = null_mean_flat.reshape((n_units, *grid_shape))
    null_sd_map = null_sd_flat.reshape((n_units, *grid_shape))
    z_map = z_flat.reshape((n_units, *grid_shape))
    valid_mask = testable_flat.reshape((n_units, *grid_shape))
    candidate_mask = candidate_flat.reshape((n_units, *grid_shape))
    cluster_labels = np.zeros((n_units, *grid_shape), dtype=np.int32)
    cluster_mass_list: list[NDArray[np.float64]] = []
    for unit_index in range(n_units):
        labels, masses = _label_clusters(
            score_flat[unit_index].reshape(grid_shape),
            candidate_mask[unit_index],
            wrap_x=wrap,
        )
        cluster_labels[unit_index] = labels
        cluster_mass_list.append(masses)

    if run_shuffle:
        null_max_masses = np.empty(
            (n_units, permutation_count),
            dtype=np.float64,
        )
        rng = np.random.default_rng(parsed_seed)
        progress_bar = (
            _make_progress_bar(
                total=permutation_count,
                n_workers=n_workers,
            )
            if display_progress
            else None
        )
        executor = None
        try:
            executor = (
                ThreadPoolExecutor(
                    max_workers=n_workers,
                    thread_name_prefix="rf-shuffle",
                )
                if n_workers > 1
                else None
            )
            completed = 0
            while completed < permutation_count:
                current_batch = min(
                    parsed_batch_size,
                    permutation_count - completed,
                )
                labels_batch = _permuted_label_batch(
                    positions,
                    stratum_indices_tuple,
                    current_batch,
                    rng,
                )
                if executor is not None and n_units == 1:
                    chunk_count = min(n_workers, current_batch)
                    chunk_size, extra = divmod(current_batch, chunk_count)
                    chunk_ranges: list[tuple[int, int]] = []
                    chunk_start = 0
                    for chunk_index in range(chunk_count):
                        chunk_stop = (
                            chunk_start
                            + chunk_size
                            + (chunk_index < extra)
                        )
                        chunk_ranges.append((chunk_start, chunk_stop))
                        chunk_start = chunk_stop

                    def calculate_chunk(
                        chunk_range: tuple[int, int],
                    ) -> NDArray[np.float64]:
                        start, stop = chunk_range
                        chunk_labels = labels_batch[start:stop]
                        flat_bins = chunk_labels + (
                            np.arange(stop - start, dtype=np.int64)[:, None]
                            * n_positions
                        )
                        return _null_masses_for_unit(
                            response_values[0],
                            flat_bins,
                            n_positions,
                            counts,
                            valid_flat,
                            null_mean_flat[0],
                            null_sd_flat[0],
                            testable_flat[0],
                            grid_shape,
                            threshold,
                            alternative,
                            wrap,
                        )

                    chunk_results = executor.map(calculate_chunk, chunk_ranges)
                    for (start, stop), chunk_masses in zip(
                        chunk_ranges,
                        chunk_results,
                        strict=True,
                    ):
                        null_max_masses[
                            0,
                            completed + start : completed + stop,
                        ] = chunk_masses
                else:
                    flat_bins = labels_batch + (
                        np.arange(current_batch, dtype=np.int64)[:, None]
                        * n_positions
                    )

                    def calculate(unit_index: int) -> NDArray[np.float64]:
                        return _null_masses_for_unit(
                            response_values[unit_index],
                            flat_bins,
                            n_positions,
                            counts,
                            valid_flat,
                            null_mean_flat[unit_index],
                            null_sd_flat[unit_index],
                            testable_flat[unit_index],
                            grid_shape,
                            threshold,
                            alternative,
                            wrap,
                        )

                    if executor is None:
                        batch_results = map(calculate, range(n_units))
                    else:
                        batch_results = executor.map(calculate, range(n_units))
                    for unit_index, unit_masses in enumerate(batch_results):
                        null_max_masses[
                            unit_index,
                            completed : completed + current_batch,
                        ] = unit_masses
                completed += current_batch
                if progress_bar is not None:
                    progress_bar.update(current_batch)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            if progress_bar is not None:
                progress_bar.close()
    else:
        null_max_masses = np.empty((n_units, 0), dtype=np.float64)

    cluster_pvalue_list: list[NDArray[np.float64]] = []
    significant_mask = np.zeros((n_units, *grid_shape), dtype=bool)
    filled_mask = np.zeros_like(significant_mask)
    final_mask = np.zeros_like(significant_mask)
    for unit_index, masses in enumerate(cluster_mass_list):
        if run_shuffle:
            pvalues = _cluster_pvalues(masses, null_max_masses[unit_index])
            significant_cluster_ids = np.flatnonzero(pvalues <= parsed_alpha) + 1
            significant_labels = np.where(
                np.isin(cluster_labels[unit_index], significant_cluster_ids),
                cluster_labels[unit_index],
                0,
            ).astype(np.int32, copy=False)
            significant_mask[unit_index] = significant_labels > 0
            if fill_holes:
                filled_mask[unit_index] = _fill_single_holes(
                    significant_labels,
                    valid_mask[unit_index],
                    wrap_x=wrap,
                    min_hole_neighbors=minimum_neighbors,
                )
            final_mask[unit_index] = (
                significant_mask[unit_index] | filled_mask[unit_index]
            )
        else:
            pvalues = np.full(masses.shape, np.nan, dtype=np.float64)
            final_mask[unit_index] = candidate_mask[unit_index]
        cluster_pvalue_list.append(pvalues)

    cluster_masses = tuple(_freeze(item) for item in cluster_mass_list)
    cluster_pvalues = tuple(_freeze(item) for item in cluster_pvalue_list)
    arrays = (
        output_unit_ids,
        response_map,
        null_mean_map,
        null_sd_map,
        z_map,
        valid_mask,
        candidate_mask,
        cluster_labels,
        significant_mask,
        filled_mask,
        final_mask,
        null_max_masses,
    )
    for array in arrays:
        _freeze(array)

    return {
        "unit_ids": output_unit_ids,
        "response_map": response_map,
        "null_mean_map": null_mean_map,
        "null_sd_map": null_sd_map,
        "z_map": z_map,
        "valid_mask": valid_mask,
        "candidate_mask": candidate_mask,
        "cluster_labels": cluster_labels,
        "cluster_masses": cluster_masses,
        "cluster_pvalues": cluster_pvalues,
        "significant_mask": significant_mask,
        "filled_mask": filled_mask,
        "final_mask": final_mask,
        "null_max_masses": null_max_masses,
        "is_significance_tested": run_shuffle,
        "n_workers": n_workers,
    }
