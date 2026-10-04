"""Explicit RF detection, with legacy array views delegated here."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from ._detector import _RF_DETECTION_OPTION_DEFAULTS
from ._validation import (
    _EDGE_ATOL_S,
    _axis_name,
    _bool_value,
    _bounded_integer,
    _nonnegative_integer,
    _readonly_array,
)
from .results import RFResult
if TYPE_CHECKING:
    from .models import RFMap

__all__ = ["detect_rf"]

_BATCH_MAP_KEYS = {
    "response_map",
    "null_mean_map",
    "null_sd_map",
    "z_map",
    "valid_mask",
    "candidate_mask",
    "cluster_labels",
    "significant_mask",
    "filled_mask",
    "final_mask",
}


_RF_RESULT_EXECUTION_OPTIONS = {"batch_size", "n_jobs"}


def _center_only_mask(
        result: Mapping[str, Any],
        *,
        alternative: str,
        wrap_x: bool,
        show_progress: bool,
) -> NDArray[np.uint8]:
    """Reduce each non-empty final RF to one response-weighted grid bin."""

    final_mask = np.asarray(result["final_mask"], dtype=bool)
    response_map = np.asarray(result["response_map"], dtype=np.float64)
    null_mean_map = np.asarray(result["null_mean_map"], dtype=np.float64)
    if not (
            final_mask.shape == response_map.shape == null_mean_map.shape
            and final_mask.ndim in {2, 3}
    ):
        raise RuntimeError("RF detector returned incompatible center-map arrays")

    is_single = final_mask.ndim == 2
    mask_batch = final_mask[np.newaxis, ...] if is_single else final_mask
    response_batch = response_map[np.newaxis, ...] if is_single else response_map
    null_batch = null_mean_map[np.newaxis, ...] if is_single else null_mean_map
    centers = np.zeros(mask_batch.shape, dtype=np.uint8)

    nonempty_units = np.flatnonzero(np.any(mask_batch, axis=(1, 2)))
    if nonempty_units.size == 0:
        return centers[0] if is_single else centers

    unit_indices: Any = nonempty_units
    if show_progress:
        unit_indices = tqdm(
            nonempty_units,
            desc="Center",
            unit="unit",
        )

    direction = -1.0 if alternative == "less" else 1.0
    n_x = mask_batch.shape[2]
    for unit_index in unit_indices:
        unit_mask = mask_batch[unit_index]
        candidate_flat = np.flatnonzero(unit_mask)
        candidate_y, candidate_x = np.unravel_index(
            candidate_flat,
            unit_mask.shape,
        )

        effect = direction * (
                response_batch[unit_index] - null_batch[unit_index]
        )
        effect = np.where(np.isfinite(effect), np.maximum(effect, 0.0), 0.0)
        weights = effect.ravel()[candidate_flat]
        if not np.any(weights > 0.0):
            weights = np.ones(candidate_flat.size, dtype=np.float64)

        y_coordinates = candidate_y.astype(np.float64, copy=False)
        x_coordinates = candidate_x.astype(np.float64, copy=False)
        dy = np.abs(y_coordinates[:, None] - y_coordinates[None, :])
        dx = np.abs(x_coordinates[:, None] - x_coordinates[None, :])
        if wrap_x:
            dx = np.minimum(dx, n_x - dx)
        costs = ((dy * dy + dx * dx) * weights[None, :]).sum(axis=1)

        minimum_cost = float(costs.min())
        tie_tolerance = (
                16.0
                * np.finfo(np.float64).eps
                * max(1.0, abs(minimum_cost))
        )
        tied = np.flatnonzero(
            np.isclose(costs, minimum_cost, rtol=0.0, atol=tie_tolerance)
        )
        tied_weights = weights[tied]
        highest_local_weight = float(tied_weights.max())
        tied = tied[tied_weights == highest_local_weight]
        selected_flat = int(candidate_flat[int(tied[0])])
        centers[unit_index].flat[selected_flat] = 1

    return centers[0] if is_single else centers



def _aligned_trial_mapping(
        trials: Mapping[str, Any],
        maps: Sequence["RFMap"],
) -> dict[str, Any]:
    """Validate plain trial arrays and align their rows to RF unit IDs."""

    if not isinstance(trials, Mapping):
        raise TypeError("trials must be a mapping of trial arrays")
    required = {
        "responses",
        "position_ids",
        "unit_ids",
        "shape",
        "x_positions",
        "y_positions",
        "time_range_s",
    }
    missing = required.difference(trials)
    if missing:
        raise ValueError(f"trials is missing required keys: {sorted(missing)}")

    first = maps[0]
    for rf_map in maps:
        if rf_map.n_time_bins != 1:
            raise ValueError(
                "RF detection requires exactly one response-window time bin; "
                "call sum(earlier_s, later_s) first"
            )
        if rf_map.shape != first.shape:
            raise ValueError("all RFMaps must share one RF grid shape")
        if not np.array_equal(rf_map.x_positions, first.x_positions):
            raise ValueError("all RFMaps must share x positions")
        if not np.array_equal(rf_map.y_positions, first.y_positions):
            raise ValueError("all RFMaps must share y positions")
        if not np.allclose(
                rf_map.time_window_s,
                first.time_window_s,
                rtol=0.0,
                atol=_EDGE_ATOL_S,
        ):
            raise ValueError("all RFMaps must share one response window")

    try:
        trial_shape = tuple(trials["shape"])
    except TypeError as exc:
        raise ValueError("trials['shape'] must be a (n_y, n_x) pair") from exc
    if trial_shape != (first.n_y, first.n_x):
        raise ValueError("trial RF grid shape does not match this RFMap")
    if not np.array_equal(np.asarray(trials["x_positions"]), first.x_positions):
        raise ValueError("trial x positions do not match this RFMap")
    if not np.array_equal(np.asarray(trials["y_positions"]), first.y_positions):
        raise ValueError("trial y positions do not match this RFMap")
    try:
        trial_window = np.asarray(trials["time_range_s"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("trials['time_range_s'] must contain two numbers") from exc
    if trial_window.shape != (2,) or not np.allclose(
            trial_window,
            first.time_window_s,
            rtol=0.0,
            atol=_EDGE_ATOL_S,
    ):
        raise ValueError(
            "trial response window does not match this RFMap's sole time bin"
        )

    responses = np.asarray(trials["responses"])
    if responses.ndim == 1:
        responses = responses[np.newaxis, :]
    unit_ids = np.asarray(trials["unit_ids"])
    if responses.ndim != 2 or unit_ids.ndim != 1:
        raise ValueError("trial responses and unit_ids must be 2-D and 1-D")
    if responses.shape[0] != unit_ids.size:
        raise ValueError("trial response rows must align with unit_ids")
    if np.unique(unit_ids).size != unit_ids.size:
        raise ValueError("trial unit_ids must be unique")

    rows_by_unit_id = {unit_id: index for index, unit_id in enumerate(unit_ids)}
    row_indices: list[int] = []
    for rf_map in maps:
        if rf_map.unit_id not in rows_by_unit_id:
            raise KeyError(f"trial responses do not contain unit_id {rf_map.unit_id}")
        row_indices.append(rows_by_unit_id[rf_map.unit_id])

    aligned = dict(trials)
    if row_indices == list(range(unit_ids.size)):
        aligned["responses"] = responses
    elif row_indices == list(range(row_indices[0], row_indices[0] + len(maps))):
        aligned["responses"] = responses[row_indices[0] : row_indices[-1] + 1]
    else:
        aligned["responses"] = responses[row_indices]
    aligned["unit_ids"] = np.asarray([item.unit_id for item in maps], dtype=np.int64)
    return aligned



def _single_detection_result(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the singleton unit axis from a plain batch-result dictionary."""

    result: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "unit_ids":
            result[key] = _readonly_array(value)
        elif key in _BATCH_MAP_KEYS or key == "null_max_masses":
            result[key] = value[0]
        elif key in {"cluster_masses", "cluster_pvalues"}:
            result[key] = value[0]
        else:
            result[key] = value
    return result



def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value



def _rf_detection_parameters(
        *,
        is_shuffle: bool,
        drop_bins: int,
        options: Mapping[str, Any],
) -> dict[str, Any]:
    unknown = sorted(set(options).difference(_RF_DETECTION_OPTION_DEFAULTS))
    if unknown:
        labels = ", ".join(repr(name) for name in unknown)
        raise TypeError(f"unexpected RF detection option(s): {labels}")

    _bounded_integer(
        options.get("batch_size", _RF_DETECTION_OPTION_DEFAULTS["batch_size"]),
        "batch_size",
        minimum=1,
    )
    n_jobs = options.get("n_jobs", _RF_DETECTION_OPTION_DEFAULTS["n_jobs"])
    if n_jobs is not None:
        _bounded_integer(n_jobs, "n_jobs", minimum=1)

    parameters = {
        name: _json_scalar(options.get(name, default))
        for name, default in _RF_DETECTION_OPTION_DEFAULTS.items()
        if name not in _RF_RESULT_EXECUTION_OPTIONS
    }
    parameters["is_shuffle"] = is_shuffle
    # A no-shuffle size cutoff is deliberately absent from shuffled-result
    # identity because shuffled output must not depend on it.
    parameters["drop_bins"] = None if is_shuffle else drop_bins
    return parameters



def _rf_result_manifest(
        maps: Sequence["RFMap"],
        parameters: Mapping[str, Any],
) -> dict[str, Any]:
    from ._cache import _rf_source_path

    first = maps[0]
    if any(_rf_source_path(item.source_path) != _rf_source_path(first.source_path) for item in maps):
        raise ValueError("RF detection maps must share one source")
    return {
        "schema_name": "rfmapping-rf-result",
        "detector_algorithm": (
            "cluster-permutation-v1"
            if parameters["is_shuffle"]
            else "pooled-spatial-z-v3-zero-filled"
        ),
        "center_algorithm": "response-weighted-medoid-v1",
        "nonshuffle_filter": "drop-small-components-inclusive-v1",
        "grid_shape": [first.n_y, first.n_x],
        "storage_shape": [len(maps), first.n_y, first.n_x],
        "x_positions": first.x_positions.tolist(),
        "y_positions": first.y_positions.tolist(),
        "source_path": _rf_source_path(first.source_path),
        "time_range_s": list(first.time_window_s),
        "response_units": first.metadata.get("responseUnits", "spike_count"),
        "response_normalization": first.metadata.get("responseNormalization", "none"),
        "parameters": dict(parameters),
    }



def _rf_result_key_arrays(aligned: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "responses": aligned["responses"],
        "position_ids": aligned["position_ids"],
        "stratum_ids": aligned.get("stratum_ids"),
        "unit_ids": aligned["unit_ids"],
        "x_positions": aligned["x_positions"],
        "y_positions": aligned["y_positions"],
    }



def _rf_detection_inputs(
        maps: Sequence["RFMap"],
        trials: Mapping[str, Any] | None,
        *,
        run_shuffle: bool,
        exclude_zero_bins: bool,
        collapse_axis: str | None,
) -> dict[str, Any]:
    """Prepare aligned detector inputs without computing or caching results."""
    if run_shuffle:
        if trials is None:
            raise ValueError("trials are required when is_shuffle=True")
        aligned = _aligned_trial_mapping(trials, maps)
        if collapse_axis is not None:
            positions = np.asarray(aligned["position_ids"])
            n_positions = maps[0].n_y * maps[0].n_x
            if np.any(positions < 0) or np.any(positions >= n_positions):
                raise ValueError(f"position_ids must be between 0 and {n_positions - 1}")
            aligned["position_ids"] = (
                positions % maps[0].n_x
                if collapse_axis == "x"
                else positions // maps[0].n_x
            )
    else:
        first = maps[0]
        pooled = np.stack(
            [np.asarray(rf_map.to_2d_array(), dtype=np.float64) for rf_map in maps]
        )
        pooled[np.isnan(pooled)] = 0
        if exclude_zero_bins:
            pooled[~np.isfinite(pooled)] = 0
        if collapse_axis is not None:
            spatial_axis = 0 if collapse_axis == "x" else 1
            pooled = pooled.sum(axis=spatial_axis + 1, keepdims=True)
        position_ids = np.arange(pooled.shape[1] * pooled.shape[2], dtype=np.int64)
        responses = pooled.reshape(len(maps), -1)

        aligned = {
            "responses": responses,
            "position_ids": position_ids,
            "stratum_ids": None,
            "unit_ids": np.asarray(
                [rf_map.unit_id for rf_map in maps],
                dtype=np.int64,
            ),
            "shape": (first.n_y, first.n_x),
            "x_positions": first.x_positions,
            "y_positions": first.y_positions,
            "time_range_s": first.time_window_s,
        }
        if collapse_axis is not None or exclude_zero_bins:
            aligned = _aligned_trial_mapping(aligned, maps)

    if collapse_axis is not None:
        aligned["shape"] = (
            (1, maps[0].n_x) if collapse_axis == "x" else (maps[0].n_y, 1)
        )
        collapsed_positions = "y_positions" if collapse_axis == "x" else "x_positions"
        aligned[collapsed_positions] = np.asarray([0.0])

    return aligned



def _compute_rf_output(
        aligned: Mapping[str, Any],
        *,
        is_batch: bool,
        detect: Any,
        run_shuffle: bool,
        exclude_zero_bins: bool,
        collapse_axis: str | None,
        drop_bins: int,
        parameters: Mapping[str, Any],
        options: Mapping[str, Any],
        show_progress: bool,
        center_progress: bool,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Compute masks and centers from prepared inputs; perform no persistence."""
    if exclude_zero_bins:
        mask = np.zeros((len(aligned["unit_ids"]), *aligned["shape"]), dtype=np.uint8)
        center = np.zeros_like(mask)
        for unit_index, responses in enumerate(aligned["responses"]):
            valid = np.isfinite(responses) & (responses != 0)
            if not valid.any():
                continue
            # One response per retained position makes the detector's null
            # mean/SD equal to this unit's spatial mean/population SD.
            unit_aligned = dict(aligned)
            unit_aligned["responses"] = responses[np.newaxis, valid]
            unit_aligned["position_ids"] = aligned["position_ids"][valid]
            unit_aligned["unit_ids"] = aligned["unit_ids"][unit_index:unit_index + 1]
            result = detect(
                unit_aligned,
                _aligned=True,
                is_shuffle=False,
                drop_bins=drop_bins,
                show_progress=False,
                **options,
            )
            mask[unit_index] = np.asarray(result["final_mask"]).reshape(aligned["shape"])
            center[unit_index] = _center_only_mask(
                result,
                alternative=parameters["alternative"],
                wrap_x=bool(parameters["wrap_x"]),
                show_progress=False,
            ).reshape(aligned["shape"])
        if not is_batch:
            mask, center = mask[0], center[0]
        mask = _readonly_array(mask)
        center = _readonly_array(center)
    else:
        result = detect(
            aligned,
            _aligned=run_shuffle or collapse_axis is not None,
            is_shuffle=run_shuffle,
            drop_bins=drop_bins,
            show_progress=show_progress,
            **options,
        )
        mask = _readonly_array(result["final_mask"], dtype=np.uint8)
        center = _readonly_array(
            _center_only_mask(
                result,
                alternative=parameters["alternative"],
                wrap_x=bool(parameters["wrap_x"]),
                show_progress=center_progress,
            ),
            dtype=np.uint8,
        )

    return mask, center



def _rf_output_arrays(
        *,
        cache: dict[str, Any],
        maps: Sequence["RFMap"],
        is_batch: bool,
        trials: Mapping[str, Any] | None,
        detect: Any,
        rf_type: str,
        is_shuffle: bool,
        drop_bins: int,
        result_path: str | Path | None,
        show_progress: bool,
        center_progress: bool,
        options: Mapping[str, Any],
        collapse_axis: str | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Return one reusable mask-center result for the public RF views."""

    run_shuffle = _bool_value(is_shuffle, "is_shuffle")
    parsed_drop_bins = _nonnegative_integer(drop_bins, "drop_bins")
    options = dict(options)
    if rf_type not in ("excitatory", "inhibitory"):
        raise ValueError("rf_type must be 'excitatory' or 'inhibitory'")
    # Preserve explicit alternative= calls from the original public API.
    if rf_type == "inhibitory":
        options.setdefault("alternative", "less")
    exclude_zero_bins = _bool_value(
        options.pop("exclude_zero_bins", False), "exclude_zero_bins",
    )
    if run_shuffle and exclude_zero_bins:
        raise ValueError("exclude_zero_bins is only supported when is_shuffle=False")
    parameters = _rf_detection_parameters(
        is_shuffle=run_shuffle,
        drop_bins=parsed_drop_bins,
        options=options,
    )
    parameters["exclude_zero_bins"] = exclude_zero_bins

    target: Path | None = None
    if result_path is not None:
        target = Path(result_path)
        if target.suffix.lower() != ".npz":
            raise ValueError("result_path must end with '.npz'")

    aligned = _rf_detection_inputs(
        maps, trials, run_shuffle=run_shuffle,
        exclude_zero_bins=exclude_zero_bins, collapse_axis=collapse_axis,
    )

    # Hash the inputs even for the in-memory fast path. In no-shuffle mode the
    # key comes only from the pooled map; a supplied trial mapping is ignored.
    manifest = _rf_result_manifest(maps, parameters)
    if collapse_axis is not None:
        manifest["collapse_axis"] = collapse_axis
        manifest["grid_shape"] = list(aligned["shape"])
        manifest["storage_shape"] = [len(maps), *aligned["shape"]]
        manifest["x_positions"] = aligned["x_positions"].tolist()
        manifest["y_positions"] = aligned["y_positions"].tolist()
    from ._cache import build_rf_result_cache_key, load_rf_result

    result_key = build_rf_result_cache_key(
        arrays=_rf_result_key_arrays(aligned),
        manifest=manifest,
    )
    memory_hit = (
            cache.get("result_key") == result_key
            and "mask_2d" in cache
            and "center_2d" in cache
    )
    mask: NDArray[np.uint8] | None = (
        cache["mask_2d"] if memory_hit else None
    )
    center: NDArray[np.uint8] | None = (
        cache["center_2d"] if memory_hit else None
    )

    if memory_hit and target is None:
        assert mask is not None and center is not None
        return mask, center

    disk_hit = False
    if target is not None:
        stored = load_rf_result(target, expected_cache_key=result_key)
        if stored is not None:
            expected_unit_ids = np.asarray(
                [rf_map.unit_id for rf_map in maps],
                dtype=np.int64,
            )
            if not np.array_equal(stored["unit_ids"], expected_unit_ids):
                raise RuntimeError(
                    "RF result unit IDs do not match the requested maps"
                )
            expected_shape = tuple(manifest["storage_shape"])
            stored_mask = stored["mask_2d"]
            stored_center = stored["center_2d"]
            if stored_mask.shape != expected_shape:
                raise RuntimeError(
                    "RF result shape does not match the requested maps"
                )
            mask = stored_mask if is_batch else stored_mask[0]
            center = stored_center if is_batch else stored_center[0]
            disk_hit = True

    if mask is None or center is None:
        mask, center = _compute_rf_output(
            aligned, is_batch=is_batch, detect=detect, run_shuffle=run_shuffle,
            exclude_zero_bins=exclude_zero_bins, collapse_axis=collapse_axis,
            drop_bins=parsed_drop_bins, parameters=parameters, options=options,
            show_progress=show_progress, center_progress=center_progress,
        )

    cache.clear()
    cache.update(
        {
            "result_key": result_key,
            "parameters": parameters,
            "manifest": manifest,
            "mask_2d": mask,
            "center_2d": center,
        }
    )

    if target is not None and not disk_hit:
        from ._cache import save_rf_result

        save_rf_result(
            target,
            mask_2d=mask if is_batch else mask[np.newaxis, ...],
            center_2d=center if is_batch else center[np.newaxis, ...],
            unit_ids=[rf_map.unit_id for rf_map in maps],
            manifest=manifest,
            cache_key=result_key,
        )
    return mask, center



def _legacy_rf_2d(
        self,
        trials: Mapping[str, Any] | None = None,
        *,
        rf_type: str = "excitatory",
        is_shuffle: bool = False,
        drop_bins: int = 2,
        is_center: bool = False,
        result_path: str | Path | None = None,
        show_progress: bool = True,
        **options: Any,
) -> NDArray[np.uint8]:
    """Return the final 2-D RF mask or its discrete weighted center.

    ``rf_type="inhibitory"`` selects spatially decreased responses instead
    of the default increased responses. An explicit ``alternative``
    overrides ``rf_type`` for compatibility with existing calls.
    ``is_center=False`` returns the complete mask; ``True`` returns one
    response-weighted bin for each non-empty RF.  A ``.npz``
    ``result_path`` persists both views, so switching ``is_center`` later
    does not repeat detection.  ``drop_bins`` is used only by exploratory
    ``is_shuffle=False`` runs and removes connected components whose size
    is less than or equal to the cutoff.
    ``exclude_zero_bins=True`` additionally omits missing and zero responses
    from each unit's spatial mean, SD, and candidate mask. Batch results
    retain a leading unit axis; single-unit results have shape ``(y, x)``.
    """

    center_only = _bool_value(is_center, "is_center")
    progress = _bool_value(show_progress, "show_progress")
    mask, center = _rf_output_arrays(
        cache=self._rf_result_cache,
        maps=self._rf_input_maps,
        is_batch=self._rf_is_batch,
        trials=trials,
        detect=self._detect_rf,
        rf_type=rf_type,
        is_shuffle=is_shuffle,
        drop_bins=drop_bins,
        result_path=result_path,
        show_progress=progress,
        center_progress=progress and center_only,
        options=options,
    )
    return center if center_only else mask


def _legacy_rf_1d(
        self,
        trials: Mapping[str, Any] | None = None,
        axis: str = "x",
        *,
        collapse_from_2d: bool = False,
        detected_rf: Mapping[str, Any] | None = None,
        rf_type: str = "excitatory",
        is_shuffle: bool = False,
        drop_bins: int = 2,
        is_center: bool = False,
        result_path: str | Path | None = None,
        show_progress: bool = True,
        **options: Any,
) -> NDArray[np.uint8]:
    """Detect RF after summing responses onto x or y (default: mean + 1 SD).

    ``collapse_from_2d=True`` purely projects the supplied ``detected_rf``
    result, loaded with ``read_rf_result``. It never starts detection or
    saves a result. Use ``sum_to_1d`` to aggregate response values instead.
    ``rf_type="inhibitory"`` selects decreased responses (mean - 0.75 SD).
    An explicit ``alternative`` overrides ``rf_type``.
    ``exclude_zero_bins=True`` filters missing and zero responses after
    summing onto the requested axis. Batch results retain a leading unit axis.
    """

    normalized_axis = _axis_name(axis)
    if _bool_value(collapse_from_2d, "collapse_from_2d"):
        from .results import _rf_result_for_map

        if detected_rf is None:
            raise ValueError("collapse_from_2d=True requires a loaded detected_rf result")
        if trials is not None or is_shuffle or result_path is not None or options:
            raise ValueError("projecting detected_rf does not accept detection or saving options")
        center_only = _bool_value(is_center, "is_center")
        matrices = [
            _rf_result_for_map(detected_rf, item, center_only=center_only)
            for item in self._rf_input_maps
        ]
        matrix = np.stack(matrices) if self._rf_is_batch else matrices[0]
    else:
        if detected_rf is not None:
            raise ValueError("detected_rf requires collapse_from_2d=True")
        center_only = _bool_value(is_center, "is_center")
        progress = _bool_value(show_progress, "show_progress")
        options.setdefault("cluster_forming_z", 0.75 if rf_type == "inhibitory" else 1.0)
        mask, center = _rf_output_arrays(
            cache=self._rf_result_cache,
            maps=self._rf_input_maps,
            is_batch=self._rf_is_batch,
            trials=trials,
            detect=self._detect_rf,
            rf_type=rf_type,
            is_shuffle=is_shuffle,
            drop_bins=drop_bins,
            result_path=result_path,
            show_progress=progress,
            center_progress=progress and center_only,
            options=options,
            collapse_axis=normalized_axis,
        )
        matrix = center if center_only else mask
    collapsed_axis = (0 if normalized_axis == "x" else 1) + int(self._rf_is_batch)
    return _readonly_array(
        np.any(matrix != 0, axis=collapsed_axis),
        dtype=np.uint8,
    )



def _detect_trials(
    trials: Mapping[str, Any], *, maps: Sequence["RFMap"], is_batch: bool,
    is_shuffle: bool = True, drop_bins: int = 1, show_progress: bool = True,
    _aligned: bool = False, **options: Any,
) -> dict[str, Any]:
    """Adapt prepared map/trial inputs to the shared numeric detector."""
    from ._detector import detect_rf as detect_arrays

    aligned = trials if _aligned else _aligned_trial_mapping(trials, maps)
    batch = detect_arrays(
        aligned["responses"], aligned["position_ids"], aligned["shape"],
        stratum_ids=aligned.get("stratum_ids"), unit_ids=aligned["unit_ids"],
        is_shuffle=is_shuffle, drop_bins=drop_bins,
        show_progress=_bool_value(show_progress, "show_progress"), **options,
    )
    return batch if is_batch else _single_detection_result(batch)


def detect_rf(
    maps, trials: Mapping[str, Any] | None = None, *,
    dimension: str = "2d", axis: str = "x", rf_type: str = "excitatory",
    is_shuffle: bool = False, drop_bins: int = 2,
    result_path: str | Path | None = None, show_progress: bool = False,
    **options: Any,
) -> RFResult:
    """Explicitly detect RFs in a prepared RFMap or RFMapList.

    The response window must already be reduced to one time bin. ``dimension``
    selects a 2-D detection or a direct 1-D detection after spatial summation.
    To project an existing result, use ``RFResult.project`` instead. Results
    always retain ``(unit, y, x)`` axes, including single-unit inputs. Supplying
    ``result_path`` explicitly enables disk caching/saving; otherwise this call
    performs no file I/O and stores no detection state on the input maps.
    """
    from functools import partial
    from .models import RFMap, RFMapList

    if not isinstance(maps, (RFMap, RFMapList)):
        raise TypeError("maps must be an RFMap or RFMapList")
    if dimension not in ("2d", "1d"):
        raise ValueError("dimension must be '2d' or '1d'")
    input_maps = (maps,) if isinstance(maps, RFMap) else tuple(maps)
    collapse_axis = _axis_name(axis) if dimension == "1d" else None
    if dimension == "1d":
        options.setdefault("cluster_forming_z", 0.75 if rf_type == "inhibitory" else 1.0)
    progress = _bool_value(show_progress, "show_progress")
    cache = {}
    mask, center = _rf_output_arrays(
        cache=cache, maps=input_maps, is_batch=True, trials=trials,
        detect=partial(_detect_trials, maps=input_maps, is_batch=True),
        rf_type=rf_type, is_shuffle=is_shuffle, drop_bins=drop_bins,
        result_path=result_path, show_progress=progress, center_progress=progress,
        options=options, collapse_axis=collapse_axis,
    )
    return RFResult(
        mask_2d=mask, center_2d=center,
        unit_ids=[item.unit_id for item in input_maps],
        manifest=cache["manifest"], cache_key=cache["result_key"],
    )
