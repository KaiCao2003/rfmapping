"""Object model for per-unit RF mapping JSON data."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from Utils.json_tools import read_formatted_json

__all__ = ["RFMap", "RFMapList", "asrfmap", "load_rf_maps"]


_EDGE_ATOL_S = 1e-12
_STRUCTURAL_JSON_FIELDS = {
    "unitsSpikeCounts",
    "unitsSpikeCountsSize",
    "unitPool",
    "xPositions",
    "yPositions",
    "timeBinEdges",
    "stimulusPresentationCounts",
}


def _readonly_array(values: Any, *, dtype: Any | None = None) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _integer(value: Any, label: str) -> int:
    parsed = _number(value, label)
    if not parsed.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(parsed)


def _flat_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list) or any(
        isinstance(item, (list, dict)) for item in value
    ):
        raise ValueError(f"{label} must be a one-dimensional array")
    return value


def _counts_are_numeric(value: Any) -> bool:
    if isinstance(value, list):
        for child in value:
            if not _counts_are_numeric(child):
                return False
        return True
    return isinstance(value, Real) and not isinstance(value, bool)


def _presentation_matrix(value: Any, n_y: int, n_x: int) -> NDArray[np.float64]:
    if isinstance(value, Real) and not isinstance(value, bool):
        if n_y != 1 or n_x != 1:
            raise ValueError("stimulusPresentationCounts must be a y-by-x array")
        rows: list[list[Any]] = [[value]]
    elif isinstance(value, list):
        if all(not isinstance(item, list) for item in value):
            if n_y == 1 and len(value) == n_x:
                rows = [value]
            elif n_x == 1 and len(value) == n_y:
                rows = [[item] for item in value]
            else:
                raise ValueError(
                    "stimulusPresentationCounts dimensions do not match "
                    "unitsSpikeCountsSize"
                )
        elif all(isinstance(item, list) for item in value):
            rows = value
        else:
            raise ValueError(
                "stimulusPresentationCounts must be a rectangular y-by-x array"
            )
    else:
        raise ValueError("stimulusPresentationCounts must be a y-by-x array")

    if len(rows) != n_y:
        raise ValueError(
            "stimulusPresentationCounts y dimension does not match "
            "unitsSpikeCountsSize"
        )
    if any(len(row) != n_x for row in rows):
        raise ValueError(
            "stimulusPresentationCounts x dimension does not match "
            "unitsSpikeCountsSize"
        )

    normalized = np.empty((n_y, n_x), dtype=float)
    for y_index, row in enumerate(rows):
        for x_index, item in enumerate(row):
            parsed = _number(
                item,
                f"stimulusPresentationCounts[{y_index}][{x_index}]",
            )
            if parsed < 0 or not parsed.is_integer():
                raise ValueError(
                    "stimulusPresentationCounts values must be non-negative integers"
                )
            normalized[y_index, x_index] = parsed
    normalized.setflags(write=False)
    return normalized


def _coerce_lookup_integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc


def _axis_name(axis: str) -> str:
    if not isinstance(axis, str):
        raise ValueError("axis must be 'x' or 'y'")
    normalized = axis.strip().lower()
    if normalized not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    return normalized


def _bool_value(value: bool, label: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be bool")
    return bool(value)


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

    row_indices: list[int] = []
    for rf_map in maps:
        matches = np.flatnonzero(unit_ids == rf_map.unit_id)
        if matches.size == 0:
            raise KeyError(f"trial responses do not contain unit_id {rf_map.unit_id}")
        row_indices.append(int(matches[0]))

    aligned = dict(trials)
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


@dataclass(frozen=True, slots=True, repr=False)
class RFMap:
    """RF mapping data for one unit.

    ``spike_counts`` always has axes ``(y, x, time_bin)``. Time-summed maps
    retain a singleton time dimension so they remain RFMap objects.
    """

    unit_index: int
    unit_id: int
    spike_counts: NDArray[Any]
    x_positions: NDArray[np.float64]
    y_positions: NDArray[np.float64]
    time_bin_edges_s: NDArray[np.float64]
    presentation_counts: NDArray[np.float64] | None
    metadata: Mapping[str, Any]
    source_path: Path

    def __repr__(self) -> str:
        return (
            f"RFMap(unit_index={self.unit_index}, unit_id={self.unit_id}, "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"time_window_s={self.time_window_s}, "
            f"source_path={str(self.source_path)!r})"
        )

    # Public data and geometry properties

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.spike_counts.shape

    @property
    def axes(self) -> tuple[str, str, str]:
        return ("y", "x", "time")

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.spike_counts.dtype

    @property
    def n_y(self) -> int:
        return self.spike_counts.shape[0]

    @property
    def n_x(self) -> int:
        return self.spike_counts.shape[1]

    @property
    def n_time_bins(self) -> int:
        return self.spike_counts.shape[2]

    @property
    def time_window_s(self) -> tuple[float, float]:
        return (float(self.time_bin_edges_s[0]), float(self.time_bin_edges_s[-1]))

    @property
    def duration_s(self) -> float:
        return self.time_window_s[1] - self.time_window_s[0]

    @property
    def time_bin_centers_s(self) -> NDArray[np.float64]:
        centers = (self.time_bin_edges_s[:-1] + self.time_bin_edges_s[1:]) / 2
        centers.setflags(write=False)
        return centers

    @property
    def time_bin_widths_s(self) -> NDArray[np.float64]:
        widths = np.diff(self.time_bin_edges_s)
        widths.setflags(write=False)
        return widths

    # Data inspection helpers

    def summary(self) -> dict[str, Any]:
        """Return a compact description without printing the full count array."""

        return {
            "unit_index": self.unit_index,
            "unit_id": self.unit_id,
            "shape": self.shape,
            "axes": self.axes,
            "dtype": self.dtype,
            "time_window_s": self.time_window_s,
            "duration_s": self.duration_s,
            "has_presentation_counts": self.presentation_counts is not None,
            "metadata_keys": tuple(self.metadata),
            "public_data": [
                "unit_index",
                "unit_id",
                "spike_counts",
                "x_positions",
                "y_positions",
                "time_bin_edges_s",
                "presentation_counts",
                "metadata",
                "source_path",
            ],
            "source_path": self.source_path,
        }

    # Array conversion

    def where(self, value: Real) -> tuple[NDArray[np.intp], ...]:
        """Return native-axis indices whose spike count equals ``value``.

        This is equivalent to ``np.where(self.spike_counts == value)`` and
        returns ``(y, x, time)`` index arrays. A summed RFMap still retains a
        singleton time axis, so its returned time indices are all zero.
        """

        _number(value, "value")
        return tuple(
            _readonly_array(indices, dtype=np.intp)
            for indices in np.where(self.spike_counts == value)
        )

    def to_2d_array(self) -> NDArray[Any]:
        """Return a read-only ``(y, x)`` array from a time-summed RFMap.

        This method never silently sums or selects from a multi-bin timeline.
        Call :meth:`sum` first when the object contains more than one time bin.
        """

        if self.n_time_bins != 1:
            raise ValueError(
                "to_2d_array() requires exactly one time bin; call "
                "rf_map.sum(earlier_s, later_s) first"
            )
        result = self.spike_counts[..., 0]
        result.setflags(write=False)
        return result

    def to_1d_array(self, axis: str = "x") -> NDArray[Any]:
        """Project a summed 2-D count map onto horizontal x or vertical y."""

        normalized_axis = _axis_name(axis)
        matrix = self.to_2d_array()
        collapsed_axis = 0 if normalized_axis == "x" else 1
        return _readonly_array(matrix.sum(axis=collapsed_axis))

    # Time-window operations

    def _available_edges_message(self) -> str:
        return f"Available time bin edges (s): {self.time_bin_edges_s.tolist()}"

    def _coerce_time(self, value: float, label: str) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"{label} must be a finite number. {self._available_edges_message()}"
            )
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{label} must be a finite number. {self._available_edges_message()}"
            ) from exc
        if not math.isfinite(parsed):
            raise ValueError(
                f"{label} must be a finite number. {self._available_edges_message()}"
            )
        return parsed

    def _edge_index(self, value: float, label: str) -> int:
        exact_matches = np.flatnonzero(self.time_bin_edges_s == value)
        if exact_matches.size:
            # Source edges are unique. A zero-width derived RFMap deliberately
            # stores [t, t]; choosing the first copy makes [t, t) empty.
            return int(exact_matches[0])

        matches = np.flatnonzero(
            np.isclose(
                self.time_bin_edges_s,
                value,
                rtol=0.0,
                atol=_EDGE_ATOL_S,
            )
        )
        if matches.size == 0:
            raise ValueError(
                f"{label}={value!r} is not in timeBinEdges. "
                f"{self._available_edges_message()}"
            )
        if matches.size > 1:
            matching_edges = self.time_bin_edges_s[matches].tolist()
            raise ValueError(
                f"{label}={value!r} is within {_EDGE_ATOL_S:g} s of multiple "
                f"timeBinEdges {matching_edges}; use an exact edge value. "
                f"{self._available_edges_message()}"
            )
        return int(matches[0])

    def _time_indices(
        self,
        earlier_s: float,
        later_s: float,
        *,
        allow_empty: bool,
    ) -> tuple[int, int, float, float]:
        earlier = self._coerce_time(earlier_s, "earlier_s")
        later = self._coerce_time(later_s, "later_s")
        if later < earlier or (not allow_empty and later == earlier):
            relation = ">=" if allow_empty else ">"
            raise ValueError(
                f"later_s must be {relation} earlier_s. "
                f"Received earlier_s={earlier!r}, later_s={later!r}. "
                f"{self._available_edges_message()}"
            )

        start_index = self._edge_index(earlier, "earlier_s")
        stop_index = self._edge_index(later, "later_s")
        if stop_index < start_index or (not allow_empty and stop_index == start_index):
            relation = ">=" if allow_empty else ">"
            raise ValueError(
                f"later_s must resolve to an edge {relation} the earlier_s edge. "
                f"{self._available_edges_message()}"
            )
        return (
            start_index,
            stop_index,
            float(self.time_bin_edges_s[start_index]),
            float(self.time_bin_edges_s[stop_index]),
        )

    def sum(self, earlier_s: float, later_s: float) -> RFMap:
        """Return this unit summed over the half-open interval [earlier, later)."""

        start, stop, canonical_start, canonical_stop = self._time_indices(
            earlier_s,
            later_s,
            allow_empty=True,
        )
        summed_counts = self.spike_counts[..., start:stop].sum(
            axis=-1,
            keepdims=True,
        )
        summed_edges = np.asarray(
            [canonical_start, canonical_stop],
            dtype=float,
        )
        summed_metadata = deepcopy(dict(self.metadata))
        if "VSTimeWindow" in summed_metadata:
            summed_metadata["VSTimeWindow"] = [canonical_start, canonical_stop]
        if "timeWindowMs" in summed_metadata:
            summed_metadata["timeWindowMs"] = [
                canonical_start * 1000.0,
                canonical_stop * 1000.0,
            ]
        if "timeBinWidthMs" in summed_metadata:
            summed_metadata["timeBinWidthMs"] = (
                canonical_stop - canonical_start
            ) * 1000.0

        return _make_rf_map(
            unit_index=self.unit_index,
            unit_id=self.unit_id,
            spike_counts=summed_counts,
            x_positions=self.x_positions,
            y_positions=self.y_positions,
            time_bin_edges_s=summed_edges,
            presentation_counts=self.presentation_counts,
            metadata=summed_metadata,
            source_path=self.source_path,
        )

    # RF detection

    def detect_rf(
        self,
        trials: Mapping[str, Any],
        *,
        is_shuffle: bool = True,
        show_progress: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        """Return plain cluster-permutation result arrays for this unit."""

        from Utils.rf_detection import detect_rf

        progress = _bool_value(show_progress, "show_progress")
        aligned = _aligned_trial_mapping(trials, (self,))
        batch = detect_rf(
            aligned["responses"],
            aligned["position_ids"],
            (self.n_y, self.n_x),
            stratum_ids=aligned.get("stratum_ids"),
            unit_ids=aligned["unit_ids"],
            is_shuffle=is_shuffle,
            show_progress=progress,
            **options,
        )
        return _single_detection_result(batch)

    def rf_2d(
        self,
        trials: Mapping[str, Any],
        *,
        is_shuffle: bool = True,
        return_center: bool = False,
        show_progress: bool = True,
        **options: Any,
    ) -> NDArray[np.uint8]:
        """Return the final 2-D RF mask or its discrete weighted center."""

        center_only = _bool_value(return_center, "return_center")
        progress = _bool_value(show_progress, "show_progress")
        result = self.detect_rf(
            trials,
            is_shuffle=is_shuffle,
            show_progress=progress,
            **options,
        )
        mask = result["final_mask"]
        if center_only:
            mask = _center_only_mask(
                result,
                alternative=options.get("alternative", "greater"),
                wrap_x=bool(options.get("wrap_x", True)),
                show_progress=progress,
            )
        return _readonly_array(
            mask,
            dtype=np.uint8,
        )

    def rf_1d(
        self,
        trials: Mapping[str, Any],
        axis: str = "x",
        *,
        is_shuffle: bool = True,
        return_center: bool = False,
        show_progress: bool = True,
        **options: Any,
    ) -> NDArray[np.uint8]:
        """Project this unit's final 2-D RF mask onto x or y."""

        normalized_axis = _axis_name(axis)
        matrix = self.rf_2d(
            trials,
            is_shuffle=is_shuffle,
            return_center=return_center,
            show_progress=show_progress,
            **options,
        )
        collapsed_axis = 0 if normalized_axis == "x" else 1
        return _readonly_array(
            np.any(matrix != 0, axis=collapsed_axis),
            dtype=np.uint8,
        )


class RFMapList(Sequence[RFMap]):
    """Ordered per-unit RFMaps with numeric-string unit-ID lookup."""

    __slots__ = (
        "_maps",
        "_maps_by_unit_id",
        "_maps_by_unit_index",
        "source_path",
    )

    def __init__(self, maps: Sequence[RFMap], source_path: str | Path):
        self._maps = list(maps)
        if not self._maps:
            raise ValueError("RFMapList requires at least one RFMap")
        self.source_path = Path(source_path)
        self._maps_by_unit_id = {rf_map.unit_id: rf_map for rf_map in self._maps}
        if len(self._maps_by_unit_id) != len(self._maps):
            raise ValueError("RFMap unit IDs must be unique")
        self._maps_by_unit_index = {
            rf_map.unit_index: rf_map for rf_map in self._maps
        }
        if len(self._maps_by_unit_index) != len(self._maps):
            raise ValueError("RFMap original unit indices must be unique")

    def __repr__(self) -> str:
        unit_ids = self.unit_ids
        if len(unit_ids) <= 8:
            unit_ids_text = repr(unit_ids)
        else:
            shown = ", ".join(str(unit_id) for unit_id in unit_ids[:6])
            unit_ids_text = f"[{shown}, ..., {unit_ids[-1]}]"
        return (
            f"RFMapList(n_units={self.n_units}, unit_ids={unit_ids_text}, "
            f"shape={self.shape}, source_path={str(self.source_path)!r})"
        )

    # Sequence and unit lookup

    def __len__(self) -> int:
        return len(self._maps)

    def __iter__(self) -> Iterator[RFMap]:
        return iter(self._maps)

    @overload
    def __getitem__(self, index: int) -> RFMap: ...

    @overload
    def __getitem__(self, index: slice) -> list[RFMap]: ...

    @overload
    def __getitem__(self, index: str) -> RFMap: ...

    def __getitem__(self, index: int | slice | str) -> RFMap | list[RFMap]:
        if isinstance(index, str):
            try:
                unit_id = int(index)
            except ValueError as exc:
                raise KeyError(
                    f"unit_id key {index!r} must be an integer string"
                ) from exc
            return self.by_unit_id(unit_id)
        return self._maps[index]

    def by_index(self, unit_index: int) -> RFMap:
        """Return a unit by its original index in the JSON file."""

        index = _coerce_lookup_integer(unit_index, "unit_index")
        try:
            return self._maps_by_unit_index[index]
        except KeyError as exc:
            available_indices = sorted(self._maps_by_unit_index)
            raise IndexError(
                f"unit_index {index} is unavailable. Available original unit "
                f"indices: {available_indices}"
            ) from exc

    def by_unit_id(self, unit_id: int) -> RFMap:
        """Return a unit by cluster/unit ID."""

        parsed_id = _coerce_lookup_integer(unit_id, "unit_id")
        try:
            return self._maps_by_unit_id[parsed_id]
        except KeyError as exc:
            raise KeyError(
                f"unit_id {parsed_id} is unavailable. Available unit IDs: "
                f"{self.unit_ids}"
            ) from exc

    # Public list properties

    @property
    def n_units(self) -> int:
        return len(self._maps)

    @property
    def unit_indices(self) -> list[int]:
        return [rf_map.unit_index for rf_map in self]

    @property
    def unit_ids(self) -> list[int]:
        return [rf_map.unit_id for rf_map in self]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self.n_units, *self._maps[0].shape)

    # Time-window and array conversion

    def sum(
        self,
        earlier_s: float,
        later_s: float,
        *,
        show_progress: bool = True,
    ) -> RFMapList:
        """Sum the requested time window independently for every unit."""

        progress = _bool_value(show_progress, "show_progress")
        maps: Any = self._maps
        if progress:
            maps = tqdm(
                self._maps,
                desc="Sum",
                unit="unit",
            )
        return RFMapList(
            [rf_map.sum(earlier_s, later_s) for rf_map in maps],
            self.source_path,
        )

    def to_4d_array(self) -> NDArray[Any]:
        """Stack units as a read-only ``(unit, y, x, time)`` array."""

        return _readonly_array(np.stack([rf_map.spike_counts for rf_map in self]))

    def where(self, value: Real) -> tuple[NDArray[np.intp], ...]:
        """Return native-axis indices whose spike count equals ``value``.

        This is equivalent to ``np.where(self.to_4d_array() == value)`` and
        returns ``(unit, y, x, time)`` index arrays. The first array contains
        RFMapList positions, not recorded unit IDs, and can contain duplicates
        when one unit has multiple matching bins.
        """

        _number(value, "value")
        return tuple(
            _readonly_array(indices, dtype=np.intp)
            for indices in np.where(self.to_4d_array() == value)
        )

    def to_2d_array(self) -> NDArray[Any]:
        """Stack every unit's 2-D count map as ``(unit, y, x)``."""

        return _readonly_array(np.stack([rf_map.to_2d_array() for rf_map in self]))

    def to_1d_array(self, axis: str = "x") -> NDArray[Any]:
        """Stack every unit's 1-D projection as ``(unit, x)`` or ``(unit, y)``."""

        return _readonly_array(
            np.stack([rf_map.to_1d_array(axis=axis) for rf_map in self])
        )

    # RF detection

    def _detect_rf(
        self,
        trials: Mapping[str, Any],
        *,
        is_shuffle: bool = True,
        show_progress: bool = True,
        **options: Any,
    ) -> dict[str, Any]:
        """Return plain cluster-permutation arrays for every unit."""

        from Utils.rf_detection import detect_rf

        progress = _bool_value(show_progress, "show_progress")
        aligned = _aligned_trial_mapping(trials, self._maps)
        first = self._maps[0]
        return detect_rf(
            aligned["responses"],
            aligned["position_ids"],
            (first.n_y, first.n_x),
            stratum_ids=aligned.get("stratum_ids"),
            unit_ids=aligned["unit_ids"],
            is_shuffle=is_shuffle,
            show_progress=progress,
            **options,
        )

    def rf_2d(
        self,
        trials: Mapping[str, Any],
        *,
        is_shuffle: bool = True,
        return_center: bool = False,
        show_progress: bool = True,
        **options: Any,
    ) -> NDArray[np.uint8]:
        """Stack final 2-D RF masks or discrete centers by unit."""

        center_only = _bool_value(return_center, "return_center")
        progress = _bool_value(show_progress, "show_progress")
        result = self._detect_rf(
            trials,
            is_shuffle=is_shuffle,
            show_progress=progress,
            **options,
        )
        mask = result["final_mask"]
        if center_only:
            mask = _center_only_mask(
                result,
                alternative=options.get("alternative", "greater"),
                wrap_x=bool(options.get("wrap_x", True)),
                show_progress=progress,
            )
        return _readonly_array(
            mask,
            dtype=np.uint8,
        )

    def rf_1d(
        self,
        trials: Mapping[str, Any],
        axis: str = "x",
        *,
        is_shuffle: bool = True,
        return_center: bool = False,
        show_progress: bool = True,
        **options: Any,
    ) -> NDArray[np.uint8]:
        """Project the same final 2-D RF results onto x or y."""

        normalized_axis = _axis_name(axis)
        matrix = self.rf_2d(
            trials,
            is_shuffle=is_shuffle,
            return_center=return_center,
            show_progress=show_progress,
            **options,
        )
        collapsed_axis = 1 if normalized_axis == "x" else 2
        return _readonly_array(
            np.any(matrix != 0, axis=collapsed_axis),
            dtype=np.uint8,
        )


def _make_rf_map(
    *,
    unit_index: int,
    unit_id: int,
    spike_counts: Any,
    x_positions: Any,
    y_positions: Any,
    time_bin_edges_s: Any,
    presentation_counts: NDArray[np.float64] | None,
    metadata: Mapping[str, Any],
    source_path: str | Path,
) -> RFMap:
    return RFMap(
        unit_index=int(unit_index),
        unit_id=int(unit_id),
        spike_counts=_readonly_array(spike_counts),
        x_positions=_readonly_array(x_positions, dtype=float),
        y_positions=_readonly_array(y_positions, dtype=float),
        time_bin_edges_s=_readonly_array(time_bin_edges_s, dtype=float),
        presentation_counts=presentation_counts,
        metadata=MappingProxyType(deepcopy(dict(metadata))),
        source_path=Path(source_path),
    )


def asrfmap(
    array: Any,
    *,
    start_time: float = 0.0,
    end_time: float | None = None,
    time_bin: float | None = None,
) -> RFMap:
    """Convert a 2-D or 3-D numeric array into one :class:`RFMap`.

    Array axes are ``(y, x)`` or ``(y, x, time)``. A 2-D input is promoted to
    one time bin. ``start_time``, ``end_time``, and ``time_bin`` are measured
    in seconds. When neither ``end_time`` nor ``time_bin`` is supplied, the
    time bins use unit-width index edges. Supplying either one derives the
    other; supplying both verifies that ``time_bin * n_time_bins`` equals
    ``end_time - start_time`` within the module's edge tolerance.

    The returned object owns a read-only copy of the input. Spatial positions
    default to zero-based array indices, and unit identifiers default to zero.
    """

    try:
        spike_counts = np.array(array, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Unable to convert array to RFMap: {exc}") from exc

    if spike_counts.ndim not in {2, 3}:
        raise ValueError(
            "array must be 2-D (y, x) or 3-D (y, x, time)"
        )
    if any(size <= 0 for size in spike_counts.shape):
        raise ValueError("array dimensions must be positive")
    if (
        np.issubdtype(spike_counts.dtype, np.bool_)
        or not np.issubdtype(spike_counts.dtype, np.number)
        or np.issubdtype(spike_counts.dtype, np.complexfloating)
    ):
        raise ValueError("array values must be real numeric values")
    if not np.all(np.isfinite(spike_counts)):
        raise ValueError("array values must be finite")
    if np.any(spike_counts < 0):
        raise ValueError("array values must be non-negative")

    if spike_counts.ndim == 2:
        spike_counts = spike_counts[..., np.newaxis]
    n_y, n_x, n_time_bins = spike_counts.shape

    parsed_start = _number(start_time, "start_time")
    parsed_end = None if end_time is None else _number(end_time, "end_time")
    parsed_time_bin = (
        None if time_bin is None else _number(time_bin, "time_bin")
    )

    if parsed_time_bin is not None and parsed_time_bin <= 0:
        raise ValueError("time_bin must be greater than zero")

    if parsed_end is None:
        effective_time_bin = 1.0 if parsed_time_bin is None else parsed_time_bin
        parsed_end = parsed_start + effective_time_bin * n_time_bins
        if not math.isfinite(parsed_end):
            raise ValueError("derived end_time must be finite")
    else:
        duration = parsed_end - parsed_start
        if not math.isfinite(duration):
            raise ValueError("end_time - start_time must be finite")
        if duration <= 0:
            raise ValueError("end_time must be greater than start_time")
        if parsed_time_bin is None:
            effective_time_bin = duration / n_time_bins
        else:
            effective_time_bin = parsed_time_bin
            expected_duration = effective_time_bin * n_time_bins
            if not math.isclose(
                expected_duration,
                duration,
                rel_tol=0.0,
                abs_tol=_EDGE_ATOL_S,
            ):
                raise ValueError(
                    "time_bin * n_time_bins must equal end_time - start_time; "
                    f"{effective_time_bin:g} * {n_time_bins} = "
                    f"{expected_duration:g}, but end_time - start_time = "
                    f"{duration:g}"
                )

    time_bin_edges_s = np.linspace(
        parsed_start,
        parsed_end,
        n_time_bins + 1,
        dtype=float,
    )
    if not np.all(np.diff(time_bin_edges_s) > 0):
        raise ValueError(
            "time bin edges are not strictly increasing at float precision"
        )

    return _make_rf_map(
        unit_index=0,
        unit_id=0,
        spike_counts=spike_counts,
        x_positions=np.arange(n_x, dtype=float),
        y_positions=np.arange(n_y, dtype=float),
        time_bin_edges_s=time_bin_edges_s,
        presentation_counts=None,
        metadata={},
        source_path=Path("<array>"),
    )


def load_rf_maps(path: str | Path) -> RFMapList:
    """Load one RF mapping JSON file into ordered, per-unit RFMap objects."""

    source_path = Path(path)
    raw = read_formatted_json(source_path)
    if not isinstance(raw, dict):
        raise ValueError("RF mapping JSON must contain an object at the top level")
    required = {
        "unitsSpikeCounts",
        "unitsSpikeCountsSize",
        "unitPool",
        "xPositions",
        "yPositions",
        "timeBinEdges",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing JSON keys: {', '.join(missing)}")

    size_values = _flat_list(raw["unitsSpikeCountsSize"], "unitsSpikeCountsSize")
    if len(size_values) != 4:
        raise ValueError("unitsSpikeCountsSize must contain four values")
    shape = tuple(
        _integer(value, "unitsSpikeCountsSize value") for value in size_values
    )
    if any(value <= 0 for value in shape):
        raise ValueError("unitsSpikeCountsSize values must be positive")
    n_units, n_y, n_x, n_time_bins = shape

    if not _counts_are_numeric(raw["unitsSpikeCounts"]):
        raise ValueError(
            "unitsSpikeCounts contains a value that is not numeric "
            "(JSON numbers only; bool is invalid)"
        )
    try:
        spike_counts = np.asarray(raw["unitsSpikeCounts"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Unable to parse unitsSpikeCounts: {exc}") from exc
    if spike_counts.shape != shape:
        raise ValueError(
            f"unitsSpikeCounts has shape {spike_counts.shape}, expected {shape}"
        )
    if not np.all(np.isfinite(spike_counts)) or np.any(spike_counts < 0):
        raise ValueError("unitsSpikeCounts values must be finite and non-negative")
    spike_counts.setflags(write=False)

    unit_pool = tuple(
        _integer(value, "unitPool value")
        for value in _flat_list(raw["unitPool"], "unitPool")
    )
    if len(unit_pool) != n_units:
        raise ValueError("unitPool length does not match unit count")
    if len(set(unit_pool)) != len(unit_pool):
        raise ValueError("unitPool must contain unique unit IDs")

    x_positions = _readonly_array(
        [
            _number(value, "xPositions value")
            for value in _flat_list(raw["xPositions"], "xPositions")
        ],
        dtype=float,
    )
    y_positions = _readonly_array(
        [
            _number(value, "yPositions value")
            for value in _flat_list(raw["yPositions"], "yPositions")
        ],
        dtype=float,
    )
    time_bin_edges_s = _readonly_array(
        [
            _number(value, "timeBinEdges value")
            for value in _flat_list(raw["timeBinEdges"], "timeBinEdges")
        ],
        dtype=float,
    )
    if len(x_positions) != n_x:
        raise ValueError("xPositions length does not match x dimension")
    if len(y_positions) != n_y:
        raise ValueError("yPositions length does not match y dimension")
    if len(time_bin_edges_s) != n_time_bins + 1:
        raise ValueError("timeBinEdges must contain nTimeBins + 1 edges")
    if not np.all(np.diff(time_bin_edges_s) > 0):
        raise ValueError("timeBinEdges must be strictly increasing")

    presentation_counts = None
    if "stimulusPresentationCounts" in raw:
        presentation_counts = _presentation_matrix(
            raw["stimulusPresentationCounts"],
            n_y,
            n_x,
        )
        zero_presentations = presentation_counts == 0
        if np.any(spike_counts[:, zero_presentations, :] != 0):
            raise ValueError(
                "stimulusPresentationCounts is zero where spike counts are nonzero"
            )

    metadata = {
        key: deepcopy(value)
        for key, value in raw.items()
        if key not in _STRUCTURAL_JSON_FIELDS
    }
    maps = [
        _make_rf_map(
            unit_index=unit_index,
            unit_id=unit_id,
            spike_counts=spike_counts[unit_index],
            x_positions=x_positions,
            y_positions=y_positions,
            time_bin_edges_s=time_bin_edges_s,
            presentation_counts=presentation_counts,
            metadata=metadata,
            source_path=source_path,
        )
        for unit_index, unit_id in enumerate(unit_pool)
    ]
    return RFMapList(maps, source_path)
