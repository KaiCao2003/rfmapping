"""RF response objects and explicit transformations of their values."""
from __future__ import annotations
import math
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, overload
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from ._compat import _RFDetectionCompatibility
from ._validation import (
    _EDGE_ATOL_S, _axis_name, _bool_value, _coerce_lookup_integer,
    _number, _presentation_matrix, _readonly_array,
)

__all__ = ["RFMap", "RFMapList", "asrfmap"]

@dataclass(frozen=True, slots=True, repr=False)
class RFMap(_RFDetectionCompatibility):
    """RF mapping data for one unit.

    ``spike_counts`` always has axes ``(y, x, time_bin)`` and retains its
    historical field name after explicit conversion to firing rates. Time-summed maps keep a singleton time dimension.
    """

    _rf_is_batch = False

    unit_index: int
    unit_id: int
    spike_counts: NDArray[Any]
    x_positions: NDArray[np.float64]
    y_positions: NDArray[np.float64]
    time_bin_edges_s: NDArray[np.float64]
    presentation_counts: NDArray[np.float64] | None
    metadata: Mapping[str, Any]
    source_path: Path
    _rf_result_cache: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __repr__(self) -> str:
        return (
            f"RFMap(unit_index={self.unit_index}, unit_id={self.unit_id}, "
            f"shape={self.shape}, dtype={self.dtype}, "
            f"time_window_s={self.time_window_s}, "
            f"source_path={str(self.source_path)!r})"
        )

    def __call__(
        self,
        earlier_s: float | None = None,
        later_s: float | None = None,
    ) -> RFMap:
        """Sum a time window, defaulting omitted bounds to available edges."""

        start = self.time_window_s[0] if earlier_s is None else earlier_s
        stop = self.time_window_s[1] if later_s is None else later_s
        return self.sum(start, stop)

    def __sub__(self, other: object) -> RFMap:
        """Subtract compatible singleton-bin maps element by element.

        Call :meth:`sum` on both operands first. The returned map keeps the
        left operand's time window and may contain negative values.
        """

        if not isinstance(other, RFMap):
            return NotImplemented

        if self.n_time_bins != 1 or other.n_time_bins != 1:
            raise ValueError(
                "RFMap subtraction requires exactly one time bin in each "
                "operand; call sum() first"
            )
        if self.unit_id != other.unit_id:
            raise ValueError("RFMaps must have the same unit_id")
        if self.shape[:2] != other.shape[:2]:
            raise ValueError("RFMaps must have the same spatial shape")
        if not np.array_equal(self.x_positions, other.x_positions):
            raise ValueError("RFMaps must have identical x positions")
        if not np.array_equal(self.y_positions, other.y_positions):
            raise ValueError("RFMaps must have identical y positions")
        if (self.presentation_counts is None) != (
            other.presentation_counts is None
        ):
            raise ValueError("RFMaps must have identical presentation counts")
        if (
            self.presentation_counts is not None
            and not np.array_equal(
                self.presentation_counts,
                other.presentation_counts,
            )
        ):
            raise ValueError("RFMaps must have identical presentation counts")
        difference = (
            np.asarray(self.spike_counts, dtype=np.float64)
            - np.asarray(other.spike_counts, dtype=np.float64)
        )

        metadata = deepcopy(dict(self.metadata))
        metadata.update(
            {
                "operation": "rfmap_subtraction",
                "lhs_time_window_s": list(self.time_window_s),
                "rhs_time_window_s": list(other.time_window_s),
                "lhs_source_path": str(self.source_path),
                "rhs_source_path": str(other.source_path),
            }
        )

        return _make_rf_map(
            unit_index=self.unit_index,
            unit_id=self.unit_id,
            spike_counts=difference,
            x_positions=self.x_positions,
            y_positions=self.y_positions,
            time_bin_edges_s=self.time_bin_edges_s,
            presentation_counts=self.presentation_counts,
            metadata=metadata,
            source_path=Path("<difference>"),
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
        """Return native-axis indices whose loaded value equals ``value``.

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
        """Extract a prepared singleton spatial axis without aggregation."""

        normalized_axis = _axis_name(axis)
        matrix = self.to_2d_array()
        collapsed_axis = 0 if normalized_axis == "x" else 1
        if matrix.shape[collapsed_axis] != 1:
            raise ValueError("to_1d_array() requires a singleton spatial axis; call sum_to_1d() first")
        return _readonly_array(np.squeeze(matrix, axis=collapsed_axis))

    def sum_to_1d(
        self,
        axis: str = "x",
        *,
        rf_only: bool = False,
        detected_rf: Mapping[str, Any] | None = None,
    ) -> RFMap:
        """Sum whole rows/columns, preserving time bins and response units.

        ``axis`` is the retained coordinate. With ``rf_only=True``, a row or
        column contributes when any of its bins belongs to the supplied 2-D
        RF. No detection, temporal aggregation, or rate conversion is run.
        Missing contributing values propagate; an empty RF yields NaN values
        and zero exposure, rather than an observed zero response.
        """
        axis = _axis_name(axis)
        rf_only = _bool_value(rf_only, "rf_only")
        collapsed_axis = 0 if axis == "x" else 1
        selected = np.ones(self.shape[collapsed_axis], dtype=bool)
        if rf_only:
            from .results import _rf_result_for_map

            if detected_rf is None:
                raise ValueError("rf_only=True requires a loaded detected_rf result")
            mask = _rf_result_for_map(detected_rf, self)
            selected = np.any(mask, axis=1 - collapsed_axis)
        elif detected_rf is not None:
            raise ValueError("detected_rf requires rf_only=True")

        def collapse(values):
            return np.compress(selected, values, axis=collapsed_axis).sum(
                axis=collapsed_axis, keepdims=True,
            )

        values = collapse(self.spike_counts)
        if not selected.any():
            values = np.full(values.shape, np.nan)
        presentations = None if self.presentation_counts is None else collapse(self.presentation_counts)
        metadata = deepcopy(dict(self.metadata))
        metadata["spatialAggregation"] = {
            "operation": "sum", "retained_axis": axis, "rf_only": rf_only,
            "selected_indices": np.flatnonzero(selected).tolist(),
            "x_positions": self.x_positions.tolist(),
            "y_positions": self.y_positions.tolist(),
        }
        if rf_only:
            metadata["spatialAggregation"]["rf_cache_key"] = detected_rf.get("cache_key")
        if "occupancyTimeSec" in metadata:
            occupancy = np.asarray(metadata["occupancyTimeSec"]).reshape(self.n_y, self.n_x)
            metadata["occupancyTimeSec"] = collapse(occupancy).tolist()
        # Zero denotes the collapsed coordinate, not a physical RF center.
        return _make_rf_map(
            unit_index=self.unit_index, unit_id=self.unit_id, spike_counts=values,
            x_positions=self.x_positions if axis == "x" else [0.0],
            y_positions=[0.0] if axis == "x" else self.y_positions,
            time_bin_edges_s=self.time_bin_edges_s,
            presentation_counts=presentations, metadata=metadata,
            source_path=self.source_path,
        )

    def to_firing_rate(
        self, *, presentation_counts: Any | None = None,
        reconstruct_presentations: bool = False,
    ) -> RFMap:
        """Convert counts to Hz using supplied or stored presentation counts.

        ``reconstruct_presentations=True`` explicitly permits reading legacy
        session logs when presentations are absent. Alternatively pass the
        result of ``resolve_presentation_counts``. Already normalized Hz maps
        retain their values and missing bins.
        """
        from .rates import RATE_NORMALIZATION, counts_to_rates

        stored_rates = self.metadata.get("responseUnits") == "Hz"
        legacy_rates = stored_rates and self.metadata.get("responseNormalization") == "occupancyTimeSec"
        reconstruct = _bool_value(reconstruct_presentations, "reconstruct_presentations")
        if legacy_rates and "spatialAggregation" in self.metadata:
            raise ValueError("convert legacy rates with to_firing_rate() before sum_to_1d()")
        if stored_rates and not legacy_rates:
            if presentation_counts is not None:
                raise ValueError("already normalized Hz maps do not need presentation_counts")
            return self
        presentations = self.presentation_counts if presentation_counts is None else _presentation_matrix(
            np.asarray(presentation_counts).tolist(), self.n_y, self.n_x,
        )
        provenance = {"method": "supplied" if presentation_counts is not None else "stimulusPresentationCounts"}
        if presentations is None and reconstruct:
            from .rates import resolve_presentation_counts

            if "spatialAggregation" in self.metadata:
                raise ValueError("reconstruct presentations before sum_to_1d()")
            presentations, provenance = resolve_presentation_counts(
                self.metadata, self.source_path, self.x_positions, self.y_positions,
            )
        if presentations is None:
            raise ValueError("to_firing_rate() requires presentation_counts; reconstruct legacy exposure explicitly")
        values = self.spike_counts
        if legacy_rates:
            occupancy = self.metadata.get("occupancyTimeSec")
            if occupancy is None:
                raise ValueError("Legacy display-normalized rates require occupancyTimeSec")
            values = values * np.asarray(occupancy).reshape(self.n_y, self.n_x, 1)
        rates = counts_to_rates(values, presentations, self.time_bin_edges_s)
        metadata = deepcopy(dict(self.metadata))
        metadata["responseUnits"] = "Hz"
        metadata["responseNormalization"] = RATE_NORMALIZATION
        metadata["presentationCountSource"] = provenance
        return _make_rf_map(
            unit_index=self.unit_index, unit_id=self.unit_id, spike_counts=rates,
            x_positions=self.x_positions, y_positions=self.y_positions,
            time_bin_edges_s=self.time_bin_edges_s,
            presentation_counts=presentations, metadata=metadata, source_path=self.source_path,
        )

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
        """Sum values over [earlier, later), without changing the operation for Hz."""
        return self._aggregate_time(earlier_s, later_s, mean_rate=False)

    def mean_rate(self, earlier_s: float, later_s: float) -> RFMap:
        """Time-weighted mean Hz over [earlier, later), preserving missing bins."""
        if self.metadata.get("responseUnits") != "Hz":
            raise ValueError("mean_rate() requires Hz values; call to_firing_rate() first")
        return self._aggregate_time(earlier_s, later_s, mean_rate=True)

    def _aggregate_time(self, earlier_s: float, later_s: float, *, mean_rate: bool) -> RFMap:

        start, stop, canonical_start, canonical_stop = self._time_indices(
            earlier_s,
            later_s,
            allow_empty=not mean_rate,
        )
        if mean_rate:
            from .rates import aggregate_rate

            summed_counts = aggregate_rate(self.spike_counts, self.time_bin_edges_s, start, stop)
        else:
            summed_counts = self.spike_counts[..., start:stop].sum(axis=-1, keepdims=True)
        summed_edges = np.asarray(
            [canonical_start, canonical_stop],
            dtype=float,
        )
        summed_metadata = deepcopy(dict(self.metadata))
        summed_metadata["timeAggregation"] = "mean_rate" if mean_rate else "sum"
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


class RFMapList(_RFDetectionCompatibility, Sequence[RFMap]):
    """Ordered per-unit RFMaps with numeric-string unit-ID lookup."""

    _rf_is_batch = True

    __slots__ = (
        "_maps",
        "_maps_by_unit_id",
        "_maps_by_unit_index",
        "_rf_result_cache",
        "source_path",
    )

    def __init__(self, maps: Sequence[RFMap], source_path: str | Path):
        self._maps = list(maps)
        self._rf_result_cache: dict[str, Any] = {}
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
    def __getitem__(self, index: int) -> RFMap:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[RFMap]:
        ...

    @overload
    def __getitem__(self, index: str) -> RFMap:
        ...

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
        """Return a unit by its original index in the source file."""

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
        """Sum values in the requested window per unit, including Hz values."""

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

    def mean_rate(
        self, earlier_s: float, later_s: float, *, show_progress: bool = True,
    ) -> RFMapList:
        """Calculate each unit's explicit time-weighted mean Hz."""
        maps = tqdm(self._maps, desc="Mean rate", unit="unit") if _bool_value(
            show_progress, "show_progress",
        ) else self._maps
        return RFMapList([item.mean_rate(earlier_s, later_s) for item in maps], self.source_path)

    def to_firing_rate(
        self, *, presentation_counts: Any | None = None,
        reconstruct_presentations: bool = False,
    ) -> RFMapList:
        """Convert each unit using stored or explicitly supplied presentations."""
        reconstruct = _bool_value(reconstruct_presentations, "reconstruct_presentations")
        if reconstruct and presentation_counts is None and self[0].presentation_counts is None:
            first = self[0].to_firing_rate(reconstruct_presentations=True)
            if first is not self[0]:
                # One recording shares trial geometry: reconstruct its exposure once.
                for item in self:
                    if (item.source_path != self[0].source_path
                            or not np.array_equal(item.x_positions, self[0].x_positions)
                            or not np.array_equal(item.y_positions, self[0].y_positions)):
                        raise ValueError("presentation reconstruction requires one source and spatial grid")
                maps = [first]
                for item in self._maps[1:]:
                    converted = item.to_firing_rate(presentation_counts=first.presentation_counts)
                    metadata = deepcopy(dict(converted.metadata))
                    metadata["presentationCountSource"] = deepcopy(first.metadata["presentationCountSource"])
                    maps.append(replace(converted, metadata=MappingProxyType(metadata)))
                return RFMapList(maps, self.source_path)
        return RFMapList(
            [item.to_firing_rate(presentation_counts=presentation_counts,
                                 reconstruct_presentations=reconstruct) for item in self],
            self.source_path,
        )

    def sum_to_1d(
        self,
        axis: str = "x",
        *,
        rf_only: bool = False,
        detected_rf: Mapping[str, Any] | None = None,
    ) -> RFMapList:
        """Sum each unit's spatial rows/columns, matching detected RFs by ID."""
        return RFMapList(
            [item.sum_to_1d(axis, rf_only=rf_only, detected_rf=detected_rf) for item in self],
            self.source_path,
        )

    def to_4d_array(self) -> NDArray[Any]:
        """Stack units as a read-only ``(unit, y, x, time)`` array."""

        return _readonly_array(np.stack([rf_map.spike_counts for rf_map in self]))

    def where(self, value: Real) -> tuple[NDArray[np.intp], ...]:
        """Return native-axis indices whose loaded value equals ``value``.

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
        """Stack every unit's 2-D map as ``(unit, y, x)``."""

        return _readonly_array(np.stack([rf_map.to_2d_array() for rf_map in self]))

    def to_1d_array(self, axis: str = "x") -> NDArray[Any]:
        """Stack every unit's 1-D projection as ``(unit, x)`` or ``(unit, y)``."""

        return _readonly_array(
            np.stack([rf_map.to_1d_array(axis=axis) for rf_map in self])
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
        presentation_counts=None if presentation_counts is None else _readonly_array(presentation_counts),
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
    if np.any(np.isinf(spike_counts)):
        raise ValueError("array values must be finite or NaN")
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
