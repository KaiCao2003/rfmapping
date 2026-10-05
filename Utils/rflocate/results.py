"""Detected RF data and pure projections, independent of detection and files."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["RFResult"]

RF_RESULT_CACHE_SCHEMA_VERSION = 1
_RESULT_FIELDS = (
    "schema_version", "mask_2d", "center_2d", "unit_ids", "manifest", "cache_key",
)


def _canonical_manifest_json(manifest: Mapping[str, Any]) -> str:
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a JSON-compatible mapping")
    try:
        serialized = json.dumps(
            dict(manifest),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"manifest must be a JSON-compatible mapping: {exc}"
        ) from exc

    # Parsing once catches encoder extensions or unusual mapping keys that do
    # not round-trip as one ordinary JSON object.
    parsed = json.loads(serialized)
    if not isinstance(parsed, dict):
        raise ValueError("manifest must encode one JSON object")
    if json.dumps(
        parsed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) != serialized:
        raise ValueError("manifest must use unambiguous JSON object keys")
    return serialized


def _binary_array(value: Any, name: str) -> NDArray[np.uint8]:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is invalid: {exc}") from exc
    if (
        raw.dtype.hasobject
        or not (
            np.issubdtype(raw.dtype, np.bool_)
            or np.issubdtype(raw.dtype, np.number)
        )
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be a binary numeric array")
    if raw.ndim == 2:
        raw = raw[np.newaxis, ...]
    if raw.ndim != 3 or 0 in raw.shape:
        raise ValueError(f"{name} must have shape (unit, y, x)")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must contain only zero and one")
    return np.array(raw, dtype=np.uint8, copy=True, order="C")


def _unit_id_array(value: Any, n_units: int) -> NDArray[np.int64]:
    raw = np.asarray(value)
    if (
        raw.ndim != 1
        or raw.size != n_units
        or np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("unit_ids must be a 1-D integer array aligned to units")
    if np.issubdtype(raw.dtype, np.unsignedinteger) and raw.size:
        if int(raw.max()) > np.iinfo(np.int64).max:
            raise ValueError("unit_ids values must fit in signed 64-bit integers")
    unit_ids = np.array(raw, dtype=np.int64, copy=True, order="C")
    if np.unique(unit_ids).size != n_units:
        raise ValueError("unit_ids must be unique")
    return unit_ids


def _validated_result_arrays(
    mask_2d: Any,
    center_2d: Any,
    unit_ids: Any,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.int64]]:
    mask = _binary_array(mask_2d, "mask_2d")
    center = _binary_array(center_2d, "center_2d")
    if center.shape != mask.shape:
        raise ValueError("mask_2d and center_2d must have identical shapes")

    if np.any((center != 0) & (mask == 0)):
        raise ValueError("every center_2d bin must also belong to mask_2d")

    mask_nonempty = np.any(mask != 0, axis=(1, 2))
    center_counts = np.sum(center != 0, axis=(1, 2))
    if not np.array_equal(center_counts, mask_nonempty.astype(np.int64)):
        raise ValueError(
            "center_2d must contain exactly one bin for each non-empty unit "
            "and zero bins for each empty unit"
        )

    ids = _unit_id_array(unit_ids, mask.shape[0])
    for array in (mask, center, ids):
        array.setflags(write=False)
    return mask, center, ids


def _cache_key(value: Any, name: str = "cache_key") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _rf_source_path(source_path: str | os.PathLike[str]) -> str:
    source = os.fspath(source_path)
    if source.startswith("<") and source.endswith(">"):
        return source
    return str(Path(source).resolve())


def _rf_result_for_map(
    result: Mapping[str, Any],
    rf_map: Any,
    *,
    center_only: bool = False,
) -> NDArray[np.uint8]:
    """Select a saved 2-D detection by unit ID without running detection.

    New results carry source and grid information, which must match the map.
    Legacy results lack that information: their caller supplies the pairing,
    and only the saved unit ID and spatial shape can be checked.
    """
    manifest = result["manifest"]
    if "collapse_axis" in manifest:
        raise ValueError("detected_rf must contain a 2-D detection, not a 1-D detection")

    for name in ("x_positions", "y_positions"):
        if name in manifest and not np.array_equal(
            manifest[name], getattr(rf_map, name),
        ):
            raise ValueError(f"detected_rf {name} do not match this RFMap")
    if "source_path" in manifest and _rf_source_path(
        manifest["source_path"],
    ) != _rf_source_path(rf_map.source_path):
        raise ValueError("detected_rf source_path does not match this RFMap")

    unit_rows = np.flatnonzero(np.asarray(result["unit_ids"]) == rf_map.unit_id)
    if unit_rows.size == 0:
        raise KeyError(f"detected_rf does not contain unit_id {rf_map.unit_id}")
    if unit_rows.size != 1:
        raise ValueError(f"detected_rf contains duplicate unit_id {rf_map.unit_id}")
    values = np.asarray(result["center_2d" if center_only else "mask_2d"])
    if values.shape != (len(result["unit_ids"]), rf_map.n_y, rf_map.n_x):
        raise ValueError("detected_rf spatial shape does not match this RFMap")
    selected = values[int(unit_rows[0])].view()
    selected.setflags(write=False)
    return selected


@dataclass(frozen=True, eq=False)
class RFResult(Mapping[str, Any]):
    """Validated detection data with read-only arrays in ``(unit, y, x)`` order.

    Construction copies arrays and JSON metadata so caller-owned inputs stay
    independent. Fields and the top-level manifest cannot be reassigned.
    Mapping access preserves existing ``result["mask_2d"]`` call sites.
    Methods only select or project existing masks; none load files or detect RF.
    """

    mask_2d: NDArray[np.uint8]
    center_2d: NDArray[np.uint8]
    unit_ids: NDArray[np.int64]
    manifest: Mapping[str, Any]
    cache_key: str
    schema_version: int = RF_RESULT_CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, (bool, np.bool_))
            or not isinstance(self.schema_version, (int, np.integer))
            or self.schema_version != RF_RESULT_CACHE_SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported RF result schema version {self.schema_version}")
        mask, center, ids = _validated_result_arrays(
            self.mask_2d, self.center_2d, self.unit_ids,
        )
        manifest = json.loads(_canonical_manifest_json(self.manifest))
        object.__setattr__(self, "mask_2d", mask)
        object.__setattr__(self, "center_2d", center)
        object.__setattr__(self, "unit_ids", ids)
        object.__setattr__(self, "manifest", MappingProxyType(manifest))
        object.__setattr__(self, "cache_key", _cache_key(self.cache_key))
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def __getitem__(self, key: str) -> Any:
        if key not in _RESULT_FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(_RESULT_FIELDS)

    def __len__(self) -> int:
        return len(_RESULT_FIELDS)

    def for_unit(self, unit_id: int, *, center_only: bool = False) -> NDArray[np.uint8]:
        """Return a read-only ``(y, x)`` mask or center selected by saved unit ID."""
        rows = np.flatnonzero(self.unit_ids == unit_id)
        if not rows.size:
            raise KeyError(f"detected_rf does not contain unit_id {unit_id}")
        values = self.center_2d if center_only else self.mask_2d
        return values[int(rows[0])]

    def for_map(self, rf_map: Any, *, center_only: bool = False) -> NDArray[np.uint8]:
        """Return this map's 2-D mask after checking unit, grid, and saved source."""
        return _rf_result_for_map(self, rf_map, center_only=center_only)

    def project(self, axis: str = "x", *, center_only: bool = False) -> NDArray[np.uint8]:
        """Project masks onto x or y, keeping ``(unit, 1, x)`` / ``(unit, y, 1)``.

        A projected bin is set when any existing RF bin lies along the collapsed
        axis. This projects detection masks; response summation belongs to RFMap.
        """
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        values = self.center_2d if center_only else self.mask_2d
        projected = np.any(values, axis=1 if axis == "x" else 2, keepdims=True)
        projected = projected.astype(np.uint8)
        projected.setflags(write=False)
        return projected

