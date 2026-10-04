"""Shared RF scalar, shape, and read-only array validation."""
from __future__ import annotations
import math
import operator
from numbers import Integral, Real
from typing import Any
import numpy as np
from numpy.typing import NDArray

_EDGE_ATOL_S = 1e-12

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



def _spatial_axis(value: Any, label: str, expected_length: int) -> list[Any]:
    """Restore MATLAB's scalar encoding for a singleton spatial axis."""

    if (
        expected_length == 1
        and isinstance(value, Real)
        and not isinstance(value, bool)
    ):
        return [value]
    return _flat_list(value, label)



def _counts_are_numeric(value: Any, *, allow_null: bool = False) -> bool:
    if isinstance(value, np.ndarray):
        return (
            np.issubdtype(value.dtype, np.number)
            and not np.issubdtype(value.dtype, np.complexfloating)
        )
    if isinstance(value, list):
        for child in value:
            if not _counts_are_numeric(child, allow_null=allow_null):
                return False
        return True
    if value is None:
        return allow_null
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



def _bounded_integer(value: Any, label: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return parsed



def _nonnegative_integer(value: Any, label: str) -> int:
    return _bounded_integer(value, label, minimum=0)
