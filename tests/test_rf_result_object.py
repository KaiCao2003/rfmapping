"""Detected results expose explicit data access without detection or file IO."""

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from Utils.rflocate.results import RFResult


def _inputs():
    return {
        "mask_2d": np.array([
            [[0, 0], [1, 1], [0, 1]],
            [[0, 0], [0, 0], [0, 0]],
        ], dtype=np.uint8),
        "center_2d": np.array([
            [[0, 0], [0, 1], [0, 0]],
            [[0, 0], [0, 0], [0, 0]],
        ], dtype=np.uint8),
        "unit_ids": np.array([23, 7]),
        "manifest": {
            "x_positions": [-30., 30.],
            "y_positions": [-20., 0., 20.],
            "source_path": "<array>",
        },
        "cache_key": "detected-inputs",
    }


def test_result_owns_arrays_and_preserves_mapping_access():
    inputs = _inputs()
    result = RFResult(**inputs)
    inputs["mask_2d"][:] = 0
    inputs["unit_ids"][:] = 99
    inputs["manifest"]["x_positions"][0] = 100
    assert result.mask_2d.sum() == 3
    np.testing.assert_array_equal(result.unit_ids, [23, 7])
    assert result.manifest["x_positions"] == [-30., 30.]
    assert result["mask_2d"] is result.mask_2d
    assert result.get("cache_key") == "detected-inputs"
    assert set(result) == {
        "schema_version", "mask_2d", "center_2d", "unit_ids", "manifest", "cache_key",
    }
    assert len(result) == 6
    with pytest.raises(KeyError):
        result["missing"]
    assert json.loads(json.dumps(dict(result.manifest))) == dict(result.manifest)
    for values in (result.mask_2d, result.center_2d, result.unit_ids):
        with pytest.raises(ValueError, match="read-only"):
            values.flat[0] = 1
    with pytest.raises(FrozenInstanceError):
        result.cache_key = "other"
    with pytest.raises(TypeError):
        result.manifest["source_path"] = "other"


def test_result_selects_by_unit_id_and_checks_map_alignment():
    result = RFResult(**_inputs())
    np.testing.assert_array_equal(result.for_unit(7), np.zeros((3, 2)))
    np.testing.assert_array_equal(result.for_unit(23, center_only=True), result.center_2d[0])
    assert not result.for_unit(23).flags.writeable
    with pytest.raises(KeyError, match="unit_id 99"):
        result.for_unit(99)
    rf_map = SimpleNamespace(
        unit_id=23, n_y=3, n_x=2, source_path="<array>",
        x_positions=[-30., 30.], y_positions=[-20., 0., 20.],
    )
    np.testing.assert_array_equal(result.for_map(rf_map), result.mask_2d[0])
    np.testing.assert_array_equal(
        result.for_map(rf_map, center_only=True), result.center_2d[0],
    )
    rf_map.source_path = "<different-array>"
    with pytest.raises(ValueError, match="source_path"):
        result.for_map(rf_map)


@pytest.mark.parametrize("axis, expected", [
    ("x", [[[1, 1]], [[0, 0]]]),
    ("y", [[[0], [1], [1]], [[0], [0], [0]]]),
])
def test_projection_preserves_units_singleton_axes_and_empty_rf(axis, expected):
    result = RFResult(**_inputs())
    projection = result.project(axis=axis)
    np.testing.assert_array_equal(projection, expected)
    assert projection.dtype == np.uint8
    assert not projection.flags.writeable
    center = result.project(axis=axis, center_only=True)
    assert center.shape == projection.shape
    np.testing.assert_array_equal(center.sum(axis=(1, 2)), [1, 0])
    with pytest.raises(ValueError, match="axis"):
        result.project("time")


@pytest.mark.parametrize("field, value, message", [
    ("unit_ids", [7, 7], "unique"),
    ("mask_2d", np.ones((2, 3, 2)) * 2, "zero and one"),
    ("center_2d", np.zeros((2, 3, 2)), "exactly one"),
    ("manifest", {"threshold": float("nan")}, "JSON-compatible"),
    ("schema_version", 99, "schema version"),
    ("cache_key", "", "non-empty string"),
])
def test_in_memory_results_use_saved_result_validation(field, value, message):
    inputs = {**_inputs(), field: value}
    with pytest.raises(ValueError, match=message):
        RFResult(**inputs)
