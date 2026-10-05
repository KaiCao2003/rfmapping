"""Saved RF results are validated data, independent of detector cache lookup."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from Utils.rf_cache import (
    RFResultCacheError, _rf_result_for_map, _rf_source_path,
    load_rf_result, read_rf_result, rf_result_path, save_rf_result,
)
from Utils.rflocate.results import RFResult


@pytest.mark.parametrize("source_name", ["regular.rfmap", "regular.json", "regular"])
@pytest.mark.parametrize("dimension", ["2d", "1d"])
@pytest.mark.parametrize("rf_type", ["excitatory", "inhibitory"])
def test_result_path_keeps_dimension_and_polarity_distinct(tmp_path, source_name, dimension, rf_type):
    suffix = "_inhibitory" if rf_type == "inhibitory" else ""
    suffix += "_1d" if dimension == "1d" else ""
    assert rf_result_path(tmp_path / source_name, dimension=dimension, rf_type=rf_type) == (
        tmp_path / f"regular{suffix}.npz"
    )


def _save_result(path):
    # The saved order intentionally differs from sorted unit IDs.
    mask = np.array([[[1, 1, 0]], [[0, 0, 0]]], dtype=np.uint8)
    center = np.array([[[0, 1, 0]], [[0, 0, 0]]], dtype=np.uint8)
    save_rf_result(path, mask_2d=mask, center_2d=center, unit_ids=[23, 7],
                   manifest={"collapse_axis": "x"}, cache_key="input-key")
    return mask, center


def test_read_saved_result_preserves_unit_order_and_singleton_axes(tmp_path):
    path = tmp_path / "regular_1d.npz"
    mask, center = _save_result(path)
    result = read_rf_result(path)
    assert isinstance(result, RFResult)
    np.testing.assert_array_equal(result["mask_2d"], mask)
    np.testing.assert_array_equal(result["center_2d"], center)
    np.testing.assert_array_equal(result["unit_ids"], [23, 7])
    assert result["manifest"] == {"collapse_axis": "x"}
    assert not result["mask_2d"].flags.writeable
    assert not result["center_2d"].flags.writeable
    assert not result["unit_ids"].flags.writeable
    assert load_rf_result(path, expected_cache_key="different-key") is None
    cached = load_rf_result(path, expected_cache_key="input-key")
    assert isinstance(cached, RFResult)
    assert cached.cache_key == "input-key"


def test_missing_saved_result_is_an_error_but_missing_cache_is_a_miss(tmp_path):
    path = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        read_rf_result(path)
    assert load_rf_result(path, expected_cache_key="input-key") is None
    assert not path.exists()


@pytest.mark.parametrize("corruption", ["schema", "unit_ids", "center", "missing_field"])
def test_saved_result_reader_rejects_malformed_archive(tmp_path, corruption):
    path = tmp_path / "regular.npz"
    _save_result(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if corruption == "schema":
        arrays["schema_version"] = np.asarray(99)
    elif corruption == "unit_ids":
        arrays["unit_ids"] = np.asarray([23, 23])
    elif corruption == "center":
        arrays["center_2d"][0, 0, 0] = 1
    else:
        del arrays["manifest_json"]
    np.savez(path, **arrays)
    with pytest.raises(RFResultCacheError):
        read_rf_result(path)
    with pytest.raises(RFResultCacheError):
        load_rf_result(path, expected_cache_key="input-key")


def _saved_2d_result_and_map(tmp_path, *, include_geometry):
    mask = np.array([
        [[1, 1], [0, 0], [0, 0]],
        [[0, 0], [1, 1], [0, 1]],
    ], dtype=np.uint8)
    center = np.array([
        [[0, 1], [0, 0], [0, 0]],
        [[0, 0], [0, 1], [0, 0]],
    ], dtype=np.uint8)
    rf_map = SimpleNamespace(
        unit_id=7, unit_index=0, n_y=3, n_x=2,
        x_positions=np.array([-30., 30.]),
        y_positions=np.array([-20., 0., 20.]),
        source_path=tmp_path / "regular.rfmap",
    )
    manifest = {}
    if include_geometry:
        manifest = {
            "x_positions": rf_map.x_positions.tolist(),
            "y_positions": rf_map.y_positions.tolist(),
            "source_path": _rf_source_path(rf_map.source_path),
        }
    path = tmp_path / "regular.npz"
    save_rf_result(
        path, mask_2d=mask, center_2d=center, unit_ids=[23, 7],
        manifest=manifest, cache_key="input-key",
    )
    return read_rf_result(path), rf_map


@pytest.mark.parametrize("include_geometry", [False, True])
@pytest.mark.parametrize("center_only", [False, True])
def test_saved_2d_result_selects_unit_id_for_new_and_legacy_files(
    tmp_path, include_geometry, center_only,
):
    result, rf_map = _saved_2d_result_and_map(
        tmp_path, include_geometry=include_geometry,
    )
    selected = _rf_result_for_map(result, rf_map, center_only=center_only)
    field = "center_2d" if center_only else "mask_2d"
    np.testing.assert_array_equal(selected, result[field][1])
    assert not selected.flags.writeable


@pytest.mark.parametrize("field", ["x_positions", "y_positions", "source_path"])
def test_saved_2d_result_rejects_geometry_or_source_mismatch(tmp_path, field):
    result, rf_map = _saved_2d_result_and_map(tmp_path, include_geometry=True)
    if field == "source_path":
        rf_map.source_path = tmp_path / "different-session.rfmap"
    else:
        setattr(rf_map, field, getattr(rf_map, field)[::-1])
    with pytest.raises(ValueError, match=field):
        _rf_result_for_map(result, rf_map)


def test_saved_2d_result_accepts_equivalent_source_paths(tmp_path, monkeypatch):
    result, rf_map = _saved_2d_result_and_map(tmp_path, include_geometry=True)
    monkeypatch.chdir(tmp_path)
    rf_map.source_path = "./regular.rfmap"
    np.testing.assert_array_equal(
        _rf_result_for_map(result, rf_map), result["mask_2d"][1],
    )


def test_in_memory_source_identity_is_not_resolved_as_a_filename():
    assert _rf_source_path("<array>") == "<array>"


def test_saved_result_cannot_substitute_a_1d_detection_for_2d(tmp_path):
    result, rf_map = _saved_2d_result_and_map(tmp_path, include_geometry=False)
    result = replace(result, manifest={**result.manifest, "collapse_axis": "x"})
    with pytest.raises(ValueError, match="2-D detection"):
        _rf_result_for_map(result, rf_map)


def test_saved_2d_result_requires_the_requested_unit_id(tmp_path):
    result, rf_map = _saved_2d_result_and_map(tmp_path, include_geometry=False)
    rf_map.unit_id = 99
    with pytest.raises(KeyError, match="unit_id 99"):
        _rf_result_for_map(result, rf_map)


def test_legacy_saved_2d_result_still_requires_matching_spatial_shape(tmp_path):
    result, rf_map = _saved_2d_result_and_map(tmp_path, include_geometry=False)
    rf_map.n_y = 2
    with pytest.raises(ValueError, match="spatial shape"):
        _rf_result_for_map(result, rf_map)
