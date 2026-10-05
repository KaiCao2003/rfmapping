"""The public RF API shares scientific behavior with the compatibility views."""

from dataclasses import replace

import numpy as np
import pytest

from Utils.rflocate import RFMapList, RFResult, asrfmap, detect_rf, load_rf, save_rf


@pytest.mark.parametrize("dimension,axis", [("2d", "x"), ("1d", "x"), ("1d", "y")])
@pytest.mark.parametrize("rf_type", ["excitatory", "inhibitory"])
@pytest.mark.parametrize("exclude_zero_bins", [False, True])
def test_explicit_detection_matches_existing_numeric_behavior(dimension, axis, rf_type, exclude_zero_bins):
    values = np.full((7, 30), 4.)
    values[2:5, 5:9] = 15
    values[2:5, 18:22] = .5
    values[0, 0] = 0
    raw = asrfmap(values)
    options = dict(rf_type=rf_type, exclude_zero_bins=exclude_zero_bins,
                   drop_bins=0, show_progress=False)
    result = detect_rf(raw, dimension=dimension, axis=axis, **options)
    assert isinstance(result, RFResult)
    assert raw._rf_result_cache == {}
    assert result.unit_ids.tolist() == [raw.unit_id]
    assert not result.mask_2d.flags.writeable
    method = raw.rf_2d if dimension == "2d" else raw.rf_1d
    legacy_options = {} if dimension == "2d" else {"axis": axis}
    mask = method(**options, **legacy_options)
    center = method(is_center=True, **options, **legacy_options)
    expected_shape = (1, 7, 30) if dimension == "2d" else (
        (1, 1, 30) if axis == "x" else (1, 7, 1)
    )
    assert result.mask_2d.shape == expected_shape
    np.testing.assert_array_equal(result.mask_2d.reshape(mask.shape), mask)
    np.testing.assert_array_equal(result.center_2d.reshape(center.shape), center)


def test_detect_save_load_and_project_have_one_result_contract(tmp_path, monkeypatch):
    first = asrfmap([[1, 1, 1, 1], [1, 12, 9, 1]], end_time=.2)
    second = replace(first, unit_index=1, unit_id=23)
    maps = RFMapList([second, first], "<array>")
    result = detect_rf(maps, drop_bins=0, cluster_forming_z=1)
    path = tmp_path / "detected.npz"
    save_rf(result, path)
    loaded = load_rf(path)
    assert isinstance(loaded, RFResult)
    assert loaded.unit_ids.tolist() == [23, 0]
    assert loaded.cache_key == result.cache_key

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("loaded results must be projected without detection")

    monkeypatch.setattr("Utils.rflocate._detector.detect_rf", unexpected_detection)
    np.testing.assert_array_equal(loaded.project("x"), result.mask_2d.any(axis=1, keepdims=True))
    np.testing.assert_array_equal(loaded.for_unit(0), result.mask_2d[1])
    output = first.sum_to_1d(rf_only=True, detected_rf=loaded)
    np.testing.assert_array_equal(output.spike_counts, first.spike_counts[1:2])


def test_public_detection_cache_preserves_schema_and_never_mutates_maps(tmp_path, monkeypatch):
    source = asrfmap([[1, 1, 1, 1], [1, 12, 9, 1]], end_time=.2)
    path = tmp_path / "detected.npz"
    first = detect_rf(source, result_path=path, drop_bins=0)

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("matching saved result should be reused")

    monkeypatch.setattr("Utils.rflocate._detector.detect_rf", unexpected_detection)
    second = detect_rf(source, result_path=path, drop_bins=0)
    np.testing.assert_array_equal(first.mask_2d, second.mask_2d)
    assert first.cache_key == second.cache_key
    assert not source._rf_result_cache
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version", "mask_2d", "center_2d", "unit_ids", "manifest_json", "cache_key",
        }


def test_public_detection_requires_explicit_time_reduction():
    source = asrfmap(np.ones((2, 4, 3)), time_bin=.1)
    with pytest.raises(ValueError, match="one time bin"):
        detect_rf(source)
