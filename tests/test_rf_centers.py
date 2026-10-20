"""Each final connected RF sector owns its own weighted center."""

from dataclasses import replace

import numpy as np
import pytest

from RF.rflocate import asrfmap, detect_rf, load_rf, save_rf
from RF.rflocate.detection import _center_only_mask


def _centers(mask, effect, *, wrap_x=False, alternative="greater"):
    return _center_only_mask(
        {"final_mask": np.asarray(mask), "response_map": np.asarray(effect),
         "null_mean_map": np.zeros_like(effect)},
        alternative=alternative, wrap_x=wrap_x, show_progress=False,
    )


@pytest.mark.parametrize("alternative, sign", [("greater", 1), ("less", -1)])
def test_each_sector_uses_only_its_own_effect_weights(alternative, sign):
    mask = [[1, 1, 1, 0, 1, 1, 1]]
    effect = sign * np.array([[1., 1., 10., 0., 1000., 100., 100.]])
    center = _centers(mask, effect, alternative=alternative)
    np.testing.assert_array_equal(center, [[0, 0, 1, 0, 1, 0, 0]])
    effect[:, 4:] *= 100
    np.testing.assert_array_equal(
        _centers(mask, effect, alternative=alternative), center,
    )


def test_centers_use_four_connectivity_and_optional_horizontal_seam():
    mask = np.array([[1, 0, 0, 0, 1], [0, 1, 0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(_centers(mask, mask), mask)
    np.testing.assert_array_equal(
        _centers(mask, mask, wrap_x=True),
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
    )


def test_zero_weight_sector_uses_equal_weights_independently_and_keeps_empty_units():
    mask = np.array([[[1, 1, 1, 0, 1]], [[0, 0, 0, 0, 0]]])
    effect = np.array([[[np.nan, -1., 0., 0., 9.]], [[0., 0., 0., 0., 0.]]])
    np.testing.assert_array_equal(
        _centers(mask, effect),
        [[[0, 1, 0, 0, 1]], [[0, 0, 0, 0, 0]]],
    )


@pytest.mark.parametrize("axis", ["x", "y"])
def test_direct_1d_detection_has_one_center_per_run(axis, tmp_path):
    values = np.zeros((2, 30))
    values[:, 4:7] = [8, 10, 8]
    values[:, 18:21] = [8, 12, 8]
    if axis == "y":
        values = values.T
    source = asrfmap(values)
    result = detect_rf(source.sum_to_1d(axis=axis), sigma=1.0, wrap_x=False, drop_bins=2)
    np.testing.assert_array_equal(np.flatnonzero(result.center_2d), [5, 19])
    path = tmp_path / "sectors.npz"
    save_rf(result, path)
    loaded = load_rf(path)
    np.testing.assert_array_equal(loaded.center_2d, result.center_2d)
    assert loaded.manifest["center_algorithm"] == "component-response-weighted-medoid-v2"


def test_projection_preserves_centers_when_sectors_merge(tmp_path):
    values = np.zeros((3, 12))
    values[0, 2:5] = [8, 12, 8]
    values[2, 4:7] = [8, 12, 8]
    result = detect_rf(asrfmap(values), wrap_x=False, drop_bins=0)
    assert result.center_2d.sum() == 2
    projected = replace(
        result, mask_2d=result.project("x"),
        center_2d=result.project("x", center_only=True),
    )
    assert projected.center_2d.sum() == 2
    path = tmp_path / "projected.npz"
    save_rf(projected, path)
    np.testing.assert_array_equal(load_rf(path).center_2d, projected.center_2d)


def test_previous_center_algorithm_cache_is_recomputed(tmp_path, monkeypatch):
    import RF.rflocate.detection as detection

    values = np.zeros((1, 30))
    values[0, 4:7] = [8, 10, 8]
    values[0, 18:21] = [8, 12, 8]
    source = asrfmap(values)
    path = tmp_path / "legacy.npz"
    manifest_builder = detection._rf_result_manifest

    def legacy_manifest(*args, **kwargs):
        manifest = manifest_builder(*args, **kwargs)
        manifest["center_algorithm"] = "response-weighted-medoid-v1"
        return manifest

    with monkeypatch.context() as context:
        context.setattr(detection, "_rf_result_manifest", legacy_manifest)
        legacy = detect_rf(source, wrap_x=False, drop_bins=2)
    legacy_centers = np.zeros_like(legacy.center_2d)
    legacy_centers[0, 0, 19] = 1
    save_rf(replace(legacy, center_2d=legacy_centers), path)
    assert load_rf(path).center_2d.sum() == 1
    source.rf_2d(result_path=path, wrap_x=False, show_progress=False)
    updated = load_rf(path)
    assert updated.cache_key != legacy.cache_key
    assert updated.center_2d.sum() == 2
    np.testing.assert_array_equal(updated.mask_2d, legacy.mask_2d)
    np.testing.assert_array_equal(load_rf(path).center_2d, updated.center_2d)
