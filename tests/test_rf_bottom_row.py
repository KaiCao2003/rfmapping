"""Bottom-row reference changes the mean while retaining the full-map SD."""

from dataclasses import replace

import numpy as np
import pytest

from RF.rflocate import RFMapList, asrfmap, detect_rf, load_rf, save_rf
from RF.rflocate._cache import build_rf_result_cache_key
from RF.rflocate._detector import detect_rf as detect_arrays
from RF.rflocate.detection import _prepare_rf_output, _rf_result_key_arrays


@pytest.mark.parametrize("sigma", [0.5, 1.5, -0.5, -1.5])
def test_bottom_row_uses_entire_last_rows_mean_and_full_maps_population_sd(sigma):
    values = np.array([[0, 2, 20], [6, 6, 6], [4, 6, 8]], dtype=float)
    baseline = values[-1].mean()
    sd = values.std(ddof=0)
    threshold = baseline + sigma * sd
    expected = values >= threshold if sigma > 0 else values <= threshold
    result = detect_rf(values, reference="bottom_row", sigma=sigma,
                       drop_bins=0, wrap_x=False)
    np.testing.assert_array_equal(result.mask_2d, [expected])
    assert np.all(result.center_2d <= result.mask_2d)

    numeric = detect_arrays(
        values.reshape(1, -1), np.arange(values.size), values.shape,
        reference="bottom_row", is_shuffle=False, drop_bins=0,
        cluster_forming_z=abs(sigma), alternative="greater" if sigma > 0 else "less",
        wrap_x=False, show_progress=False,
    )
    np.testing.assert_allclose(numeric["null_mean_map"], baseline)
    np.testing.assert_allclose(numeric["null_sd_map"], sd)
    np.testing.assert_array_equal(numeric["final_mask"], [expected])


@pytest.mark.parametrize("sigma, expected", [
    (1, [[0, 1], [0, 1]]), (-1, [[1, 0], [1, 0]]),
])
def test_bottom_row_includes_exact_thresholds(sigma, expected):
    # Mean 2 and population SD 2 make the signed thresholds exactly 4 and 0.
    result = detect_rf([[0, 4], [0, 4]], reference="bottom_row", sigma=sigma,
                       drop_bins=0, wrap_x=False)
    np.testing.assert_array_equal(result.mask_2d, [expected])


@pytest.mark.parametrize("sigma", [0.5, -0.5])
def test_nan_bins_are_zero_filled_and_zero_bins_remain_in_both_statistics(sigma):
    values = np.array([[np.nan, 4, 8], [0, np.nan, 6], [4, 4, np.nan]])
    pooled = np.nan_to_num(values)
    threshold = pooled[-1].mean() + sigma * pooled.std(ddof=0)
    expected = pooled >= threshold if sigma > 0 else pooled <= threshold
    result = detect_rf(values, reference="bottom_row", sigma=sigma,
                       drop_bins=0, wrap_x=False)
    filled_result = detect_rf(pooled, reference="bottom_row", sigma=sigma,
                              drop_bins=0, wrap_x=False)
    np.testing.assert_array_equal(result.mask_2d, [expected])
    assert result.cache_key == filled_result.cache_key
    if sigma < 0:
        assert result.mask_2d[0, 0, 0] == result.mask_2d[0, 1, 0] == 1
    assert np.isnan(values[0, 0])


@pytest.mark.parametrize("sigma", [1.5, -1.5])
@pytest.mark.parametrize("value", [0, 5])
def test_constant_map_has_no_detections(sigma, value):
    result = detect_rf(np.full((3, 4), value), reference="bottom_row",
                       sigma=sigma, drop_bins=0)
    assert not result.mask_2d.any()
    assert not result.center_2d.any()


def test_nonshuffle_bottom_row_ignores_supplied_trials_and_trial_provenance():
    source = asrfmap([[0, 2, 20], [6, 6, 6], [4, 6, 8]], end_time=.2)
    options = dict(reference="bottom_row", sigma=1, drop_bins=0, wrap_x=False)
    original = detect_rf(source, **options)
    supplied = {"responses": np.array([[999.]]), "response_units": "Hz",
                "provenance": {"trial_file": "unused"}}
    with_trials = detect_rf(source, supplied, **options)
    np.testing.assert_array_equal(with_trials.mask_2d, original.mask_2d)
    np.testing.assert_array_equal(with_trials.center_2d, original.center_2d)
    assert with_trials.cache_key == original.cache_key
    assert "trial_provenance" not in with_trials.manifest
    np.testing.assert_array_equal(supplied["responses"], [[999.]])


@pytest.mark.parametrize("sigma", [0.5, -0.5])
def test_batch_preserves_unit_order_and_uses_each_units_map_values(sigma):
    values = np.array([[0, 2, 20], [6, 6, 6], [4, 6, 8]], dtype=float)
    first = replace(asrfmap(values, end_time=.2), unit_id=7)
    second = replace(asrfmap(40 - values, end_time=.2), unit_id=47, unit_index=1)
    maps = RFMapList([second, first], first.source_path)
    options = dict(reference="bottom_row", sigma=sigma, drop_bins=0, wrap_x=False)
    # Unequal presentation counts make rate maps differ beyond a global scale.
    rates = maps.to_firing_rate(presentation_counts=np.array([[1, 2, 3], [4, 2, 1], [1, 3, 2]]))
    for inputs in (maps, rates):
        batch = detect_rf(inputs, **options)
        assert batch.unit_ids.tolist() == [47, 7]
        for row, source in enumerate(inputs):
            pooled = source.to_2d_array()
            threshold = pooled[-1].mean() + sigma * pooled.std(ddof=0)
            expected = pooled >= threshold if sigma > 0 else pooled <= threshold
            single = detect_rf(source, **options)
            np.testing.assert_array_equal(batch.mask_2d[row], expected)
            np.testing.assert_array_equal(batch.mask_2d[row], single.mask_2d[0])
            np.testing.assert_array_equal(batch.center_2d[row], single.center_2d[0])
        assert batch.manifest["response_units"] == inputs[0].metadata.get("responseUnits", "spike_count")
        assert batch.manifest["response_normalization"] == inputs[0].metadata.get("responseNormalization", "none")


@pytest.mark.parametrize("sign", [1, -1])
def test_centers_weight_response_relative_to_bottom_rows_shared_mean(sign):
    values = np.vstack([20 + sign * np.array([1, 2, 13]), [20, 20, 20]])
    result = detect_rf(values, reference="bottom_row", sigma=sign * .1,
                       drop_bins=0, wrap_x=False)
    np.testing.assert_array_equal(result.mask_2d, [[[1, 1, 1], [0, 0, 0]]])
    np.testing.assert_array_equal(result.center_2d, [[[0, 0, 1], [0, 0, 0]]])


@pytest.mark.parametrize("sign", [1, -1])
def test_bottom_row_retains_wrapped_component_size_filter(sign):
    values = np.vstack([5 + sign * np.array([3, 0, 0, 3]), [5, 5, 5, 5]])
    options = dict(reference="bottom_row", sigma=sign * 2, drop_bins=1)
    wrapped = detect_rf(values, wrap_x=True, **options)
    separate = detect_rf(values, wrap_x=False, **options)
    np.testing.assert_array_equal(wrapped.mask_2d, [[[1, 0, 0, 1], [0, 0, 0, 0]]])
    assert wrapped.center_2d.sum() == 1
    assert not separate.mask_2d.any()


@pytest.mark.parametrize("sign", [1, -1])
def test_bottom_row_drop_zero_retains_small_components_and_drop_two_removes_them(sign):
    values = np.full((2, 9), 5.)
    values[0, [0, 2, 3, 5, 6, 7]] += sign * 3
    options = dict(reference="bottom_row", sigma=sign, wrap_x=False)
    unfiltered = detect_rf(values, drop_bins=0, **options)
    filtered = detect_rf(values, drop_bins=2, **options)
    np.testing.assert_array_equal(unfiltered.mask_2d[0, 0], [1, 0, 1, 1, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(filtered.mask_2d[0, 0], [0, 0, 0, 0, 0, 1, 1, 1, 0])
    assert not unfiltered.mask_2d[0, 1].any()
    assert not filtered.mask_2d[0, 1].any()
    assert unfiltered.cache_key != filtered.cache_key


def test_bottom_row_manifest_version_and_save_load_do_not_reuse_trial_algorithm(tmp_path):
    source = asrfmap([[0, 2, 20], [6, 6, 6], [4, 6, 8]])
    result = detect_rf(source, reference="bottom_row", sigma=1, drop_bins=0)
    assert result.manifest["parameters"]["reference"] == "bottom_row"
    assert result.manifest["detector_algorithm"] == "bottom-row-mean-global-sd-v1"
    assert dict(result.manifest["baseline"]) == {
        "row_index": 2, "y_position": float(source.y_positions[-1]),
        "mean_scope": "bottom_row", "sd_scope": "full_map", "sd_ddof": 0,
    }
    aligned, _, _, manifest, key = _prepare_rf_output(
        maps=[source], trials=None, rf_type="excitatory", is_shuffle=False,
        drop_bins=0, options={"reference": "bottom_row", "cluster_forming_z": 1.0},
    )
    assert result.cache_key == key
    old_manifest = dict(manifest, detector_algorithm="bottom-row-trial-z-v1")
    old_key = build_rf_result_cache_key(arrays=_rf_result_key_arrays(aligned), manifest=old_manifest)
    assert result.cache_key != old_key
    assert result.cache_key != detect_rf(source, sigma=1, drop_bins=0).cache_key
    path = tmp_path / "bottom_row.npz"
    save_rf(result, path)
    loaded = load_rf(path)
    assert loaded.cache_key == result.cache_key
    assert loaded.manifest == result.manifest
    np.testing.assert_array_equal(loaded.mask_2d, result.mask_2d)
    np.testing.assert_array_equal(loaded.center_2d, result.center_2d)


@pytest.mark.parametrize("options", [
    {"is_shuffle": True}, {"exclude_zero_bins": True},
])
def test_bottom_row_requires_unfiltered_nonshuffle_maps(options):
    with pytest.raises(ValueError, match="bottom_row"):
        detect_rf([[8, 5], [4, 6]], reference="bottom_row", drop_bins=2, **options)


def test_bottom_row_requires_multiple_spatial_rows_and_no_collapse():
    with pytest.raises(ValueError, match="two spatial rows"):
        detect_rf([[8, 5]], reference="bottom_row", drop_bins=2)
    source = asrfmap([[8, 5], [4, 6]])
    with pytest.raises(ValueError, match="uncollapsed"):
        source.rf_1d(reference="bottom_row", is_shuffle=False)
