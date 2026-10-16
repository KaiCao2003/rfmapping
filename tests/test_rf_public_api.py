"""The public RF API shares scientific behavior with the compatibility views."""

from dataclasses import replace

import numpy as np
import pytest

from RF.rflocate import RFMapList, RFResult, asrfmap, detect_rf, load_rf, save_rf


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
    sigma = 1.5 if dimension == "2d" else 1.0
    if rf_type == "inhibitory":
        sigma = -1.5 if dimension == "2d" else -.75
    prepared = raw if dimension == "2d" else raw.sum_to_1d(axis=axis)
    result = detect_rf(prepared, sigma=sigma, exclude_zero_bins=exclude_zero_bins,
                       drop_bins=0)
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
    result = detect_rf(maps, drop_bins=0, sigma=1)
    path = tmp_path / "detected.npz"
    save_rf(result, path)
    loaded = load_rf(path)
    assert isinstance(loaded, RFResult)
    assert loaded.unit_ids.tolist() == [23, 0]
    assert loaded.cache_key == result.cache_key

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("loaded results must be projected without detection")

    monkeypatch.setattr("RF.rflocate._detector.detect_rf", unexpected_detection)
    np.testing.assert_array_equal(loaded.project("x"), result.mask_2d.any(axis=1, keepdims=True))
    np.testing.assert_array_equal(loaded.for_unit(0), result.mask_2d[1])
    output = first.sum_to_1d(rf_only=True, detected_rf=loaded)
    np.testing.assert_array_equal(output.spike_counts, first.spike_counts[1:2])


def test_public_detection_never_uses_file_caching_or_mutates_maps(tmp_path, monkeypatch):
    source = asrfmap([[1, 1, 1, 1], [1, 12, 9, 1]], end_time=.2)
    path = tmp_path / "detected.npz"
    def unexpected_cache(*args, **kwargs):
        raise AssertionError("canonical detection must not enter the legacy file-cache wrapper")

    monkeypatch.setattr("RF.rflocate.detection._rf_output_arrays", unexpected_cache)
    first = detect_rf(source, drop_bins=0)
    second = detect_rf(source, drop_bins=0)
    np.testing.assert_array_equal(first.mask_2d, second.mask_2d)
    assert first.cache_key == second.cache_key
    assert not source._rf_result_cache
    assert not path.exists()
    with pytest.raises(TypeError, match="result_path"):
        detect_rf(source, result_path=path, drop_bins=0)
    save_rf(first, path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version", "mask_2d", "center_2d", "unit_ids", "manifest_json", "cache_key",
        }


def test_public_detection_requires_explicit_time_reduction():
    source = asrfmap(np.ones((2, 4, 3)), time_bin=.1)
    with pytest.raises(ValueError, match="one time bin"):
        detect_rf(source, drop_bins=2)


@pytest.mark.parametrize("reference", ["global", "bottom_row"])
def test_public_detection_requires_explicit_drop_bins(reference):
    with pytest.raises(TypeError, match="drop_bins"):
        detect_rf([[8, 5], [4, 6]], reference=reference)


@pytest.mark.parametrize("sigma", [1.8, -1.0])
@pytest.mark.parametrize("shape", [(20,), (1, 20), (20, 1), (4, 5)])
def test_signed_sigma_thresholds_the_input_grid(sigma, shape):
    values = np.full(20, 10.0)
    values[4:6] = [0, 3]
    values[12:14] = [22, 18]
    values = values.reshape(shape)
    threshold = values.mean() + sigma * values.std()
    expected = values >= threshold if sigma > 0 else values <= threshold
    result = detect_rf(values, sigma=sigma, drop_bins=0, wrap_x=False)
    expected_shape = (1, 20) if values.ndim == 1 else values.shape
    assert result.mask_2d.shape == (1, *expected_shape)
    np.testing.assert_array_equal(result.mask_2d.ravel(), expected.ravel())
    assert result.unit_ids.tolist() == [0]
    assert result.manifest["parameters"]["reference"] == "global"
    assert "method" not in result.manifest["parameters"]
    assert result.center_2d.any()
    assert np.all(result.center_2d <= result.mask_2d)


@pytest.mark.parametrize("sigma", [1.0, -1.0])
def test_raw_vector_and_rfmap_preserve_masks_and_weighted_centers(sigma):
    values = np.full(30, 10.0)
    values[5:8] = [0, 3, 5]
    values[20:23] = [20, 17, 15]
    source = replace(asrfmap(values[None]), unit_id=47)
    result = detect_rf(values, sigma=sigma, drop_bins=0)
    mapped = detect_rf(source, sigma=sigma, drop_bins=0)
    assert mapped.unit_ids.tolist() == [47]
    np.testing.assert_array_equal(mapped.mask_2d, result.mask_2d)
    np.testing.assert_array_equal(mapped.center_2d, result.center_2d)
    expected_center = 21 if sigma > 0 else 6
    assert result.center_2d.sum() == 1
    assert result.center_2d[0, 0, expected_center] == 1


def test_persisted_results_distinguish_sigma_sign(tmp_path):
    values = np.full(30, 10.0)
    values[5:8] = [0, 1, 2]
    values[20:23] = [20, 19, 18]
    path = tmp_path / "detected.npz"
    increased = detect_rf(values, sigma=1, drop_bins=0)
    save_rf(increased, path)
    decreased = detect_rf(values, sigma=-1, drop_bins=0)
    save_rf(decreased, path)
    assert increased.cache_key != decreased.cache_key
    assert increased.mask_2d.any() and decreased.mask_2d.any()
    assert not np.any(increased.mask_2d & decreased.mask_2d)
    loaded = load_rf(path)
    assert loaded.cache_key == decreased.cache_key
    assert loaded.manifest["parameters"]["alternative"] == "less"
    assert loaded.manifest["parameters"]["cluster_forming_z"] == 1
    np.testing.assert_array_equal(loaded.mask_2d, decreased.mask_2d)


@pytest.mark.parametrize("sigma", [1.0, -1.0])
def test_signed_detection_preserves_wrapped_component_size_filter(sigma):
    values = np.zeros(12)
    values[[0, -1]] = 9
    if sigma < 0:
        values = 9 - values
    wrapped = detect_rf(values, sigma=sigma, drop_bins=1, wrap_x=True)
    separate = detect_rf(values, sigma=sigma, drop_bins=1, wrap_x=False)
    assert wrapped.mask_2d.sum() == 2
    assert wrapped.center_2d.sum() == 1
    assert not separate.mask_2d.any()


@pytest.mark.parametrize("axis", ["x", "y"])
def test_prepared_1d_result_preserves_provenance_for_native_singleton_grid(axis):
    values = np.zeros((1, 12) if axis == "x" else (12, 1))
    values.flat[4:7] = [8, 10, 8]
    source = asrfmap(values)
    native_result = detect_rf(source, sigma=1, drop_bins=0)
    prepared = source.sum_to_1d(axis=axis, missing_as_zero=True)
    result = detect_rf(prepared, sigma=1, drop_bins=0)
    np.testing.assert_array_equal(result.mask_2d, native_result.mask_2d)
    assert result.cache_key != native_result.cache_key
    assert "collapse_axis" not in native_result.manifest
    assert result.manifest["collapse_axis"] == axis
    assert result.manifest["spatial_aggregation"] == [prepared.metadata["spatialAggregation"]]
    with pytest.raises(ValueError, match="2-D detection, not a 1-D detection"):
        source.sum_to_1d(axis=axis, rf_only=True, detected_rf=result)


def test_native_bar_coverage_metadata_is_not_a_computed_1d_projection():
    values = np.ones((1, 120))
    values[:, 40:45] = 10
    source = asrfmap(values)
    source = replace(source, metadata={**source.metadata,
                     "spatialAggregation": "binary_final_bin_trial_coverage"})
    result = detect_rf(source, sigma=2, drop_bins=2)
    assert "collapse_axis" not in result.manifest
    assert result.manifest["spatial_aggregation"] == ["binary_final_bin_trial_coverage"]
    assert result.mask_2d.sum() == 5
    projected = source.sum_to_1d("x", rf_only=True, detected_rf=result)
    assert projected.to_1d_array("x").shape == (120,)


def test_aggregation_cache_identity_tracks_missing_convention_and_each_units_source_rows():
    values = np.zeros((2, 12))
    values[:, 4:7] = [8, 10, 8]
    source = asrfmap(values)
    propagated = source.sum_to_1d()
    zero_filled = source.sum_to_1d(missing_as_zero=True)
    first = detect_rf(propagated, sigma=1, drop_bins=0)
    filled = detect_rf(zero_filled, sigma=1, drop_bins=0)
    np.testing.assert_array_equal(first.mask_2d, filled.mask_2d)
    assert first.cache_key != filled.cache_key

    def select_row(row):
        mask = np.zeros((1, 2, 12), dtype=np.uint8)
        mask[0, row, 4:7] = 1
        return source.sum_to_1d(rf_only=True, detected_rf={
            "mask_2d": mask, "unit_ids": [source.unit_id], "manifest": {},
        })

    top = select_row(0)
    bottom = replace(select_row(1), unit_index=1, unit_id=7)
    same_rows = RFMapList([top, replace(top, unit_index=1, unit_id=7)], "<array>")
    other_row = RFMapList([top, bottom], "<array>")
    repeated = detect_rf(same_rows, sigma=1, drop_bins=0)
    changed = detect_rf(other_row, sigma=1, drop_bins=0)
    np.testing.assert_array_equal(repeated.mask_2d, changed.mask_2d)
    assert repeated.cache_key != changed.cache_key
    assert changed.manifest["spatial_aggregation"][1]["selected_indices"] == [1]


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("sigma", [1.0, -1.0])
def test_public_shuffle_uses_prepared_grid_and_aligned_unit_ids(axis, sigma):
    values = np.full((3, 8), 10.0)
    values[1, 2:4] = [0, 2]
    values[1, 5:7] = [20, 18]
    source = asrfmap(values)
    prepared = source.sum_to_1d(axis=axis)
    maps = RFMapList([replace(prepared, unit_id=7, unit_index=1), prepared], "<array>")
    positions = np.tile(np.arange(values.size), 8)
    responses = np.tile(values.ravel(), 8)
    trials = dict(
        responses=np.stack([responses, responses * 2]),
        position_ids=positions % source.n_x if axis == "x" else positions // source.n_x,
        shape=prepared.shape[:2],
        unit_ids=np.array([0, 7]),
        x_positions=prepared.x_positions,
        y_positions=prepared.y_positions,
        time_range_s=prepared.time_window_s,
    )
    options = dict(is_shuffle=True, n_permutations=39, random_seed=17,
                   wrap_x=False, n_jobs=1, show_progress=False)
    result = detect_rf(maps, trials, sigma=sigma, drop_bins=2, **options)
    legacy_options = dict(cluster_forming_z=abs(sigma),
                          rf_type="excitatory" if sigma > 0 else "inhibitory")
    assert result.unit_ids.tolist() == [7, 0]
    np.testing.assert_array_equal(
        result.mask_2d, maps.rf_2d(trials, **options, **legacy_options),
    )
    np.testing.assert_array_equal(
        result.center_2d, maps.rf_2d(trials, is_center=True, **options, **legacy_options),
    )


@pytest.mark.parametrize("options", [
    {"dimension": "1d"}, {"axis": "x"}, {"rf_type": "inhibitory"},
    {"cluster_forming_z": 2}, {"alternative": "less"},
    {"method": "global"},
])
def test_public_detection_rejects_redundant_selectors(options):
    with pytest.raises(TypeError):
        detect_rf([0, 1, 0], sigma=1, drop_bins=2, **options)


@pytest.mark.parametrize("sigma", [0, np.nan, np.inf, True])
def test_public_detection_requires_finite_nonzero_sigma(sigma):
    with pytest.raises(ValueError, match="sigma"):
        detect_rf([0, 1, 0], sigma=sigma, drop_bins=2)
