"""RF transformations use supplied data and leave analysis choices explicit."""

from dataclasses import replace

import numpy as np
import pytest

from Utils.rf_cache import read_rf_result, save_rf_result
from Utils.rfmap import RFMap, RFMapList, asrfmap


def _raw_map(tmp_path):
    first_bin = np.array([[1, 2], [10, 20], [3, 4]])
    values = np.stack([first_bin, first_bin * 10], axis=-1)
    return replace(
        asrfmap(values, time_bin=0.1), unit_id=7,
        x_positions=np.array([-30., 30.]),
        y_positions=np.array([-20., 0., 20.]),
        presentation_counts=np.array([[1, 2], [3, 4], [5, 6]]),
        metadata={"responseUnits": "spike_count", "responseNormalization": "none",
                  "occupancyTimeSec": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]},
        source_path=tmp_path / "recording.rfmap",
    )


def _saved_detection(tmp_path, masks, unit_ids, rf_map=None):
    masks = np.asarray(masks, dtype=np.uint8)
    centers = np.zeros_like(masks)
    for index, mask in enumerate(masks):
        bins = np.flatnonzero(mask)
        if bins.size:
            centers[index].flat[bins[0]] = 1
    manifest = {} if rf_map is None else {
        "x_positions": rf_map.x_positions.tolist(),
        "y_positions": rf_map.y_positions.tolist(),
        "source_path": str(rf_map.source_path),
    }
    path = tmp_path / "detected.npz"
    save_rf_result(path, mask_2d=masks, center_2d=centers,
                   unit_ids=unit_ids, manifest=manifest, cache_key="supplied-detection")
    return read_rf_result(path)


@pytest.mark.parametrize("axis, expected, selected", [
    ("x", [[10, 20]], [1]),
    ("y", [[1], [10], [3]], [0]),
])
def test_rf_only_sum_selects_whole_rows_or_columns_and_preserves_time(
    tmp_path, monkeypatch, axis, expected, selected,
):
    raw = _raw_map(tmp_path)
    result = _saved_detection(tmp_path, [[[0, 0], [1, 0], [0, 0]]], [7], raw)

    def unexpected_analysis(*args, **kwargs):
        raise AssertionError("transform must use its loaded inputs without detection or loading")

    monkeypatch.setattr("Utils.rflocate.detection._rf_output_arrays", unexpected_analysis)
    monkeypatch.setattr("Utils.rflocate.io._read_rf_source", unexpected_analysis)
    monkeypatch.setattr("Utils.rflocate._cache._read_rf_result", unexpected_analysis)
    collapsed = raw.sum_to_1d(axis=axis, rf_only=True, detected_rf=result)
    expected = np.asarray(expected)
    assert isinstance(collapsed, RFMap)
    np.testing.assert_array_equal(collapsed.spike_counts,
                                  np.stack([expected, expected * 10], axis=-1))
    np.testing.assert_array_equal(collapsed.time_bin_edges_s, raw.time_bin_edges_s)
    assert collapsed.unit_id == raw.unit_id
    assert collapsed.source_path == raw.source_path
    assert collapsed.metadata["responseUnits"] == "spike_count"
    assert collapsed.metadata["spatialAggregation"]["selected_indices"] == selected
    assert collapsed.metadata["spatialAggregation"]["rf_cache_key"] == "supplied-detection"
    collapsed_axis = 0 if axis == "x" else 1
    np.testing.assert_array_equal(
        collapsed.presentation_counts,
        np.take(raw.presentation_counts, selected, axis=collapsed_axis).sum(
            axis=collapsed_axis, keepdims=True),
    )
    np.testing.assert_allclose(
        collapsed.metadata["occupancyTimeSec"],
        np.take(raw.metadata["occupancyTimeSec"], selected, axis=collapsed_axis).sum(
            axis=collapsed_axis, keepdims=True),
    )
    retained_positions = "x_positions" if axis == "x" else "y_positions"
    np.testing.assert_array_equal(getattr(collapsed, retained_positions), getattr(raw, retained_positions))
    assert not collapsed.spike_counts.flags.writeable
    assert "spatialAggregation" not in raw.metadata


def test_multiple_rf_rows_sum_whole_responses_not_masked_pixels(tmp_path):
    raw = _raw_map(tmp_path)
    result = _saved_detection(tmp_path, [[[1, 0], [0, 0], [1, 0]]], [7], raw)
    actual = raw.sum_to_1d(rf_only=True, detected_rf=result)
    np.testing.assert_array_equal(actual.spike_counts, [[[4, 40], [6, 60]]])
    np.testing.assert_array_equal(raw.sum_to_1d().spike_counts, [[[14, 140], [26, 260]]])


@pytest.mark.parametrize("axis", ["x", "y"])
def test_empty_rf_has_missing_response_and_zero_exposure(tmp_path, axis):
    raw = _raw_map(tmp_path)
    result = _saved_detection(tmp_path, np.zeros((1, 3, 2)), [7], raw)
    collapsed = raw.sum_to_1d(axis=axis, rf_only=True, detected_rf=result)
    assert np.isnan(collapsed.spike_counts).all()
    assert not collapsed.presentation_counts.any()
    assert not np.asarray(collapsed.metadata["occupancyTimeSec"]).any()
    assert collapsed.metadata["spatialAggregation"]["selected_indices"] == []


@pytest.mark.parametrize("axis, values, mask", [
    ("x", [[1, 2]], [[0, 1]]),
    ("y", [[1], [2], [3]], [[0], [1], [0]]),
])
def test_rf_sum_supports_already_singleton_grids(tmp_path, axis, values, mask):
    raw = asrfmap(values)
    result = _saved_detection(tmp_path, [mask], [raw.unit_id], raw)
    collapsed = raw.sum_to_1d(axis=axis, rf_only=True, detected_rf=result)
    np.testing.assert_array_equal(collapsed.spike_counts, raw.spike_counts)
    np.testing.assert_array_equal(collapsed.to_1d_array(axis=axis), np.asarray(values).ravel())


def test_batch_rf_sum_aligns_saved_units_and_preserves_batch_order(tmp_path):
    first = _raw_map(tmp_path)
    second = replace(first, unit_id=23, unit_index=1, spike_counts=first.spike_counts * 2)
    result = _saved_detection(tmp_path, [
        [[0, 0], [0, 0], [0, 1]],
        [[0, 0], [1, 0], [0, 0]],
    ], [23, 7], first)
    maps = RFMapList([first, second], first.source_path)
    collapsed = maps.sum_to_1d(rf_only=True, detected_rf=result)
    assert collapsed.unit_ids == [7, 23]
    np.testing.assert_array_equal(collapsed.to_4d_array(), [
        [[[10, 100], [20, 200]]], [[[6, 60], [8, 80]]],
    ])


def test_nan_input_propagates_only_when_its_row_contributes(tmp_path):
    values = np.array([[np.nan, 0], [10, 20], [3, 4]])
    raw = asrfmap(values)
    assert np.isnan(raw.spike_counts[0, 0, 0])
    assert raw.spike_counts[0, 1, 0] == 0
    result = _saved_detection(tmp_path, [[[0, 0], [1, 0], [0, 0]]], [raw.unit_id], raw)
    np.testing.assert_array_equal(
        raw.sum_to_1d(rf_only=True, detected_rf=result).to_1d_array(), [10, 20],
    )
    np.testing.assert_allclose(raw.sum_to_1d().to_1d_array(), [np.nan, 24])


def test_array_extraction_requires_explicit_spatial_and_time_aggregation(tmp_path):
    raw = _raw_map(tmp_path)
    with pytest.raises(ValueError, match="exactly one time bin"):
        raw.sum_to_1d().to_1d_array()
    with pytest.raises(ValueError, match="singleton spatial axis"):
        raw.sum(0, 0.2).to_1d_array()
    prepared = raw.sum(0, 0.2).sum_to_1d()
    np.testing.assert_array_equal(prepared.to_1d_array(), [154, 286])


def test_rf_only_sum_requires_explicit_loaded_result():
    raw = asrfmap([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="requires a loaded detected_rf"):
        raw.sum_to_1d(rf_only=True)
    with pytest.raises(ValueError, match="requires rf_only=True"):
        raw.sum_to_1d(detected_rf={})


@pytest.mark.parametrize("option", [
    {"trials": {}}, {"is_shuffle": True}, {"result_path": "unused.npz"},
    {"cluster_forming_z": 1.5},
])
def test_saved_mask_projection_rejects_detection_and_saving_options(tmp_path, option):
    raw = asrfmap([[1, 2], [3, 4]])
    result = _saved_detection(tmp_path, [[[1, 0], [0, 0]]], [raw.unit_id], raw)
    with pytest.raises(ValueError, match="does not accept detection or saving options"):
        raw.rf_1d(collapse_from_2d=True, detected_rf=result, **option)


def test_saved_mask_projection_does_not_start_detection_without_a_result():
    raw = asrfmap([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="requires a loaded detected_rf"):
        raw.rf_1d(collapse_from_2d=True)


def test_legacy_rate_conversion_precedes_spatial_sum_to_preserve_exposure():
    raw = replace(
        asrfmap([[2], [3]], end_time=1.0),
        presentation_counts=np.array([[1], [1]]),
        metadata={"responseUnits": "Hz", "responseNormalization": "occupancyTimeSec",
                  "occupancyTimeSec": [[1], [2]]},
    )
    with pytest.raises(ValueError, match="before sum_to_1d"):
        raw.sum_to_1d().to_firing_rate()
    converted = raw.to_firing_rate().sum_to_1d()
    np.testing.assert_array_equal(converted.spike_counts, [[[8]]])
    assert converted.metadata["responseUnits"] == "Hz"


@pytest.mark.parametrize("is_batch", [False, True])
def test_mean_rate_rejects_empty_window_but_sum_preserves_empty_sum(is_batch):
    raw = replace(asrfmap([[2, 3]], end_time=1.0), metadata={"responseUnits": "Hz"})
    source = RFMapList([raw], "<array>") if is_batch else raw
    with pytest.raises(ValueError, match="later_s"):
        source.mean_rate(0.0, 0.0)
    empty = source.sum(0.0, 0.0)
    if is_batch:
        empty = empty[0]
    np.testing.assert_array_equal(empty.spike_counts, [[[0], [0]]])
