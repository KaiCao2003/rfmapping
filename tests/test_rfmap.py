from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import Utils.rfmap as rfmap_module
from Utils.rfmap import RFMap, RFMapList, asrfmap, load_rf_maps


def _payload(
    counts: np.ndarray | None = None,
    *,
    unit_pool: list[int] | None = None,
    time_bin_edges: list[float] | None = None,
) -> dict[str, object]:
    if counts is None:
        counts = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    if unit_pool is None:
        unit_pool = [41, 7]
    if time_bin_edges is None:
        time_bin_edges = [-0.1, 0.0, 0.1, 0.2]

    return {
        "unitsSpikeCounts": counts.tolist(),
        "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": unit_pool,
        "xPositions": [-10.0, 10.0],
        "yPositions": [-5.0, 5.0],
        "timeBinEdges": time_bin_edges,
        "stimulusPresentationCounts": [[2, 3], [4, 5]],
        "sessionName": "fixture-session",
    }


def _write_payload(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "rfmapping.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_asrfmap_promotes_2d_array_and_uses_index_defaults() -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    expected = values.copy()

    rf_map = asrfmap(values)

    assert isinstance(rf_map, RFMap)
    assert rf_map.shape == (2, 3, 1)
    assert rf_map.dtype == np.dtype(np.float32)
    assert rf_map.unit_index == 0
    assert rf_map.unit_id == 0
    assert rf_map.presentation_counts is None
    assert dict(rf_map.metadata) == {}
    assert rf_map.source_path == Path("<array>")
    np.testing.assert_array_equal(rf_map.to_2d_array(), expected)
    np.testing.assert_array_equal(rf_map.x_positions, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(rf_map.y_positions, [0.0, 1.0])
    np.testing.assert_array_equal(rf_map.time_bin_edges_s, [0.0, 1.0])
    assert not rf_map.spike_counts.flags.writeable
    assert not rf_map.x_positions.flags.writeable
    assert not rf_map.y_positions.flags.writeable
    assert not rf_map.time_bin_edges_s.flags.writeable

    # Construction must neither freeze nor retain a mutable view of the input.
    assert values.flags.writeable
    values[0, 0] = 99.0
    np.testing.assert_array_equal(rf_map.to_2d_array(), expected)


def test_asrfmap_keeps_3d_time_axis_and_builds_explicit_edges() -> None:
    values = np.arange(18).reshape(2, 3, 3)

    rf_map = asrfmap(
        values,
        start_time=-0.1,
        end_time=0.2,
        time_bin=0.1,
    )

    assert rf_map.shape == (2, 3, 3)
    np.testing.assert_array_equal(rf_map.spike_counts, values)
    np.testing.assert_allclose(
        rf_map.time_bin_edges_s,
        [-0.1, 0.0, 0.1, 0.2],
        rtol=0.0,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected_edges"),
    [
        ({}, [0.0, 1.0, 2.0, 3.0]),
        ({"start_time": -0.1, "time_bin": 0.1}, [-0.1, 0.0, 0.1, 0.2]),
        ({"start_time": -0.1, "end_time": 0.2}, [-0.1, 0.0, 0.1, 0.2]),
    ],
)
def test_asrfmap_derives_missing_time_values(
    kwargs: dict[str, float],
    expected_edges: list[float],
) -> None:
    rf_map = asrfmap(np.zeros((2, 2, 3)), **kwargs)

    np.testing.assert_allclose(
        rf_map.time_bin_edges_s,
        expected_edges,
        rtol=0.0,
        atol=1e-15,
    )


def test_asrfmap_accepts_array_like_input_and_float_roundoff() -> None:
    rf_map = asrfmap(
        [[1, 2], [3, 4]],
        start_time=0.0,
        end_time=0.1 + 0.2,
        time_bin=0.3,
    )

    np.testing.assert_array_equal(rf_map.to_2d_array(), [[1, 2], [3, 4]])
    np.testing.assert_allclose(rf_map.time_bin_edges_s, [0.0, 0.3])


@pytest.mark.parametrize(
    "values",
    [
        np.asarray(1.0),
        np.zeros(3),
        np.zeros((1, 1, 1, 1)),
        np.empty((0, 2)),
        np.empty((2, 0)),
        np.empty((2, 2, 0)),
        np.asarray([[True]]),
        np.asarray([[1.0 + 2.0j]]),
        np.asarray([[np.nan]]),
        np.asarray([[np.inf]]),
        np.asarray([[-1.0]]),
        np.asarray([["1"]]),
    ],
)
def test_asrfmap_rejects_invalid_arrays(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="array|dimensions|values"):
        asrfmap(values)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_time": True},
        {"start_time": np.nan},
        {"end_time": np.inf},
        {"end_time": 0.0},
        {"end_time": -0.1},
        {"time_bin": 0.0},
        {"time_bin": -0.1},
        {"time_bin": True},
    ],
)
def test_asrfmap_rejects_invalid_time_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="time"):
        asrfmap(np.zeros((2, 2, 3)), **kwargs)  # type: ignore[arg-type]


def test_asrfmap_rejects_inconsistent_time_bin_count() -> None:
    with pytest.raises(
        ValueError,
        match=r"time_bin \* n_time_bins must equal end_time - start_time",
    ):
        asrfmap(
            np.zeros((2, 2, 3)),
            start_time=-0.1,
            end_time=0.2,
            time_bin=0.2,
        )


def test_load_rf_maps_returns_one_object_per_unit_in_source_order(
    tmp_path: Path,
) -> None:
    counts = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload(counts)))

    assert isinstance(rf_maps, RFMapList)
    assert len(rf_maps) == 2
    assert all(isinstance(rf_map, RFMap) for rf_map in rf_maps)
    assert [rf_map.unit_id for rf_map in rf_maps] == [41, 7]

    first = rf_maps.by_index(0)
    second = rf_maps.by_index(1)
    assert first is rf_maps.by_unit_id(41)
    assert second is rf_maps.by_unit_id(7)
    np.testing.assert_array_equal(first.spike_counts, counts[0])
    np.testing.assert_array_equal(second.spike_counts, counts[1])

    assert first.spike_counts.shape == (2, 2, 3)
    np.testing.assert_array_equal(first.x_positions, [-10.0, 10.0])
    np.testing.assert_array_equal(first.y_positions, [-5.0, 5.0])
    np.testing.assert_array_equal(first.time_bin_edges_s, [-0.1, 0.0, 0.1, 0.2])
    np.testing.assert_array_equal(first.presentation_counts, [[2, 3], [4, 5]])
    assert first.metadata["sessionName"] == "fixture-session"


def test_rf_map_list_retrieval_errors_are_unambiguous(tmp_path: Path) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))

    with pytest.raises(IndexError):
        rf_maps.by_index(-1)
    with pytest.raises(IndexError):
        rf_maps.by_index(2)
    with pytest.raises(KeyError):
        rf_maps.by_unit_id(0)


def test_rf_map_list_never_confuses_an_index_with_a_unit_id(
    tmp_path: Path,
) -> None:
    payload = _payload(unit_pool=[1, 0])
    rf_maps = load_rf_maps(_write_payload(tmp_path, payload))

    assert rf_maps.by_index(1).unit_id == 0
    assert rf_maps.by_unit_id(1) is rf_maps.by_index(0)
    assert rf_maps.by_unit_id(1) is not rf_maps.by_index(1)


def test_rf_map_list_string_index_uses_unit_id_and_preserves_sequence_indexing(
    tmp_path: Path,
) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))
    summed_rf_maps = rf_maps.sum(0.0, 0.2)

    assert rf_maps["41"] is rf_maps.by_unit_id(41)
    assert summed_rf_maps["41"] is summed_rf_maps.by_unit_id(41)
    assert rf_maps[0] is rf_maps.by_index(0)
    assert rf_maps[-1] is rf_maps.by_index(1)
    assert rf_maps[:1] == [rf_maps.by_index(0)]


@pytest.mark.parametrize("unit_id", ["999", "not-a-unit-id"])
def test_rf_map_list_invalid_string_index_raises_key_error(
    tmp_path: Path,
    unit_id: str,
) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))

    with pytest.raises(KeyError):
        rf_maps[unit_id]


def test_rf_map_list_uses_original_indices_after_reordering(tmp_path: Path) -> None:
    loaded = load_rf_maps(_write_payload(tmp_path, _payload()))
    reordered = RFMapList([loaded[1], loaded[0]], loaded.source_path)

    assert reordered[0] is loaded.by_index(1)
    assert reordered[-1] is loaded.by_index(0)
    assert reordered.by_index(0) is loaded.by_index(0)
    assert reordered.by_index(1) is loaded.by_index(1)


def test_load_rf_maps_allows_missing_presentation_counts(tmp_path: Path) -> None:
    payload = _payload()
    payload.pop("stimulusPresentationCounts")

    rf_maps = load_rf_maps(_write_payload(tmp_path, payload))

    assert all(rf_map.presentation_counts is None for rf_map in rf_maps)


def test_sum_returns_a_new_single_bin_rf_map(tmp_path: Path) -> None:
    counts = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    original = load_rf_maps(_write_payload(tmp_path, _payload(counts))).by_index(1)

    summed = original.sum(0.0, 0.2)

    assert isinstance(summed, RFMap)
    assert summed is not original
    assert summed.unit_id == 7
    assert summed.spike_counts.shape == (2, 2, 1)
    np.testing.assert_array_equal(
        summed.spike_counts[..., 0],
        counts[1, ..., 1:3].sum(axis=-1),
    )
    np.testing.assert_array_equal(summed.time_bin_edges_s, [0.0, 0.2])

    # Summing must not mutate the source object.
    np.testing.assert_array_equal(original.spike_counts, counts[1])
    np.testing.assert_array_equal(
        original.time_bin_edges_s,
        [-0.1, 0.0, 0.1, 0.2],
    )


@pytest.mark.parametrize("edge", [-0.1, 0.0, 0.1, 0.2])
def test_sum_equal_endpoints_returns_one_zero_bin(
    tmp_path: Path,
    edge: float,
) -> None:
    original = load_rf_maps(_write_payload(tmp_path, _payload())).by_index(0)

    summed = original.sum(edge, edge)

    assert summed.spike_counts.shape == (2, 2, 1)
    np.testing.assert_array_equal(summed.spike_counts, np.zeros((2, 2, 1)))
    np.testing.assert_array_equal(summed.time_bin_edges_s, [edge, edge])


def test_sum_uses_strict_edges_with_small_float_tolerance(
    tmp_path: Path,
) -> None:
    original = load_rf_maps(_write_payload(tmp_path, _payload())).by_index(0)

    tolerated = original.sum(5e-13, 0.2 - 5e-13)
    np.testing.assert_array_equal(
        tolerated.spike_counts[..., 0],
        original.spike_counts[..., 1:3].sum(axis=-1),
    )

    with pytest.raises(ValueError):
        original.sum(2e-12, 0.2)
    with pytest.raises(ValueError):
        original.sum(0.0, 0.2 - 2e-12)
    with pytest.raises(ValueError):
        original.sum(0.05, 0.2)


def test_sum_prefers_exact_edge_and_rejects_ambiguous_tolerance(
    tmp_path: Path,
) -> None:
    counts = np.arange(8, dtype=float).reshape(1, 2, 2, 2)
    rf_map = load_rf_maps(
        _write_payload(
            tmp_path,
            _payload(
                counts,
                unit_pool=[5],
                time_bin_edges=[0.0, 1e-12, 2e-12],
            ),
        )
    ).by_index(0)

    exact = rf_map.sum(1e-12, 2e-12)
    np.testing.assert_array_equal(exact.spike_counts[..., 0], counts[0, ..., 1])

    with pytest.raises(ValueError, match="multiple timeBinEdges"):
        rf_map.sum(0.5e-12, 2e-12)


def test_sum_rejects_reversed_window(tmp_path: Path) -> None:
    original = load_rf_maps(_write_payload(tmp_path, _payload())).by_index(0)

    with pytest.raises(ValueError):
        original.sum(0.1, 0.0)


def test_rf_map_exposes_data_fields_getters_summary_and_compact_repr(
    tmp_path: Path,
) -> None:
    rf_map = load_rf_maps(_write_payload(tmp_path, _payload())).by_unit_id(41)

    expected_fields = {
        "unit_index",
        "unit_id",
        "spike_counts",
        "x_positions",
        "y_positions",
        "time_bin_edges_s",
        "presentation_counts",
        "metadata",
        "source_path",
    }
    summary = rf_map.summary()
    assert summary["unit_index"] == 0
    assert summary["unit_id"] == 41
    assert summary["shape"] == (2, 2, 3)
    assert summary["time_window_s"] == (-0.1, 0.2)
    assert set(summary["public_data"]) == expected_fields

    rendered = repr(rf_map)
    assert "RFMap(" in rendered
    assert "unit_id=41" in rendered
    assert "shape=(2, 2, 3)" in rendered
    assert "spike_counts=array" not in rendered
    assert len(rendered) < 300


def test_rf_map_array_conversions(
    tmp_path: Path,
) -> None:
    counts = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    rf_map = load_rf_maps(_write_payload(tmp_path, _payload(counts))).by_index(0)

    np.testing.assert_array_equal(rf_map.spike_counts, counts[0])
    with pytest.raises(ValueError, match="one time bin|single time bin|sum"):
        rf_map.to_2d_array()

    summed = rf_map.sum(0.0, 0.2)
    expected_2d = counts[0, ..., 1:3].sum(axis=-1)
    expected_x = expected_2d.sum(axis=0)
    expected_y = expected_2d.sum(axis=1)

    np.testing.assert_array_equal(summed.to_2d_array(), expected_2d)
    np.testing.assert_array_equal(summed.to_1d_array(axis="x"), expected_x)
    np.testing.assert_array_equal(summed.to_1d_array(axis="y"), expected_y)
    assert not rf_map.spike_counts.flags.writeable
    assert not summed.to_2d_array().flags.writeable
    assert not summed.to_1d_array(axis="x").flags.writeable


def test_rf_map_where_matches_numpy_native_axis_indices() -> None:
    values = np.asarray(
        [
            [[0, 1], [2, 0], [3, 4]],
            [[5, 6], [0, 7], [8, 0]],
        ],
        dtype=np.int16,
    )
    rf_map = asrfmap(values)

    result = rf_map.where(0)
    expected = np.where(values == 0)

    assert len(result) == 3
    for actual_indices, expected_indices in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual_indices, expected_indices)
        assert actual_indices.dtype == np.dtype(np.intp)
        assert not actual_indices.flags.writeable


def test_rf_map_where_keeps_the_summed_singleton_time_axis() -> None:
    rf_map = asrfmap(np.asarray([[0, 1], [2, 0]], dtype=np.float32))

    y_indices, x_indices, time_indices = rf_map.where(0)

    np.testing.assert_array_equal(y_indices, [0, 1])
    np.testing.assert_array_equal(x_indices, [0, 1])
    np.testing.assert_array_equal(time_indices, [0, 0])
    assert all(indices.size == 0 for indices in rf_map.where(99))


@pytest.mark.parametrize("axis", ["z", "horizontal", 0, None, True])
def test_rf_map_array_projection_rejects_invalid_axes(
    tmp_path: Path,
    axis: object,
) -> None:
    rf_map = load_rf_maps(_write_payload(tmp_path, _payload())).by_index(0)
    summed = rf_map.sum(0.0, 0.2)

    with pytest.raises(ValueError, match="axis"):
        summed.to_1d_array(axis=axis)  # type: ignore[arg-type]


def test_rf_map_list_exposes_shared_getters_and_compact_repr(
    tmp_path: Path,
) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))

    assert rf_maps.n_units == 2
    assert rf_maps.shape == (2, 2, 2, 3)
    assert rf_maps.unit_indices == [0, 1]
    assert rf_maps.unit_ids == [41, 7]

    rendered = repr(rf_maps)
    assert "RFMapList(" in rendered
    assert "n_units=2" in rendered
    assert "unit_ids=[41, 7]" in rendered
    assert "spike_counts" not in rendered
    assert len(rendered) < 300


def test_rf_map_list_batch_sum_and_array_conversions(tmp_path: Path) -> None:
    counts = np.arange(24, dtype=float).reshape(2, 2, 2, 3)
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload(counts)))

    np.testing.assert_array_equal(rf_maps.to_4d_array(), counts)
    with pytest.raises(ValueError, match="one time bin|single time bin|sum"):
        rf_maps.to_2d_array()

    summed = rf_maps.sum(0.0, 0.2)
    expected_3d = counts[..., 1:3].sum(axis=-1)

    assert isinstance(summed, RFMapList)
    assert summed is not rf_maps
    assert summed.source_path == rf_maps.source_path
    assert summed.unit_ids == rf_maps.unit_ids
    assert summed.shape == (2, 2, 2, 1)
    assert all(rf_map.n_time_bins == 1 for rf_map in summed)
    np.testing.assert_array_equal(summed.to_4d_array(), expected_3d[..., None])
    np.testing.assert_array_equal(summed.to_2d_array(), expected_3d)
    np.testing.assert_array_equal(
        summed.to_1d_array(axis="x"),
        expected_3d.sum(axis=1),
    )
    np.testing.assert_array_equal(
        summed.to_1d_array(axis="y"),
        expected_3d.sum(axis=2),
    )
    assert not summed.to_4d_array().flags.writeable
    assert not summed.to_2d_array().flags.writeable
    assert not summed.to_1d_array(axis="x").flags.writeable


def test_rf_map_list_where_matches_numpy_and_returns_list_positions(
    tmp_path: Path,
) -> None:
    counts = np.ones((2, 2, 3, 2), dtype=np.int32)
    counts[0, 0, 1, 0] = 0
    counts[0, 1, 2, 1] = 0
    counts[1, 1, 0, 0] = 0
    payload = _payload(
        counts,
        unit_pool=[41, 7],
        time_bin_edges=[-0.1, 0.0, 0.1],
    )
    payload["xPositions"] = [-10.0, 0.0, 10.0]
    payload["stimulusPresentationCounts"] = [[2, 3, 4], [5, 6, 7]]
    rf_maps = load_rf_maps(_write_payload(tmp_path, payload))

    result = rf_maps.where(0)
    expected = np.where(counts == 0)

    assert len(result) == 4
    for actual_indices, expected_indices in zip(result, expected, strict=True):
        np.testing.assert_array_equal(actual_indices, expected_indices)
        assert actual_indices.dtype == np.dtype(np.intp)
        assert not actual_indices.flags.writeable

    # The first result contains RFMapList offsets, including duplicates. It is
    # not a recorded unit-ID array.
    np.testing.assert_array_equal(result[0], [0, 0, 1])
    zero_offsets = np.unique(result[0])
    zero_unit_ids = np.asarray(rf_maps.unit_ids)[zero_offsets]
    np.testing.assert_array_equal(zero_unit_ids, [41, 7])


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), None, "0", [], np.nan, np.inf, -np.inf],
)
def test_where_rejects_non_finite_numeric_scalar(
    tmp_path: Path,
    value: object,
) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))

    with pytest.raises(ValueError, match="value must be (numeric|finite)"):
        rf_maps[0].where(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="value must be (numeric|finite)"):
        rf_maps.where(value)  # type: ignore[arg-type]



def _formal_trial_data(
    *,
    time_range_s: tuple[float, float] = (0.0, 0.1),
) -> dict[str, object]:
    return {
        "responses": np.asarray([[1.0, 2.0, 5.0, 8.0]]),
        "position_ids": np.asarray([0, 0, 1, 1]),
        "stratum_ids": np.zeros(4, dtype=np.int64),
        "shape": (1, 2),
        "unit_ids": np.asarray([0]),
        "x_positions": np.asarray([0.0, 1.0]),
        "y_positions": np.asarray([0.0]),
        "time_range_s": time_range_s,
        "polarity": "on",
    }


def _center_detection_result(
    final_mask: object,
    response_map: object,
    null_mean_map: object,
) -> dict[str, np.ndarray]:
    return {
        "final_mask": np.asarray(final_mask, dtype=bool),
        "response_map": np.asarray(response_map, dtype=float),
        "null_mean_map": np.asarray(null_mean_map, dtype=float),
    }


def test_rf_center_uses_effect_weight_and_projects_the_same_2d_bin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rf_map = asrfmap(np.zeros((1, 3)))
    details = _center_detection_result(
        [[True, True, True]],
        [[4.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0]],
    )
    forwarded_options: list[dict[str, object]] = []

    def fake_detect_rf(
        self: RFMap,
        trials: object,
        *,
        is_shuffle: bool = True,
        **options: object,
    ) -> dict[str, np.ndarray]:
        forwarded_options.append(options)
        return details

    monkeypatch.setattr(RFMap, "detect_rf", fake_detect_rf)

    full_mask = rf_map.rf_2d(
        {},
        show_progress=False,
    )
    center_2d = rf_map.rf_2d(
        {},
        return_center=True,
        show_progress=False,
    )
    center_x = rf_map.rf_1d(
        {},
        axis="x",
        return_center=True,
        show_progress=False,
    )
    center_y = rf_map.rf_1d(
        {},
        axis="y",
        return_center=True,
        show_progress=False,
    )

    np.testing.assert_array_equal(full_mask, [[1, 1, 1]])
    # Bins 0 and 1 have equal medoid cost; bin 0 wins on local weight.
    np.testing.assert_array_equal(center_2d, [[1, 0, 0]])
    np.testing.assert_array_equal(center_x, [1, 0, 0])
    np.testing.assert_array_equal(center_y, [1])
    assert center_2d.dtype == np.uint8
    assert not center_2d.flags.writeable
    assert all(options == {"show_progress": False} for options in forwarded_options)


def test_rf_center_respects_less_tail_and_horizontal_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rf_map = asrfmap(np.zeros((1, 5)))
    details = _center_detection_result(
        [[True, False, False, True, True]],
        [[1.0, 0.0, 0.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0, 0.0, 0.0]],
    )

    def fake_detect_rf(
        self: RFMap,
        trials: object,
        *,
        is_shuffle: bool = True,
        **options: object,
    ) -> dict[str, np.ndarray]:
        return details

    monkeypatch.setattr(RFMap, "detect_rf", fake_detect_rf)

    plain = rf_map.rf_2d(
        {},
        return_center=True,
        show_progress=False,
        wrap_x=False,
    )
    wrapped = rf_map.rf_2d(
        {},
        return_center=True,
        show_progress=False,
        wrap_x=True,
    )

    np.testing.assert_array_equal(plain, [[0, 0, 0, 1, 0]])
    np.testing.assert_array_equal(wrapped, [[0, 0, 0, 0, 1]])

    less_details = _center_detection_result(
        [[True, True, True]],
        [[0.0, 3.0, 3.0]],
        [[4.0, 4.0, 4.0]],
    )

    def fake_less_detect_rf(
        self: RFMap,
        trials: object,
        *,
        is_shuffle: bool = True,
        **options: object,
    ) -> dict[str, np.ndarray]:
        return less_details

    monkeypatch.setattr(RFMap, "detect_rf", fake_less_detect_rf)
    less_center = asrfmap(np.zeros((1, 3))).rf_2d(
        {},
        return_center=True,
        show_progress=False,
        alternative="less",
        wrap_x=False,
    )
    np.testing.assert_array_equal(less_center, [[1, 0, 0]])


def test_rf_center_batch_skips_empty_units_and_uses_equal_weight_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _payload(
        np.zeros((3, 1, 3, 1)),
        unit_pool=[0, 1, 2],
        time_bin_edges=[0.0, 1.0],
    )
    payload["xPositions"] = [0.0, 1.0, 2.0]
    payload["yPositions"] = [0.0]
    payload["stimulusPresentationCounts"] = [[1, 1, 1]]
    maps = load_rf_maps(_write_payload(tmp_path, payload))
    details = _center_detection_result(
        [
            [[True, False, True]],
            [[False, False, False]],
            [[False, True, False]],
        ],
        np.zeros((3, 1, 3)),
        np.zeros((3, 1, 3)),
    )
    progress_calls: list[tuple[list[int], dict[str, object]]] = []

    def fake_detect_rf(
        self: RFMapList,
        trials: object,
        *,
        is_shuffle: bool = True,
        **options: object,
    ) -> dict[str, np.ndarray]:
        assert options == {"show_progress": True}
        return details

    def fake_tqdm(iterable: object, **options: object) -> object:
        values = np.asarray(iterable).tolist()
        progress_calls.append((values, options))
        return iterable

    monkeypatch.setattr(RFMapList, "_detect_rf", fake_detect_rf)
    monkeypatch.setattr(rfmap_module, "tqdm", fake_tqdm)

    centers = maps.rf_2d({}, return_center=True, show_progress=True)

    np.testing.assert_array_equal(
        centers,
        [
            [[1, 0, 0]],
            [[0, 0, 0]],
            [[0, 1, 0]],
        ],
    )
    assert progress_calls == [([0, 2], {"desc": "Center", "unit": "unit"})]
    assert centers.shape == (3, 1, 3)
    assert centers.dtype == np.uint8
    assert not centers.flags.writeable

    progress_calls.clear()
    empty_details = _center_detection_result(
        np.zeros((3, 1, 3), dtype=bool),
        np.zeros((3, 1, 3)),
        np.zeros((3, 1, 3)),
    )

    def fake_empty_detect_rf(
        self: RFMapList,
        trials: object,
        *,
        is_shuffle: bool = True,
        **options: object,
    ) -> dict[str, np.ndarray]:
        return empty_details

    monkeypatch.setattr(RFMapList, "_detect_rf", fake_empty_detect_rf)
    empty_centers = maps.rf_2d({}, return_center=True, show_progress=True)
    np.testing.assert_array_equal(empty_centers, np.zeros((3, 1, 3)))
    assert progress_calls == []


def test_rf_map_list_sum_progress_is_one_optional_unit_bar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))
    progress_calls: list[tuple[int, dict[str, object]]] = []

    def fake_tqdm(iterable: object, **options: object) -> object:
        progress_calls.append((len(iterable), options))  # type: ignore[arg-type]
        return iterable

    monkeypatch.setattr(rfmap_module, "tqdm", fake_tqdm)

    summed = rf_maps.sum(-0.1, 0.0, show_progress=True)
    silent = rf_maps.sum(-0.1, 0.0, show_progress=False)

    assert len(summed) == len(rf_maps)
    assert len(silent) == len(rf_maps)
    assert progress_calls == [(2, {"desc": "Sum", "unit": "unit"})]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"return_center": 1}, "return_center"),
        ({"show_progress": 1}, "show_progress"),
    ],
)
def test_rf_output_rejects_non_boolean_controls_before_detection(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    rf_map = asrfmap(np.zeros((1, 1)))

    def unexpected_detection(*_args: object, **_options: object) -> None:
        pytest.fail("invalid output controls must fail before RF detection")

    monkeypatch.setattr(RFMap, "detect_rf", unexpected_detection)
    with pytest.raises(ValueError, match=message):
        rf_map.rf_2d({}, **kwargs)  # type: ignore[arg-type]


def test_rf_map_list_sum_rejects_non_boolean_progress_control(
    tmp_path: Path,
) -> None:
    rf_maps = load_rf_maps(_write_payload(tmp_path, _payload()))

    with pytest.raises(ValueError, match="show_progress"):
        rf_maps.sum(-0.1, 0.0, show_progress=1)  # type: ignore[arg-type]


def test_formal_rf_detection_uses_the_summed_objects_only_time_bin() -> None:
    rf_map = asrfmap(
        np.zeros((1, 2)),
        start_time=0.0,
        end_time=0.1,
    )
    trials = _formal_trial_data()

    details = rf_map.detect_rf(
        trials,
        is_shuffle=False,
        cluster_forming_z=1.5,
    )
    mask_2d = rf_map.rf_2d(
        trials,
        is_shuffle=False,
        cluster_forming_z=1.5,
    )
    center_2d = rf_map.rf_2d(
        trials,
        is_shuffle=False,
        return_center=True,
        show_progress=False,
        cluster_forming_z=1.5,
    )
    mask_x = rf_map.rf_1d(
        trials,
        axis="x",
        is_shuffle=False,
        cluster_forming_z=1.5,
    )
    mask_y = rf_map.rf_1d(
        trials,
        axis="y",
        is_shuffle=False,
        cluster_forming_z=1.5,
    )

    assert not details["is_significance_tested"]
    np.testing.assert_array_equal(details["unit_ids"], [0])
    assert not details["unit_ids"].flags.writeable
    assert "unit_id" not in details
    np.testing.assert_array_equal(details["final_mask"], [[False, True]])
    np.testing.assert_array_equal(mask_2d, [[0, 1]])
    np.testing.assert_array_equal(center_2d, [[0, 1]])
    np.testing.assert_array_equal(mask_x, [0, 1])
    np.testing.assert_array_equal(mask_y, [1])
    assert mask_2d.dtype == np.uint8
    assert not mask_2d.flags.writeable


def test_formal_rf_detection_rejects_unsummed_or_mismatched_windows() -> None:
    unsummed = asrfmap(
        np.zeros((1, 2, 2)),
        start_time=0.0,
        end_time=0.2,
    )

    with pytest.raises(ValueError, match="exactly one|sum"):
        unsummed.rf_2d(_formal_trial_data(time_range_s=(0.0, 0.2)))

    summed = unsummed.sum(0.0, 0.1)
    with pytest.raises(ValueError, match="response window"):
        summed.rf_2d(_formal_trial_data(time_range_s=(0.1, 0.2)))


def test_formal_rf_defaults_to_shuffle_and_exposes_cluster_details() -> None:
    rf_map = asrfmap(
        np.zeros((1, 2)),
        start_time=0.0,
        end_time=0.1,
    )
    details = rf_map.detect_rf(
        _formal_trial_data(),
        cluster_forming_z=1.5,
        n_permutations=19,
        random_seed=3,
        batch_size=7,
    )

    assert details["is_significance_tested"]
    assert details["null_max_masses"].shape == (19,)
    assert np.isfinite(details["cluster_pvalues"]).all()
    assert details["n_workers"] == 1


def test_rf_map_list_aligns_formal_trial_rows_by_unit_id(tmp_path: Path) -> None:
    counts = np.zeros((2, 1, 2, 1), dtype=float)
    payload = _payload(
        counts,
        unit_pool=[11, 22],
        time_bin_edges=[0.0, 0.1],
    )
    payload["xPositions"] = [0.0, 1.0]
    payload["yPositions"] = [0.0]
    payload["stimulusPresentationCounts"] = [[2, 2]]
    rf_maps = load_rf_maps(_write_payload(tmp_path, payload))
    trials = {
        "responses": np.asarray([
            [8.0, 5.0, 2.0, 1.0],  # unit 22: first position candidate
            [1.0, 2.0, 5.0, 8.0],  # unit 11: second position candidate
        ]),
        "position_ids": np.asarray([0, 0, 1, 1]),
        "stratum_ids": np.zeros(4, dtype=np.int64),
        "shape": (1, 2),
        "unit_ids": np.asarray([22, 11]),
        "x_positions": np.asarray([0.0, 1.0]),
        "y_positions": np.asarray([0.0]),
        "time_range_s": (0.0, 0.1),
    }

    result = rf_maps._detect_rf(
        trials,
        is_shuffle=False,
        cluster_forming_z=1.5,
    )

    assert result["unit_ids"].tolist() == [11, 22]
    np.testing.assert_array_equal(
        result["final_mask"][:, 0, :],
        [[False, True], [True, False]],
    )

    masks_2d = rf_maps.rf_2d(trials, is_shuffle=False)
    centers_2d = rf_maps.rf_2d(
        trials,
        is_shuffle=False,
        return_center=True,
        show_progress=False,
    )
    masks_1d = rf_maps.rf_1d(trials, is_shuffle=False)
    assert masks_2d.shape == (2, 1, 2)
    assert masks_1d.shape == (2, 2)
    assert masks_2d.dtype == np.uint8
    assert masks_1d.dtype == np.uint8
    assert not masks_2d.flags.writeable
    assert not masks_1d.flags.writeable
    np.testing.assert_array_equal(masks_2d[:, 0, :], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(centers_2d[:, 0, :], [[0, 1], [1, 0]])
    np.testing.assert_array_equal(masks_1d, [[0, 1], [1, 0]])


def test_removed_spatial_sd_helpers_are_not_part_of_the_api() -> None:
    rf_map = asrfmap(np.zeros((1, 1)))
    rf_maps = RFMapList([rf_map], Path("<array>"))

    assert not hasattr(rf_map, "spatial_sd_2d")
    assert not hasattr(rf_map, "spatial_sd_1d")
    assert not hasattr(rf_maps, "spatial_sd_2d")
    assert not hasattr(rf_maps, "spatial_sd_1d")
