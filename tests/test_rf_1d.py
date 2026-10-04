from dataclasses import replace

import numpy as np
import pytest

from Utils.rfmap import RFMap, RFMapList, asrfmap
from Utils.rf_cache import read_rf_result


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rf_1d_thresholds_collapsed_responses(axis):
    profile = np.zeros(30)
    profile[5:8] = [8, 10, 8]
    profile[18:21] = 5
    values = np.tile(profile, (7, 1))
    if axis == "y":
        values = values.T
    rf_map = asrfmap(values)
    collapsed = values.sum(axis=0 if axis == "x" else 1)
    expected = collapsed >= collapsed.mean() + collapsed.std()
    mask = rf_map.rf_1d(axis=axis, show_progress=False)
    np.testing.assert_array_equal(mask, expected)
    assert not mask.flags.writeable
    assert mask.dtype == np.uint8
    assert not np.array_equal(
        mask, rf_map.rf_1d(axis=axis, cluster_forming_z=1.5, show_progress=False),
    )
    maps = RFMapList([rf_map, replace(rf_map, unit_index=1, unit_id=7)], "<array>")
    np.testing.assert_array_equal(
        maps.rf_1d(axis=axis, show_progress=False), [expected, expected],
    )


def test_rf_1d_can_preserve_the_original_2d_projection(tmp_path, monkeypatch):
    rf_map = asrfmap([[10, 10, 0, 0, 0, 0], [0, 0, 10, 10, 0, 0]])
    options = dict(drop_bins=0, wrap_x=False, show_progress=False)
    assert not rf_map.rf_1d(**options).any()
    result_path = tmp_path / "result_2d.npz"
    mask = rf_map.rf_2d(result_path=result_path, **options)
    detected_rf = read_rf_result(result_path)
    rf_map.rf_2d(result_path=result_path, cluster_forming_z=1.0, **options)
    lower_threshold_rf = read_rf_result(result_path)

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("projection must use the supplied detection")

    monkeypatch.setattr(RFMap, "_detect_rf", unexpected_detection)
    np.testing.assert_array_equal(
        rf_map.rf_1d(collapse_from_2d=True, detected_rf=detected_rf),
        mask.any(axis=0),
    )
    np.testing.assert_array_equal(
        rf_map.rf_1d(collapse_from_2d=True, detected_rf=lower_threshold_rf),
        [1, 1, 1, 1, 0, 0],
    )


def test_rf_1d_saved_center_reuses_collapsed_result(tmp_path, monkeypatch):
    values = np.zeros((7, 30))
    values[:, 5:8] = [8, 10, 8]
    path = tmp_path / "result_1d.npz"
    rf_map = asrfmap(values)
    mask = rf_map.rf_1d(result_path=path, show_progress=False)

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("the saved result should be reused")

    monkeypatch.setattr(RFMap, "_detect_rf", unexpected_detection)
    for source in (rf_map, asrfmap(values)):
        center = source.rf_1d(is_center=True, result_path=path, show_progress=False)
        assert center.sum() == 1
        assert center[6] == 1
        assert np.all(center <= mask)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["mask_2d"].shape == (1, 1, 30)
        assert saved["center_2d"].shape == (1, 1, 30)


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("rf_type", ["excitatory", "inhibitory"])
@pytest.mark.parametrize("exclude_zero_bins", [False, True])
def test_batch_saved_views_match_single_unit_views(
        tmp_path, monkeypatch, axis, rf_type, exclude_zero_bins):
    values = np.full((7, 30), 4.0)
    values[:, 5:8] = 12
    values[:, 17:20] = 0.5
    values[0, 0] = 0
    if axis == "y":
        values = values.T
    maps = [asrfmap(values), replace(asrfmap(values * 2), unit_id=7, unit_index=1)]
    options = dict(axis=axis, rf_type=rf_type, exclude_zero_bins=exclude_zero_bins,
                   show_progress=False)
    expected_mask = np.stack([rf_map.rf_1d(**options) for rf_map in maps])
    expected_center = np.stack([rf_map.rf_1d(is_center=True, **options) for rf_map in maps])
    result_path = tmp_path / "batch_1d.npz"
    batch = RFMapList(maps, "<array>")
    np.testing.assert_array_equal(batch.rf_1d(result_path=result_path, **options), expected_mask)

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("the saved batch result should be reused")

    monkeypatch.setattr(RFMapList, "_detect_rf", unexpected_detection)
    fresh = RFMapList(maps, "<array>")
    for source in (batch, fresh):
        actual = source.rf_1d(is_center=True, result_path=result_path, **options)
        np.testing.assert_array_equal(actual, expected_center)
        assert not actual.flags.writeable


def test_rf_1d_treats_unpresented_and_nan_columns_as_zero():
    values = np.zeros((2, 30))
    values[:, 5:8] = 10
    values[:, 20] = np.nan
    presentations = np.ones_like(values)
    presentations[0, 25] = 0
    rf_map = replace(asrfmap(np.zeros_like(values)), spike_counts=values[..., None],
                     presentation_counts=presentations)
    expected = np.zeros(30, dtype=np.uint8)
    expected[5:8] = 1
    np.testing.assert_array_equal(rf_map.rf_1d(show_progress=False), expected)
    np.testing.assert_array_equal(rf_map.sum_to_1d().to_1d_array(), values.sum(axis=0))


def test_rf_1d_batch_requires_matching_response_windows():
    first = asrfmap([[0, 1, 0]], end_time=0.1)
    second = replace(asrfmap([[0, 1, 0]], end_time=0.2), unit_index=1, unit_id=7)
    maps = RFMapList([first, second], "<array>")
    with pytest.raises(ValueError, match="share one response window"):
        maps.rf_1d(show_progress=False)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rf_1d_shuffle_uses_collapsed_trial_positions(axis):
    values = np.zeros((3, 5))
    values[1, 2:4] = 10
    rf_map = asrfmap(values)
    trials = dict(
        responses=np.tile(values.ravel(), (1, 4)),
        position_ids=np.tile(np.arange(values.size), 4),
        shape=values.shape,
        unit_ids=np.array([rf_map.unit_id]),
        x_positions=rf_map.x_positions,
        y_positions=rf_map.y_positions,
        time_range_s=rf_map.time_window_s,
    )
    options = dict(is_shuffle=True, n_permutations=20, show_progress=False)
    mask = rf_map.rf_1d(trials, axis=axis, **options)
    collapsed = asrfmap(values.sum(axis=0, keepdims=True) if axis == "x"
                       else values.sum(axis=1, keepdims=True))
    collapsed_trials = dict(trials, shape=collapsed.shape[:2],
                            x_positions=collapsed.x_positions,
                            y_positions=collapsed.y_positions,
                            position_ids=trials["position_ids"] % 5 if axis == "x"
                            else trials["position_ids"] // 5)
    expected = collapsed.rf_2d(collapsed_trials, cluster_forming_z=1.0, **options)
    np.testing.assert_array_equal(mask, expected.ravel())
    trials["position_ids"][0] = values.size
    with pytest.raises(ValueError, match="position_ids must be between"):
        rf_map.rf_1d(trials, axis=axis, **options)
