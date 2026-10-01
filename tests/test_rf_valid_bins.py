import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from Utils import rf_detection
from Utils.rf_analysis import rf_bin_qc
from Utils.rfmap import RFMapList, asrfmap


def _map(values, unit_id=0, *, occupancy=None, presentations=None):
    values = np.asarray(values, dtype=float)
    metadata = {}
    if occupancy is not None:
        metadata["occupancyTimeSec"] = np.asarray(occupancy).ravel().tolist()
    if presentations is None:
        presentations = np.ones_like(values)
    return replace(
        asrfmap(np.zeros_like(values), end_time=0.2),
        unit_index=unit_id,
        unit_id=unit_id,
        spike_counts=values[..., None].copy(),
        presentation_counts=np.asarray(presentations),
        metadata=metadata,
    )


def _record_detection(monkeypatch):
    calls = []
    original = rf_detection.detect_rf

    def record(responses, position_ids, shape, **options):
        result = original(responses, position_ids, shape, **options)
        calls.append((np.asarray(responses), np.asarray(position_ids), result))
        return result

    monkeypatch.setattr(rf_detection, "detect_rf", record)
    return calls


def test_rf_2d_standardizes_only_206_finite_nonzero_bins(monkeypatch):
    values = np.linspace(1.0, 8.0, 210).reshape(7, 30)
    values.flat[[0, 1]] = 0
    values.flat[[2, 3]] = np.nan
    rf_map = _map(values)
    calls = _record_detection(monkeypatch)

    mask = rf_map.rf_2d(
        exclude_zero_bins=True, cluster_forming_z=1.0,
        drop_bins=0, wrap_x=False, show_progress=False,
    )
    valid = np.isfinite(values) & (values != 0)
    sample = values[valid]
    responses, positions, result = calls[0]
    assert responses.shape == (1, 206)
    np.testing.assert_array_equal(positions, np.flatnonzero(valid))
    np.testing.assert_array_equal(responses[0], sample)
    np.testing.assert_allclose(result["null_mean_map"][0][valid], sample.mean())
    np.testing.assert_allclose(result["null_sd_map"][0][valid], sample.std(ddof=0))
    expected = valid & (values >= sample.mean() + sample.std(ddof=0))
    np.testing.assert_array_equal(mask, expected)
    assert not mask[~valid].any()


def test_batch_uses_each_units_valid_positions_and_matches_single_maps():
    first = np.ones((7, 30))
    first[3, 5:9] = [8, 10, 10, 8]
    first.flat[[0, 1]] = 0
    first.flat[[2, 3]] = np.nan
    second = np.ones((7, 30))
    second[4, 20:24] = [8, 10, 10, 8]
    second.flat[[30, 31]] = 0
    second.flat[[32, 33]] = np.nan
    second.flat[34] = np.inf
    sources = [_map(first, 10), _map(second, 20)]
    batch = RFMapList(sources, "<array>")
    options = dict(exclude_zero_bins=True, wrap_x=False, show_progress=False)
    expected_masks = np.stack([source.rf_2d(**options) for source in sources])
    expected_centers = np.stack([
        source.rf_2d(is_center=True, **options) for source in sources
    ])
    np.testing.assert_array_equal(batch.rf_2d(**options), expected_masks)
    np.testing.assert_array_equal(
        batch.rf_2d(is_center=True, **options), expected_centers,
    )
    assert expected_masks.sum(axis=(1, 2)).tolist() == [4, 4]
    assert expected_centers.sum(axis=(1, 2)).tolist() == [1, 1]


@pytest.mark.parametrize("gap", [0.0, np.nan])
def test_excluded_bin_cannot_bridge_components_or_become_center(gap):
    values = np.full((1, 20), 10.0)
    values[0, 5:10] = [1, 1, gap, 1, 1]
    rf_map = _map(values)
    options = dict(
        exclude_zero_bins=True, alternative="less", cluster_forming_z=1.0,
        wrap_x=False, show_progress=False,
    )
    assert not rf_map.rf_2d(drop_bins=2, **options).any()
    assert not rf_map.rf_2d(drop_bins=2, is_center=True, **options).any()
    mask = rf_map.rf_2d(drop_bins=0, **options)
    expected = np.zeros_like(values, dtype=np.uint8)
    expected[0, [5, 6, 8, 9]] = 1
    np.testing.assert_array_equal(mask, expected)
    center = rf_map.rf_2d(drop_bins=0, is_center=True, **options)
    assert center.sum() == 1
    assert center[0, 7] == 0
    assert np.all(center <= mask)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rf_1d_ignores_all_missing_and_zero_collapsed_positions(axis, monkeypatch):
    values = np.ones((2, 30))
    values[:, 5:8] = [8, 10, 8]
    values[:, 20] = 0
    values[:, 21] = np.nan
    if axis == "y":
        values = values.T
    rf_map = _map(values)
    calls = _record_detection(monkeypatch)
    options = dict(exclude_zero_bins=True, drop_bins=0, show_progress=False)
    mask = rf_map.rf_1d(axis=axis, **options)
    collapsed = np.nansum(values, axis=0 if axis == "x" else 1)
    valid = collapsed != 0
    sample = collapsed[valid]
    expected = valid & (collapsed >= sample.mean() + sample.std(ddof=0))
    np.testing.assert_array_equal(mask, expected)
    responses, positions, result = calls[0]
    assert responses.shape == (1, 28)
    np.testing.assert_array_equal(positions, np.flatnonzero(valid))
    np.testing.assert_allclose(result["null_mean_map"][result["valid_mask"]], sample.mean())
    np.testing.assert_allclose(result["null_sd_map"][result["valid_mask"]], sample.std(ddof=0))
    center = rf_map.rf_1d(axis=axis, is_center=True, **options)
    assert center.sum() == 1
    assert not center[[20, 21]].any()
    assert np.all(center <= mask)


@pytest.mark.parametrize("axis", ["x", "y"])
def test_rf_1d_keeps_finite_responses_in_partly_missing_columns(axis, monkeypatch):
    values = np.ones((2, 30))
    values[:, 5:8] = [8, 10, 8]
    values[0, 5] = np.inf
    values[0, 6] = np.nan
    if axis == "y":
        values = values.T
    rf_map = _map(values)
    calls = _record_detection(monkeypatch)
    mask = rf_map.rf_1d(
        axis=axis, exclude_zero_bins=True, drop_bins=0, show_progress=False,
    )
    finite_values = np.where(np.isfinite(values), values, np.nan)
    collapsed = np.nansum(finite_values, axis=0 if axis == "x" else 1)
    expected = collapsed >= collapsed.mean() + collapsed.std(ddof=0)
    np.testing.assert_array_equal(mask, expected)
    responses, positions, result = calls[0]
    np.testing.assert_array_equal(positions, np.arange(30))
    np.testing.assert_array_equal(responses[0], collapsed)
    assert responses[0, 5] == 8
    assert responses[0, 6] == 10
    np.testing.assert_allclose(
        result["null_mean_map"][result["valid_mask"]], collapsed.mean(),
    )


@pytest.mark.parametrize("method", ["rf_2d", "rf_1d"])
@pytest.mark.parametrize("mismatch", ["window", "x_positions", "y_positions"])
def test_excluding_zero_bins_preserves_batch_geometry_checks(method, mismatch):
    first = _map(np.ones((2, 5)), 10)
    second = _map(np.ones((2, 5)), 20)
    if mismatch == "window":
        second = replace(second, time_bin_edges_s=np.array([0.0, 0.1]))
        message = "share one response window"
    else:
        second = replace(second, **{mismatch: getattr(second, mismatch) + 0.5})
        message = "share " + mismatch.replace("_", " ")
    batch = RFMapList([first, second], "<array>")
    with pytest.raises(ValueError, match=message):
        getattr(batch, method)(exclude_zero_bins=True, show_progress=False)


@pytest.mark.parametrize("method", ["rf_2d", "rf_1d"])
@pytest.mark.parametrize("is_batch", [False, True])
def test_excluding_zero_bins_rejects_shuffle(method, is_batch):
    source = _map(np.ones((2, 5)))
    if is_batch:
        source = RFMapList([source], "<array>")
    with pytest.raises(ValueError, match="exclude_zero_bins"):
        getattr(source, method)(
            exclude_zero_bins=True, is_shuffle=True, show_progress=False,
        )


def test_zero_bin_option_has_distinct_cache_identity_and_reuses_results(tmp_path, monkeypatch):
    values = np.array([[0, 0, 10, 10, 10, 10, 10, 10, 12, 12]], dtype=float)
    rf_map = _map(values)
    path = tmp_path / "rf_result.npz"
    calls = _record_detection(monkeypatch)
    options = dict(drop_bins=0, result_path=path, show_progress=False)
    default_mask = rf_map.rf_2d(**options)
    with np.load(path, allow_pickle=False) as saved:
        default_key = saved["cache_key"].item()
        manifest = json.loads(saved["manifest_json"].item())
        assert manifest["parameters"]["exclude_zero_bins"] is False
    np.testing.assert_array_equal(
        rf_map.rf_2d(exclude_zero_bins=False, **options), default_mask,
    )
    assert len(calls) == 1

    mask = rf_map.rf_2d(exclude_zero_bins=True, **options)
    assert len(calls) == 2
    assert not default_mask.any()
    np.testing.assert_array_equal(mask, [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    with np.load(path, allow_pickle=False) as saved:
        assert saved["cache_key"].item() != default_key
        manifest = json.loads(saved["manifest_json"].item())
        assert manifest["parameters"]["exclude_zero_bins"] is True
    for source in (rf_map, _map(values)):
        center = source.rf_2d(exclude_zero_bins=True, is_center=True, **options)
        assert center.sum() == 1
        assert np.all(center <= mask)
    assert len(calls) == 2


def test_rf_bin_qc_accepts_inclusive_limits_and_requires_both():
    accepted = np.ones((7, 30))
    accepted.flat[[0, 1]] = [np.nan, 0]
    accepted.flat[[2, 3]] = 0
    occupancy = np.ones_like(accepted)
    occupancy.flat[1] = 0
    too_missing = np.ones_like(accepted)
    too_missing.flat[[0, 2]] = np.nan
    too_zero = np.ones_like(accepted)
    too_zero.flat[[0, 2, 3]] = 0
    nonfinite = np.ones_like(accepted)
    nonfinite.flat[[0, 1]] = [np.inf, 0]
    maps = [
        _map(accepted, 10, occupancy=occupancy),
        _map(too_missing, 11, occupancy=occupancy),
        _map(too_zero, 12, occupancy=occupancy),
        _map(np.ones_like(accepted), 13, occupancy=occupancy),
        _map(nonfinite, 14, occupancy=occupancy),
    ]
    qc = rf_bin_qc(RFMapList(maps, "<array>"))
    np.testing.assert_array_equal(qc["unit_ids"], [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(qc["missing_bins"], [2, 3, 1, 1, 2])
    np.testing.assert_array_equal(qc["zero_bins"], [2, 0, 3, 0, 0])
    np.testing.assert_array_equal(qc["valid_bins"], [206, 207, 206, 209, 208])
    np.testing.assert_array_equal(qc["keep"], [True, False, False, True, True])


def test_rf_bin_qc_counts_unpresented_zero_as_missing_only():
    accepted = np.ones((7, 30))
    accepted.flat[0] = np.nan
    accepted.flat[[1, 2, 3]] = 0
    too_zero = np.ones_like(accepted)
    too_zero.flat[[1, 2, 3, 4]] = 0
    presentations = np.ones_like(accepted)
    presentations.flat[1] = 0
    maps = [
        _map(accepted, 10, presentations=presentations),
        _map(too_zero, 20, presentations=presentations),
    ]
    qc = rf_bin_qc(RFMapList(maps, "<array>"))
    np.testing.assert_array_equal(qc["missing_bins"], [2, 1])
    np.testing.assert_array_equal(qc["zero_bins"], [2, 3])
    np.testing.assert_array_equal(qc["valid_bins"], [206, 206])
    np.testing.assert_array_equal(qc["keep"], [True, False])


@pytest.mark.parametrize("fill", [0.0, np.nan])
def test_rf_bin_qc_rejects_units_without_valid_bins(fill):
    maps = [_map(np.full((7, 30), fill), 10)]
    qc = rf_bin_qc(
        RFMapList(maps, "<array>"), max_missing_bins=210, max_zero_bins=210,
    )
    np.testing.assert_array_equal(qc["valid_bins"], [0])
    np.testing.assert_array_equal(qc["keep"], [False])


def test_notebook_default_qc_limits_are_two():
    notebook = Path(__file__).resolve().parents[1] / "locate_rf.ipynb"
    source = "".join(json.loads(notebook.read_text())["cells"][1]["source"])
    namespace = {"Path": Path}
    exec(compile(source, str(notebook), "exec"), namespace)
    assert namespace["max_missing_bins"] == 2
    assert namespace["max_zero_bins"] == 2
