import json
from dataclasses import replace

import numpy as np
import pytest

from Utils import rf_detection
from Utils.rfmap import RFMapList, asrfmap


def _source(values, is_batch=False):
    first = asrfmap(np.asarray(values, dtype=float))
    if not is_batch:
        return first
    second = replace(first, unit_index=1, unit_id=7)
    return RFMapList([first, second], "<array>")


@pytest.mark.parametrize("method", ["rf_2d", "rf_1d"])
@pytest.mark.parametrize("is_batch", [False, True])
def test_default_remains_excitatory_and_legacy_alternative_takes_precedence(
    method, is_batch,
):
    values = np.full((1, 30), 10.0)
    values[0, 5:8] = [0, 1, 2]
    values[0, 20:23] = [20, 19, 18]
    detect = getattr(_source(values, is_batch), method)
    options = dict(drop_bins=0, show_progress=False)
    excitatory = detect(**options)
    inhibitory = detect(rf_type="inhibitory", **options)
    assert excitatory.any() and inhibitory.any()
    assert not np.any(excitatory & inhibitory)
    np.testing.assert_array_equal(
        detect(rf_type="excitatory", **options), excitatory,
    )
    np.testing.assert_array_equal(
        detect(alternative="less", **options), inhibitory,
    )
    np.testing.assert_array_equal(
        detect(rf_type="inhibitory", alternative="greater", **options),
        excitatory,
    )


@pytest.mark.parametrize("is_batch", [False, True])
def test_inhibitory_threshold_includes_zero_responses_and_weights_suppression(is_batch):
    values = np.full((1, 20), 10.0)
    values[0, 5:7] = [0, 5]
    source = _source(values, is_batch)
    options = dict(rf_type="inhibitory", drop_bins=0, show_progress=False)
    expected = (values <= values.mean() - 1.5 * values.std()).astype(np.uint8)
    expected_center = np.zeros_like(expected)
    expected_center[0, 5] = 1
    if is_batch:
        expected = np.stack([expected, expected])
        expected_center = np.stack([expected_center, expected_center])
    mask = source.rf_2d(**options)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(
        source.rf_2d(is_center=True, **options), expected_center,
    )
    assert mask.dtype == np.uint8
    assert not mask.flags.writeable
    excluded = source.rf_2d(exclude_zero_bins=True, **options)
    assert not excluded[..., 5].any()
    assert excluded[..., 6].all()


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("is_batch", [False, True])
@pytest.mark.parametrize("collapse_from_2d", [False, True])
def test_inhibitory_1d_uses_requested_axis_and_detection_mode(
    axis, is_batch, collapse_from_2d,
):
    values = np.full((3, 30), 10.0)
    values[:, 5:8] = [0, 1, 2]
    if axis == "y":
        values = values.T
    source = _source(values, is_batch)
    options = dict(rf_type="inhibitory", drop_bins=0, show_progress=False)
    if collapse_from_2d:
        spatial_axis = (1 if axis == "x" else 2) if is_batch else (
            0 if axis == "x" else 1
        )
        mask_2d = source.rf_2d(**options)
        center_2d = source.rf_2d(is_center=True, **options)
        expected = mask_2d.any(axis=spatial_axis)
        options = dict(detected_rf={
            "mask_2d": mask_2d if is_batch else mask_2d[None],
            "center_2d": center_2d if is_batch else center_2d[None],
            "unit_ids": source.unit_ids if is_batch else [source.unit_id],
            "manifest": {},
        })
    else:
        collapsed = values.sum(axis=0 if axis == "x" else 1)
        expected = collapsed <= collapsed.mean() - 0.75 * collapsed.std()
        if is_batch:
            expected = np.stack([expected, expected])
    mask = source.rf_1d(axis=axis, collapse_from_2d=collapse_from_2d, **options)
    np.testing.assert_array_equal(mask, expected)
    center = source.rf_1d(
        axis=axis, collapse_from_2d=collapse_from_2d, is_center=True, **options,
    )
    assert np.all(center.sum(axis=-1) == 1)
    assert np.all(center <= mask)


@pytest.mark.parametrize("is_batch", [False, True])
def test_inhibitory_shuffle_matches_mirrored_excitatory_responses(is_batch, monkeypatch):
    values = np.full((3, 12), 10.0)
    values[1, 4:7] = [0, 1, 2]
    inhibitory = _source(values, is_batch)
    excitatory = _source(10 - values, is_batch)
    unit = inhibitory[0] if is_batch else inhibitory
    unit_ids = [0, 7] if is_batch else [0]
    responses = np.tile(values.ravel(), 12)
    trials = dict(
        responses=np.tile(responses, (len(unit_ids), 1)),
        position_ids=np.tile(np.arange(values.size), 12),
        shape=values.shape,
        unit_ids=np.array(unit_ids),
        x_positions=unit.x_positions,
        y_positions=unit.y_positions,
        time_range_s=unit.time_window_s,
    )
    reflected_trials = dict(trials, responses=10 - trials["responses"])
    options = dict(
        is_shuffle=True, n_permutations=39, random_seed=17,
        wrap_x=False, n_jobs=1, show_progress=False,
    )
    results = []
    original = rf_detection.detect_rf

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(rf_detection, "detect_rf", record)
    mask = inhibitory.rf_2d(trials, rf_type="inhibitory", **options)
    expected = excitatory.rf_2d(reflected_trials, **options)
    assert mask.any()
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_allclose(
        results[0]["null_max_masses"], results[1]["null_max_masses"], atol=1e-12,
    )
    np.testing.assert_allclose(
        results[0]["cluster_masses"], results[1]["cluster_masses"], atol=1e-12,
    )
    np.testing.assert_array_equal(
        results[0]["cluster_pvalues"], results[1]["cluster_pvalues"],
    )
    np.testing.assert_array_equal(
        inhibitory.rf_2d(trials, rf_type="inhibitory", is_center=True, **options),
        excitatory.rf_2d(reflected_trials, is_center=True, **options),
    )


def test_polarities_have_distinct_cached_results_and_reuse_saved_inhibitory_center(
    tmp_path, monkeypatch,
):
    values = np.full((1, 30), 10.0)
    values[0, 5:8] = [0, 1, 2]
    values[0, 20:23] = [20, 19, 18]
    source = _source(values)
    path = tmp_path / "rf_result.npz"
    options = dict(result_path=path, show_progress=False)
    excitatory = source.rf_2d(**options)
    with np.load(path, allow_pickle=False) as saved:
        excitatory_key = saved["cache_key"].item()
    inhibitory = source.rf_2d(rf_type="inhibitory", **options)
    assert not np.array_equal(inhibitory, excitatory)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["cache_key"].item() != excitatory_key
        manifest = json.loads(saved["manifest_json"].item())
        assert manifest["parameters"]["alternative"] == "less"

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("the saved inhibitory result should be reused")

    monkeypatch.setattr(rf_detection, "detect_rf", unexpected_detection)
    for candidate in (source, _source(values)):
        center = candidate.rf_2d(rf_type="inhibitory", is_center=True, **options)
        assert center.sum() == 1
        assert np.all(center <= inhibitory)
        np.testing.assert_array_equal(
            candidate.rf_2d(alternative="less", **options), inhibitory,
        )


@pytest.mark.parametrize("method", ["rf_2d", "rf_1d"])
@pytest.mark.parametrize("is_batch", [False, True])
@pytest.mark.parametrize("rf_type", ["both", "less", None])
def test_invalid_rf_type_is_rejected(method, is_batch, rf_type):
    source = _source([[1, 2, 3]], is_batch)
    with pytest.raises(ValueError, match="rf_type"):
        getattr(source, method)(rf_type=rf_type, show_progress=False)
