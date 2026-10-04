"""RF source loading preserves zero responses and the declared rate units."""

import json

import numpy as np
import pytest
from scipy.io import savemat

from Utils.rfmap import load_rf_maps
from Utils.rf_rates import resolve_presentation_counts


def _write_source(path, values, edges, **metadata):
    values = np.asarray(values, dtype=object)
    n_units, n_y, n_x, _ = values.shape
    source = {
        "unitsSpikeCounts": values.tolist(),
        "unitsSpikeCountsSize": list(values.shape),
        "unitPool": list(range(7, 7 + n_units)),
        "xPositions": list(range(n_x)),
        "yPositions": list(range(n_y)),
        "timeBinEdges": edges,
        **metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(source))
    return path


@pytest.mark.parametrize("use_rates", [False, True])
def test_json_null_nan_and_zero_remain_distinct_in_counts_and_rates(tmp_path, use_rates):
    source = _write_source(
        tmp_path / "zero_responses.rfmap",
        [[[[None, 2], [np.nan, 4], [0, 6]]]],
        [0.0, 0.05, 0.2],
        stimulusPresentationCounts=[[2, 2, 2]],
    )
    maps = load_rf_maps(source, unit_firing_rate=use_rates)
    expected = [[[np.nan, 2 / 0.3], [np.nan, 4 / 0.3], [0, 6 / 0.3]]] if use_rates else [
        [[np.nan, 2], [np.nan, 4], [0, 6]],
    ]
    np.testing.assert_allclose(maps[0].spike_counts, expected)
    assert maps[0].metadata["responseUnits"] == ("Hz" if use_rates else "spike_count")


def test_loaded_unequal_width_bins_have_explicit_sum_and_mean_rate(tmp_path):
    source = _write_source(
        tmp_path / "unequal_width.rfmap",
        [[[[2, 3, 8], [6, 9, 24]]]],
        [0.0, 0.05, 0.2, 0.4],
        stimulusPresentationCounts=[[2, 3]],
    )
    counts = load_rf_maps(source)
    np.testing.assert_array_equal(counts[0].spike_counts, [[[2, 3, 8], [6, 9, 24]]])
    rates = load_rf_maps(source, unit_firing_rate=True)
    np.testing.assert_allclose(rates[0].spike_counts, [[[20, 10, 20], [40, 20, 40]]])
    np.testing.assert_allclose(counts.to_firing_rate()[0].spike_counts, rates[0].spike_counts)
    np.testing.assert_allclose(counts.sum(0.0, 0.4)[0].spike_counts, [[[13], [39]]])
    np.testing.assert_allclose(rates.sum(0.0, 0.4)[0].spike_counts, [[[50], [100]]])
    np.testing.assert_allclose(rates.mean_rate(0.0, 0.4)[0].spike_counts, [[[16.25], [32.5]]])
    np.testing.assert_allclose(rates.mean_rate(0.05, 0.4)[0].spike_counts,
                               [[[11 / (2 * 0.35)], [33 / (3 * 0.35)]]])
    assert rates.mean_rate(0.05, 0.4)[0].time_window_s == (0.05, 0.4)
    with pytest.raises(ValueError, match="requires Hz"):
        counts.mean_rate(0.0, 0.4)


def test_legacy_display_hz_requires_explicit_exposure_reconstruction(tmp_path, monkeypatch):
    session = tmp_path / "261001_13"
    (session / "data").mkdir(parents=True)
    trials = [
        {"Square_PositionX": x, "Square_PositionY": 0, "Square_Luminance": 1}
        for x in (0, 0, 1, 1)
    ]
    savemat(session / "261001.mat", {"trials": trials})
    np.save(session / "data/on_list_times.npy", [0, 0.1, 0.3, 0.6, 1.0])
    source = _write_source(
        session / "data/rfmapping/good/ProbeA/regular_unitsSpikeCounts_261001_13.rfmap",
        [[[[2 / 0.3, 4 / 0.3], [3 / 0.7, 6 / 0.7]]]],
        [0.0, 0.05, 0.2],
        responseUnits="Hz", responseNormalization="occupancyTimeSec",
        occupancyTimeSec=[[0.3, 0.7]],
    )
    def unexpected_reconstruction(*args, **kwargs):
        raise AssertionError("loading must not reconstruct presentation counts")

    monkeypatch.setattr("Utils.rf_rates.resolve_presentation_counts", unexpected_reconstruction)
    loaded = load_rf_maps(source)
    np.testing.assert_allclose(loaded[0].spike_counts, [[[2 / 0.3, 4 / 0.3], [3 / 0.7, 6 / 0.7]]])
    assert loaded[0].metadata["responseNormalization"] == "occupancyTimeSec"
    with pytest.raises(ValueError, match="presentation_counts"):
        loaded.to_firing_rate()
    with pytest.raises(ValueError, match="presentation_counts"):
        load_rf_maps(source, unit_firing_rate=True)
    presentations, provenance = resolve_presentation_counts(
        loaded[0].metadata, source, loaded[0].x_positions, loaded[0].y_positions,
    )
    rates = loaded.to_firing_rate(presentation_counts=presentations)
    np.testing.assert_allclose(rates[0].spike_counts, [[[20, 4 / 0.3], [30, 20]]])
    np.testing.assert_array_equal(rates[0].presentation_counts, [[2, 2]])
    np.testing.assert_allclose(rates.mean_rate(0.0, 0.2)[0].spike_counts, [[[15], [22.5]]])
    assert rates[0].metadata["responseNormalization"] == "presentation_count_time"
    assert rates[0].metadata["presentationCountSource"]["method"] == "supplied"
    assert provenance["method"] == "raw_trial_footprint"
    monkeypatch.setattr("Utils.rf_rates.resolve_presentation_counts", resolve_presentation_counts)
    reconstructed = loaded[0].to_firing_rate(reconstruct_presentations=True)
    np.testing.assert_allclose(reconstructed.spike_counts, rates[0].spike_counts)
    np.testing.assert_array_equal(reconstructed.presentation_counts, [[2, 2]])
    assert reconstructed.metadata["presentationCountSource"] == provenance


def test_precomputed_normalized_hz_loads_without_presentation_metadata(tmp_path):
    source = _write_source(
        tmp_path / "precomputed.rfmap", [[[[4, 16], [5, 25]]]],
        [0.0, 0.05, 0.2],
        responseUnits="Hz", responseNormalization="already_normalized",
    )
    rates = load_rf_maps(source)
    np.testing.assert_array_equal(rates[0].spike_counts, [[[4, 16], [5, 25]]])
    np.testing.assert_allclose(rates.sum(0.0, 0.2)[0].spike_counts, [[[20], [30]]])
    np.testing.assert_allclose(rates.mean_rate(0.0, 0.2)[0].spike_counts, [[[13], [20]]])
    assert rates[0].metadata["responseUnits"] == "Hz"
    assert rates[0].metadata["responseNormalization"] == "already_normalized"
    assert rates[0].presentation_counts is None


def test_unpresented_counts_convert_to_missing_rates_without_erasing_observed_zero(tmp_path):
    source = _write_source(
        tmp_path / "exposure.rfmap", [[[[0], [0], [None]]]],
        [0.0, 0.1], stimulusPresentationCounts=[[2, 0, 0]],
    )
    counts = load_rf_maps(source)
    np.testing.assert_allclose(counts[0].spike_counts, [[[0], [0], [np.nan]]])
    rates = counts.to_firing_rate()
    np.testing.assert_allclose(rates[0].spike_counts, [[[0], [np.nan], [np.nan]]])


def test_batch_reconstructs_presentations_once_only_when_explicitly_requested(tmp_path, monkeypatch):
    source = _write_source(
        tmp_path / "legacy_counts.rfmap",
        [[[[2, 4], [3, 6]]], [[[4, 8], [6, 12]]]],
        [0.0, 0.1, 0.3],
    )
    calls = []
    provenance = {"method": "raw_trial_footprint", "analyzed_trials": 5}

    def reconstruct(raw, source_path, x_positions, y_positions):
        calls.append(source_path)
        assert source_path == source
        np.testing.assert_array_equal(x_positions, [0, 1])
        np.testing.assert_array_equal(y_positions, [0])
        return np.array([[2, 3]]), provenance

    monkeypatch.setattr("Utils.rf_rates.resolve_presentation_counts", reconstruct)
    loaded = load_rf_maps(source)
    for candidate in (loaded, loaded[0]):
        with pytest.raises(ValueError, match="presentation_counts"):
            candidate.to_firing_rate()
    with pytest.raises(ValueError, match="presentation_counts"):
        load_rf_maps(source, unit_firing_rate=True)
    assert calls == []

    rates = loaded.to_firing_rate(reconstruct_presentations=True)
    assert calls == [source]
    np.testing.assert_allclose(rates.to_4d_array(), [
        [[[10, 10], [10, 10]]], [[[20, 20], [20, 20]]],
    ])
    for unit in rates:
        np.testing.assert_array_equal(unit.presentation_counts, [[2, 3]])
        assert unit.metadata["presentationCountSource"] == provenance
    assert loaded[0].presentation_counts is None
