"""RF source loading preserves zero responses and the declared rate units."""

import json

import numpy as np
import pytest
from scipy.io import savemat

from Utils.rfmap import load_rf_maps


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
def test_json_null_nan_and_zero_load_as_zero_in_counts_and_rates(tmp_path, use_rates):
    source = _write_source(
        tmp_path / "zero_responses.rfmap",
        [[[[None, 2], [np.nan, 4], [0, 6]]]],
        [0.0, 0.05, 0.2],
        stimulusPresentationCounts=[[2, 2, 2]],
    )
    maps = load_rf_maps(source, unit_firing_rate=use_rates)
    expected = [[[0, 2 / 0.3], [0, 4 / 0.3], [0, 6 / 0.3]]] if use_rates else [
        [[0, 2], [0, 4], [0, 6]],
    ]
    np.testing.assert_allclose(maps[0].spike_counts, expected)
    assert np.isfinite(maps[0].spike_counts).all()
    assert maps[0].metadata["responseUnits"] == ("Hz" if use_rates else "spike_count")
    assert maps[0].metadata["zeroBinPolicy"] == "null_and_nan_as_zero"


def test_loaded_unequal_width_bins_sum_counts_or_average_hz(tmp_path):
    source = _write_source(
        tmp_path / "unequal_width.rfmap",
        [[[[2, 3, 8], [6, 9, 24]]]],
        [0.0, 0.05, 0.2, 0.4],
        stimulusPresentationCounts=[[2, 3]],
    )
    counts = load_rf_maps(source, unit_firing_rate=False)
    rates = load_rf_maps(source)
    np.testing.assert_allclose(rates[0].spike_counts, [[[20, 10, 20], [40, 20, 40]]])
    np.testing.assert_allclose(counts.sum(0.0, 0.4)[0].spike_counts, [[[13], [39]]])
    np.testing.assert_allclose(rates.sum(0.0, 0.4)[0].spike_counts, [[[16.25], [32.5]]])
    np.testing.assert_allclose(rates.sum(0.05, 0.4)[0].spike_counts,
                               [[[11 / (2 * 0.35)], [33 / (3 * 0.35)]]])
    assert rates.sum(0.05, 0.4)[0].time_window_s == (0.05, 0.4)


def test_legacy_display_hz_restores_counts_before_presentation_normalization(tmp_path):
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
    rates = load_rf_maps(source)
    np.testing.assert_allclose(rates[0].spike_counts, [[[20, 4 / 0.3], [30, 20]]])
    np.testing.assert_array_equal(rates[0].presentation_counts, [[2, 2]])
    np.testing.assert_allclose(rates.sum(0.0, 0.2)[0].spike_counts, [[[15], [22.5]]])
    assert rates[0].metadata["responseNormalization"] == "presentation_count_time"
    assert rates[0].metadata["presentationCountSource"]["method"] == "raw_trial_footprint"


def test_precomputed_normalized_hz_loads_without_presentation_metadata(tmp_path):
    source = _write_source(
        tmp_path / "precomputed.rfmap", [[[[4, 16], [5, 25]]]],
        [0.0, 0.05, 0.2],
        responseUnits="Hz", responseNormalization="already_normalized",
    )
    rates = load_rf_maps(source)
    np.testing.assert_array_equal(rates[0].spike_counts, [[[4, 16], [5, 25]]])
    np.testing.assert_allclose(rates.sum(0.0, 0.2)[0].spike_counts, [[[13], [20]]])
    assert rates[0].metadata["responseUnits"] == "Hz"
    assert rates[0].metadata["responseNormalization"] == "already_normalized"
    assert rates[0].presentation_counts is None
