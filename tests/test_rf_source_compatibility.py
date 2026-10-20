from __future__ import annotations

import json

import numpy as np
import pytest

from Utils.rfmap import load_rf_maps


@pytest.fixture(params=["json", "indexed_npz"])
def write_source(request, tmp_path, write_indexed_rfmap):
    def write(payload):
        path = tmp_path / "regular_unitsSpikeCounts_260101_1.rfmap"
        if request.param == "indexed_npz":
            return write_indexed_rfmap(path, payload)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return write


def test_source_formats_preserve_unit_order_axes_counts_and_normalization(write_source):
    counts = np.arange(36, dtype=float).reshape(2, 2, 3, 3)
    occupancy = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]])
    path = write_source({
        "unitsSpikeCounts": counts.tolist(),
        "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [41, 7],
        "xPositions": [-12.0, 0.0, 12.0],
        "yPositions": [-6.0, 6.0],
        "timeBinEdges": [-0.1, 0.0, 0.1, 0.2],
        "occupancyTimeSec": occupancy.tolist(),
        "responseUnits": "spike_count",
        "responseNormalization": "none",
    })

    raw = load_rf_maps(path, unit_firing_rate=False)
    rates = load_rf_maps(path)
    assert raw.unit_ids == rates.unit_ids == [41, 7]
    np.testing.assert_array_equal(raw.to_4d_array(), counts)
    np.testing.assert_array_equal(raw[0].x_positions, [-12.0, 0.0, 12.0])
    np.testing.assert_array_equal(raw[0].y_positions, [-6.0, 6.0])
    np.testing.assert_array_equal(raw[0].time_bin_edges_s, [-0.1, 0.0, 0.1, 0.2])
    np.testing.assert_array_equal(rates.to_4d_array(), counts / occupancy[None, :, :, None])
    np.testing.assert_array_equal(
        raw.sum(0.0, 0.2, show_progress=False).to_2d_array(), counts[..., 1:].sum(axis=-1),
    )
    assert raw[0].metadata["responseUnits"] == "spike_count"
    assert raw[0].metadata["responseNormalization"] == "none"
    assert rates[0].metadata["responseUnits"] == "Hz"
    assert rates[0].metadata["responseNormalization"] == "occupancyTimeSec"
    assert rates[0].sum(0.0, 0.2).metadata["responseUnits"] == "Hz"


def test_unpresented_bins_stay_missing_and_do_not_block_detection(write_source):
    path = write_source({
        "unitsSpikeCounts": [[[[0.0], [20.0], [0.0]], [[0.0], [0.0], [10.0]]]],
        "unitsSpikeCountsSize": [1, 2, 3, 1],
        "unitPool": [7],
        "xPositions": [-12.0, 0.0, 12.0],
        "yPositions": [-6.0, 6.0],
        "timeBinEdges": [0.0, 0.2],
        "occupancyTimeSec": [[0.0, 2.0, 2.0], [0.0, 0.0, 2.0]],
    })
    with np.errstate(divide="raise", invalid="raise"):
        maps = load_rf_maps(path)
    np.testing.assert_array_equal(
        maps.to_2d_array(), [[[np.nan, 10.0, 0.0], [np.nan, np.nan, 5.0]]],
    )
    np.testing.assert_array_equal(maps.to_1d_array(axis="x"), [[np.nan, 10.0, 5.0]])
    np.testing.assert_array_equal(maps.to_1d_array(axis="y"), [[10.0, 5.0]])
    expected_mask = [[[0, 1, 0], [0, 0, 0]]]
    for selected in (maps, load_rf_maps(path, unit_firing_rate=False)):
        np.testing.assert_array_equal(
            selected.rf_2d(cluster_forming_z=1.0, drop_bins=0, wrap_x=False, show_progress=False),
            expected_mask,
        )


def test_saved_rates_accept_missing_unpresented_bins_without_renormalizing(write_source):
    path = write_source({
        "unitsSpikeCounts": [[[[None], [10.0], [0.0], [5.0]]]],
        "unitsSpikeCountsSize": [1, 1, 4, 1],
        "unitPool": [7],
        "xPositions": [-12.0, 0.0, 12.0, 24.0],
        "yPositions": [0.0],
        "timeBinEdges": [0.0, 0.2],
        "occupancyTimeSec": [[0.0, 2.0, 2.0, 2.0]],
        "stimulusPresentationCounts": [[0, 10, 10, 10]],
        "responseUnits": "Hz",
        "responseNormalization": "already_normalized",
    })
    for use_firing_rate in (False, True):
        maps = load_rf_maps(path, unit_firing_rate=use_firing_rate)
        np.testing.assert_array_equal(maps.to_2d_array(), [[[np.nan, 10.0, 0.0, 5.0]]])
        assert maps[0].metadata["responseNormalization"] == "already_normalized"
        np.testing.assert_array_equal(
            maps.rf_2d(cluster_forming_z=1.0, drop_bins=0, show_progress=False),
            [[[0, 1, 0, 0]]],
        )
