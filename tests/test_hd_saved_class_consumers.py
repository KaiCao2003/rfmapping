"""Downstream analyses retain the HD class already stored in each TC file."""

import json
from pathlib import Path

import numpy as np
import pytest

from Utils import pitch_direction_comparison as pitch
from RF.rflocate import RFResult, save_rf
from scripts import analyze_adn_rf_correlations as adn


@pytest.fixture
def saved_hd_session(tmp_path):
    root = tmp_path / "recordings"
    data = root / "m14/260609/260609_1/data"
    tc_path = data / "tuning_curves/ProbeA/tuning_curves.tc"
    tc_path.parent.mkdir(parents=True)
    kilosort = data / "kilosort"
    kilosort.mkdir()
    counts = np.ones((3, 180), dtype=int)
    counts[:, [10, 50, 100]] = 4
    saved = {
        "unit_id": [23, 7, 99],
        "angle_bin_edges_deg": np.linspace(0, 360, 181).tolist(),
        "spike_counts": counts.tolist(),
        "occupancy_time_s": np.ones(180).tolist(),
        "firing_rate_hz": counts.astype(float).tolist(),
        # These p values would reject every unit if a reader reclassified it.
        "unit_data": {"hd_class": [3, 0, 3], "rayleigh_p": [.5, .5, .5],
                      "shuffle_p": [.5, .5, .5]},
        "metadata": {"epoch_intervals_s": [[0., 1.2]],
                     "kilosort_dir": str(kilosort),
                     "classification": {"method": "saved_source_definition",
                                        "kappa_cutoff": .1}},
    }
    tc_path.write_text(json.dumps(saved))
    return root, data, tc_path, saved


def test_pitch_builder_uses_saved_hd_classes(saved_hd_session, tmp_path, monkeypatch):
    root, data, tc_path, saved = saved_hd_session
    frames = np.arange(60)
    times = frames / 50
    processed = data / "processed"
    processed.mkdir()
    (processed / "head_direction.json").write_text(json.dumps({"hp4": {
        "frames": frames.tolist(), "head_direction_deg": np.linspace(3, 357, 60).tolist(),
    }}))
    (processed / "pitch_direction.json").write_text(json.dumps({"hp4": {
        "frames": frames.tolist(), "pitch_direction_deg": np.linspace(-60, 60, 60).tolist(),
    }}))
    (data / "session_info.json").write_text(json.dumps({"session_info": {}}))
    (data / "probeA").mkdir()
    np.save(data / "probeA/adc_spike_time.npy", 10 + np.arange(1, 7) / 10)
    np.save(Path(saved["metadata"]["kilosort_dir"]) / "spike_clusters.npy",
            np.array([23, 7, 99, 23, 7, 99]))
    monkeypatch.setattr(pitch, "get_exposure_timestamps",
                        lambda info, directory: (times, 10., {"source": "test_clock"}))
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}

    report = pitch.build_paired_tuning(root, tmp_path / "pitch", "m14", "260609", 1, "A")

    with np.load(report["output"]) as result:
        np.testing.assert_array_equal(result["unit_id"], [23, 99])
        np.testing.assert_array_equal(result["hd_counts"].sum(axis=1), [2, 2])
        np.testing.assert_array_equal(result["pitch_counts"].sum(axis=1), [2, 2])
    assert report["units"] == 2
    assert report["class_source"] == str(tc_path)
    assert {path: path.read_bytes() for path in before} == before


def test_adn_collection_uses_saved_hd_classes(saved_hd_session, tmp_path, monkeypatch):
    root, _, tc_path, saved = saved_hd_session
    rf_path = (root / "m14/260609/260609_3/data/rfmapping/good/-100_400_1ms/ProbeA"
               / "regular_unitsSpikeCounts_260609_3.rfmap")
    rf_path.parent.mkdir(parents=True)
    counts = np.array([[[1, 8, 1], [1, 1, 1]],
                       [[1, 1, 1], [1, 6, 1]],
                       [[7, 1, 1], [1, 1, 1]]])[..., None]
    rf_path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [99, 7, 23], "xPositions": [-10, 0, 10], "yPositions": [-5, 5],
        "timeBinEdges": [0, .2],
    }))
    masks = np.zeros((3, 2, 3), dtype=np.uint8)
    masks[0, 0, 1] = masks[1, 1, 1] = masks[2, 0, 0] = 1
    save_rf(RFResult(masks, masks, [99, 7, 23], {
        "time_range_s": [0, .2],
        "parameters": {"alternative": "greater", "wrap_x": False, "exclude_zero_bins": False},
    }, "saved-rf-result"), rf_path.with_suffix(".npz"))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"recordings": [{
        "mouse": "m14", "date": "260609", "probe": "A", "unit_ids": [7, 23, 99],
    }]}))
    monkeypatch.setattr(adn, "RECORDINGS", [("m14", "260609", 1, 3, "A")])
    output = tmp_path / "adn"
    output.mkdir()
    before = {path: path.read_bytes() for path in (tc_path, rf_path, rf_path.with_suffix(".npz"))}

    units, centers, sources, _ = adn.collect(root, selection, output)

    assert units.loc[units.hd_class == 3, "unit_id"].tolist() == [23, 99]
    assert centers.loc[centers.hd_class == 3, "unit_id"].tolist() == [23, 99]
    assert sources[0]["not_current_hd3_ids"] == [7]
    assert sources[0]["saved_hd_classification"] == saved["metadata"]["classification"]
    assert {path: path.read_bytes() for path in before} == before
