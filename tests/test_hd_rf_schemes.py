import json

import numpy as np
import pandas as pd
import pytest

from Utils import hd_rf_schemes
from Utils.rf_cache import save_rf_result


def test_native_global_rate_peak_is_not_projection_or_mask_peak(tmp_path, monkeypatch):
    ids = [7, 8, 9, 10, 11, 12]
    index = pd.MultiIndex.from_product([["m19"], ["260827"], ["A"], ids],
                                      names=["mouse", "date", "probe", "unit_id"])
    hd = pd.DataFrame([[0., 3.]] * len(ids), index=index, columns=[6., 18.])
    hd.attrs["unit_info"] = {key: {"hd_class": 2 if key[-1] == 12 else 3} for key in index}
    monkeypatch.setattr(hd_rf_schemes, "load_tc", lambda *a, **k: hd)
    tc = tmp_path / "tuning_curves.json"
    tc.write_text(json.dumps({"metadata": {"angle_convention_note": "counter-clockwise"}}))
    before = tc.read_bytes()
    # Hz: [[10, 6, NaN, NaN], [0, 6, NaN, NaN]]. Global maximum is
    # x=-10; both raw-count maximum and horizontal projection favor x=10.
    counts = np.array([
        [[20, 30, 0, 0], [0, 6, 0, 0]],
        [[2, 5, 0, 0], [1, 1, 0, 0]],  # Four tied positive maxima, still eligible.
        [[0, 0, 0, 0], [0, 0, 0, 0]],
        [[20, 0, 0, 0], [0, 0, 0, 0]],
        [[20, 0, 0, 0], [0, 0, 0, 0]],
    ])[..., None]
    rf = tmp_path / "rf.json"
    rf.write_text(json.dumps(dict(
        unitsSpikeCounts=counts.tolist(), unitsSpikeCountsSize=list(counts.shape),
        unitPool=[7, 8, 9, 10, 12], xPositions=[-10, 10, 90, 150], yPositions=[-20, 20],
        timeBinEdges=[0., .2], occupancyTimeSec=[[2, 5, 0, 0], [1, 1, 0, 0]],
        stimulusPresentationCounts=[[10, 25, 0, 0], [5, 5, 0, 0]],
    )))
    masks = np.zeros((2, 2, 4), dtype=bool)
    masks[1, 0, 1] = True  # Unit 7 is detected; its global maximum lies outside the mask.
    save_rf_result(rf.with_suffix(".npz"), unit_ids=[8, 7], mask_2d=masks,
                   center_2d=masks, manifest={}, cache_key="test")

    pairs, provenance = hd_rf_schemes.load_peak_pairs(tc, rf, hd_is_clockwise=False)
    rows = pairs.set_index("unit_id")

    assert pairs.unit_id.tolist() == [7, 8, 9, 10, 11]
    assert rows.loc[7, "rf_ego_deg"] == -10.
    assert rows.loc[7, "rf_peak_y"] == -20.
    assert rows.loc[7, "rf_peak_hz"] == 10.
    assert rows.loc[7, "rf_saved_2d"] and not rows.loc[7, "rf_peak_in_saved_2d_mask"]
    assert rows.loc[7, "rf_zero_bins"] == 1 and rows.loc[7, "peak_valid"]
    assert rows.loc[7, "rf_finite_bins"] == 4
    assert rows.loc[8, "rf_peak_ties"] == 4 and rows.loc[8, "peak_valid"]
    assert rows.loc[8, "rf_peak_x_index"] == 0 and rows.loc[8, "rf_peak_y_index"] == 0
    assert rows.loc[9, "exclusion_reason"] == "no_positive_finite_rf_response"
    assert rows.loc[10, "rf_zero_bins"] == 3 and rows.loc[10, "peak_valid"]
    assert rows.loc[11, "exclusion_reason"] == "missing_rf_map"
    assert rows.loc[7, "hd_native_preferred_deg"] == 18.
    assert rows.loc[7, "hd_preferred_deg"] == 342.
    assert provenance["hd_angle_sign"] == -1
    assert tc.read_bytes() == before
    schemes, _ = hd_rf_schemes.select_schemes(pairs)
    assert schemes["scheme1"].unit_id.tolist() == [7, 8, 10]
    assert schemes["scheme2"].unit_id.tolist() == [7]


def test_explicit_detection_path_overrides_adjacent_file(tmp_path, monkeypatch):
    index = pd.MultiIndex.from_tuples([("m20", "260922", "A", 7)],
                                      names=["mouse", "date", "probe", "unit_id"])
    hd = pd.DataFrame([[0., 3.]], index=index, columns=[6., 18.])
    hd.attrs["unit_info"] = {index[0]: {"hd_class": 3}}
    monkeypatch.setattr(hd_rf_schemes, "load_tc", lambda *a, **k: hd)
    tc = tmp_path / "tuning_curves.json"
    tc.write_text(json.dumps({"metadata": {}}))
    rf = tmp_path / "rf.json"
    counts = np.array([[[[20], [6]], [[0], [6]]]])
    rf.write_text(json.dumps(dict(
        unitsSpikeCounts=counts.tolist(), unitsSpikeCountsSize=list(counts.shape),
        unitPool=[7], xPositions=[-10, 10], yPositions=[-20, 20],
        timeBinEdges=[0., .2], occupancyTimeSec=[[1, 1], [1, 1]],
        stimulusPresentationCounts=[[5, 5], [5, 5]],
    )))
    empty = np.zeros((1, 2, 2), dtype=np.uint8)
    save_rf_result(rf.with_suffix(".npz"), unit_ids=[7], mask_2d=empty,
                   center_2d=empty, manifest={}, cache_key="test-empty")
    detection_path = tmp_path / "relation" / "rf_detection.npz"
    detection_path.parent.mkdir()
    mask = np.array([[[0, 1], [0, 0]]], dtype=np.uint8)
    save_rf_result(detection_path, unit_ids=[7], mask_2d=mask, center_2d=mask,
                   manifest={}, cache_key="test-selected")

    pairs, provenance = hd_rf_schemes.load_peak_pairs(
        tc, rf, mouse="m20", date=260922, probe="A", rf_detection_path=detection_path,
    )

    assert pairs.loc[0, "rf_saved_2d"]
    assert not pairs.loc[0, "rf_peak_in_saved_2d_mask"]
    assert provenance["rf_detection_source"] == str(detection_path)
    assert str(detection_path) in provenance["sources_sha256"]
    with pytest.raises(FileNotFoundError):
        hd_rf_schemes.load_peak_pairs(tc, rf, rf_detection_path=tmp_path / "missing.npz")


def test_top_three_bins_are_independent_and_ties_choose_lower_bin():
    angles = [0, 360, 6, 60, 66, 240, 246, 120, 126, 300, 0]
    pairs = pd.DataFrame(dict(
        unit_id=np.arange(len(angles)), hd_plus_rf_deg=angles,
        peak_valid=[True] * 10 + [False], rf_saved_2d=[True] * 9 + [False, True],
        rf_detection_record_present=[True] * 11, exclusion_reason=[""] * 10 + ["missing_rf_map"],
    ))
    original = pairs.copy(deep=True)
    schemes, manifest = hd_rf_schemes.select_schemes(pairs)
    assert manifest["selected_bin_indices"] == [0, 5, 10]
    assert manifest["bin_counts"][0] == 3
    assert manifest["cutoff_tied_bin_indices"] == [5, 10, 20]
    assert manifest["cutoff_tie_crosses_selection"]
    assert manifest["counts"] == {"scheme1": 10, "scheme2": 9, "scheme3": 7}
    assert set(schemes["scheme3"].unit_id) == {0, 1, 2, 3, 4, 7, 8}
    assert schemes["scheme2"].loc[1, "hd_plus_rf_bin"] == 0
    assert manifest["missing"] == [{"unit_id": 10, "exclusion_reason": "missing_rf_map"}]
    pd.testing.assert_frame_equal(pairs, original)


def test_empty_detected_cohort_reports_zero_counts_without_inventing_pairs():
    pairs = pd.DataFrame(dict(unit_id=[1], hd_plus_rf_deg=[12.], peak_valid=[True],
                              rf_saved_2d=[False], rf_detection_record_present=[True],
                              exclusion_reason=[""]))
    schemes, manifest = hd_rf_schemes.select_schemes(pairs)
    assert manifest["counts"] == {"scheme1": 1, "scheme2": 0, "scheme3": 0}
    assert manifest["bin_counts"] == [0] * 30
    assert manifest["selected_bin_indices"] == [0, 1, 2]
    assert schemes["scheme3"].empty
