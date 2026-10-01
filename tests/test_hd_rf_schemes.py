import json

import numpy as np
import pandas as pd

from Utils import hd_rf_schemes


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
    )))
    masks = np.zeros((2, 2, 4), dtype=bool)
    masks[1, 0, 1] = True  # Unit 7 is detected; its global maximum lies outside the mask.
    np.savez(rf.with_suffix(".npz"), unit_ids=[8, 7], mask_2d=masks)

    pairs, provenance = hd_rf_schemes.load_peak_pairs(tc, rf, hd_is_clockwise=False)
    rows = pairs.set_index("unit_id")

    assert pairs.unit_id.tolist() == [7, 8, 9, 10, 11]
    assert rows.loc[7, "rf_ego_deg"] == -10.
    assert rows.loc[7, "rf_peak_y"] == -20.
    assert rows.loc[7, "rf_peak_hz"] == 10.
    assert rows.loc[7, "rf_saved_2d"] and not rows.loc[7, "rf_peak_in_saved_2d_mask"]
    assert rows.loc[7, "rf_missing_bins"] == 4 and rows.loc[7, "peak_valid"]
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
