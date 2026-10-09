from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
from Utils.rf_cache import read_rf_result, rf_result_path, save_rf_result
from Utils.rfmap import RFMapList, asrfmap


def test_minimum_bin_keeps_source_ties_and_missing_curves():
    profiles = pd.DataFrame(
        [[9, 5, 1, 2], [5, 1, 1, np.nan], [np.nan] * 4],
        index=[7, 9, 11], columns=[30, -150, -30, 150],
    )
    np.testing.assert_array_equal(
        comparison.peak_angles(profiles, [7, 9, 11, 13], use_min=True),
        [-30, -150, np.nan, np.nan],
    )
    np.testing.assert_array_equal(
        comparison.peak_angles(profiles, [7, 9, 11, 13]),
        [30, 30, np.nan, np.nan],
    )


@pytest.mark.parametrize("mode", ["native", "aligned", "sum"])
@pytest.mark.parametrize("rf_first", [False, True])
def test_pairing_uses_hd_maximum_and_rf_minimum_without_changing_curves(mode, rf_first):
    hd = pd.DataFrame([[0, 9, 3, 1], [2, 3, 9, 1]], index=[7, 11], columns=[-180, -90, 0, 90])
    rf = pd.DataFrame([[9, 6, 4, 1], [3, 1, 5, 9]], index=hd.index, columns=hd.columns)
    originals = hd.copy(), rf.copy()
    if rf_first:
        result = comparison.prepare_comparison(rf, hd, mode=mode, reference_min=True)
        assert result["order"] == [11, 7]
        np.testing.assert_array_equal(result["reference_peak_deg"], [-90, 90])
        np.testing.assert_array_equal(result["matched_peak_deg"], [0, -90])
    else:
        result = comparison.prepare_comparison(hd, rf, mode=mode, matched_min=True)
        assert result["order"] == [7, 11]
        np.testing.assert_array_equal(result["reference_peak_deg"], [-90, 0])
        np.testing.assert_array_equal(result["matched_peak_deg"], [90, -90])
    pd.testing.assert_frame_equal(hd, originals[0])
    pd.testing.assert_frame_equal(rf, originals[1])


@pytest.mark.parametrize("unit_ids", [
    [40, 7, 13, 9],
    ["m15:260630:A:40", "m15:260630:A:7", "m15:260630:A:13", "m15:260630:A:9"],
    ["zebra", "alpha", "middle", "missing"],
])
@pytest.mark.parametrize("mode", ["native", "aligned", "sum"])
@pytest.mark.parametrize("reference_min", [False, True])
def test_paired_peak_ties_keep_reference_rows_regardless_of_unit_labels(
    unit_ids, mode, reference_min,
):
    reference = pd.DataFrame(
        [[0, 1, 9, 2], [0, 3, 9, 1], [0, 9, 2, 1], [np.nan] * 4],
        index=unit_ids, columns=[-180, -90, 0, 90],
    )
    if reference_min:
        reference = 10 - reference
    matched = pd.DataFrame(
        [[1, 9, 4, 0], [2, 1, 4, 9], [9, 3, 1, 0], [3, 1, 9, 0]],
        index=unit_ids, columns=reference.columns,
    ).iloc[::-1]

    result = comparison.prepare_comparison(
        reference, matched, mode=mode, reference_min=reference_min,
    )

    assert result["order"] == [unit_ids[i] for i in [2, 0, 1, 3]]
    np.testing.assert_array_equal(result["reference_peak_deg"], [-90, 0, 0, np.nan])
    np.testing.assert_array_equal(result["matched_peak_deg"], [-180, -90, 90, 0])


def test_explicit_pair_order_overrides_peak_and_missing_row_order():
    reference = pd.DataFrame([[0, 1], [0, 2], [np.nan, np.nan]],
                             index=["z", "a", "missing"], columns=[-90, 90])
    matched = pd.DataFrame([[1, 0], [0, 1], [1, 0]],
                           index=reference.index, columns=reference.columns)
    order = ["missing", "a", "z"]

    result = comparison.prepare_comparison(reference, matched, order=order)

    assert result["order"] == order
    np.testing.assert_array_equal(result["reference_peak_deg"], [np.nan, 90, 90])


@pytest.mark.parametrize("is_batch", [False, True])
@pytest.mark.parametrize("method,z", [("rf_2d", 1.5), ("rf_1d", 0.75)])
def test_inhibitory_defaults_use_requested_sd_cutoff(is_batch, method, z):
    values = np.arange(1, 61, dtype=float).reshape(2, 30)
    first = asrfmap(values)
    source = RFMapList([first, replace(first, unit_index=1, unit_id=7)], "<array>") if is_batch else first
    response = values if method == "rf_2d" else values.sum(axis=0)
    expected = response <= response.mean() - z * response.std()
    if is_batch:
        expected = np.stack([expected, expected])
    actual = getattr(source, method)(rf_type="inhibitory", drop_bins=0, show_progress=False)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("rf_type,selected", [("2d", [11]), ("1d", [13]), ("either", [11, 13])])
def test_explicit_selection_uses_inhibitory_files_and_preserves_raw_responses(tmp_path, rf_type, selected):
    source = tmp_path / "regular.rfmap"
    first = asrfmap(np.arange(1, 31, dtype=float)[None, :], end_time=0.2)
    maps = RFMapList([replace(first, unit_index=i, unit_id=unit_id) for i, unit_id in enumerate([7, 11, 13])], source)
    for suffix, row in [("", 0), ("_inhibitory", 1), ("_inhibitory_1d", 2)]:
        mask = np.zeros((3, 1, 30), dtype=np.uint8)
        mask[row, 0, :3] = 1
        center = np.zeros_like(mask)
        center[row, 0, 1] = 1
        save_rf_result(source.with_name(f"{source.stem}{suffix}.npz"),
                       mask_2d=mask, center_2d=center, unit_ids=[7, 11, 13],
                       manifest={}, cache_key=f"test{suffix}")
    all_profiles = comparison.rf_profiles(maps)
    dimensions = ("2d", "1d") if rf_type == "either" else (rf_type,)
    results = [read_rf_result(rf_result_path(source, dimension=dimension, rf_type="inhibitory"))
               for dimension in dimensions]
    inhibitory = comparison.select_rf_profiles(all_profiles, *results)
    expected_keys = [("A", unit_id) for unit_id in selected]
    pd.testing.assert_frame_equal(inhibitory, all_profiles.loc[expected_keys])
    assert set(inhibitory.attrs["unit_info"]) == set(expected_keys)
