from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
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


@pytest.mark.parametrize("plot", [comparison.plot_sort, comparison.plot_align, comparison.plot_sum])
@pytest.mark.parametrize("rf_first", [False, True])
def test_pairing_uses_hd_maximum_and_rf_minimum_without_changing_curves(plot, rf_first):
    hd = pd.DataFrame([[0, 9, 3, 1], [2, 3, 9, 1]], index=[7, 11], columns=[-180, -90, 0, 90])
    rf = pd.DataFrame([[9, 6, 4, 1], [3, 1, 5, 9]], index=hd.index, columns=hd.columns)
    originals = hd.copy(), rf.copy()
    if rf_first:
        result = plot(rf, hd, reference_min=True, show=False)
        assert result["order"] == [11, 7]
        np.testing.assert_array_equal(result["reference_peak_deg"], [-90, 90])
        np.testing.assert_array_equal(result["matched_peak_deg"], [0, -90])
    else:
        result = plot(hd, rf, matched_min=True, show=False)
        assert result["order"] == [7, 11]
        np.testing.assert_array_equal(result["reference_peak_deg"], [-90, 0])
        np.testing.assert_array_equal(result["matched_peak_deg"], [90, -90])
    pd.testing.assert_frame_equal(hd, originals[0])
    pd.testing.assert_frame_equal(rf, originals[1])
    for figure, _ in result["figures"]:
        plt.close(figure)


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
def test_loader_selects_inhibitory_files_and_preserves_raw_responses(tmp_path, monkeypatch, rf_type, selected):
    source = tmp_path / "regular.rfmap"
    first = asrfmap(np.arange(1, 31, dtype=float)[None, :], end_time=0.2)
    maps = RFMapList([replace(first, unit_index=i, unit_id=unit_id) for i, unit_id in enumerate([7, 11, 13])], source)
    monkeypatch.setattr(comparison, "load_rf_maps", lambda *args, **kwargs: maps)
    for suffix, row in [("", 0), ("_inhibitory", 1), ("_inhibitory_1d", 2)]:
        mask = np.zeros((3, 1, 30), dtype=np.uint8)
        mask[row, 0, :3] = 1
        np.savez(source.with_name(f"{source.stem}{suffix}.npz"), mask_2d=mask, unit_ids=[7, 11, 13])
    all_profiles = comparison.load_rf_profiles(source)
    inhibitory = comparison.load_rf_profiles(source, rf_type=rf_type, rf_detection="inhibitory")
    expected_keys = [("A", unit_id) for unit_id in selected]
    pd.testing.assert_frame_equal(inhibitory, all_profiles.loc[expected_keys])
    assert set(inhibitory.attrs["unit_info"]) == set(expected_keys)
