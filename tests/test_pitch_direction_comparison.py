import numpy as np
import pandas as pd
import json

from Utils.pitch_direction_comparison import circular_linear_association, preferred_angles
from Utils.pitch_direction_comparison import (
    rf_vertical_profiles, pitch_rf_regression, load_pitch_rf_pair, cache_name,
    select_min_occupancy, select_pitch_rf_pair, resample_rf_profiles,
)


def test_pitch_peaks_remain_signed_and_missing_curves_have_no_peak():
    hd = pd.DataFrame([[0, 2, 1], [np.nan, np.nan, np.nan]], columns=[6., 18., 30.])
    pitch = pd.DataFrame([[4, 1, 0], [0, 0, 0]], columns=[-60., 0., 60.])
    peaks = preferred_angles(hd, pitch)
    np.testing.assert_allclose(peaks.iloc[0], [18, -60])
    assert peaks.iloc[1].isna().all()


def test_circular_linear_association_is_invariant_to_heading_zero():
    hd = np.arange(0., 360., 12.)
    pitch = 20 * np.sin(np.deg2rad(hd)) - 15
    original = circular_linear_association(hd, pitch, n_permutations=100, seed=3)
    rotated = circular_linear_association((hd + 80) % 360, pitch, n_permutations=100, seed=3)
    np.testing.assert_allclose(original["r"], 1.)
    assert original == rotated
    assert original["p"] == 1 / 101


def test_rf_vertical_profile_uses_rates_and_differs_from_2d_max(tmp_path):
    # First elevation has the largest single bin (10), second the largest sum
    # (6+6). Unequal exposure makes the distinction depend on normalization.
    counts = np.array([[[20, 0], [6, 6]]])[..., None]
    path = tmp_path / "rf.json"
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7], "xPositions": [-10, 10], "yPositions": [-20, 20],
        "timeBinEdges": [0, .2], "occupancyTimeSec": [[2, 1], [1, 1]],
        "stimulusPresentationCounts": [[10, 5], [5, 5]],
    }))
    profiles, peaks = rf_vertical_profiles(path, mouse="m", date="d", probe="A")
    np.testing.assert_array_equal(profiles.columns, [-20, 20])
    np.testing.assert_allclose(profiles.iloc[0], [12, 10])
    assert peaks.iloc[0].rf_vertical_profile_peak_deg == -20
    assert peaks.iloc[0].rf_2d_max_elevation_deg == 20
    np.savez(tmp_path / cache_name("m", "d", 1, "A"), unit_id=[7],
             hd_counts=[[1., 1.]], hd_occupancy_s=[1., 1.], hd_edges=[0, 180, 360],
             pitch_counts=[[2., 1.]], pitch_occupancy_s=[1., 1.], pitch_edges=[-90, 0, 90])
    pitch, native_rf, native_peaks = load_pitch_rf_pair(
        tmp_path, path, mouse="m", date="d", pitch_session=1, probe="A",
    )
    np.testing.assert_array_equal(native_rf.columns, [-20, 20])
    first = select_pitch_rf_pair(pitch, native_rf, native_peaks)
    second = select_pitch_rf_pair(pitch, native_rf, native_peaks, rf_peak_method="max_bin_2d")
    assert first[2].index.equals(second[2].index)
    assert first[2].rf_peak_deg.iloc[0] == -20
    assert second[2].rf_peak_deg.iloc[0] == 20
    display = resample_rf_profiles(first[1])
    assert len(display.columns) == 91
    assert display.loc[:, -45].isna().all()
    assert first[2].rf_peak_deg.iloc[0] == -20


def test_pitch_rf_regression_preserves_signed_linear_angles():
    peaks = pd.DataFrame({"pitch_peak_deg": [-40, -20, 0, 20], "rf_peak_deg": [-25, -15, -5, 5]})
    result = pitch_rf_regression(peaks)
    np.testing.assert_allclose([result["slope"], result["intercept"], result["r_squared"]], [.5, -5, 1])


def test_occupancy_mask_is_explicit_and_keeps_native_bins():
    curves = pd.DataFrame([[2., 3., 4.]], columns=[-60., 0., 60.])
    curves.attrs["occupancy_s"] = [.5, 1., 2.]
    selected = select_min_occupancy(curves, min_occupancy_s=1.)
    assert np.isnan(selected.iloc[0, 0])
    np.testing.assert_array_equal(selected.columns, curves.columns)
    np.testing.assert_array_equal(curves.iloc[0], [2., 3., 4.])


def test_pitch_rf_selection_reports_missing_excluded_and_unmeasured_units():
    def keys(units):
        return pd.MultiIndex.from_tuples(
            [("m", "d", "A", unit) for unit in units], names=["mouse", "date", "probe", "unit_id"],
        )
    pitch = pd.DataFrame([[2., 1.], [2., 1.], [0., 0.], [2., 1.]],
                         index=keys([1, 2, 3, 4]), columns=[-45., 45.])
    rf = pd.DataFrame([[3., 1.], [3., 1.], [3., 1.], [3., 1.]],
                      index=keys([2, 3, 4, 5]), columns=[-20., 20.])
    native_peaks = pd.DataFrame({"rf_vertical_profile_peak_deg": [-20.] * 4,
                                "rf_2d_max_elevation_deg": [20.] * 4}, index=rf.index)
    picked_pitch, native_rf, peaks, audit = select_pitch_rf_pair(
        pitch, rf, native_peaks, rf_unit_ids=[2, 3, 5],
    )
    assert peaks.index.get_level_values("unit_id").to_list() == [2]
    assert picked_pitch.index.equals(native_rf.index)
    assert audit.loc[("m", "d", "A", 1), "missing_rf"]
    assert audit.loc[("m", "d", "A", 3), "no_positive_pitch_peak"]
    assert audit.loc[("m", "d", "A", 4), "outside_rf_selection"]
    assert audit.loc[("m", "d", "A", 5), "missing_pitch"]
    assert len(pitch) == len(rf) == 4


def test_pitch_plots_use_supplied_statistics_and_full_unit_labels(monkeypatch):
    from matplotlib import pyplot as plt
    from Utils import pitch_direction_comparison as pitch_module

    def unexpected(*args, **kwargs):
        raise AssertionError("plotting must not recompute statistics")
    monkeypatch.setattr(pitch_module, "circular_linear_association", unexpected)
    monkeypatch.setattr(pitch_module, "pitch_rf_regression", unexpected)
    keys = pd.MultiIndex.from_tuples([("m14", "260609", "A", 7), ("m14", "260609", "B", 7)])
    peaks = pd.DataFrame({"hd_peak_deg": [30., 90.], "pitch_peak_deg": [-45., 45.],
                          "rf_peak_deg": [-20., 20.]}, index=keys)
    circular = pitch_module.plot_peak_comparison(peaks, {"r": .5, "p": .2, "n": 2})
    assert "R=0.500" in circular.axes[0].get_title()
    regression = pitch_module.plot_pitch_rf_regression(
        peaks, {"n": 2, "slope": 1., "intercept": 0., "r_squared": .8, "p": .1},
    )
    assert "R²=0.800" in regression.axes[0].get_title()
    pitch = pd.DataFrame([[1., 2.], [3., 4.]], index=keys, columns=[-45., 45.])
    rf = pd.DataFrame([[2., 1.], [4., 3.]], index=keys, columns=[-20., 20.])
    order = keys[::-1].to_list()
    figures = pitch_module.plot_pitch_rf_pair(pitch, rf, order=order)
    assert [tick.get_text() for tick in figures[0].axes[0].get_yticklabels()] == [
        "m14:260609:B:7", "m14:260609:A:7",
    ]
    np.testing.assert_array_equal(figures[0].axes[0].images[0].get_array(), [[3., 4.], [1., 2.]])
    for figure in [circular, regression, *figures]:
        plt.close(figure)
