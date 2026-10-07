"""Physical angular ranges must survive loading, display and explicit pooling."""

import json

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison


def _table(angles, values, *, mouse="m14", unit=7, angular_range=None):
    source = comparison.profile_table([values], [unit], angles, probe="A")
    return comparison._prepare_profiles(source, mouse=mouse, date="260609", range=angular_range)


def test_tc_range_returns_fresh_legacy_tick_lists():
    assert comparison.tcRange(True) == [-180, -90, 0, 90, 180]
    assert comparison.tcRange(False) == [180, 270, 0, 90, 180]
    ticks = comparison.tcRange(True)
    assert isinstance(ticks, list)
    ticks[0] = 999
    assert comparison.tcRange(True) == [-180, -90, 0, 90, 180]


def test_m14_loader_preserves_native_thirty_ten_degree_bins(tmp_path):
    angles = np.arange(-145., 150., 10.)
    values = np.arange(1., 31.)
    counts = values.reshape(1, 1, 30, 1)
    path = tmp_path / "rf.json"
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7], "xPositions": angles.tolist(), "yPositions": [0.],
        "timeBinEdges": [0., .2], "occupancyTimeSec": [np.ones(30).tolist()],
    }))

    result = comparison.load_rf(path, mouse="m14", date="260609", range=[-150, 150])

    np.testing.assert_array_equal(result.columns, angles)
    np.testing.assert_array_equal(result.iloc[0], values)
    assert result.attrs["range"] == [-150, 150]
    assert comparison.peak_angles(result, result.index)[0] == 145.


def test_rf_loader_preserves_missing_values_and_records_only_measured_zero_bins(tmp_path):
    counts = np.ones((3, 1, 30, 1))
    counts[0, 0, :2, 0] = np.nan
    counts[1, 0, :3, 0] = np.nan
    counts[2, 0, :2, 0] = 0
    payload = counts.tolist()
    payload[0][0][0][0] = None
    path = tmp_path / "rf.json"
    path.write_text(json.dumps({
        "unitsSpikeCounts": payload, "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7, 11, 13], "xPositions": np.arange(-145., 150., 10.).tolist(),
        "yPositions": [0.], "timeBinEdges": [0., .2],
    }))
    profiles = comparison.load_rf(path, mouse="m14", date="260609", range=[-150, 150])
    np.testing.assert_array_equal(profiles.isna().sum(axis=1), [2, 3, 0])
    assert [info["zero_bins"] for info in profiles.attrs["unit_info"].values()] == [0, 0, 2]
    assert comparison.rf_pick(profiles, max_zero_bins=0).index.get_level_values("unit_id").tolist() == [7, 11]
    assert len(comparison.rf_pick(profiles, max_zero_bins=None)) == 3

    selected = comparison.rf_pick(profiles, max_zero_bins=0)
    pooled = comparison.resample_profiles(selected, range=comparison.tcRange(True), fill_value=0.)
    assert pooled.loc[profiles.index[0]].isna().any()
    assert pooled.iloc[:, np.abs(pooled.columns) > 150].eq(0).all().all()
    # Select on the native grid before interpolation introduces padding zeros.
    assert pooled.index.get_level_values("unit_id").tolist() == [7, 11]
    assert comparison.rf_pick(pooled, max_zero_bins=0).empty


def test_rf_loader_preserves_native_nonstandard_grid_and_single_time_bin(tmp_path, monkeypatch):
    angles = [15., 105., 195., 285.]
    values = np.array([1., 7., 3., 2.])
    counts = values.reshape(1, 1, 4, 1)
    path = tmp_path / "rf.json"
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7], "xPositions": angles, "yPositions": [0.],
        "timeBinEdges": [.4, .7],
    }))

    def unexpected_transform(*args, **kwargs):
        raise AssertionError("RF loading must not run analysis or transformations")

    for name in ("smooth_profiles", "rf_pick", "load_detected_rf", "resample_profiles"):
        monkeypatch.setattr(comparison, name, unexpected_transform)
    result = comparison.load_rf(path, mouse="m19", date="260827", range=[0, 360])

    assert result.index.tolist() == [("m19", "260827", "A", 7)]
    np.testing.assert_array_equal(result.columns, angles)
    np.testing.assert_array_equal(result.iloc[0], values)
    assert result.attrs["label"] == "RF"


@pytest.mark.parametrize("shape", [(1, 2, 4, 1), (1, 1, 4, 2)])
def test_rf_loader_rejects_sources_requiring_spatial_or_time_aggregation(tmp_path, shape):
    counts = np.ones(shape)
    path = tmp_path / "rf.json"
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(shape),
        "unitPool": [7], "xPositions": [15., 105., 195., 285.],
        "yPositions": list(np.arange(shape[1], dtype=float)),
        "timeBinEdges": np.linspace(0., .2, shape[3] + 1).tolist(),
    }))

    with pytest.raises(ValueError, match="sum_to_1d|time bin"):
        comparison.load_rf(path, mouse="m19", date="260827")


def test_explicit_resampling_wraps_angles_and_preserves_partial_support():
    index = pd.MultiIndex.from_tuples([("A", 7)], names=["probe", "unit_id"])
    profiles = pd.DataFrame([[1., 5., 2.]], index=index, columns=[30., 90., 150.])
    profiles.attrs = {"range": [0, 180], "unit_info": {("A", 7): {"zero_bins": 0}}}
    original = profiles.copy(deep=True)

    result = comparison.resample_profiles(profiles, range=[0, 360], bins=6)

    np.testing.assert_array_equal(result.columns, [30., 90., 150., -150., -90., -30.])
    np.testing.assert_array_equal(result.iloc[0, :3], [1., 5., 2.])
    assert result.iloc[0, 3:].isna().all()
    assert result.attrs["range"] == [0, 360]
    pd.testing.assert_frame_equal(profiles, original)
    assert profiles.attrs == original.attrs


def test_partial_field_padding_is_explicit_and_independent_of_response_kind():
    profiles = pd.DataFrame([[1., 5., 2.]], index=["unit"], columns=[30., 90., 150.])
    profiles.attrs = {"range": [0, 180], "response_kind": "RF"}

    missing = comparison.resample_profiles(profiles, range=[0, 360], bins=6)
    zero = comparison.resample_profiles(profiles, range=[0, 360], bins=6, fill_value=0.)
    assert missing.iloc[0, 3:].isna().all()
    np.testing.assert_array_equal(zero.iloc[0, 3:], 0.)
    np.testing.assert_array_equal(missing.iloc[0, :3], zero.iloc[0, :3])

    missing = comparison.align_profiles(profiles, profiles.index, [0.])
    zero = comparison.align_profiles(profiles, profiles.index, [0.], fill_value=0.)
    outside = (missing.columns > -180) & (missing.columns < 0)
    assert missing.iloc[0, outside].isna().all()
    np.testing.assert_array_equal(zero.iloc[0, outside], 0.)
    assert missing.loc["unit", -180.] == zero.loc["unit", -180.] == 2.


def test_comparison_padding_is_chosen_separately_for_each_panel():
    profiles = pd.DataFrame([[1., 5., 2.]], index=["unit"], columns=[30., 90., 150.])
    profiles.attrs["range"] = [0, 180]
    result = comparison.prepare_comparison(profiles, profiles, mode="aligned",
                                           reference_fill_value=0.)
    assert result["reference"].eq(0).any().any()
    assert not result["reference"].isna().any().any()
    assert result["matched"].isna().any().any()


def test_explicit_hd_preparation_preserves_source_order_for_tied_peaks(monkeypatch):
    angles = np.arange(6., 360., 12.)
    values = np.ones(30)
    values[[0, 15]] = 9.  # 6 and -174 degrees tie; source order chooses 6.
    source = comparison.profile_table([values], [7], angles, probe="A")
    monkeypatch.setattr(comparison, "load_hd_profiles", lambda *args, **kwargs: source)

    profiles = comparison.load_hd_profiles("unused.tc", probe="A")
    profiles = comparison.resample_profiles(profiles, range=comparison.tcRange(False))
    result = comparison.recording_profiles(profiles, mouse="m19", date="260827")

    np.testing.assert_array_equal(result.columns, source.columns)
    np.testing.assert_array_equal(result.iloc[0], values)
    assert result.attrs["range"] == comparison.tcRange(False)
    assert comparison.peak_angles(result, result.index)[0] == 6.


def test_display_conversion_preserves_values_and_requires_same_physical_extent():
    angles = np.arange(-174., 180., 12.)
    source = _table(angles, np.arange(30.), angular_range=comparison.tcRange(True))
    original = source.copy(deep=True)

    result = comparison.convert_profile_coordinates(source, range=comparison.tcRange(False))

    np.testing.assert_array_equal(result.columns, source.columns)
    np.testing.assert_array_equal(result, source)
    assert result.attrs["range"] == comparison.tcRange(False)
    pd.testing.assert_frame_equal(source, original)
    assert source.attrs["range"] == comparison.tcRange(True)
    with pytest.raises(ValueError):
        comparison.convert_profile_coordinates(source, range=[-150, 150])


def test_partial_heatmap_has_real_extent_and_all_thirty_bins():
    source = _table(np.arange(-145., 150., 10.), np.arange(30.), angular_range=[-150, 150])
    figure, axes = comparison.plot_profiles(source, list(source.index), "RF", show=False)
    try:
        np.testing.assert_allclose(axes.get_xlim(), [-150., 150.])
        np.testing.assert_allclose(axes.images[0].get_extent()[:2], [-150., 150.])
        np.testing.assert_array_equal(axes.images[0].get_array(), source.to_numpy())
        assert axes.images[0].get_array().shape == (1, 30)
    finally:
        plt.close(figure)


def test_partial_smoothing_does_not_join_opposite_field_edges():
    values = np.zeros(30)
    values[0] = 10.
    partial = _table(np.arange(-145., 150., 10.), values, angular_range=[-150, 150])
    full = _table(np.arange(-174., 180., 12.), values, angular_range=[-180, 180])
    assert comparison.smooth_profiles(partial, 1).iloc[0, -1] == 0.
    assert comparison.smooth_profiles(full, 1).iloc[0, -1] > 0.


def test_positive_full_circle_labels_keep_signed_angles_for_unwrapped_sums():
    angles = np.arange(6., 360., 12.)
    values = np.zeros(30)
    values[27] = 10.  # 330 degrees is -30 in the internal signed representation.
    source = _table(angles, values, angular_range=[0, 360])
    peak = comparison.peak_angles(source, source.index)[0]
    assert peak == -30.
    shifted = comparison.align_profiles(source, source.index, [138.], is_wrap=False)
    assert shifted.iloc[0].idxmax() == 108.


def test_partial_field_crossing_signed_seam_smooths_its_real_neighbors():
    angles = np.arange(5., 300., 10.)
    values = np.zeros(30)
    values[17] = 10.  # 175 and 185 degrees are adjacent measured bins.
    source = _table(angles, values, angular_range=[0, 300])
    smoothed = comparison.smooth_profiles(source, 1)
    assert smoothed.iloc[0].loc[-175.] > 0.
    assert smoothed.iloc[0].loc[5.] == 0.
    aligned = comparison.align_profiles(source, source.index, [0.], is_wrap=False)
    assert aligned.iloc[0].loc[168.] > 0.
    assert aligned.iloc[0].loc[-180.] > 0.
    assert np.isnan(aligned.iloc[0].loc[-24.])  # 336 degrees was never measured.


def test_combine_same_grid_reorders_by_angle_without_losing_values():
    angles = (np.arange(6., 360., 12.) + 180.) % 360. - 180.
    first = _table(angles, np.arange(30.), mouse="m14", angular_range=comparison.tcRange(True))
    second = _table(angles[::-1], np.arange(100., 130.), mouse="m15",
                    angular_range=comparison.tcRange(True))

    result = comparison.combine(first, second)

    np.testing.assert_array_equal(result.columns, first.columns)
    np.testing.assert_array_equal(result.loc[first.index[0]], first.iloc[0])
    np.testing.assert_array_equal(result.loc[second.index[0]], second.reindex(columns=first.columns).iloc[0])
    assert not result.isna().any().any()


def test_mixed_grids_require_separate_resampling_and_keep_finite_rows_and_gaps():
    full = _table(np.arange(-174., 180., 12.), np.ones(30), mouse="m20", angular_range=[-180, 180])
    partial_values = np.linspace(1., 2., 30)
    partial_values[15] = np.nan
    partial = _table(np.arange(-145., 150., 10.), partial_values,
                     mouse="m14", angular_range=[-150, 150])
    with pytest.raises(ValueError, match="resample_profiles explicitly"):
        comparison.combine(full, partial)

    prepared = [comparison.resample_profiles(table, range=[-180, 180])
                for table in (full, partial)]
    result = comparison.combine(*prepared)

    assert result.shape == (2, 30)
    assert set(result.index) == set(full.index) | set(partial.index)
    np.testing.assert_array_equal(result.loc[full.index[0]], np.ones(30))
    row = result.loc[partial.index[0]]
    assert np.isfinite(row).sum() >= 20
    assert row.loc[abs(np.asarray(row.index, dtype=float)) > 150].isna().all()
    assert np.isnan(row.loc[6.])  # Missing native 5° bin must not be bridged.
    assert partial.isna().sum().iloc[15] == 1  # Pooling cannot mutate source gaps.


@pytest.mark.parametrize("wrapped", [True, False])
def test_shifted_partial_domain_retains_unobserved_gap(wrapped):
    source = _table(np.arange(-145., 150., 10.), np.ones(30), angular_range=[-150, 150])
    result = comparison.align_profiles(source, source.index, [84.], is_wrap=wrapped)
    angles = np.asarray(result.columns, dtype=float)
    original_angle = angles - 84.
    if wrapped:
        original_angle = (original_angle + 180.) % 360. - 180.
        assert result.shape[1] == 30
        assert np.isfinite(result.iloc[0].loc[-180.])  # Shifted support crosses the ±180 seam.
    else:
        assert result.shape[1] == 61
    assert np.isnan(result.to_numpy()[0, abs(original_angle) > 150.]).all()
    np.testing.assert_allclose(result.to_numpy()[0, abs(original_angle) < 140.], 1.)


@pytest.mark.parametrize("angular_range", [[0], [0, 0], [-180, 181],
                                           [0, 90, 45, 180], [0, 90, 180, 270, 360, 450],
                                           [0, float("nan")]])
def test_invalid_angular_range_is_rejected(angular_range):
    with pytest.raises(ValueError):
        _table(np.arange(-174., 180., 12.), np.ones(30), angular_range=angular_range)


@pytest.mark.parametrize("angular_range", [[-150, 150], [-180, -90, 0, 90, 180],
                                           [180, 270, 0, 90, 180], [0, 360]])
def test_ascending_or_wrapped_ranges_up_to_one_turn_are_supported(angular_range):
    result = _table(np.arange(-174., 180., 12.), np.ones(30), angular_range=angular_range)
    assert result.shape == (1, 30)
    assert result.attrs["range"] == angular_range
