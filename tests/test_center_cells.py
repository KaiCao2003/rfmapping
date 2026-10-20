"""Scientific contracts for center-cell geometry, sampling and shuffled nulls."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import linregress

from EBB.center_cells import (
    CenterCellConfig, _smooth_angular_counts, analyze_center_cells, classify_center_cohort,
    assign_spikes_to_behavior, circular_shift_spikes, prepare_center_behavior,
    select_moving_spikes, behavior_intervals, analyze_center_theta,
)
from Utils.theta_analysis import extract_theta_phase


def config(**changes):
    return CenterCellConfig(**{"position_smoothing_s": 0., **changes})


def ring_behavior(*, heading_constant=False, duration=60., **changes):
    times = np.arange(0., duration + .025, .05)
    angle = np.deg2rad(times * 29.)
    radius = 8 + 6 * np.sin(times * 1.37)
    xy = 20 + radius[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    heading = np.zeros(len(times)) if heading_constant else times * 51 % 360
    cfg = config(angle_bin_deg=90., angle_smoothing_deg=90.,
                 min_spatial_coverage_exclusive=0., min_spikes_exclusive=5,
                 **changes)
    return prepare_center_behavior(times, xy, heading, 40., config=cfg)


def test_center_geometry_preserves_native_clockwise_angles_and_raw_heading():
    xy = np.array([[20., 30.], [10., 20.], [20., 10.], [30., 20.], [20., 20.]])
    heading = np.array([0., 360., -360., 0., 0.])
    result = prepare_center_behavior(np.arange(5.), xy, heading, [40., 40.], config=config())
    np.testing.assert_allclose(result["cb_deg"][:4], [0., 90., 180., 270.])
    assert np.isnan(result["cb_deg"][-1])
    np.testing.assert_allclose(result["cd_cm"], [10., 10., 10., 10., 0.])
    np.testing.assert_array_equal(result["raw_hd_cw_deg"], heading)
    np.testing.assert_array_equal(result["xy_cm_y_down"], xy)
    np.testing.assert_array_equal(result["hd_deg"], np.zeros(5))


def test_tracking_gaps_neither_collect_spikes_nor_inflate_occupancy():
    times = np.array([0., .1, .2, 10., 10.1, 10.2])
    xy = np.array([[1., 1.], [2., 1.], [3., 1.], [20., 20.], [21., 20.], [22., 20.]])
    behavior = prepare_center_behavior(times, xy, np.zeros(6), 40., config=config())
    np.testing.assert_allclose(behavior["speed_cm_s"], 10.)
    assert behavior["exposure_s"].sum() == pytest.approx(.5)
    spikes = np.array([-.01, .05, .1, 5., 10.1, 10.21])
    np.testing.assert_array_equal(assign_spikes_to_behavior(behavior, spikes), [-1, 0, 1, -1, 4, -1])
    np.testing.assert_allclose(select_moving_spikes(behavior, spikes), [.05, .1, 10.1])
    result = analyze_center_cells(behavior, {17: spikes}, n_shuffles=0)
    assert result["tuning"][17]["hd"]["occupancy_s"].sum() == pytest.approx(.5)
    assert result["summary"].loc[17, "n_speed_spikes"] == 3


def test_position_smoothing_stops_at_excluded_tracking_and_does_not_create_jump_speed():
    times = np.arange(8) * .1
    xy = np.c_[[1., 2., 3., 50., 75., 100., 101., 102.], np.ones(8)]
    valid = np.array([True, True, True, False, False, True, True, True])
    behavior = prepare_center_behavior(times, xy, np.zeros(8), [200., 20.],
                                       valid_mask=valid, config=CenterCellConfig(position_smoothing_s=.2))
    assert np.isnan(behavior["smoothed_xy_cm_y_down"][3:5]).all()
    assert behavior["smoothed_xy_cm_y_down"][2, 0] < 4
    assert behavior["smoothed_xy_cm_y_down"][5, 0] > 99
    assert np.nanmax(behavior["speed_cm_s"]) < 11
    np.testing.assert_array_equal(assign_spikes_to_behavior(behavior, times[3:5]), [-1, -1])


def test_spatial_boundary_does_not_silently_clip_invalid_positions():
    behavior = prepare_center_behavior(np.arange(4.), [[-1, 1], [0, 1], [1, 1], [41, 1]],
                                       [0, 0, 0, 0], 40., config=config())
    np.testing.assert_array_equal(behavior["valid_position_mask"], [False, True, True, False])
    assert np.isnan(behavior["cd_cm"][[0, 3]]).all()


def test_circle_coverage_excludes_square_corners_and_uses_radius_for_distance():
    centers = np.arange(1.25, 10., 2.5)
    x, y = np.meshgrid(centers, centers, indexing="ij")
    xy = np.column_stack((x.ravel(), y.ravel()))
    inside = np.linalg.norm(xy - 5, axis=1) <= 5
    behavior = prepare_center_behavior(
        np.arange(len(xy)) * .1, xy, np.zeros(len(xy)), 10.,
        arena_shape="circle", config=config(speed_threshold_cm_s=0.),
    )
    np.testing.assert_array_equal(behavior["valid_position_mask"], inside)
    assert behavior["spatial_bin_mask"].sum() == 12
    assert behavior["spatial_coverage"] == 1.
    assert behavior["max_center_distance_cm"] == 5.
    np.testing.assert_array_equal(
        assign_spikes_to_behavior(behavior, behavior["times_s"][~inside]),
        np.full(4, -1),
    )
    result = analyze_center_cells(behavior, {1: behavior["times_s"]}, n_shuffles=0)
    assert result["tuning"][1]["cd"]["edges"][-1] == 6.
    assert result["tuning"][1]["cd"]["spike_counts"].sum() == inside.sum()


def test_angular_author_smoothing_pools_counts_and_occupancy_and_retains_missing():
    counts = np.array([1., 9., 0., 0.])
    occupancy = np.array([1., 3., 1., 1.])
    rates, _, _ = _smooth_angular_counts(counts, occupancy, 3)
    assert rates[0] == pytest.approx(10 / 5)
    assert rates[0] != pytest.approx((0 + 1 + 3) / 3)
    occupancy[2] = 0.
    rates, _, _ = _smooth_angular_counts(counts, occupancy, 3)
    assert np.isnan(rates[2])
    assert np.isfinite(rates[[0, 1, 3]]).all()


def test_zero_spike_units_and_unvisited_bins_remain_distinguishable():
    times = np.arange(20.) * .1
    xy = np.c_[np.linspace(21., 29., len(times)), np.full(len(times), 20.)]
    behavior = prepare_center_behavior(times, xy, np.zeros(len(times)), 40., config=config())
    result = analyze_center_cells(behavior, {"silent": np.array([])}, n_shuffles=0)
    assert list(result["summary"].index) == ["silent"]
    row = result["summary"].loc["silent"]
    assert row["n_speed_spikes"] == 0
    assert pd.isna(row["cb_is_cell"])
    assert "insufficient_spikes" in row["cb_status"]
    for feature in ("hd", "cb", "cd"):
        curve = result["tuning"]["silent"][feature]
        assert np.all(curve["spike_counts"] == 0)
        assert np.all(curve["rate_hz"][curve["occupancy_s"] > 0] == 0)
        assert np.isnan(curve["rate_hz"][curve["occupancy_s"] == 0]).all()
        assert np.isnan(curve["smoothed_rate_hz"][curve["occupancy_s"] == 0]).all()


def test_distance_first_bin_and_zero_degree_bin_are_not_omitted():
    distance = np.array([.1, .5, .9, 1.3, 2.1, 2.5, 3.1, 3.5])
    times = np.arange(len(distance)) * .1
    xy = np.c_[20 + distance, np.full(len(times), 20.)]
    behavior = prepare_center_behavior(times, xy, np.full(len(times), 360.), 40., config=config())
    result = analyze_center_cells(behavior, {1: times}, n_shuffles=0)
    cd, hd = result["tuning"][1]["cd"], result["tuning"][1]["hd"]
    np.testing.assert_array_equal(cd["spike_counts"][:2], [4, 4])
    assert cd["spike_counts"].sum() == 8
    assert hd["spike_counts"][0] == 8
    assert len(hd["centers"]) == 120


def test_split_halves_use_exact_spike_times_and_split_frame_exposure():
    times = np.arange(5) * .25
    xy = np.c_[10 + times * 4, np.full(5, 10.)]
    behavior = prepare_center_behavior(times, xy, np.zeros(5), 40., config=config())
    spikes = np.array([.49, .51])
    np.testing.assert_array_equal(assign_spikes_to_behavior(behavior, spikes), [2, 2])
    result = analyze_center_cells(behavior, {1: spikes}, n_shuffles=0)
    hd = result["tuning"][1]["hd"]
    np.testing.assert_array_equal(hd["half_spike_counts"].sum(axis=1), [1, 1])
    np.testing.assert_allclose(hd["half_occupancy_s"].sum(axis=1), [.5, .5])
    np.testing.assert_allclose(hd["half_occupancy_s"].sum(axis=0), hd["occupancy_s"])


def test_circular_shift_is_continuous_clock_time_and_reselects_gap_support():
    times = np.array([0., .1, .2, .3, 10., 10.1, 10.2])
    xy = np.c_[[1., 2., 3., 4., 20., 21., 22.], np.ones(7)]
    behavior = prepare_center_behavior(times, xy, np.zeros(7), 40., config=config())
    spikes = np.array([.11, .15, .20])
    shifted = circular_shift_spikes(spikes, [0., 10.2], 2.037)
    np.testing.assert_allclose(shifted, spikes + 2.037)
    assert len(select_moving_spikes(behavior, spikes)) == 3
    assert len(select_moving_spikes(behavior, shifted)) == 0
    np.testing.assert_allclose(circular_shift_spikes([10.1], [0., 10.2], .3), [.2])


def test_cd_regression_uses_unsmoothed_visited_rates():
    distances = np.tile([1., 3., 5., 7.], 10)
    times = np.arange(len(distances)) * .1
    xy = np.c_[20 + distances, np.full(len(times), 20.)]
    behavior = prepare_center_behavior(times, xy, np.zeros(len(times)), 40., config=config())
    spikes = np.repeat(times, np.tile([0, 1, 4, 9], 10))
    result = analyze_center_cells(behavior, {1: spikes}, n_shuffles=0)
    curve = result["tuning"][1]["cd"]
    visited = curve["occupancy_s"] > 0
    reference = linregress(curve["centers"][visited], curve["rate_hz"][visited])
    smoothed_reference = linregress(curve["centers"][visited], curve["smoothed_rate_hz"][visited])
    row = result["summary"].loc[1]
    assert row["cd_slope_hz_per_cm"] == pytest.approx(reference.slope)
    assert row["cd_r2"] == pytest.approx(reference.rvalue ** 2)
    assert row["cd_r2"] != pytest.approx(smoothed_reference.rvalue ** 2)


def test_pooled_nulls_are_reproducible_per_statistic_and_exclude_ineligible_units():
    behavior = ring_behavior()
    times = behavior["times_s"]
    spikes = {"a": times[::3], "b": times[1::5] + .012, "silent": np.array([])}
    first = analyze_center_cells(behavior, spikes, n_shuffles=7, seed=17)
    second = analyze_center_cells(behavior, spikes, n_shuffles=7, seed=17)
    np.testing.assert_array_equal(first["shuffle_offsets_s"], second["shuffle_offsets_s"])
    assert (first["shuffle_offsets_s"][:2] >= 20).all()
    assert (first["shuffle_offsets_s"][:2] <= 40).all()
    assert np.isnan(first["shuffle_offsets_s"][2]).all()
    assert first["metadata"]["null_scope"] == "session_local_pooled_eligible_units"
    assert first["metadata"]["pooled_eligible_unit_ids"] == ["a", "b"]
    for statistic, values in first["nulls"].items():
        assert values.shape == (3, 7)
        assert np.isnan(values[2]).all()
        np.testing.assert_array_equal(values, second["nulls"][statistic])
        finite = values[np.isfinite(values)]
        if len(finite):
            assert first["thresholds"][statistic] == pytest.approx(np.percentile(finite, 99))
    for feature in ("hd", "cb", "cd"):
        assert pd.isna(first["summary"].loc["silent", f"{feature}_is_cell"])


def test_feature_specific_coverage_does_not_conflate_hd_and_center_bearing():
    behavior = ring_behavior(heading_constant=True)
    result = analyze_center_cells(behavior, {1: behavior["times_s"][::4]}, n_shuffles=3)
    row = result["summary"].loc[1]
    assert not row["hd_coverage_pass"]
    assert row["cb_coverage_pass"]
    assert "incomplete_angular_coverage" in row["hd_status"]
    assert "incomplete_angular_coverage" not in row["cb_status"]
    assert pd.isna(row["hd_is_cell"])
    assert np.isnan(result["nulls"]["hd_mvl"]).all()
    assert np.isfinite(result["nulls"]["cb_mvl"]).all()


def test_more_than_100_spikes_and_more_than_80_percent_coverage_are_strict():
    behavior = ring_behavior()
    behavior["config"]["min_spikes_exclusive"] = 100
    behavior["config"]["min_spatial_coverage_exclusive"] = .8
    behavior["spatial_coverage"] = .8
    times = select_moving_spikes(behavior, behavior["times_s"])
    units = {100: times[:100], 101: times[:101]}
    result = analyze_center_cells(behavior, units, n_shuffles=0)
    assert "insufficient_spikes" in result["summary"].loc[100, "cd_status"]
    assert "insufficient_spikes" not in result["summary"].loc[101, "cd_status"]
    assert "insufficient_spatial_coverage" in result["summary"].loc[101, "cd_status"]
    behavior["spatial_coverage"] = .81
    result = analyze_center_cells(behavior, units, n_shuffles=0)
    assert "insufficient_spatial_coverage" not in result["summary"].loc[101, "cd_status"]


def test_short_trial_reports_unassessable_not_negative_classification():
    behavior = ring_behavior(duration=40.)
    result = analyze_center_cells(behavior, {1: behavior["times_s"]}, n_shuffles=3)
    row = result["summary"].loc[1]
    assert "insufficient_trial_duration" in row["cb_status"]
    assert pd.isna(row["cb_is_cell"])
    assert np.isnan(result["shuffle_offsets_s"]).all()


@pytest.mark.parametrize("times", [[0., 0.], [1., 0.], [0., np.nan]])
def test_rejects_ambiguous_behavior_clock(times):
    with pytest.raises(ValueError, match="strictly increasing"):
        prepare_center_behavior(times, [[1, 1], [2, 2]], [0, 0], 40.)


def test_missing_heading_does_not_destroy_center_distance_support():
    times = np.arange(8) * .1
    behavior = prepare_center_behavior(times, np.c_[21 + np.arange(8), np.full(8, 20)],
                                       np.full(8, np.nan), 40., config=config())
    result = analyze_center_cells(behavior, {1: times}, n_shuffles=0)
    assert result["tuning"][1]["cd"]["spike_counts"].sum() == 8
    assert result["tuning"][1]["hd"]["spike_counts"].sum() == 0
    assert result["tuning"][1]["cb"]["spike_counts"].sum() == 0


def test_cohort_reclassification_pools_nulls_without_mutating_sessions_or_merging_unit_ids():
    behavior = ring_behavior()
    times = behavior["times_s"]
    first = analyze_center_cells(behavior, {12: times[::3]}, n_shuffles=4, seed=1)
    second = analyze_center_cells(behavior, {12: times[1::5], 27: []}, n_shuffles=4, seed=2)
    original_first = first["summary"].copy(deep=True)
    original_second = second["summary"].copy(deep=True)
    cohort = classify_center_cohort({"mouse1/session1": first, "mouse1/session2": second})
    pd.testing.assert_frame_equal(first["summary"], original_first)
    pd.testing.assert_frame_equal(second["summary"], original_second)
    assert len(cohort["summary"]) == 3
    assert ("mouse1/session1", 12) in cohort["summary"].index
    assert ("mouse1/session2", 12) in cohort["summary"].index
    assert cohort["metadata"]["null_scope"] == "cohort_pooled_eligible_session_units"
    assert "not assumed independent" in cohort["metadata"]["interpretation"]
    for stat in first["nulls"]:
        pooled = np.r_[first["nulls"][stat].ravel(), second["nulls"][stat].ravel()]
        finite = pooled[np.isfinite(pooled)]
        if len(finite):
            threshold = np.percentile(finite, 99)
            assert cohort["thresholds"][stat] == pytest.approx(threshold)
            assert cohort["results_by_session"]["mouse1/session1"]["thresholds"][stat] == pytest.approx(threshold)
    assert pd.isna(cohort["summary"].loc[("mouse1/session2", 27), "cb_is_cell"])


def test_cohort_rejects_incompatible_analysis_parameters():
    behavior = ring_behavior()
    result = analyze_center_cells(behavior, {1: behavior["times_s"][::3]}, n_shuffles=0)
    other = {**result, "config": {**result["config"], "distance_bin_cm": 4.}}
    with pytest.raises(ValueError, match="incompatible nulls"):
        classify_center_cohort({"first": result, "second": other})


def test_behavior_intervals_preserve_selected_exposure_and_real_gaps():
    times = np.array([0., .1, .2, 10., 10.1, 10.2])
    xy = np.c_[[1., 2., 3., 20., 21., 22.], np.ones(6)]
    behavior = prepare_center_behavior(times, xy, np.zeros(6), 40., config=config())
    selected = behavior["moving_mask"].copy()
    selected[1] = False
    intervals = behavior_intervals(behavior, selected)
    np.testing.assert_allclose(intervals, [[0., .05], [.15, .25], [9.95, 10.2]])
    assert np.diff(intervals, axis=1).sum() == pytest.approx(behavior["exposure_s"][selected].sum())
    assert behavior_intervals(behavior, np.zeros(6, dtype=bool)).shape == (0, 2)


def test_center_theta_keeps_spike_coverage_and_phase_support_gates_distinct():
    behavior = ring_behavior()
    behavior["config"].update(min_spikes_exclusive=100, min_spatial_coverage_exclusive=.8)
    behavior["spatial_coverage"] = .81
    times = np.arange(0., 60., .002)
    theta = extract_theta_phase(times, np.cos(2 * np.pi * 8 * times), sample_rate_hz=500.)
    spikes = np.arange(1.5, 58., .125)
    result = analyze_center_theta(
        behavior, {"locked": spikes, "sparse": spikes[:100], "edge": np.full(101, .01)}, theta,
    )
    table = result["summary"]
    assert table.loc["locked", "is_phase_locked"]
    assert table.loc["locked", "mean_vector_length"] > .99
    assert pd.isna(table.loc["sparse", "is_phase_locked"])
    assert table.loc["sparse", "theta_status"] == "insufficient_spikes"
    assert "sparse" not in result["units"]
    assert pd.isna(table.loc["edge", "is_phase_locked"])
    assert table.loc["edge", "phase_locking_status"] == "insufficient_phase_spikes"

    behavior["spatial_coverage"] = .8
    excluded = analyze_center_theta(behavior, {"locked": spikes}, theta)["summary"].loc["locked"]
    assert pd.isna(excluded["is_phase_locked"])
    assert pd.isna(excluded["is_theta_rhythmic"])
    assert pd.isna(excluded["precession_candidate"])
    assert excluded["mean_vector_length"] == pytest.approx(table.loc["locked", "mean_vector_length"])
