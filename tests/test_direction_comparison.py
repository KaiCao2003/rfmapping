import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from Utils.direction_comparison import (
    gen_overlap_list, profile_table, peak_angles, sorted_by_peak, align_profiles,
    plot_profiles, plot_comparison_heatmaps, compare_both_orders,
    ego_coordinates, allo_coordinates, profile_coordinates, paired_peak_angles,
    load_hd_profiles, load_ebc_profiles, peak_direction_sums, plot_peak_direction_sums,
    TuningCurveCollection, convert_profile_coordinates, smooth_profiles,
)
from Utils.tuning_curve_utils import save_hd_unit_lists
from Utils.statistic_utils import compare_direction_angles
from spatial_cell_analysis import save_egocentric_rfmap


def test_profile_table_wraps_source_angles_without_reversing_sign_or_values():
    source_angles = np.array([0, 90, 180, 270, 246, 360, -90])
    values = np.array([[1, 2, np.nan, 4, 5, 6, 7]])
    table = profile_table(values, [7], source_angles, probe="B")
    np.testing.assert_array_equal(table.columns, [0, 90, -180, -90, -114, 0, -90])
    np.testing.assert_array_equal(table, values)
    np.testing.assert_array_equal(source_angles, [0, 90, 180, 270, 246, 360, -90])
    assert table.index.tolist() == [("B", 7)]


def test_profile_loaders_keep_native_angles_and_response_order(tmp_path):
    path = tmp_path / "tuning_curves.tc"
    path.write_text(json.dumps({
        "unit_id": [7, 9], "unit_data": {
            "hd_class": [2, 1], "rayleigh_p": [.001, .001], "shuffle_p": [.001, .2],
        },
        "spike_counts": [[1, 2, 3, 4], [4, 3, 2, 1]],
        "occupancy_time_s": [1, 1, 1, 1],
    }))
    hd = load_hd_profiles(path, bins=4, hd_class=2)
    np.testing.assert_array_equal(hd.columns, [45, 135, -135, -45])
    np.testing.assert_allclose(hd.loc[("A", 7)], [1, 2, 3, 4])
    assert list(hd.index) == [("A", 7)]

    path = tmp_path / "egocentric_rate_map.rfmap"
    save_egocentric_rfmap(
        path, np.array([[[1], [2], [3], [4]]]),
        {"distance_edges": np.array([0, 8]), "theta_edges": np.arange(0, 361, 90)},
        [7], [0, 1],
    )
    ebc = load_ebc_profiles(path)
    np.testing.assert_array_equal(ebc.columns, hd.columns)
    np.testing.assert_allclose(ebc, hd)


def test_overlap_preserves_probe_identity_and_peak_ties():
    assert gen_overlap_list([4, 2, 2, 1], np.array([2, 3, 4])) == [2, 4]
    assert gen_overlap_list([("A", 2), ("B", 2)], [("B", 2)]) == [("B", 2)]
    profiles = profile_table([[2, 2, 0, 0], [2, 2, 0, 0], [0, 0, 3, 0]],
                             [9, 2, 5], [45, 135, 225, 315])
    np.testing.assert_array_equal(peak_angles(profiles, [9, 2, 5]), [45, 45, -135])
    assert sorted_by_peak(profiles, [9, 5, 2]) == [5, 2, 9]


@pytest.fixture(params=["unit_id", "recording"])
def partial_profiles(request):
    angles = np.arange(-174., 180., 12.)
    keys = ([7, 9, 11, 13] if request.param == "unit_id" else
            [("m19", "260918", "A", 7), ("m19", "260918", "A", 9),
             ("m19", "260918", "B", 7), ("m20", "260918", "A", 7)])
    shared, _, matched_only, later = keys
    profiles = {
        "HD": profile_table(2 + np.cos(np.deg2rad(angles[None, :] - np.array([54, -30])[:, None])),
                            [7, 9], angles),
        "RF": profile_table(2 + np.cos(np.deg2rad(angles[None, :] - np.array([66, -54, 102])[:, None])),
                            [7, 11, 13], angles),
    }
    if request.param == "recording":
        profiles["HD"].index = pd.MultiIndex.from_tuples(keys[:2], names=["mouse", "date", "probe", "unit_id"])
        profiles["RF"].index = pd.MultiIndex.from_tuples([shared, matched_only, later],
                                                       names=["mouse", "date", "probe", "unit_id"])
    return profiles, shared


def test_peak_angles_returns_nan_for_missing_or_all_missing_rows():
    table = profile_table([[0, 2, 1], [np.nan, np.nan, np.nan]], [7, 9], [-120, 0, 120])
    np.testing.assert_array_equal(peak_angles(table, [9, 7, 11]), [np.nan, 0, np.nan])
    assert peak_angles(table, []).shape == (0,)
    np.testing.assert_array_equal(peak_angles(table.iloc[:0], [7]), [np.nan])


def test_explicit_overlap_preserves_recording_identity_and_pairing(partial_profiles):
    profiles, shared = partial_profiles
    units = profiles["HD"].index.intersection(profiles["RF"].index)
    for reference, matched in (("HD", "RF"), ("RF", "HD")):
        first, second = profiles[reference].loc[units], profiles[matched].loc[units]
        peaks = paired_peak_angles(first, second)
        assert peaks.index.tolist() == [shared]
        assert peaks.index.names == first.index.names
        assert peaks.loc[shared, "peak_sum_deg"] == 120
        result = plot_comparison_heatmaps({reference: first, matched: second},
                                         sorted_by=reference, apply_to=matched, show=False)
        pd.testing.assert_frame_equal(result["peaks"], peaks)
        for (figure, axis), table in zip(result["figures"], (first, second), strict=True):
            row = table.loc[shared].to_numpy()
            np.testing.assert_allclose(axis.images[0].get_array(), (row / row.max())[None, :])
            plt.close(figure)


@pytest.mark.parametrize("plot_name", ["plot_sort", "plot_align", "plot_sum"])
def test_pair_plots_accept_external_id_and_range_selection(plot_name):
    from Utils import direction_comparison

    reference = profile_table(np.eye(3), [7, 9, 13], [-120, 0, 120])
    matched = profile_table(np.eye(3), [7, 11, 13], [-120, 0, 120])
    overlap = reference.index.intersection(matched.index)
    for limits, expected in (((7, 11), [7]), ((7, 13), [7, 13]), ((100, 200), [])):
        units = overlap[(overlap >= limits[0]) & (overlap <= limits[1])]
        result = getattr(direction_comparison, plot_name)(
            reference.loc[units], matched.loc[units], show=False,
        )
        assert result["order"] == expected
        assert result["peaks"].index.tolist() == expected
        for figure, axis in result["figures"]:
            if expected:
                assert axis.images[0].get_array().shape[0] == len(expected)
            else:
                assert any(text.get_text() == "No units to plot." for text in axis.texts)
            plt.close(figure)


@pytest.mark.parametrize("function_name", ["paired_peak_angles", "plot_sort", "plot_align", "plot_sum"])
def test_pair_functions_reject_unmatched_and_duplicate_unit_keys(partial_profiles, function_name):
    from Utils import direction_comparison

    profiles, shared = partial_profiles
    function = getattr(direction_comparison, function_name)
    with pytest.raises(ValueError):
        function(profiles["HD"], profiles["RF"])
    first = profiles["HD"].loc[[shared]]
    second = profiles["RF"].loc[[shared]]
    with pytest.raises(ValueError):
        function(pd.concat([first, first]), second)


@pytest.mark.parametrize("function_name", ["paired_peak_angles", "plot_sort", "plot_align", "plot_sum"])
@pytest.mark.parametrize("order", [[7], [7, 9, 11], [7, 7], []])
def test_pair_order_must_be_a_complete_permutation(function_name, order):
    from Utils import direction_comparison

    table = profile_table(np.eye(2), [7, 9], [-90, 90])
    with pytest.raises(ValueError):
        getattr(direction_comparison, function_name)(table, table, order=order)


@pytest.mark.parametrize("function_name", ["paired_peak_angles", "plot_sort", "plot_align", "plot_sum"])
@pytest.mark.parametrize("removed", ["unit_mode", "unit_range", "units", "smoothing_bins"])
def test_pair_functions_reject_removed_selection_and_smoothing_options(function_name, removed):
    from Utils import direction_comparison

    table = profile_table([[1, 2]], [7], [-90, 90])
    with pytest.raises(TypeError):
        getattr(direction_comparison, function_name)(table, table, **{removed: None})


@pytest.mark.parametrize("align", [True, "sum"])
def test_alignment_retains_full_input_order_and_missing_rows(align, monkeypatch):
    angles = np.arange(-174., 180., 12.)
    reference = profile_table([[np.nan] * 30, 2 + np.cos(np.deg2rad(angles - 54))], [7, 9], angles)
    matched = profile_table([2 + np.cos(np.deg2rad(angles + 30)), [np.nan] * 30], [7, 9], angles)
    profiles = {"HD": reference, "RF": matched.iloc[::-1]}
    originals = {name: table.copy() for name, table in profiles.items()}
    monkeypatch.setattr(plt, "show", lambda: None)
    native = compare_both_orders(profiles, "HD", "RF")
    aligned = compare_both_orders(profiles, "HD", "RF", align=align, previous=native)
    for reference_name in ("HD", "RF"):
        result = aligned[reference_name]
        assert result["order"] == native[reference_name]["order"]
        pd.testing.assert_frame_equal(result["peaks"], native[reference_name]["peaks"])
        missing_reference = result["peaks"].reference_peak_deg.isna().to_numpy()
        missing_matched = result["peaks"].matched_peak_deg.isna().to_numpy()
        np.testing.assert_array_equal(result["offset_deg"][missing_reference], [0])
        np.testing.assert_array_equal(result["matched_offset_deg"][missing_reference], [0])
        assert result["peaks"].peak_sum_deg.isna().all()
        for (_, axis), missing in zip(result["figures"], (missing_reference, missing_matched), strict=True):
            assert np.isnan(axis.images[0].get_array().filled(np.nan)[missing]).all()
        for item in (native[reference_name], result):
            for figure, _ in item["figures"]:
                plt.close(figure)
    for name, table in profiles.items():
        pd.testing.assert_frame_equal(table, originals[name])


def test_no_overlap_requires_explicit_empty_inputs():
    reference = profile_table([[1, 2]], [7], [-90, 90])
    matched = profile_table([[2, 1]], [9], [-90, 90])
    with pytest.raises(ValueError):
        paired_peak_angles(reference, matched)
    units = reference.index.intersection(matched.index)
    first, second = reference.loc[units], matched.loc[units]
    assert paired_peak_angles(first, second).empty
    result = plot_comparison_heatmaps({"HD": first, "RF": second}, sorted_by="HD", apply_to="RF",
                                     align="sum", show=False)
    assert result["order"] == []
    assert result["peaks"].empty
    for figure, _ in result["figures"]:
        plt.close(figure)


def test_peak_sums_pair_by_probe_and_unit_and_wrap_both_directions():
    angles = [-170, -30, 30, 60, 150, 170]
    hd = profile_table(np.eye(6)[[4, 0, 1]], [7, 9, 11], angles, probe="A")
    rf = profile_table(np.eye(6)[[1, 3]], [9, 7], angles, probe="A")
    other_probe = profile_table([np.eye(6)[5]], [7], angles, probe="B")
    all_rf = pd.concat([rf, other_probe])
    units = hd.index.intersection(all_rf.index)
    hd, rf = hd.loc[units], all_rf.loc[units]
    sums = peak_direction_sums(hd, rf)
    assert list(sums.index) == [("A", 9), ("A", 7)]
    np.testing.assert_array_equal(sums, [[-170, -30, 160], [150, 60, -150]])
    reversed_sums = peak_direction_sums(hd, rf, order=[("A", 7), ("A", 9)])
    np.testing.assert_array_equal(reversed_sums, sums.iloc[::-1])
    # A tied source peak uses its first bin; +180 is stored as -180.
    tied = profile_table([[1, 1]], [7], [150, 170])
    matching = profile_table([[1, 0]], [7], [30, 60])
    np.testing.assert_array_equal(peak_direction_sums(tied, matching).iloc[0], [150, 30, -180])


def test_peak_sum_histogram_counts_units_and_shows_rayleigh_statistics():
    angles = [-120, 0, 120]
    hd = profile_table(np.eye(3), [7, 9, 11], angles)
    rf = profile_table(np.eye(3)[::-1], [7, 9, 11], angles)
    table = peak_direction_sums(hd, rf)
    with plt.style.context("dark_background"):
        figure = plot_peak_direction_sums(table)
    histogram, = figure.axes
    assert sum(bar.get_height() for bar in histogram.patches) == 3
    assert len(histogram.patches) == 30
    assert histogram.get_title() == "HD + RF"
    assert histogram.get_xlabel() == "Peak sum (°)"
    assert histogram.texts[0].get_text().startswith("n = 3 · R = 1.00\nRayleigh p = ")
    assert figure.get_facecolor() == (1, 1, 1, 1)
    assert all(axis.get_facecolor() == (1, 1, 1, 1) for axis in figure.axes)
    plt.close(figure)


@pytest.mark.parametrize("reference, matched", [("HD", "RF"), ("RF", "HD")])
def test_sum_heatmaps_match_histogram_peaks_and_keep_reference_centered(reference, matched):
    angles = np.arange(-174., 180., 12.)
    hd = profile_table(2 + np.cos(np.deg2rad(angles[None, :] - np.array([114, -18])[:, None])),
                       [267, 7], angles, probe="A")
    rf = profile_table(2 + np.cos(np.deg2rad(angles[None, :] - np.array([174, 66])[:, None])),
                       [7, 267], angles, probe="A")
    profiles = {"HD": hd, "RF": rf}
    native = plot_comparison_heatmaps(profiles, sorted_by=reference, apply_to=matched, show=False)
    offset = plot_comparison_heatmaps(profiles, sorted_by=reference, apply_to=matched,
                                      align=True, order=native["order"], show=False)
    summed = plot_comparison_heatmaps(profiles, sorted_by=reference, apply_to=matched,
                                      align="sum", order=native["order"], show=False)
    histogram = peak_direction_sums(hd, rf, order=native["order"])
    assert summed["order"] == offset["order"] == native["order"]
    np.testing.assert_array_equal(summed["peak_sum_deg"], histogram.peak_sum_deg)
    np.testing.assert_array_equal(summed["offset_deg"], -summed["reference_peak_deg"])
    np.testing.assert_array_equal(summed["matched_offset_deg"], summed["reference_peak_deg"])
    np.testing.assert_array_equal(offset["matched_offset_deg"], offset["offset_deg"])
    centers = np.arange(-180, 181, 12)
    reference_image = summed["figures"][0][1].images[0].get_array()
    matched_image = summed["figures"][1][1].images[0].get_array()
    np.testing.assert_array_equal(centers[np.argmax(reference_image, axis=1)], [0, 0])
    np.testing.assert_array_equal(centers[np.argmax(matched_image, axis=1)], histogram.peak_sum_deg)
    np.testing.assert_array_equal(reference_image, offset["figures"][0][1].images[0].get_array())
    assert " + " in summed["figures"][1][1].get_xlabel()
    assert summed["mode"] == "sum"
    for item in (native, offset, summed):
        for figure, _ in item["figures"]:
            plt.close(figure)


def test_sum_and_offset_orders_are_saved_separately(tmp_path, monkeypatch):
    angles = np.arange(-174., 180., 12.)
    hd = profile_table([np.cos(np.deg2rad(angles - 114)) + 2], [267], angles)
    hd.attrs["is_ego_deg"] = False
    rf = profile_table([np.cos(np.deg2rad(angles - 66)) + 2], [267], angles)
    rf.attrs["is_ego_deg"] = True
    monkeypatch.setattr(plt, "show", lambda: None)
    profiles = {"HD": hd, "RF": rf}
    native = compare_both_orders(profiles, "HD", "RF", order_dir=tmp_path)
    for reference, names in (("HD", ("HD", "RF")), ("RF", ("RF", "HD"))):
        for (_, axis), name in zip(native[reference]["figures"], names, strict=True):
            np.testing.assert_array_equal(axis.get_xticks(), [-180, -90, 0, 90, 180])
    compare_both_orders(profiles, "HD", "RF", align=True, previous=native, order_dir=tmp_path)
    compare_both_orders(profiles, "HD", "RF", align="sum", previous=native, order_dir=tmp_path)
    assert len(list(tmp_path.glob("*_order.csv"))) == 6
    offset = pd.read_csv(tmp_path / "HD_RF_aligned_order.csv")
    summed = pd.read_csv(tmp_path / "HD_RF_sum_order.csv")
    np.testing.assert_array_equal(offset[["offset_deg", "matched_offset_deg"]], [[-114, -114]])
    np.testing.assert_array_equal(summed[["offset_deg", "matched_offset_deg", "peak_sum_deg"]],
                                  [[-114, 114, -180]])


def test_peak_sums_optionally_wrap_and_keep_missing_pairs_nan():
    angles = [-174, -90, 66, 114, 174]
    first = profile_table(np.eye(5)[[3, 4, 0, 1]], [7, 9, 11, 13], angles)
    second = profile_table(np.vstack([np.eye(5)[[2, 4, 0]], np.full(5, np.nan)]), [7, 9, 11, 13], angles)
    options = {"order": [7, 9, 11, 13]}
    raw = paired_peak_angles(first, second, is_wrap=False, **options)
    wrapped = paired_peak_angles(first, second, is_wrap=True, **options)
    np.testing.assert_array_equal(raw.peak_sum_deg, [180, 348, -348, np.nan])
    np.testing.assert_array_equal(wrapped.peak_sum_deg, [-180, -12, 12, np.nan])
    np.testing.assert_array_equal(raw.iloc[:, :2], wrapped.iloc[:, :2])
    assert raw.attrs["is_wrap"] is False
    assert wrapped.attrs["is_wrap"] is True


def test_unwrapped_sum_heatmaps_and_csv_keep_both_reference_orders(tmp_path, monkeypatch):
    angles = np.arange(-174., 180., 12.)
    first = profile_table(
        2 + np.cos(np.deg2rad(angles[None, :] - np.array([174, -174, 114, -54])[:, None])),
        [7, 9, 11, 13], angles,
    )
    second = profile_table(
        2 + np.cos(np.deg2rad(angles[None, :] - np.array([66, -174, 174, -18])[:, None])),
        [11, 9, 7, 17], angles,
    )
    profiles = {"HD": first, "RF": second}
    monkeypatch.setattr(plt, "show", lambda: None)
    units = first.index.intersection(second.index)
    profiles = {name: table.loc[units] for name, table in profiles.items()}
    native = compare_both_orders(profiles, "HD", "RF")
    summed = compare_both_orders(
        profiles, "HD", "RF", align="sum", is_wrap=False,
        previous=native, order_dir=tmp_path,
    )
    expected_by_unit = {7: 348, 9: -348, 11: 180}
    for reference, matched in (("HD", "RF"), ("RF", "HD")):
        result = summed[reference]
        assert result["order"] == native[reference]["order"]
        assert set(result["order"]) == set(units)
        expected = np.array([expected_by_unit[unit] for unit in result["order"]])
        np.testing.assert_array_equal(result["peak_sum_deg"], expected)
        np.testing.assert_array_equal(result["peaks"].peak_sum_deg, expected)
        assert result["peaks"].attrs["is_wrap"] is False
        reference_axis = result["figures"][0][1]
        matched_axis = result["figures"][1][1]
        assert reference_axis.get_xlim() == (-180, 180)
        assert matched_axis.get_xlim() == (-360, 360)
        assert "wrapped" not in matched_axis.get_xlabel()
        image = matched_axis.images[0]
        values = np.ma.filled(image.get_array(), np.nan)
        left, right = image.get_extent()[:2]
        centers = left + (np.arange(values.shape[1]) + .5) * (right - left) / values.shape[1]
        paired = np.isfinite(expected)
        np.testing.assert_allclose(centers[np.nanargmax(values[paired], axis=1)], expected[paired])
        saved = pd.read_csv(tmp_path / f"{reference}_{matched}_sum_order.csv")
        assert saved.unit.tolist() == result["order"]
        np.testing.assert_array_equal(saved.peak_sum_deg, expected)
    for results in (native, summed):
        for result in results.values():
            for figure, _ in result["figures"]:
                plt.close(figure)


def test_unwrapped_sum_summary_counts_peaks_outside_signed_angle_range():
    angles = [-174, 66, 114, 174]
    first = profile_table(np.eye(4)[[3, 0, 2]], [7, 9, 11], angles)
    second = profile_table(np.eye(4)[[3, 0, 1]], [7, 9, 11], angles)
    table = peak_direction_sums(first, second, order=[7, 9, 11], is_wrap=False)
    assert table.attrs["is_wrap"] is False
    figure = plot_peak_direction_sums(table)
    histogram, = figure.axes
    expected, _ = np.histogram([348, -348, 180], bins=np.arange(-360, 361, 12))
    np.testing.assert_array_equal([bar.get_height() for bar in histogram.patches], expected)
    assert sum(bar.get_height() for bar in histogram.patches) == 3
    for axis in figure.axes:
        assert axis.get_xlim() == (-360, 360)
        assert "wrapped" not in axis.get_xlabel()
    plt.close(figure)


@pytest.mark.parametrize("reference_peak, matched_peak, expected", [
    (114., 66., -48.), (-18., 174., -168.), (174., -174., 12.),
])
def test_same_offset_across_resolutions_and_circular_boundary(reference_peak, matched_peak, expected):
    angle30 = (np.arange(30) + .5) * 12
    angle60 = np.arange(-180., 180., 6.)
    first = profile_table([2 + np.cos(np.deg2rad(angle30 - reference_peak))], [7], angle30)
    second = profile_table([2 + np.cos(np.deg2rad(angle60 - matched_peak))], [7], angle60)
    original = first.copy()
    offset = -peak_angles(first, [7])
    aligned = align_profiles(first, [7], offset)
    matched = align_profiles(second, [7], offset)
    assert aligned.shape == matched.shape == (1, 30)
    np.testing.assert_array_equal(aligned.columns, np.arange(-180, 180, 12))
    assert aligned.columns[np.argmax(aligned.iloc[0])] == 0
    assert matched.columns[np.argmax(matched.iloc[0])] == expected
    np.testing.assert_array_equal(original, first)


def test_alignment_keeps_missing_intervals_and_centered_image():
    profiles = profile_table([[1, np.nan, 3, 0]], [9], [0, -90, -180, 90], probe="A")
    aligned = align_profiles(profiles, [("A", 9)], [0])
    assert np.isnan(aligned.loc[("A", 9), -48])
    assert aligned.loc[("A", 9), 0] == 1
    result = plot_comparison_heatmaps({"A": profiles, "B": profiles}, sorted_by="A", apply_to="B", align=True, show=False)
    assert result["order"] == [("A", 9)]
    np.testing.assert_array_equal(result["reference_peak_deg"], [-180])
    np.testing.assert_array_equal(result["offset_deg"], [180])
    for fig, ax in result["figures"]:
        values = ax.images[0].get_array()[0]
        left, right, *_ = ax.images[0].get_extent()
        centers = left + (np.arange(len(values)) + .5) * (right - left) / len(values)
        assert centers[np.argmax(values)] == 0
        assert len(values) == 31  # 30 bins, with the circular edge bin shown as two halves.
        assert centers[1] - centers[0] == 12
        np.testing.assert_equal(values[0], values[-1])
        assert fig.get_facecolor() == (1, 1, 1, 1)
        plt.close(fig)


def test_pair_axes_are_independent_and_alignment_retains_native_order():
    angles = np.arange(45, 360, 90)
    first = profile_table([[1, 4, 0, 2], [4, 2, 0, 1]], [7, 3], angles)
    second = profile_table([[1, 0, 4, 2], [2, 4, 0, 1]], [7, 3], angles)
    custom_axis = {"ticks": (-180, 0, 180), "ticklabels": ("−180°", "0°", "+180°")}
    profiles = {"HD": first, "RF": second}
    native = plot_comparison_heatmaps(profiles, sorted_by="HD", apply_to="RF",
                                      sorted_by_axis=ego_coordinates, apply_to_axis=custom_axis, show=False)
    aligned = plot_comparison_heatmaps(profiles, sorted_by="HD", apply_to="RF",
                                       sorted_by_axis=ego_coordinates, apply_to_axis=custom_axis,
                                       align=True, order=native["order"], show=False)
    assert native["order"] == aligned["order"] == [3, 7]
    np.testing.assert_array_equal(aligned["reference_peak_deg"], [45, 135])
    np.testing.assert_array_equal(aligned["offset_deg"], [-45, -135])
    assert [tick.get_text() for tick in native["figures"][1][1].get_xticklabels()] == list(custom_axis["ticklabels"])
    assert [tick.get_text() for tick in aligned["figures"][1][1].get_xticklabels()] == list(custom_axis["ticklabels"])
    image = aligned["figures"][1][1].images[0].get_array()
    np.testing.assert_allclose(np.arange(-180, 181, 12)[np.argmax(image, axis=1)], [90, 90], atol=6)
    for item in (native, aligned):
        for fig, _ in item["figures"]:
            plt.close(fig)


@pytest.mark.parametrize("axis, expected_centers, expected_values", [
    (None, [-135, -45, 45, 135], [1, .2, .6, np.nan]),
    ({"ticklabels": (0, 90, 180, 270, 360)}, [45, 135, 225, 315], [.6, np.nan, 1, .2]),
])
def test_native_heatmap_wraps_source_centers_into_display_window(axis, expected_centers, expected_values):
    profiles = profile_table([[1, 5, np.nan, 3]], [7], [-45, -135, 135, 45])
    assert profile_coordinates(profiles) == ego_coordinates
    original = profiles.copy()
    fig, ax = plot_profiles(profiles, [7], "HD", axis=axis, show=False)
    values = ax.images[0].get_array()[0]
    left, right, *_ = ax.images[0].get_extent()
    centers = left + (np.arange(len(values)) + .5) * (right - left) / len(values)
    np.testing.assert_array_equal(centers, expected_centers)
    np.testing.assert_allclose(values.filled(np.nan), expected_values)
    expected_labels = [-180, -90, 0, 90, 180] if axis is None else axis["ticklabels"]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == list(map(str, expected_labels))
    np.testing.assert_array_equal(profiles, original)
    plt.close(fig)


def test_wrapping_native_angles_preserves_statistics_and_offset_sign():
    reference = np.array([7, 35, 89, 152, 223, 304])
    matched = (reference + 67) % 360
    options = dict(n_permutations=99, random_seed=4)
    original = compare_direction_angles(reference, matched, **options)
    wrapped = compare_direction_angles((reference + 180) % 360 - 180,
                                        (matched + 180) % 360 - 180, **options)
    assert wrapped["same_unit"]["one_to_one_offset_deg"] == pytest.approx(67)
    for key in ("circular_correlation_rho", "alignment_resultant", "permutation_p"):
        assert wrapped["same_unit"][key] == pytest.approx(original["same_unit"][key])
    assert wrapped["pairwise"] == pytest.approx(original["pairwise"])


def test_hd_saved_lists_are_exact_classes_and_sorted(tmp_path):
    save_hd_unit_lists({"unit_id": [9, 7, 3, 5, 11], "unit_data": {"hd_class": [2, 1, None, 2, 0]}}, tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path / "hd_cells_1.npy"), [7])
    np.testing.assert_array_equal(np.load(tmp_path / "hd_cells_2.npy"), [5, 9])
    assert np.load(tmp_path / "hd_cells_2.npy").dtype == np.int64


@pytest.fixture
def write_tc(tmp_path):
    def write(date, session, probe, *, mouse="m19", edges=None):
        path = (tmp_path / mouse / str(date) / f"{date}_{session}"
                / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "unit_id": [7, 9], "unit_data": {
                "hd_class": [2, 1], "rayleigh_p": [.001, .001], "shuffle_p": [.001, .2],
            },
            "spike_counts": [list(range(1, 61)), [1] * 60],
            "occupancy_time_s": [1] * 60,
            "angle_bin_edges_deg": np.linspace(0, 360, 61).tolist() if edges is None else edges,
        }))
        return path
    return write


def test_hd_loader_defaults_to_unsmoothed_count_occupancy_aggregation(write_tc):
    path = write_tc(260918, 9, "A")
    data = json.loads(path.read_text())
    data["spike_counts"][0] = [20] + [0] * 59
    data["occupancy_time_s"][:2] = [1, 3]
    data["occupancy_time_s"][10:12] = [0, 0]
    path.write_text(json.dumps(data))
    table = load_hd_profiles(path, hd_class=2)
    expected = np.zeros((1, 30))
    expected[0, 0] = 5
    expected[0, 5] = np.nan
    np.testing.assert_array_equal(table, expected)


def test_collection_repeated_loads_match_sessions_but_separate_mice_dates_and_probes(write_tc):
    curves = TuningCurveCollection()
    datasets = [("m19", 260918, 2, "A", "HD2"), ("m19", 260918, 9, "A", "HD9"),
                ("m19", 260918, 2, "B", "HD2_B"), ("m19", 260921, 2, "A", "later"),
                ("m20", 260918, 2, "A", "other_mouse")]
    for mouse, date, session, probe, prefix in datasets:
        path = write_tc(date, session, probe, mouse=mouse)
        table = curves.load_tc(path, mouse=mouse, date=date, probe=probe, prefix=prefix, hd_class=2)
        assert table.shape == (1, 30)
        assert table.index.names == ["mouse", "date", "probe", "unit_id"]
        assert table.index.tolist() == [(mouse, str(date), probe, 7)]
    assert gen_overlap_list(curves.profiles["HD2"].index, curves.profiles["HD9"].index) == [("m19", "260918", "A", 7)]
    assert gen_overlap_list(curves.profiles["HD2"].index, curves.profiles["later"].index) == []
    assert gen_overlap_list(curves.profiles["HD2"].index, curves.profiles["HD2_B"].index) == []
    with pytest.raises(ValueError):
        paired_peak_angles(curves.profiles["HD2"], curves.profiles["other_mouse"])
    original = curves.profiles["HD2"].copy()
    curves.load_tc(write_tc(260918, 2, "A"), mouse="m19", date=260918, probe="A", prefix="HD2", hd_class=2)
    pd.testing.assert_frame_equal(curves.profiles["HD2"], original)
    assert len(curves.profiles) == 5


def test_collection_direct_source_and_coordinate_flags(write_tc):
    path = write_tc(260918, 9, "A")
    curves = TuningCurveCollection()
    ccw = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="HD9")
    ego = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="ego", is_ego_deg=True)
    assert curves.profiles["HD9"] is ccw
    np.testing.assert_array_equal(ccw.columns, (np.arange(6., 360., 12.) + 180) % 360 - 180)
    np.testing.assert_array_equal(ego.columns, ccw.columns)
    np.testing.assert_array_equal(ccw.to_numpy(), ego.to_numpy())
    assert ccw.attrs["is_ego_deg"] is False
    assert ego.attrs["is_ego_deg"] is True
    assert profile_coordinates(ccw) == allo_coordinates
    assert profile_coordinates(ego) == ego_coordinates
    absolute = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="absolute")
    pd.testing.assert_frame_equal(ccw, absolute)

    signed_path = write_tc(260918, 10, "A", edges=np.linspace(-180, 180, 61).tolist())
    signed = load_hd_profiles(signed_path)
    np.testing.assert_array_equal(signed.columns, np.arange(-174., 180., 12.))


def test_collection_loads_direct_path_with_literal_braces(tmp_path, write_tc):
    original_path = write_tc(260918, 9, "B")
    custom_path = tmp_path / "m20/260921/ProbeA/custom exports/curve_{date}_{sessionID}_{probe}.tc"
    custom_path.parent.mkdir(parents=True)
    original_path.rename(custom_path)
    curves = TuningCurveCollection()
    loaded = curves.load_tc(str(custom_path), mouse=19, date=260918, probe="B", prefix="custom")
    expected = curves.add(load_hd_profiles(custom_path, probe="B"),
                           mouse="19", date="260918", prefix="expected", is_ego_deg=False)
    pd.testing.assert_frame_equal(loaded, expected)
    assert loaded.index.tolist() == [("19", "260918", "B", 7), ("19", "260918", "B", 9)]


def test_collection_hd_keeps_source_90_and_270_and_display_flag_only_changes_labels(write_tc):
    path = write_tc(260918, 9, "A")
    data = json.loads(path.read_text())
    counts = np.zeros((2, 60))
    counts[0, 14:16] = 10
    counts[1, 44:46] = 10
    data["spike_counts"] = counts.tolist()
    data["unit_data"]["hd_class"] = [2, 2]
    data["unit_data"]["shuffle_p"] = [.001, .001]
    path.write_text(json.dumps(data))
    curves = TuningCurveCollection()
    allo = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="allo", is_ego_deg=False)
    ego = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="ego", is_ego_deg=True)
    np.testing.assert_array_equal(peak_angles(allo, allo.index), [90, -90])
    np.testing.assert_array_equal(peak_angles(ego, ego.index), [90, -90])
    np.testing.assert_array_equal(allo.columns, ego.columns)
    np.testing.assert_array_equal(allo.to_numpy(), ego.to_numpy())
    comparison_peaks = paired_peak_angles(allo, ego, order=allo.index.tolist())
    np.testing.assert_array_equal(comparison_peaks, [[90, 90, -180], [-90, -90, -180]])
    assert allo_coordinates == {"ticklabels": (180, 270, 0, 90, 180)}
    assert ego_coordinates == {"ticklabels": (-180, -90, 0, 90, 180)}
    images = []
    for table, coordinates in (
        (allo, allo_coordinates), (ego, ego_coordinates),
    ):
        original = table.copy()
        figure, axis = plot_profiles(table, table.index.tolist(), "HD", show=False)
        values = axis.images[0].get_array()
        left, right, *_ = axis.images[0].get_extent()
        centers = left + (np.arange(values.shape[1]) + .5) * (right - left) / values.shape[1]
        np.testing.assert_array_equal(centers[np.argmax(values, axis=1)], [90, -90])
        np.testing.assert_array_equal(axis.get_xticks(), [-180, -90, 0, 90, 180])
        assert [tick.get_text() for tick in axis.get_xticklabels()] == list(map(str, coordinates["ticklabels"]))
        assert axis.get_xlim() == (-180, 180)
        screen_x = axis.transAxes.inverted().transform(axis.transData.transform([[90, 0], [-90, 0]]))[:, 0]
        # Native HD 90 is right; native HD 270 wraps to -90 and is left.
        np.testing.assert_allclose(screen_x, [.75, .25])
        images.append(values)
        pd.testing.assert_frame_equal(table, original)
        assert table.attrs == original.attrs
        plt.close(figure)
    np.testing.assert_array_equal(images[0], images[1])
    assert sorted_by_peak(allo, allo.index) == sorted_by_peak(ego, ego.index)


@pytest.mark.parametrize("reference, matched", [("allo", "ego"), ("ego", "allo")])
def test_native_allo_ego_sum_heatmaps_and_histogram_use_the_same_paired_peaks(reference, matched):
    curves = TuningCurveCollection()
    for name, angles, peaks, is_ego in (
        ("allo", np.arange(6., 360., 12.), [-90, 90, -30], False),
        ("ego", np.arange(-174., 180., 12.), [-90, 90, 150], True),
    ):
        values = 2 + np.cos(np.deg2rad(angles[None, :] - np.array(peaks)[:, None]))
        curves.add(profile_table(values, [7, 9, 11], angles, probe="A"),
                   mouse="m19", date=260918, prefix=name, is_ego_deg=is_ego)
    originals = {name: table.copy() for name, table in curves.profiles.items()}
    native = plot_comparison_heatmaps(curves.profiles, sorted_by=reference, apply_to=matched, show=False)
    summed = plot_comparison_heatmaps(curves.profiles, sorted_by=reference, apply_to=matched,
                                      align="sum", order=native["order"], show=False)
    expected_units = [7, 11, 9] if reference == "allo" else [7, 9, 11]
    expected_sums = [-180, 120, -180] if reference == "allo" else [-180, -180, 120]
    assert summed["order"] == native["order"] == [("m19", "260918", "A", unit) for unit in expected_units]
    np.testing.assert_array_equal(summed["peak_sum_deg"], expected_sums)
    for (_, axis), name in zip(native["figures"], (reference, matched), strict=True):
        np.testing.assert_array_equal(axis.get_xticks(), [-180, -90, 0, 90, 180])
    for (_, axis), expected in zip(summed["figures"], ([0, 0, 0], expected_sums), strict=True):
        values = axis.images[0].get_array()
        np.testing.assert_array_equal(np.arange(-180, 181, 12)[np.argmax(values, axis=1)], expected)
        np.testing.assert_array_equal(axis.get_xticks(), [-180, -90, 0, 90, 180])
        assert axis.get_xlim() == (-180, 180)
    summary = peak_direction_sums(curves.profiles["allo"], curves.profiles["ego"], order=native["order"])
    np.testing.assert_array_equal(summary.peak_sum_deg, expected_sums)
    figure = plot_peak_direction_sums(summary)
    histogram = np.array([bar.get_height() for bar in figure.axes[0].patches])
    assert histogram.sum() == 3
    assert histogram[0] == 2
    assert histogram[25] == 1
    for name, table in curves.profiles.items():
        pd.testing.assert_frame_equal(table, originals[name])
        assert table.attrs == originals[name].attrs
    for item in (native, summed):
        for panel, _ in item["figures"]:
            plt.close(panel)
    plt.close(figure)


@pytest.mark.parametrize("reference, matched", [("HD", "RF"), ("RF", "HD")])
@pytest.mark.parametrize("align", [False, True, "sum"])
def test_native_order_images_and_sum_do_not_change_with_hd_display_flag(reference, matched, align):
    allo_angles = np.arange(6., 360., 12.)
    ego_angles = np.arange(-174., 180., 12.)
    units = [267, 7, 9]
    hd = profile_table(
        2 + np.cos(np.deg2rad(allo_angles[None, :] - np.array([246, 90, 270])[:, None])),
        units, allo_angles, probe="A",
    )
    hd.attrs["is_ego_deg"] = False
    rf = profile_table(
        2 + np.cos(np.deg2rad(ego_angles[None, :] - np.array([66, -90, 90])[:, None])),
        units, ego_angles, probe="A",
    )
    profiles = {"HD": hd, "RF": rf}
    original = {"HD": convert_profile_coordinates(hd, is_ego_deg=True), "RF": rf}
    results = [
        plot_comparison_heatmaps(tables, sorted_by=reference, apply_to=matched,
                                 align=align, show=False, figsize=(8, 4))
        for tables in (profiles, original)
    ]
    np.testing.assert_array_equal(results[0]["peak_sum_deg"], [-48, 0, 0] if reference == "HD" else [0, -48, 0])
    assert results[0]["order"] == results[1]["order"]
    np.testing.assert_array_equal(profiles["HD"].columns, original["HD"].columns)
    np.testing.assert_array_equal(profiles["HD"], original["HD"])
    assert profiles["HD"].attrs["is_ego_deg"] is False
    assert original["HD"].attrs["is_ego_deg"] is True
    pd.testing.assert_frame_equal(results[0]["peaks"], results[1]["peaks"])
    for (figure, axis), (old_figure, old_axis) in zip(results[0]["figures"], results[1]["figures"], strict=True):
        np.testing.assert_array_equal(axis.images[0].get_array(), old_axis.images[0].get_array())
        np.testing.assert_array_equal(figure.get_size_inches(), [8, 4])
        if align:
            assert axis.get_xlim() == old_axis.get_xlim() == (-180, 180)
            np.testing.assert_array_equal(axis.get_xticks(), [-180, -90, 0, 90, 180])
        plt.close(figure)
        plt.close(old_figure)


def test_smoothing_wraps_angles_without_mixing_rows_filling_gaps_or_changing_input():
    angles = np.arange(-174., 180., 12.)
    values = np.zeros((3, 30))
    values[0, 0] = 1
    values[1] = 4
    values[1, 5:8] = np.nan
    values[2] = np.nan
    profiles = profile_table(values, [7, 9, 11], angles, probe="A")
    profiles.attrs["is_ego_deg"] = True
    original = profiles.copy()
    pd.testing.assert_frame_equal(smooth_profiles(profiles), original)
    result = smooth_profiles(profiles, smoothing_bins=1)
    assert result.iloc[0, -1] > 0
    assert result.iloc[0, -1] == pytest.approx(result.iloc[0, 1])
    assert result.iloc[0].sum() == pytest.approx(1)
    np.testing.assert_allclose(result.iloc[1].dropna(), 4)
    np.testing.assert_array_equal(np.isnan(result), np.isnan(profiles))
    assert result.attrs == profiles.attrs
    pd.testing.assert_frame_equal(profiles, original)

    shuffled = profiles.iloc[:, np.roll(np.arange(30)[::-1], 7)]
    shuffled_result = smooth_profiles(shuffled, smoothing_bins=1)
    pd.testing.assert_frame_equal(shuffled_result.reindex(columns=profiles.columns), result)


def test_explicit_smoothing_controls_peaks_images_and_histogram_together(monkeypatch):
    angles = np.arange(-174., 180., 12.)
    row = np.zeros(30)
    row[2] = 4
    row[14:17] = 3
    hd = profile_table([row, np.roll(row, 4)], [7, 9], angles)
    rf = profile_table(
        2 + np.cos(np.deg2rad(angles[None, :] - np.array([66, -54])[:, None])),
        [7, 9], angles,
    )
    np.testing.assert_array_equal(peak_angles(hd, hd.index), [-150, -102])
    monkeypatch.setattr(plt, "show", lambda: None)
    profiles = {name: smooth_profiles(table, 1) for name, table in {"HD": hd, "RF": rf}.items()}
    results = compare_both_orders(profiles, "HD", "RF", align="sum", figsize=(7, 3))
    np.testing.assert_array_equal(results["HD"]["reference_peak_deg"], [6, 54])
    for result in results.values():
        sums = result["peaks"].peak_sum_deg.to_numpy()
        np.testing.assert_array_equal(result["peak_sum_deg"], sums)
        image = result["figures"][1][1].images[0].get_array()
        np.testing.assert_array_equal(np.arange(-180, 181, 12)[np.argmax(image, axis=1)], sums)
        figure = plot_peak_direction_sums(result["peaks"])
        expected_counts, _ = np.histogram(sums, bins=np.linspace(-180, 180, 31))
        np.testing.assert_array_equal([bar.get_height() for bar in figure.axes[0].patches], expected_counts)
        plt.close(figure)
        for figure, _ in result["figures"]:
            np.testing.assert_array_equal(figure.get_size_inches(), [7, 3])
            plt.close(figure)


@pytest.mark.parametrize("figsize", [None, (8, 4)])
def test_three_hundred_units_keep_compact_figures_and_every_row(figsize):
    angles = np.arange(-174., 180., 12.)
    profiles = profile_table(np.tile(2 + np.cos(np.deg2rad(angles - 6)), (300, 1)),
                             np.arange(300), angles)
    options = {} if figsize is None else {"figsize": figsize}
    figure, axis = plot_profiles(profiles, profiles.index.tolist(), "HD", show=False, **options)
    np.testing.assert_array_equal(figure.get_size_inches(), (10, 6) if figsize is None else figsize)
    assert axis.images[0].get_array().shape == (300, 30)
    assert len(axis.get_yticks()) == 300
    plt.close(figure)

    figure = plot_peak_direction_sums(paired_peak_angles(profiles, profiles), **options)
    np.testing.assert_array_equal(figure.get_size_inches(), (6, 4) if figsize is None else figsize)
    assert sum(bar.get_height() for bar in figure.axes[0].patches) == 300
    plt.close(figure)


@pytest.mark.parametrize("is_ego", [False, True])
def test_collection_add_wraps_native_angles_independently_of_display_flag(is_ego):
    angles = np.arange(186., 540., 12.)
    source = profile_table([np.arange(1., 31.)], [7], angles, probe="A")
    original = source.copy()
    table = TuningCurveCollection().add(source, mouse="m19", date=260918, prefix="curve", is_ego_deg=is_ego)
    expected = (angles + 180) % 360 - 180
    np.testing.assert_array_equal(table.columns, expected)
    np.testing.assert_array_equal(table.to_numpy(), source.to_numpy())
    assert table.attrs["is_ego_deg"] is is_ego
    pd.testing.assert_frame_equal(source, original)
    assert source.attrs == original.attrs


@pytest.mark.parametrize("first_name", ["allo", "ego"])
def test_collection_combines_native_angles_and_keeps_first_display_flag(first_name):
    curves = TuningCurveCollection()
    angles = np.arange(-174., 180., 12.)
    values = 2 + np.cos(np.deg2rad(angles + 90))
    curves.add(profile_table([values], [7], angles, probe="A"),
               mouse="m19", date=260918, prefix="allo", is_ego_deg=False)
    curves.add(profile_table([values[::-1]], [7], angles[::-1], probe="B"),
               mouse="m19", date=260918, prefix="ego", is_ego_deg=True)
    originals = {name: table.copy() for name, table in curves.profiles.items()}
    second_name = "ego" if first_name == "allo" else "allo"
    combined = curves.combine("pooled", [first_name, second_name])
    assert combined.attrs["is_ego_deg"] is (first_name == "ego")
    np.testing.assert_array_equal(combined.columns, originals[first_name].columns)
    np.testing.assert_array_equal(combined.iloc[0], combined.iloc[1])
    np.testing.assert_array_equal(peak_angles(combined, combined.index), [-90, -90])
    assert profile_coordinates(combined) == (ego_coordinates if first_name == "ego" else allo_coordinates)
    for name, original in originals.items():
        pd.testing.assert_frame_equal(curves.profiles[name], original)
        assert curves.profiles[name].attrs == original.attrs


def test_collection_ebc_resamples_to_30_bins_without_filling_missing_intervals(tmp_path):
    path = (tmp_path / "260918/260918_11/data/spatial_cells/ProbeB/baseline"
            / "egocentric_rate_map.rfmap")
    path.parent.mkdir(parents=True)
    values = np.arange(60., dtype=float)
    values[10:14] = np.nan
    save_egocentric_rfmap(
        path, values[None, :, None],
        {"distance_edges": np.array([0, 8]), "theta_edges": np.arange(0, 361, 6)},
        [7], [0, 1],
    )
    curves = TuningCurveCollection()
    table = curves.load_tc(path, mouse="m19", date=260918, probe="B", prefix="EBC11",
                           kind="EBC", is_ego_deg=True)
    centers = np.arange(-174., 180., 12.)
    np.testing.assert_array_equal(table.columns, centers)
    expected = np.interp(centers, np.arange(3, 360, 6), values, period=360)
    np.testing.assert_allclose(table.iloc[0], expected)
    assert np.isnan(table.loc[("m19", "260918", "B", 7), 66.])
    assert np.isnan(table.loc[("m19", "260918", "B", 7), 78.])
    assert np.isfinite(table.loc[("m19", "260918", "B", 7), 90.])

    allo = curves.load_tc(path, mouse="m19", date=260918, probe="B", prefix="allo",
                          kind="EBC", is_ego_deg=False)
    np.testing.assert_array_equal(allo.columns, table.columns)
    np.testing.assert_allclose(allo, table)


def test_collection_combines_by_angle_and_rejects_duplicate_units():
    angles = np.arange(6., 360., 12.)
    values = np.arange(1., 31.)
    values[:2] = 40
    first = profile_table([values], [7], angles, probe="A")
    second = profile_table([values[::-1]], [7], angles[::-1], probe="B")
    curves = TuningCurveCollection()
    table = curves.add(first, mouse="m19", date=260918, prefix="first")
    np.testing.assert_array_equal(table.columns, first.columns)
    np.testing.assert_array_equal(peak_angles(table, table.index), peak_angles(first, first.index))
    curves.add(second, mouse="m19", date=260918, prefix="second")
    curves.add(first, mouse="m19", date=260921, prefix="later")
    curves.add(first, mouse="m20", date=260918, prefix="other_mouse")
    single = curves.combine("single", ["first"])
    pd.testing.assert_frame_equal(single, table)
    assert single.attrs["is_ego_deg"] is True
    same_order = curves.combine("same_order", ["first", "later"])
    np.testing.assert_array_equal(peak_angles(same_order, same_order.index), [6., 6.])
    combined = curves.combine("pooled", ["first", "second", "later", "other_mouse"])
    assert combined.index.tolist() == [
        ("m19", "260918", "A", 7), ("m19", "260918", "B", 7),
        ("m19", "260921", "A", 7), ("m20", "260918", "A", 7),
    ]
    np.testing.assert_array_equal(combined.columns, first.columns)
    np.testing.assert_array_equal(combined.iloc[0], combined.iloc[1])
    np.testing.assert_array_equal(combined.iloc[0], combined.iloc[2])
    np.testing.assert_array_equal(combined.iloc[0], combined.iloc[3])
    np.testing.assert_array_equal(peak_angles(combined, combined.index), [6., 6., 6., 6.])
    assert combined.attrs["is_ego_deg"] is True
    with pytest.raises(ValueError, match="overlapping"):
        curves.combine("duplicate", ["first", "first"])
    assert "duplicate" not in curves.profiles
    curves.add(first, mouse="m19", date=260918, prefix="same_units")
    with pytest.raises(ValueError, match="overlapping"):
        curves.combine("duplicate", ["first", "same_units"])
    curves.combine("pooled", ["first", "second", "later"])
    assert len(curves.profiles["pooled"]) == 3


def test_collection_peak_sum_keeps_unit_identity_without_repeating_unit_labels():
    curves = TuningCurveCollection()
    native = profile_table([np.arange(30.)], [7], np.arange(-174., 180., 12.), probe="B")
    table = curves.add(native, mouse="m19", date=260918, prefix="curve")
    figure = plot_peak_direction_sums(peak_direction_sums(table, table), hd_label="custom")
    assert table.index.tolist() == [("m19", "260918", "B", 7)]
    assert figure.axes[0].get_title() == "custom + RF"
    assert sum(bar.get_height() for bar in figure.axes[0].patches) == 1
    assert figure.axes[0].texts[0].get_text().endswith("Rayleigh p = —")
    plt.close(figure)


def test_collection_preserves_zero_and_missing_profiles_on_the_30_bin_grid():
    values = np.array([[0.] * 60, [1., np.nan] * 30, [2.] * 60])
    profiles = profile_table(values, [7, 9, 11], np.arange(3., 360., 6.), probe="A")
    curves = TuningCurveCollection()
    table = curves.add(profiles, mouse="m19", date=260918, prefix="supported")
    assert table.index.tolist() == [("m19", "260918", "A", unit) for unit in (7, 9, 11)]
    np.testing.assert_array_equal(table.iloc[0], np.zeros(30))
    assert table.iloc[1].isna().all()
    np.testing.assert_array_equal(table.iloc[2], np.full(30, 2.))


@pytest.mark.parametrize("has_localization", [False, True])
def test_rf_loaders_keep_all_file_units_independently_of_localization(tmp_path, has_localization):
    from Utils.direction_comparison import load_rf, load_rf_centers, load_rf_profiles

    path = tmp_path / "custom RF exports" / "example.rfmap"
    path.parent.mkdir(parents=True)
    angles = np.arange(-174., 180., 12.)
    counts = np.array([np.full((1, 30, 1), value) for value in (100., 4., 0., 8.)])
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [9, 7, 11, 13], "xPositions": angles.tolist(), "yPositions": [0.],
        "timeBinEdges": [0., .2], "stimulusPresentationCounts": [[1] * 30],
    }))
    center = np.zeros((1, 1, 30), dtype=bool)
    center[0, 0, 3] = True
    if has_localization:
        np.savez(path.with_suffix(".npz"), mask_2d=center, center_2d=center, unit_ids=[7])
        centers = load_rf_centers(path, probe="A")
        assert centers.index.tolist() == [("A", 7)]
        np.testing.assert_array_equal(centers, [[angles[3], 0.]])
    curves = TuningCurveCollection()
    table = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="RF2",
                           kind="RF", is_ego_deg=True)
    assert table.index.tolist() == [("m19", "260918", "A", unit) for unit in (9, 7, 11, 13)]
    np.testing.assert_allclose(table.to_numpy(), counts[:, 0, :, 0])
    direct = load_rf(path, mouse="m19", date=260918, probe="A")
    np.testing.assert_allclose(direct, table)
    assert direct.index.equals(table.index)
    native = load_rf_profiles(path, probe="A")
    assert native.index.tolist() == [("A", unit) for unit in (9, 7, 11, 13)]
    np.testing.assert_allclose(native, table)
    np.testing.assert_array_equal(table.columns, angles)
    ccw = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="ccw",
                         kind="RF", is_ego_deg=False)
    np.testing.assert_array_equal(ccw.columns, angles)
    np.testing.assert_array_equal(ccw.to_numpy(), table.to_numpy())

    data = json.loads(path.read_text())
    counts[1] = np.arange(30).reshape(1, 30, 1)
    data["unitsSpikeCounts"] = counts.tolist()
    path.write_text(json.dumps(data))
    raw = curves.load_tc(path, mouse="m19", date=260918, probe="A", prefix="raw",
                         kind="RF", is_ego_deg=True, smoothing_bins=0)
    np.testing.assert_array_equal(raw.loc[("m19", "260918", "A", 7)], np.arange(30))
    assert raw.index.equals(table.index)


@pytest.mark.parametrize("threshold, expected_ids", [(0, [7]), (2, [7, 9]), (3, [7, 9, 11]), (None, [7, 9, 11, 13])])
def test_rf_missing_bin_limit_uses_native_projection_and_keeps_measured_zeros(tmp_path, threshold, expected_ids):
    from Utils.direction_comparison import load_rf, load_rf_profiles, plot_sort

    path = tmp_path / "rates.rfmap"
    rates = np.ones((4, 2, 30, 1))
    rates[0] = 0  # Measured zero responses are not missing.
    rates[1, :, :2] = np.nan
    rates[1, 0, 2:5] = np.nan  # The other y row still measures these x bins.
    rates[1, :, 10] = 5
    rates[2, :, :3] = np.nan
    rates[3] = np.nan
    stored = rates.astype(object)
    stored[np.isnan(rates)] = None
    path.write_text(json.dumps({
        "unitsSpikeCounts": stored.tolist(), "unitsSpikeCountsSize": list(rates.shape),
        "unitPool": [7, 9, 11, 13], "xPositions": np.arange(-174., 180., 12.).tolist(),
        "yPositions": [-12., 12.], "timeBinEdges": [0., .2], "responseUnits": "Hz",
    }))
    options = {} if threshold is None else {"max_missing_bins": threshold}
    native = load_rf_profiles(path, **options)
    assert native.index.tolist() == [("A", unit) for unit in expected_ids]
    table = load_rf(path, mouse="m19", date=260918, **options)
    assert table.index.tolist() == [("m19", "260918", "A", unit) for unit in expected_ids]
    np.testing.assert_allclose(table, native)
    np.testing.assert_array_equal(table.iloc[0], np.zeros(30))
    if 9 in expected_ids:
        row = native.loc[("A", 9)].to_numpy()
        assert np.isnan(row[:2]).all()
        np.testing.assert_array_equal(row[2:5], [1., 1., 1.])
        np.testing.assert_array_equal(row[9:12], [2., 10., 2.])  # No smoothing.
        assert np.isnan(row).sum() == 2

    collection = TuningCurveCollection()
    loaded = collection.load_tc(path, mouse="m19", date=260918, probe="A", prefix="RF", kind="RF", **options)
    np.testing.assert_allclose(loaded, table)
    assert loaded.index.equals(table.index)
    reference = collection.add(profile_table(np.ones((4, 30)), [7, 9, 11, 13], native.columns, probe="A"),
                               mouse="m19", date=260918, prefix="HD")
    result = plot_sort(reference.loc[table.index], table, show=False)
    assert set(result["order"]) == set(table.index)
    for figure, _ in result["figures"]:
        plt.close(figure)


@pytest.mark.parametrize("coverage_field", ["stimulusPresentationCounts", "occupancyTimeSec"])
@pytest.mark.parametrize("missing_bins", [2, 3])
def test_raw_rf_missing_bins_use_coverage_after_collapsing_y(tmp_path, coverage_field, missing_bins):
    from Utils.direction_comparison import load_rf, load_rf_profiles

    path = tmp_path / "counts.rfmap"
    coverage = np.ones((2, 30))
    coverage[:, :missing_bins] = 0
    coverage[0, 10] = 0
    counts = coverage[None, :, :, None].copy()
    counts[:, :, 11:] = 0  # Many measured zero bins must not exclude the unit.
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7], "xPositions": np.arange(-174., 180., 12.).tolist(),
        "yPositions": [-12., 12.], "timeBinEdges": [0., .2], coverage_field: coverage.tolist(),
    }))
    native = load_rf_profiles(path, max_missing_bins=2)
    table = load_rf(path, mouse="m19", date=260918, max_missing_bins=2)
    assert len(native) == len(table) == (1 if missing_bins == 2 else 0)
    unfiltered = load_rf_profiles(path)
    row = unfiltered.iloc[0].to_numpy()
    assert np.isnan(row).sum() == missing_bins
    assert row[10] == 1.
    np.testing.assert_array_equal(row[11:], np.zeros(19))


def test_ebc_loaders_keep_all_file_units_including_zero_and_missing_rows(tmp_path):
    from Utils.direction_comparison import load_ebc

    path = tmp_path / "egocentric_rate_map.rfmap"
    values = np.array([np.full((30, 1), value) for value in (2., 0., np.nan)])
    save_egocentric_rfmap(
        path, values, {"distance_edges": np.array([0, 8]), "theta_edges": np.arange(0, 361, 12)},
        [7, 9, 11], [0, 1],
    )
    native = load_ebc_profiles(path, probe="B")
    assert native.index.tolist() == [("B", unit) for unit in (7, 9, 11)]
    np.testing.assert_allclose(native, values[:, :, 0])
    table = load_ebc(path, mouse="m19", date=260918, probe="B")
    assert table.index.tolist() == [("m19", "260918", "B", unit) for unit in (7, 9, 11)]
    np.testing.assert_allclose(table, values[:, :, 0])


@pytest.mark.parametrize("threshold, expected_ids", [(0, [7]), (2, [7, 9]), (3, [7, 9, 11]), (None, [7, 9, 11, 13])])
def test_rf_zero_bin_limit_keeps_boundary_and_can_be_disabled(tmp_path, threshold, expected_ids):
    from Utils.direction_comparison import load_rf, load_rf_profiles, rf_pick

    path = tmp_path / "zeros.rfmap"
    counts = np.ones((4, 2, 30, 1))
    for row, zero_bins in zip(counts, [0, 2, 3, 30], strict=True):
        row[:, :zero_bins] = 0
    counts[0, :, 0] = 1e-12  # Small measured responses are not exact zeros.
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7, 9, 11, 13], "xPositions": np.arange(-174., 180., 12.).tolist(),
        "yPositions": [-12., 12.], "timeBinEdges": [0., .2],
    }))
    native = load_rf_profiles(path)
    assert [info["zero_bins"] for info in native.attrs["unit_info"].values()] == [0, 2, 3, 30]
    selected = rf_pick(native, max_zero_bins=threshold)
    assert selected.index.tolist() == [("A", unit) for unit in expected_ids]
    loaded = load_rf_profiles(path, max_zero_bins=threshold)
    pd.testing.assert_frame_equal(loaded, selected)
    table = load_rf(path, mouse="m19", date=260827, max_zero_bins=threshold)
    assert table.index.tolist() == [("m19", "260827", "A", unit) for unit in expected_ids]
    np.testing.assert_allclose(table, selected)
    assert native.index.tolist() == [("A", unit) for unit in (7, 9, 11, 13)]


def test_rf_zero_and_missing_bin_limits_both_apply_without_counting_nan_as_zero(tmp_path):
    from Utils.direction_comparison import load_rf_profiles, rf_pick

    path = tmp_path / "partial.rfmap"
    counts = np.ones((5, 1, 30, 1))
    for row, zero_bins, missing_bins in zip(counts, [2, 3, 1, 0, 0], [2, 1, 3, 0, 30], strict=True):
        row[:, :zero_bins] = 0
        row[:, 30 - missing_bins:] = np.nan
    stored = counts.astype(object)
    stored[np.isnan(counts)] = None
    path.write_text(json.dumps({
        "unitsSpikeCounts": stored.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7, 9, 11, 13, 15], "xPositions": np.arange(-174., 180., 12.).tolist(),
        "yPositions": [0.], "timeBinEdges": [0., .2], "responseUnits": "Hz",
    }))
    native = load_rf_profiles(path)
    info = native.attrs["unit_info"]
    assert [value["zero_bins"] for value in info.values()] == [2, 3, 1, 0, 0]
    assert [value["missing_bins"] for value in info.values()] == [2, 1, 3, 0, 30]
    for missing_limit, zero_limit, expected_ids in (
        (2, 2, [7, 13]), (2, None, [7, 9, 13]),
        (None, 2, [7, 11, 13, 15]), (None, None, [7, 9, 11, 13, 15]),
    ):
        selected = rf_pick(native, max_missing_bins=missing_limit, max_zero_bins=zero_limit)
        assert selected.index.tolist() == [("A", unit) for unit in expected_ids]
        loaded = load_rf_profiles(path, max_missing_bins=missing_limit, max_zero_bins=zero_limit)
        pd.testing.assert_frame_equal(loaded, selected)


@pytest.mark.parametrize("smoothing_bins", [0, 1])
def test_rf_zero_bin_count_uses_measured_native_bins_before_resampling_and_smoothing(tmp_path, smoothing_bins):
    from Utils.direction_comparison import load_rf, load_rf_profiles, rf_pick

    path = tmp_path / "native.rfmap"
    counts = np.ones((1, 2, 60, 1))
    counts[:, :, [0, 2, 10, 11]] = 0
    coverage = np.ones((2, 60))
    coverage[:, [10, 11]] = 0  # Unpresented zeros are missing bins, not measured zeros.
    path.write_text(json.dumps({
        "unitsSpikeCounts": counts.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [7], "xPositions": np.arange(3., 360., 6.).tolist(),
        "yPositions": [-12., 12.], "timeBinEdges": [0., .2],
        "stimulusPresentationCounts": coverage.tolist(),
    }))
    native = load_rf_profiles(path, smoothing_bins=smoothing_bins)
    assert native.attrs["unit_info"][("A", 7)] == {"missing_bins": 2, "zero_bins": 2}
    assert (native.iloc[0] == 0).sum() == (2 if smoothing_bins == 0 else 0)
    table = load_rf(path, mouse="m19", date=260827, smoothing_bins=smoothing_bins)
    assert table.shape == (1, 30)
    assert (table.iloc[0] == 0).sum() == 0
    assert table.attrs["unit_info"][("m19", "260827", "A", 7)] == {"missing_bins": 2, "zero_bins": 2}
    assert rf_pick(table, max_zero_bins=1).empty
    assert len(rf_pick(table, max_zero_bins=2)) == 1
    assert load_rf(path, mouse="m19", date=260827, smoothing_bins=smoothing_bins, max_zero_bins=1).empty


@pytest.fixture
def localized_rf_source(tmp_path):
    source = tmp_path / "localized.rfmap"
    counts = np.array([np.full((1, 30, 1), value) for value in (1., 2., 0., 3., 4.)])
    counts[0, 0, :, 0] = np.arange(1., 31.)
    counts[1, :, :3] = np.nan
    counts[3, :, :2] = 0
    stored = counts.astype(object)
    stored[np.isnan(counts)] = None
    source.write_text(json.dumps({
        "unitsSpikeCounts": stored.tolist(), "unitsSpikeCountsSize": list(counts.shape),
        "unitPool": [9, 7, 11, 13, 15], "xPositions": np.arange(-174., 180., 12.).tolist(),
        "yPositions": [0.], "timeBinEdges": [0., .2], "responseUnits": "Hz",
    }))
    # Saved results can contain a QC subset in a different order from the source.
    mask_2d = np.zeros((3, 2, 30), dtype=np.uint8)
    mask_2d[[0, 2], 1, 5] = 1
    np.savez(source.with_suffix(".npz"), unit_ids=[13, 9, 7], mask_2d=mask_2d)
    mask_1d = np.zeros((3, 1, 30), dtype=np.uint8)
    mask_1d[[0, 2], 0, 9] = 1
    np.savez(source.with_name(source.stem + "_1d.npz"), unit_ids=[11, 7, 9], mask_2d=mask_1d)
    return source


@pytest.mark.parametrize("rf_type, expected_ids", [(None, [9, 7, 11, 13, 15]), ("2d", [7, 13]), ("1d", [9, 11]), ("either", [9, 7, 11, 13])])
@pytest.mark.parametrize("smoothing_bins", [0, 1])
def test_rf_localization_filter_matches_saved_ids_and_keeps_source_values(localized_rf_source, rf_type, expected_ids, smoothing_bins):
    from Utils.direction_comparison import load_rf, load_rf_profiles

    native = load_rf_profiles(localized_rf_source, smoothing_bins=smoothing_bins)
    selected = load_rf_profiles(localized_rf_source, rf_type=rf_type, smoothing_bins=smoothing_bins)
    assert selected.index.tolist() == [("A", unit) for unit in expected_ids]
    np.testing.assert_allclose(selected, native.loc[selected.index], equal_nan=True)
    table = load_rf(localized_rf_source, mouse="m19", date=260827, rf_type=rf_type, smoothing_bins=smoothing_bins)
    assert table.index.tolist() == [("m19", "260827", "A", unit) for unit in expected_ids]
    np.testing.assert_allclose(table, selected, equal_nan=True)
    curves = TuningCurveCollection()
    collection = curves.load_tc(localized_rf_source, mouse="m19", date=260827, probe="A", prefix="RF",
                                kind="RF", rf_type=rf_type, smoothing_bins=smoothing_bins)
    assert collection.index.equals(table.index)
    np.testing.assert_allclose(collection, table, equal_nan=True)


@pytest.mark.parametrize("missing_limit, zero_limit, expected_ids", [(2, None, [9, 11, 13]), (None, 2, [9, 7, 13]), (2, 2, [9, 13])])
def test_rf_localization_filter_combines_with_native_bin_limits(localized_rf_source, missing_limit, zero_limit, expected_ids):
    from Utils.direction_comparison import load_rf, load_rf_profiles

    options = dict(rf_type="either", max_missing_bins=missing_limit, max_zero_bins=zero_limit)
    native = load_rf_profiles(localized_rf_source, **options)
    assert native.index.tolist() == [("A", unit) for unit in expected_ids]
    table = load_rf(localized_rf_source, mouse="m19", date=260827, **options)
    assert table.index.tolist() == [("m19", "260827", "A", unit) for unit in expected_ids]
    np.testing.assert_allclose(table, native, equal_nan=True)


@pytest.mark.parametrize("rf_type", ["2d", "1d", "either"])
def test_rf_localization_filter_can_return_no_detected_units(localized_rf_source, rf_type):
    from Utils.direction_comparison import load_rf, load_rf_profiles

    for path in (localized_rf_source.with_suffix(".npz"), localized_rf_source.with_name(localized_rf_source.stem + "_1d.npz")):
        np.savez(path, unit_ids=[13, 9], mask_2d=np.zeros((2, 1, 30), dtype=np.uint8))
    native = load_rf_profiles(localized_rf_source, rf_type=rf_type)
    table = load_rf(localized_rf_source, mouse="m19", date=260827, rf_type=rf_type)
    assert native.shape == table.shape == (0, 30)
    assert table.index.names == ["mouse", "date", "probe", "unit_id"]


def test_rf_localization_filter_requires_only_requested_result_files(localized_rf_source):
    from Utils.direction_comparison import load_rf_profiles

    result_2d = localized_rf_source.with_suffix(".npz")
    result_1d = localized_rf_source.with_name(localized_rf_source.stem + "_1d.npz")
    result_1d.unlink()
    assert load_rf_profiles(localized_rf_source).shape[0] == 5
    assert load_rf_profiles(localized_rf_source, rf_type="2d").index.tolist() == [("A", 7), ("A", 13)]
    for rf_type in ("1d", "either"):
        with pytest.raises(FileNotFoundError, match="_1d.npz"):
            load_rf_profiles(localized_rf_source, rf_type=rf_type)
    result_2d.unlink()
    for rf_type in ("2d", "either"):
        with pytest.raises(FileNotFoundError, match="localized.npz"):
            load_rf_profiles(localized_rf_source, rf_type=rf_type)
    result_2d.write_text("unused localization")
    result_1d.write_text("unused localization")
    assert load_rf_profiles(localized_rf_source, rf_type=None).shape[0] == 5
    with pytest.raises(ValueError):
        load_rf_profiles(localized_rf_source, rf_type="other")
