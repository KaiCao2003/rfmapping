from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
from Utils.plotting import plot_keyed_heatmap
from Utils.rflocate import RFMapList, asrfmap


def test_rf_profiles_consumes_explicit_time_and_spatial_transforms():
    values = np.arange(24., dtype=float).reshape(2, 4, 3)
    source = asrfmap(values, start_time=-.1, time_bin=.1)
    source = replace(source, unit_id=7, x_positions=np.array([15., 105., 195., 285.]))
    maps = RFMapList([source], "<array>")

    prepared = maps.sum(0., .2, show_progress=False).sum_to_1d(axis="x")
    profiles = comparison.rf_profiles(prepared, probe="B")

    assert profiles.index.tolist() == [("B", 7)]
    np.testing.assert_array_equal(profiles.columns, source.x_positions)
    np.testing.assert_array_equal(profiles.iloc[0], values[..., 1:].sum(axis=(0, 2)))
    profiles.iloc[0, 0] = 999.
    np.testing.assert_array_equal(source.spike_counts, values)


def test_rf_profiles_wraps_loaded_csv_without_changing_native_values():
    tc = pd.DataFrame(
        [[0., np.nan, 4.], [2., 3., 5.]],
        index=pd.Index([23, 7], name="unit_id"), columns=[195., -45., 285.],
    )
    original = tc.copy(deep=True)

    profiles = comparison.rf_profiles(tc, probe="B")

    assert profiles.index.tolist() == [("B", 23), ("B", 7)]
    np.testing.assert_array_equal(profiles.columns, tc.columns)
    np.testing.assert_array_equal(profiles.to_numpy(), tc.to_numpy())
    assert profiles.attrs["unit_info"] == {
        ("B", 23): {"zero_bins": 1}, ("B", 7): {"zero_bins": 0},
    }
    profiles.iloc[0, 0] = 999.
    pd.testing.assert_frame_equal(tc, original)


def test_saved_rf_selection_uses_unit_ids_and_preserves_native_responses():
    index = pd.MultiIndex.from_tuples([("A", 7), ("A", 11), ("A", 13)], names=["probe", "unit_id"])
    profiles = pd.DataFrame([[1., 5.], [2., np.nan], [0., 3.]], index=index, columns=[195., 285.])
    profiles.attrs = {"unit_info": {key: {"zero_bins": i} for i, key in enumerate(index)}}
    first = {"unit_ids": np.array([13, 7]), "mask_2d": np.array([[[0, 0]], [[1, 0]]])}
    second = {"unit_ids": np.array([11]), "mask_2d": np.array([[[0, 1]]])}

    selected = comparison.select_rf_profiles(profiles, first, second)

    pd.testing.assert_frame_equal(selected, profiles.loc[[("A", 7), ("A", 11)]])
    assert list(selected.attrs["unit_info"]) == [("A", 7), ("A", 11)]
    selected.attrs["unit_info"][("A", 7)]["zero_bins"] = 99
    assert profiles.attrs["unit_info"][("A", 7)]["zero_bins"] == 0
    pooled = profiles.copy()
    pooled.index = pd.MultiIndex.from_tuples([("A", 7), ("B", 11), ("B", 13)], names=index.names)
    with pytest.raises(ValueError, match="one probe"):
        comparison.select_rf_profiles(pooled, first)


def test_recording_profiles_only_adds_identity_and_label():
    index = pd.MultiIndex.from_tuples([("B", 7)], names=["probe", "unit_id"])
    profiles = pd.DataFrame([[2., np.nan, 5.]], index=index, columns=[90., 210., 330.])
    profiles.attrs = {"range": [0, 360], "unit_info": {("B", 7): {"zero_bins": 0}}}

    result = comparison.recording_profiles(profiles, mouse="m19", date=260827, label="RF")

    assert result.index.tolist() == [("m19", "260827", "B", 7)]
    np.testing.assert_array_equal(result.columns, profiles.columns)
    np.testing.assert_array_equal(result.to_numpy(), profiles.to_numpy())
    assert result.attrs["range"] == [0, 360]
    assert result.attrs["label"] == "RF"
    assert result.attrs["unit_info"] == {("m19", "260827", "B", 7): {"zero_bins": 0}}


@pytest.mark.parametrize("transform", [comparison.normalize_tc, comparison.zscore_tc])
def test_transform_preserves_table_identity_and_does_not_mutate_source(transform):
    index = pd.MultiIndex.from_tuples(
        [("mouse2", "B", 7), ("mouse1", "A", 3)], names=["mouse", "probe", "unit_id"],
    )
    source = pd.DataFrame(
        [[1, 3, 8], [2, 7, 5]], index=index,
        columns=pd.Index([90, -150, -30], name="angle_deg"),
    )
    source.attrs = {
        "label": "RF", "range": comparison.tcRange(False),
        "unit_info": {key: {"hd_class": 2} for key in index},
    }
    original = source.copy(deep=True)

    result = transform(source)

    assert result is not source
    assert result.shape == source.shape
    pd.testing.assert_index_equal(result.index, source.index)
    pd.testing.assert_index_equal(result.columns, source.columns)
    assert all(np.issubdtype(dtype, np.floating) for dtype in result.dtypes)
    assert result.attrs == source.attrs
    result.iloc[0, 0] = 1000
    pd.testing.assert_frame_equal(source, original)
    assert source.attrs == original.attrs


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_normalize_tc_uses_each_finite_row_max_and_preserves_missing_bins():
    source = pd.DataFrame([
        [2, 8, np.nan, 4],
        [0, 0, np.nan, 0],
        [np.nan, np.nan, np.nan, np.nan],
        [np.inf, 2, 4, -np.inf],
        [0, 0, 7, 0],
    ])
    original = source.copy(deep=True)

    result = comparison.normalize_tc(source)

    np.testing.assert_allclose(result, [
        [.25, 1, np.nan, .5],
        [0, 0, np.nan, 0],
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, .5, 1, np.nan],
        [0, 0, 1, 0],
    ], equal_nan=True)
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_zscore_tc_uses_finite_bins_and_population_sd_per_unit():
    source = pd.DataFrame([
        [0, 2, 4, np.nan],
        [100, 120, 140, np.nan],
        [np.inf, 0, 4, -np.inf],
    ])
    original = source.copy(deep=True)

    result = comparison.zscore_tc(source)

    expected = np.sqrt(1.5)
    np.testing.assert_allclose(result, [
        [-expected, 0, expected, np.nan],
        [-expected, 0, expected, np.nan],
        [np.nan, -1, 1, np.nan],
    ], equal_nan=True)
    np.testing.assert_allclose(np.nanmean(result, axis=1), 0, atol=1e-15)
    np.testing.assert_allclose(np.nanstd(result, axis=1, ddof=0), 1)
    pd.testing.assert_frame_equal(source, original)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_zscore_tc_marks_constant_single_bin_and_missing_curves_undefined():
    source = pd.DataFrame([
        [7, 7, 7, np.nan],
        [0, 0, 0, 0],
        [np.nan, 3, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
    ])

    assert comparison.zscore_tc(source).isna().all().all()


@pytest.mark.parametrize("transform", [comparison.normalize_tc, comparison.zscore_tc])
def test_transforms_preserve_peak_and_minimum_angles_for_nonconstant_rates(transform):
    source = pd.DataFrame(
        [[8, 1, 4, np.nan], [2, 7, 4, 1]],
        index=[11, 7], columns=[30, -150, -30, 150],
    )
    transformed = transform(source)
    affine_transformed = transform(3 * source + 5)

    for use_min in (False, True):
        expected = comparison.peak_angles(source, source.index, use_min=use_min)
        for result in (transformed, affine_transformed):
            np.testing.assert_array_equal(
                comparison.peak_angles(result, source.index, use_min=use_min), expected,
            )


def test_keyed_heatmap_plots_supplied_values_without_normalization():
    values = {7: np.array([2., np.nan, 8.]), 3: np.array([30., 10., 20.])}
    figure, axes = plot_keyed_heatmap(
        values, [3, 7], column_order=[2, 0, 1],
        xticks=[-180, 0, 180], xticklabels=[-180, 0, 180], show=False,
    )
    try:
        np.testing.assert_allclose(
            np.ma.filled(axes.images[0].get_array(), np.nan),
            [[20, 30, 10], [8, 2, np.nan]], equal_nan=True,
        )
        assert axes.images[0].get_clim() == (2, 30)
        assert figure.axes[1].get_ylabel() == "Response"
    finally:
        plt.close(figure)


def test_plot_profiles_preserves_signed_zscores_and_explicit_color_scale():
    source = pd.DataFrame(
        [[2, 8, 0, 4], [0, 1, 8, 3]], index=[7, 3], columns=[0, 90, -180, -90],
    )
    standardized = comparison.zscore_tc(source)
    with plt.style.context("dark_background"):
        figure, axes = comparison.plot_profiles(
            standardized, [3, 7], "RF", show=False,
            cmap="RdBu_r", vmin=-3, vmax=3, colorbar_label="Z-score (SD)",
        )
    try:
        np.testing.assert_allclose(
            axes.images[0].get_array(), standardized.loc[[3, 7], [-180, -90, 0, 90]],
        )
        assert axes.images[0].get_clim() == (-3, 3)
        assert axes.images[0].get_cmap().name == "RdBu_r"
        assert figure.axes[1].get_ylabel() == "Z-score (SD)"
        assert figure.get_facecolor() == (1, 1, 1, 1)
        assert axes.get_facecolor() == (1, 1, 1, 1)
        assert axes.xaxis.label.get_color() == "black"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("mode", ["native", "aligned", "sum"])
def test_prepared_pair_plots_preserve_values_and_display_options(mode, monkeypatch):
    angles = np.arange(-180., 180., 12.)
    reference = comparison.zscore_tc(pd.DataFrame(
        [2 + np.cos(np.deg2rad(angles + 60))], index=[7], columns=angles,
    ))
    matched = comparison.zscore_tc(pd.DataFrame(
        [4 + np.sin(np.deg2rad(angles))], index=[7], columns=angles,
    ))
    result = comparison.prepare_comparison(reference, matched, mode=mode)
    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("plot must consume the prepared comparison")
    monkeypatch.setattr(comparison, "paired_peak_angles", unexpected_calculation)
    monkeypatch.setattr(comparison, "align_profiles", unexpected_calculation)
    figures = comparison.plot_comparison_heatmaps(
        result, show=False,
        cmap="RdBu_r", vmin=-2, vmax=2, colorbar_label="Z-score (SD)",
    )
    try:
        for panel_index, ((figure, axes), source) in enumerate(zip(
            figures, (reference, matched), strict=True,
        )):
            expected = source.to_numpy()
            if mode != "native":
                # A reference peak at -60 degrees shifts alignment by +5 bins;
                # the matched sum panel shifts by -5 bins instead.
                shift = -5 if mode == "sum" and panel_index == 1 else 5
                expected = np.roll(expected, shift, axis=1)
                expected = np.c_[expected, expected[:, 0]]
            np.testing.assert_allclose(axes.images[0].get_array(), expected, atol=1e-14)
            assert axes.images[0].get_clim() == (-2, 2)
            assert axes.images[0].get_cmap().name == "RdBu_r"
            assert figure.axes[1].get_ylabel() == "Z-score (SD)"
    finally:
        for figure, _ in figures:
            plt.close(figure)
