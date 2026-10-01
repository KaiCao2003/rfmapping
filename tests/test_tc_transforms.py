import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
from Utils.plotting import plot_keyed_heatmap


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


@pytest.mark.parametrize("plot", [comparison.plot_sort, comparison.plot_align, comparison.plot_sum])
def test_pair_plot_wrappers_forward_display_options_without_rescaling(plot):
    angles = np.arange(-180., 180., 12.)
    reference = comparison.zscore_tc(pd.DataFrame(
        [2 + np.cos(np.deg2rad(angles + 60))], index=[7], columns=angles,
    ))
    matched = comparison.zscore_tc(pd.DataFrame(
        [4 + np.sin(np.deg2rad(angles))], index=[7], columns=angles,
    ))
    result = plot(
        reference, matched, show=False,
        cmap="RdBu_r", vmin=-2, vmax=2, colorbar_label="Z-score (SD)",
    )
    try:
        for panel_index, ((figure, axes), source) in enumerate(zip(
            result["figures"], (reference, matched), strict=True,
        )):
            expected = source.to_numpy()
            if plot is not comparison.plot_sort:
                # A reference peak at -60 degrees shifts alignment by +5 bins;
                # the matched sum panel shifts by -5 bins instead.
                shift = -5 if plot is comparison.plot_sum and panel_index == 1 else 5
                expected = np.roll(expected, shift, axis=1)
                expected = np.c_[expected, expected[:, 0]]
            np.testing.assert_allclose(axes.images[0].get_array(), expected, atol=1e-14)
            assert axes.images[0].get_clim() == (-2, 2)
            assert axes.images[0].get_cmap().name == "RdBu_r"
            assert figure.axes[1].get_ylabel() == "Z-score (SD)"
    finally:
        for figure, _ in result["figures"]:
            plt.close(figure)
