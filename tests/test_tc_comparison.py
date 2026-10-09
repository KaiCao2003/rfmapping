import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba

from Utils import tc_comparison


@pytest.fixture
def paired_curves():
    hd = pd.DataFrame(
        [
            [.1, .2, 1., .4, .3, .1],
            [.2, .1, .1, .1, 1., .1],
            [1., .2, .1, .1, .2, .1],
            [np.nan] * 6,
            [.1, .2, .4, 1., .3, .1],
            [.2, 1., .1, .3, .2, .1],
        ],
        index=["u3", "hd-only", "u1", "missing", "u4", "u2"],
        columns=[0., 60., 120., 180., 240., 300.],
    )
    rf = pd.DataFrame(
        [
            [.2, 1., .2, .1, .3, .1],
            [.1, .2, .3, .2, 1., .1],
            [.2, .1, .3, .1, .2, 1.],
            [.1, .2, .3, .2, .1, 1.],
            [.2, .3, 1., .1, .2, .1],
            [.2, .3, .1, 1., .2, .1],
        ],
        index=["u2", "u4", "rf-only", "missing", "u3", "u1"],
        columns=[-150., -90., -30., 30., 90., 150.],
    )
    hd.attrs.update(label="Fixture HD", range=[0, 360], response_units="normalized")
    rf.attrs.update(label="Fixture RF", range=[-180, 180], response_units="normalized")
    return hd, rf


def test_statistics_match_unit_identity_and_keep_native_peak_angles(paired_curves):
    hd, rf = paired_curves
    original_hd, original_rf = hd.copy(deep=True), rf.copy(deep=True)

    result = tc_comparison.calculate_tc_statistics(
        hd, rf, bins=6, n_permutations=19, random_seed=7,
    )

    assert result["peaks"].index.tolist() == ["u1", "u2", "u3", "u4"]
    np.testing.assert_array_equal(
        result["peaks"][["reference_peak_deg", "matched_peak_deg"]],
        [[0., 30.], [60., -90.], [120., -30.], [180., 90.]],
    )
    # Peak sums are 30, 330, 90, and 270 degrees, with empty bins retained.
    np.testing.assert_array_equal(result["counts"], [1, 1, 0, 0, 1, 1])
    np.testing.assert_allclose(np.rad2deg(result["edges"]), np.arange(0., 361., 60.))
    np.testing.assert_allclose(result["rayleigh_score"], 1.5)
    np.testing.assert_allclose(result["rayleigh_p"], np.exp(-.75))
    assert result["direction"]["same_unit"]["unit_count"] == 4
    assert result["direction"]["pairwise"]["unit_count"] == 4
    for statistics in result["direction"].values():
        assert 0 < statistics["permutation_p"] <= 1

    native = result["comparisons"][0]
    assert native["reference"].loc["missing"].isna().all()
    pd.testing.assert_frame_equal(native["reference"], hd.loc[native["order"]])
    pd.testing.assert_frame_equal(native["matched"], rf.loc[native["order"]])
    pd.testing.assert_frame_equal(hd, original_hd)
    pd.testing.assert_frame_equal(rf, original_rf)


def test_peak_statistics_are_invariant_to_positive_per_unit_scaling(paired_curves):
    hd, rf = paired_curves
    original = tc_comparison.calculate_tc_statistics(
        hd, rf, bins=6, n_permutations=19, random_seed=7,
    )
    scaled = tc_comparison.calculate_tc_statistics(
        hd.mul([3., 4., 2., 6., 1.5, 9.], axis=0),
        rf.mul([2., 7., 3., 4., 8., 5.], axis=0),
        bins=6, n_permutations=19, random_seed=7,
    )

    pd.testing.assert_frame_equal(scaled["peaks"], original["peaks"])
    np.testing.assert_array_equal(scaled["counts"], original["counts"])
    np.testing.assert_allclose(scaled["rayleigh_score"], original["rayleigh_score"])
    assert scaled["direction"] == original["direction"]


def test_two_measured_pairs_keep_histogram_without_direction_test(paired_curves):
    hd, rf = paired_curves
    result = tc_comparison.calculate_tc_statistics(
        hd.loc[["u1", "missing", "u2"]], rf, bins=6, n_permutations=19,
    )

    assert result["peaks"].index.tolist() == ["u1", "u2"]
    assert result["direction"] is None
    np.testing.assert_array_equal(result["counts"], [1, 0, 0, 0, 0, 1])


def test_plot_uses_prepared_statistics_and_renders_opaque_white_figures(
    paired_curves, monkeypatch,
):
    hd, rf = paired_curves
    statistics = tc_comparison.calculate_tc_statistics(
        hd, rf, bins=6, n_permutations=19, random_seed=7,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("plotting must consume the supplied statistics")

    monkeypatch.setattr(tc_comparison, "calculate_tc_statistics", unexpected)
    monkeypatch.setattr(tc_comparison, "compare_direction_angles", unexpected)
    monkeypatch.setattr(tc_comparison, "rayleigh_test", unexpected)
    monkeypatch.setattr(tc_comparison, "display", unexpected)
    figures = []
    try:
        with plt.style.context("dark_background"):
            figures = tc_comparison.plot_tc_comparison(
                hd, rf, statistics, figsize=(3, 3), show=False,
            )
        polar = [(figure, axis) for figure, axis in figures if axis.name == "polar"]
        assert len(polar) == 1
        np.testing.assert_array_equal(
            [bar.get_height() for bar in polar[0][1].patches], statistics["counts"],
        )
        assert "n = 4" in polar[0][1].get_title()
        assert any(axis.images for _, axis in figures)
        assert any("Same-unit" in axis.get_title() for _, axis in figures)
        assert any("pair distances" in axis.get_title() for _, axis in figures)
        for figure, _ in figures:
            assert figure.get_facecolor() == (1, 1, 1, 1)
            for axis in figure.axes:
                assert axis.get_facecolor() == (1, 1, 1, 1)
                assert to_rgba(axis.xaxis.label.get_color()) == (0, 0, 0, 1)
                assert to_rgba(axis.yaxis.label.get_color()) == (0, 0, 0, 1)
            canvas = FigureCanvasAgg(figure)
            canvas.draw()
            pixels = np.asarray(canvas.buffer_rgba())
            assert (pixels[..., 3] == 255).all()
            assert (pixels[..., :3] < 245).any()
    finally:
        for figure, _ in figures:
            plt.close(figure)
