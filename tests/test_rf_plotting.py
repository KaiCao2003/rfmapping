"""Plotting consumes explicit unit and population arrays without analysis."""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import direction_comparison as comparison
from Utils import rf_plotting
from Utils.plotting import plot_tuning_curves_for_cluster
from Utils.rfmap import plot_1d_rfmap
from Utils.rflocate import RFMapList, RFResult, asrfmap


def test_rf_unit_plot_uses_supplied_horizontal_response(tmp_path):
    data = {
        "unit_id": 17, "response": np.array([[1., 2., 3.], [4., 5., 6.]]),
        "response_1d": np.array([11., 22., 33.]),
        "mask_2d": np.array([[0, 1, 0], [0, 1, 0]], dtype=bool),
        "center_2d": np.array([[0, 0, 0], [0, 1, 0]], dtype=bool),
        "mask_1d": np.array([0, 1, 0], dtype=bool),
        "center_1d": np.array([0, 1, 0], dtype=bool),
    }
    with plt.style.context("dark_background"):
        figure, axes = rf_plotting.plot_rf_unit(data, output_path=tmp_path / "unit", show=False)
    np.testing.assert_array_equal(axes[1, 0].lines[0].get_ydata(), data["response_1d"])
    np.testing.assert_array_equal(axes[0, 0].images[0].get_array(), data["response"])
    assert figure.get_facecolor() == (1, 1, 1, 1)
    assert all(axis.get_facecolor() == (1, 1, 1, 1) for axis in figure.axes)
    assert (tmp_path / "unit.svg").is_file()


def test_unit_plot_data_sums_whole_rf_rows_without_detection(monkeypatch):
    response = np.array([[1., 2.], [10., 20.], [3., 4.]])
    rf_map = asrfmap(response)
    mask = np.array([[[0, 0], [1, 0], [0, 0]]], dtype=np.uint8)
    detected = RFResult(
        mask_2d=mask, center_2d=mask, unit_ids=[0], manifest={}, cache_key="saved-2d",
    )
    analysis = {
        "summed": RFMapList([rf_map], rf_map.source_path), "result_2d": detected,
        "mask_2d": detected.mask_2d, "center_2d": detected.center_2d,
        "mask_1d": detected.project("x")[:, 0],
        "center_1d": detected.project("x", center_only=True)[:, 0],
    }

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("plot preparation must use the supplied 2-D RF")

    monkeypatch.setattr("Utils.rflocate._detector.detect_rf", unexpected_detection)
    all_rows = rf_plotting.rf_unit_plot_data(analysis, 0, rf_only=False)
    rf_rows = rf_plotting.rf_unit_plot_data(analysis, 0, rf_only=True)
    np.testing.assert_array_equal(all_rows["response_1d"], [14., 26.])
    np.testing.assert_array_equal(rf_rows["response_1d"], [10., 20.])
    np.testing.assert_array_equal(rf_rows["response"], response)
    np.testing.assert_array_equal(rf_rows["mask_1d"], [1, 0])


def test_population_plot_does_not_count_or_smooth(monkeypatch):
    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("population plots must consume supplied arrays")
    monkeypatch.setattr(rf_plotting, "rf_population_counts", unexpected_calculation)
    counts = {
        "mask_2d": np.array([[0.25, 1.5, 0.75]]),
        "center_2d": np.array([[0., 1., 0.]]),
        "mask_1d": np.array([0.5, 3., 0.5]),
        "center_1d": np.array([0., 1., 0.]),
    }
    figures = rf_plotting.plot_rf_population({"chosen": counts}, show=False)
    assert set(figures) == {"chosen"}
    _, axes = figures["chosen"]
    np.testing.assert_array_equal(axes[0, 0].images[0].get_array(), counts["mask_2d"])
    np.testing.assert_array_equal(axes[1, 0].lines[0].get_ydata(), counts["mask_1d"])


def test_peak_sum_plot_uses_explicit_counts_and_statistics(monkeypatch):
    table = pd.DataFrame({"peak_sum_deg": [-90., 0., 0., 90.]})
    summary = comparison.peak_sum_statistics(table)
    assert summary["counts"].sum() == 4
    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("peak-sum plots must not run statistical tests")
    monkeypatch.setattr(comparison, "rayleigh_uniformity", unexpected_calculation)
    figure = comparison.plot_peak_direction_sums(summary)
    assert "n = 4" in figure.axes[0].texts[0].get_text()
    plt.close(figure)


@pytest.mark.parametrize("renderer", [plot_tuning_curves_for_cluster, plot_1d_rfmap])
def test_shared_rf_renderer_preserves_values_and_white_background(renderer):
    responses = np.array([[2., 8., 4.], [5., 15., 10.]])
    with plt.style.context("dark_background"):
        figure, axis = renderer(responses, [7, 11], isHeatmap=True)
    np.testing.assert_array_equal(axis.images[0].get_array(), responses)
    assert figure.get_facecolor() == (1, 1, 1, 1)
    assert all(ax.get_facecolor() == (1, 1, 1, 1) for ax in figure.axes)
    assert axis.xaxis.label.get_color() == "black"
    assert axis.yaxis.label.get_color() == "black"
    plt.close(figure)
