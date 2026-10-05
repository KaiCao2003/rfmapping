import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from Utils.plotting import plot_hd_tuning_curve, plot_phase_bar, plot_head_turn_bar
from Utils.statistic_utils import (
    smooth_tuning_curve, summarize_phase_intervals, summarize_head_turns_from_intervals,
)


def test_hd_plot_preserves_missing_bins_and_smoothing_uses_the_intact_grid():
    curve = pd.Series([0., 4., np.nan, 2., 0., 1.], index=np.arange(0., 360., 60.), name=7)
    rendered = plot_hd_tuning_curve(curve, 7)
    pd.testing.assert_series_equal(rendered["curve"], curve)
    np.testing.assert_allclose(rendered["line_ax"].lines[0].get_ydata(), curve, equal_nan=True)
    smoothed = smooth_tuning_curve(curve, sigma=.75)
    finite = curve.notna().to_numpy()
    expected = gaussian_filter1d(curve.fillna(0).to_numpy(), .75, mode="wrap")
    expected /= gaussian_filter1d(finite.astype(float), .75, mode="wrap")
    expected[~finite] = np.nan
    np.testing.assert_allclose(smoothed, expected, equal_nan=True)
    assert smoothed.index.equals(curve.index)
    assert curve.iloc[1] == 4.
    plt.close(rendered["line_fig"])


def test_phase_summary_uses_inclusive_intervals_and_renderer_keeps_supplied_means(monkeypatch):
    summary = summarize_phase_intervals(["first", "second"], [[(0, 2), (3, 4)], [(5, 5)]],
                                        [1., np.nan, 3., 6., 8., 9.])
    np.testing.assert_array_equal(summary["phase_trial_values"][0], [2., 7.])
    np.testing.assert_array_equal(summary["phase_means"], [4.5, 9.])
    summary["phase_means"] = np.array([20., 30.])
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, axis = plot_phase_bar(summary)
    np.testing.assert_array_equal([bar.get_height() for bar in axis.patches], [20., 30.])
    np.testing.assert_array_equal(summary["phase_trial_values"][0], [2., 7.])
    plt.close(fig)


def test_head_turn_events_are_summarized_before_rendering(monkeypatch):
    summary = summarize_head_turns_from_intervals(
        ["phase"], [[(0, 4), (5, 9)]],
        [{"start_frame_actual": 1, "end_frame": 2},
         {"start_frame_actual": 7, "end_frame": 8, "is_skipped": True}],
        fps=5, value_mode="rate",
    )
    np.testing.assert_array_equal(summary["phase_trial_counts"][0], [1, 0])
    np.testing.assert_array_equal(summary["phase_trial_values"][0], [1., 0.])
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, axis = plot_head_turn_bar(summary)
    assert axis.get_ylabel() == "Head turn count/s"
    assert axis.patches[0].get_height() == .5
    assert "line_ax" not in summary
    plt.close(fig)


def test_changed_renderers_keep_white_canvas_under_dark_global_style(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    with plt.style.context("dark_background"):
        rendered = plot_hd_tuning_curve(pd.Series([1., 2.], index=[0., 180.]), 7)
        figure, axis = plot_phase_bar({"labels": ["phase"], "phase_trial_values": [np.array([1.])],
                                       "phase_means": np.array([1.]), "phase_stds": np.array([0.])})
        for fig, ax in [(rendered["line_fig"], rendered["line_ax"]), (figure, axis)]:
            assert fig.get_facecolor() == (1., 1., 1., 1.)
            assert ax.get_facecolor() == (1., 1., 1., 1.)
            assert ax.xaxis.label.get_color() == "black"
            plt.close(fig)
        assert plt.rcParams["axes.facecolor"] == "black"
