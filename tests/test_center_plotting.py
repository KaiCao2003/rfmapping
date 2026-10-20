"""Saved CB/CD plots retain native coordinates, missing bins, and class state."""

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from EBB.center_cell_plotting import plot_center_population, plot_center_tuning
from Utils.theta_plotting import plot_lfp_reference, plot_theta_unit


@pytest.fixture
def center_result():
    rows, tuning = [], {}
    for unit in (7, 11):
        row = {"unit_id": unit, "cd_slope_hz_per_cm": .01, "cd_intercept_hz": 0.}
        tuning[unit] = {}
        for feature, edges in (
            ("cb", np.arange(5) * 90.),
            ("cd", np.array([0., 100., 250., 420., 600.])),
        ):
            rates = np.array([0., 2., np.nan, 5.]) * (unit / 7)
            tuning[unit][feature] = {
                "edges": edges, "centers": (edges[:-1] + edges[1:]) / 2,
                "rate_hz": rates.copy(), "smoothed_rate_hz": rates.copy(),
                "occupancy_s": np.array([1., 1., 0., 1.]),
            }
            row.update({
                f"{feature}_status": "classified" if unit == 7 else "insufficient_spikes",
                f"{feature}_is_cell": True if unit == 7 else pd.NA,
                f"{feature}_strength_threshold": .31,
                f"{feature}_stability_threshold": .41,
                f"{feature}_stability": .65,
                "cd_r2" if feature == "cd" else "cb_mvl": .55,
            })
        rows.append(row)
    return {"summary": pd.DataFrame(rows).set_index("unit_id"), "tuning": tuning}


def assert_white(figure):
    assert figure.get_facecolor() == (1., 1., 1., 1.)
    for axis in figure.axes:
        assert axis.get_facecolor() == (1., 1., 1., 1.)
        assert axis.xaxis.label.get_color() == "black"
        assert axis.yaxis.label.get_color() == "black"


@pytest.mark.parametrize("style", ["line", "polar", "heatmap", "donut"])
def test_cb_styles_keep_zero_missing_native_bearing_and_saved_thresholds(center_result, style):
    with plt.style.context("dark_background"):
        figure = plot_center_tuning(center_result, 7, plot_style=style)
        axis = figure.axes[0]
        assert_white(figure)
        assert "CB: positive" in axis.get_title()
        assert "threshold 0.31" in axis.get_title()
        assert "threshold 0.41" in axis.get_title()
        expected = center_result["tuning"][7]["cb"]["smoothed_rate_hz"]
        if style in ("line", "polar"):
            values = axis.lines[1].get_ydata()
            if style == "polar":
                np.testing.assert_allclose(axis.lines[1].get_xdata()[:-1], np.deg2rad([45, 135, 225, 315]))
                assert values[-1] == values[0]
                values = values[:-1]
            np.testing.assert_allclose(values, expected, equal_nan=True)
        else:
            values = axis.collections[0].get_array().ravel()
            np.testing.assert_allclose(values.filled(np.nan), expected, equal_nan=True)
            assert values[0] == 0.
            assert values.mask[2]
        if style in ("polar", "donut"):
            assert axis.get_theta_direction() == -1
            assert axis.get_theta_offset() == pytest.approx(np.pi / 2)
        if style == "polar":
            assert axis.get_ylabel() == ""
            assert axis.get_xlabel() == "Radius: firing rate (Hz)"
        if style == "donut":
            assert not len(axis.get_yticks())
            assert "cm" not in axis.get_xlabel()
        assert plt.rcParams["axes.facecolor"] == "black"
        plt.close(figure)


@pytest.mark.parametrize("style", ["line", "heatmap"])
def test_cd_plot_retains_native_cm_and_never_wraps_distance(center_result, style):
    figure = plot_center_tuning(center_result, 7, feature="cd", plot_style=style)
    axis = figure.axes[0]
    curve = center_result["tuning"][7]["cd"]
    assert axis.name == "rectilinear"
    assert axis.get_xlim() == (0., 600.)
    assert axis.get_xlabel() == "Center distance (cm)"
    if style == "line":
        np.testing.assert_array_equal(axis.lines[0].get_xdata(), curve["centers"])
        fit = axis.lines[2].get_ydata()
        assert fit[0] == .5
        assert np.isnan(fit[2])
    else:
        np.testing.assert_array_equal(axis.collections[0].get_coordinates()[0, :, 0], curve["edges"])
    plt.close(figure)
    with pytest.raises(ValueError, match="CD plot_style"):
        plot_center_tuning(center_result, 7, feature="cd", plot_style="polar")


def test_population_keeps_native_edges_requested_unit_order_and_unassessed_state(center_result):
    with plt.style.context("dark_background"):
        figure = plot_center_population(center_result, feature="cd", unit_ids=[11, 7])
        assert_white(figure)
        axis = figure.axes[0]
        mesh = axis.collections[0]
        np.testing.assert_array_equal(mesh.get_coordinates()[0, :, 0], [0, 100, 250, 420, 600])
        expected = np.stack([center_result["tuning"][unit]["cd"]["smoothed_rate_hz"] for unit in (11, 7)])
        np.testing.assert_allclose(mesh.get_array().reshape(expected.shape).filled(np.nan), expected, equal_nan=True)
        assert [label.get_text() for label in axis.get_yticklabels()] == ["11: unassessed", "7: positive"]
        assert axis.get_xlim() == (0., 600.)
        plt.close(figure)


def test_empty_population_and_silent_unit_remain_valid_displays(center_result):
    figure = plot_center_population(center_result, unit_ids=[])
    assert "No units selected" in figure.axes[0].texts[0].get_text()
    plt.close(figure)
    center_result["tuning"][11]["cb"]["smoothed_rate_hz"][:] = [0, 0, np.nan, 0]
    figure = plot_center_tuning(center_result, 11, plot_style="donut")
    assert figure.axes[0].collections[0].get_clim() == (0., 1.)
    assert "unassessed" in figure.axes[0].get_title()
    assert "insufficient_spikes" in figure.axes[0].get_title()
    plt.close(figure)


def test_theta_plotting_is_separate_and_retains_compatibility_imports():
    from EBB.center_cell_plotting import plot_theta_unit as compatible_theta_plot
    from EBB.center_cell_plotting import plot_lfp_reference as compatible_lfp_plot

    assert compatible_theta_plot is plot_theta_unit
    assert compatible_lfp_plot is plot_lfp_reference
    references = {"session": {
        "spectrum": {"frequency_hz": np.array([1., 5., 10., 40.]),
                     "psd_uv2_per_hz": np.array([1., 4., 3., 1.])},
        "trace": {"time_from_midpoint_s": np.array([-1., 0., 1.]),
                  "voltage_uv": np.array([1., 3., 2.]),
                  "filtered_uv": np.array([0., 2., 1.]),
                  "amplitude_uv": np.array([1., 2., 1.]), "midpoint_s": 30.},
        "summary": {"power_4_11_over_1_40": .6},
    }}
    with plt.style.context("dark_background"):
        for figure in (plot_theta_unit({}), plot_lfp_reference(references, channel=12)):
            assert_white(figure)
            plt.close(figure)


@pytest.mark.parametrize("final_candidate, expected", [(pd.NA, "unassessed"), (False, "not a candidate")])
def test_theta_plot_honors_final_saved_classification_over_raw_candidate(final_candidate, expected):
    precession = {
        "acg_lags_cycles": np.array([0., 1., 2.]), "acg_counts": np.array([0., 4., 2.]),
        "frequency_per_cycle": np.array([.5, 1., 1.25, 1.5, 1.9]),
        "spectrum": np.array([1., 1., 5., 1., 1.]), "peak_frequency_per_cycle": 1.25,
        "precession_candidate": True, "eligible": True, "phase_precession_index": .7,
        "parameters": {"reference_bands": [[.4, .8], [1.6, 1.9]], "spectrum_convention": "power"},
    }
    rhythmicity = {
        "acg_lags_s": np.array([0., .1, .2]), "acg_counts": np.array([0., 4., 2.]),
        "frequency_hz": np.array([1., 8., 12.]), "spectrum": np.array([1., 5., 1.]),
        "peak_frequency_hz": 8., "theta_index": .8,
        "parameters": {"theta_band_hz": [4., 11.], "reference_band_hz": [1., 40.],
                       "spectrum_convention": "power"},
    }
    locking = {
        "phase_bin_edges_rad": np.linspace(-np.pi, np.pi, 4),
        "phase_counts": np.array([1., 4., 2.]), "phase_occupancy_s": np.ones(3),
        "mean_vector_length": .5, "rayleigh_p": .01, "preferred_phase_rad": 0.,
    }
    summary = pd.Series({
        "precession_candidate": final_candidate, "precession_status": "computed",
        "is_theta_rhythmic": pd.NA, "rhythmicity_status": "computed",
        "is_phase_locked": pd.NA, "phase_locking_status": "insufficient_phase_spikes",
        "theta_status": "descriptive_only_insufficient_spatial_coverage",
    })
    figure = plot_theta_unit(
        {"rhythmicity": rhythmicity, "phase_locking": locking, "phase_precession": precession},
        summary=summary,
    )
    assert f"Precession test · {expected}" in figure.axes[5].get_title()
    assert "unassessed · computed" in figure.axes[1].get_title()
    assert "unassessed · insufficient phase spikes" in figure.axes[2].get_title()
    assert "insufficient spatial coverage" in figure._suptitle.get_text()
    plt.close(figure)


def test_lfp_reference_keeps_psd_and_explains_missing_movement_excerpt():
    references = {"session": {
        "spectrum": {"frequency_hz": np.array([1., 5., 10., 40.]),
                     "psd_uv2_per_hz": np.array([1., 4., 3., 1.])},
        "trace": {"status": "no_movement", "midpoint_s": np.nan,
                  "time_from_midpoint_s": np.array([]), "voltage_uv": np.array([]),
                  "filtered_uv": np.array([]), "amplitude_uv": np.array([])},
        "summary": {"power_4_11_over_1_40": .6},
    }}
    figure = plot_lfp_reference(references, channel=12)
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[1].get_title() == "Channel 12"
    assert "No movement exposure" in figure.axes[1].texts[0].get_text()
    assert not figure.axes[1].lines
    plt.close(figure)
