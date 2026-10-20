import copy

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest

from EBB.spatial_cell_plotting import MAP_PLOTS, draw_map, prepare_spatial_profiles
from Utils.plotting import plot_egocentric_heatmap, plot_egocentric_polar


def test_curves_use_distance_sums_and_saved_hd_without_changing_loaded_data():
    # The full distance sum peaks at a different angle from the maximum-distance column.
    rates = np.array([[[9., 0., 0.], [6., 7., 0.], [0., 0., 0.], [0., 0., 0.]],
                      np.zeros((4, 3))])
    session = dict(unit_ids=np.array([7, 9]), egocentric_rate_maps=rates,
                   distance_edges=np.array([0., 8., 16., 24.]),
                   allocentric_tuning_curves=np.full((2, 60), 999.))
    counts = np.zeros((1, 180))
    counts[0, [0, 6]] = [30., 12.]
    hd_tc = dict(unit_id=[7], spike_counts=counts.tolist(),
                 occupancy_time_s=np.ones(180).tolist(),
                 angle_bin_edges_deg=np.linspace(0, 360, 181).tolist(),
                 unit_data=dict(hd_class=[3], rayleigh_p=[1.], shuffle_p=[1.]))
    original_session, original_hd = copy.deepcopy(session), copy.deepcopy(hd_tc)

    prepared = prepare_spatial_profiles(session, hd_tc)

    np.testing.assert_array_equal(prepared["egocentric_tuning_profiles"][0],
                                  [[9., 13., 0., 0.], [9., 6., 0., 0.],
                                   [0., 7., 0., 0.], [0., 0., 0., 0.]])
    np.testing.assert_array_equal(prepared["allocentric_tuning_curves"][0],
                                  np.r_[5., 2., np.zeros(28)])
    assert np.isnan(prepared["allocentric_tuning_curves"][1]).all()
    np.testing.assert_array_equal(prepared["hd_tc_present"], [True, False])
    np.testing.assert_array_equal(prepared["hd_theta_edges"], np.arange(0, 361, 12))
    for key, value in original_session.items():
        np.testing.assert_array_equal(session[key], value)
    assert hd_tc == original_hd

    figure, axis = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        draw_map(axis, dict(theta_edges=np.linspace(0, 360, 5), distance_edges=session["distance_edges"],
                            egocentric_tuning_profiles=prepared["egocentric_tuning_profiles"][0]),
                 next(plot for plot in MAP_PLOTS if plot.kind == "ego_tuning"))
        assert [line.get_label() for line in axis.lines] == ["Full map", "≤8 cm", "8–16 cm", ">16 cm"]
        np.testing.assert_array_equal(axis.lines[0].get_ydata(), [9., 13., 0., 0., 9.])
        assert axis.get_theta_direction() == -1
        assert [label.get_text() for label in axis.get_xticklabels()] == ["0°", "90°", "±180°", "−90°"]
    finally:
        plt.close(figure)


def test_signed_ebc_axes_preserve_left_right_values_and_native_inputs():
    edges = np.arange(0., 361., 90.)
    # Four distinct native bearing sectors; 270..360 is negative (left of forward).
    rates = np.arange(1., 5.)[:, None]
    figure = plt.figure()
    heatmap = figure.add_subplot(121)
    polar = figure.add_subplot(122, projection="polar")
    try:
        image = plot_egocentric_heatmap(heatmap, [0., 1.], edges, rates, "EBC", "Hz")
        np.testing.assert_array_equal(image.get_array().ravel(), [3., 4., 1., 2.])
        assert heatmap.get_ylim() == (-180., 180.)
        plot_egocentric_polar(polar, edges, np.array([0., 1.]), rates)
        assert [label.get_text() for label in polar.get_xticklabels()] == ["0°", "90°", "±180°", "−90°"]
        figure.canvas.draw()
        center = polar.transData.transform((0., 0.))
        assert polar.transData.transform((-np.pi / 2, .5))[0] < center[0]
        assert polar.transData.transform((np.pi / 2, .5))[0] > center[0]
        np.testing.assert_array_equal(rates[:, 0], [1., 2., 3., 4.])
        np.testing.assert_array_equal(edges, [0., 90., 180., 270., 360.])
    finally:
        plt.close(figure)


def test_missing_saved_hd_is_marked_without_drawing_a_replacement_curve():
    figure, axis = plt.subplots(subplot_kw={"projection": "polar"})
    try:
        draw_map(axis, {"hd_tc_present": False},
                 next(plot for plot in MAP_PLOTS if plot.kind == "hd_tuning"))
        assert not axis.lines
        assert axis.texts[0].get_text() == "No saved HD TC for this unit"
    finally:
        plt.close(figure)


@pytest.mark.parametrize("polar", [False, True])
def test_zero_spike_maps_do_not_show_negative_rates_or_counts(polar):
    values = np.zeros((4, 2))
    values[0, 0] = np.nan
    original = values.copy()
    figure, axis = plt.subplots(subplot_kw={"projection": "polar" if polar else None})
    try:
        if polar:
            image = plot_egocentric_polar(axis, np.arange(0., 361., 90.), np.array([0., 1., 2.]), values)
        else:
            image = plot_egocentric_heatmap(axis, [0., 1., 2.], np.arange(0., 361., 90.), values, "EBC", "Hz")
        assert image.get_clim() == (0., 1.)
        np.testing.assert_array_equal(image.colorbar.get_ticks(), [0.])
        np.testing.assert_array_equal(values, original)
    finally:
        plt.close(figure)
