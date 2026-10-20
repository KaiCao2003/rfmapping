"""ADN scatter plots retain shared shank coordinates and every RF center."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from Utils import retinotopography_scatter as scatter


MICE = ["m14", "m15", "m19", "m20", "m21"]


@pytest.fixture
def adn_data():
    units = pd.DataFrame([
        ("m14", "260609", "A", 7, 0, 300., 10.),
        ("m14", "260609", "A", 8, 0, 225., 350.),
        ("m14", "260609", "A", 9, 0, 900., 30.),
        ("m14", "260609", "A", 10, 1, 800., 20.),
        ("m14", "260609", "A", 11, 1, 780., 400.),
        ("m14", "260609", "B", 7, 0, 1400., 270.),
        ("m14", "260609", "B", 8, 0, 1200., 90.),
        ("m15", "260630", "A", 7, 0, 1000., 50.),
        ("m15", "260630", "A", 8, 0, 750., 60.),
        ("m15", "260701", "A", 7, 0, 3000., 80.),
        ("m15", "260701", "A", 8, 0, 2700., 90.),
        ("m19", "260827", "A", 7, 0, np.nan, 20.),
        ("m20", "260921", "A", 7, 0, 10., np.nan),
    ], columns=scatter.IDENTITY + ["shank", "y_um", "hd_preferred_deg"])
    selection = units.loc[units.unit_id.ne(9), scatter.IDENTITY].copy()
    centers = pd.DataFrame([
        ("m14", "260609", "A", 8, 0., -20.),
        ("m14", "260609", "A", 8, 0., 20.),
        ("m14", "260609", "B", 8, 180., 0.),
        ("m15", "260630", "A", 8, -100., 0.),
        ("m15", "260701", "A", 8, 90., 0.),
        ("m14", "260609", "A", 9, 60., 0.),
    ], columns=scatter.IDENTITY + ["rf_x_deg", "rf_y_deg"])
    return units, selection, centers


@pytest.fixture
def adn_data_with_deeper_unit(adn_data):
    units, selection, centers = adn_data
    bottom = units.iloc[[0]].assign(unit_id=12, y_um=150.)
    return (pd.concat([units, bottom], ignore_index=True),
            pd.concat([selection, bottom[scatter.IDENTITY]], ignore_index=True),
            centers)


def test_adn_depth_uses_each_selected_shank_top_without_scaling(adn_data):
    units, selection, _ = adn_data
    before = units.copy(deep=True)
    selected = scatter.select_adn_depth(units, selection)

    assert len(selected) == 10
    assert 9 not in selected.unit_id.to_numpy()
    np.testing.assert_allclose(selected.depth_um, [0, 75, 0, 20, 0, 200, 0, 250, 0, 300])
    np.testing.assert_allclose(selected.topmost_hd_y_um,
                               [300, 300, 800, 800, 1400, 1400, 1000, 1000, 3000, 3000])
    assert len(selected.groupby(scatter.SHANK)) == 5
    np.testing.assert_allclose(selected.groupby(scatter.SHANK).depth_um.min(), 0)
    pd.testing.assert_frame_equal(units, before)


def test_hd_and_all_rf_centers_share_origin_even_when_top_cell_has_no_rf(adn_data):
    units, selection, centers = adn_data
    selected = scatter.select_adn_depth(units, selection)
    selected_before, centers_before = selected.copy(deep=True), centers.copy(deep=True)
    hd = scatter.prepare_depth_angles(selected, centers, kind="hd")
    rf = scatter.prepare_depth_angles(selected, centers, kind="rf")

    assert len(hd) == 10
    assert hd[scatter.IDENTITY].duplicated().sum() == 0
    np.testing.assert_allclose(hd.angle_deg, [10, 350, 20, 40, 270, 90, 50, 60, 80, 90])
    assert len(rf) == 5
    assert len(rf[scatter.IDENTITY].drop_duplicates()) == 4
    np.testing.assert_allclose(rf.depth_um, [75, 75, 200, 250, 300])
    np.testing.assert_allclose(rf.angle_deg, [0, 0, -180, -100, 90])
    np.testing.assert_allclose(rf.rf_y_deg, [-20, 20, 0, 0, 0])
    assert (rf.depth_um > 0).all()
    pd.testing.assert_frame_equal(selected, selected_before)
    pd.testing.assert_frame_equal(centers, centers_before)


@pytest.mark.parametrize("group_key, other_value", [
    ("mouse", "m15"), ("date", "260610"), ("probe", "B"), ("shank", 1),
])
def test_normalized_depth_uses_each_selected_shank_span_without_mutation(
    adn_data, group_key, other_value,
):
    units, selection, _ = adn_data
    selected = scatter.select_adn_depth(units.iloc[:2], selection)
    other_shank = selected.copy().assign(unit_id=[17, 18], y_um=[300., 150.],
                                         depth_um=[0., 150.])
    other_shank[group_key] = other_value
    selected = pd.concat([selected, other_shank], ignore_index=True)
    before = selected.copy(deep=True)
    normalized = scatter.normalize_shank_depth(selected)

    np.testing.assert_allclose(normalized.depth_fraction, [0, 1, 0, 1])
    np.testing.assert_allclose(normalized.groupby(scatter.SHANK).depth_fraction.min(), 0)
    np.testing.assert_allclose(normalized.groupby(scatter.SHANK).depth_fraction.max(), 1)
    pd.testing.assert_frame_equal(normalized.drop(columns="depth_fraction"), before)
    pd.testing.assert_frame_equal(selected, before)


def test_rf_fraction_shares_hd_span_when_top_and_bottom_have_no_rf(adn_data_with_deeper_unit):
    units, selection, centers = adn_data_with_deeper_unit
    normalized = scatter.normalize_shank_depth(scatter.select_adn_depth(units, selection))
    before, centers_before = normalized.copy(deep=True), centers.copy(deep=True)
    hd = scatter.prepare_depth_angles(normalized, centers, kind="hd",
                                      depth_column="depth_fraction")
    rf = scatter.prepare_depth_angles(normalized, centers, kind="rf",
                                      depth_column="depth_fraction")

    group = hd.loc[hd.mouse.eq("m14") & hd.probe.eq("A") & hd.shank.eq(0)]
    np.testing.assert_array_equal(group.unit_id, [7, 8, 12])
    np.testing.assert_allclose(group.depth_fraction, [0, .5, 1])
    np.testing.assert_allclose(rf.depth_fraction, [.5, .5, 1, 1, 1])
    np.testing.assert_allclose(rf.rf_y_deg, [-20, 20, 0, 0, 0])
    assert len(rf[scatter.IDENTITY].drop_duplicates()) == 4
    pd.testing.assert_frame_equal(normalized, before)
    pd.testing.assert_frame_equal(centers, centers_before)


@pytest.mark.parametrize("count", [1, 2])
def test_zero_span_shank_stays_at_zero(adn_data, count):
    units, selection, _ = adn_data
    coincident = units.iloc[:count].assign(y_um=300.)
    selected = scatter.select_adn_depth(coincident, selection)
    normalized = scatter.normalize_shank_depth(selected)

    np.testing.assert_array_equal(normalized.depth_fraction, np.zeros(count))


@pytest.mark.parametrize("kind, expected_points, expected_units", [
    ("hd", 10, 10),
    ("rf", 5, 4),
])
def test_mouse_and_pooled_panels_preserve_every_unbinned_point(
    adn_data, kind, expected_points, expected_units,
):
    units, selection, centers = adn_data
    points = scatter.prepare_depth_angles(scatter.select_adn_depth(units, selection),
                                          centers, kind=kind)
    before = points.copy(deep=True)
    panels = scatter.prepare_depth_panels(points, MICE, missing_mice=["m21"])

    assert [table.attrs["label"] for table in panels] == MICE + ["All"]
    assert sum(len(table) for table in panels[:-1]) == expected_points
    assert len(panels[-1]) == expected_points
    assert panels[-1].attrs["n_points"] == expected_points
    assert panels[-1].attrs["n_units"] == expected_units
    for mouse, table in zip(MICE, panels[:-1], strict=True):
        expected = points.loc[points.mouse.eq(mouse)].reset_index(drop=True)
        pd.testing.assert_frame_equal(table.reset_index(drop=True), expected)
        assert table.attrs["n_points"] == len(expected)
    pd.testing.assert_frame_equal(panels[-1].reset_index(drop=True),
                                  points.reset_index(drop=True))
    assert panels[2].empty and panels[4].empty
    assert panels[2].attrs["missing_selection"] is False
    assert panels[4].attrs["missing_selection"] is True
    assert panels[-1].attrs["missing_mice"] == ["m21"]
    if kind == "rf":
        np.testing.assert_array_equal(panels[0][["angle_deg", "depth_um"]].to_numpy(),
                                      [[0., 75.], [0., 75.], [-180., 200.]])
    pd.testing.assert_frame_equal(points, before)


def test_panels_reject_points_from_mouse_without_adn_selection(adn_data):
    units, selection, centers = adn_data
    points = scatter.prepare_depth_angles(scatter.select_adn_depth(units, selection),
                                          centers, kind="rf")
    with pytest.raises(ValueError, match="missing ADN selection cannot contribute"):
        scatter.prepare_depth_panels(points, MICE, missing_mice=["m14"])


def test_plot_draws_exact_points_with_six_white_downward_depth_panels(adn_data, monkeypatch):
    units, selection, centers = adn_data
    points = scatter.prepare_depth_angles(scatter.select_adn_depth(units, selection),
                                          centers, kind="rf")
    panels = scatter.prepare_depth_panels(points, MICE, missing_mice=["m21"])
    before = deepcopy(panels)

    def no_binning(*args, **kwargs):
        raise AssertionError("The renderer must draw unbinned coordinates")

    monkeypatch.setattr(np, "histogram2d", no_binning)
    with plt.style.context("dark_background"):
        figure, axes = scatter.plot_depth_scatter(
            panels, xlabel="RF center (°; egocentric)", title="ADN RF centers",
            angle_range=(-180., 180.), depth_max=400.,
        )
        try:
            assert len(axes) == 6
            assert len(figure.axes) == 6
            assert figure.get_facecolor() == (1., 1., 1., 1.)
            for axis, table in zip(axes, panels, strict=True):
                assert axis.get_facecolor() == (1., 1., 1., 1.)
                assert axis.xaxis.label.get_color() == "black"
                assert axis.get_ylim() == (400., 0.)
                assert axis.get_xlim() == (-180., 180.)
                assert axis.get_xlabel() == "RF center (°; egocentric)"
                assert not axis.images
                assert len(axis.collections) == 1
                np.testing.assert_array_equal(axis.collections[0].get_offsets(),
                                              table[["angle_deg", "depth_um"]].to_numpy())
            assert "0 cells · 0 points" in axes[2].get_title()
            assert [text.get_text() for text in axes[4].texts] == ["ADN selection needed"]
            assert "Excludes m21" in axes[5].get_title()
        finally:
            plt.close(figure)
        assert plt.rcParams["axes.facecolor"] == "black"

    for table, original in zip(panels, before, strict=True):
        pd.testing.assert_frame_equal(table, original)
        assert table.attrs == original.attrs


@pytest.mark.parametrize("kind, angle_range", [("rf", (-180., 180.)), ("hd", (0., 360.))])
def test_normalized_plot_draws_exact_fractional_coordinates(
    adn_data_with_deeper_unit, kind, angle_range,
):
    units, selection, centers = adn_data_with_deeper_unit
    normalized = scatter.normalize_shank_depth(scatter.select_adn_depth(units, selection))
    points = scatter.prepare_depth_angles(normalized, centers, kind=kind,
                                          depth_column="depth_fraction")
    panels = scatter.prepare_depth_panels(points, MICE, missing_mice=["m21"])
    before = deepcopy(panels)
    figure, axes = scatter.plot_depth_scatter(
        panels, xlabel="Angle (°)", title="ADN normalized depth",
        angle_range=angle_range, depth_max=1., depth_column="depth_fraction",
    )
    try:
        assert len(axes) == 6
        assert .5 in points.depth_fraction.to_numpy()
        for axis, table in zip(axes, panels, strict=True):
            assert axis.get_ylim() == (1., 0.)
            assert axis.get_ylabel() == "Normalized depth within shank"
            np.testing.assert_array_equal(axis.collections[0].get_offsets(),
                                          table[["angle_deg", "depth_fraction"]].to_numpy())
    finally:
        plt.close(figure)
    for table, original in zip(panels, before, strict=True):
        pd.testing.assert_frame_equal(table, original)
        assert table.attrs == original.attrs
