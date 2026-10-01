from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import hd_rf_population_video as population
from Utils.ebc_camera import project


def cylinder_data(*, hd=(90.,), world_zero=0., projection=None):
    if projection is None:
        projection = np.array([[2., .3, .2, 640.], [.1, -1.8, .4, 480.],
                               [.002, .001, .01, 1.]])
    ego = np.deg2rad(30.)
    count = len(hd)
    return dict(
        arena_type="cylinder", session=Path("260921_9"), probe="A", hd_mode="real",
        hd_source={"method": "saved head_direction.json"}, hd=np.asarray(hd, dtype=float),
        xyz=np.tile([20., 30., 10.], (count, 1)), projection=projection,
        cylinder_center_raw=np.array([0., 0.]), cylinder_radius_raw=100., raw_units_per_cm=10.,
        hd_world_zero_deg=world_zero, frame_period_s=1 / 60., trace_window_s=12.,
        times=np.arange(count) / 60., exposure_times=np.arange(max(3, count)) / 60.,
        model=dict(method="first_harmonic_circular_regression", scheme="scheme1", n=3,
                   cos_coefficients=[np.cos(ego), 0., 0.], sin_coefficients=[np.sin(ego), 0., 0.],
                   rho_circular=.5, fit_mae_deg=12., selected_on_sum=False),
    )


def expected_wall_point(head, angle_deg, radius):
    direction = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
    along = head[:2] @ direction
    distance = -along + np.sqrt(along ** 2 + radius ** 2 - head[:2] @ head[:2])
    point = head.copy()
    point[:2] += distance * direction
    assert np.linalg.norm(point[:2]) == pytest.approx(radius)
    return point, distance


@pytest.mark.parametrize("world_zero", [0., 25.])
def test_physical_hd_and_rf_points_are_projected_without_image_angle_rotation(world_zero):
    data = cylinder_data(world_zero=world_zero)
    original_hd = data["hd"].copy()
    overlay = population.PopulationOverlay(data, 1280, 1024, 60.)
    geometry = overlay.cylinder_geometry(0)
    head = data["xyz"][0]
    hd_ccw = world_zero - 90.
    hd_tip = head + 50 * np.array([np.cos(np.deg2rad(hd_ccw)), np.sin(np.deg2rad(hd_ccw)), 0.])
    rf_ccw = world_zero - 120.
    rf_points = np.stack([expected_wall_point(head, angle, 100.)[0]
                          for angle in (rf_ccw, rf_ccw - 6., rf_ccw + 6.)])
    boundary = [expected_wall_point(head, hd_ccw + angle, 100.) for angle in (0., 90., 180., 270.)]

    np.testing.assert_allclose(overlay.positions[0], project(head, data["projection"]))
    np.testing.assert_allclose(geometry["hd_tip_px"], project(hd_tip, data["projection"]))
    np.testing.assert_allclose(geometry["rf_points_px"], project(rf_points, data["projection"]))
    np.testing.assert_allclose(geometry["wall_endpoints_px"],
                               project(np.stack([point for point, _ in boundary]), data["projection"]))
    np.testing.assert_allclose(geometry["wall_distances_raw"], [distance for _, distance in boundary])
    np.testing.assert_array_equal(data["hd"], original_hd)
    assert not hasattr(overlay, "fixed_origin")
    # A perspective projection changes this bearing from the raster +90° direction.
    direction = geometry["hd_tip_px"] - overlay.positions[0]
    direction /= np.linalg.norm(direction)
    assert not np.allclose(direction, population.screen_direction(90.))


def test_head_plane_wall_uses_current_xyz_height_with_perspective_projection():
    data = cylinder_data(hd=(90., 90.))
    data["xyz"][1, 2] = 60.
    overlay = population.PopulationOverlay(data, 1280, 1024, 60.)
    first, second = overlay.cylinder_geometry(0), overlay.cylinder_geometry(1)
    assert not np.allclose(first["wall_pixels"], second["wall_pixels"])
    for row, geometry in enumerate((first, second)):
        wall = np.c_[overlay.wall_xy, np.full(len(overlay.wall_xy), data["xyz"][row, 2])]
        np.testing.assert_allclose(geometry["wall_pixels"], project(wall, data["projection"]))


def test_camera_arrows_start_at_projected_head_and_geometry_stays_in_camera_panel(monkeypatch):
    data = cylinder_data()
    overlay = population.PopulationOverlay(data, 1280, 1024, 60.)
    arrows = []

    def record_arrow(draw, start, end, color):
        arrows.append((np.asarray(start), np.asarray(end), color))

    monkeypatch.setattr(overlay, "projected_arrow", record_arrow)
    frame = Image.new("RGB", (1280, 1024), "#202020")
    before = np.asarray(frame).copy()
    canvas = overlay.draw(frame, 0, 0)
    geometry = overlay.cylinder_geometry(0)

    assert len(arrows) == 2
    for start, _, _ in arrows:
        np.testing.assert_allclose(start, project(data["xyz"][0], data["projection"]))
    np.testing.assert_allclose(arrows[0][1], geometry["hd_tip_px"])
    np.testing.assert_allclose(arrows[1][1], geometry["rf_points_px"][0])
    assert [color for _, _, color in arrows] == [population.HD_COLOR, population.RF_COLOR]
    np.testing.assert_array_equal(np.asarray(frame), before)
    assert canvas.size == (1650, 1324)
    assert canvas.getpixel((1285, 1000)) == (255, 255, 255)


@pytest.mark.parametrize("missing", ["clock", "row", "xyz", "hd"])
def test_unavailable_cylinder_geometry_preserves_every_camera_pixel(missing):
    data = cylinder_data()
    row = 0
    if missing == "clock":
        data["exposure_times"][0] = np.nan
    elif missing == "row":
        row = -1
    elif missing == "xyz":
        data["xyz"][0, 0] = np.nan
    else:
        data["hd"][0] = np.nan
    overlay = population.PopulationOverlay(data, 1280, 1024, 60.)
    frame = Image.new("RGB", (1280, 1024), "#202020")
    canvas = overlay.draw(frame, 0, row)
    np.testing.assert_array_equal(np.asarray(canvas.crop((0, 0, 1280, 1024))), np.asarray(frame))


def test_projected_arrowhead_follows_projected_tip_vector():
    lines, polygons = [], []

    class Draw:
        def line(self, points, **settings):
            lines.append(points)

        def polygon(self, points, **settings):
            polygons.append(points)

    start, tip = np.array([10., 20.]), np.array([40., 60.])
    population.PopulationOverlay.projected_arrow(Draw(), start, tip, population.HD_COLOR)
    direction, normal = np.array([.6, .8]), np.array([-.8, .6])
    np.testing.assert_allclose(lines[0], [start, tip])
    np.testing.assert_allclose(polygons[0], [tip, tip - 16 * direction + 8 * normal,
                                           tip - 16 * direction - 8 * normal])
