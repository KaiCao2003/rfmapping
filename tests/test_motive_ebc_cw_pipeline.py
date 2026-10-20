"""Independent clockwise Motive geometry and saved-map regression checks."""

import importlib
import json

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

import numpy as np
import pytest

from EBB import ebc_analysis


def independent_circle_rays(xy, yaw_deg, radius, bearings):
    """Intersect front/right basis rays with the circle using polynomial roots."""
    distances = np.empty((len(xy), len(bearings)))
    for frame, (point, yaw) in enumerate(zip(xy, yaw_deg, strict=True)):
        yaw = np.deg2rad(yaw)
        front = np.array([np.cos(yaw), np.sin(yaw)])
        right = np.array([front[1], -front[0]])
        for angle, bearing in enumerate(np.deg2rad(bearings)):
            direction = np.cos(bearing) * front + np.sin(bearing) * right
            roots = np.roots([direction @ direction, 2 * point @ direction,
                              point @ point - radius**2])
            distances[frame, angle] = max(roots.real)
    return distances


@pytest.mark.parametrize("yaw", [0., 90., 180., 270.])
def test_motive_front_right_left_at_cardinal_headings(yaw):
    point = np.array([[3., 4.]])
    bearings = np.array([0., 90., -90., 180., -180.])
    expected = independent_circle_rays(point, [yaw], 10., bearings)
    actual = ebc_analysis.circle_distances(point, np.array([yaw]), 10., bearings)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual[:, 3], actual[:, 4], atol=1e-12)


def test_motive_bin_centers_use_the_clockwise_side_of_the_heading():
    xy = np.array([[3., 4.], [-2., 6.], [5., -1.], [-4., -3.]])
    yaw = np.array([0., 37., 163., 283.])
    np.testing.assert_array_equal(ebc_analysis.THETA_DEG, np.arange(3., 360., 6.))
    expected = independent_circle_rays(xy, yaw, 15., ebc_analysis.THETA_DEG)
    actual = ebc_analysis.circle_distances(xy, yaw, 15.)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert not np.allclose(actual, expected[:, ::-1])


def independent_histograms(distances, edges, counts, dt):
    occupancy = np.zeros((distances.shape[1], len(edges) - 1))
    spikes = np.zeros((len(counts), *occupancy.shape))
    for frame, row in enumerate(distances):
        for angle, distance in enumerate(row):
            # The native bins are right-closed, including zero in the first bin.
            matches = np.flatnonzero((distance > edges[:-1]) & (distance <= edges[1:]))
            distance_bin = 0 if distance == 0 else matches[0]
            occupancy[angle, distance_bin] += dt
            spikes[:, angle, distance_bin] += counts[:, frame]
    return occupancy, spikes


def test_circle_workflow_preserves_clockwise_counts_occupancy_and_saved_maps(tmp_path):
    xy = np.array([[3., 4.], [-2., 6.], [5., -1.], [-4., -3.]])
    yaw = np.array([0., 37., 163., 283.])
    counts = np.array([[0, 3, 0, 1], [2, 0, 5, 0]])
    data = dict(session=tmp_path, probe="A", phase="baseline", xy=xy, hd=yaw,
                counts=counts, unit_ids=np.array([7, 19]), dt=.02,
                center_raw=np.array([.1, .2]), cm_per_unit=100.,
                interval=np.array([0., .08]), position_held=np.zeros(4, dtype=bool),
                initial_outside=np.zeros(4, dtype=bool),
                source={"position_path": "raw-Motive-pose.csv",
                        "camera_timing": {"frame_timestamp_mapping": "zero_based_motive_frame_id_to_original_ttl_timestamp"}})

    results = ebc_analysis.compute_circle_ebc(data, sigma=0)

    for name, radius in (("inner", 15.), ("outer", 30.)):
        item = results[name]
        distances = independent_circle_rays(xy, yaw, radius, ebc_analysis.THETA_DEG)
        occupancy, spikes = independent_histograms(distances, item["distance_edges"], counts, data["dt"])
        np.testing.assert_allclose(item["occupancy"], occupancy, atol=1e-15)
        np.testing.assert_allclose(occupancy.sum(axis=1), len(xy) * data["dt"])
        np.testing.assert_array_equal(spikes.sum(axis=2),
                                      np.repeat(counts.sum(axis=1)[:, None], 60, axis=1))
        expected = np.divide(spikes, occupancy, out=np.full_like(spikes, np.nan),
                             where=occupancy > 0)
        with np.load(item["paths"]["full"]) as saved:
            metadata = json.loads(saved["metadata"].tobytes())
            assert "clockwise" in metadata["angleConvention"]
            assert "counterclockwise from +X" in metadata["headingConvention"]
            assert metadata["cameraTiming"] == data["source"]["camera_timing"]
            np.testing.assert_array_equal(saved["yPositions"], ebc_analysis.THETA_DEG)
            np.testing.assert_array_equal(saved["unitPool"], data["unit_ids"])
            for unit_index, unit_id in enumerate(data["unit_ids"]):
                np.testing.assert_allclose(saved[f"unit_{unit_id}"][..., 0], expected[unit_index], equal_nan=True)
        for band, lower, upper in (("0-30", 0., 30.), ("30-60", 30., 60.)):
            if name == "outer":
                centers = (item["distance_edges"][:-1] + item["distance_edges"][1:]) / 2
                selected = (centers > lower) & (centers <= upper)
                finite = np.isfinite(expected[..., selected]).any(axis=2)
                collapsed = np.where(finite, np.nansum(expected[..., selected], axis=2), np.nan)
                with np.load(item["paths"][band]) as saved:
                    for unit_index, unit_id in enumerate(data["unit_ids"]):
                        np.testing.assert_allclose(saved[f"unit_{unit_id}"][:, 0, 0],
                                                   collapsed[unit_index], equal_nan=True)


@pytest.mark.parametrize("module", ["analysis", "geometry", "camera", "pose", "replay",
                                     "overlay", "video", "workflow"])
def test_old_ebc_imports_resolve_to_the_same_implementation(module):
    legacy = importlib.import_module(f"Utils.ebc_{module}")
    canonical = importlib.import_module(f"EBB.ebc_{module}")
    assert legacy is canonical


@pytest.mark.parametrize("ignore_hd_ttl", [False, True])
def test_cylinder_cli_forwards_loading_options_without_another_implementation(
    tmp_path, monkeypatch, ignore_hd_ttl,
):
    from scripts import run_cylinder_spatial as cli

    assert cli.load_motive_session is ebc_analysis.load_motive_session
    assert cli.compute_circle_ebc is ebc_analysis.compute_circle_ebc
    calls = []
    loaded = object()

    def load(session, **options):
        calls.append((session, options))
        return loaded

    def compute(data):
        assert data is loaded
        return {"inner": {"paths": {"full": tmp_path / "map.rfmap"}}}

    monkeypatch.setattr(cli, "load_motive_session", load)
    monkeypatch.setattr(cli, "compute_circle_ebc", compute)
    argv = [str(tmp_path), "--probe", "B", "--phase", "probe"]
    if ignore_hd_ttl:
        argv.append("--ignore-hd-ttl")
    cli.main(argv)
    assert calls == [(tmp_path, dict(probe="B", phase="probe", ignore_hd_ttl=ignore_hd_ttl))]


def test_cylinder_coverage_signed_bearings_and_white_canvas_in_dark_style():
    occupancy = np.arange(60 * 20).reshape(60, 20) + 1.
    results = {name: dict(distance_edges=np.arange(21), occupancy=occupancy)
               for name in ("inner", "outer")}
    with plt.style.context("dark_background"):
        figure = ebc_analysis.plot_circle_coverage({"xy": np.array([[0., 0.], [1., 2.]])}, results)
    assert figure.get_facecolor() == to_rgba("white")
    for axis in figure.axes:
        assert axis.get_facecolor() == to_rgba("white")
        assert to_rgba(axis.xaxis.label.get_color()) == to_rgba("black")
        assert to_rgba(axis.yaxis.label.get_color()) == to_rgba("black")
    for axis in figure.axes[1:3]:
        np.testing.assert_array_equal(axis.get_ylim(), [-180., 180.])
        np.testing.assert_array_equal(axis.get_yticks(), [-180., -90., 0., 90., 180.])
        displayed = axis.collections[0].get_array()
        np.testing.assert_array_equal(displayed, np.roll(occupancy, 30, axis=0))
    plt.close(figure)
