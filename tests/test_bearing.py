import numpy as np
import pytest

from Utils.bearing import bearing


@pytest.mark.parametrize(
    "hd, start, end, expected",
    [
        (0, -10, 10, (-10, 10)),
        (90, 0, 90, (-90, 0)),
        (450, 360, 450, (-90, 0)),
        (-90, 0, 90, (90, -180)),
        (0, 350, 10, (-10, 10)),
        (0, 170, 190, (170, -170)),
        (0, 0, 270, (0, -90)),
    ],
)
def test_center_and_counterclockwise_edge_order(hd, start, end, expected):
    result = bearing(0, 0, 1, hd, start, end)
    assert isinstance(result, tuple)
    assert all(isinstance(value, float) for value in result)
    np.testing.assert_allclose(result, expected, atol=1e-12)
    # Include both the signed seam and an arc wider than 180 degrees.
    np.testing.assert_allclose(
        (result[1] - result[0]) % 360, (end - start) % 360, atol=1e-12,
    )


def test_observer_position_changes_edge_bearings():
    # From (0.5, 0.5), the cardinal edges lie southeast and northwest.
    np.testing.assert_allclose(
        bearing(0.5, 0.5, 1, 0, 0, 90), (-45, 135), atol=1e-12,
    )
    np.testing.assert_allclose(
        bearing(0.5, 0, 1, 0, 0, 90), (0, 116.56505117707799), atol=1e-12,
    )


@pytest.mark.parametrize("scale", [1e-200, 0.001, 3.74, 1000, 1e200])
def test_length_unit_and_scale_do_not_change_bearings(scale):
    np.testing.assert_allclose(
        bearing(0.5 * scale, 0.5 * scale, scale, 45, 0, 90),
        (-90, 90), atol=1e-12,
    )


def test_rotating_the_whole_geometry_preserves_ego_bearings():
    rotation = np.deg2rad(37)
    x, y = 0.2, -0.4
    rotated_x = x * np.cos(rotation) - y * np.sin(rotation)
    rotated_y = x * np.sin(rotation) + y * np.cos(rotation)
    np.testing.assert_allclose(
        bearing(rotated_x, rotated_y, 1, 57, 177, 257),
        bearing(x, y, 1, 20, 140, 220), atol=1e-12,
    )


def test_broadcasts_positions_headings_and_edges_without_mutating_inputs():
    x = np.array([[0], [0.5]])
    y = x.copy()
    hd = np.array([0, 90, 180])
    start, end = bearing(x, y, 1, hd, [0, 0, 0], 90)
    np.testing.assert_allclose(start, [[0, -90, -180], [-45, -135, 135]])
    np.testing.assert_allclose(end, [[90, 0, -90], [135, 45, -45]])
    np.testing.assert_array_equal(x, [[0], [0.5]])
    np.testing.assert_array_equal(y, x)
    np.testing.assert_array_equal(hd, [0, 90, 180])


@pytest.mark.parametrize("r", [0, -1])
def test_rejects_nonpositive_radius(r):
    with pytest.raises(ValueError, match="r must be positive"):
        bearing(0, 0, r, 0, 0, 90)


@pytest.mark.parametrize("position", [(1, 0), (0, -1), (2, 0)])
def test_rejects_observers_on_or_outside_circle(position):
    with pytest.raises(ValueError, match="inside the circle"):
        bearing(*position, 1, 0, 0, 90)


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_rejects_nonfinite_inputs(index, invalid):
    args = [0, 0, 1, 0, 0, 90]
    args[index] = invalid
    with pytest.raises(ValueError, match="finite"):
        bearing(*args)
