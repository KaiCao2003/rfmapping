import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from Utils.plotting import angular_ticks, plot_direction_comparison, plot_keyed_heatmap
from Utils.statistic_utils import compare_direction_angles


@pytest.mark.parametrize("labels, expected", [
    ((-180, -90, 0, 90, 180), (-180, -90, 0, 90, 180)),
    ((180, 270, 0, 90, 180), (-180, -90, 0, 90, 180)),
    ((180, 90, 0, 270, 180), (180, 90, 0, -90, -180)),
])
def test_angular_ticks_infers_ego_and_both_allo_directions_without_changing_labels(labels, expected):
    labels = np.array(labels)
    original = labels.copy()
    np.testing.assert_array_equal(angular_ticks(labels), expected)
    np.testing.assert_array_equal(labels, original)


def test_keyed_heatmap_preserves_probe_keys_missing_bins_and_input_arrays(monkeypatch):
    profiles = {
        ("A", 7): np.array([0, 2, np.nan, 4]),
        ("B", 7): np.zeros(4),
        ("A", 9): np.full(4, np.nan),
        ("B", 9): np.array([3, 9, 6, 0]),
    }
    originals = {key: row.copy() for key, row in profiles.items()}
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    with plt.style.context("dark_background"), plt.rc_context({"image.origin": "lower"}):
        figure, axis = plot_keyed_heatmap(
            profiles, [("B", 9), ("B", 7), ("A", 7), ("A", 9)],
            column_order=[2, 0, 3, 1], show=False,
            xlabel="EBC bearing (°)", title="Shared units",
            xticks=(-180, -90, 0, 90, 180), xticklabels=(-180, -90, 0, 90, 180),
        )
    try:
        np.testing.assert_allclose(
            axis.images[0].get_array().filled(np.nan),
            [[6, 3, 0, 9], [0, 0, 0, 0], [np.nan, 0, 4, 2], [np.nan] * 4],
            equal_nan=True,
        )
        assert [tick.get_text() for tick in axis.get_yticklabels()] == ["B:9", "B:7", "A:7", "A:9"]
        assert axis.images[0].origin == "upper"
        assert axis.get_ylim() == (3.5, -0.5)
        assert axis.get_xlim() == (-180, 180)
        assert axis.images[0].get_clim() == (0, 9)
        assert axis.get_xlabel() == "EBC bearing (°)"
        assert axis.get_title() == "Shared units"
        assert figure.get_facecolor() == (1, 1, 1, 1)
        assert all(ax.get_facecolor() == (1, 1, 1, 1) for ax in figure.axes)
        assert axis.xaxis.label.get_color() == "black"
        assert show_calls == []
        for key in profiles:
            np.testing.assert_array_equal(profiles[key], originals[key])
    finally:
        plt.close(figure)


@pytest.mark.parametrize("figsize", [None, (8, 4)])
def test_keyed_heatmap_keeps_all_rows_in_a_fixed_size_figure(figsize):
    profiles = {unit: np.roll(np.arange(30), unit) for unit in range(300)}
    kwargs = {} if figsize is None else {"figsize": figsize}
    with plt.style.context("dark_background"):
        figure, axis = plot_keyed_heatmap(
            profiles, list(profiles), column_order=np.arange(30),
            xticks=(0, 90, 180, 270, 360), xticklabels=(0, 90, 180, 270, 360),
            show=False, **kwargs,
        )
    try:
        np.testing.assert_array_equal(figure.get_size_inches(), (10, 6) if figsize is None else figsize)
        np.testing.assert_allclose(axis.images[0].get_array(), np.stack(list(profiles.values())))
        assert axis.images[0].get_array().shape == (300, 30)
        np.testing.assert_array_equal(axis.get_yticks(), np.arange(300))
        assert len(axis.get_yticklabels()) == 300
        assert [tick.get_text() for tick in axis.get_yticklabels()] == [str(unit) for unit in range(300)]
        assert figure.get_facecolor() == (1, 1, 1, 1)
        assert all(ax.get_facecolor() == (1, 1, 1, 1) for ax in figure.axes)
        assert axis.yaxis.label.get_color() == "black"
    finally:
        plt.close(figure)


def test_direction_comparison_uses_the_same_style_and_counts_all_ordered_pairs(monkeypatch):
    reference = np.array([7, 35, 89, 152, 223, 304])
    matched = (reference + 67) % 360
    statistics = compare_direction_angles(reference, matched, n_permutations=9)
    reference_ego = np.array([7, 35, 89, 152, -137, -56])
    matched_ego = np.array([74, 102, 156, -141, -70, 11])
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    with plt.style.context("dark_background"):
        scatter, density = plot_direction_comparison(
            reference_ego, matched_ego, statistics, reference_label="HD", matched_label="RF",
            angle_ticks=(-180, -90, 0, 90, 180),
            reference_ticklabels=(180, 90, 0, 270, 180),
            matched_ticklabels=(-180, -90, 0, 90, 180),
            density_bin_count=12, show=False,
        )
    try:
        np.testing.assert_array_equal(scatter[1].collections[0].get_offsets(), np.c_[reference_ego, matched_ego])
        assert density[1].collections[0].get_array().sum() == reference.size ** 2
        assert scatter[1].get_xlim() == (-180, 180)
        assert density[1].get_xlim() == (-5, 185)
        assert "HD" in scatter[1].get_xlabel()
        assert "RF" in scatter[1].get_ylabel()
        assert [tick.get_text() for tick in scatter[1].get_xticklabels()] == ["180", "90", "0", "270", "180"]
        assert [tick.get_text() for tick in scatter[1].get_yticklabels()] == ["-180", "-90", "0", "90", "180"]
        annotation = scatter[1].texts[0].get_text().splitlines()
        assert len(annotation) == 2
        assert annotation[0].startswith("Circular-circular rho = ")
        assert annotation[1].startswith("p = ")
        assert "Mantel ρ" in density[1].texts[0].get_text()
        for figure, axis in (scatter, density):
            assert figure.get_facecolor() == (1, 1, 1, 1)
            assert all(item.get_facecolor() == (1, 1, 1, 1) for item in figure.axes)
            assert axis.xaxis.label.get_color() == "black"
        assert show_calls == []
    finally:
        plt.close(scatter[0])
        plt.close(density[0])


def test_direction_comparison_wraps_native_points_into_independent_axis_ranges():
    reference = np.array([0, 90, -90])
    matched = np.array([0, -90, 90])
    statistics = compare_direction_angles(reference, -matched % 360, n_permutations=9)
    allo_ticks = (0, 90, 180, 270, 360)
    ego_ticks = (-180, -90, 0, 90, 180)
    with plt.style.context("dark_background"):
        scatter, density = plot_direction_comparison(
            reference, matched, statistics, reference_label="HD", matched_label="EBC",
            angle_ticks=ego_ticks, reference_ticks=allo_ticks, matched_ticks=ego_ticks,
            reference_ticklabels=allo_ticks, matched_ticklabels=ego_ticks, show=False,
        )
    try:
        np.testing.assert_array_equal(scatter[1].collections[0].get_offsets(), np.c_[[0, 90, 270], matched])
        np.testing.assert_array_equal(reference, [0, 90, -90])
        np.testing.assert_array_equal(matched, [0, -90, 90])
        assert scatter[1].get_xlim() == (0, 360)
        assert scatter[1].get_ylim() == (-180, 180)
        np.testing.assert_array_equal(scatter[1].get_xticks(), allo_ticks)
        np.testing.assert_array_equal(scatter[1].get_yticks(), ego_ticks)
        assert [tick.get_text() for tick in scatter[1].get_xticklabels()] == ["0", "90", "180", "270", "360"]
        assert [tick.get_text() for tick in scatter[1].get_yticklabels()] == ["-180", "-90", "0", "90", "180"]
        assert density[1].get_xlim() == (-5, 185)
        assert density[1].get_ylim() == (-5, 185)
        assert density[1].collections[0].get_array().sum() == reference.size ** 2
        for figure, axis in (scatter, density):
            assert figure.get_facecolor() == (1, 1, 1, 1)
            assert all(item.get_facecolor() == (1, 1, 1, 1) for item in figure.axes)
            assert axis.xaxis.label.get_color() == "black"
            assert axis.yaxis.label.get_color() == "black"
    finally:
        plt.close(scatter[0])
        plt.close(density[0])


@pytest.mark.parametrize("allo_axis", ["reference", "matched"])
def test_direction_comparison_uses_native_positions_with_ego_or_allo_labels(allo_axis):
    reference = np.array([-150, -90, 0, 90, 150])
    matched = np.array([-120, -90, 30, 90, 120])
    ego_ticks = (-180, -90, 0, 90, 180)
    allo_ticks = (-180, -90, 0, 90, 180)
    allo_labels = (180, 270, 0, 90, 180)
    reference_ticks = allo_ticks if allo_axis == "reference" else ego_ticks
    matched_ticks = allo_ticks if allo_axis == "matched" else ego_ticks
    reference_labels = allo_labels if allo_axis == "reference" else ego_ticks
    matched_labels = allo_labels if allo_axis == "matched" else ego_ticks
    statistics = compare_direction_angles(reference, matched, n_permutations=9)
    scatter, density = plot_direction_comparison(
        reference, matched, statistics,
        reference_label="HD" if allo_axis == "reference" else "RF",
        matched_label="HD" if allo_axis == "matched" else "RF",
        reference_ticklabels=reference_labels, matched_ticklabels=matched_labels,
        show=False,
    )
    try:
        axis = scatter[1]
        np.testing.assert_array_equal(axis.collections[0].get_offsets(), np.c_[reference, matched])
        assert axis.get_xlim() == (reference_ticks[0], reference_ticks[-1])
        assert axis.get_ylim() == (matched_ticks[0], matched_ticks[-1])
        np.testing.assert_array_equal(axis.get_xticks(), reference_ticks)
        np.testing.assert_array_equal(axis.get_yticks(), matched_ticks)
        assert [tick.get_text() for tick in axis.get_xticklabels()] == list(map(str, reference_labels))
        assert [tick.get_text() for tick in axis.get_yticklabels()] == list(map(str, matched_labels))
        screen = axis.transAxes.inverted().transform(axis.transData.transform([[-90, -90], [90, 90]]))
        np.testing.assert_allclose(screen[:, 0], [.25, .75])
        np.testing.assert_allclose(screen[:, 1], [.25, .75])
        assert density[1].get_xlim() == (-5, 185)
        assert density[1].get_ylim() == (-5, 185)
        assert density[1].collections[0].get_array().sum() == reference.size ** 2
    finally:
        plt.close(scatter[0])
        plt.close(density[0])
