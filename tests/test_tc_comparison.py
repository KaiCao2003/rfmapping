"""Requested comparison bins apply to cached curves and every comparison panel."""

import json

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from RF import tc_generation
from HD.tc_generation import generate_hd_tc
from Utils.direction_comparison import rebin_profiles, save_tc, tc_loader
from Utils.tc_comparison import calculate_tc_statistics, plot_tc_comparison
from HD.profiles import prep_hd_tc, load_hd_comparison
from RF.profiles import prep_rf_tc, load_rf_comparison


def test_rebin_sums_adjacent_bins_and_preserves_the_circular_seam():
    values = np.arange(1., 61.).reshape(2, 30)
    source = pd.DataFrame(
        values, index=["m14:260609:A:7", "m14:260609:A:11"],
        columns=(np.arange(6., 360., 12.) + 180) % 360 - 180,
    )
    source.attrs = {"label": "HD", "range": [0, 360], "response_units": "Hz"}
    original = source.copy(deep=True)

    actual = rebin_profiles(source, bins=15)

    np.testing.assert_array_equal(actual, values[:, ::2] + values[:, 1::2])
    np.testing.assert_allclose(actual.columns, (np.arange(12., 360., 24.) + 180) % 360 - 180)
    assert actual.columns[7] == -180
    assert actual.attrs == source.attrs
    pd.testing.assert_frame_equal(source, original)
    with pytest.raises(ValueError):
        rebin_profiles(source, bins=16)


@pytest.fixture
def cached_curves(tmp_path, monkeypatch):
    values = np.arange(1., 31.) ** 2
    values = np.stack([values, np.roll(values, 6), np.roll(values, 17), values])
    values[-1, :3] = 0  # Three native zero bins would become only one after grouping.
    ids = pd.Index([f"m14:260609:A:{unit}" for unit in (7, 11, 23, 31)], name="unit_id")
    originals = {}
    for session, name, angles in (
        (1, "hd_class3_ProbeA.csv", (np.arange(6., 360., 12.) + 180) % 360 - 180),
        (3, "rf_excitatory_x_2d_rfonly_ProbeA.csv", np.arange(-145., 150., 10.)),
    ):
        path = tmp_path / "m14/260609" / f"260609_{session}/data/tc_comparison" / name
        path.parent.mkdir(parents=True)
        save_tc(pd.DataFrame(values, index=ids, columns=angles), path)
        originals[path] = path.read_bytes()

    def unexpected_source_read(*args, **kwargs):
        pytest.fail("Cached curves must be rebinned without reading the original sources")

    for name in ("load_rfmap", "load_rf", "load_rf_tc"):
        monkeypatch.setattr(tc_generation, name, unexpected_source_read)
    hd = prep_hd_tc(load_hd_comparison("m14", tmp_path, "A", "260609", 1), bins=15)
    rf = prep_rf_tc(load_rf_comparison("m14", tmp_path, "A", "260609", 3, rf_range=[-150, 150]), bins=15)
    return hd, rf, values, ids, originals


def test_cached_hd_and_rf_use_grouped_values_without_rewriting_csvs(cached_curves):
    hd, rf, values, ids, originals = cached_curves
    grouped = values[:, ::2] + values[:, 1::2]
    normalized = grouped / grouped.max(axis=1, keepdims=True)

    np.testing.assert_array_equal(hd, normalized)
    np.testing.assert_array_equal(rf, normalized[:3])
    assert hd.index.equals(ids)
    assert rf.index.equals(ids[:3])
    np.testing.assert_allclose(rf.columns, np.arange(-140., 150., 20.))
    assert rf.attrs["range"] == [-150, 150]
    assert {path: path.read_bytes() for path in originals} == originals


def test_first_hd_generation_is_explicit_and_comparison_keeps_saved_30_bins(tmp_path):
    session_dir = tmp_path / "m14/260609/260609_1"
    source = session_dir / "data/tuning_curves/ProbeA/tuning_curves.tc"
    source.parent.mkdir(parents=True)
    edges = np.linspace(0, 360, 181)
    angles = np.deg2rad((edges[:-1] + edges[1:]) / 2)
    rates = np.exp(.6 * np.cos(angles))[None, :]
    occupancy = np.linspace(1, 3, 180)
    counts = rates * occupancy
    source.write_text(json.dumps({
        "metadata": {"classification": {}}, "unit_id": [23],
        "angle_bin_edges_deg": edges.tolist(), "occupancy_time_s": occupancy.tolist(),
        "firing_rate_hz": rates.tolist(), "spike_counts": counts.tolist(),
        "unit_data": {"hd_class": [3], "rayleigh_p": [.001], "shuffle_p": [.001]},
    }))
    original_source = source.read_bytes()

    with pytest.raises(FileNotFoundError):
        load_hd_comparison("m14", tmp_path, "A", "260609", 1)
    output = session_dir / "data/tc_comparison/hd_class3_ProbeA.csv"
    assert not output.exists()
    generate_hd_tc(source, output, bins=30, hd_class=3, unit_prefix="m14:260609:A")
    hd15 = prep_hd_tc(load_hd_comparison("m14", tmp_path, "A", "260609", 1), bins=15)
    output = session_dir / "data/tc_comparison/hd_class3_ProbeA.csv"
    saved = tc_loader(output)
    original_cache = output.read_bytes()
    expected30 = counts.reshape(1, 30, 6).sum(axis=2) / occupancy.reshape(30, 6).sum(axis=1)
    expected15 = expected30[:, ::2] + expected30[:, 1::2]

    assert saved.shape == (1, 30)
    assert saved.index.tolist() == ["m14:260609:A:23"]
    np.testing.assert_array_equal(saved, expected30)
    np.testing.assert_array_equal(hd15, expected15 / expected15.max(axis=1, keepdims=True))

    hd30 = prep_hd_tc(load_hd_comparison("m14", tmp_path, "A", "260609", 1), bins=30)
    np.testing.assert_array_equal(hd30, expected30 / expected30.max(axis=1, keepdims=True))
    assert output.read_bytes() == original_cache
    assert source.read_bytes() == original_source


def test_statistics_bins_do_not_change_curves_or_peak_comparisons(cached_curves):
    hd, rf, *_ = cached_curves
    statistics = [
        calculate_tc_statistics(hd, rf, bins=bins, n_permutations=19)
        for bins in (15, 30)
    ]
    pd.testing.assert_frame_equal(statistics[0]["peaks"], statistics[1]["peaks"])
    np.testing.assert_equal(statistics[0]["direction"], statistics[1]["direction"])
    for first, second in zip(statistics[0]["comparisons"], statistics[1]["comparisons"]):
        pd.testing.assert_frame_equal(first["reference"], second["reference"])
        pd.testing.assert_frame_equal(first["matched"], second["matched"])

    for bins, stats in zip((15, 30), statistics):
        assert len(stats["counts"]) == bins
        assert stats["counts"].sum() == 3
        for comparison in stats["comparisons"]:
            assert comparison["reference"].shape == (3, 15)
            assert comparison["matched"].shape == (3, 15)
            if comparison["mode"] != "native":
                assert 0 in comparison["reference"].columns

        figures = plot_tc_comparison(hd, rf, stats, figsize=(3, 3), show=False)
        try:
            for _, axis in figures[:12]:
                assert axis.images[0].get_array().shape == (3, 15)
            assert len(figures[12][1].patches) == bins
            assert figures[-1][1].collections[0].get_coordinates().shape[:2] == (bins + 1, bins + 1)
        finally:
            for figure, _ in figures:
                plt.close(figure)

        unwrapped = calculate_tc_statistics(hd, rf, bins=bins, is_wrap=False, n_permutations=19)
        for comparison in unwrapped["comparisons"]:
            assert comparison["reference"].shape == (3, 15)
            expected = 31 if comparison["mode"] == "sum" else 15
            assert comparison["matched"].shape == (3, expected)
