import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import spatial_cell_analysis as spatial
import spatial_cell_plotting as plotting


def test_load_data_uses_saved_camera_and_spike_times(tmp_path, monkeypatch):
    monkeypatch.setattr(spatial, "recording_root", tmp_path)
    monkeypatch.setattr(spatial, "date", "260827")
    monkeypatch.setattr(spatial, "recording_number", 10)
    session_dir = tmp_path / "260827" / "260827_10"
    data_dir = session_dir / "data"
    probe_dir = data_dir / "probeA"
    probe_dir.mkdir(parents=True)
    kilosort_dir = session_dir / "kilosort" / "ProbeA" / "kilosort_10"
    kilosort_dir.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [7, 9], "KSLabel": ["good", "mua"]}).to_csv(
        kilosort_dir / "cluster_KSLabel.tsv", sep="\t", index=False
    )
    saved_spike_times = np.asarray([100.125, 100.25, 100.75, 101.0, 101.5])
    np.save(probe_dir / "adc_spike_time.npy", saved_spike_times[:, None])
    np.save(kilosort_dir / "spike_clusters.npy", np.asarray([7, 7, 9, 7, 9]))
    relative_camera_times = np.asarray([0.0, 0.25, 0.5, 1.0, 1.25, 2.0])
    (data_dir / "sync_data.json").write_text(json.dumps({
        "exposure_sampling_number_list_mid": relative_camera_times.tolist(),
        "exposure_sampling_number_list_mid_raw": (relative_camera_times + 100).tolist(),
    }), encoding="utf-8")
    (data_dir / "session_info.json").write_text(
        json.dumps({"session_info": {}}), encoding="utf-8"
    )
    pd.DataFrame({
        "frame": [0, 1, 2, 3, 5],
        "center_x": [500.0] * 5,
        "center_y": [500.0] * 5,
        "hd_deg": [0.0, 90.0, 180.0, 270.0, 0.0],
    }).to_csv(session_dir / "260827.csv", index=False)
    pd.DataFrame({
        "interval_type": ["baseline"], "start": [0.25], "end": [1.25],
    }).to_csv(data_dir / "interval_table.csv", index=False)

    pose, pose_times, spike_times, spike_clusters, good_unit_ids = spatial.load_data()

    np.testing.assert_array_equal(pose["frame"], [1, 2, 3])
    np.testing.assert_array_equal(pose_times, [0.25, 0.5, 1.0])
    np.testing.assert_array_equal(spike_times, [0.25, 0.75, 1.0])
    np.testing.assert_array_equal(spike_clusters, [7, 9, 7])
    assert good_unit_ids == [7]
    np.testing.assert_array_equal(
        np.load(probe_dir / "adc_spike_time.npy").reshape(-1), saved_spike_times
    )


@pytest.mark.parametrize(
    "position, expected",
    [
        ((370, 485), [275, 0, 275, 550]),
        ((920, 485), [275, 550, 275, 0]),
        ((645, 210), [0, 275, 550, 275]),
        ((645, 760), [550, 275, 0, 275]),
        ((370, 210), [0, 0, 550, 550]),
        ((920, 760), [550, 550, 0, 0]),
    ],
)
def test_cardinal_rays_on_walls_and_corners(position, expected):
    with np.errstate(divide="raise", invalid="raise"):
        distances = spatial.d(
            np.array([0.0, 90.0, 180.0, 270.0]),
            np.array([position[0]]),
            np.array([position[1]]),
            np.array([0.0]),
        )
    np.testing.assert_allclose(distances[0], expected, atol=1e-10)


def test_center_rays_follow_square_geometry_and_relative_heading():
    theta = np.arange(-360.0, 721.0, 3.0)
    head_direction = np.array([0.0, 37.0, 90.0])
    angles = np.deg2rad(head_direction[:, None] + theta)
    expected = 275 / np.maximum(np.abs(np.sin(angles)), np.abs(np.cos(angles)))
    distances = spatial.d(
        theta, np.full(3, 645.0), np.full(3, 485.0), head_direction,
    )
    np.testing.assert_allclose(distances, expected)


@pytest.mark.parametrize(
    "frames, spikes, expected",
    [
        ([0, 1, 3], [0, 0.4, 0.5, 1, 1.9, 2, 2.1, 3], [3, 3, 2]),
        ([0, 1, 3], [], [0, 0, 0]),
        ([1], [1, 1], [2]),
    ],
)
def test_spikes_at_endpoints_and_midpoint_ties(frames, spikes, expected):
    counts = spatial.count_spikes_by_frame(np.array(spikes), np.array(frames))
    np.testing.assert_array_equal(counts, expected)


@pytest.fixture
def session_maps():
    pose = pd.DataFrame({
        "center_x": [420.0, 500.0, 620.0, 780.0, 850.0, 650.0],
        "center_y": [260.0, 410.0, 580.0, 690.0, 300.0, 470.0],
        "hd_deg": [13.0, 72.0, 156.0, 244.0, 301.0, 358.0],
    })
    return spatial.prepare_session_maps(pose, np.arange(len(pose)) * 0.5)


def test_units_share_occupancy_and_preserve_firing_rates(session_maps):
    for values in session_maps.values():
        values.setflags(write=False)

    frames = session_maps["frame_times"]
    for spikes_per_frame in (1, 2, 0):
        maps = spatial.compute_maps(np.repeat(frames, spikes_per_frame), session_maps)
        for key in ("egocentric_occupancy", "allocentric_occupancy"):
            assert maps[key] is session_maps[key]
        assert maps["allocentric_occupancy"].sum() == len(frames) * 0.5
        assert maps["allocentric_spike_map"].sum() == len(frames) * spikes_per_frame
        assert maps["spike_frame_counts"].sum() == len(frames) * spikes_per_frame
        for key in (
            "egocentric_rate_map", "allocentric_rate_map", "allocentric_tuning_curve",
        ):
            rates = maps[key]
            assert np.isfinite(rates).any()
            np.testing.assert_allclose(rates[np.isfinite(rates)], spikes_per_frame * 2)


def test_unoccupied_bins_remain_nan():
    with np.errstate(divide="raise", invalid="raise"):
        rates = spatial.compute_rate_map(np.zeros((2, 2)), np.zeros((2, 2)), 1, "nearest")
    assert np.isnan(rates).all()


def test_all_plot_exports_use_configured_limits_and_opaque_white_background(
    session_maps, monkeypatch, tmp_path,
):
    monkeypatch.setattr(plotting, "save_root_directory", tmp_path)
    maps = spatial.compute_maps(session_maps["frame_times"], session_maps)
    maps["spike_x_cm"] = np.repeat(maps["trajectory_x_cm"], maps["spike_frame_counts"])
    maps["spike_y_cm"] = np.repeat(maps["trajectory_y_cm"], maps["spike_frame_counts"])
    maps["x_edges"] = np.linspace(0, 30, 41)
    maps["y_edges"] = np.linspace(0, 45, 41)
    plotting.prepare_output_directories()

    with plt.style.context("dark_background"), plt.rc_context({"savefig.transparent": True}):
        figure = plotting.plot_maps(maps, 7, {"recording_number": 2, "probe_name": "A"})
        try:
            figure.canvas.draw()
            assert figure.get_facecolor() == to_rgba("white")
            for axis in figure.axes:
                assert axis.get_facecolor() == to_rgba("white")
                assert to_rgba(axis.title.get_color()) == to_rgba("black")
                assert to_rgba(axis.xaxis.label.get_color()) == to_rgba("black")
            panels = [axis for axis in figure.axes if axis.get_title()]
            assert len(panels) == 10
            trajectory_axis = next(
                axis for axis in panels if axis.get_title() == "Trajectory and spike positions"
            )
            np.testing.assert_allclose(trajectory_axis.get_xlim(), [0, 30])
            np.testing.assert_allclose(trajectory_axis.get_ylim(), [45, 0])
            plotting.save_figure(figure, "spatial_maps", 7)
            plotting.save_individual_plots(maps, 7)
        finally:
            plt.close(figure)

    expected_plots = {
        "spatial_maps", "egocentric_time_map", "egocentric_spike_map",
        "egocentric_rate_map", "egocentric_rate_map_polar", "egocentric_tuning_curve",
        "allocentric_occupancy", "allocentric_spike_map", "allocentric_rate_map",
        "trajectory_spike_positions", "allocentric_tuning_curve",
    }
    expected_files = {
        tmp_path / plot / extension / f"7.{extension}"
        for plot in expected_plots for extension in ("png", "svg")
    }
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == expected_files
    for path in expected_files:
        assert path.stat().st_size > 0
        if path.suffix == ".png":
            pixels = plt.imread(path)
            np.testing.assert_array_equal(pixels[0, 0], [1, 1, 1, 1])
            assert np.all(pixels[:, :, 3] == 1)


def test_saved_results_reproduce_maps_without_raw_inputs(session_maps, monkeypatch, tmp_path):
    import json

    monkeypatch.setattr(spatial, "save_root_directory", tmp_path)
    expected = spatial.compute_maps(session_maps["frame_times"], session_maps)
    metadata = spatial.save_session(session_maps, [7])
    spatial.save_unit(expected, 7)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    loaded_metadata, shared = plotting.load_results(tmp_path)
    actual = plotting.load_unit(tmp_path, shared, 7)
    assert loaded_metadata["unit_ids"] == [7]
    for key, value in expected.items():
        np.testing.assert_equal(actual[key], value)
    np.testing.assert_equal(actual["spike_x_cm"], session_maps["trajectory_x_cm"])
    with np.load(tmp_path / "units/7.npz", allow_pickle=False) as archive:
        assert not set(archive.files).intersection(spatial.SESSION_KEYS)
