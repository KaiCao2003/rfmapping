import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba

import spatial_cell_analysis as spatial
import spatial_cell_plotting as plotting


@pytest.fixture
def recording_inputs(tmp_path, monkeypatch):
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
    return session_dir


def test_load_data_uses_saved_camera_and_spike_times(recording_inputs):
    pose, pose_times, spike_times, spike_clusters, good_unit_ids, source = spatial.load_data()

    np.testing.assert_array_equal(pose["frame"], [1, 2, 3])
    np.testing.assert_array_equal(pose_times, [0.25, 0.5, 1.0])
    np.testing.assert_array_equal(spike_times, [0.25, 0.75, 1.0])
    np.testing.assert_array_equal(spike_clusters, [7, 9, 7])
    assert good_unit_ids == [7]
    assert source["selected_interval_s"] == [0.25, 1.25]
    assert source["adc_time_origin_s"] == 100
    assert source["camera_timing"]["timestamp_reference"] == "saved_exposure_midpoint"
    np.testing.assert_array_equal(
        np.load(recording_inputs / "data/probeA/adc_spike_time.npy").reshape(-1),
        [100.125, 100.25, 100.75, 101.0, 101.5],
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
    session_maps, tmp_path,
):
    maps = spatial.compute_maps(session_maps["frame_times"], session_maps)
    maps["spike_x_cm"] = np.repeat(maps["trajectory_x_cm"], maps["spike_frame_counts"])
    maps["spike_y_cm"] = np.repeat(maps["trajectory_y_cm"], maps["spike_frame_counts"])
    maps["x_edges"] = np.linspace(0, 30, 41)
    maps["y_edges"] = np.linspace(0, 45, 41)
    plotting.prepare_output_directories(tmp_path)
    metadata = {
        "date": "260831", "recording_number": 2, "probe_name": "B", "phase_key": "test",
    }

    with plt.style.context("dark_background"), plt.rc_context({"savefig.transparent": True}):
        figure = plotting.plot_maps(maps, 7, metadata)
        try:
            figure.canvas.draw()
            assert figure._suptitle.get_text() == "260831, rec 2, ProbeB, test, unit 7"
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
            plotting.save_figure(figure, "spatial_maps", 7, tmp_path)
            plotting.save_individual_plots(maps, 7, tmp_path)
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
    monkeypatch.setattr(spatial, "save_root_directory", tmp_path)
    expected = spatial.compute_maps(session_maps["frame_times"], session_maps)
    # Unoccupied bins must survive serialization without becoming zeros.
    expected["egocentric_rate_map"][0, 0] = np.nan
    metadata = spatial.save_session(session_maps, [7], {})
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


def test_analysis_then_plotting_cli_with_raw_recording_removed(recording_inputs, tmp_path):
    result_path = tmp_path / "results"
    label_path = recording_inputs / "kilosort/ProbeA/kilosort_10/cluster_KSLabel.tsv"
    pd.DataFrame({"cluster_id": [7, 9], "KSLabel": ["good", "good"]}).to_csv(
        label_path, sep="\t", index=False,
    )
    script_directory = Path(spatial.__file__).parent
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    analysis_command = [
        sys.executable, str(script_directory / "spatial_cell_analysis.py"),
        "--recording-root", str(tmp_path), "--date", "260827",
        "--recording-number", "10", "--probe", "A", "--phase", "baseline",
        "--output", str(result_path), "--workers", "2",
    ]
    subprocess.run(analysis_command, check=True, env=environment, capture_output=True)
    assert {str(path.relative_to(result_path)) for path in result_path.rglob("*")
            if path.is_file()} == {"metadata.json", "session.npz", "units/7.npz", "units/9.npz"}
    metadata, shared = plotting.load_results(result_path)
    assert metadata["date"] == "260827"
    assert metadata["recording_number"] == 10
    assert metadata["source"]["selected_interval_s"] == [0.25, 1.25]
    assert metadata["frame_dt_s"] == 0.375
    expected = spatial.compute_maps(np.array([0.25, 1.0]), spatial.prepare_session_maps(
        pd.DataFrame({
            "center_x": [500.0] * 3, "center_y": [500.0] * 3,
            "hd_deg": [90.0, 180.0, 270.0],
        }), np.array([0.25, 0.5, 1.0]),
    ))
    loaded = plotting.load_unit(result_path, shared, 7)
    for key, value in expected.items():
        np.testing.assert_equal(loaded[key], value)

    selected_path = tmp_path / "selected"
    selected_command = analysis_command.copy()
    selected_command[selected_command.index("--output") + 1] = str(selected_path)
    subprocess.run(
        [*selected_command, "--units", "9"],
        check=True, env=environment, capture_output=True,
    )
    selected_metadata, selected_shared = plotting.load_results(selected_path)
    assert selected_metadata["unit_ids"] == [9]
    assert {path.name for path in (selected_path / "units").iterdir()} == {"9.npz"}
    selected_unit = plotting.load_unit(selected_path, selected_shared, 9)
    assert selected_unit["spike_frame_counts"].sum() == 1
    for key, value in plotting.load_unit(result_path, shared, 9).items():
        np.testing.assert_equal(selected_unit[key], value)

    # Refuse to mix a rerun's arrays with this manifest or existing units.
    manifest_bytes = (result_path / "metadata.json").read_bytes()
    rerun = subprocess.run(analysis_command, env=environment, capture_output=True)
    assert rerun.returncode != 0
    assert (result_path / "metadata.json").read_bytes() == manifest_bytes

    recording_inputs.rename(tmp_path / "unavailable_recording")
    subprocess.run(
        [sys.executable, str(script_directory / "spatial_cell_plotting.py"),
         str(result_path)],
        check=True, env=environment, capture_output=True,
    )
    figures = list((result_path / "plots").rglob("*.*"))
    assert len(figures) == 44
    assert all(path.stem in {"7", "9"} and path.stat().st_size > 0 for path in figures)


def test_missing_saved_camera_times_report_required_upstream_input(recording_inputs):
    (recording_inputs / "data/sync_data.json").write_text(json.dumps({
        "recording_interval": [[0, 10]],
    }))
    with pytest.raises(KeyError, match="No saved camera timestamps.*upstream"):
        spatial.load_data()


def test_pose_input_requires_processed_columns(recording_inputs):
    (recording_inputs / "260827.csv").write_text("Format Version,1.25,Take Name,example\n")
    with pytest.raises(ValueError, match="Usecols do not match columns"):
        spatial.load_data()


def test_plot_selection_uses_saved_unit_ids(session_maps, tmp_path, monkeypatch):
    monkeypatch.setattr(spatial, "save_root_directory", tmp_path)
    maps = spatial.compute_maps(session_maps["frame_times"], session_maps)
    metadata = spatial.save_session(session_maps, [7, 9], {})
    for unit_id in metadata["unit_ids"]:
        spatial.save_unit(maps, unit_id)
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    plotted = []
    monkeypatch.setattr(plotting, "MAP_PLOTS", ())
    monkeypatch.setattr(plotting, "plot_maps", lambda maps, unit, meta: plt.figure())
    monkeypatch.setattr(plotting, "save_figure", lambda fig, kind, unit, out: plotted.append(unit))
    plotting.main([str(tmp_path), "--units", "9", "9"])
    assert plotted == [9]
    with pytest.raises(SystemExit):
        plotting.main([str(tmp_path), "--units", "8"])
    assert plotted == [9]


def test_failed_analysis_has_no_completed_manifest(recording_inputs, tmp_path, monkeypatch):
    class FailedExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, unit_ids):
            raise RuntimeError("unit analysis failed")

    monkeypatch.setattr(spatial, "ProcessPoolExecutor", FailedExecutor)
    monkeypatch.setattr(spatial, "save_root_directory", tmp_path)
    monkeypatch.setattr(spatial, "worker_data", None)
    result_path = tmp_path / "incomplete"
    with pytest.raises(RuntimeError, match="unit analysis failed"):
        spatial.main(["--output", str(result_path)])
    assert (result_path / "session.npz").is_file()
    assert not (result_path / "metadata.json").exists()
    with pytest.raises(FileNotFoundError, match="No completed spatial analysis"):
        plotting.load_results(result_path)


@pytest.mark.parametrize("name, version", [("unknown", 1), ("rfmapping-spatial-cells", 2)])
def test_plotting_rejects_unknown_result_schema(tmp_path, name, version):
    (tmp_path / "metadata.json").write_text(json.dumps({
        "schema_name": name, "schema_version": version,
    }))
    with pytest.raises(ValueError, match="Unsupported spatial-cell result schema"):
        plotting.load_results(tmp_path)
