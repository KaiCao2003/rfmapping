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
from Utils.rfmap import load_rf_maps

import spatial_cell_analysis as spatial


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


def test_tuning_matrix_preserves_constant_firing_rates(session_maps):
    for values in session_maps.values():
        values.setflags(write=False)
    frames = session_maps["frame_times"]
    for spikes_per_frame in (1, 2, 0):
        rates = spatial.compute_tuning_matrix(
            np.repeat(frames, spikes_per_frame), session_maps,
        )
        assert rates.shape == (60, 20)
        assert np.isfinite(rates).any()
        np.testing.assert_allclose(rates[np.isfinite(rates)], spikes_per_frame * 2)


def test_unoccupied_bins_remain_nan():
    with np.errstate(divide="raise", invalid="raise"):
        rates = spatial.compute_rate_map(np.zeros((2, 2)), np.zeros((2, 2)), 1, "nearest")
    assert np.isnan(rates).all()


def test_rfmap_export_preserves_matrices_units_coordinates_and_nan(tmp_path):
    expected = np.array([
        [[0.125, np.nan], [0.0, 19.75], [3.5, 0.01]],
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
    ])
    session = {
        "distance_edges": np.array([0, 4, 10]),
        "theta_edges": np.array([0, 90, 180, 360]),
    }
    path = tmp_path / "egocentric_rate_map.rfmap"
    spatial.save_egocentric_rfmap(path, expected, session, [9, 7], [5.0, 8.5])
    assert list(tmp_path.iterdir()) == [path]
    payload = json.loads(path.read_text())
    assert payload["unitsSpikeCounts"][0][0][1] == [None]
    assert payload["responseUnits"] == "Hz"
    assert payload["xBinEdges"] == [0, 4, 10]
    assert payload["yBinEdges"] == [0, 90, 180, 360]
    for options in ({}, {"unit_firing_rate": False}):
        maps = load_rf_maps(path, **options)
        assert maps.shape == (2, 3, 2, 1)
        assert maps.unit_ids == [9, 7]
        np.testing.assert_array_equal(maps.to_2d_array(), expected)
        np.testing.assert_array_equal(maps[0].x_positions, [2, 7])
        np.testing.assert_array_equal(maps[0].y_positions, [45, 135, 270])
        np.testing.assert_array_equal(maps[0].time_bin_edges_s, [5.0, 8.5])


def test_analysis_saves_one_file_and_rerun_replaces_it(recording_inputs, tmp_path):
    result_path = tmp_path / "results" / "egocentric_rate_map.rfmap"
    label_path = recording_inputs / "kilosort/ProbeA/kilosort_10/cluster_KSLabel.tsv"
    pd.DataFrame({"cluster_id": [7, 9], "KSLabel": ["good", "good"]}).to_csv(
        label_path, sep="\t", index=False,
    )
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    analysis_command = [
        sys.executable, str(Path(spatial.__file__)),
        "--recording-root", str(tmp_path), "--date", "260827",
        "--recording-number", "10", "--probe", "A", "--phase", "baseline",
        "--output", str(result_path), "--workers", "2",
    ]
    subprocess.run(analysis_command, check=True, env=environment, capture_output=True)
    assert list(result_path.parent.iterdir()) == [result_path]
    maps = load_rf_maps(result_path)
    assert maps.unit_ids == [7, 9]
    expected = spatial.compute_tuning_matrix(
        np.array([0.25, 1.0]),
        spatial.prepare_session_maps(
            pd.DataFrame({
                "center_x": [500.0] * 3, "center_y": [500.0] * 3,
                "hd_deg": [90.0, 180.0, 270.0],
            }),
            np.array([0.25, 0.5, 1.0]),
        ),
    )
    np.testing.assert_array_equal(maps.by_unit_id(7).to_2d_array(), expected)

    subprocess.run(
        [*analysis_command, "--units", "9"], check=True, env=environment, capture_output=True,
    )
    assert list(result_path.parent.iterdir()) == [result_path]
    selected = load_rf_maps(result_path)
    assert selected.unit_ids == [9]
    np.testing.assert_array_equal(
        selected[0].to_2d_array(), maps.by_unit_id(9).to_2d_array(),
    )
    recording_inputs.rename(tmp_path / "unavailable_recording")
    np.testing.assert_array_equal(load_rf_maps(result_path).to_2d_array(), selected.to_2d_array())


@pytest.mark.parametrize("is_save", [False, True])
def test_plotting_notebook_displays_one_unit_and_saves_only_when_requested(
    tmp_path, is_save,
):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    path = tmp_path / "tuning.rfmap"
    values = np.arange(24, dtype=float).reshape(2, 4, 3)
    values[1, 0, 0] = np.nan
    spatial.save_egocentric_rfmap(
        path, values,
        {"distance_edges": np.linspace(0, 12, 4), "theta_edges": np.linspace(0, 360, 5)},
        [7, 9], [0.0, 1.0],
    )
    notebook_path = Path(spatial.__file__).with_name("spatial_cell_plotting.ipynb")
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)
    parameters = next(cell for cell in nb.cells if cell.id == "parameters")
    parameters.source = (
        f"result_path = Path({str(path)!r})\nunit_id = 9\nis_save = {is_save!r}\n"
        "save_path = result_path.with_name(f'unit_{unit_id}.png')\n"
    )
    plot_cell = next(cell for cell in nb.cells if cell.id == "plot")
    plot_cell.source = (
        "plt.style.use('dark_background')\nplt.rcParams['savefig.transparent'] = True\n"
        + plot_cell.source
        + "\nnp.testing.assert_array_equal(axis.images[0].get_array(), rfmap.to_2d_array())\n"
        "assert rfmap.unit_id == 9\n"
        "assert axis.get_xlabel() == 'Distance to boundary (cm)'\n"
        "assert axis.get_ylabel() == 'Egocentric angle (deg)'\n"
        "assert axis.get_ylim() == (0, 360)\n"
        "assert axis.get_xlim() == (0, 12)\n"
        "assert figure.get_facecolor() == (1, 1, 1, 1)\n"
        "assert axis.get_facecolor() == (1, 1, 1, 1)\n"
        "assert axis.xaxis.label.get_color() == 'black'\n"
    )
    client = NotebookClient(
        nb, timeout=60, kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    images = [
        output for cell in nb.cells for output in cell.get("outputs", [])
        if "image/png" in output.get("data", {})
    ]
    assert len(images) == 1
    expected_files = {path, tmp_path / "unit_9.png"} if is_save else {path}
    assert set(tmp_path.iterdir()) == expected_files
    if is_save:
        pixels = plt.imread(tmp_path / "unit_9.png")
        np.testing.assert_array_equal(pixels[0, 0], [1, 1, 1, 1])
        assert np.all(pixels[:, :, 3] == 1)


def test_load_data_reads_adc_when_saved_camera_times_are_missing(recording_inputs):
    adc_dir = recording_inputs / "node/experiment/recording/continuous/ADC"
    adc_dir.mkdir(parents=True)
    np.save(adc_dir / "timestamps.npy", 100 + np.arange(2100) / 1000)
    signal = np.zeros((2100, 2), dtype=np.int16)
    signal[:, 1] = 20000
    signal[:20, 1] = 0  # Basler stays low before and after acquisition.
    signal[2050:, 1] = 0
    for center in (50, 250, 500, 1000, 1250, 2000):
        signal[center - 10:center + 10, 1] = 0
    signal.tofile(adc_dir / "continuous.dat")
    (recording_inputs / "data/session_info.json").write_text(json.dumps({
        "session_info": {
            "base_path": str(recording_inputs), "record_nodes": "node",
            "experiment_id": "experiment", "recording_name": "recording",
            "continuous_ADC_folder": "ADC", "ADC_input_channel": 2,
        },
    }))
    (recording_inputs / "data/sync_data.json").write_text(json.dumps({
        "recording_interval": [[0, 10]],
    }))
    pose, times, spikes, clusters, units, source = spatial.load_data()
    np.testing.assert_array_equal(pose["frame"], [1, 2, 3])
    np.testing.assert_allclose(times, [0.25, 0.5, 1.0])
    np.testing.assert_allclose(spikes, [0.25, 0.75, 1.0])
    assert source["camera_timing"]["timestamp_reference"] == "adc_exposure_midpoint"
    assert source["camera_timing"]["camera_output"] == "basler"
    assert source["camera_timing"]["camera_ttl_active_high"] is False
    assert source["camera_timing"]["ttl_pulse_count"] == 6


@pytest.mark.parametrize("basler, optihub2", [(True, True), (False, False)])
def test_camera_output_requires_exactly_one_bool(recording_inputs, basler, optihub2):
    with pytest.raises(ValueError, match="Set exactly one"):
        spatial.load_data(basler_output=basler, optihub2_output=optihub2)


def test_only_optihub2_drops_one_trailing_motive_frame(recording_inputs):
    pd.DataFrame({
        "frame": np.arange(7), "center_x": 500.0, "center_y": 500.0, "hd_deg": 0.0,
    }).to_csv(recording_inputs / "260827.csv", index=False)
    pose, times, *_, source = spatial.load_data(basler_output=False, optihub2_output=True)
    np.testing.assert_array_equal(pose["frame"], [1, 2, 3, 4])
    np.testing.assert_array_equal(times, [0.25, 0.5, 1.0, 1.25])
    assert source["camera_timing"]["camera_output"] == "optihub2"
    assert source["camera_timing"]["dropped_trailing_motive_frame"] is True
    with pytest.raises(ValueError, match="Pose frame IDs exceed"):
        spatial.load_data(basler_output=True, optihub2_output=False)


def test_pose_input_requires_processed_columns(recording_inputs):
    (recording_inputs / "260827.csv").write_text("Format Version,1.25,Take Name,example\n")
    with pytest.raises(ValueError, match="Usecols do not match columns"):
        spatial.load_data()


def test_failed_analysis_keeps_previous_result(recording_inputs, tmp_path, monkeypatch):
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
    result_path = tmp_path / "result.rfmap"
    result_path.write_text("previous complete result")
    with pytest.raises(RuntimeError, match="unit analysis failed"):
        spatial.run_analysis(output=result_path)
    assert result_path.read_text() == "previous complete result"
    assert not result_path.with_name("result.rfmap.tmp").exists()
    assert spatial.worker_data is None
