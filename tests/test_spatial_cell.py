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


def test_four_rfmap_exports_partition_bin_centers_and_preserve_missing_values(tmp_path):
    values = np.array([
        [[1, 2, 4, 8], [np.nan, 3, np.nan, 5], [np.nan] * 4, [0] * 4],
        [[2, 4, 8, 16], [1, np.nan, 3, 4], [0] * 4, [np.nan] * 4],
    ], dtype=float)
    original = values.copy()
    session = {
        # Centers at 8 and 16 cm belong to the lower of the adjacent bands.
        "distance_edges": np.array([0, 4, 12, 20, 24]),
        "theta_edges": np.linspace(0, 360, 5),
    }
    paths = spatial.save_egocentric_rfmaps(
        tmp_path / "tuning.rfmap", values, session, [9, 7], [5.0, 8.5],
    )
    assert [path.name for path in paths] == [
        "tuning.rfmap", "tuning_0-8.rfmap",
        "tuning_8-16.rfmap", "tuning_16-.rfmap",
    ]
    assert set(tmp_path.iterdir()) == set(paths)
    np.testing.assert_array_equal(values, original)
    np.testing.assert_array_equal(load_rf_maps(paths[0]).to_2d_array(), values)
    expected_curves = [
        [[3, 3, np.nan, 0], [6, 1, 0, np.nan]],
        [[4, np.nan, np.nan, 0], [8, 3, 0, np.nan]],
        [[8, 5, np.nan, 0], [16, 4, 0, np.nan]],
    ]
    for path, expected, bounds, centers in zip(
        paths[1:], expected_curves, ([0, 8], [8, 16], [16, 24]),
        ([2, 8], [16], [22]),
    ):
        payload = json.loads(path.read_text())
        assert payload["unitsSpikeCounts"][0][2][0] == [None]
        assert "distanceRangeCm" not in payload
        assert payload["distanceSelection"] == "bin_centers"
        assert payload["distanceBinCentersCm"] == centers
        assert payload["responseAggregation"] == "sum_over_distance"
        assert payload["xUnits"] == "cm"
        assert payload["xBinEdges"] == bounds
        assert payload["yBinEdges"] == [0, 90, 180, 270, 360]
        for options in ({}, {"unit_firing_rate": False}):
            maps = load_rf_maps(path, **options)
            assert maps.shape == (2, 4, 1, 1)
            assert maps.unit_ids == [9, 7]
            np.testing.assert_array_equal(maps.to_2d_array()[:, :, 0], expected)
            np.testing.assert_array_equal(maps[0].x_positions, [np.mean(bounds)])
            np.testing.assert_array_equal(maps[0].y_positions, [45, 135, 225, 315])
            np.testing.assert_array_equal(maps[0].time_bin_edges_s, [5.0, 8.5])
    curves = np.stack([load_rf_maps(path).to_2d_array()[:, :, 0] for path in paths[1:]])
    np.testing.assert_array_equal(np.nansum(curves, axis=0), np.nansum(values, axis=2))


def test_empty_distance_bands_save_missing_curves(tmp_path):
    paths = spatial.save_egocentric_rfmaps(
        tmp_path / "tuning.rfmap", np.ones((1, 4, 1)),
        {"distance_edges": np.array([0, 4]), "theta_edges": np.linspace(0, 360, 5)},
        [7], [0, 1],
    )
    np.testing.assert_array_equal(load_rf_maps(paths[1]).to_2d_array(), np.ones((1, 4, 1)))
    for path, bounds in zip(paths[2:], ([8, 16], [16, 16])):
        maps = load_rf_maps(path)
        assert maps.shape == (1, 4, 1, 1)
        assert np.isnan(maps.to_2d_array()).all()
        assert maps[0].metadata["distanceBinCentersCm"] == []
        assert maps[0].metadata["xBinEdges"] == bounds


def test_analysis_saves_four_files_and_rerun_replaces_them(recording_inputs, tmp_path):
    result_path = tmp_path / "results" / "tuning.rfmap"
    result_paths = [
        result_path, result_path.with_name("tuning_0-8.rfmap"),
        result_path.with_name("tuning_8-16.rfmap"),
        result_path.with_name("tuning_16-.rfmap"),
    ]
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
    assert set(result_path.parent.iterdir()) == set(result_paths)
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
    first_run = {path: load_rf_maps(path) for path in result_paths}
    for path, band_maps in list(first_run.items())[1:]:
        assert band_maps.unit_ids == [7, 9]
        lower, upper = band_maps[0].metadata["xBinEdges"]
        distances = maps[0].x_positions
        selected_distances = (distances > lower) & (distances <= upper)
        np.testing.assert_allclose(
            band_maps.by_unit_id(7).to_2d_array()[:, 0],
            expected[:, selected_distances].sum(axis=1),
        )

    subprocess.run(
        [*analysis_command, "--units", "9"], check=True, env=environment, capture_output=True,
    )
    assert set(result_path.parent.iterdir()) == set(result_paths)
    recording_inputs.rename(tmp_path / "unavailable_recording")
    for path in result_paths:
        selected = load_rf_maps(path)
        assert selected.unit_ids == [9]
        np.testing.assert_array_equal(
            selected[0].to_2d_array(), first_run[path].by_unit_id(9).to_2d_array(),
        )


@pytest.mark.parametrize("is_save", [False, True])
def test_plotting_notebook_displays_unit_and_population_and_saves_when_requested(
    tmp_path, is_save,
):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    path = tmp_path / "tuning.rfmap"
    values = np.array([
        [[2, np.nan, 4], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
        [[np.nan, 0, 0], [1, 1, 1], [0, 0, 0], [0, 0, 0]],
        np.zeros((4, 3)),
        [[1, 1, 1], [2, 2, 2], [np.nan, np.nan, np.nan], [4, 4, 4]],
        np.full((4, 3), np.nan),
    ])
    spatial.save_egocentric_rfmap(
        path, values,
        {"distance_edges": np.linspace(0, 12, 4), "theta_edges": np.linspace(0, 360, 5)},
        [7, 9, 11, 13, 19], [0.0, 1.0],
    )
    notebook_path = Path(spatial.__file__).with_name("spatial_cell_plotting.ipynb")
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)
    parameters = next(cell for cell in nb.cells if cell.id == "parameters")
    parameters.source = (
        f"result_path = Path({str(path)!r})\nrf_maps = load_rf_maps(result_path)\n"
    )
    plot_cell = next(cell for cell in nb.cells if cell.id == "plot")
    plot_cell.source = (
        f"unit_id = 9\nis_save = {is_save!r}\n"
        "save_path = result_path.with_name(f'unit_{unit_id}.png')\n"
        "plt.style.use('dark_background')\nplt.rcParams['savefig.transparent'] = True\n"
        + plot_cell.source[plot_cell.source.index("rfmap = rf_maps.by_unit_id"):]
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
    heatmap_cell = next(cell for cell in nb.cells if cell.id == "all-units-heatmap")
    heatmap_cell.source = heatmap_cell.source.replace(
        "is_save_heatmap = False", f"is_save_heatmap = {is_save!r}",
    ) + "\n" + "\n".join([
        "np.testing.assert_array_equal(sorted_unit_ids, [9, 7, 13])",
        "np.testing.assert_array_equal(excluded_unit_ids, [11, 19])",
        "np.testing.assert_array_equal(unit_ids, [7, 9, 13])",
        "assert rate_maps.shape == (3, 4, 3)",
        "assert angle_profiles.shape == (3, 4)",
        "assert peak_bin.shape == (3,)",
        "expected_heatmap = np.array([",
        "    [1, 0, 0, 0], [0, 1, 0.5, 0], [0.5, 0.25, 1, np.nan],",
        "])",
        "np.testing.assert_array_equal(",
        "    np.ma.filled(heatmap_axis.images[0].get_array(), np.nan), expected_heatmap,",
        ")",
        "assert heatmap_axis.get_xlim() == (-180, 180)",
        "assert heatmap_axis.get_ylim() == (2.5, -0.5)",
        "assert [tick.get_text() for tick in heatmap_axis.get_xticklabels()] == ['180', '90', '0', '270', '180']",
        "assert [tick.get_text() for tick in heatmap_axis.get_yticklabels()] == ['9', '7', '13']",
        "assert heatmap_figure.axes[1].get_ylabel() == 'Normalized response'",
        "assert heatmap_axis.images[0].get_clim() == (0, 1)",
        "assert heatmap_figure.get_facecolor() == (1, 1, 1, 1)",
        "assert heatmap_axis.get_facecolor() == (1, 1, 1, 1)",
        "assert heatmap_axis.xaxis.label.get_color() == 'black'",
    ])
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
    assert len(images) == 2
    image_paths = [tmp_path / "unit_9.png", tmp_path / "all_units_angle_heatmap.png"]
    expected_files = {path, *image_paths} if is_save else {path}
    assert set(tmp_path.iterdir()) == expected_files
    if is_save:
        for image_path in image_paths:
            pixels = plt.imread(image_path)
            np.testing.assert_array_equal(pixels[0, 0], [1, 1, 1, 1])
            assert np.all(pixels[:, :, 3] == 1)


@pytest.mark.parametrize(
    "distance_band, column, bounds",
    [((None, 8), 0, (0, 8)), ((8, 16), 1, (8, 16)), ((16, None), 2, (16, 24))],
)
def test_plotting_notebook_reads_distance_bearing_maps(tmp_path, distance_band, column, bounds):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    path = tmp_path / "bearing.rfmap"
    values = np.arange(1, 25, dtype=float).reshape(2, 4, 3)
    spatial.save_egocentric_rfmap(
        path, values,
        {"distance_edges": np.array([0, 4, 20, 24]), "theta_edges": np.linspace(0, 360, 5)},
        [7, 9], [0, 1], distance_band_cm=distance_band,
    )
    notebook_path = Path(spatial.__file__).with_name("spatial_cell_plotting.ipynb")
    nb = nbformat.read(notebook_path, as_version=4)
    parameters = next(cell for cell in nb.cells if cell.id == "parameters")
    parameters.source = f"result_path = Path({str(path)!r})\nrf_maps = load_rf_maps(result_path)\n"
    plot_cell = next(cell for cell in nb.cells if cell.id == "plot")
    plot_cell.source = (
        "unit_id = 9\nis_save = False\n"
        + plot_cell.source[plot_cell.source.index("rfmap = rf_maps.by_unit_id"):]
        + f"\nassert axis.get_xlim() == {bounds!r}\n"
        "assert axis.get_xlabel() == 'Distance to boundary (cm)'\n"
        "assert axis.get_ylim() == (0, 360)\n"
        "assert figure.axes[1].get_ylabel() == 'Hz'\n"
        f"np.testing.assert_array_equal(axis.images[0].get_array()[:, 0], {values[1, :, column].tolist()!r})\n"
    )
    heatmap_cell = next(cell for cell in nb.cells if cell.id == "all-units-heatmap")
    heatmap_cell.source += (
        f"\nnp.testing.assert_array_equal(angle_profiles, {values[:, :, column].tolist()!r})\n"
    )
    client = NotebookClient(
        nb, timeout=60, kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    assert set(tmp_path.iterdir()) == {path}


def test_population_sort_matches_hd_rf_reference_and_rendered_peak_positions(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    notebook_path = Path(spatial.__file__).with_name("spatial_cell_plotting.ipynb")
    from Utils.direction_comparison import to_rf_angles, profile_table, sorted_by_peak
    # Equal native peaks and unsorted IDs expose both RF-reference tie rules.
    profiles = np.array([
        [0, 0, 10, 10], [0, 0, 0, 5], [0, 0, 10, 0], [10, 0, 0, 0],
        [0, 0, 0, 0], [np.nan] * 4,
    ])
    unit_ids = [40, 9, 7, 13, 99, 100]
    angles = np.array([45, 135, 225, 315])
    keys = [("A", unit_id) for unit_id in unit_ids[:4]]
    peak_by_key = {key: angles[np.argmax(profile)] for key, profile in zip(keys, profiles[:4])}
    reference = profile_table(profiles[:4], unit_ids[:4], to_rf_angles(angles), probe="A")
    ordered_keys = sorted_by_peak(reference, keys)
    expected_ids = [unit_id for _, unit_id in ordered_keys]
    assert expected_ids == [13, 9, 7, 40]
    to_layout_x = to_rf_angles
    columns = np.argsort(to_layout_x(angles))
    expected_rows = np.array([profiles[unit_ids.index(unit_id), columns] for unit_id in expected_ids])
    expected_rows /= expected_rows.max(axis=1, keepdims=True)
    peak_x = [float(to_layout_x(peak_by_key[key])) for key in ordered_keys]
    path = tmp_path / "tuning.rfmap"
    spatial.save_egocentric_rfmap(
        path, profiles[:, :, None],
        {"distance_edges": np.array([0, 8]), "theta_edges": np.linspace(0, 360, 5)},
        unit_ids, [0, 1],
    )
    nb = nbformat.read(notebook_path, as_version=4)
    nb.cells = [cell for cell in nb.cells if cell.id in {"imports", "parameters", "all-units-heatmap"}]
    parameters = next(cell for cell in nb.cells if cell.id == "parameters")
    parameters.source = (
        f"result_path = Path({str(path)!r})\nrf_maps = load_rf_maps(result_path)\n"
        "plt.rcParams['image.origin'] = 'lower'\n"
    )
    heatmap_cell = next(cell for cell in nb.cells if cell.id == "all-units-heatmap")
    heatmap_cell.source += "\n" + "\n".join([
        f"np.testing.assert_array_equal(sorted_unit_ids, {expected_ids!r})",
        "np.testing.assert_array_equal(excluded_unit_ids, [99, 100])",
        f"np.testing.assert_array_equal(heatmap_axis.images[0].get_array(), {expected_rows.tolist()!r})",
        "from matplotlib.backend_bases import MouseEvent",
        "heatmap_figure.canvas.draw()",
        f"for row, peak_x in enumerate({peak_x!r}):",
        "    px, py = heatmap_axis.transData.transform((peak_x, row))",
        "    event = MouseEvent('motion_notify_event', heatmap_figure.canvas, px, py)",
        "    assert heatmap_axis.images[0].get_cursor_data(event) == 1.0",
    ])
    client = NotebookClient(
        nb, timeout=60, kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()


def test_population_heatmap_explains_when_all_profiles_are_zero_or_missing(tmp_path):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    path = tmp_path / "empty.rfmap"
    spatial.save_egocentric_rfmap(
        path, [np.zeros((4, 1)), np.full((4, 1), np.nan)],
        {"distance_edges": np.array([0, 8]), "theta_edges": np.linspace(0, 360, 5)},
        [7, 9], [0, 1],
    )
    notebook_path = Path(spatial.__file__).with_name("spatial_cell_plotting.ipynb")
    nb = nbformat.read(notebook_path, as_version=4)
    nb.cells = [cell for cell in nb.cells if cell.id in {"imports", "parameters", "all-units-heatmap"}]
    parameters = next(cell for cell in nb.cells if cell.id == "parameters")
    parameters.source = f"result_path = Path({str(path)!r})\nrf_maps = load_rf_maps(result_path)\n"
    heatmap_cell = next(cell for cell in nb.cells if cell.id == "all-units-heatmap")
    heatmap_cell.source += "\n" + "\n".join([
        "assert n_units == 0",
        "np.testing.assert_array_equal(excluded_unit_ids, [7, 9])",
        "assert rate_maps.shape == (0, 4, 1)",
        "assert angle_profiles.shape == (0, 4)",
        "assert peak_bin.size == 0",
        "assert len(heatmap_axis.images) == 0",
        "assert heatmap_axis.texts[0].get_text() == 'No nonzero responses in this RFMap.'",
        "assert heatmap_figure.get_facecolor() == (1, 1, 1, 1)",
        "assert heatmap_axis.get_facecolor() == (1, 1, 1, 1)",
    ])
    client = NotebookClient(
        nb, timeout=60, kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.km = client.create_kernel_manager()
    client.km.kernel_spec.argv[0] = sys.executable
    client.execute()
    assert set(tmp_path.iterdir()) == {path}


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


def test_failed_analysis_keeps_all_previous_results(recording_inputs, tmp_path, monkeypatch):
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
    result_paths = [
        result_path, result_path.with_name("result_0-8.rfmap"),
        result_path.with_name("result_8-16.rfmap"),
        result_path.with_name("result_16-.rfmap"),
    ]
    for path in result_paths:
        path.write_text("previous complete result")
    with pytest.raises(RuntimeError, match="unit analysis failed"):
        spatial.run_analysis(output=result_path)
    for path in result_paths:
        assert path.read_text() == "previous complete result"
        assert not path.with_name(path.name + ".tmp").exists()
    assert spatial.worker_data is None
