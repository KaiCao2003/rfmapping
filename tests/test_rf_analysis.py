import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

from Utils.direction_comparison import tc_loader
from Utils.rflocate import workflow as rf_analysis
from Utils.rflocate.workflow import analyze_rf_file, save_rf_unit_lists
from Utils.rflocate import RFMapList, RFResult, load_rf, load_rfmap, load_rf_tc, save_rf_tc
from Utils.rflocate.plotting import (
    export_rf_units, plot_rf_population, plot_rf_unit,
    rf_population_counts, rf_unit_plot_data,
)
from Utils.tc_preparation import prepare_rf_comparison


def _write_indexed_source(path, *, counts=None, occupancy=None, unit_ids=None):
    if counts is None:
        counts = np.full((3, 7, 30, 3), 4, dtype=np.uint32)
        counts[..., 0] = 300
        counts[0, 3, 5:9, 1:] = 24
        counts[1, 4, 19:23, 1:] = 26
        counts[1].reshape(210, 3)[[1, 2], 1:] = 0
        counts[2].reshape(210, 3)[[1, 2, 3], 1:] = 0
        occupancy = np.linspace(1.0, 2.0, 210).reshape(7, 30)
        occupancy[0, 0] = 0
        counts[:, 0, 0, :] = 0
        unit_ids = [7, 11, 23]
    if occupancy is None:
        occupancy = np.ones(counts.shape[1:3])
    if unit_ids is None:
        unit_ids = np.arange(counts.shape[0])
    metadata = {
        "unitsSpikeCountsSize": list(counts.shape),
        "responseUnits": "spike_count",
        "VSTimeWindow": [-0.1, 0.2],
    }
    arrays = dict(
        metadata=np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8),
        unitPool=np.asarray(unit_ids, dtype=np.int64),
        xPositions=np.arange(counts.shape[2], dtype=float) * 12.0,
        yPositions=np.arange(counts.shape[1], dtype=float) * 10.0,
        timeBinEdges=np.array([-0.1, 0.0, 0.1, 0.2]),
        occupancyTimeSec=np.asarray(occupancy),
        stimulusPresentationCounts=np.full(counts.shape[1:3], 2),
    )
    arrays.update({f"unit_{int(unit_id)}": values for unit_id, values in zip(unit_ids, counts)})
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return counts, occupancy


def _output_paths(source, rf_type="excitatory"):
    if rf_type == "inhibitory":
        source = source.with_name(source.stem + "_inhibitory" + source.suffix)
    return dict(
        result_2d=source.with_suffix(".npz"),
        result_1d=source.with_name(source.stem + "_1d.npz"),
        units_2d=source.with_name(source.stem + "_units_with_rf.npy"),
        units_1d=source.with_name(source.stem + "_units_with_rf_1d.npy"),
        summary=source.with_name(source.stem + "_analysis.json"),
    )


@pytest.mark.parametrize("collapse_from_2d", [False, True])
@pytest.mark.parametrize("rf_type", ["excitatory", "inhibitory"])
def test_analysis_matches_public_rf_views_and_preserves_rate_window(tmp_path, collapse_from_2d, rf_type):
    source = tmp_path / "regular.rfmap"
    counts, occupancy = _write_indexed_source(source)
    rf_options = {} if rf_type == "excitatory" else dict(rf_type=rf_type)
    result = analyze_rf_file(source, probe="B", collapse_from_2d=collapse_from_2d, **rf_options)
    assert result["raw"].unit_ids == [7]
    assert result["summed"].unit_ids == [7]
    assert result["raw"][0].metadata["responseNormalization"] == "presentation_count_time"
    assert result["summed"][0].time_window_s == (0.0, 0.2)
    expected_rates = counts[:1, ..., 1:].sum(axis=-1) / (2 * 0.2)
    np.testing.assert_allclose(result["summed"].to_2d_array(), expected_rates)

    raw = load_rfmap(source).to_firing_rate()
    expected = RFMapList([raw[0]], source).mean_rate(0.0, 0.2)
    options = dict(exclude_zero_bins=False, show_progress=False, rf_type=rf_type)
    z_2d = 1.5 if rf_type == "inhibitory" else 1.8
    z_1d = 0.75 if rf_type == "inhibitory" else 1.0
    mask_2d = expected.rf_2d(cluster_forming_z=z_2d, **options)
    center_2d = expected.rf_2d(cluster_forming_z=z_2d, is_center=True, **options)
    if collapse_from_2d:
        mask_1d = mask_2d.any(axis=1)
        center_1d = center_2d.any(axis=1)
    else:
        mask_1d = expected.rf_1d(cluster_forming_z=z_1d, **options)
        center_1d = expected.rf_1d(cluster_forming_z=z_1d, is_center=True, **options)
    for key, values in (
        ("mask_2d", mask_2d), ("center_2d", center_2d),
        ("mask_1d", mask_1d), ("center_1d", center_1d),
    ):
        np.testing.assert_array_equal(result[key], values)
    qc = result["bin_qc"]
    np.testing.assert_array_equal(qc["unit_ids"], [7, 11, 23])
    assert "missing_bins" not in qc
    np.testing.assert_array_equal(qc["zero_bins"], [1, 3, 4])
    np.testing.assert_array_equal(qc["valid_bins"], [209, 207, 206])
    np.testing.assert_array_equal(qc["keep"], [True, False, False])
    assert result["units_with_rf"] == [
        ("B", unit_id) for unit_id, present in zip([7], center_2d.any(axis=(1, 2))) if present
    ]
    assert result["units_with_rf_1d"] == [
        ("B", unit_id) for unit_id, present in zip([7], center_1d.any(axis=1)) if present
    ]
    expected_paths = _output_paths(source, rf_type)
    assert {key: Path(value) for key, value in result["output_paths"].items()} == expected_paths
    for path in expected_paths.values():
        assert path.is_file()
    for key, expected_mask, expected_center in (
        ("result_2d", mask_2d, center_2d),
        ("result_1d", mask_1d, center_1d),
    ):
        with np.load(expected_paths[key], allow_pickle=False) as saved:
            if key == "result_2d":
                np.testing.assert_array_equal(saved["mask_2d"], expected_mask)
                np.testing.assert_array_equal(saved["center_2d"], expected_center)
            else:
                np.testing.assert_array_equal(saved["mask_2d"].any(axis=1), expected_mask)
                np.testing.assert_array_equal(saved["center_2d"].any(axis=1), expected_center)
    summary = json.loads(expected_paths["summary"].read_text())
    assert summary["bin_qc"]["unit_ids"] == [7, 11, 23]
    assert summary["bin_qc"]["keep"] == [True, False, False]
    assert summary["parameters"]["time_range_s"] == [0.0, 0.2]
    assert summary["parameters"]["cluster_forming_z_2d"] == z_2d
    assert summary["parameters"]["cluster_forming_z_1d"] == z_1d
    assert summary["parameters"]["collapse_from_2d"] is collapse_from_2d
    assert summary["parameters"]["rf_type"] == rf_type


def test_collapsed_analysis_projects_existing_detection_without_another_detector(tmp_path, monkeypatch):
    source = tmp_path / "projected.rfmap"
    _write_indexed_source(source)

    def unexpected_detection(*args, **kwargs):
        raise AssertionError("projecting an existing RF must not call rf_1d detection")

    monkeypatch.setattr(RFMapList, "rf_1d", unexpected_detection)
    result = analyze_rf_file(
        source, probe="A", collapse_from_2d=True,
        cluster_forming_z_2d=1.8, cluster_forming_z_1d=100.0,
    )
    np.testing.assert_array_equal(result["mask_1d"], result["mask_2d"].any(axis=1))
    np.testing.assert_array_equal(result["center_1d"], result["center_2d"].any(axis=1))
    parent = load_rf(result["output_paths"]["result_2d"])
    projected = load_rf(result["output_paths"]["result_1d"])
    assert projected["manifest"]["collapse_axis"] == "x"
    assert projected["manifest"]["projection_source_cache_key"] == parent["cache_key"]


def test_inhibitory_analysis_detects_suppression_and_preserves_excitatory_outputs(tmp_path):
    source = tmp_path / "suppression.rfmap"
    counts = np.full((1, 7, 30, 3), 24, dtype=np.uint32)
    counts[0, :, 5:9, 1:] = 4
    counts[0, 0, 0, 1:] = 0
    _write_indexed_source(source, counts=counts, unit_ids=[17])
    excitatory = analyze_rf_file(source, probe="B")
    previous = {key: path.read_bytes() for key, path in excitatory["output_paths"].items()}

    inhibitory = analyze_rf_file(source, probe="B", rf_type="inhibitory")

    assert inhibitory["units_with_rf"] == [("B", 17)]
    assert inhibitory["units_with_rf_1d"] == [("B", 17)]
    assert inhibitory["mask_2d"][0, :, 5:9].all()
    assert inhibitory["mask_1d"][0, 5:9].all()
    assert not inhibitory["mask_2d"][0, 0, 0]
    assert inhibitory["output_paths"] == _output_paths(source, "inhibitory")
    assert {key: path.read_bytes() for key, path in excitatory["output_paths"].items()} == previous


@pytest.mark.parametrize("save_results", [False, True])
def test_both_analysis_loads_once_and_matches_separate_detections(tmp_path, monkeypatch, save_results):
    source = tmp_path / "both.rfmap"
    counts = np.full((2, 7, 30, 3), 24, dtype=np.uint32)
    counts[0] = 4
    counts[0, :, 5:9, 1:] = 24
    counts[1, :, 19:23, 1:] = 4
    _write_indexed_source(source, counts=counts, unit_ids=[7, 11])
    expected = {
        rf_type: analyze_rf_file(source, probe="B", rf_type=rf_type, save_results=False)
        for rf_type in ("excitatory", "inhibitory")
    }
    loads = []

    def load(path):
        loads.append(path)
        return load_rfmap(path)

    monkeypatch.setattr(rf_analysis, "load_rfmap", load)
    analyses = analyze_rf_file(source, probe="B", rf_type="both", save_results=save_results)

    assert loads == [source]
    assert analyses["excitatory"]["raw"] is analyses["inhibitory"]["raw"]
    assert analyses["excitatory"]["summed"] is analyses["inhibitory"]["summed"]
    assert analyses["excitatory"]["bin_qc"] is analyses["inhibitory"]["bin_qc"]
    for rf_type, analysis in analyses.items():
        for key in ("mask_2d", "center_2d", "mask_1d", "center_1d"):
            np.testing.assert_array_equal(analysis[key], expected[rf_type][key])
        for key in ("units_with_rf", "units_with_rf_1d", "output_paths"):
            assert analysis[key] == expected[rf_type][key]
        assert all(path.is_file() == save_results for path in analysis["output_paths"].values())
    if not save_results:
        assert set(tmp_path.iterdir()) == {source}


def test_analysis_without_saving_writes_no_outputs(tmp_path):
    source = tmp_path / "read_only.rfmap"
    _write_indexed_source(source)
    result = analyze_rf_file(source, probe="A", save_results=False)
    assert result["mask_2d"].shape == (1, 7, 30)
    assert set(tmp_path.iterdir()) == {source}
    assert not any(Path(path).exists() for path in result["output_paths"].values())


def test_analysis_saves_empty_unit_lists_without_pickle(tmp_path):
    source = tmp_path / "flat.rfmap"
    _write_indexed_source(source, counts=np.full((1, 7, 30, 3), 4), unit_ids=[17])
    result = analyze_rf_file(source, probe="B")
    assert result["units_with_rf"] == []
    assert result["units_with_rf_1d"] == []
    assert not result["mask_2d"].any()
    assert not result["mask_1d"].any()
    for key in ("units_2d", "units_1d"):
        saved = np.load(result["output_paths"][key], allow_pickle=False)
        assert saved.shape == (0, 2)
    summary = json.loads(Path(result["output_paths"]["summary"]).read_text())
    assert summary["units_with_rf"] == []
    assert summary["units_with_rf_1d"] == []


@pytest.mark.parametrize("save_results", [False, True])
def test_analysis_qc_failure_preserves_existing_outputs(tmp_path, save_results):
    source = tmp_path / "excluded.rfmap"
    _write_indexed_source(source, counts=np.zeros((1, 7, 30, 3)), unit_ids=[17])
    paths = _output_paths(source)
    previous = {}
    for key, path in paths.items():
        previous[key] = ("existing " + key).encode()
        path.write_bytes(previous[key])
    with pytest.raises(ValueError, match="(?i)(no units|no unit|all units)"):
        analyze_rf_file(source, probe="B", save_results=save_results)
    assert {key: path.read_bytes() for key, path in paths.items()} == previous


@pytest.mark.parametrize("collapse_from_2d", [False, True])
@pytest.mark.parametrize("rf_type", ["excitatory", "inhibitory"])
def test_cli_matches_direct_analysis_from_arbitrary_cwd_and_quoted_path(tmp_path, collapse_from_2d, rf_type):
    source = tmp_path / "map input 'quoted'.rfmap"
    _write_indexed_source(source)
    options = dict(
        time_range_s=(0.0, 0.1), max_zero_bins=2,
        cluster_forming_z_2d=1.2, cluster_forming_z_1d=0.8,
        drop_bins=1, wrap_x=False, collapse_from_2d=collapse_from_2d,
        rf_type=rf_type,
    )
    expected = analyze_rf_file(source, probe="C", save_results=False, **options)
    script = Path(__file__).resolve().parents[1] / "locate_rf.py"
    cwd = tmp_path / "unrelated working directory"
    cwd.mkdir()
    command = [
        sys.executable, str(script), str(source), "--probe", "C",
        "--time-range", "0.0", "0.1",
        "--max-zero-bins", "2", "--cluster-forming-z-2d", "1.2",
        "--cluster-forming-z-1d", "0.8", "--drop-bins", "1", "--no-wrap-x", "--no-plots",
    ]
    if collapse_from_2d:
        command.append("--collapse-from-2d")
    if rf_type == "inhibitory":
        command.extend(("--rf-type", rf_type))
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
    assert source.stem in completed.stdout
    assert not source.with_name(f"{source.stem}_figures").exists()
    paths = _output_paths(source, rf_type)
    with np.load(paths["result_2d"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["mask_2d"], expected["mask_2d"])
        np.testing.assert_array_equal(saved["center_2d"], expected["center_2d"])
    with np.load(paths["result_1d"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["mask_2d"].any(axis=1), expected["mask_1d"])
        np.testing.assert_array_equal(saved["center_2d"].any(axis=1), expected["center_1d"])
    summary = json.loads(paths["summary"].read_text())
    for key, value in options.items():
        assert summary["parameters"][key] == (list(value) if key == "time_range_s" else value)
    assert summary["units_with_rf"] == [list(pair) for pair in expected["units_with_rf"]]
    assert summary["units_with_rf_1d"] == [list(pair) for pair in expected["units_with_rf_1d"]]


def test_cli_qc_failure_returns_nonzero_without_outputs(tmp_path):
    source = tmp_path / "all excluded.rfmap"
    _write_indexed_source(source, counts=np.zeros((1, 7, 30, 3)), unit_ids=[17])
    script = Path(__file__).resolve().parents[1] / "locate_rf.py"
    completed = subprocess.run(
        [sys.executable, str(script), str(source), "--probe", "B"],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "ValueError" in completed.stderr
    assert set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("custom_plot_dir", [False, True])
def test_cli_both_saves_both_rf_types_and_source_specific_plots(tmp_path, custom_plot_dir):
    script = Path(__file__).resolve().parents[1] / "locate_rf.py"
    cwd = tmp_path / "unrelated working directory"
    cwd.mkdir()
    source_names = ["both input 'quoted'.rfmap"]
    if not custom_plot_dir:
        source_names.append("other stimulus.rfmap")
    previous_exports = {}
    for source_name in source_names:
        source = tmp_path / source_name
        _write_indexed_source(source)
        default_plot_dir = source.with_name(f"{source.stem}_figures")
        plot_dir = tmp_path / "custom figures 'quoted'" if custom_plot_dir else default_plot_dir
        command = [sys.executable, str(script), str(source), "--probe", "B", "--rf-type", "both"]
        if custom_plot_dir:
            command.extend(("--plot-output-dir", str(plot_dir)))
        completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
        expected_exports = set()
        for rf_type, z_2d, z_1d in (("excitatory", 1.8, 1.0), ("inhibitory", 1.5, 0.75)):
            assert f"ProbeB {rf_type}:" in completed.stdout
            paths = _output_paths(source, rf_type)
            assert all(path.is_file() for path in paths.values())
            summary = json.loads(paths["summary"].read_text())
            assert summary["parameters"]["rf_type"] == rf_type
            assert summary["parameters"]["cluster_forming_z_2d"] == z_2d
            assert summary["parameters"]["cluster_forming_z_1d"] == z_1d
            figure_dir = plot_dir / ("inhibitory/rfmap" if rf_type == "inhibitory" else "rfmap")
            expected_exports.update(
                figure_dir / f"{name}{suffix}"
                for name in ("ProbeB/rf_summary", "units/B/7")
                for suffix in (".png", ".svg")
            )
            assert not (figure_dir / "all").exists()
        assert set(plot_dir.rglob("*.png")) | set(plot_dir.rglob("*.svg")) == expected_exports
        for path in expected_exports:
            assert path.stat().st_size > 0
            if path.suffix == ".png":
                with Image.open(path) as image:
                    rgba = image.convert("RGBA")
                    assert rgba.getextrema()[3] == (255, 255)
                    assert rgba.getpixel((0, 0)) == (255, 255, 255, 255)
        if custom_plot_dir:
            assert not default_plot_dir.exists()
        assert {path: path.read_bytes() for path in previous_exports} == previous_exports
        previous_exports.update({path: path.read_bytes() for path in expected_exports})


@pytest.mark.parametrize("rf_type", ["excitatory", "inhibitory"])
@pytest.mark.parametrize("is_save", [False, True])
@pytest.mark.parametrize("probes", [("A",), ("A", "B")])
@pytest.mark.parametrize("rf_only", [False, True])
def test_notebook_saves_both_types_for_configured_probes(tmp_path, rf_type, is_save, probes, rf_only):
    notebook = Path(__file__).resolve().parents[1] / "locate_rf.ipynb"
    source = "".join(json.loads(notebook.read_text())["cells"][2]["source"])
    counts = np.full((3, 7, 30, 3), 24, dtype=np.uint32)
    counts[0] = 4
    counts[0, 2:4, 5:9, 1:] = 24
    counts[1, :, 19:23, 1:] = 4
    counts[2].reshape(210, 3)[[1, 2, 3], 1:] = 0
    sources = {}
    rf_dir = tmp_path / "260630/260630_3/data/rfmapping/good/-100_400_1ms"
    for probe in probes:
        probe_dir = rf_dir / f"Probe{probe}"
        probe_dir.mkdir(parents=True)
        sources[probe] = probe_dir / "regular_unitsSpikeCounts_260630_3.rfmap"
        _write_indexed_source(sources[probe], counts=counts, unit_ids=[7, 11, 23])

    namespace = dict(
        analyze_rf_file=analyze_rf_file, save_rf_unit_lists=save_rf_unit_lists,
        prepare_rf_comparison=prepare_rf_comparison,
        RFMapList=RFMapList, load_rfmap=load_rfmap, load_rf_tc=load_rf_tc, save_rf_tc=save_rf_tc,
        base_dir=tmp_path, probes=probes, is_rotation=False,
        date=260630, sessionID=3, rf_time_range=(0.0, 0.2),
        max_zero_bins=2, collapse_from_2d=False,
        is_save=is_save, rf_type=rf_type, rf_only=rf_only,
    )
    exec(compile(source, str(notebook), "exec"), namespace)
    data_dir = tmp_path / "260630/260630_3/data"
    expected_tc_paths = set()
    expected_comparison_paths = set()
    window_counts = counts[..., 1:].sum(axis=-1)
    for probe, rf_source in sources.items():
        analyses = namespace["rf_analyses_by_probe"][probe]
        for detection_type, unit_id in (("excitatory", 7), ("inhibitory", 11)):
            analysis = analyses[detection_type]
            paths = _output_paths(rf_source, detection_type)
            detected = analysis["result_2d"]
            assert isinstance(detected, RFResult)
            np.testing.assert_array_equal(detected.unit_ids, [7, 11])
            np.testing.assert_array_equal(detected.mask_2d, analysis["mask_2d"])
            np.testing.assert_array_equal(detected.center_2d, analysis["center_2d"])
            assert analysis["units_with_rf"] == [(probe, unit_id)]
            assert analysis["units_with_rf_1d"] == [(probe, unit_id)]
            assert all(path.is_file() == is_save for path in paths.values())
            suffix = "_inhibitory" if rf_only and detection_type == "inhibitory" else ""
            filename = f"{rf_source.stem}_Probe{probe}{suffix}_1d{'_rfonly' if rf_only else ''}.csv"
            tc_path = data_dir / filename
            expected_tc_paths.add(tc_path)
            assert tc_path.is_file()
            loaded = load_rf_tc(tc_path)
            expected_ids = [7, 11] if rf_only else [7, 11, 23]
            assert loaded.index.tolist() == expected_ids
            assert analysis["tc_1d"].index.tolist() == expected_ids
            np.testing.assert_array_equal(loaded.columns, np.arange(30) * 12.0)
            expected_tc = window_counts.sum(axis=1).astype(float)
            if rf_only:
                expected_tc = expected_tc[:2]
                for row, mask in enumerate(detected.mask_2d):
                    selected_rows = mask.any(axis=1)
                    expected_tc[row] = (
                        window_counts[row, selected_rows].sum(axis=0)
                        if selected_rows.any() else np.full(30, np.nan)
                    )
                empty_row = 1 if detection_type == "excitatory" else 0
                assert loaded.iloc[empty_row].isna().all()
            np.testing.assert_allclose(loaded.to_numpy(), expected_tc, equal_nan=True)
            np.testing.assert_allclose(analysis["tc_1d"].to_numpy(), expected_tc, equal_nan=True)
            comparison_path = data_dir / "tc_comparison" / (
                f"rf_{detection_type}_x_2d{'_rfonly' if rf_only else ''}_Probe{probe}.csv"
            )
            assert comparison_path.is_file() == is_save
            if is_save:
                expected_comparison_paths.add(comparison_path)
                comparison = tc_loader(comparison_path)
                assert comparison.index.tolist() == [f"{tmp_path.name}:260630:{probe}:{unit_id}"]
                np.testing.assert_array_equal(comparison.columns, loaded.columns)
                np.testing.assert_allclose(
                    comparison.to_numpy(), loaded.loc[[unit_id]].to_numpy(), equal_nan=True,
                )
                with np.load(paths["result_2d"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["mask_2d"], analysis["mask_2d"])
                    np.testing.assert_array_equal(saved["center_2d"], analysis["center_2d"])
                    np.testing.assert_array_equal(saved["unit_ids"], [7, 11])
                with np.load(paths["result_1d"], allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["mask_2d"][:, 0], analysis["mask_1d"])
                    np.testing.assert_array_equal(saved["center_2d"][:, 0], analysis["center_1d"])
                for key in ("units_2d", "units_1d"):
                    np.testing.assert_array_equal(np.load(paths[key], allow_pickle=False), [[probe, str(unit_id)]])
            else:
                assert set(rf_source.parent.iterdir()) == {rf_source}
    assert set(data_dir.glob("*.csv")) == expected_tc_paths
    assert set((data_dir / "tc_comparison").glob("*.csv")) == expected_comparison_paths
    for detection_type, unit_id in (("excitatory", 7), ("inhibitory", 11)):
        suffix = "_inhibitory" if detection_type == "inhibitory" else ""
        for stem in ("units_with_rf", "units_with_rf_1d"):
            list_dir = sources[probes[0]].parent if len(probes) == 1 else rf_dir
            path = list_dir / f"{stem}{suffix}.npy"
            assert path.is_file() == is_save
            if is_save:
                np.testing.assert_array_equal(
                    np.load(path, allow_pickle=False), [[probe, str(unit_id)] for probe in probes],
                )


def test_unit_preview_uses_selected_rf_and_exports_opaque_white(tmp_path):
    source = tmp_path / "suppression.rfmap"
    counts = np.full((1, 7, 30, 3), 24, dtype=np.uint32)
    counts[0, :, 5:9, 1:] = 4
    _write_indexed_source(source, counts=counts, unit_ids=[17])
    analysis = analyze_rf_file(source, probe="B", rf_type="inhibitory", save_results=False)
    output = tmp_path / "preview"
    with plt.rc_context({"figure.facecolor": "black", "axes.facecolor": "black", "text.color": "white"}):
        fig, axes = plot_rf_unit(rf_unit_plot_data(analysis, 17), output_path=output, show=False)
    np.testing.assert_array_equal(axes[0, 0].images[0].get_array(), analysis["summed"][0].to_2d_array())
    np.testing.assert_array_equal(axes[0, 1].images[0].get_array(), analysis["mask_2d"][0])
    np.testing.assert_array_equal(axes[1, 1].images[0].get_array()[0], analysis["mask_1d"][0])
    rows, columns = np.nonzero(analysis["center_2d"][0])
    np.testing.assert_array_equal(axes[0, 1].collections[0].get_offsets(), np.column_stack((columns, rows)))
    assert fig.get_facecolor() == (1, 1, 1, 1)
    assert all(ax.get_facecolor() == (1, 1, 1, 1) for ax in fig.axes)
    assert all(ax.title.get_color() == "black" for ax in axes.flat)
    assert output.with_suffix(".svg").is_file()
    with Image.open(output.with_suffix(".png")) as image:
        rgba = image.convert("RGBA")
        assert rgba.getextrema()[3] == (255, 255)
        assert rgba.getpixel((0, 0)) == (255, 255, 255, 255)


def test_population_and_unit_exports_use_all_probes_without_inline_figures(tmp_path):
    analyses_by_probe = {}
    for probe, start in (("A", 5), ("B", 17)):
        counts = np.full((1, 7, 30, 3), 24, dtype=np.uint32)
        counts[0, :, start:start + 4, 1:] = 4
        source = tmp_path / f"{probe}.rfmap"
        _write_indexed_source(source, counts=counts, unit_ids=[17])
        analyses_by_probe[probe] = analyze_rf_file(source, probe=probe, rf_type="both", save_results=False)
    output_dir = tmp_path / "figures"
    open_figures = set(plt.get_fignums())
    population_counts = rf_population_counts(analyses_by_probe, rf_type="inhibitory")
    population_counts["all_smoothed"] = {
        key: gaussian_filter(values.astype(float), sigma=1.0, mode="nearest")
        for key, values in population_counts["all"].items()
    }
    figures = plot_rf_population(
        population_counts, rf_type="inhibitory", output_dir=output_dir, show=False,
    )
    assert set(figures) == {"ProbeA", "ProbeB", "all", "all_smoothed"}
    for probe, analyses in analyses_by_probe.items():
        axes = figures[f"Probe{probe}"][1]
        for column, key in ((0, "mask_2d"), (1, "center_2d")):
            np.testing.assert_array_equal(axes[0, column].images[0].get_array(), analyses["inhibitory"][key].sum(axis=0))
    for column, key in ((0, "mask_1d"), (1, "center_1d")):
        expected = sum(analyses["inhibitory"][key].sum(axis=0) for analyses in analyses_by_probe.values())
        np.testing.assert_array_equal(figures["all"][1][1, column].lines[0].get_ydata(), expected)
    count = export_rf_units(analyses_by_probe, output_dir, rf_type="inhibitory")
    assert count == 2
    figure_dir = output_dir / "inhibitory/rfmap"
    for probe in ("A", "B"):
        for suffix in (".png", ".svg"):
            assert (figure_dir / "units" / probe / f"17{suffix}").is_file()
    assert (figure_dir / "all/rf_summary.png").is_file()
    assert set(plt.get_fignums()) == open_figures
