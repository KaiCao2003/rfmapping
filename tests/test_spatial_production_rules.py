"""Keep every spatial entry on the production boundary and pose-file rules."""

import importlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def analysis(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "spatial_rules_under_test", PROJECT_ROOT / "EBB/spatial_cell_analysis.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mouse,date,recording,options,bounds,channel",
    [
        ("m99", "261014", 1, [], (225, 720, 280, 785), 6),
        ("m99", "261014", 1, ["--boundary", "old"], (370, 920, 210, 760), 6),
        ("m19", "260831", 2, [], (370, 920, 210, 760), 1),
        ("m22", "261009", 3, [], (230, 740, 335, 835), 6),
        ("m22", "261009", 4, [], (364.4598693847656, 920.0496215820312, 218.121337890625, 763.5052490234375), 1),
        ("m22", "261009", 4, ["--boundary", "old"], (364.4598693847656, 920.0496215820312, 218.121337890625, 763.5052490234375), 1),
        ("m22", "261009", 3, ["--boundary", "old"], (230, 740, 335, 835), 6),
        ("m22", "261009", 3,
         ["--bounds-px", "10", "20", "30", "40", "--camera-input-channel", "3"],
         (10, 20, 30, 40), 3),
    ],
)
def test_cli_resolves_production_boundary_precedence(
    analysis, monkeypatch, tmp_path, mouse, date, recording, options, bounds, channel,
):
    captured = {}
    monkeypatch.setattr(analysis, "run_analysis", lambda **kwargs: captured.update(kwargs))
    analysis.main([
        "--recording-root", str(tmp_path / mouse), "--date", date,
        "--recording-number", str(recording), *options,
    ])
    assert captured["bounds_px"] == bounds
    assert captured["camera_input_channel"] == channel
    assert (analysis.x_min, analysis.x_max, analysis.y_min, analysis.y_max) == bounds


@pytest.fixture
def recording(tmp_path):
    session = tmp_path / "m99/261014/261014_1"
    data = session / "data"
    (data / "probeA").mkdir(parents=True)
    kilosort = session / "kilosort/ProbeA/kilosort_1"
    kilosort.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [7, 9], "KSLabel": ["good", "good"]}).to_csv(
        kilosort / "cluster_KSLabel.tsv", sep="\t", index=False,
    )
    np.save(kilosort / "spike_clusters.npy", [7, 7])
    np.save(data / "probeA/adc_spike_time.npy", [100.02, 100.04])
    times = np.arange(5) * .02
    (data / "session_info.json").write_text(json.dumps({"session_info": {}}))
    (data / "sync_data.json").write_text(json.dumps({
        "exposure_sampling_number_list_mid": times.tolist(),
        "exposure_sampling_number_list_mid_raw": (times + 100).tolist(),
    }))
    pd.DataFrame({"interval_type": ["baseline"], "start": [0.], "end": [.08]}).to_csv(
        data / "interval_table.csv", index=False,
    )
    pose = pd.DataFrame({"frame": np.arange(5), "center_x": [500.] * 5,
                         "center_y": [500.] * 5, "hd_deg": [0., 90., 180., 270., 0.]})
    pose.to_csv(session / "261014.csv", index=False)
    pose.assign(center_x=600.).to_csv(session / "261014_yolo.csv", index=False)
    return session


@pytest.mark.parametrize("has_tak,basler,expected_x,filename", [
    (False, True, 500., "261014.csv"),
    (True, True, 600., "261014_yolo.csv"),
    (True, False, 500., "261014.csv"),
])
def test_pose_reader_uses_production_csv_rule(analysis, recording, has_tak, basler, expected_x, filename):
    if has_tak:
        (recording / "261014.tak").touch()
    pose, _, _, _, unit_ids, metadata = analysis.load_data(
        session_dir=recording, basler_output=basler, optihub2_output=not basler,
    )
    np.testing.assert_array_equal(pose["raw_center_x"], np.full(5, expected_x))
    assert Path(metadata["pose_path"]) == recording / filename
    assert unit_ids == [7, 9]


def test_missing_yolo_csv_is_not_replaced_by_motive_csv(analysis, recording):
    (recording / "261014.tak").touch()
    (recording / "261014_yolo.csv").unlink()
    with pytest.raises(FileNotFoundError, match="261014_yolo.csv"):
        analysis.load_data(session_dir=recording)


@pytest.mark.parametrize("module_name", ["scripts.spatial_cell_analysis", "Utils.spatial_cell_analysis"])
def test_historical_imports_share_canonical_module(module_name):
    from EBB import spatial_cell_analysis
    assert importlib.import_module(module_name) is spatial_cell_analysis


@pytest.mark.parametrize("relative_path", ["scripts/spatial_cell_analysis.py", "Utils/spatial_cell_analysis.py"])
def test_historical_cli_calls_canonical_main(relative_path, monkeypatch):
    from EBB import spatial_cell_analysis
    calls = []
    monkeypatch.setattr(spatial_cell_analysis, "main", lambda: calls.append(True))
    runpy.run_path(str(PROJECT_ROOT / relative_path), run_name="__main__")
    assert calls == [True]


def test_production_motive_alignment_keeps_original_timestamp_prefix():
    from Utils.motive_pose import matched_motive_frame_count
    exposures = np.array([1., 1.02, 1.04])
    timing = {}
    assert matched_motive_frame_count(np.arange(4), exposures, timing) == 3
    assert timing["motive_frame_alignment"] == "drop_single_trailing_motive_frame"
    with pytest.raises(ValueError, match="frame/TTL count mismatch"):
        matched_motive_frame_count(np.arange(5), exposures, {})
    assert matched_motive_frame_count(np.arange(5), exposures, {}, ignore_hd_ttl=True) == 3
    np.testing.assert_array_equal(exposures, [1., 1.02, 1.04])
