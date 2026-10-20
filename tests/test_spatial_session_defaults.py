import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LoadedInputs(Exception):
    pass


@pytest.mark.parametrize(
    "source",
    [
        PROJECT_ROOT / "EBB/spatial_cell_analysis.py",
        PROJECT_ROOT / "scripts/spatial_cell_analysis.py",
        PROJECT_ROOT / "Utils/spatial_cell_analysis.py",
        PROJECT_ROOT / "spatial_cell_analysis.py",
        "/home/kai/scripts/spatial_cell_analysis.py",
    ],
)
@pytest.mark.parametrize(
    "mouse,date,recording,expected_channel,expected_bounds",
    [
        ("m19", "260831", 2, 1, (370, 920, 210, 760)),
        ("m20", "260918", 11, 1, (370, 920, 210, 760)),
        ("m20", "260921", 11, 6, (225, 720, 280, 785)),
        ("m21", "261006", 11, 6, (225, 720, 280, 785)),
        ("m22", "261008", 11, 6, (225, 720, 280, 785)),
        ("m22", "261009", 3, 6, (230, 740, 335, 835)),
        ("m22", "261009", 4, 1, (364.4598693847656, 920.0496215820312, 218.121337890625, 763.5052490234375)),
    ],
)
def test_analysis_defaults_resolve_session_geometry_and_camera(
    source, mouse, date, recording, expected_channel, expected_bounds, monkeypatch, tmp_path,
):
    if not Path(source).is_file():
        pytest.skip("Production entry is only installed on hhw9l84")
    spec = importlib.util.spec_from_file_location("defaults_under_test", source)
    analysis = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, analysis)
    spec.loader.exec_module(analysis)
    analysis = sys.modules[spec.name]
    for name in ("x_min", "x_max", "y_min", "y_max", "cm_per_px"):
        monkeypatch.setattr(analysis, name, getattr(analysis, name))

    monkeypatch.setattr(analysis, "recording_root", tmp_path / mouse)
    monkeypatch.setattr(analysis, "date", date)
    monkeypatch.setattr(analysis, "recording_number", recording)
    captured = {}

    def fake_load_data(**kwargs):
        captured.update(kwargs)
        metadata = {"camera_timing": {
            "camera_output": "basler", "ttl_pulse_count": 2,
            "measured_rate_hz": 50., "timestamp_reference": "test",
        }}
        return pd.DataFrame(), np.array([0., .02]), np.array([]), np.array([], dtype=int), [7], metadata

    monkeypatch.setattr(analysis, "load_data", fake_load_data)
    monkeypatch.setattr(analysis, "prepare_session_maps", lambda *args: (_ for _ in ()).throw(LoadedInputs()))
    with pytest.raises(LoadedInputs):
        analysis.run_analysis(output=tmp_path / "unused.rfmap", workers=1)
    assert captured["camera_input_channel"] == expected_channel
    assert tuple(captured["bounds_px"]) == expected_bounds


@pytest.mark.parametrize(
    "source",
    [
        PROJECT_ROOT / "EBB/spatial_cell_analysis.py",
        PROJECT_ROOT / "scripts/spatial_cell_analysis.py",
        PROJECT_ROOT / "Utils/spatial_cell_analysis.py",
        PROJECT_ROOT / "spatial_cell_analysis.py",
        "/home/kai/scripts/spatial_cell_analysis.py",
    ],
)
def test_analysis_explicit_camera_and_geometry_override_session_defaults(source, monkeypatch, tmp_path):
    if not Path(source).is_file():
        pytest.skip("Production entry is only installed on hhw9l84")
    spec = importlib.util.spec_from_file_location("defaults_override_under_test", source)
    analysis = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, analysis)
    spec.loader.exec_module(analysis)
    analysis = sys.modules[spec.name]
    for name in ("x_min", "x_max", "y_min", "y_max", "cm_per_px"):
        monkeypatch.setattr(analysis, name, getattr(analysis, name))

    monkeypatch.setattr(analysis, "recording_root", tmp_path / "m22")
    monkeypatch.setattr(analysis, "date", "261009")
    monkeypatch.setattr(analysis, "recording_number", 3)
    captured = {}
    metadata = {"camera_timing": {
        "camera_output": "basler", "ttl_pulse_count": 2,
        "measured_rate_hz": 50., "timestamp_reference": "test",
    }}
    monkeypatch.setattr(analysis, "load_data", lambda **kwargs: (
        captured.update(kwargs) or pd.DataFrame(), np.array([0., .02]),
        np.array([]), np.array([], dtype=int), [7], metadata,
    ))
    monkeypatch.setattr(analysis, "prepare_session_maps", lambda *args: (_ for _ in ()).throw(LoadedInputs()))
    with pytest.raises(LoadedInputs):
        analysis.run_analysis(
            output=tmp_path / "unused.rfmap", workers=1,
            camera_input_channel=3, bounds_px=(1, 2, 3, 4),
        )
    assert captured["camera_input_channel"] == 3
    assert tuple(captured["bounds_px"]) == (1, 2, 3, 4)


@pytest.mark.parametrize(
    "source,session,bounds_arg,expected_bounds,expected_channel",
    [
        (
            "/home/kai/scripts/EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m19/260831/260831_2",
            None,
            (370, 920, 210, 760),
            1,
        ),
        (
            "/home/kai/scripts/EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m20/260918/260918_11",
            None,
            (370, 920, 210, 760),
            1,
        ),
        (
            "/home/kai/scripts/EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m20/260921/260921_11",
            None,
            (225, 720, 280, 785),
            6,
        ),
        (
            "/home/kai/scripts/EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m21/261006/261006_11",
            None,
            (225, 720, 280, 785),
            6,
        ),
        (
            "/home/kai/scripts/EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m22/261008/261008_11",
            None,
            (225, 720, 280, 785),
            6,
        ),
        (
            "/home/kai/scripts/EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m22/261009/261009_3",
            None,
            (230, 740, 335, 835),
            6,
        ),
        (
            PROJECT_ROOT / "EBB/ebc_analysis.py",
            "/mnt/senzailab/Kai/#Recording/m22/261009/261009_3",
            (370, 920, 210, 760),
            (370, 920, 210, 760),
            6,
        ),
    ],
)
def test_legacy_ebc_loader_forwards_geometry_and_session_camera(
    source, session, bounds_arg, expected_bounds, expected_channel, monkeypatch,
):
    if not Path(source).is_file():
        pytest.skip("Production adapter is only installed on hhw9l84")
    if str(source).startswith("/home/kai/scripts/"):
        monkeypatch.syspath_prepend("/home/kai/scripts")
        for name in tuple(sys.modules):
            if name.split(".")[0] in {"Utils", "HD", "EBB", "RF"}:
                monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("legacy_ebc_under_test", source)
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, adapter)
    spec.loader.exec_module(adapter)
    captured = {}

    def fake_load_data(**kwargs):
        captured.update(kwargs)
        pose = pd.DataFrame({
            "frame": [0, 1], "center_x": [400., 500.],
            "center_y": [400., 500.], "hd_deg": [0., 90.],
            "raw_center_x": [400., 500.], "raw_center_y": [400., 500.],
            "position_held": [False, False], "initial_outside": [False, False],
        })
        return pose, np.array([0., .02]), np.array([]), np.array([], dtype=int), [7], {"ok": True}

    monkeypatch.setattr(adapter.spatial, "load_data", fake_load_data)
    adapter.frame_counts = lambda spikes, clusters, good, times: (
        np.array([7]), np.zeros((1, len(times)), dtype=np.int32),
    )
    options = {} if bounds_arg is None else {"bounds_px": bounds_arg}
    result = adapter.load_basler_session(session, **options)
    assert tuple(captured["bounds_px"]) == expected_bounds
    assert captured["camera_input_channel"] == expected_channel
    assert result["unit_ids"].tolist() == [7]


@pytest.mark.parametrize(
    "mouse,date,recording,expected_channel,expected_bounds",
    [
        ("m19", "260831", 2, 1, (370, 920, 210, 760)),
        ("m20", "260918", 11, 1, (370, 920, 210, 760)),
        ("m20", "260921", 11, 6, (225, 720, 280, 785)),
        ("m21", "261006", 11, 6, (225, 720, 280, 785)),
        ("m22", "261008", 11, 6, (225, 720, 280, 785)),
        ("m22", "261009", 3, 6, (230, 740, 335, 835)),
        ("m22", "261009", 4, 1, (364.4598693847656, 920.0496215820312, 218.121337890625, 763.5052490234375)),
    ],
)
@pytest.mark.parametrize("source", [PROJECT_ROOT / "scripts/ebc_video.py", Path("/home/kai/scripts/ebc_video.py")])
def test_production_video_loader_resolves_all_audited_sessions(
    source, mouse, date, recording, expected_channel, expected_bounds, monkeypatch, tmp_path,
):
    if not source.is_file():
        pytest.skip("Production video entry is only installed on hhw9l84")
    module_root = Path("/home/kai/scripts") if str(source).startswith("/home/kai/scripts/") else PROJECT_ROOT
    monkeypatch.syspath_prepend(str(module_root))
    for name in tuple(sys.modules):
        if name.split(".")[0] in {"Utils", "HD", "EBB", "RF"}:
            monkeypatch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("video_defaults_under_test", source)
    video = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, video)
    spec.loader.exec_module(video)
    assert Path(video.spatial.__file__).resolve() == module_root / "EBB/spatial_cell_analysis.py"

    session = tmp_path / mouse / date / f"{date}_{recording}"
    kilosort = session / "kilosort" / "ProbeA" / f"kilosort_{recording}"
    kilosort.mkdir(parents=True)
    (session / "data").mkdir()
    pose = pd.DataFrame({
        "frame": [0, 1], "center_x": [400., 500.],
        "center_y": [400., 500.], "hd_deg": [0., 90.],
    })
    interval = pd.DataFrame({"interval_type": ["baseline"], "start": [0.], "end": [.02]})
    monkeypatch.setattr(video, "read_formatted_json", lambda *args: {"session_info": {}})
    camera = {}
    monkeypatch.setattr(
        video, "get_exposure_timestamps",
        lambda *args, **kwargs: (camera.update(kwargs) or np.array([0., .02]), 0., {"ok": True}),
    )
    monkeypatch.setattr(
        video.pd, "read_csv",
        lambda path, **kwargs: pose.copy() if Path(path).name.endswith(".csv")
        and Path(path).name != "interval_table.csv" else interval,
    )
    selected_bounds = {}
    def hold(raw, bounds):
        selected_bounds["bounds"] = tuple(bounds)
        return raw, np.zeros(len(raw), dtype=bool), np.zeros(len(raw), dtype=bool)
    monkeypatch.setattr(video, "hold_rectangle_positions", hold)

    data = video.load_video_data(session)
    assert selected_bounds["bounds"] == expected_bounds
    assert camera["camera_input_channel"] == expected_channel
    assert tuple(data["bounds_px"]) == expected_bounds
