"""Overlay geometry must be selected from the exact saved analysis samples."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from EBB import ebc_geometry, ebc_workflow as workflow


@pytest.fixture
def saved_analysis(tmp_path):
    directory = tmp_path / "analysis"
    directory.mkdir()
    frames = dict(
        frame_ids=np.array([1, 3, 4]), frame_times=np.array([.02, .06, .08]),
        exposure_times=np.arange(6) * .02,
        raw_xy_px=np.array([[50., 60.], [500., 50.], [np.nan, np.nan]]),
        corrected_xy_px=np.array([[50., 60.], [50., 60.], [np.nan, np.nan]]),
        position_held=np.array([False, True, False]), initial_outside=np.zeros(3, dtype=bool),
        hd_deg=np.array([0., 45., 90.]), ebc_hd_deg=np.array([0., 60., 90.]),
        theta_deg=np.arange(3., 360., 6.),
        # Deliberately unlike recomputed wall distances, to catch hidden calculation.
        theta_d_cm=np.arange(180, dtype=float).reshape(3, 60) + .125,
    )
    frames["theta_d_cm"][2] = np.nan
    audit = dict(bounds_px=[10., 210., 20., 120.], arena_size_cm=[40., 40.],
                 probe_name="A", phase_key="baseline", source=dict(
                     adc_time_origin_s=123., camera_timing={"source": "saved clock"},
                     kilosort_dir="absent-sorting", spike_times_path="absent-spikes.npy",
                     selected_interval_s=[.02, .08]))
    np.savez_compressed(directory / "position_correction.npz", **frames)
    (directory / "audit.json").write_text(json.dumps(audit))
    return directory, frames, audit


def test_saved_overlay_uses_eight_native_columns_without_loading_raw_data_or_intersections(
    saved_analysis, monkeypatch,
):
    directory, frames, audit = saved_analysis
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Overlay must not calculate wall intersections")

    monkeypatch.setattr(workflow, "compute_ebc_rays", forbidden)
    monkeypatch.setattr(ebc_geometry, "_rectangle_intersections", forbidden)
    data = workflow.prepare_rectangle_video(dict(analysis_dir=directory, session=directory.parent))
    selected = np.array([0, 7, 15, 22, 30, 37, 45, 52])
    info = data["ebc_info"]
    np.testing.assert_array_equal(info.distances_cm, frames["theta_d_cm"][:, selected])
    np.testing.assert_array_equal(info.bearings_deg, [3., 45., 93., 135., -177., -135., -87., -45.])
    np.testing.assert_array_equal(data["frame_ids"], frames["frame_ids"])
    np.testing.assert_array_equal(data["times"], frames["frame_times"])
    np.testing.assert_array_equal(data["exposure_times"], frames["exposure_times"])
    np.testing.assert_array_equal(data["position_held"], frames["position_held"])
    np.testing.assert_array_equal(data["hd"], (90 - frames["ebc_hd_deg"]) % 360)
    assert data["adc_time_origin_s"] == 123.
    assert data["provenance"]["selected_angle_indices"] == selected.tolist()
    np.testing.assert_allclose(data["xy"][:2], [[8, 24], [8, 24]])
    # Projection uses the saved calibrated heading, not the raw 45-degree heading.
    angle = np.deg2rad(90. - frames["ebc_hd_deg"][1] - info.bearings_deg)
    np.testing.assert_allclose(info.endpoints_cm[1, :, :2],
                               [8, 24] + info.distances_cm[1, :, None] * np.c_[np.cos(angle), np.sin(angle)])
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    assert before == after


def test_missing_saved_geometry_never_triggers_generation(tmp_path):
    with pytest.raises(FileNotFoundError):
        workflow.load_saved_ebc_frames(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_geometry_audit_contains_the_saved_samples_and_preserves_missing_frames(saved_analysis):
    directory, frames, _ = saved_analysis
    data = workflow.prepare_rectangle_video(dict(analysis_dir=directory, session=directory.parent))
    target = directory / "frame_geometry.csv"
    workflow.save_geometry_audit(data, dict(source_first_frame=0, source_last_frame=5), target)
    import pandas as pd
    saved = pd.read_csv(target)
    np.testing.assert_array_equal(saved.video_frame, np.arange(6))
    assert saved.loc[[0, 2, 5], "x_cm"].isna().all()
    for column, index in zip(data["ebc_info"].bearings_deg,
                             data["provenance"]["selected_angle_indices"]):
        np.testing.assert_array_equal(saved.loc[frames["frame_ids"], f"ray_{column:g}_distance_cm"],
                                       frames["theta_d_cm"][:, index])


def test_pipeline_video_entry_only_passes_saved_analysis_directory(tmp_path, monkeypatch):
    from scripts import ebc_video as entry

    config_seen = []
    output = tmp_path / "video"
    args = SimpleNamespace(analysis_dir=None, probe="B", phase="test_phase", workers=2,
                           audio_workers=2, video_encoder="libx264", start=1., duration=2., silent=True)

    def run(command, **kwargs):
        config = json.loads(Path(command[2]).read_text())
        config_seen.append(config)
        (output / "overlay.json").write_text('{"frames": 100}')

    monkeypatch.setattr(entry.subprocess, "run", run)
    assert entry.export_rectangle_with_shared_runner(args, tmp_path, output) == {"frames": 100}
    config = config_seen[0]
    assert config["analysis_dir"] == str(tmp_path / "data/spatial_cells/ProbeB/test_phase")
    assert not {"boundary", "pose", "pose_path", "camera_timing", "interval_s"} & config.keys()
    with pytest.raises(SystemExit) as error:
        entry.main([str(tmp_path), "--arena-type", "cylinder"])
    assert error.value.code == 2
