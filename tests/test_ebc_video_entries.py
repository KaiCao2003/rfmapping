import json

import numpy as np
import pytest

import ebc_video_circle as circle
import ebc_video_rectangle as rectangle
from Utils import ebc_workflow as workflow


@pytest.mark.parametrize("entry, adapter", [(rectangle, "prepare_rectangle_video"),
                                            (circle, "prepare_cylinder_video")])
def test_entries_use_explicit_config_and_shared_export(tmp_path, monkeypatch, entry, adapter):
    directory = tmp_path / "settings"
    directory.mkdir()
    path = directory / "recording.json"
    path.write_text(json.dumps(dict(session="../recording", pose_path="pose.csv", video_path="raw.avi",
                                    output_dir="rendered", kilosort_dir="sorting", probe="B",
                                    boundary=dict(bounds_px=[10, 200, 20, 180], size_cm=33))))
    loaded, exported = [], []
    prepared = {"prepared": True}

    def prepare(config):
        loaded.append(config)
        return prepared

    def export(data, config, **settings):
        exported.append((data, config, settings))
        return {"completed": True}

    monkeypatch.setattr(entry, adapter, prepare)
    monkeypatch.setattr(workflow, "export_prepared_video", export)
    monkeypatch.chdir(tmp_path)
    result = entry.main([str(path), "--start", "1.25", "--duration", ".5", "--silent"])
    assert result == {"completed": True}
    config = loaded[0]
    assert config["session"] == tmp_path / "recording"
    assert config["pose_path"] == directory / "pose.csv"
    assert config["video_path"] == directory / "raw.avi"
    assert config["output_dir"] == directory / "rendered"
    assert config["boundary"] == dict(bounds_px=[10, 200, 20, 180], size_cm=33)
    assert exported == [(prepared, config, dict(start_s=1.25, duration_s=.5, silent=True))]


@pytest.mark.parametrize("entry", [rectangle, circle])
def test_entries_require_a_config_instead_of_a_recording_default(entry):
    with pytest.raises(SystemExit) as error:
        entry.main([])
    assert error.value.code == 2


def test_audited_clock_keeps_declared_gaps_and_rejects_another_recording(tmp_path):
    session = tmp_path / "recording"
    qc = dict(session=str(session), adc_time_origin_s=100., decoded_video_frame_count=5,
              uncertain_video_frame_ranges_inclusive=[[2, 2]])
    (tmp_path / "video_clock_qc.json").write_text(json.dumps(qc))
    np.save(tmp_path / "video_adc_times.npy", [10., 10.04, np.nan, 10.12, 10.16])
    np.save(tmp_path / "video_motive_rows.npy", [0., 2., np.nan, 6., 8.])
    times, rows, origin, _ = workflow.load_video_clock(tmp_path, session)
    assert np.isnan(times[2]) and np.isnan(rows[2])
    assert origin == 100.
    assert workflow.measured_video_rate(times) == pytest.approx(25.)
    with pytest.raises(ValueError, match="different session"):
        workflow.load_video_clock(tmp_path, tmp_path / "other")


def test_shared_export_selects_frames_from_the_measured_clock(tmp_path, monkeypatch):
    data = dict(exposure_times=np.array([10., 10.04, np.nan, 10.12, 10.16]), video_frame_count=5)
    config = dict(video_path=tmp_path / "source.avi", output_dir=tmp_path / "result",
                  video_workers=2, video_encoder="libx264")
    calls, audits = [], []
    summary = dict(frames=3, source_first_frame=1, source_last_frame=3)

    def export(*args, **kwargs):
        calls.append((args, kwargs))
        return summary

    monkeypatch.setattr(workflow, "export_ebc_overlay", export)
    monkeypatch.setattr(workflow, "save_geometry_audit", lambda *args: audits.append(args))
    result = workflow.export_prepared_video(data, config, start_s=.04, duration_s=.12, silent=True)
    args, settings = calls[0]
    assert args == (data, config["video_path"], config["output_dir"] / "overlay.mp4")
    assert settings["frame_rate"] == pytest.approx(25.)
    assert (settings["first_frame"], settings["stop_frame"]) == (1, 4)
    assert result == summary
    assert audits == [(data, summary, config["output_dir"] / "frame_geometry.csv")]
