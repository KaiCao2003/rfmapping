from pathlib import Path

import numpy as np
import pytest

import ebc_video_rectangle as rectangle
import ebc_video_sep as sep
from Utils import ebc_video


def test_rectangle_entry_preserves_arena_and_audio_settings(tmp_path, monkeypatch):
    session = tmp_path / "260921_11"
    output = tmp_path / "videos"
    loaded, exported = [], []
    data = {"session": session}

    def load(selected_session, **settings):
        loaded.append((selected_session, settings))
        return data

    def export(*args, **settings):
        exported.append((args, settings))
        return {12: output / "12.mp4"}

    monkeypatch.setattr(rectangle.video, "load_video_data", load)
    monkeypatch.setattr(rectangle.video, "export_good_unit_videos", export)
    monkeypatch.setattr(rectangle, "BASLER_BOUNDS_PX", (10., 510., 20., 620.))
    result = rectangle.main([str(session), "--probe", "B", "--phase", "recovery",
                             "--output-dir", str(output)])
    assert loaded == [(session, dict(probe="B", phase="recovery",
                                    bounds_px=(10., 510., 20., 620.), arena_size_cm=41.))]
    assert exported[0][0] == (data, session / "260921.avi", output)
    assert exported[0][1]["duration_s"] is None
    assert exported[0][1]["audio_source"] == rectangle.audio_source
    assert exported[0][1]["gain"] == rectangle.audio_gain
    assert result == {12: output / "12.mp4"}


def test_rectangle_geometry_uses_the_recording_bounds_instead_of_module_defaults():
    data = dict(xy=np.array([[10., 10.]]), hd=np.array([0.]),
                bounds_px=(100., 500., 200., 800.), arena_size_cm=20.)
    positions, endpoints, distances, valid = ebc_video.overlay_geometry(data)
    np.testing.assert_allclose(positions, [[300., 500.]])
    np.testing.assert_allclose(endpoints[0, 0], [300., 200.])
    np.testing.assert_allclose(distances[0, 0], 10.)
    assert valid.all()


def test_sep_entry_exports_and_renders_the_same_unit_and_time_range(tmp_path, monkeypatch):
    session, output = tmp_path / "260922_3", tmp_path / "sep"
    exported, rendered = [], []
    monkeypatch.setattr(sep, "export_session", lambda *args, **kwargs: exported.append((args, kwargs)))
    monkeypatch.setattr(sep, "render", lambda *args, **kwargs: rendered.append((args, kwargs)) or {"frames": 150})
    result = sep.main([str(session), "--unit", "318", "--probe", "B", "--output-dir", str(output),
                       "--start", "500", "--duration", "6", "--fps", "25"])
    replay = output / "session_overlay.json"
    assert exported == [((session, 318, replay), dict(probe="B", fps=25., preview_start=500.))]
    assert rendered == [((replay, output), dict(start_s=500., duration_s=6., fps=25.))]
    assert result == {"frames": 150}


def test_sep_entry_can_reuse_existing_replay_without_loading_raw_data(tmp_path, monkeypatch):
    replay, output = tmp_path / "existing.json", tmp_path / "sep"
    rendered = []

    def unexpected_export(*args, **kwargs):
        pytest.fail("Existing replay should not load or export session data")

    monkeypatch.setattr(sep, "export_session", unexpected_export)
    monkeypatch.setattr(sep, "render", lambda *args, **kwargs: rendered.append((args, kwargs)))
    sep.main(["--input", str(replay), "--output-dir", str(output)])
    assert rendered == [((replay, output), dict(start_s=None, duration_s=None, fps=None))]


@pytest.mark.parametrize("arguments", [[], ["recording"], ["recording", "--input", "existing.json"]])
def test_sep_entry_requires_one_source_and_a_unit_for_raw_export(tmp_path, arguments):
    with pytest.raises(SystemExit) as result:
        sep.main([*arguments, "--output-dir", str(tmp_path)])
    assert result.value.code == 2
