from pathlib import Path
import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from Utils import ebc_video
from Utils.ebc_video import _Overlay, _select_encoder, export_ebc_overlay, overlay_geometry


@pytest.fixture(autouse=True)
def fixed_test_arena(monkeypatch):
    # Geometry expectations must not depend on the editable session settings.
    monkeypatch.setattr(ebc_video, "BASLER_BOUNDS_PX", (225., 720., 280., 785.))


def test_auto_encoder_checks_runtime_and_reports_missing_nvenc(monkeypatch, capsys):
    reason = "[h264_nvenc] Cannot load libnvidia-encode.so.1"
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 1, stdout="", stderr=reason))
    assert _select_encoder("auto", 1650, 1024, "25/1") == "libx264"
    output = capsys.readouterr().out
    assert reason in output
    assert "8 CPU threads" in output
    with pytest.raises(RuntimeError, match="Cannot load libnvidia-encode"):
        _select_encoder("h264_nvenc", 1650, 1024, "25/1")


def test_auto_encoder_uses_nvenc_after_successful_initialization(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, stdout="", stderr=""))
    assert _select_encoder("auto", 1650, 1024, "25/1") == "h264_nvenc"


def test_video_rays_use_clockwise_hd_and_ccw_bearings_in_centimeters():
    data = dict(xy=np.array([[20.5, 20.5], [20.5, 10.]]), hd=np.array([0., 90.]))
    position, endpoints, distances, valid = overlay_geometry(data)
    np.testing.assert_allclose(position[0], [472.5, 532.5])
    np.testing.assert_allclose(endpoints[0, ::2], [[472.5, 280], [225, 532.5], [472.5, 785], [720, 532.5]])
    np.testing.assert_allclose(distances[0, ::2], 20.5)
    np.testing.assert_allclose(distances[0, 1::2], 20.5 * np.sqrt(2))
    # A 90-degree CW HD faces right; the positive ego bearing remains to its left.
    np.testing.assert_allclose(endpoints[1, 0], [720, 785 - 10 * 505 / 41])
    np.testing.assert_allclose(distances[1, ::2], [20.5, 31, 20.5, 10])
    assert valid.all()


def test_video_does_not_draw_boundary_intersections_for_outside_positions():
    data = dict(xy=np.array([[-.1, 15.], [41.1, 20.], [0., 20.]]), hd=np.array([0., 180., 270.]))
    _, endpoints, distances, valid = overlay_geometry(data)
    np.testing.assert_array_equal(valid, [False, False, True])
    assert np.isnan(endpoints[:2]).all()
    assert np.isnan(distances[:2]).all()
    assert distances[2, 0] == 0
    np.testing.assert_allclose(endpoints[2, 0], [225, 785 - 20 * 505 / 41])


def test_missing_video_pose_does_not_reuse_the_previous_marker():
    data = dict(xy=np.array([[20.5, 20.5]]), hd=np.array([0.]), times=np.array([2.9]),
                session=Path("test_session"), phase="baseline")
    overlay = _Overlay(data, 1280, 1024, 25.)
    frame = Image.new("RGB", (1280, 1024), "#777777")
    tracked = overlay.draw(frame, 4, 0)
    missing = overlay.draw(frame, 5, -1)
    assert tracked.getpixel((472, 532)) != frame.getpixel((472, 532))
    assert missing.getpixel((472, 532)) == frame.getpixel((472, 532))
    assert missing.getpixel((1649, 1023)) == (255, 255, 255)


def test_rectangle_loader_uses_camera_frames_and_filters_the_selected_phase(tmp_path, monkeypatch):
    session = tmp_path / "260921_11"
    directory = session / "data"
    directory.mkdir(parents=True)
    (session / "kilosort/ProbeA/kilosort_3").mkdir(parents=True)
    (directory / "session_info.json").write_text(json.dumps({"session_info": {}}))
    (directory / "interval_table.csv").write_text("interval_type,start,end\nbaseline,3.005,3.12\n")
    left, right, top, bottom = ebc_video.BASLER_BOUNDS_PX
    pd.DataFrame(dict(frame=[0, 1, 3], center_x=[left, (left + right) / 2, right],
                      center_y=[bottom, (top + bottom) / 2, top], hd_deg=[0., 370., -90.])).to_csv(
        session / "260921.csv", index=False)
    exposures = np.array([3., 3.01, 3.08, 3.11])

    def exposure_clock(*args, **kwargs):
        assert kwargs["camera_ttl_active_high"] is False
        return exposures, 1000., {"source": "test exposures"}

    monkeypatch.setattr(ebc_video, "get_exposure_timestamps", exposure_clock)
    data = ebc_video.load_video_data(session)
    np.testing.assert_array_equal(data["frame_ids"], [1, 3])
    np.testing.assert_allclose(data["xy"], [[20.5, 20.5], [41., 41.]])
    np.testing.assert_allclose(data["hd"], [10., 270.])
    np.testing.assert_allclose(data["times"], [3.01, 3.11])
    np.testing.assert_array_equal(data["exposure_times"], exposures)
    assert data["adc_time_origin_s"] == 1000.


def test_rectangle_audio_clock_uses_exposures_with_clip_offset():
    data = dict(exposure_times=np.array([3., 3.01, 4.5, 4.51]), adc_time_origin_s=1000.)
    result = dict(fps=100., source_first_frame=1, frames=2)
    clock = ebc_video._video_audio_clock(data, result)
    np.testing.assert_allclose(ebc_video._clock_times(np.array([0., .005, .01]), clock),
                               [1003.01, 1003.755, 1004.5])
    np.testing.assert_allclose(ebc_video._clock_times(np.array([1003.755]), (clock[1], clock[0])), [.005])


@pytest.mark.parametrize("workers", [1, 2])
def test_video_clip_preserves_source_frame_order_and_handles_avi_overcount(tmp_path, monkeypatch, workers):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video export requires system ffmpeg and ffprobe")
    source = tmp_path / "source.avi"
    # Distinct gray levels let us inspect exact source order after H.264 encoding.
    levels = [30, 60, 90, 120, 150]
    payload = b"".join(bytes([value]) * (1280 * 1024 * 3) for value in levels)
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", "1280x1024", "-r", "25", "-i", "pipe:0", "-c:v", "libx264",
                    "-threads", "2", "-pix_fmt", "yuv420p", str(source)], input=payload, check=True)
    data = dict(xy=np.full((3, 2), 20.5), hd=np.zeros(3), times=np.array([.04, .12, .16]),
                frame_ids=np.array([1, 3, 4]), session=Path("test_session"), phase="baseline")
    output = tmp_path / "clip.mp4"
    result = export_ebc_overlay(data, source, output, start_s=.04, duration_s=.12, workers=workers)
    assert result["frames"] == 3
    assert result["source_first_frame"] == 1
    assert result["source_last_frame"] == 3
    assert result["missing_pose_frames"] == 1
    decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-f", "rawvideo",
                              "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    images = np.frombuffer(decoded, np.uint8).reshape(3, 1024, 1650, 3)
    np.testing.assert_allclose(images[:, 50, 50, 0], [60, 90, 120], atol=3)
    np.testing.assert_allclose(images[1, 532, 472], [90, 90, 90], atol=3)

    run = subprocess.run

    def overcount_header(args, **kwargs):
        result = run(args, **kwargs)
        if args[0] == "ffprobe":
            info = json.loads(result.stdout)
            info["streams"][0]["nb_frames"] = "6"  # Five frames actually decode.
            result.stdout = json.dumps(info)
        return result

    monkeypatch.setattr(subprocess, "run", overcount_header)
    result = export_ebc_overlay(data, source, tmp_path / "full.mp4", workers=workers)
    assert result["frames"] == 5
    assert result["source_last_frame"] == 4
    assert result["missing_pose_frames"] == 2
    assert not list(tmp_path.glob("*.partial.mp4"))
