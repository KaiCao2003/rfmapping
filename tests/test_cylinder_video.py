import csv
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import ImageDraw
from scipy.io import savemat

import ebc_video_circle as batch
from Utils import ebc_camera as camera


@pytest.fixture
def recording(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video integration tests require ffmpeg and ffprobe")
    session = tmp_path / "260922_3"
    processed = session / "data/processed"
    processed.mkdir(parents=True)
    alignment = tmp_path / "alignment"
    alignment.mkdir()
    levels = np.array([30, 60, 90, 120, 150], dtype=np.uint8)
    payload = b"".join(bytes([int(value)]) * (1280 * 1024 * 3) for value in levels)
    # The AVI rate deliberately disagrees with the measured 25 Hz exposure clock.
    video = session / "260922.avi"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x1024", "-r", "1105338/10000", "-i", "pipe:0",
        "-c:v", "ffv1", "-threads", "1", str(video),
    ], input=payload, check=True)
    columns = pd.MultiIndex.from_tuples(
        [("meta", "id", "meta", "Frame")]
        + [("hp4", "id", kind, axis) for kind in ("Position", "Rotation") for axis in "XYZ"]
    )
    pose = np.c_[np.arange(5), np.zeros((5, 2)), np.full(5, 100.), np.zeros((5, 3))]
    pose[2, 1:] = np.nan
    motive = processed / "trimmed_input.csv"
    pd.DataFrame(pose, columns=columns).to_csv(motive, index=False)
    calibration = dict(screen_bottom_center=[0., 0., 0.], screen_diamter=200.,
                       motive_units_per_real_mm=1.)
    (session / "260922.calib").write_text(json.dumps(calibration))
    projection = [[2., 0., 0., 640.], [0., -2., 0., 512.], [0., 0., 0., 1.]]
    registration = alignment / "camera_registration.json"
    registration.write_text(json.dumps(dict(world_to_video_projection_raw=projection)))
    reference = alignment / "reference.json"
    reference.write_text(json.dumps(dict(reference_quaternion_xyzw=[0., 0., 0., 1.])))
    video_times = alignment / "video_adc_times.npy"
    np.save(video_times, 10. + np.arange(5) / 25.)
    np.save(alignment / "video_motive_rows.npy", np.arange(5))
    replay = alignment / "trials.json"
    replay.write_text(json.dumps(dict(meta=dict(probe="B"), trials=[
        dict(on_s=9., off_s=11., screen_deg=0., luminance=1.),
    ])))
    return dict(session=session, alignment=alignment, motive=motive, levels=levels,
                registration=registration, reference=reference, video_times=video_times,
                replay=replay, output=tmp_path / "output")


def render_full(recording):
    return camera.render(
        recording["session"], recording["replay"], recording["registration"],
        recording["video_times"], recording["output"], full_video=True,
        motive_csv_path=recording["motive"], hd_reference_path=recording["reference"],
        hd_reference_world_deg=0.,
    )


@pytest.mark.parametrize("times", [
    [10., 10.04, np.nan, 10.12, 10.16],
    [10., 10.04, 10.04, 10.12, 10.16],
])
def test_full_renderer_rejects_partial_or_nonincreasing_clock(recording, times):
    np.save(recording["video_times"], times)
    with pytest.raises(ValueError, match="every decoded AVI frame"):
        render_full(recording)
    assert not recording["output"].exists()


def test_full_renderer_rejects_a_finite_clock_shorter_than_the_avi(recording):
    np.save(recording["video_times"], 10. + np.arange(4) / 25.)
    recording["output"].mkdir()
    audit_path = recording["output"] / "frame_geometry.csv"
    audit_path.write_text("previous successful audit\n")
    with pytest.raises(ValueError, match="clock"):
        render_full(recording)
    assert audit_path.read_text() == "previous successful audit\n"
    assert not (recording["output"] / ".frame_geometry.partial.csv").exists()


def test_full_renderer_keeps_every_frame_and_missing_pose_at_measured_rate(recording, monkeypatch):
    labels = []
    draw_text = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        labels.append(text)
        return draw_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    result = render_full(recording)
    assert result["frames"] == 5
    assert result["fps"] == pytest.approx(25.)
    assert result["duration_s"] == pytest.approx(.2)
    assert result["first_decoded_video_frame"] == 0
    assert result["last_decoded_video_frame"] == 4
    assert result["missing_pose_frames"] == 1
    assert result["unit_id"] is None
    assert Path(result["video"]).name == "overlay_silent.mp4"
    assert sum(text.startswith("Probe B  |") for text in labels) == 5
    assert not any("Unit 318" in text for text in labels)
    assert labels.count("Tracking unavailable") == 1
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-of", "json", result["video"],
    ]))["streams"][0]
    assert float(Fraction(stream["avg_frame_rate"])) == pytest.approx(25.)
    decoded = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", result["video"], "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ])
    images = np.frombuffer(decoded, np.uint8).reshape(-1, 1088, 1280, 3)
    assert len(images) == 5
    np.testing.assert_allclose(images[:, 82, 50, 0], recording["levels"], atol=3)
    # Missing tracking leaves the original image in place, not the previous arrow.
    np.testing.assert_allclose(images[2, 544, 640], [90, 90, 90], atol=3)
    assert images[0, 544, 640, 1] > images[0, 544, 640, 0] + 100
    audit = pd.read_csv(recording["output"] / "frame_geometry.csv")
    np.testing.assert_array_equal(audit.video_frame, np.arange(5))
    np.testing.assert_array_equal(audit.pose_valid, [True, True, False, True, True])
    assert np.isnan(audit.bearing_deg.iloc[2])


def test_batch_exports_all_good_units_and_caches_each_electrode_once(recording, monkeypatch):
    session, alignment = recording["session"], recording["alignment"]
    origin = 100.
    (alignment / "video_clock_qc.json").write_text(json.dumps(
        dict(session=str(session), adc_time_origin_s=origin)))
    savemat(session / "260922.mat", dict(trials=[
        dict(Square_PositionX=0., Square_Luminance=1.),
        dict(Square_PositionX=90., Square_Luminance=.5),
    ]))
    np.save(session / "data/on_list_times.npy", origin + np.array([9., 10.08, 11.]))
    vs_alignment = alignment / "vs.json"
    vs_alignment.write_text(json.dumps(dict(beta_deg=44.7956488331414)))
    settings = dict(session=session, probe="B", clock_dir=alignment,
                    registration_path=recording["registration"], vs_alignment_path=vs_alignment,
                    hd_reference_path=recording["reference"], motive_csv_path=recording["motive"],
                    kilosort_dir=session / "kilosort/ProbeB/kilosort_3", save_path=recording["output"])
    for key, value in settings.items():
        monkeypatch.setattr(batch, key, value)
    units = {10: 266, 20: 266, 30: 280}
    monkeypatch.setattr(batch, "_good_unit_channels", lambda path: units)
    monkeypatch.setattr(batch, "_open_ephys_source", lambda session, probe:
                        dict(binary="raw.dat", timestamps="timestamps.npy"))
    rendered, cached, exported, scratch_videos = [], [], [], []

    def render(*args, **kwargs):
        rendered.append(kwargs)
        replay = json.loads(Path(args[1]).read_text())
        assert replay["meta"] == dict(probe="B")
        np.testing.assert_allclose([trial["on_s"] for trial in replay["trials"]], [9., 10.08])
        output = Path(args[4])
        assert output != recording["output"]
        video = output / "overlay_silent.mp4"
        video.write_bytes(b"shared video")
        scratch_videos.append(video)
        snapshot = output / "overlay_0000.png"
        snapshot.write_bytes(b"preview")
        (output / "frame_geometry.csv").write_text("video_frame\n0\n")
        result = dict(video=str(video), snapshots=[str(snapshot)], frames=5,
                      fps=25., duration_s=.2)
        (output / "render_manifest.json").write_text(json.dumps(result))
        return result

    def cache(source, channels, clock, duration, directory, **kwargs):
        cached.append((channels, clock, duration))
        return {channel: dict(channel=channel, noise_sigma=1.) for channel in channels}

    def export(video, directory, channel_units, probe, track, duration, *audio_settings):
        assert Path(video) == scratch_videos[0]
        assert Path(video).read_bytes() == b"shared video"
        exported.append((sorted(channel_units), probe, track["channel"], duration))

    monkeypatch.setattr(batch, "render", render)
    monkeypatch.setattr(batch, "_cache_continuous_audio", cache)
    monkeypatch.setattr(batch, "_export_channel_videos", export)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", lambda max_workers, mp_context:
                        ThreadPoolExecutor(max_workers=max_workers))
    manifest = batch.main([])
    assert len(rendered) == 1 and rendered[0]["full_video"] is True
    assert rendered[0]["hd_reference_world_deg"] == rendered[0]["vs_zero_deg"]
    assert len(cached) == 1 and cached[0][0] == [266, 280]
    np.testing.assert_allclose(cached[0][1][0], np.arange(6) / 25.)
    np.testing.assert_allclose(cached[0][1][1], origin + 10. + np.arange(6) / 25.)
    assert sorted(exported) == [([10, 20], "B", 266, .2), ([30], "B", 280, .2)]
    assert manifest["units"] == [10, 20, 30]
    assert [Path(path).name for path in manifest["videos"]] == ["10.mp4", "20.mp4", "30.mp4"]
    saved = recording["output"]
    assert (saved / "overlay_silent.mp4").read_bytes() == b"shared video"
    assert (saved / "overlay_0000.png").read_bytes() == b"preview"
    assert (saved / "frame_geometry.csv").read_text() == "video_frame\n0\n"
    render_manifest = json.loads((saved / "render_manifest.json").read_text())
    assert render_manifest["video"] == manifest["video"] == str(saved / "overlay_silent.mp4")
    assert render_manifest["snapshots"] == manifest["snapshots"] == [str(saved / "overlay_0000.png")]
    saved_manifest = json.loads((saved / "manifest.json").read_text())
    assert saved_manifest["video"] == manifest["video"]
    assert saved_manifest["snapshots"] == manifest["snapshots"]
    assert str(scratch_videos[0].parent) not in (saved / "manifest.json").read_text()
    assert str(scratch_videos[0].parent) not in (saved / "render_manifest.json").read_text()
    assert not scratch_videos[0].exists()


def test_full_renderer_preserves_declared_sync_gap_without_geometry(recording, monkeypatch):
    times = np.load(recording["video_times"])
    times[1] = np.nan
    np.save(recording["video_times"], times)
    rows = np.arange(5, dtype=float)
    rows[1] = np.nan
    np.save(recording["alignment"] / "video_motive_rows.npy", rows)
    labels = []
    draw_text = ImageDraw.ImageDraw.text

    def capture_text(self, xy, text, *args, **kwargs):
        labels.append(text)
        return draw_text(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", capture_text)
    result = camera.render(
        recording["session"], recording["replay"], recording["registration"],
        recording["video_times"], recording["output"], full_video=True,
        motive_csv_path=recording["motive"], hd_reference_path=recording["reference"],
        hd_reference_world_deg=0., sync_valid=np.isfinite(times),
    )
    assert result["frames"] == 5
    assert result["fps"] == pytest.approx(25.)
    assert result["unsynchronized_frames"] == 1
    assert labels.count("Sync unavailable") == 1
    assert labels.count("Tracking unavailable") == 1
    assert "Probe B  |  260922_3  |  Frame 1" in labels
    assert not any("nan" in text.lower() for text in labels)
    decoded = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", result["video"], "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ])
    images = np.frombuffer(decoded, np.uint8).reshape(-1, 1088, 1280, 3)
    assert len(images) == 5
    np.testing.assert_allclose(images[:, 82, 50, 0], recording["levels"], atol=3)
    # A missing clock keeps its source image and omits the otherwise valid pose.
    np.testing.assert_allclose(images[1, 32:1056], 60, atol=3)
    audit = pd.read_csv(recording["output"] / "frame_geometry.csv")
    np.testing.assert_array_equal(audit.video_frame, np.arange(5))
    np.testing.assert_array_equal(audit.sync_valid, [True, False, True, True, True])
    gap = audit.iloc[1]
    assert not gap.pose_valid
    assert gap.motive_row == -1 and gap.trial_index == -1
    assert gap[["adc_time_s", "raw_hd_deg", "video_hd_deg", "fused_yaw_deg",
                "video_hd_vs_deg", "head_x", "head_y", "head_z", "vs_visible",
                "vs_world_deg", "target_x", "target_y", "bearing_deg"]].isna().all()
    with (recording["output"] / "frame_geometry.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == [
        "output_frame", "video_frame", "motive_row", "adc_time_s", "raw_hd_deg",
        "video_hd_deg", "fused_yaw_deg", "video_hd_vs_deg", "head_x", "head_y", "head_z",
        "trial_index", "vs_visible", "pose_valid", "sync_valid", "vs_world_deg",
        "target_x", "target_y", "bearing_deg",
    ]
    assert rows[1]["adc_time_s"] == rows[1]["vs_visible"] == rows[1]["bearing_deg"] == ""
    assert not (recording["output"] / ".frame_geometry.partial.csv").exists()
