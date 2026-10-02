import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from Utils import ebc_geometry as geometry
from Utils import ebc_overlay as drawing
from Utils import ebc_video as video
from Utils.ebc_workflow import prepare_video_data


@pytest.fixture
def recording(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video integration tests require ffmpeg and ffprobe")
    path = tmp_path / "source.avi"
    levels = np.array([30, 60, 90, 120, 150], dtype=np.uint8)
    frames = np.broadcast_to(levels[:, None, None, None], (5, 240, 320, 3)).copy()
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "320x240", "-r", "31", "-i", "pipe:0", "-c:v", "ffv1", "-threads", "1", str(path),
    ], input=frames.tobytes(), check=True)
    return dict(path=path, levels=levels, frames=frames, directory=tmp_path)


def prepared_data(directory, shape):
    times = np.array([10., 10.04, np.nan, 10.12, 10.16])
    if shape == "rectangle":
        boundary = geometry.rectangle_boundary((60., 260., 20., 220.), 20.)
        xyz = np.tile([10., 10., 0.], (5, 1))
    else:
        boundary = geometry.cylinder_boundary(
            dict(screen_bottom_center=[100., 200., 0.], screen_diamter=200., motive_units_per_real_mm=1.),
            dict(world_to_video_projection_raw=[[1., 0., 0., 60.], [0., -1., 0., 320.], [0., 0., 0., 1.]]),
        )
        xyz = np.tile([10., 20., 5.], (5, 1))
    valid = np.isfinite(times)
    xyz[~valid] = np.nan
    position = geometry.PositionInfo(np.arange(5), times, xyz, np.full(5, 90.), valid,
                                     dict(pose_path="synthetic recording"))
    config = dict(session=directory, probe="A", phase="test", kilosort_dir=directory / "sorting")
    data = prepare_video_data(config, position, boundary, times, 100., dict(source="measured test clock"))
    return data


def decoded_video(path):
    streams = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path),
    ]))["streams"]
    stream = streams[0]
    pixels = subprocess.check_output([
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    frames = np.frombuffer(pixels, np.uint8).reshape(-1, int(stream["height"]), int(stream["width"]), 3)
    return stream, frames


@pytest.mark.parametrize("shape", ["rectangle", "cylinder"])
def test_draw_uses_prepared_rays_and_preserves_raw_sync_gap(tmp_path, monkeypatch, shape):
    data = prepared_data(tmp_path, shape)

    def unexpected(*args, **kwargs):
        pytest.fail("Drawing must not calculate geometry or project points")

    monkeypatch.setattr(geometry, "compute_ebc_rays", unexpected)
    monkeypatch.setattr(drawing, "project_points", unexpected)
    overlay = drawing.EBCOverlay(data, 320, 240, 25.)
    frame = Image.new("RGB", (320, 240), (90, 90, 90))
    assert len(overlay.info.bearings_deg) == 8
    valid = np.asarray(overlay.draw(frame, 0, 0))
    assert np.any(valid[:240, :320] != 90)
    gap = np.asarray(overlay.draw(frame, 2, 2))
    np.testing.assert_array_equal(gap[:240, :320], np.asarray(frame))
    absent = np.asarray(overlay.draw(frame, 2, -1))
    np.testing.assert_array_equal(absent[:240, :320], np.asarray(frame))
    np.testing.assert_array_equal(gap[700:, :320], 255)


@pytest.mark.parametrize("shape", ["rectangle", "cylinder"])
def test_both_sources_keep_frames_and_measured_rate_with_one_or_two_workers(recording, shape):
    data = prepared_data(recording["directory"], shape)
    outputs = []
    for workers in (1, 2):
        output = recording["directory"] / f"{shape}_{workers}.mp4"
        result = video.export_ebc_overlay(data, recording["path"], output, workers=workers,
                                          video_encoder="libx264", frame_rate=25., first_frame=0, stop_frame=5)
        stream, frames = decoded_video(output)
        assert result["frames"] == len(frames) == 5
        assert result["fps"] == float(Fraction(stream["avg_frame_rate"])) == 25.
        assert result["duration_s"] == pytest.approx(.2)
        assert (result["source_first_frame"], result["source_last_frame"]) == (0, 4)
        assert frames.shape[1:3] == (760, 690)
        np.testing.assert_allclose(frames[:, 10, 10, 0], recording["levels"], atol=3)
        np.testing.assert_allclose(frames[2, :240, :320].astype(float), 90., atol=3)
        outputs.append(frames)
    np.testing.assert_array_equal(outputs[0], outputs[1])


def continuous_source(directory):
    sample_rate = 30_000
    relative = np.arange(18_000) / sample_rate
    voltage = np.column_stack((1000. * np.sin(2 * np.pi * 1000. * relative),
                                1000. * np.sin(2 * np.pi * 2000. * relative))).astype("<i2")
    binary, timestamps = directory / "continuous.dat", directory / "timestamps.npy"
    voltage.tofile(binary)
    np.save(timestamps, 109.8 + relative)
    return dict(binary=binary, timestamps=timestamps, sample_rate=sample_rate,
                num_channels=2, bit_volts=np.array([.195, .195]))


def test_continuous_electrode_audio_is_silent_through_unknown_clock_intervals(tmp_path):
    data = prepared_data(tmp_path, "cylinder")
    clock = video._video_audio_clock(data, dict(fps=25., source_first_frame=0, frames=5))
    samples = np.concatenate(list(video._continuous_audio_blocks(
        continuous_source(tmp_path), [0, 1], clock, .2, sample_rate=48_000, block_s=.02,
    )))
    assert samples.shape == (9600, 2)
    np.testing.assert_array_equal(samples[2400:5520], 0.)
    assert np.std(samples[240:1680]) > 1.
    assert np.std(samples[6240:9120]) > 1.


def test_spike_clock_uses_only_adjacent_known_exposures_and_retains_clip_offset():
    clock = (np.arange(6) / 25. - .08, np.array([100., np.nan, 102., 103., 104., 105.]))
    spikes = np.array([99., 100., 100.5, 101.5, 102., 102.25, 103., 104.5, 105., 106.])
    result = video._spike_times_on_video_clock(spikes, clock)
    np.testing.assert_allclose(result, [np.nan, np.nan, np.nan, np.nan, 0., .01, .04, .1, np.nan, np.nan],
                               equal_nan=True, atol=1e-12)


def test_spike_clock_does_not_bridge_an_internal_gap_or_a_clock_without_pairs():
    spikes = np.array([100.5, 101., 102., 103., 103.5, 104.5])
    clock = (np.arange(6) / 25., np.array([100., 101., np.nan, 103., 104., 105.]))
    np.testing.assert_allclose(video._spike_times_on_video_clock(spikes, clock),
                               [.02, np.nan, np.nan, .12, .14, .18], equal_nan=True, atol=1e-12)
    missing_pairs = (np.arange(3) / 25., np.array([100., np.nan, 102.]))
    assert np.isnan(video._spike_times_on_video_clock(spikes, missing_pairs)).all()


def test_spike_loader_keeps_terminal_frame_spikes_and_silent_good_units(tmp_path):
    data = prepared_data(tmp_path, "cylinder")
    sorting = Path(data["source"]["kilosort_dir"])
    sorting.mkdir()
    (sorting / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n2\tgood\n5\tgood\n7\tnoise\n")
    np.save(sorting / "spike_clusters.npy", [2, 2, 2, 2, 2, 2, 7])
    spike_path = Path(data["source"]["spike_times_path"])
    spike_path.parent.mkdir(parents=True)
    # 110.06 falls in the unknown clock interval; 110.18 belongs to the
    # final image, after its measured exposure at 110.16 but before 110.20.
    np.save(spike_path, [110.02, 110.06, 110.13, 110.18, 110.20, 110.21, 110.19])
    result = dict(fps=25., source_first_frame=0, frames=5)
    units, spikes, timing = video._load_audio_spikes(data, result)
    assert units == [2, 5]
    np.testing.assert_allclose(spikes[2], [.02, .13, .18], atol=1e-12)
    assert spikes[5].size == 0
    assert timing == data["camera_timing"]

    data["source"]["selected_interval_s"] = [10.12, 10.16]
    units, spikes, _ = video._load_audio_spikes(data, result)
    assert units == [2, 5]
    np.testing.assert_allclose(spikes[2], [.13], atol=1e-12)
    assert spikes[5].size == 0


@pytest.mark.parametrize("header_extra, audited_count, accepted", [(1, None, True),
                                                                  (2, None, False),
                                                                  (1, 6, False)])
def test_decode_eof_only_allows_one_unaudited_header_overcount(recording, monkeypatch,
                                                            header_extra, audited_count, accepted):
    data = prepared_data(recording["directory"], "rectangle")
    if audited_count is not None:
        data["video_frame_count"] = audited_count
    original_run = subprocess.run

    def run(command, *args, **kwargs):
        result = original_run(command, *args, **kwargs)
        if command[0] == "ffprobe" and str(recording["path"]) in command:
            payload = json.loads(result.stdout)
            payload["streams"][0]["nb_frames"] = str(5 + header_extra)
            result.stdout = json.dumps(payload)
        return result

    monkeypatch.setattr(video.subprocess, "run", run)
    output = recording["directory"] / "eof.mp4"
    if accepted:
        result = video.export_ebc_overlay(data, recording["path"], output, workers=1,
                                          video_encoder="libx264", frame_rate=25.)
        _, frames = decoded_video(output)
        assert result["frames"] == len(frames) == 5
        assert result["container_frame_count"] == 6
    else:
        with pytest.raises(RuntimeError, match="decode stopped at frame 5"):
            video.export_ebc_overlay(data, recording["path"], output, workers=1,
                                     video_encoder="libx264", frame_rate=25.)
        assert not output.exists()
        assert not output.with_name("eof.partial.mp4").exists()


def test_audited_frame_count_disagreement_is_rejected_before_export(recording):
    data = prepared_data(recording["directory"], "cylinder")
    data["video_frame_count"] = 7
    output = recording["directory"] / "wrong_count.mp4"
    with pytest.raises(ValueError, match="audited full-video clock"):
        video.export_ebc_overlay(data, recording["path"], output, workers=1,
                                 video_encoder="libx264", frame_rate=25.)
    assert not output.exists()


def test_good_unit_batch_caches_shared_electrodes_and_keeps_outputs(recording, monkeypatch):
    data = prepared_data(recording["directory"], "cylinder")
    source = continuous_source(recording["directory"])
    monkeypatch.setattr(video, "_open_ephys_source", lambda *args: source)
    monkeypatch.setattr(video, "_good_unit_channels", lambda *args: {2: 0, 5: 0, 9: 1})
    cached = []
    original_cache = video._cache_continuous_audio

    def cache(raw_source, channels, *args, **kwargs):
        cached.append(list(channels))
        return original_cache(raw_source, channels, *args, **kwargs)

    monkeypatch.setattr(video, "_cache_continuous_audio", cache)
    directory = recording["directory"] / "good_units"
    manifest = video.export_good_unit_videos(
        data, recording["path"], directory, workers=1, audio_workers=2,
        video_encoder="libx264", frame_rate=25., first_frame=0, stop_frame=5,
        audio_gate_sigma=0., audio_expander_ratio=1.,
    )
    assert cached == [[0, 1]]
    assert manifest["units"] == [2, 5, 9]
    saved = json.loads((directory / "manifest.json").read_text())
    assert saved == manifest
    for key in ("video", "preview"):
        assert Path(saved[key]).is_file()
    assert "ebc_batch_" not in json.dumps(saved)
    assert saved["provenance"]["timing"]["source"] == "measured test clock"
    sounds = []
    for unit in (2, 5, 9):
        path = directory / f"{unit}.mp4"
        stream, frames = decoded_video(path)
        assert len(frames) == 5
        sound = subprocess.check_output([
            "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
            "-f", "s16le", "-acodec", "pcm_s16le", "pipe:1",
        ])
        sounds.append(np.frombuffer(sound, dtype="<i2"))
        assert np.max(np.abs(sounds[-1])) > 0
    np.testing.assert_array_equal(sounds[0], sounds[1])
    assert not np.array_equal(sounds[0], sounds[2])
