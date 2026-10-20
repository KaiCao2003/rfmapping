import json
import shutil
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from Utils import ebc_video
from Utils.ebc_video import _click_blocks, spike_video_seconds, write_spike_clicks


def test_batch_loads_spikes_once_and_keeps_silent_good_units(tmp_path, monkeypatch):
    kilosort = tmp_path / "kilosort"
    kilosort.mkdir()
    (kilosort / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n7\tgood\n8\tmua\n9\tgood\n")
    cluster_path = kilosort / "spike_clusters.npy"
    time_path = tmp_path / "adc_spike_time.npy"
    np.save(cluster_path, [7, 7, 8, 7])
    np.save(time_path, [110.02, 110.05, 110.06, 110.15])
    loaded = []
    load = np.load

    def track_load(path, **kwargs):
        loaded.append(path)
        return load(path, **kwargs)

    monkeypatch.setattr(np, "load", track_load)
    data = {"session": tmp_path, "exposure_times": np.array([10., 10.04, 10.08, 10.12, 10.16]),
            "adc_time_origin_s": 100., "camera_timing": {}, "source": {"kilosort_dir": kilosort,
            "spike_times_path": time_path, "selected_interval_s": [10.02, 10.14]}}
    units, seconds, _ = ebc_video._load_audio_spikes(
        data, {"fps": 25., "source_first_frame": 1, "frames": 3},
    )
    assert units == [7, 9]
    np.testing.assert_allclose(seconds[7], [.01], atol=1e-12)
    assert seconds[9].size == 0
    assert loaded == [cluster_path, time_path]


def test_spike_audio_uses_exposure_clock_and_fractional_frames():
    # A dropped pose at frame 1 must not remove its exposure or shift the sound.
    exposures = np.array([2.9, 2.94, 2.984, 3.024])
    spikes = np.array([2.88, 2.9, 2.92, 2.962, 3.024, 3.044])
    seconds = spike_video_seconds(spikes, exposures, 25.)
    np.testing.assert_allclose(seconds, [-.02, 0., .02, .06, .12, .14], atol=1e-14)
    np.testing.assert_allclose(seconds[3:] - 1 / 25., [.02, .08, .10], atol=1e-14)


def test_click_tails_and_coincident_spikes_survive_audio_block_boundaries():
    click = np.array([1., -.5, .25], dtype=np.float32)
    samples = np.array([3, 3, 4, 9])
    blocks = np.concatenate(list(_click_blocks(samples, click, 10, 4)))
    np.testing.assert_array_equal(blocks, [0, 0, 0, 2, 0, 0, .25, 0, 0, 1])


def test_click_wav_has_exact_sample_onsets_and_clip_duration(tmp_path):
    path = tmp_path / "spikes.wav"
    result = write_spike_clicks(path, [-1., .025, .075, .1], .1, gain=.35)
    with wave.open(str(path)) as source:
        assert source.getframerate() == 48000
        assert source.getnchannels() == 1
        assert source.getnframes() == 4800
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    assert result["spike_count"] == 2
    assert not audio[:1200].any()
    assert audio[1200] > 0 and audio[3600] > 0
    assert not audio[1392:3600].any()
    assert not audio[3792:].any()


def test_dense_bursts_are_scaled_globally_and_silent_clips_are_supported(tmp_path):
    path = tmp_path / "burst.wav"
    result = write_spike_clicks(path, np.r_[np.full(20, .01), .05], .1, gain=1.)
    assert result["applied_gain"] == .95 / 20
    with wave.open(str(path)) as source:
        audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
    assert np.max(np.abs(audio)) <= round(.95 * 32767)
    assert audio[480] / audio[2400] > 19.9
    result = write_spike_clicks(path, [], .1)
    with wave.open(str(path)) as source:
        assert not any(source.readframes(source.getnframes()))
    assert result["spike_count"] == 0


@pytest.fixture
def continuous_source(tmp_path):
    sample_rate = 30_000
    t = np.arange(sample_rate * 2) / sample_rate
    signal = np.c_[5000 * np.sin(2 * np.pi * 1200 * t) + 4000 * np.sin(2 * np.pi * 20 * t)
                   + 2000 * np.sin(2 * np.pi * 11_000 * t),
                   3000 * np.sin(2 * np.pi * 900 * t)]
    binary = tmp_path / "continuous.dat"
    np.rint(signal).astype("<i2").tofile(binary)
    timestamps = tmp_path / "timestamps.npy"
    np.save(timestamps, 100.003 + t)
    return dict(binary=binary, timestamps=timestamps, sample_rate=sample_rate,
                num_channels=2, bit_volts=np.array([.195, .195]))


def test_real_voltage_retains_pitch_and_rejects_lfp_and_high_frequency_noise(continuous_source):
    clock = (np.array([0., .5]), np.array([100.403, 100.903]))
    signal = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [1, 0], clock, .5,
    )))
    assert signal.shape == (24_000, 2)
    spectrum = abs(np.fft.rfft(signal, axis=0))
    frequencies = np.fft.rfftfreq(len(signal), 1 / 48_000)
    np.testing.assert_array_equal(frequencies[spectrum.argmax(axis=0)], [900., 1200.])
    amplitude = lambda hz: spectrum[np.argmin(abs(frequencies - hz)), 1]
    assert amplitude(20) < .01 * amplitude(1200)
    assert amplitude(11_000) < .15 * amplitude(1200)


def test_voltage_filter_and_resampling_have_no_block_boundary_clicks(continuous_source):
    # An offset/drifting clock puts block edges between both raw and audio samples.
    clock = (np.array([0., .4]), np.array([100.412345, 100.812789]))
    whole = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [0, 1], clock, .4, block_s=1.,
    )))
    divided = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [0, 1], clock, .4, block_s=.013731,
    )))
    np.testing.assert_allclose(divided, whole, rtol=0, atol=1e-4)


def test_continuous_clock_is_inverse_of_spike_clock_with_end_extrapolation():
    exposures = np.array([2.9, 2.94, 2.984, 3.024])
    first_frame, fps, origin = 1, 25., 2300.
    clock = ((np.arange(len(exposures)) - first_frame) / fps, exposures + origin)
    seconds = np.array([0., .001, .04, .07, .08, .10])
    oe_times = ebc_video._clock_times(seconds, clock)
    recovered = spike_video_seconds(oe_times - origin, exposures, fps) - first_frame / fps
    np.testing.assert_allclose(recovered, seconds, atol=1e-12)


def test_audio_pads_missing_recording_tail_with_silence(continuous_source):
    clock = (np.array([0., .1]), np.array([101.96, 102.06]))
    audio = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [0], clock, .1, block_s=.01,
    )))[:, 0]
    assert np.max(abs(audio[:1920])) > 100
    assert not audio[int(np.ceil((102.003 - 101.96) * 48_000)):].any()


def test_unknown_audio_clock_mutes_partial_and_whole_blocks_then_resumes(continuous_source, monkeypatch):
    video_seconds = np.arange(9) / 10.
    measured_times = 100.403 + video_seconds
    full_clock = (video_seconds, measured_times)
    baseline = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [0, 1], full_clock, .8, block_s=.2,
    )))
    unknown_times = measured_times.copy()
    unknown_times[2:5] = np.nan
    clock = (video_seconds, unknown_times)
    filtered_samples = []
    filter_voltage = ebc_video.sosfilt

    def record_filtered_samples(sos, voltage, **kwargs):
        filtered_samples.append(len(voltage))
        return filter_voltage(sos, voltage, **kwargs)

    monkeypatch.setattr(ebc_video, "sosfilt", record_filtered_samples)
    blocks = ebc_video._continuous_audio_blocks(continuous_source, [0, 1], clock, .8, block_s=.2)
    first = next(blocks)
    # Its final timestamps are unknown: read only the finite first 100 ms,
    # plus filter warmup, rather than consuming the remaining two-second file.
    assert filtered_samples[0] < .36 * continuous_source["sample_rate"]
    second = next(blocks)
    assert not second.any()
    assert len(filtered_samples) == 1
    audio = np.concatenate([first, second, *blocks])
    valid = np.isfinite(ebc_video._clock_times(np.arange(len(audio)) / 48_000, clock))
    assert not audio[~valid].any()
    assert np.isfinite(audio).all()
    assert np.max(abs(audio[-4800:])) > 100
    np.testing.assert_allclose(audio[valid], baseline[valid], rtol=0, atol=1e-4)


def test_audio_clock_with_unknown_start_warms_up_at_first_measured_time(continuous_source):
    clock = (np.arange(5) / 10., np.array([np.nan, np.nan, 100.603, 100.703, 100.803]))
    blocks = list(ebc_video._continuous_audio_blocks(
        continuous_source, [1], clock, .4, block_s=.1,
    ))
    assert not blocks[0].any() and not blocks[1].any()
    resumed = np.concatenate(blocks[2:])
    reference = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [1], (np.array([0., .2]), np.array([100.603, 100.803])), .2,
    )))
    assert np.max(abs(resumed)) > 100
    np.testing.assert_allclose(resumed, reference, rtol=0, atol=1e-4)


def test_entirely_unknown_audio_clock_returns_only_silence(continuous_source, monkeypatch):
    def unexpected_filter(*args, **kwargs):
        pytest.fail("Unknown timestamps must not read/filter any raw samples")

    monkeypatch.setattr(ebc_video, "sosfilt", unexpected_filter)
    clock = (np.array([0., .2]), np.array([np.nan, np.nan]))
    audio = np.concatenate(list(ebc_video._continuous_audio_blocks(
        continuous_source, [0, 1], clock, .2, block_s=.1,
    )))
    assert audio.shape == (9600, 2)
    assert not audio.any()


def _channel_fixture(tmp_path, labels, clusters, template_ids):
    (tmp_path / "cluster_KSLabel.tsv").write_text(labels)
    templates = np.zeros((3, 5, 3), dtype=np.float32)
    templates[0, 2] = [5., 2., 0.]  # Unwhitening changes the peak from column 0 to 1.
    templates[1, 2, 2] = 8.
    templates[2, 2, 0] = 9.
    for name, values in dict(templates=templates, channel_map=[8, 4, 1],
                             whitening_mat_inv=np.diag([1., 3., 1.]),
                             spike_clusters=clusters, spike_templates=template_ids).items():
        np.save(tmp_path / f"{name}.npy", values)


def test_main_channel_uses_cluster_membership_unwhitening_and_channel_map(tmp_path):
    _channel_fixture(tmp_path, "cluster_id\tKSLabel\n10\tgood\n11\tgood\n3\tmua\n",
                     [10, 10, 11, 10, 3], [0, 1, 1, 0, 2])
    assert ebc_video._good_unit_channels(tmp_path) == {10: 4, 11: 1}


def test_original_silent_good_unit_keeps_its_template_channel(tmp_path):
    _channel_fixture(tmp_path, "cluster_id\tKSLabel\n0\tgood\n2\tgood\n",
                     [0, 0, 1], [0, 0, 1])
    assert ebc_video._good_unit_channels(tmp_path) == {0: 4, 2: 8}


def test_continuous_cache_has_exact_duration_and_oe_voltage_gain(tmp_path, continuous_source):
    clock = (np.array([0., 1.2]), np.array([100.403, 101.603]))
    tracks = ebc_video._cache_continuous_audio(continuous_source, [0, 1], clock, 1.2, tmp_path)
    assert len(tracks) == 2
    track = tracks[0]
    raw = np.fromfile(track["path"], dtype="<f4")
    pcm = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, .35, gate_sigma=0)), dtype="<i2")
    assert len(pcm) == len(raw) == 57_600
    # OE limits the incoming voltage to +/-1000 uV and then applies a fixed
    # conversion. Loud artifacts elsewhere must not change the volume here.
    np.testing.assert_allclose(pcm, np.clip(raw, -1000, 1000) * .35 / .02, atol=.51)


def test_expander_reduces_background_and_retains_large_spikes_without_time_shift(tmp_path):
    rate = 48_000
    rng = np.random.default_rng(14)
    signal = rng.normal(0., 20., rate * 2).astype("<f4")
    peaks = np.array([0, 11999, 47999, 48024, 95999])
    for center, amplitude in zip(peaks, [200, -220, -280, 240, -160]):
        indices = np.arange(max(0, center - 20), min(len(signal), center + 21))
        signal[indices] = amplitude * np.exp(-.5 * ((indices - center) / 6)**2)
    path = tmp_path / "channel.f32"
    signal.tofile(path)
    track = dict(path=path, sample_rate=rate, noise_sigma=20.)
    before = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, .35, gate_sigma=0)), dtype="<i2")
    after = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, .35)), dtype="<i2")
    whole = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, .35, block_s=3.)), dtype="<i2")
    divided = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, .35, block_s=.013731)), dtype="<i2")
    assert len(before) == len(after) == len(signal)
    np.testing.assert_allclose(after[peaks], before[peaks], rtol=.02)
    for peak in peaks:
        left, right = max(0, peak - 5), min(len(signal), peak + 6)
        assert left + abs(after[left:right]).argmax() == peak
    np.testing.assert_array_equal(after, whole)
    np.testing.assert_array_equal(after, divided)
    background = abs(signal) < 30
    assert np.sqrt(np.mean(after[background].astype(float)**2)) < .2 * np.sqrt(np.mean(before[background].astype(float)**2))
    assert np.sqrt(np.mean(after.astype(float)**2)) < .7 * np.sqrt(np.mean(before.astype(float)**2))


def test_expander_has_continuous_gain_below_threshold_and_unity_gain_above(tmp_path):
    # At a fixed input below threshold, a 4:1 expander multiplies amplitude by
    # (input/threshold)^3. This is not a binary open/closed window.
    signal = np.repeat(np.array([10., 20., 30., 100.], dtype="<f4"), 480)
    path = tmp_path / "channel.f32"
    signal.tofile(path)
    track = dict(path=path, sample_rate=48_000, noise_sigma=10.)
    pcm = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, .35)), dtype="<i2")
    np.testing.assert_allclose(pcm[[479, 959, 1439, 1919]],
                               np.array([10 / 27, 20 * (2 / 3)**3, 30, 100]) * .35 / .02,
                               atol=.51)
    bypass = b"".join(ebc_video._continuous_pcm(track, .35, expander_ratio=1))
    assert bypass == b"".join(ebc_video._continuous_pcm(track, .35, gate_sigma=0))


def test_voltage_limiter_does_not_wrap_int16_at_high_gain(tmp_path):
    signal = np.array([0, 100, -100, 10_000, -10_000], dtype="<f4")
    path = tmp_path / "channel.f32"
    signal.tofile(path)
    track = dict(path=path, sample_rate=48_000, noise_sigma=10.)
    pcm = np.frombuffer(b"".join(ebc_video._continuous_pcm(track, 1., gate_sigma=0)), dtype="<i2")
    np.testing.assert_array_equal(pcm, [0, 5000, -5000, round(.95 * 32767), -round(.95 * 32767)])


def test_gate_estimates_background_robustly_and_keeps_a_silent_track_silent(tmp_path, monkeypatch):
    rng = np.random.default_rng(12)
    quiet = rng.normal(0., 2., (48_000, 2)).astype(np.float32)
    quiet[:, 1] = 0
    artifact = quiet * 50
    monkeypatch.setattr(ebc_video, "_continuous_audio_blocks",
                        lambda *args, **kwargs: iter([quiet, artifact, quiet]))
    tracks = ebc_video._cache_continuous_audio({}, [0, 1], None, 3., tmp_path)
    assert tracks[0]["noise_sigma"] == pytest.approx(2., rel=.08)
    assert tracks[1]["noise_sigma"] == 0.
    assert not any(b"".join(ebc_video._continuous_pcm(tracks[1], .35)))


def test_voltage_track_mux_preserves_video_and_audio_duration(tmp_path, continuous_source):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video mux requires ffmpeg and ffprobe")
    video = tmp_path / "picture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=white:s=64x64:r=25:d=0.2",
                    "-c:v", "libx264", "-threads", "1", str(video)], check=True)
    clock = (np.array([0., .2]), np.array([100.403, 100.603]))
    track = ebc_video._cache_continuous_audio(continuous_source, [0], clock, .2, tmp_path)[0]
    output = tmp_path / "7.mp4"
    ebc_video._mux_continuous_audio(video, output, 7, "A", track, .2, .35)
    result = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                            capture_output=True, text=True, check=True)
    streams = json.loads(result.stdout)["streams"]
    audio = next(item for item in streams if item["codec_type"] == "audio")
    assert audio["sample_rate"] == "48000"
    assert float(audio["duration"]) == pytest.approx(.2, abs=1 / 48000)
    hashes = [subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                              "-c", "copy", "-f", "md5", "-"],
                             capture_output=True, check=True).stdout for path in (video, output)]
    assert hashes[0] == hashes[1]
    assert not list(tmp_path.glob("*.partial.mp4"))


def test_batch_encodes_shared_electrode_once_and_preserves_unit_media(tmp_path, continuous_source, monkeypatch):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Video mux requires ffmpeg and ffprobe")
    video = tmp_path / "picture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=white:s=64x64:r=25:d=0.2",
                    "-c:v", "libx264", "-threads", "1", str(video)], check=True)
    monkeypatch.setattr(ebc_video, "_open_ephys_source", lambda *args: continuous_source)
    monkeypatch.setattr(ebc_video, "_good_unit_channels", lambda *args: {7: 0, 9: 0, 8: 1})
    monkeypatch.setattr(ebc_video, "export_ebc_overlay", lambda *args, **kwargs:
                        dict(video=video, frames=5, fps=25., source_first_frame=0))
    monkeypatch.setattr(ebc_video, "ProcessPoolExecutor",
                        lambda max_workers, **kwargs: ThreadPoolExecutor(max_workers=max_workers))
    processed_channels = []
    continuous_pcm = ebc_video._continuous_pcm

    def count_pcm(track, *args, **kwargs):
        processed_channels.append(track["channel"])
        return continuous_pcm(track, *args, **kwargs)

    monkeypatch.setattr(ebc_video, "_continuous_pcm", count_pcm)
    data = dict(session=tmp_path, probe="A", source={"kilosort_dir": tmp_path},
                exposure_times=np.array([0., .2]), adc_time_origin_s=100.403)
    outputs = ebc_video.export_good_unit_videos(data, video, tmp_path / "units", audio_workers=2)
    assert sorted(processed_channels) == [0, 1]
    assert [path.name for path in outputs] == ["7.mp4", "9.mp4", "8.mp4"]

    def stream_hash(path, stream):
        return subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", f"0:{stream}:0",
                               "-c", "copy", "-f", "md5", "-"],
                              capture_output=True, check=True).stdout

    assert stream_hash(outputs[0], "a") == stream_hash(outputs[1], "a")
    assert stream_hash(outputs[0], "a") != stream_hash(outputs[2], "a")
    clock = ebc_video._video_audio_clock(data, dict(frames=5, fps=25., source_first_frame=0))
    baseline_track = ebc_video._cache_continuous_audio(continuous_source, [0], clock, .2, tmp_path)[0]
    baseline = tmp_path / "direct.mp4"
    ebc_video._mux_continuous_audio(video, baseline, 7, "A", baseline_track, .2, .35)
    assert stream_hash(outputs[0], "a") == stream_hash(baseline, "a")

    def packet_times(path):
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a", "-show_packets",
             "-show_entries", "packet=pts,dts,duration,side_data_list", "-of", "json", str(path)],
            capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)["packets"]

    assert packet_times(outputs[0]) == packet_times(baseline)
    original_video = stream_hash(video, "v")
    for path in outputs:
        assert stream_hash(path, "v") == original_video
        info = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
                              capture_output=True, text=True, check=True)
        audio = next(item for item in json.loads(info.stdout)["streams"] if item["codec_type"] == "audio")
        assert float(audio["duration"]) == pytest.approx(.2, abs=1 / 48000)
        assert f"unit {path.stem};" in audio["tags"]["handler_name"]
    assert not list((tmp_path / "units").glob("*.partial.mp4"))
