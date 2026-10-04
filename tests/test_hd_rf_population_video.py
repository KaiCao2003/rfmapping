import json
import subprocess
from concurrent.futures import Future
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

import hd_rf_population_video as population
from hd_rf_population_video import PopulationOverlay, screen_direction, wrapped_segments


def regression_model(session, scheme="scheme1"):
    # RF ego=HD+40 is representable exactly and its world-frame sum varies.
    angle = np.deg2rad(40.)
    return dict(method="first_harmonic_circular_regression", scheme=scheme, n=18,
                cos_coefficients=[0., np.cos(angle), -np.sin(angle)],
                sin_coefficients=[0., np.sin(angle), np.cos(angle)],
                rho_circular=1., fit_mae_deg=0.,
                selected_on_sum=scheme == "scheme3",
                provenance={"hd_source": str(session / "data/tuning_curves/ProbeA/tuning_curves.tc")})


def test_population_rays_use_clockwise_world_angles(monkeypatch):
    labels = []
    original_text = ImageDraw.ImageDraw.text

    def record_text(draw, xy, text, *args, **kwargs):
        labels.append(str(text).lower())
        return original_text(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    session = Path("260827_11")
    data = dict(session=session, probe="A", hd=np.array([90.]),
                xy=np.array([[20.5, 20.5]]), bounds_px=(370, 920, 210, 760),
                arena_size_cm=41., model=regression_model(session), times=np.array([.04]),
                exposure_times=np.array([0., .04]), trace_window_s=12.,
                hd_mode="real", hd_source={"method": "tracked_hd"}, frame_period_s=1 / 25)
    overlay = PopulationOverlay(data, 1280, 1024, 25.)
    np.testing.assert_allclose(overlay.positions[0], [645., 485.])
    np.testing.assert_allclose(overlay.rf_ego, [130.], atol=1e-8)
    np.testing.assert_allclose(overlay.rf_allo, [220.], atol=1e-8)
    np.testing.assert_allclose(screen_direction([0., 90., 180., 270.]),
                               [[0., -1.], [1., 0.], [0., 1.], [-1., 0.]], atol=1e-12)
    arrows = []
    original_arrow = overlay.arrow

    def record_arrow(draw, start, angle, length, color):
        arrows.append(angle)
        original_arrow(draw, start, angle, length, color)

    monkeypatch.setattr(overlay, "arrow", record_arrow)
    canvas = overlay.draw(Image.new("RGB", (1280, 1024), "white"), 1, 0)
    # World-view arrows use HD and the sum; the ego dial is head-relative.
    np.testing.assert_allclose(arrows, [90., 220., 0., 130.])
    assert canvas.size == (1650, 1324)
    # Frames without an exposure retain the source picture and show no geometry.
    gap = overlay.draw(Image.new("RGB", (1280, 1024), "white"), 2, -1)
    assert gap.getpixel((645, 485)) == (255, 255, 255)
    assert not any("held-out" in label or "kernel" in label for label in labels)


def test_timeline_shows_signed_ego_prediction_on_right_axis(monkeypatch):
    session = Path("260827_11")
    data = dict(session=session, probe="A", hd=np.array([0., 90., 180.]),
                xy=np.ones((3, 2)), bounds_px=(370, 920, 210, 760),
                arena_size_cm=41., model=regression_model(session),
                times=np.array([0., .04, .08]), trace_window_s=12.,
                hd_mode="real", hd_source={"method": "tracked_hd"}, frame_period_s=1 / 25)
    overlay = PopulationOverlay(data, 1280, 1024, 25.)
    traces = []

    def record_segments(times, angles, gap):
        traces.append(np.asarray(angles))
        return []

    monkeypatch.setattr(population, "wrapped_segments", record_segments)
    overlay.timeline(ImageDraw.Draw(overlay.template.copy()), .08)
    np.testing.assert_allclose(traces[0], [0., 90., 180.])
    np.testing.assert_allclose(traces[1], [220., 310., 40.])


def test_trace_breaks_wrap_seams_tracking_gaps_and_missing_values():
    times = [0., .04, .08, .12, .5, .54, .58, .62, .66]
    angles = [350., 355., 1., 5., 10., 15., np.nan, 20., 25.]
    parts = wrapped_segments(times, angles, .06)
    assert len(parts) == 4
    assert [list(t) for t, _ in parts] == [[0., .04], [.08, .12], [.5, .54], [.62, .66]]


@pytest.mark.parametrize("enabled", [(True, True, True), (True, False, False),
                                      (False, True, False), (False, False, True),
                                      (False, False, False)])
def test_top_level_switches_select_only_enabled_schemes(monkeypatch, enabled):
    for number, flag in enumerate(enabled, 1):
        monkeypatch.setattr(population, f"render_scheme_{number}", flag)
    assert population.enabled_schemes() == [f"scheme{i}" for i, flag in enumerate(enabled, 1) if flag]


def test_no_enabled_scheme_returns_without_loading_or_rendering(monkeypatch):
    for number in (1, 2, 3):
        monkeypatch.setattr(population, f"render_scheme_{number}", False)

    def unexpected(*args, **kwargs):
        pytest.fail("All schemes disabled must not read models, create outputs, or start workers")

    monkeypatch.setattr(population, "fit_models", unexpected)
    monkeypatch.setattr(population, "load_population_video_data", unexpected)
    monkeypatch.setattr(population, "select_hd", unexpected)
    monkeypatch.setattr(population, "configure_nvenc", unexpected)
    monkeypatch.setattr(population.video, "_select_encoder", unexpected)
    monkeypatch.setattr(population, "ProcessPoolExecutor", unexpected)
    monkeypatch.setattr(Path, "mkdir", unexpected)
    monkeypatch.setattr(Path, "read_text", unexpected)
    monkeypatch.setattr(Path, "read_bytes", unexpected)
    assert population.main([]) == []


def test_fit_models_uses_ab_pairs_and_only_enabled_cohorts(tmp_path, monkeypatch):
    hd_file = tmp_path / "session_b/data/tuning_curves/ProbeA/tuning_curves.tc"
    rf_file = tmp_path / "session_a/data/rfmapping/source.rfmap"
    output = tmp_path / "session_c/data/hd_rf/test_run"
    detection_path = tmp_path / "prepared/detected_rf.npz"
    pairs = pd.DataFrame(dict(unit_id=[7, 11, 19, 23],
                              hd_preferred_deg=[0., 90., 180., 270.],
                              rf_ego_deg=[40., 130., -140., -50.]))
    schemes = {"scheme1": pairs, "scheme2": pairs.iloc[:3], "scheme3": pairs.iloc[1:]}
    provenance = {"hd_source": str(hd_file), "rf_source": str(rf_file)}
    calls = []

    def unexpected_rf_analysis(*args, **kwargs):
        pytest.fail("Fitting must use the supplied detection, without loading or detecting RF maps")

    def load_pairs(hd_source, rf_source, **settings):
        assert (hd_source, rf_source) == (hd_file, rf_file)
        assert settings == dict(rf_detection_path=detection_path,
                                hd_is_clockwise=False, mouse="m20", date="260922", probe="A")
        calls.append("pairs")
        return pairs, provenance

    def fit(hd, rf, *, selected_on_sum):
        calls.append((hd.tolist(), rf.tolist(), selected_on_sum))
        return dict(method="first_harmonic_circular_regression", n=len(hd), rho=.25,
                    mae_deg=12., cos_coefficients=[1., 0., 0.], sin_coefficients=[0., 1., 0.])

    monkeypatch.setattr(population, "load_rfmap", unexpected_rf_analysis)
    monkeypatch.setattr(population, "detect_rf", unexpected_rf_analysis)
    monkeypatch.setattr(population, "load_peak_pairs", load_pairs)
    monkeypatch.setattr(population, "select_schemes", lambda received: (schemes, {"counts": [4, 3, 3]}))
    monkeypatch.setattr(population, "fit_circular_conversion", fit)
    models = population.fit_models(hd_file, rf_file, ["scheme2", "scheme3"], output,
                                   mouse="m20", date="260922", probe="A", hd_is_clockwise=False,
                                   rf_detection_path=detection_path)

    assert [model["scheme"] for model in models] == ["scheme2", "scheme3"]
    assert [model["selected_units"] for model in models] == [[7, 11, 19], [11, 19, 23]]
    assert [model["provenance"] for model in models] == [provenance, provenance]
    assert calls[0] == "pairs"
    assert [call[-1] for call in calls[1:]] == [False, True]
    assert not (output / "relation/scheme1_conversion.json").exists()
    for model in models:
        saved = json.loads(Path(model["model_path"]).read_text())
        assert saved["selected_units"] == model["selected_units"]
        assert saved["provenance"]["hd_source"] == str(hd_file)


def test_real_hd_does_not_load_decoder_tc_or_spikes(tmp_path, monkeypatch):
    data = dict(session=tmp_path / "260922_3", probe="A", position_source="motive_pose.csv",
                hd_path=tmp_path / "260922_3/data/processed/head_direction.json",
                hd=np.array([0., 90., np.nan]))

    def unexpected(*args, **kwargs):
        pytest.fail("Real HD selection must use the loaded tracked directions directly")

    monkeypatch.setattr(population, "load_decoder_tuning_curves", unexpected)
    monkeypatch.setattr(population.np, "load", unexpected)
    selected = population.select_hd(data, tmp_path / "absent.tc",
                                    use_decoded_hd=False, bin_size_s=.1)
    np.testing.assert_array_equal(selected["hd"], data["hd"])
    assert selected["hd_mode"] == "real"
    assert selected["hd_source"]["session"] == str(data["session"])
    assert str(data["hd_path"]) in selected["hd_source"].values()
    assert "hd_mode" not in data


def test_decoded_hd_uses_c_units_and_converts_native_ccw_once(tmp_path, monkeypatch):
    session = tmp_path / "260922_7"
    tc_file = session / "data/tuning_curves/ProbeA/tuning_curves.tc"
    tc_file.parent.mkdir(parents=True)
    tc_file.write_text("saved C tuning curves")
    kilosort = session / "kilosort/ProbeA/kilosort_7"
    kilosort.mkdir(parents=True)
    spike_dir = session / "data/probeA"
    spike_dir.mkdir(parents=True)
    np.save(spike_dir / "adc_spike_time.npy", [100.01, 100.11])
    np.save(kilosort / "spike_clusters.npy", [10, 99])
    curves = pd.DataFrame([[5., 0.], [0., 6.]], index=[10, 99], columns=[0., 90.])
    curves.attrs["metadata"] = dict(session=session.name, adc_time_origin_raw_s=100.,
                                    angle_convention_note="0 degrees up, positive counter-clockwise",
                                    kilosort_dir=str(kilosort))
    monkeypatch.setattr(population, "tc_is_clockwise", False)
    monkeypatch.setattr(population, "load_decoder_tuning_curves", lambda path: curves)
    data = dict(session=session, probe="A", hd=np.array([180., 180.]),
                times=np.array([.05, .15]), adc_time_origin_s=100.,
                decoding_intervals_s=np.array([[0., .2]]))

    selected = population.select_hd(data, tc_file, use_decoded_hd=True, bin_size_s=.1)

    np.testing.assert_array_equal(selected["hd"], [0., 270.])
    np.testing.assert_array_equal(data["hd"], [180., 180.])
    assert selected["hd_source"]["unit_ids"] == [10, 99]
    assert selected["hd_source"]["valid_bins"] == 2
    assert selected["hd_mode"] == "decoded"


def test_frame_window_uses_measured_clock_and_retains_video_ordinals():
    data = dict(exposure_times=np.array([.5, .54, np.nan, .70, .74]), video_frame_count=5)
    assert population.frame_window(data, .15, .1) == (3, 5)
    with pytest.raises(ValueError, match="after the measured video clock"):
        population.frame_window(data, 1., .1)


@pytest.mark.parametrize("scheme", ["scheme1", "scheme2", "scheme3"])
def test_worker_exports_separate_silent_gpu_video_and_exact_frame_angles(tmp_path, monkeypatch, scheme):
    session = tmp_path / "260827_11"
    model = regression_model(session, scheme)
    model.update(model_path=f"{scheme}_conversion.json", model_sha256="source-hash")
    data = dict(session=session, video_path=session / "260827.avi", hd_path="head_direction.json",
                position_source="dark_pose.csv", camera_timing={}, frame_mapping={},
                frame_ids=np.array([0, 1, 3, 4]), times=np.array([0., .04, .12, .16]),
                exposure_times=np.arange(7) / 25., frame_period_s=1 / 25,
                hd=np.array([0., 90., 350., 180.]), hd_mode="real",
                hd_source={"method": "tracked_hd"}, video_frame_count=7, video_clock={})
    calls = []

    def export(received, source, destination, **settings):
        calls.append((received, source, destination, settings))
        return dict(source_first_frame=1, source_last_frame=3, ray_bearings_deg=[0.],
                    outside_arena_frames=0, frames=3)

    monkeypatch.setattr(population, "configure_nvenc", lambda: None)
    monkeypatch.setattr(population.video, "export_ebc_overlay", export)
    result = population.render_scheme(data, model, output_dir=tmp_path, start_s=.04,
                                      duration_s=.12, trace_window_s=8.)
    received, source, destination, settings = calls[0]
    assert source == data["video_path"] and destination == tmp_path / f"{scheme}.mp4"
    assert received["model"] == model and received["trace_window_s"] == 8.
    assert "model" not in data  # Different schemes share the original loaded data.
    assert settings == dict(start_s=.04, duration_s=.12, workers=1, frame_rate=25.,
                            first_frame=1, stop_frame=4,
                            overlay_type=PopulationOverlay, video_encoder="h264_nvenc")
    frames = pd.read_csv(tmp_path / f"{scheme}_frame_angles.csv")
    assert frames.video_frame.tolist() == [1, 3]
    np.testing.assert_allclose(frames.rf_ego_deg, [130., 30.])
    np.testing.assert_allclose(frames.rf_allo_deg, [220., 20.])
    saved = json.loads((tmp_path / f"{scheme}.json").read_text())
    assert saved == result and saved["audio"] is False
    assert saved["model_kind"] == "first_harmonic_circular_regression"
    assert saved["fitted_angular_error_deg"] == 0.
    assert "beta_deg" not in saved
    assert "fit_validation" not in saved and "ray_bearings_deg" not in saved


def test_worker_failure_does_not_write_success_outputs(tmp_path, monkeypatch):
    session = tmp_path / "260827_11"

    def fail(*args, **kwargs):
        raise RuntimeError("NVENC worker failed")

    monkeypatch.setattr(population, "configure_nvenc", lambda: None)
    monkeypatch.setattr(population.video, "export_ebc_overlay", fail)
    with pytest.raises(RuntimeError, match="NVENC worker failed"):
        population.render_scheme(dict(video_path=session / "260827.avi", frame_period_s=1 / 25,
                                      exposure_times=np.arange(7) / 25., video_frame_count=7),
                                 regression_model(session),
                                 output_dir=tmp_path, start_s=0., duration_s=.12, trace_window_s=8.)
    assert not (tmp_path / "scheme1.json").exists()
    assert not (tmp_path / "scheme1_frame_angles.csv").exists()


def test_export_uses_measured_rate_and_exact_source_frames_with_a_clock_gap(tmp_path):
    session = tmp_path / "260922_3"
    session.mkdir()
    source = session / "260922.avi"
    levels = [30, 60, 90, 120, 150]
    payload = b"".join(bytes([level]) * (1280 * 1024 * 3) for level in levels)
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x1024", "-r", "110.5338", "-i", "pipe:0",
        "-c:v", "ffv1", "-threads", "1", str(source),
    ], input=payload, check=True)
    data = dict(session=session, probe="A", phase="baseline", hd_mode="real",
                hd=np.array([0., 90., 180., 270.]), arena_type="cylinder",
                xyz=np.tile([0., 0., 100.], (4, 1)),
                projection=np.array([[1., 0., 0., 645.], [0., -1., 0., 485.], [0., 0., 0., 1.]]),
                cylinder_center_raw=np.zeros(2), cylinder_radius_raw=200., raw_units_per_cm=37.4,
                model=regression_model(session), trace_window_s=12., frame_period_s=.04,
                frame_ids=np.array([0, 1, 3, 4]), times=np.array([10., 10.04, 10.12, 10.16]),
                exposure_times=np.array([10., 10.04, np.nan, 10.12, 10.16]))
    destination = tmp_path / "overlay.mp4"

    result = population.video.export_ebc_overlay(
        data, source, destination, overlay_type=PopulationOverlay, video_encoder="libx264",
        frame_rate=25., first_frame=1, stop_frame=4,
    )

    assert result["frames"] == 3 and result["missing_pose_frames"] == 1
    assert result["source_first_frame"] == 1 and result["source_last_frame"] == 3
    assert result["duration_s"] == pytest.approx(.12)
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(destination),
    ]))
    assert len(probe["streams"]) == 1
    assert float(Fraction(probe["streams"][0]["avg_frame_rate"])) == 25.
    decoded = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(destination), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ])
    images = np.frombuffer(decoded, np.uint8).reshape(-1, 1324, 1650, 3)
    np.testing.assert_allclose(images[:, 85, 50, 0], levels[1:4], atol=3)


@pytest.mark.parametrize("worker_error", [False, True])
@pytest.mark.parametrize("decoded_hd", [False, True])
def test_main_fits_ab_and_applies_to_independent_c_with_spawn(tmp_path, monkeypatch,
                                                           worker_error, decoded_hd):
    for number, flag in enumerate((True, False, True), 1):
        monkeypatch.setattr(population, f"render_scheme_{number}", flag)
    root = tmp_path / "m20/260922"
    rf_session, hd_session, session = [root / f"260922_{number}" for number in (1, 3, 7)]
    models = [regression_model(hd_session, name) for name in ("scheme1", "scheme3")]
    data = {"session": session, "frame_period_s": 1 / 25,
            "frame_ids": np.array([0, 1]), "times": np.array([0., .04]),
            "hd": np.array([0., 90.])}
    events, submitted, exited = [], [], []
    output = session / "data/hd_rf/test_run"
    monkeypatch.setattr(population, "use_decoded_hd", False)
    monkeypatch.setattr(population, "tc_is_clockwise", False)
    rf_path = rf_session / "data/rfmapping/good/-100_400_1ms/ProbeA/regular_unitsSpikeCounts_260922_1.rfmap"
    detection_path = output / "relation/rf_detection.npz"
    response = object()

    class Rates:
        def mean_rate(self, first, last, *, show_progress):
            assert (first, last, show_progress) == (0., .2, False)
            events.append("mean_rate")
            return response

    class RawMaps:
        def to_firing_rate(self, *, reconstruct_presentations):
            assert reconstruct_presentations is True
            events.append("to_firing_rate")
            return Rates()

    def load_rfmap(source):
        assert source == rf_path
        events.append("load_rfmap")
        return RawMaps()

    def detect_rf(received, **settings):
        assert received is response
        assert settings == dict(is_shuffle=False, cluster_forming_z=1.8, drop_bins=2,
                                wrap_x=True, result_path=detection_path, show_progress=False)
        events.append("detect_rf")

    def fit_models(hd_file, rf_file, names, output_dir, **settings):
        assert names == ["scheme1", "scheme3"]
        assert hd_file == hd_session / "data/tuning_curves/ProbeA/tuning_curves.tc"
        assert rf_file == rf_path
        assert output_dir == output
        assert settings == dict(mouse="m20", date="260922", probe="A", hd_is_clockwise=False,
                                rf_detection_path=detection_path)
        events.append("models")
        return models

    def select_encoder(*args):
        assert args == ("h264_nvenc", 1650, 1324, "25")
        events.append("encoder")
        return "h264_nvenc"

    def load_data(selected_session, **settings):
        assert selected_session == session
        assert settings == dict(probe="A", phase="baseline",
                                motive_clock=population.motive_clock,
                                basler_clock=population.basler_clock,
                                video_stride=2, video_phase=0, video_segments=None,
                                camera_registration_path=population.camera_registration_path,
                                cylinder_calibration_path=population.cylinder_calibration_path)
        events.append("data")
        return data

    def select_hd(received, tc_file, **settings):
        assert received is data
        assert tc_file == session / "data/tuning_curves/ProbeA/tuning_curves.tc"
        assert settings == dict(use_decoded_hd=decoded_hd, bin_size_s=.1)
        events.append("select")
        return data

    class Pool:
        def __init__(self, *, max_workers, mp_context):
            assert max_workers == 2 and mp_context.get_start_method() == "spawn"
            events.append("pool")

        def __enter__(self):
            return self

        def submit(self, function, shared_data, model, **settings):
            assert function is population.render_scheme and shared_data is data
            submitted.append((model["scheme"], settings))
            future = Future()
            if worker_error and model["scheme"] == "scheme1":
                future.set_exception(RuntimeError("scheme1 failed"))
            else:
                future.set_result({"scheme": model["scheme"]})
            return future

        def __exit__(self, *exception):
            exited.append(True)

    monkeypatch.setattr(population, "fit_models", fit_models)
    monkeypatch.setattr(population, "load_rfmap", load_rfmap)
    monkeypatch.setattr(population, "detect_rf", detect_rf)
    monkeypatch.setattr(population, "configure_nvenc", lambda: None)
    monkeypatch.setattr(population.video, "_select_encoder", select_encoder)
    monkeypatch.setattr(population, "load_population_video_data", load_data)
    monkeypatch.setattr(population, "select_hd", select_hd)
    monkeypatch.setattr(population, "ProcessPoolExecutor", Pool)
    output.mkdir(parents=True)
    (output / "render_manifest.json").write_text('[{"scheme": "stale_success"}]')
    arguments = ["--rf-session", str(rf_session), "--hd-session", str(hd_session),
                 "--video-session", str(session), "--start", "2", "--duration", ".12",
                 "--trace-window", "8", "--output-name", "test_run"]
    if decoded_hd:
        arguments.append("--decoded-hd")
    if worker_error:
        with pytest.raises(RuntimeError, match="scheme1 failed"):
            population.main(arguments)
        assert not (output / "render_manifest.json").exists()
    else:
        assert population.main(arguments) == [{"scheme": "scheme1"}, {"scheme": "scheme3"}]
        assert json.loads((output / "render_manifest.json").read_text()) == [
            {"scheme": "scheme1"}, {"scheme": "scheme3"}]
    assert events == ["encoder", "load_rfmap", "to_firing_rate", "mean_rate", "detect_rf",
                      "models", "data", "select", "pool"]
    assert [name for name, _ in submitted] == ["scheme1", "scheme3"]
    for _, settings in submitted:
        assert settings == dict(output_dir=output, start_s=2., duration_s=.12, trace_window_s=8.)
    assert exited == [True]
