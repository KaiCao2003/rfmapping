import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

import hd_rf_population_video as population
from hd_rf_population_video import PopulationOverlay, screen_direction, wrapped_segments


def phase_model(session, scheme="scheme1", beta=40.):
    return dict(method="reversed_circular_phase", scheme=scheme, n=18,
                beta_deg=beta, rho_circular=-.2, fit_mae_deg=50.3,
                selected_on_sum=scheme == "scheme3",
                provenance={"hd_source": str(session / "data/tuning_curves/ProbeA/tuning_curves.json")})


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
                arena_size_cm=41., model=phase_model(session), times=np.array([.04]),
                exposure_times=np.array([0., .04]), trace_window_s=12.)
    overlay = PopulationOverlay(data, 1280, 1024, 25.)
    np.testing.assert_allclose(overlay.positions[0], [645., 485.])
    np.testing.assert_allclose(overlay.rf_ego, [-50.], atol=1e-8)
    np.testing.assert_allclose(overlay.rf_allo, [40.], atol=1e-8)
    np.testing.assert_allclose(screen_direction([0., 90., 180., 270.]),
                               [[0., -1.], [1., 0.], [0., 1.], [-1., 0.]], atol=1e-12)
    canvas = overlay.draw(Image.new("RGB", (1280, 1024), "white"), 1, 0)
    assert canvas.size == (1650, 1324)
    # Frames without an exposure retain the source picture and show no geometry.
    gap = overlay.draw(Image.new("RGB", (1280, 1024), "white"), 2, -1)
    assert gap.getpixel((645, 485)) == (255, 255, 255)
    assert not any("held-out" in label or "kernel" in label for label in labels)


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

    monkeypatch.setattr(population, "load_models", unexpected)
    monkeypatch.setattr(population, "load_scheme_video_data", unexpected)
    monkeypatch.setattr(population, "configure_nvenc", unexpected)
    monkeypatch.setattr(population.video, "_select_encoder", unexpected)
    monkeypatch.setattr(population, "ProcessPoolExecutor", unexpected)
    monkeypatch.setattr(Path, "mkdir", unexpected)
    monkeypatch.setattr(Path, "read_text", unexpected)
    monkeypatch.setattr(Path, "read_bytes", unexpected)
    assert population.main([]) == []


def test_load_models_reads_only_enabled_files(tmp_path):
    session = tmp_path / "260827_11"
    model = phase_model(session, "scheme2")
    (tmp_path / "scheme2_conversion.json").write_text(json.dumps(model))
    # A disabled scheme may be absent or malformed without affecting selection.
    (tmp_path / "scheme3_conversion.json").write_text("not JSON")
    loaded = population.load_models(tmp_path, ["scheme2"], session)
    assert len(loaded) == 1
    assert loaded[0]["scheme"] == "scheme2"
    assert loaded[0]["beta_deg"] == model["beta_deg"]


@pytest.mark.parametrize("change", ["source_session", "method", "scheme"])
def test_load_models_rejects_incompatible_source_and_model(tmp_path, change):
    session = tmp_path / "260827_11"
    model = phase_model(session)
    if change == "source_session":
        model["provenance"]["hd_source"] = str(tmp_path / "260827_12/data/tuning_curves/ProbeA/tuning_curves.json")
    elif change == "method":
        model["method"] = "circular_kernel"
    else:
        model["scheme"] = "scheme3"
    (tmp_path / "scheme1_conversion.json").write_text(json.dumps(model))
    with pytest.raises(ValueError):
        population.load_models(tmp_path, ["scheme1"], session)


@pytest.mark.parametrize("scheme", ["scheme1", "scheme2", "scheme3"])
def test_worker_exports_separate_silent_gpu_video_and_exact_frame_angles(tmp_path, monkeypatch, scheme):
    session = tmp_path / "260827_11"
    model = phase_model(session, scheme)
    model.update(model_path=f"{scheme}_conversion.json", model_sha256="source-hash")
    data = dict(session=session, video_path=session / "260827.avi", hd_path="head_direction.json",
                position_source="dark_pose.csv", camera_timing={}, frame_mapping={},
                frame_ids=np.array([0, 1, 3, 4]), times=np.array([0., .04, .12, .16]),
                hd=np.array([0., 90., 350., 180.]))
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
    assert settings == dict(start_s=.04, duration_s=.12, workers=1,
                            overlay_type=PopulationOverlay, video_encoder="h264_nvenc")
    frames = pd.read_csv(tmp_path / f"{scheme}_frame_angles.csv")
    assert frames.video_frame.tolist() == [1, 3]
    np.testing.assert_allclose(frames.rf_ego_deg, [-50., 50.])
    np.testing.assert_allclose(frames.rf_allo_deg, [40., 40.])
    saved = json.loads((tmp_path / f"{scheme}.json").read_text())
    assert saved == result and saved["audio"] is False
    assert saved["model_kind"] == "reversed_circular_phase"
    assert saved["fitted_angular_error_deg"] == 50.3
    assert "fit_validation" not in saved and "ray_bearings_deg" not in saved


def test_worker_failure_does_not_write_success_outputs(tmp_path, monkeypatch):
    session = tmp_path / "260827_11"

    def fail(*args, **kwargs):
        raise RuntimeError("NVENC worker failed")

    monkeypatch.setattr(population, "configure_nvenc", lambda: None)
    monkeypatch.setattr(population.video, "export_ebc_overlay", fail)
    with pytest.raises(RuntimeError, match="NVENC worker failed"):
        population.render_scheme(dict(video_path=session / "260827.avi"), phase_model(session),
                                 output_dir=tmp_path, start_s=0., duration_s=.12, trace_window_s=8.)
    assert not (tmp_path / "scheme1.json").exists()
    assert not (tmp_path / "scheme1_frame_angles.csv").exists()


@pytest.mark.parametrize("worker_error", [False, True])
def test_main_submits_enabled_models_with_spawn_and_propagates_failures(tmp_path, monkeypatch, worker_error):
    for number, flag in enumerate((True, False, True), 1):
        monkeypatch.setattr(population, f"render_scheme_{number}", flag)
    session = tmp_path / "260827/260827_11"
    models = [phase_model(session, name) for name in ("scheme1", "scheme3")]
    data = {"session": session}
    events, submitted, exited = [], [], []

    def load_models(directory, names, selected_session):
        assert names == ["scheme1", "scheme3"] and selected_session == session
        events.append("models")
        return models

    def select_encoder(*args):
        assert args == ("h264_nvenc", 1650, 1324, "25")
        events.append("encoder")
        return "h264_nvenc"

    def load_data(selected_session, **settings):
        assert selected_session == session
        assert settings == dict(probe="A", phase="baseline", wall_config="old")
        events.append("data")
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

    monkeypatch.setattr(population, "load_models", load_models)
    monkeypatch.setattr(population, "configure_nvenc", lambda: None)
    monkeypatch.setattr(population.video, "_select_encoder", select_encoder)
    monkeypatch.setattr(population, "load_scheme_video_data", load_data)
    monkeypatch.setattr(population, "ProcessPoolExecutor", Pool)
    output = tmp_path / "videos"
    output.mkdir()
    (output / "render_manifest.json").write_text('[{"scheme": "stale_success"}]')
    arguments = ["--base-dir", str(tmp_path), "--date", "260827", "--session", "11",
                 "--start", "2", "--duration", ".12", "--trace-window", "8", "--output-dir", str(output)]
    if worker_error:
        with pytest.raises(RuntimeError, match="scheme1 failed"):
            population.main(arguments)
        assert not (output / "render_manifest.json").exists()
    else:
        assert population.main(arguments) == [{"scheme": "scheme1"}, {"scheme": "scheme3"}]
        assert json.loads((output / "render_manifest.json").read_text()) == [
            {"scheme": "scheme1"}, {"scheme": "scheme3"}]
    assert events == ["models", "encoder", "data", "pool"]
    assert [name for name, _ in submitted] == ["scheme1", "scheme3"]
    for _, settings in submitted:
        assert settings == dict(output_dir=output, start_s=2., duration_s=.12, trace_window_s=8.)
    assert exited == [True]
