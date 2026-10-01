import json
import shutil

import numpy as np
import pandas as pd
import pytest

from Utils import hd_rf_video_data


@pytest.fixture
def session_source(tmp_path, monkeypatch):
    session = tmp_path / "260827_11"
    data = session / "data"
    tc = data / "tuning_curves/ProbeA/tuning_curves.json"
    tc.parent.mkdir(parents=True)
    tc.write_text(json.dumps({"metadata": {"headplate": "hp4", "ttl_qc": {
        "frame_timestamp_mapping": "pose_frame_id_minus_one_to_exposure_midpoint",
        "camera_input_channel": 2, "camera_ttl_threshold": 12345., "camera_ttl_active_high": True,
    }}}))
    (data / "processed").mkdir()
    heading = data / "processed/head_direction.json"
    heading.write_text(json.dumps({"hp4": {"frames": [1, 2, 4, 6], "hd": [0., 90., 270., 45.]}}))
    pose = pd.DataFrame({"frame": np.arange(5), "back_x": [645.] * 5, "back_y": [485.] * 5,
                         "front_x": [645., 655., 645., 635., 645.],
                         "front_y": [475., 485., 495., 485., 475.],
                         "hd_deg": [123.] * 5})  # Intentionally wrong: must never supply HD.
    pose.to_csv(data / "dark_pose.csv", index=False)
    (data / "session_info.json").write_text(json.dumps({"session_info": {"test": True}}))
    pd.DataFrame({"interval_type": ["baseline"], "start": [1.], "end": [4.]}).to_csv(
        data / "interval_table.csv", index=False,
    )
    monkeypatch.setattr(hd_rf_video_data, "_probe_video", lambda path: {
        "width": 1280, "height": 1024, "fps": 25., "declared_frame_count": 6, "decoded_frame_count": 5,
    })

    def exposures(info, directory, **kwargs):
        assert kwargs == {"camera_input_channel": 2, "camera_ttl_threshold": 12345.,
                          "camera_ttl_active_high": True}
        return np.array([.5, 1.5, 2.5, 3.5]), 10., {"ttl_pulse_count": 4}

    monkeypatch.setattr(hd_rf_video_data, "get_exposure_timestamps", exposures)
    return session, tc, heading


def test_one_based_json_joins_zero_based_pose_without_using_csv_heading(session_source):
    session, tc, heading = session_source
    original_tc = tc.read_bytes()
    data = hd_rf_video_data.load_scheme_video_data(session)
    np.testing.assert_array_equal(data["frame_ids"], [1, 3])
    np.testing.assert_array_equal(data["hd"], [90., 270.])
    np.testing.assert_array_equal(data["times"], [1.5, 3.5])
    left, right, top, bottom = data["bounds_px"]
    pixels = data["xy"] * [(right - left) / 41., -(bottom - top) / 41.] + [left, bottom]
    np.testing.assert_allclose(pixels, [[655., 485.], [635., 485.]])
    assert data["frame_mapping"]["json_frame_offset_to_video"] == -1
    assert data["frame_mapping"]["geometry_audit_frames"] == 3
    assert data["frame_mapping"]["geometry_max_error_deg"] == 0.
    assert data["frame_mapping"]["video_frames_without_ttl"] == 1
    assert data["dropped_unsynchronized_pose_frames"] == 1
    assert data["video_frame_count"] == 5
    assert len(data["exposure_times"]) == 4  # Unmeasured timing is not invented.
    assert data["video_path"] == session / "260827.avi"
    assert data["position_source"] == str(session / "data/dark_pose.csv")
    assert data["hd_path"] == str(heading)
    json.dumps(data["frame_mapping"], allow_nan=False)
    assert tc.read_bytes() == original_tc


def test_zero_based_metadata_uses_exact_ids_and_trims_extra_ttl(session_source, monkeypatch):
    session, tc, heading = session_source
    metadata = json.loads(tc.read_text())
    metadata["metadata"]["frame_mapping"] = "zero_based_csv_frame"
    tc.write_text(json.dumps(metadata))
    heading.write_text(json.dumps({"hp4": {"frames": [0, 1, 3], "hd": [0., 90., 270.]}}))
    monkeypatch.setattr(hd_rf_video_data, "get_exposure_timestamps",
                        lambda *a, **k: (np.arange(7.) + .5, 10., {}))
    data = hd_rf_video_data.load_scheme_video_data(session)
    np.testing.assert_array_equal(data["frame_ids"], [1, 3])
    assert data["frame_mapping"]["json_frame_offset_to_video"] == 0
    assert data["frame_mapping"]["trailing_ttl_without_video"] == 2
    assert len(data["exposure_times"]) == data["video_frame_count"] == 5


def test_missing_mapping_is_rejected_instead_of_guessing_from_min_frame(session_source):
    session, tc, _ = session_source
    metadata = json.loads(tc.read_text())
    del metadata["metadata"]["ttl_qc"]["frame_timestamp_mapping"]
    tc.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Explicit JSON-to-video frame mapping"):
        hd_rf_video_data.load_scheme_video_data(session)


def test_wrong_json_sign_is_rejected_by_geometry_audit(session_source):
    session, _, heading = session_source
    heading.write_text(json.dumps({"hp4": {"frames": [1, 2, 4], "hd": [0., 270., 90.]}}))
    with pytest.raises(ValueError, match="CW front/back geometry"):
        hd_rf_video_data.load_scheme_video_data(session)


def test_duplicate_json_frames_are_rejected(session_source):
    session, _, heading = session_source
    heading.write_text(json.dumps({"hp4": {"frames": [1, 1], "hd": [0., 0.]}}))
    with pytest.raises(ValueError, match="unique increasing"):
        hd_rf_video_data.load_scheme_video_data(session)


@pytest.fixture
def population_source(tmp_path, monkeypatch):
    session = tmp_path / "260922_3"
    directory = session / "data"
    processed = directory / "processed"
    processed.mkdir(parents=True)
    header = (
        ",Name,hp4,hp4,hp4,hp4,hp4,hp4\n"
        ",ID,body,body,body,body,body,body\n"
        ",,Rotation,Rotation,Rotation,Position,Position,Position\n"
        "Frame,Time (Seconds),X,Y,Z,X,Y,Z\n"
    )
    rows = [f"{i},{i / 2},0,0,45,{100 + i},{200 + i},{300 + i}" for i in range(10)]
    (processed / "trimmed_input.csv").write_text(header + "\n".join(rows) + "\n")
    pd.DataFrame({"interval_type": ["baseline"], "start": [1.], "end": [4.]}).to_csv(
        directory / "interval_table.csv", index=False,
    )
    (processed / "head_direction.json").write_text(json.dumps({
        "hp4": {"frames": [0, 2, 6, 8], "head_direction_deg": [11., 123., 246., 321.]},
    }))
    (directory / "session_info.json").write_text(json.dumps({"session_info": {"test": True}}))
    registration = tmp_path / "shared_camera.json"
    registration.write_text(json.dumps({"world_to_video_projection_raw": [
        [1., 0., 0., 10.], [0., 1., 0., 20.], [0., 0., 0., 1.],
    ]}))
    calibration = tmp_path / "shared_cylinder.calib"
    calibration.write_text(json.dumps({
        "screen_bottom_center": [12., 34., 0.], "screen_top_center": [12., 34., 100.],
        "screen_diamter": 200., "motive_units_per_real_mm": 3.74,
    }))
    monkeypatch.setattr(hd_rf_video_data, "_probe_video", lambda path: {
        "width": 1280, "height": 1024, "fps": 110.,
        "declared_frame_count": 6, "decoded_frame_count": 5,
    })
    motive_times = np.arange(10) / 2 + .5
    basler_times = np.delete(motive_times, 4)  # The missing pulse must remain an AVI-clock gap.

    def raw_exposures(info, **kwargs):
        assert info == {"test": True}
        if kwargs == {"camera_input_channel": 1, "camera_ttl_threshold": 14000., "camera_ttl_active_high": True}:
            return motive_times, 10., tmp_path / "raw_adc.dat"
        assert kwargs == {"camera_input_channel": 2, "camera_ttl_threshold": 2800., "camera_ttl_active_high": False}
        return basler_times, 10., tmp_path / "raw_adc.dat"

    monkeypatch.setattr(hd_rf_video_data, "_read_exposure_timestamps", raw_exposures)
    return session, registration, calibration


def load_population(source):
    session, registration, calibration = source
    return hd_rf_video_data.load_population_video_data(
        session, probe="A", phase="baseline", video_stride=2, video_phase=0,
        motive_clock={"camera_input_channel": 1, "camera_ttl_threshold": 14000., "camera_ttl_active_high": True},
        basler_clock={"camera_input_channel": 2, "camera_ttl_threshold": 2800., "camera_ttl_active_high": False},
        camera_registration_path=registration, cylinder_calibration_path=calibration,
    )


def test_population_reads_motive_xyz_and_json_without_yolo_or_external_clock(population_source):
    data = load_population(population_source)
    np.testing.assert_array_equal(data["frame_ids"], [1, 3])
    np.testing.assert_array_equal(data["hd"], [123., 246.])
    np.testing.assert_array_equal(data["times"], [1.5, 3.5])
    np.testing.assert_array_equal(data["xyz"], [[102., 202., 302.], [106., 206., 306.]])
    np.testing.assert_array_equal(data["projection"], [[1., 0., 0., 10.], [0., 1., 0., 20.], [0., 0., 0., 1.]])
    np.testing.assert_array_equal(data["cylinder_center_raw"], [12., 34.])
    assert data["cylinder_radius_raw"] == 100.
    assert data["raw_units_per_cm"] == pytest.approx(37.4)
    assert data["arena_type"] == "cylinder"
    assert len(data["exposure_times"]) == data["video_frame_count"] == 5
    assert np.isnan(data["exposure_times"][2])
    assert data["dropped_unsynchronized_pose_frames"] == 1
    assert data["adc_time_origin_s"] == 10.
    assert data["frame_period_s"] == 1.
    np.testing.assert_allclose(data["decoding_intervals_s"], [[1., 2.], [3., 4.]])
    assert data["video_clock"]["uncertain_video_frame_ranges_inclusive"] == [[2, 2]]
    assert data["hd_path"] == str(population_source[0] / "data/processed/head_direction.json")
    assert data["hd_source"] == "Motive processed/head_direction.json"
    assert data["camera_registration_path"] == str(population_source[1])
    assert data["cylinder_calibration_path"] == str(population_source[2])
    assert not (population_source[0] / "data/260922.csv").exists()
    assert not (population_source[0] / "data/hd_rf/camera_clock").exists()
    json.dumps(data["frame_mapping"], allow_nan=False)


def test_population_reuses_shared_geometry_with_another_session(population_source):
    session, registration, calibration = population_source
    other = session.parent / "260921_9"
    shutil.copytree(session, other)
    first = load_population(population_source)
    second = load_population((other, registration, calibration))
    assert second["session"] == other
    assert second["camera_registration_path"] == first["camera_registration_path"]
    assert second["camera_registration_sha256"] == first["camera_registration_sha256"]
    assert second["cylinder_calibration_sha256"] == first["cylinder_calibration_sha256"]
    np.testing.assert_array_equal(second["projection"], first["projection"])
    np.testing.assert_array_equal(second["hd"], first["hd"])


def test_population_rejects_incorrect_decoded_avi_count(population_source, monkeypatch):
    monkeypatch.setattr(hd_rf_video_data, "_probe_video", lambda path: {
        "width": 1280, "height": 1024, "fps": 110.,
        "declared_frame_count": 6, "decoded_frame_count": 6,
    })
    with pytest.raises(ValueError, match="explicit stride/phase gives 5 video frames, but the AVI has 6"):
        load_population(population_source)


def test_population_excludes_nonfinite_motive_position(population_source):
    session, _, _ = population_source
    path = session / "data/processed/trimmed_input.csv"
    path.write_text(path.read_text().replace("102,202,302", "nan,202,302"))
    data = load_population(population_source)
    np.testing.assert_array_equal(data["frame_ids"], [3])
    np.testing.assert_array_equal(data["hd"], [246.])


def test_population_requires_shared_camera_registration(population_source):
    _, registration, _ = population_source
    registration.unlink()
    with pytest.raises(FileNotFoundError):
        load_population(population_source)
