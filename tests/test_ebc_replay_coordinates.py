import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from EBB import ebc_replay
from EBB.ebc_pose import POSITION_CORRECTION_POLICY
from EBB.ebc_screen import ScreenEBCFigure, circle_distances, signed_bearing


@pytest.mark.parametrize("heading", [0., 90., 355.])
def test_screen_bearings_recover_clockwise_physical_endpoints(heading):
    position = np.array([3., 4.])
    bearings = np.array([0., 90., 180., 270.])
    distances, directions = circle_distances(position, heading, np.zeros(2), 10., bearings)
    endpoints = position + distances[:, None] * directions
    np.testing.assert_allclose(np.linalg.norm(endpoints, axis=1), 10., atol=1e-12)
    error = (signed_bearing(position, heading, endpoints) - bearings + 180) % 360 - 180
    np.testing.assert_allclose(error, 0., atol=1e-12)
    forward = np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
    np.testing.assert_allclose(directions[0], forward, atol=1e-12)
    np.testing.assert_allclose(directions[1], [forward[1], -forward[0]], atol=1e-12)


def test_signed_display_retains_midpoint_geometry_and_rates():
    rate = np.arange(120.).reshape(60, 2)
    edges, signed = ebc_replay.signed_ebc_display(rate)
    source_angles = np.arange(3., 360., 6.)
    wrapped = (source_angles + 180) % 360 - 180
    order = np.argsort(wrapped)
    np.testing.assert_array_equal((edges[:-1] + edges[1:]) / 2, wrapped[order])
    np.testing.assert_array_equal(signed, rate[order])
    assert signed.shape == rate.shape
    np.testing.assert_array_equal(edges[[0, -1]], [-180., 180.])


@pytest.fixture
def replay_payload(tmp_path, monkeypatch):
    session = tmp_path / "m00/260101/260101_1"
    data = session / "data"
    (data / "processed").mkdir(parents=True)
    (data / "probeA").mkdir()
    calibration = dict(fm_eye_position_model="xy_offset_rotated_by_calibrated_hd",
                       screen_center=0., hp_to_eye_center=[0., 0., 0.],
                       motive_units_per_real_mm=1., screen_bottom_center=[0., 0., 0.],
                       screen_diamter=200.)
    (session / "260101.calib").write_text(json.dumps(calibration))
    (data / "session_info.json").write_text(json.dumps({"session_info": {}}))
    times = np.arange(20) * .02
    monkeypatch.setattr(ebc_replay, "get_exposure_timestamps",
                        lambda *a, **k: (times, 0., {"timestamp_reference": "test_midpoints"}))
    columns = pd.MultiIndex.from_tuples([
        ("Frame", "Frame", "Frame", "Frame"),
        *[("Rigid Body", "hp4", kind, axis)
          for kind in ("Position", "Rotation") for axis in "XYZ"],
    ])
    values = np.zeros((20, 7))
    values[:, 0] = np.arange(20)
    values[:, 1:3] = [2., 1.]
    pose = pd.DataFrame(values, columns=columns)
    pose.to_csv(data / "processed/filtered.csv", index=False)
    raw_lines = pose.to_csv(index=False).splitlines()
    raw_header = ["Motive", "Export", raw_lines[0], raw_lines[1], "", "", raw_lines[2], raw_lines[3]]
    (session / "260101.csv").write_text("\n".join(raw_header + raw_lines[4:]) + "\n")
    # A conflicting fused/CW file must never become the planar world-Z heading.
    (data / "processed/head_direction.json").write_text(json.dumps({
        "hp4": {"frames": list(range(20)), "head_direction_deg": [123.] * 20},
    }))
    for name, diameter in (("inner", 30.), ("outer", 60.)):
        folder = data / f"spatial_cells/ProbeA/baseline/{name}"
        folder.mkdir(parents=True)
        metadata = dict(angleConvention="egocentric clockwise: 0 forward, 90 right",
                        angleSampling="bin_centers", boundaryCenterRaw=[0., 0.],
                        positionCorrectionPolicy=POSITION_CORRECTION_POLICY,
                        cmPerPositionUnit=.1, boundaryDiameterCm=diameter,
                        xBinEdges=[0., diameter / 2, diameter], yBinEdges=list(np.arange(0., 361., 6.)))
        with (folder / "egocentric_rate_map.rfmap").open("wb") as stream:
            np.savez(stream, unitPool=[1], metadata=np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8),
                     unit_1=np.arange(120.).reshape(60, 2, 1), timeBinEdges=[0., .38])
    savemat(session / "260101.mat", {"trials": np.array([
        {"Square_PositionX": 90., "Square_Size": 0., "Square_Luminance": 1.},
        {"Square_PositionX": 0., "Square_Size": 0., "Square_Luminance": 1.},
    ], dtype=object)})
    np.save(data / "on_list_times.npy", [0., .2, .4])
    kilosort = session / "kilosort/ProbeA/kilosort_1"
    kilosort.mkdir(parents=True)
    (kilosort / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n")
    np.save(kilosort / "spike_clusters.npy", [1, 1])
    np.save(data / "probeA/adc_spike_time.npy", [.04, .22])
    payload, qc = ebc_replay.export_session(session, 1, tmp_path / "replay.json", fps=50., preview_start=0.)
    return payload, qc


def test_export_retains_world_z_yaw_and_midpoint_timing(replay_payload):
    payload, qc = replay_payload
    np.testing.assert_array_equal(payload["frames"]["heading_deg"], np.zeros(20))
    np.testing.assert_allclose(payload["frames"]["time_s"], np.arange(20) * .02)
    np.testing.assert_array_equal(payload["frames"]["frame_id"], np.arange(20))
    assert payload["meta"]["egocentric_angle_convention"] == "clockwise: 0 forward, 90 right"
    assert qc["vs_ray_endpoint_max_error_cm"] < 1e-12
    assert qc["ebc_transform_max_error_model_cm"] < 1e-12
    assert all(model["ray_sample_location"] == "saved_bin_center" for model in payload["ebc"])


def test_screen_draw_uses_clockwise_vs_bearings(replay_payload):
    payload, _ = replay_payload
    figure = ScreenEBCFigure(payload, 0., .38)
    try:
        image = figure.draw(.04)
        assert image.shape == (900, 1600, 3)
        head = np.array([.2, .1])
        expected = signed_bearing(head, 0., np.array([[0., -10.]]))[0]
        assert 90 < expected < 92
        for lines in figure.bearing_lines:
            np.testing.assert_allclose(lines[0].get_ydata(), expected)
        figure.frames["valid"][3] = False
        figure.draw(.06)
        assert not figure.heading.get_visible()
        assert len(figure.curves[0][0].get_xdata()) == 0
    finally:
        figure.close()


def test_screen_rejects_old_ccw_payload():
    with pytest.raises(ValueError, match="clockwise EBC replay"):
        ScreenEBCFigure({"meta": {"angle_note": "CCW positive; 0 forward; +90 left"}}, 0., 1.)


def test_replay_holds_whole_position_and_retains_missing_rows(replay_payload, tmp_path):
    payload, _ = replay_payload
    path = Path(payload["meta"]["body_source"])
    pose = pd.read_csv(path, header=[0, 1, 2, 3])
    columns = [("Rigid Body", "hp4", "Position", axis) for axis in "XY"]
    pose.loc[0, columns] = [200., 0.]
    pose.loc[1, columns] = [3., 4.]
    pose.loc[2, columns] = [200., 40.]
    pose.loc[3, columns] = [np.nan, np.nan]
    pose.loc[4, columns] = [200., 80.]
    pose.loc[5, columns] = [6., 7.]
    pose.to_csv(path, index=False)
    raw_before = path.read_bytes()
    updated, qc = ebc_replay.export_session(path.parents[2], 1, tmp_path / "held.json",
                                           fps=50., preview_start=0.)
    frames = updated["frames"]
    assert frames["x_cm"][0] is None and frames["x_cm"][3] is None
    np.testing.assert_allclose(np.array([frames["x_cm"][i] for i in [1, 2, 4, 5]]), [.3, .3, .3, .6])
    np.testing.assert_allclose(np.array([frames["y_cm"][i] for i in [1, 2, 4, 5]]), [.4, .4, .4, .7])
    np.testing.assert_array_equal(frames["frame_id"], np.arange(20))
    np.testing.assert_allclose(frames["time_s"], np.arange(20) * .02)
    np.testing.assert_array_equal(frames["heading_deg"], np.zeros(20))
    assert qc["position_correction"]["body_held"] == 2
    assert qc["position_correction"]["body_initial_outside"] == 1
    assert path.read_bytes() == raw_before
