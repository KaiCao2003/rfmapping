import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import ebc_tuning_video as tuning


@pytest.mark.parametrize("wall", ["old", "new"])
def test_peak_ray_uses_clockwise_hd_and_counterclockwise_ego(wall):
    data = dict(session=tuning.Path("260921_11"), phase="baseline", times=np.array([10.]),
                xy=np.array([[20.5, 20.5]]), hd=np.array([90.]),
                bounds_px=tuning.WALL_CONFIGS[wall], arena_size_cm=41., unit_id=2,
                tuning_angles_deg=np.array([0., 90., 180., 270.]),
                tuning_rate=np.array([1., 4., 2., 1.]))
    overlay = tuning.TuningOverlay(data, 1280, 1024, 50.)
    left, right, top, bottom = tuning.WALL_CONFIGS[wall]
    # Heading east plus a 90-degree leftward ego preference points north.
    np.testing.assert_allclose(overlay.endpoints[0, 0], [(left + right) / 2, top])
    np.testing.assert_allclose(overlay.distances[0, 0], 20.5)
    canvas = overlay.draw(Image.new("RGB", (1280, 1024), "white"), 500, 0)
    assert canvas.size == (1650, 1024)


@pytest.mark.parametrize("clock_frames", [2, 3])
def test_loader_uses_json_hd_without_any_csv_direction_column(tmp_path, monkeypatch, clock_frames):
    session = tmp_path / "260921_11"
    directory = session / "data"
    (directory / "processed").mkdir(parents=True)
    (directory / "tuning_curves/ProbeA").mkdir(parents=True)
    (session / "kilosort/ProbeA/kilosort_11").mkdir(parents=True)
    (directory / "processed/head_direction.json").write_text(
        json.dumps({"hp4": {"frames": [0, 2], "hd": [90., 180.]}}))
    (directory / "session_info.json").write_text(json.dumps({"session_info": {}}))
    (directory / "tuning_curves/ProbeA/tuning_curves.tc").write_text(
        json.dumps({"metadata": {"ttl_qc": {"camera_input_channel": 6}}}))
    pd.DataFrame({"frame": [0, 1, 2], "front_x": [450., 460., 470.],
                  "front_y": [500., 510., 520.]}).to_csv(session / "260921.csv", index=False)
    pd.DataFrame({"interval_type": ["baseline"], "start": [0.], "end": [1.]}).to_csv(
        directory / "interval_table.csv", index=False)
    seen = []

    def exposures(info, path, **settings):
        seen.append(settings)
        return np.array([0., .02, .04])[:clock_frames], 12., {}

    monkeypatch.setattr(tuning, "get_exposure_timestamps", exposures)
    data = tuning.load_json_pose(session, "A", "baseline", "new")
    np.testing.assert_array_equal(data["hd"], [90., 180.] if clock_frames == 3 else [90.])
    np.testing.assert_array_equal(data["frame_ids"], [0, 2] if clock_frames == 3 else [0])
    assert data["dropped_unsynchronized_pose_frames"] == 3 - clock_frames
    assert seen == [dict(camera_input_channel=6, camera_ttl_active_high=False)]
