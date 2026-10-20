"""Persist the exact analyzed EBC samples and full camera clock for playback."""

import numpy as np
import pandas as pd

from EBB import ebc_geometry
from EBB import spatial_cell_analysis
from EBB.spatial_cell_results import save_position_correction


def test_saved_frame_geometry_preserves_samples_and_full_clock_without_calculation(tmp_path, monkeypatch):
    exposures = np.array([.001, .023, .051, .077, .11])
    pose = pd.DataFrame({"frame": [1, 3], "hd_deg": [15., 75.]})
    pose.attrs["exposure_times"] = exposures
    times = exposures[pose.frame.to_numpy()]
    theta = np.arange(3., 360., 6., dtype=np.float32)
    distances = np.arange(120, dtype=np.float32).reshape(2, 60) + .123
    distances[0, 5] = np.nan
    maps = dict(raw_positions_px=np.array([[10., 20.], [200., 40.]]),
                corrected_positions_px=np.array([[10., 20.], [10., 20.]]),
                position_held=np.array([False, True]), initial_outside=np.array([False, False]),
                ebc_hd_deg=np.array([16., 74.]), theta_grid=np.broadcast_to(theta, distances.shape),
                theta_d_cm=distances)

    def unexpected_calculation(*args, **kwargs):
        raise AssertionError("Saving already analyzed rays must not calculate geometry again.")

    monkeypatch.setattr(ebc_geometry, "rectangle_distances_cm", unexpected_calculation)
    monkeypatch.setattr(spatial_cell_analysis, "prepare_session_maps", unexpected_calculation)

    save_position_correction(tmp_path, pose, times, maps)

    with np.load(tmp_path / "position_correction.npz") as saved:
        np.testing.assert_array_equal(saved["theta_d_cm"], distances)
        np.testing.assert_array_equal(saved["theta_deg"], theta)
        assert saved["theta_d_cm"].dtype == distances.dtype
        assert saved["theta_deg"].dtype == theta.dtype
        np.testing.assert_array_equal(saved["exposure_times"], exposures)
        np.testing.assert_array_equal(saved["frame_ids"], [1, 3])
        np.testing.assert_array_equal(saved["frame_times"], times)
        np.testing.assert_array_equal(saved["exposure_times"][saved["frame_ids"]], saved["frame_times"])
        np.testing.assert_array_equal(saved["corrected_xy_px"], maps["corrected_positions_px"])
        np.testing.assert_array_equal(saved["raw_xy_px"], maps["raw_positions_px"])
        np.testing.assert_array_equal(saved["position_held"], maps["position_held"])
        np.testing.assert_array_equal(saved["hd_deg"], pose.hd_deg)
        np.testing.assert_array_equal(saved["ebc_hd_deg"], maps["ebc_hd_deg"])
