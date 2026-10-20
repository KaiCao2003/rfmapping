"""Saved sector centers retain their neuronal identity when loaded as rows."""
import RF.profiles as rf_profiles_module

from types import SimpleNamespace

import numpy as np
import pandas as pd

from RF.rflocate import RFResult, save_rf


def test_load_rf_centers_keeps_all_sectors_and_saved_unit_order(monkeypatch):
    centers = np.array([
        [[0, 1, 0], [0, 0, 1]],
        [[0, 0, 0], [0, 0, 0]],
        [[1, 0, 0], [0, 0, 0]],
    ], dtype=np.uint8)
    result = RFResult(centers, centers, [23, 99, 7], {}, "saved-centers")
    monkeypatch.setattr(rf_profiles_module, "load_detected_rf", lambda path: result)
    monkeypatch.setattr(rf_profiles_module, "load_rfmap", lambda path: [SimpleNamespace(
        x_positions=np.array([-60., 0., 60.]), y_positions=np.array([-20., 20.]),
    )])
    loaded = rf_profiles_module.load_rf_centers("recording.rfmap", probe="B")
    expected = pd.DataFrame(
        {"rf_x_deg": [0., 60., -60.], "rf_y_deg": [-20., 20., -20.]},
        index=pd.MultiIndex.from_tuples(
            [("B", 23), ("B", 23), ("B", 7)], names=["probe", "unit_id"],
        ),
    )
    pd.testing.assert_frame_equal(loaded, expected)


def test_load_rf_centers_selects_corrected_sidecar_and_keeps_same_x_centers(tmp_path, monkeypatch):
    source = tmp_path / "recording.rfmap"
    corrected_path = tmp_path / "corrected.npz"
    centers = np.array([[[0, 1, 0], [0, 0, 0], [0, 1, 0]]], dtype=np.uint8)
    old_centers = centers.copy()
    old_centers[0, 2, 1] = 0
    save_rf(RFResult(centers, old_centers, [23], {}, "old-centers"), source.with_suffix(".npz"))
    save_rf(RFResult(centers, centers, [23], {}, "corrected-centers"), corrected_path)
    monkeypatch.setattr(rf_profiles_module, "load_rfmap", lambda path: [SimpleNamespace(
        x_positions=np.array([-60., 0., 60.]), y_positions=np.array([-20., 0., 20.]),
    )])

    loaded = rf_profiles_module.load_rf_centers(source, probe="B", detected_rf_path=corrected_path)
    expected = pd.DataFrame(
        {"rf_x_deg": [0., 0.], "rf_y_deg": [-20., 20.]},
        index=pd.MultiIndex.from_tuples(
            [("B", 23), ("B", 23)], names=["probe", "unit_id"],
        ),
    )
    pd.testing.assert_frame_equal(loaded, expected)
    pd.testing.assert_frame_equal(rf_profiles_module.load_rf_centers(source, probe="B"), expected.iloc[:1])
