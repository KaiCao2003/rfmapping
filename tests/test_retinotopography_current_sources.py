import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import plot_retinotopography_presentation as presentation


def test_refresh_uses_canonical_centers_and_keeps_hd_cohort(tmp_path, monkeypatch):
    source = tmp_path / "prepared"
    source.mkdir()
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    hd = tmp_path / "tuning_curves.tc"
    hd.write_text("saved HD source")
    rf = tmp_path / "regular.rfmap"
    rf.write_text("unchanged response map")
    detected = rf.with_suffix(".npz")
    detected.write_text("current canonical detection")
    units = pd.DataFrame(dict(mouse=["m1"] * 2, date=["260101"] * 2, probe=["A"] * 2,
                              unit_id=[7, 9], shank=[0, 0], depth_um=[0., 25.],
                              hd_preferred_deg=[20., 80.], rf_detected_2d=[True, False]))
    units.to_csv(source / "adn_units.csv", index=False)
    old_hd = presentation.prepare_depth_angles(units, pd.DataFrame(), kind="hd")
    old_hd.to_csv(source / "hd_depth_points.csv", index=False)
    old_hd_bytes = (source / "hd_depth_points.csv").read_bytes()
    pd.DataFrame([dict(mouse="m1", date="260101", probe="A", rf_session=3,
                       rf_file=str(rf))]).to_csv(tmp_path / "sources.csv", index=False)
    verification = dict(normalized=False, depth_units="um", selection_sha256=presentation.sha(selection),
                        hd_source_sha256={str(hd): presentation.sha(hd)},
                        center_files={"historical_missing_centers.npz": "old hash"})
    (source / "verification.json").write_text(json.dumps(verification))
    result = SimpleNamespace(unit_ids=np.array([7, 9]), mask_2d=np.array([[[1, 0, 1]], [[0, 1, 0]]]),
                             manifest=dict(center_algorithm="component-response-weighted-medoid-v2",
                                           detector_algorithm="pooled-spatial-z-v3-zero-filled",
                                           parameters=dict(exclude_zero_bins=False, alternative="greater",
                                                           cluster_forming_z=1.8, drop_bins=2)))
    monkeypatch.setattr(presentation, "load_rf", lambda path: result if path == detected else pytest.fail(str(path)))

    def load_centers(path, *, probe):
        assert path == str(rf) and probe == "A"
        index = pd.MultiIndex.from_tuples([("A", 7), ("A", 7), ("A", 9)], names=["probe", "unit_id"])
        return pd.DataFrame(dict(rf_x_deg=[-30., 30., 60.], rf_y_deg=[0., 0., 12.]), index=index)

    monkeypatch.setattr(presentation, "load_rf_centers", load_centers)
    output = presentation.refresh_canonical_rf_points(source, tmp_path / "sources.csv", selection)
    assert output["center_files"] == {str(detected): presentation.sha(detected)}
    assert (output["rf_centers"], output["rf_cells"]) == (3, 2)
    points = pd.read_csv(source / "rf_depth_points.csv")
    assert points.unit_id.tolist() == [7, 7, 9]
    assert points.depth_um.tolist() == [0., 0., 25.]
    assert (source / "hd_depth_points.csv").read_bytes() == old_hd_bytes
    assert pd.read_csv(source / "adn_units.csv").rf_detected_2d.all()
