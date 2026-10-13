import numpy as np
import pandas as pd
import pytest

from Utils import ebc_analysis


@pytest.mark.parametrize("frames", [[0, 1, 2], [0, 1, 2, 3]])
def test_motive_session_matches_pose_rows_to_available_exposures(tmp_path, monkeypatch, frames):
    session = tmp_path / "mouse/day/day_9"
    kilosort = session / "kilosort/ProbeA/kilosort_9"
    kilosort.mkdir(parents=True)
    columns = pd.MultiIndex.from_tuples([
        ("Frame", "", "", ""),
        *[("Body", "hp4", kind, axis) for kind in ("Position", "Rotation") for axis in "XYZ"],
    ])
    rows = [[frame, frame, frame % 2, 0, 0, 0, frame * 30] for frame in frames]
    pose = pd.DataFrame(rows, columns=columns)
    interval = pd.DataFrame({"interval_type": ["baseline"], "start": [0.], "end": [2.]})
    monkeypatch.setattr(ebc_analysis.pd, "read_csv", lambda path, **kw:
                        pose if path.name == "filtered.csv" else interval)
    monkeypatch.setattr(ebc_analysis, "read_formatted_json", lambda path: {"session_info": {}})
    monkeypatch.setattr(ebc_analysis, "get_exposure_timestamps", lambda **kw:
                        (np.array([0., 1., 2.]), 100., {"source": "test"}))
    monkeypatch.setattr(ebc_analysis.np, "load", lambda path, **kw:
                        np.array([100., 101., 102.]) if path.name == "adc_spike_time.npy"
                        else np.array([7, 7, 7]))
    monkeypatch.setattr(ebc_analysis.spatial, "read_good_unit_ids", lambda path: [7])

    data = ebc_analysis.load_motive_session(session)

    np.testing.assert_array_equal(data["times"], [0., 1., 2.])
    np.testing.assert_array_equal(data["hd"], [0., 30., 60.])
    np.testing.assert_array_equal(data["counts"], [[1, 1, 1]])
    assert data["xy"].shape == (3, 2)
    assert len(pose) == len(frames)
