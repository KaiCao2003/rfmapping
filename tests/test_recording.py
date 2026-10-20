import json

import numpy as np
import pandas as pd
import pytest

from Utils.recording import gen_recording_interval_table


@pytest.fixture
def session(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    session_info = {
        "base_path": str(tmp_path),
        "record_nodes": "node",
        "experiment_id": "experiment",
        "recording_name": "recording",
        "continuous_ADC_folder": "ADC",
        "ADC_input_channel": 1,
    }
    (data_dir / "session_info.json").write_text(json.dumps({"session_info": session_info}))
    return tmp_path


@pytest.mark.parametrize("indices", [[[10, 20]], [[10, 20], [30, 40]], []])
@pytest.mark.parametrize("legacy", [False, True])
def test_cached_intervals_read_arrays_and_existing_strings(session, indices, legacy):
    indices = np.asarray(indices, dtype=int).reshape(-1, 2)
    sync_data = {
        "recording_interval": (indices / 100).tolist(),
        "recording_interval_indices": str(indices) if legacy else indices.tolist(),
    }
    (session / "data/sync_data.json").write_text(json.dumps(sync_data))

    success, message = gen_recording_interval_table(str(session))

    if len(indices) == 1:
        assert success
        table = pd.read_csv(session / "data/interval_table.csv")
        np.testing.assert_array_equal(table[["start_frame", "end_frame"]], indices)
        np.testing.assert_allclose(table[["start", "end"]], indices / 100)
    else:
        assert not success
        assert ("non-empty" if len(indices) == 0 else "Multiple recording intervals") in message
        assert not (session / "data/interval_table.csv").exists()


def test_generated_intervals_are_saved_as_json_arrays_and_reload(session, monkeypatch):
    adc_dir = session / "node/experiment/recording/continuous/ADC"
    adc_dir.mkdir(parents=True)
    np.save(adc_dir / "timestamps.npy", 100.0 + np.arange(10) / 10)
    np.array([0, 0, 0, 20000, 20000, 20000, 20000, 0, 0, 0], dtype=np.int16).tofile(
        adc_dir / "continuous.dat",
    )

    class EndpointTimestamps(np.ndarray):
        def __sub__(self, other):
            # Timestamp arithmetic must scale with interval endpoints, not ADC samples.
            assert self.shape == (1, 2)
            return np.asarray(self) - other

    load = np.load

    def load_timestamps(*args, **kwargs):
        return load(*args, **kwargs).view(EndpointTimestamps)

    monkeypatch.setattr("Utils.recording.np.load", load_timestamps)

    success, _ = gen_recording_interval_table(
        str(session), camera_input_channel=0, _recording_interval_min_len=2,
    )
    assert success
    sync_data = json.loads((session / "data/sync_data.json").read_text())
    assert sync_data["recording_interval_indices"] == [[3, 7]]
    np.testing.assert_allclose(sync_data["recording_interval"], [[0.3, 0.7]])

    first_table = pd.read_csv(session / "data/interval_table.csv")
    success, _ = gen_recording_interval_table(str(session), is_overwrite_interval_table=True)
    assert success
    pd.testing.assert_frame_equal(first_table, pd.read_csv(session / "data/interval_table.csv"))
