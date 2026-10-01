import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_recording_import_does_not_load_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import Utils.recording; import sys; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_torch_exposure_helper_still_runs():
    torch = pytest.importorskip("torch")
    from Utils.recording import detect_exposure_time_torch

    result = detect_exposure_time_torch(
        torch.device("cpu"), np.arange(5), np.array([0, 20000, 20000, 0, 20000]),
    )
    np.testing.assert_array_equal(result.numpy(), [[1, 2], [4, 4]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("column_indices", [False, True])
@pytest.mark.parametrize("relative", [False, True])
def test_adc_spike_time_preserves_clock_shape_and_dtype(
        tmp_path, monkeypatch, capsys, dtype, column_indices, relative):
    pytest.importorskip("pynapple")
    from Utils.kilosort_utils import gen_adc_spike_time

    timestamps = np.arange(100, 110, dtype=dtype)
    spike_samples = np.array([8, 0, 3, 3])
    if column_indices:
        spike_samples = spike_samples[:, None]
    timestamp_dir = tmp_path / "node/experiment/recording/continuous/ProbeA"
    timestamp_dir.mkdir(parents=True)
    np.save(timestamp_dir / "timestamps.npy", timestamps)
    kilosort_dir = tmp_path / "kilosort/ProbeA/kilosort_1"
    kilosort_dir.mkdir(parents=True)
    np.save(kilosort_dir / "spike_times.npy", spike_samples)

    class SpikeTimestamps(np.ndarray):
        def __sub__(self, other):
            # Only selected spike timestamps should be copied for origin subtraction.
            assert self.shape == spike_samples.shape
            return np.asarray(self) - other

    load = np.load

    def load_timestamps(path, **kwargs):
        array = load(path, **kwargs)
        return array.view(SpikeTimestamps) if Path(path).name == "timestamps.npy" else array

    monkeypatch.setattr("Utils.kilosort_utils.np.load", load_timestamps)
    session_info = dict(base_path=str(tmp_path), record_nodes="node", experiment_id="experiment",
                        recording_name="recording", continuous_probe_A_folder="ProbeA")
    success, output = gen_adc_spike_time(
        session_info, "a", str(tmp_path), 1, is_convert_to_zero=relative,
    )
    actual = load(output)
    expected = (timestamps - timestamps[0] if relative else timestamps)[spike_samples]

    assert success
    assert output == tmp_path / "data/probeA/adc_spike_time.npy"
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    log = capsys.readouterr().out
    assert f"ProbeA duration: {timestamps[-1] - timestamps[0] if relative else timestamps[-1]} " in log
    assert "11111111" not in log
