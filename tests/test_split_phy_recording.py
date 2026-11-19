from pathlib import Path
import runpy

import numpy as np
import pytest

from preprocessing.spikeinterface.split_kilosort_by_recording import (
    joint_sorting_source, split_kilosort_by_recording, update_split_phy_params,
)


@pytest.mark.parametrize("has_phy_params", [True, False])
def test_split_phy_uses_session_samples_and_preserves_joint_identity(tmp_path, has_phy_params):
    work = tmp_path / "work"
    published = tmp_path / "mouse" / "260918"
    probe = work / "Data/260918_32/ProbeA"
    kilosort = probe / "kilosort"
    kilosort.mkdir(parents=True)
    concat = probe / "concat/traces_cached_seg0.raw"
    concat.parent.mkdir()
    relative_raw = Path("Record Node 102/experiment1/recording1/continuous/OneBox-106.ProbeA/continuous.dat")
    sessions = [3, 2]
    first = np.arange(12, dtype="int16").reshape(6, 2)
    second = first[:4] + 100
    joined = np.vstack([first, second])
    joined.tofile(concat)
    joint_params = f"n_channels_dat = 2\nsample_rate = 30000.0\ndtype = 'int16'\noffset = 0\ndat_path = [\n    {str(concat)!r},\n]\n"
    if has_phy_params:
        (kilosort / "params.py").write_text(joint_params)
    np.save(kilosort / "spike_times.npy", [2, 6, 8, 9])
    np.save(kilosort / "spike_clusters.npy", [17, 17, 23, 17])
    for session, signal in zip(sessions, (first, second)):
        for raw in (work / f"260918_{session}" / relative_raw,
                    published / f"260918_{session}/260918" / relative_raw):
            raw.parent.mkdir(parents=True)
            signal.tofile(raw)

    split_kilosort_by_recording(work, published, "260918", sessions, ["A"], num_channels=2)
    if has_phy_params:
        assert (kilosort / "params.py").read_text() == joint_params
    for session, expected, global_indices in ((3, first, [2]), (2, second, [6, 8, 9])):
        for split, portable in ((probe / f"kilosort_{session}", False),
                                (published / f"260918_{session}/kilosort/ProbeA/kilosort_{session}", True)):
            params_path = split / "params.py"
            local_indices = np.load(split / "spike_times.npy")
            np.testing.assert_array_equal(expected[local_indices], joined[global_indices])
            if not has_phy_params:
                assert not params_path.exists()
                continue
            params = runpy.run_path(str(params_path))
            assert params["offset"] == 0
            source = Path(params["dat_path"][0])
            assert source.is_absolute() == (not portable)
            raw = np.memmap(split / source, mode="r", dtype=params["dtype"]).reshape(-1, params["n_channels_dat"])
            np.testing.assert_array_equal(raw, expected)
            np.testing.assert_array_equal(raw[local_indices], joined[global_indices])
            assert joint_sorting_source(params_path) == joint_sorting_source(kilosort / "params.py")
            before = params_path.read_bytes()
            update_split_phy_params(split, split / source, relative=portable)
            assert params_path.read_bytes() == before
