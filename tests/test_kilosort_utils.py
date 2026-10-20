from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pynapple")
from Utils import kilosort_utils


def test_spike_grouping_preserves_public_lists_and_sample_order():
    clusters = np.array([8, 3, 8, 3, 8])
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    requested = [8, 42, 3, 8]
    expected = [[8, [0, 2, 4], [0.1, 0.3, 0.5]], [42, [], []],
                [3, [1, 3], [0.2, 0.4]], [8, [0, 2, 4], [0.1, 0.3, 0.5]]]

    assert kilosort_utils.locate_spikes(requested, clusters, times) == expected
    grouped = kilosort_utils._locate_spike_arrays(requested, clusters, times)
    for (unit, indices, values), (expected_unit, expected_indices, expected_values) in zip(grouped, expected):
        assert unit == expected_unit
        assert isinstance(indices, np.ndarray) and isinstance(values, np.ndarray)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(values, expected_values)


def test_tuning_curves_use_returned_synced_times_without_reload(tmp_path, monkeypatch):
    kilosort_dir = tmp_path / "kilosort"
    kilosort_dir.mkdir()
    (kilosort_dir / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n3\tgood\n8\tgood\n")
    np.save(kilosort_dir / "spike_clusters.npy", np.array([8, 3, 8, 3]))
    synced_times = np.array([0.1, 0.2, 0.3, 0.4])
    calls = []

    def sync(*args, **kwargs):
        calls.append(kwargs["output_path"])
        # Deliberately do not save: gen_tuning_curves must use this returned array.
        return synced_times

    monkeypatch.setattr(kilosort_utils, "sync_spike_time_to_adc_time", sync)
    curves = object()
    monkeypatch.setattr(kilosort_utils.nap, "compute_tuning_curves", lambda **kwargs: curves)
    session_info = dict(base_path=str(tmp_path), record_nodes="node", experiment_id="experiment",
                        recording_name="recording", continuous_probe_A_folder="ProbeA")
    tsgroup, _, actual_curves, labels = kilosort_utils.gen_tuning_curves(
        tmp_path, kilosort_dir, "A", session_info, [[0, 1]], None, "kilosort_info", 60,
    )

    assert calls == [tmp_path / "data/ProbeA/adc_spike_time.npy"]
    assert actual_curves is curves
    assert labels == [["good", [3, 8]]]
    np.testing.assert_array_equal(tsgroup[3].index, [0.2, 0.4])
    np.testing.assert_array_equal(tsgroup[8].index, [0.1, 0.3])


@pytest.fixture
def summary_args(tmp_path):
    kilosort_dir = tmp_path / "kilosort/ProbeA/kilosort_1"
    kilosort_dir.mkdir(parents=True)
    (kilosort_dir / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n3\tgood\n8\tnoise\n")
    (tmp_path / "data").mkdir()
    return dict(base_dir=tmp_path, kilosort_dir=tmp_path / "kilosort", probe_name="A",
                session_info={}, interval_pairs=[], HD_tsd=None, kilosort_info_filename="info",
                num_of_bins_in_hd=60, num_shuffle=100, is_overwrite_shuffle_data=False)


def summary_frame():
    return pd.DataFrame(dict(cluster_id=[3, 8, 11], rayleigh_significant=[True, False, False],
                             shuffle_significant=[False, True, False], R=[0.6, 0.7, 0.3]))


@pytest.mark.parametrize("custom_path", [False, True])
def test_cached_summary_skips_tuning_but_keeps_requested_metadata(summary_args, monkeypatch, custom_path):
    path = Path(summary_args["base_dir"]) / "data/tc_summary_ProbeA.csv"
    if custom_path:
        path = path.with_name("custom.csv")
        summary_args["save_path"] = path
    frame = summary_frame()
    frame.to_csv(path, index=False)

    def unexpected(**kwargs):
        pytest.fail("Cached summary does not need spike/tuning calculations")

    monkeypatch.setattr(kilosort_utils, "gen_tuning_curves", unexpected)
    monkeypatch.setattr(kilosort_utils, "shuffle_neuro_data", unexpected)
    result = kilosort_utils.get_neuro_summary(
        **summary_args, is_return_cluster_KSLabel=True,
        is_return_hd_cells=True, is_return_classical_hd_cells=True,
    )

    pd.testing.assert_frame_equal(result["hd_summary"], frame)
    assert result["hd_cells"]["cluster_id"].tolist() == [3, 8]
    assert result["classical_hd_cells"]["cluster_id"].tolist() == [8]
    assert result["cluster_KSLabel"] == [["good", [3]], ["noise", [8]]]


@pytest.mark.parametrize("reason", ["missing", "overwrite", "tsgroup", "time_support", "hd_tcs"])
def test_summary_computes_when_cache_or_requested_outputs_require_it(summary_args, monkeypatch, reason):
    frame = summary_frame()
    if reason != "missing":
        frame.to_csv(Path(summary_args["base_dir"]) / "data/tc_summary_ProbeA.csv", index=False)
    if reason == "overwrite":
        summary_args["is_overwrite_shuffle_data"] = True
    elif reason in {"tsgroup", "time_support", "hd_tcs"}:
        summary_args[f"is_return_{reason}"] = True
    calls = []
    outputs = [object(), object(), object(), [["good", [3]]]]

    def generate(**kwargs):
        calls.append("generate")
        return outputs

    def shuffle(**kwargs):
        calls.append("shuffle")
        assert kwargs["tsgroup"] is outputs[0]
        assert kwargs["time_support"] is outputs[1]
        assert kwargs["hd_tuning_curves"] is outputs[2]
        return frame

    monkeypatch.setattr(kilosort_utils, "gen_tuning_curves", generate)
    monkeypatch.setattr(kilosort_utils, "shuffle_neuro_data", shuffle)
    result = kilosort_utils.get_neuro_summary(**summary_args)

    assert calls == ["generate", "shuffle"]
    assert result["hd_summary"] is frame
    if reason in {"tsgroup", "time_support", "hd_tcs"}:
        key = "hd_tuning_curves" if reason == "hd_tcs" else reason
        assert result[key] is outputs[["tsgroup", "time_support", "hd_tcs"].index(reason)]
