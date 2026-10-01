from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import pytest
import xarray as xr

import decode_head_direction_across_sessions as decoder


def write_session(path, *, fs=20.0, hd_classes=(3, 1, 1), angles=True, silent=()):
    data = path / "data"
    (data / "processed").mkdir(parents=True)
    times = np.arange(int(fs * 4)) / fs + 0.5 / fs
    source = {"frames": list(range(len(times)))}
    if angles:
        source["head_direction_deg"] = np.repeat([45.0, 135.0, 225.0, 315.0], int(fs)).tolist()
    (data / "processed/head_direction.json").write_text(json.dumps({"hp4": source}))
    (data / "session_info.json").write_text(json.dumps({"session_info": {}}))
    ks = path / "kilosort/ProbeA/kilosort_4"
    ks.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [11, 22, 33, 44], "KSLabel": ["good", "good", "good", "mua"]}).to_csv(
        ks / "cluster_KSLabel.tsv", sep="\t", index=False,
    )
    spike_times, clusters = [], []
    for index, unit in enumerate([11, 22, 33]):
        if unit not in silent:
            spike_times.extend(index + np.arange(10) / 10 + 0.025)
            clusters.extend([unit] * 10)
    (data / "probeA").mkdir()
    np.save(data / "probeA/adc_spike_time.npy", np.array(spike_times) + 100.0)
    np.save(ks / "spike_clusters.npy", np.array(clusters))
    if hd_classes is not None:
        tc_path = data / "tuning_curves/ProbeA/tuning_curves.tc"
        tc_path.parent.mkdir(parents=True)
        tc_path.write_text(json.dumps({
            "unit_id": [11, 22, 33, 44], "unit_data": {"hd_class": [*hd_classes, 3]},
            "metadata": {"session": path.name, "alpha": 0.01 if path.name == "train9" else 0.05},
        }))
    return times


def test_pooled_rates_weight_occupancy_seconds_with_different_sample_rates():
    counts = []
    for fs, seconds, spike_times in [(10.0, 2, [0.05, 1.05]), (20.0, 4, [0.05, 0.15, 0.25, 2.05])]:
        support = nap.IntervalSet(start=0, end=seconds)
        times = np.arange(int(seconds * fs)) / fs + 0.5 / fs
        feature = nap.Tsd(
            t=times, d=np.repeat([45.0, 225.0], len(times) // 2), time_support=support,
        )
        spikes = nap.TsGroup({11: nap.Ts(t=spike_times, time_support=support)}, time_support=support)
        result, _ = decoder.session_training_counts(spikes, feature, fs, 4)
        counts.append(result)

    curves, arrays = decoder.pool_training_counts(counts, [10.0, 20.0])

    np.testing.assert_array_equal(arrays["occupancy_seconds"], [3.0, 0.0, 3.0, 0.0])
    np.testing.assert_array_equal(arrays["spike_counts"], [[4.0, 0.0, 2.0, 0.0]])
    np.testing.assert_allclose(curves.values, [[4 / 3, 2 / 3]])
    np.testing.assert_array_equal(curves.head_direction_deg, [45.0, 225.0])
    # A mean of each session's rate would incorrectly yield [1.25, 0.75].
    assert not np.allclose(curves.values, [[1.25, 0.75]])


def test_angle_visited_in_only_one_training_session_survives_pooling():
    counts = []
    for angle in [45.0, 225.0]:
        support = nap.IntervalSet(start=0, end=1)
        feature = nap.Tsd(t=[0.25, 0.75], d=[angle, angle], time_support=support)
        spikes = nap.TsGroup({11: nap.Ts(t=[0.25], time_support=support)}, time_support=support)
        result, _ = decoder.session_training_counts(spikes, feature, 2.0, 4)
        counts.append(result)

    curves, arrays = decoder.pool_training_counts(counts, [2.0, 2.0])

    np.testing.assert_array_equal(curves.values, [[1.0, 1.0]])
    np.testing.assert_array_equal(arrays["visited_angle_bins"], [True, False, True, False])


def test_selection_unions_only_training_classes_and_intersects_good_labels(tmp_path):
    train9, train10 = tmp_path / "train9", tmp_path / "train10"
    write_session(train9, hd_classes=(3, 1, 1))
    write_session(train10, hd_classes=(1, 3, 1))
    write_session(tmp_path / "target5", hd_classes=(1, 1, 3))

    units, selection = decoder.select_training_units([train9, train10], "A", 3)

    np.testing.assert_array_equal(units, [11, 22])
    assert selection["target_classification_used"] is False
    assert [item["source_metadata"]["alpha"] for item in selection["training_sources"]] == [0.01, 0.05]
    assert [Path(item["session_dir"]).name for item in selection["training_sources"]] == ["train9", "train10"]


def test_target_spike_loader_preserves_silent_training_units(tmp_path):
    write_session(tmp_path, hd_classes=None, silent=(22, 33))
    spikes, selection = decoder.load_spikes(
        tmp_path, decoder.find_kilosort(tmp_path, "A"), "A", 100.0,
        nap.IntervalSet(start=0, end=4), units=[11, 22, 33],
    )
    assert list(spikes.keys()) == [11, 22, 33]
    assert len(spikes[11]) == 10
    assert len(spikes[22]) == len(spikes[33]) == 0
    assert selection["mode"] == "explicit"


def test_saved_model_reloads_exact_rates_unit_order_and_angles(tmp_path):
    counts = xr.DataArray(
        [[4.0, 0.0, 8.0, 0.0], [6.0, 0.0, 2.0, 0.0]],
        dims=("unit", "head_direction_deg"),
        coords={"unit": [11, 33], "head_direction_deg": [45.0, 135.0, 225.0, 315.0]},
        attrs={"occupancy": np.array([4.0, 0.0, 8.0, 0.0]),
               "bin_edges": [np.linspace(0, 360, 5)]},
    )
    curves, arrays = decoder.pool_training_counts([counts], [2.0])
    path = tmp_path / "model.npz"
    metadata = {"training_sessions": ["train9", "train10"]}

    decoder.save_model(path, arrays, metadata)
    reloaded = decoder.load_model(path)

    xr.testing.assert_identical(reloaded, curves)
    with np.load(path, allow_pickle=False) as stored:
        for key, values in arrays.items():
            np.testing.assert_array_equal(stored[key], values)
    assert json.loads(path.with_suffix(".metadata.json").read_text())["model_sha256"] == decoder.file_sha256(path)
    support = nap.IntervalSet(start=0.0, end=1.0)
    spikes = nap.TsGroup({
        11: nap.Ts(t=[0.25], time_support=support),
        33: nap.Ts(t=[], time_support=support),
    }, time_support=support)
    decoded, posterior = nap.decode_bayes(curves, spikes, support, 1.0, uniform_prior=True)
    restored, restored_posterior = nap.decode_bayes(reloaded, spikes, support, 1.0, uniform_prior=True)
    np.testing.assert_array_equal(restored.values, decoded.values)
    np.testing.assert_array_equal(restored_posterior.values, posterior.values)
    # Unit 11 has equal rates at both angles. Silent unit 33 still changes the likelihood.
    np.testing.assert_array_equal(decoded.values, [225.0])
    assert posterior.values[0, 1] > 0.9


def test_training_and_target_sessions_must_be_disjoint(tmp_path):
    args = decoder.parse_args([
        "--train-session", str(tmp_path), "--decode-session", str(tmp_path),
        "--model-output", str(tmp_path / "model.npz"),
    ])
    with pytest.raises(ValueError, match="disjoint"):
        decoder.run(args)


def test_target_clock_requires_audited_exact_trailing_count(tmp_path, monkeypatch):
    exposures = write_session(tmp_path, hd_classes=None, angles=False)
    source = tmp_path / "data/processed/head_direction.json"
    source.write_text(json.dumps({"hp4": {"frames": list(range(len(exposures) + 7))}}))
    monkeypatch.setattr(decoder, "get_exposure_timestamps", lambda *a, **kw: (exposures, 100.0, {}))
    with pytest.raises(ValueError, match="explicitly permitted trailing"):
        decoder.load_session_clock(tmp_path, "hp4", 1, training=False)
    clock_path = tmp_path / "audited_camera.npz"
    np.savez(clock_path, exposure_times_s=exposures, adc_time_origin_s=100.0,
             metadata_json=json.dumps({"allowed_trailing_frames": 7, "evidence": "synthetic audit"}))

    loaded = decoder.load_session_clock(tmp_path, "hp4", 1, training=False, clock_override=clock_path)

    assert len(loaded["frames"]) == len(exposures) + 7
    assert not loaded["synced"][-7:].any()
    assert loaded["timing_qc"]["allowed_trailing_frames"] == 7
    assert loaded["timing_qc"]["clock_override_sha256"] == decoder.file_sha256(clock_path)
    np.savez(clock_path, exposure_times_s=exposures, adc_time_origin_s=100.0,
             metadata_json=json.dumps({"allowed_trailing_frames": 6}))
    with pytest.raises(ValueError, match="explicitly permitted trailing"):
        decoder.load_session_clock(tmp_path, "hp4", 1, training=False, clock_override=clock_path)


def test_cross_session_run_never_reads_target_angles_or_classes_or_interval_table(tmp_path, monkeypatch):
    train9, train10, target5, target7 = [tmp_path / name for name in ("train9", "train10", "target5", "target7")]
    clocks = {
        train9 / "data": write_session(train9, hd_classes=(3, 1, 1)),
        train10 / "data": write_session(train10, fs=40.0, hd_classes=(1, 3, 1)),
        target5 / "data": write_session(target5, hd_classes=None, angles=False, silent=(22, 33)),
        target7 / "data": write_session(target7, hd_classes=None, angles=False, silent=(11, 33)),
    }
    monkeypatch.setattr(decoder, "get_exposure_timestamps", lambda info, data_dir, **kw: (
        clocks[data_dir], 100.0, {"source": "test_clock"},
    ))
    args = decoder.parse_args([
        "--train-session", str(train9), str(train10), "--decode-session", str(target5), str(target7),
        "--bins", "4", "--model-output", str(tmp_path / "model.npz"),
    ])
    outputs = decoder.run(args)

    assert len(outputs) == 2
    for output, silent_unit in zip(outputs, [22, 11]):
        payload = json.loads(output.read_text())["internal_direction"]
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        assert payload["frames"] == list(range(80))
        assert metadata["unit_ids"] == [11, 22]
        assert metadata["zero_spike_unit_ids"] == [silent_unit]
        assert metadata["target_classification_used"] is False
        assert metadata["target_refit"] is False
        assert metadata["target_tracking_use"] == "frame IDs only"
        assert metadata["model_sha256"] == decoder.file_sha256(args.model_output)
        assert metadata["training_session_dirs"] == [str(train9), str(train10)]
        assert metadata["valid_frame_count"] > 0
    with pytest.raises(FileExistsError):
        decoder.run(args)
