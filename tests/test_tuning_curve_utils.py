from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from Utils import kilosort_utils
from Utils import tuning_curve_utils


class _FakeTs:
    def __init__(self, t, time_units: str) -> None:
        assert time_units == "s"
        self.index = pd.Index(np.asarray(t, dtype=float))

    def restrict(self, _time_support):
        return self

    def value_from(self, _feature):
        return SimpleNamespace(values=np.full(len(self.index), 3.0))


class _FakeTsGroup(dict):
    pass


@pytest.mark.parametrize("source", ["npy", "sync", "sync_raw"])
def test_exposure_reads_saved_times_without_scanning_adc(tmp_path, source):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    relative_times = np.asarray([0.25, 0.5, 0.75, 2.0])
    origin = 128.0
    session_info = {}
    if source != "sync_raw":
        adc_dir = tmp_path / "node" / "experiment" / "recording" / "continuous" / "ADC"
        adc_dir.mkdir(parents=True)
        np.save(adc_dir / "timestamps.npy", np.asarray([origin, origin + 0.01]))
        session_info = {
            "base_path": str(tmp_path),
            "record_nodes": "node",
            "experiment_id": "experiment",
            "recording_name": "recording",
            "continuous_ADC_folder": "ADC",
        }

    if source == "npy":
        source_path = data_dir / "camera_frame_times.npy"
        np.save(source_path, relative_times + origin)
        (data_dir / "sync_data.json").write_text("unused JSON", encoding="utf-8")
    else:
        source_path = data_dir / "sync_data.json"
        sync_data = {
            "exposure_sampling_number_list_mid": relative_times.tolist(),
            "exposure_sampling_number_list_pairs": "[[0.1 0.4] ... [1.9 2.1]]",
        }
        if source == "sync_raw":
            sync_data["exposure_sampling_number_list_mid_raw"] = (
                relative_times + origin
            ).tolist()
        source_path.write_text(json.dumps(sync_data), encoding="utf-8")

    timestamps, adc_origin, qc = tuning_curve_utils.get_exposure_timestamps(
        session_info=session_info,
        data_dir=data_dir,
    )

    np.testing.assert_array_equal(timestamps, relative_times)
    assert adc_origin == origin
    assert qc["source_path"] == str(source_path)
    assert qc["ttl_pulse_count"] == 4
    assert qc["first_exposure_s"] == 0.25
    assert qc["last_exposure_s"] == 2.0
    assert qc["median_period_s"] == 0.25
    assert qc["timestamp_reference"] == (
        "saved_camera_frame_times" if source == "npy" else "saved_exposure_midpoint"
    )
    if source == "npy":
        np.testing.assert_array_equal(np.load(source_path), relative_times + origin)


@pytest.mark.parametrize(
    "sync_data, error",
    [
        (None, FileNotFoundError),
        ({"recording_interval": [[0, 10]]}, KeyError),
        ({"exposure_sampling_number_list_mid": []}, ValueError),
    ],
)
def test_missing_exposure_data_does_not_regenerate(tmp_path, sync_data, error):
    if sync_data is not None:
        (tmp_path / "sync_data.json").write_text(json.dumps(sync_data), encoding="utf-8")
    with pytest.raises(error):
        tuning_curve_utils.get_exposure_timestamps(session_info={}, data_dir=tmp_path)


@pytest.mark.parametrize("include_empty_unit", [False, True])
def test_tuning_curve_writes_exact_columnar_contract(
    tmp_path, monkeypatch, include_empty_unit,
) -> None:
    base_dir = tmp_path / "260630_1"
    kilosort_dir = base_dir / "kilosort4" / "ProbeA"
    kilosort_dir.mkdir(parents=True)
    unit_ids = [218, 22, 11] if include_empty_unit else [22, 11]
    pd.DataFrame({
        "cluster_id": unit_ids + [99],
        "KSLabel": ["good"] * len(unit_ids) + ["mua"],
    }).to_csv(
        kilosort_dir / "cluster_KSLabel.tsv", sep="\t", index=False
    )
    np.save(kilosort_dir / "spike_clusters.npy", np.asarray(unit_ids + [99], dtype=np.int64))
    probe_dir = base_dir / "data" / "probeA"
    probe_dir.mkdir(parents=True)
    saved_spike_times = 100.0 + np.arange(1, len(unit_ids) + 2) * 0.125
    np.save(probe_dir / "adc_spike_time.npy", saved_spike_times)

    occupancy_samples = np.full(180, 100, dtype=int)
    occupancy_samples[-1] = 0
    counts_by_unit = np.vstack((np.ones(180, dtype=int), np.full(180, 2, dtype=int)))
    if include_empty_unit:
        counts_by_unit = np.vstack((np.zeros(180, dtype=int), counts_by_unit))
    counts_by_unit[:, -1] = 0
    counts = xr.DataArray(
        counts_by_unit.T,
        dims=("angle", "unit"),
        coords={
            "angle": np.arange(1.0, 360.0, 2.0),
            "unit": np.asarray(unit_ids),
        },
        attrs={
            "bin_edges": [np.linspace(0.0, 360.0, 181)],
            "occupancy": occupancy_samples,
            "fs": 100.0,
        },
    )
    time_support = SimpleNamespace(start=np.asarray([0.0]), end=np.asarray([1.0]))

    monkeypatch.setattr(tuning_curve_utils.nap, "Ts", _FakeTs)
    monkeypatch.setattr(tuning_curve_utils.nap, "TsGroup", _FakeTsGroup)
    def compute_tuning_curves(**kwargs):
        spikes = kwargs["data"]
        assert set(spikes) == set(unit_ids)
        for index, unit_id in enumerate(unit_ids):
            np.testing.assert_array_equal(spikes[unit_id].index, [(index + 1) * 0.125])
        return counts

    monkeypatch.setattr(tuning_curve_utils.nap, "compute_tuning_curves", compute_tuning_curves)
    monkeypatch.setattr(
        kilosort_utils,
        "convert_time_list_to_nap_tsd",
        lambda *_args, **_kwargs: time_support,
    )
    monkeypatch.setattr(
        kilosort_utils,
        "_compress_times_to_epoch_clock",
        lambda times, *_args: np.asarray(times, dtype=float),
    )
    monkeypatch.setattr(
        kilosort_utils,
        "_flatten_feature_segments",
        lambda *_args: (
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 2.0]),
            np.asarray([0]),
            np.asarray([2]),
        ),
    )
    monkeypatch.setattr(kilosort_utils, "_compute_shuffle_r_numba", None)
    shuffled_units = []

    def compute_shuffle(*args):
        shuffled_units.append(args)
        return np.asarray([0.0])

    monkeypatch.setattr(kilosort_utils, "_compute_shuffle_r_numpy", compute_shuffle)

    save_path = tmp_path / "tuning_curves.json"
    result = tuning_curve_utils.tuning_curve(
        base_dir=base_dir,
        kilosort_dir=kilosort_dir,
        probe_name="A",
        interval_pairs=np.asarray([[0.0, 1.0]]),
        HD_tsd=object(),
        adc_time_origin_s=100.0,
        num_of_bins_in_hd=180,
        num_shuffle=1,
        shuffle_seed=42,
        is_save=True,
        save_path=save_path,
        metadata={"epoch": "arena"},
    )

    assert tuple(result) == (
        "metadata",
        "angle_bin_edges_deg",
        "occupancy_samples",
        "occupancy_time_s",
        "unit_id",
        "spike_counts",
        "firing_rate_hz",
        "unit_data",
    )
    assert result["unit_id"] == unit_ids
    assert result["spike_counts"] == counts_by_unit.tolist()
    assert result["firing_rate_hz"][-2][0] == 1.0
    assert result["firing_rate_hz"][-1][0] == 2.0
    assert result["firing_rate_hz"][0][-1] is None
    assert tuple(result["unit_data"]) == (
        "hd_class",
        "rate_mvl",
        "spike_angle_mrl",
        "rayleigh_score",
        "rayleigh_p",
        "rayleigh_significant",
        "shuffle_p",
        "shuffle_significant",
    )
    assert all(len(column) == len(unit_ids) for column in result["unit_data"].values())
    assert len(shuffled_units) == 2
    assert all(value is not None for value in result["unit_data"]["rate_mvl"][-2:])
    if include_empty_unit:
        assert all(column[0] is None for column in result["unit_data"].values())
    assert "schema_version" not in result
    assert "units" not in result
    assert json.loads(save_path.read_text(encoding="utf-8")) == result
    np.testing.assert_array_equal(np.load(probe_dir / "adc_spike_time.npy"), saved_spike_times)


def test_tuning_curve_skips_unit_without_spikes_in_selected_epoch(
    tmp_path,
    monkeypatch,
) -> None:
    class _EmptyRestrictedTs(_FakeTs):
        def restrict(self, _time_support):
            return _FakeTs([], time_units="s")

    base_dir = tmp_path / "260827_10"
    kilosort_dir = base_dir / "kilosort" / "ProbeA" / "kilosort_10"
    kilosort_dir.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [218], "KSLabel": ["good"]}).to_csv(
        kilosort_dir / "cluster_KSLabel.tsv",
        sep="\t",
        index=False,
    )
    np.save(
        kilosort_dir / "spike_clusters.npy",
        np.asarray([218, 218], dtype=np.int64),
    )

    probe_dir = base_dir / "data" / "probeA"
    probe_dir.mkdir(parents=True)
    np.save(probe_dir / "adc_spike_time.npy", np.asarray([100.2, 100.3]))

    counts = xr.DataArray(
        np.zeros((180, 1), dtype=int),
        dims=("angle", "unit"),
        coords={
            "angle": np.arange(1.0, 360.0, 2.0),
            "unit": np.asarray([218]),
        },
        attrs={
            "bin_edges": [np.linspace(0.0, 360.0, 181)],
            "occupancy": np.full(180, 100, dtype=int),
            "fs": 100.0,
        },
    )
    time_support = SimpleNamespace(start=np.asarray([0.0]), end=np.asarray([1.0]))

    monkeypatch.setattr(tuning_curve_utils.nap, "Ts", _EmptyRestrictedTs)
    monkeypatch.setattr(tuning_curve_utils.nap, "TsGroup", _FakeTsGroup)
    monkeypatch.setattr(
        tuning_curve_utils.nap,
        "compute_tuning_curves",
        lambda **_kwargs: counts,
    )
    monkeypatch.setattr(
        kilosort_utils,
        "convert_time_list_to_nap_tsd",
        lambda *_args, **_kwargs: time_support,
    )
    monkeypatch.setattr(
        kilosort_utils,
        "_compress_times_to_epoch_clock",
        lambda times, *_args: np.asarray(times, dtype=float),
    )
    monkeypatch.setattr(
        kilosort_utils,
        "_flatten_feature_segments",
        lambda *_args: (
            np.asarray([0.0, 1.0]),
            np.asarray([0.0, 2.0]),
            np.asarray([0]),
            np.asarray([2]),
        ),
    )
    monkeypatch.setattr(kilosort_utils, "_compute_shuffle_r_numba", None)

    def fail_if_shuffle_runs(*_args):
        raise AssertionError("Shuffle must not run for an empty epoch unit.")

    monkeypatch.setattr(
        kilosort_utils,
        "_compute_shuffle_r_numpy",
        fail_if_shuffle_runs,
    )

    result = tuning_curve_utils.tuning_curve(
        base_dir=base_dir,
        kilosort_dir=kilosort_dir,
        probe_name="A",
        interval_pairs=np.asarray([[0.0, 1.0]]),
        HD_tsd=object(),
        adc_time_origin_s=100.0,
        num_of_bins_in_hd=180,
        num_shuffle=1000,
        shuffle_seed=0,
    )

    assert result["unit_id"] == [218]
    assert result["spike_counts"] == [np.zeros(180, dtype=int).tolist()]
    for values in result["unit_data"].values():
        assert values == [None]
