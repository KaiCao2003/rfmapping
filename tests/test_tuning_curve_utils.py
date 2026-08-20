from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
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


def test_tuning_curve_writes_exact_columnar_contract(tmp_path, monkeypatch) -> None:
    base_dir = tmp_path / "260630_1"
    kilosort_dir = base_dir / "kilosort4" / "ProbeA"
    kilosort_dir.mkdir(parents=True)
    pd.DataFrame({"cluster_id": [22, 11], "KSLabel": ["good", "good"]}).to_csv(
        kilosort_dir / "cluster_KSLabel.tsv", sep="\t", index=False
    )
    np.save(kilosort_dir / "spike_times.npy", np.asarray([1, 2], dtype=np.int64))
    np.save(kilosort_dir / "spike_clusters.npy", np.asarray([22, 11], dtype=np.int64))

    continuous_dir = base_dir / "node" / "experiment" / "recording" / "continuous"
    probe_dir = continuous_dir / "ProbeA"
    probe_dir.mkdir(parents=True)
    np.save(probe_dir / "timestamps.npy", np.asarray([100.0, 100.1, 100.2, 100.3]))
    session_info = {
        "base_path": str(base_dir),
        "record_nodes": "node",
        "experiment_id": "experiment",
        "recording_name": "recording",
        "continuous_probe_A_folder": "ProbeA",
    }

    occupancy_samples = np.full(180, 100, dtype=int)
    occupancy_samples[-1] = 0
    counts_by_unit = np.vstack((np.ones(180, dtype=int), np.full(180, 2, dtype=int)))
    counts_by_unit[:, -1] = 0
    counts = xr.DataArray(
        counts_by_unit.T,
        dims=("angle", "unit"),
        coords={
            "angle": np.arange(1.0, 360.0, 2.0),
            "unit": np.asarray([22, 11]),
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
    monkeypatch.setattr(
        kilosort_utils,
        "_compute_shuffle_r_numpy",
        lambda *_args: np.asarray([0.0]),
    )

    save_path = base_dir / "data" / "tuning_curves" / "ProbeA" / "tuning_curves.tc"
    result = tuning_curve_utils.tuning_curve(
        base_dir=base_dir,
        kilosort_dir=kilosort_dir,
        probe_name="A",
        session_info=session_info,
        interval_pairs=np.asarray([[0.0, 1.0]]),
        HD_tsd=object(),
        adc_time_origin_s=100.0,
        num_of_bins_in_hd=180,
        num_shuffle=1,
        shuffle_seed=42,
        is_save=True,
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
    assert result["unit_id"] == [22, 11]
    assert result["spike_counts"] == counts_by_unit.tolist()
    assert result["firing_rate_hz"][0][0] == 1.0
    assert result["firing_rate_hz"][1][0] == 2.0
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
    assert all(len(column) == 2 for column in result["unit_data"].values())
    assert "schema_version" not in result
    assert "units" not in result
    assert json.loads(save_path.read_text(encoding="utf-8")) == result
