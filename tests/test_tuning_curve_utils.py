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


@pytest.mark.parametrize("kappa", [0.0, 0.075, 0.2, 5.0, 25.0])
@pytest.mark.parametrize("rotation", [0.0, 1.7])
def test_von_mises_kappa_recovers_concentration(kappa, rotation):
    angles = np.deg2rad(np.arange(1, 360, 2))
    rates = 7 * np.exp(kappa * (np.cos(angles - rotation) - 1))

    assert tuning_curve_utils.von_mises_kappa(rates, angles) == pytest.approx(kappa, abs=1e-10)


@pytest.mark.parametrize("rates", [np.zeros(180), np.full(180, np.nan), np.full(180, np.inf)])
def test_von_mises_kappa_is_undefined_without_complete_finite_firing(rates):
    assert np.isnan(tuning_curve_utils.von_mises_kappa(rates, np.deg2rad(np.arange(0, 360, 2))))


def _saved_hd_curves():
    angles = np.deg2rad(np.arange(0, 360, 2))
    rates = [np.exp(kappa * (np.cos(angles) - 1)).tolist() for kappa in (.6, .09, .6, .6, .6)]
    return {
        "metadata": {"classification": {"rayleigh_alpha": .01, "shuffle_alpha": .01}},
        "unit_id": [21, 12, 7, 3, 8],
        "angle_bin_edges_deg": np.arange(-1, 360, 2).tolist(),
        "occupancy_time_s": [2.0] * 180,
        "firing_rate_hz": rates,
        "spike_counts": (np.asarray(rates) * 2).tolist(),
        "unit_data": {
            "hd_class": [2, 2, 1, 0, None],
            "rayleigh_p": [.01, .001, .001, .02, None],
            "shuffle_p": [.01, .001, .02, .02, .001],
            "rayleigh_significant": [True, True, True, False, None],
            "shuffle_significant": [True, True, False, False, True],
        },
    }


def test_update_hd_classification_preserves_source_tests_and_rates(tmp_path):
    data = _saved_hd_curves()
    before = json.loads(json.dumps(data))

    assert tuning_curve_utils.update_hd_classification(data) is data

    assert data["unit_data"]["hd_class"] == [3, 2, 1, 0, None]
    assert data["unit_data"]["kappa_pass"] == [True, False, True, True, True]
    np.testing.assert_allclose(data["unit_data"]["von_mises_kappa"], [.6, .09, .6, .6, .6])
    for key in before["unit_data"]:
        if key != "hd_class":
            assert data["unit_data"][key] == before["unit_data"][key]
    for key in ("unit_id", "angle_bin_edges_deg", "occupancy_time_s", "firing_rate_hz", "spike_counts"):
        assert data[key] == before[key]
    assert data["metadata"]["classification"]["kappa_cutoff"] == .1
    tuning_curve_utils.save_hd_unit_lists(data, tmp_path)
    assert np.load(tmp_path / "hd_cells_1.npy").tolist() == [7]
    assert np.load(tmp_path / "hd_cells_2.npy").tolist() == [12]
    assert np.load(tmp_path / "hd_cells_3.npy").tolist() == [21]
    # The new columns remain valid JSON and repeated refreshes are stable.
    saved = json.loads(json.dumps(data, allow_nan=False))
    tuning_curve_utils.update_hd_classification(data)
    assert data == saved


@pytest.mark.parametrize("kappa, passes", [(.1, True), (np.nextafter(.1, 0), False)])
def test_update_hd_classification_uses_inclusive_kappa_cutoff(monkeypatch, kappa, passes):
    data = _saved_hd_curves()
    monkeypatch.setattr(tuning_curve_utils, "von_mises_kappa", lambda *_: kappa)

    tuning_curve_utils.update_hd_classification(data, kappa_cutoff=.1)

    assert data["unit_data"]["kappa_pass"] == [passes] * 5
    assert data["unit_data"]["hd_class"][:2] == [3 if passes else 2] * 2
    assert data["unit_data"]["hd_class"][2:] == [1, 0, None]


@pytest.mark.parametrize("missing", ["occupancy", "rate", "zero_firing"])
def test_update_hd_classification_cannot_promote_unavailable_kappa(missing):
    data = _saved_hd_curves()
    if missing == "occupancy":
        data["occupancy_time_s"][0] = 0
    elif missing == "rate":
        for rates in data["firing_rate_hz"]:
            rates[0] = None
    else:
        data["firing_rate_hz"] = np.zeros((5, 180)).tolist()

    tuning_curve_utils.update_hd_classification(data)

    assert data["unit_data"]["hd_class"] == [2, 2, 1, 0, None]
    assert data["unit_data"]["von_mises_kappa"] == [None] * 5
    assert data["unit_data"]["kappa_pass"] == [None] * 5


def test_unbounded_kappa_passes_without_serializing_infinity():
    data = _saved_hd_curves()
    rates = np.zeros(180)
    rates[0] = 1
    data["firing_rate_hz"] = [rates.tolist()] * 5

    assert tuning_curve_utils.von_mises_kappa(rates, np.deg2rad(np.arange(0, 360, 2))) == np.inf
    tuning_curve_utils.update_hd_classification(data)

    assert data["unit_data"]["von_mises_kappa"] == [None] * 5
    assert data["unit_data"]["kappa_pass"] == [True] * 5
    assert data["unit_data"]["hd_class"] == [3, 3, 1, 0, None]
    json.dumps(data, allow_nan=False)


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


def test_saved_empty_camera_times_are_invalid(tmp_path):
    (tmp_path / "sync_data.json").write_text(json.dumps({
        "exposure_sampling_number_list_mid": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="No complete camera timing data"):
        tuning_curve_utils.get_exposure_timestamps(session_info={}, data_dir=tmp_path)


@pytest.mark.parametrize("active_high", [True, False])
@pytest.mark.parametrize("sync_exists", [True, False])
@pytest.mark.parametrize("active_at_boundary", [True, False])
def test_missing_saved_times_read_raw_adc_pulses_across_chunks(
    tmp_path, active_high, sync_exists, active_at_boundary,
):
    adc_dir = tmp_path / "node/experiment/recording/continuous/ADC"
    adc_dir.mkdir(parents=True)
    session_info = {
        "base_path": str(tmp_path), "record_nodes": "node",
        "experiment_id": "experiment", "recording_name": "recording",
        "continuous_ADC_folder": "ADC", "ADC_input_channel": 2,
    }
    origin = 100.0
    times = origin + np.arange(1_000_030) / 1000
    np.save(adc_dir / "timestamps.npy", times)
    signal = np.zeros((len(times), 2), dtype=np.int16)
    starts = np.array([10, 999_990, 1_000_010])
    ends = starts + 15
    for start, end in zip(starts, ends):
        signal[start:end, 1] = 20000
    if not active_high:
        signal[:, 1] = 20000 - signal[:, 1]
    if active_at_boundary:
        signal[:5, 1] = 20000 if active_high else 0
        signal[-3:, 1] = 20000 if active_high else 0
    signal.tofile(adc_dir / "continuous.dat")
    if sync_exists:
        (tmp_path / "sync_data.json").write_text(json.dumps({
            "recording_interval": [[0, 1010]],
        }))

    actual, adc_origin, qc = tuning_curve_utils.get_exposure_timestamps(
        session_info, tmp_path, camera_input_channel=1,
        camera_ttl_threshold=14000, camera_ttl_active_high=active_high,
    )

    np.testing.assert_allclose(actual, (times[starts] + times[ends]) / 2 - origin)
    assert adc_origin == origin
    assert qc["ttl_pulse_count"] == 3
    assert qc["timestamp_reference"] == "adc_exposure_midpoint"
    assert qc["camera_ttl_active_high"] is active_high
    assert qc["source_path"] == str(adc_dir / "continuous.dat")
    assert not (tmp_path / "camera_frame_times.npy").exists()


@pytest.mark.parametrize("include_empty_unit", [False, True])
@pytest.mark.parametrize("classification_bins", [None, 180])
def test_tuning_curve_writes_exact_columnar_contract(
    tmp_path, monkeypatch, include_empty_unit, classification_bins,
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

    occupancy_samples = np.tile(np.arange(1, 7) * 100, 30)
    occupancy_samples[-6:] = 0
    counts_by_unit = np.vstack((np.arange(180) % 17 + 1, np.arange(180) % 11 + 2))
    if include_empty_unit:
        counts_by_unit = np.vstack((np.zeros(180, dtype=int), counts_by_unit))
    counts_by_unit[:, -6:] = 0
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
    time_support = tuning_curve_utils.nap.IntervalSet(start=0.0, end=1.0)
    hd_tsd = tuning_curve_utils.nap.Tsd(t=[0.0, 1.0], d=[1.0, 181.0])

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
    classification_options = (
        {} if classification_bins is None else {"classification_bins": classification_bins}
    )
    result = tuning_curve_utils.tuning_curve(
        base_dir=base_dir,
        kilosort_dir=kilosort_dir,
        probe_name="A",
        interval_pairs=np.asarray([[0.0, 1.0]]),
        HD_tsd=hd_tsd,
        adc_time_origin_s=100.0,
        num_of_bins_in_hd=180,
        num_shuffle=1,
        shuffle_seed=42,
        is_save=True,
        save_path=save_path,
        metadata={"epoch": "arena"},
        timestamp_reference="adc_exposure_midpoint",
        **classification_options,
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
    assert result["metadata"]["timestamp_reference"] == "adc_exposure_midpoint"
    assert result["metadata"]["num_angle_bins"] == 180
    assert result["spike_counts"] == counts_by_unit.tolist()
    assert result["occupancy_samples"] == occupancy_samples.tolist()
    np.testing.assert_allclose(result["occupancy_time_s"], occupancy_samples / 100)
    np.testing.assert_allclose(result["angle_bin_edges_deg"], np.linspace(0, 360, 181))
    assert result["firing_rate_hz"][-2][0] == 1.0
    assert result["firing_rate_hz"][-1][0] == 2.0
    assert result["firing_rate_hz"][0][-1] is None
    expected_raw_rates = np.full(counts_by_unit.shape, np.nan, dtype=float)
    np.divide(
        counts_by_unit, occupancy_samples / 100,
        out=expected_raw_rates, where=occupancy_samples > 0,
    )
    np.testing.assert_allclose(
        np.asarray(result["firing_rate_hz"], dtype=float), expected_raw_rates,
    )
    assert tuple(result["unit_data"]) == (
        "rate_mvl",
        "spike_angle_mrl",
        "rayleigh_score",
        "rayleigh_p",
        "rayleigh_significant",
        "shuffle_p",
        "shuffle_significant",
        "hd_class",
        "von_mises_kappa",
        "kappa_pass",
    )
    assert all(len(column) == len(unit_ids) for column in result["unit_data"].values())
    assert len(shuffled_units) == 2
    assert all(value is not None for value in result["unit_data"]["rate_mvl"][-2:])
    expected_bins = 30 if classification_bins is None else classification_bins
    assert result["metadata"]["classification"]["num_angle_bins"] == expected_bins
    expected_edges = np.linspace(0, 360, expected_bins + 1)
    expected_centers = (expected_edges[:-1] + expected_edges[1:]) / 2
    expected_occupancy = occupancy_samples.reshape(expected_bins, -1).sum(axis=1)
    expected_counts = counts_by_unit.reshape(len(unit_ids), expected_bins, -1).sum(axis=2)
    expected_rates = np.zeros(expected_counts.shape, dtype=float)
    np.divide(
        expected_counts, expected_occupancy / 100,
        out=expected_rates, where=expected_occupancy > 0,
    )
    expected_vectors = np.exp(1j * np.deg2rad(expected_centers))
    for unit_index in range(len(unit_ids) - 2, len(unit_ids)):
        expected_mvl = (
            abs(expected_rates[unit_index] @ expected_vectors)
            / expected_rates[unit_index].sum()
        )
        assert result["unit_data"]["rate_mvl"][unit_index] == pytest.approx(expected_mvl)
        expected_score, expected_p = tuning_curve_utils.rayleigh_test(
            expected_counts[unit_index], expected_occupancy / 100, expected_centers,
        )
        assert result["unit_data"]["rayleigh_score"][unit_index] == pytest.approx(expected_score)
        assert result["unit_data"]["rayleigh_p"][unit_index] == pytest.approx(expected_p)
    for shuffle_args in shuffled_units:
        np.testing.assert_allclose(shuffle_args[9], expected_edges)
        np.testing.assert_array_equal(shuffle_args[10], np.maximum(expected_occupancy, 1))
        np.testing.assert_allclose(shuffle_args[11], expected_vectors.real)
        np.testing.assert_allclose(shuffle_args[12], expected_vectors.imag)
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
    time_support = tuning_curve_utils.nap.IntervalSet(start=0.0, end=1.0)
    hd_tsd = tuning_curve_utils.nap.Tsd(t=[0.0, 1.0], d=[1.0, 181.0])

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
        HD_tsd=hd_tsd,
        adc_time_origin_s=100.0,
        num_of_bins_in_hd=180,
        num_shuffle=1000,
        shuffle_seed=0,
    )

    assert result["unit_id"] == [218]
    assert result["spike_counts"] == [np.zeros(180, dtype=int).tolist()]
    for values in result["unit_data"].values():
        assert values == [None]


def test_head_direction_support_excludes_missing_invalid_and_paused_frames():
    camera_times = np.asarray([0.0, 0.01, 0.02, 0.03, 0.10, 0.11, 0.12, 0.13])
    frames = np.asarray([0, 1, 3, 4, 5, 6, 7])
    angles = np.asarray([1.0, np.nan, 21.0, 181.0, 181.0, np.nan, 183.0])

    feature, fs = tuning_curve_utils.make_head_direction_tsd(camera_times, frames, angles)

    assert fs == pytest.approx(100.0)
    np.testing.assert_allclose(feature.index, camera_times[[0, 3, 4, 5, 7]])
    np.testing.assert_allclose(feature.values, [1.0, 21.0, 181.0, 181.0, 183.0])
    np.testing.assert_allclose(feature.time_support.values, [
        [-0.005, 0.005], [0.025, 0.035], [0.095, 0.115], [0.125, 0.135],
    ])


@pytest.mark.parametrize("shuffle_backend", ["numpy", "numba"])
@pytest.mark.parametrize("include_singleton", [False, True])
def test_tracking_gap_spikes_do_not_change_counts_rates_or_shuffle(
    tmp_path, monkeypatch, shuffle_backend, include_singleton,
):
    if shuffle_backend == "numpy":
        monkeypatch.setattr(kilosort_utils, "_compute_shuffle_r_numba", None)
    elif kilosort_utils._compute_shuffle_r_numba is None:
        pytest.skip("Numba is unavailable")

    kilosort_dir = tmp_path / "kilosort"
    kilosort_dir.mkdir()
    (kilosort_dir / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n")
    probe_dir = tmp_path / "data/probeA"
    probe_dir.mkdir(parents=True)
    camera_times = np.arange(1001) / 100
    frame_ids = np.r_[np.arange(101), np.arange(900, 1001)]
    angles = np.r_[np.ones(101), np.full(101, 181.0)]
    clean_spikes = [0.5, 9.5]
    expected_support = [[-0.005, 1.005], [8.995, 10.005]]
    if include_singleton:
        frame_ids = np.insert(frame_ids, 101, 500)
        angles = np.insert(angles, 101, 91.0)
        clean_spikes.insert(1, 5.0)
        expected_support.insert(1, [4.995, 5.005])
    feature, fs = tuning_curve_utils.make_head_direction_tsd(camera_times, frame_ids, angles)

    results = []
    for spikes in (clean_spikes, sorted(clean_spikes + [3.0, 7.0])):
        np.save(kilosort_dir / "spike_clusters.npy", np.ones(len(spikes), dtype=int))
        np.save(probe_dir / "adc_spike_time.npy", 100 + np.asarray(spikes))
        results.append(tuning_curve_utils.tuning_curve(
            tmp_path, kilosort_dir, "A", [[-0.01, 10.01]], feature, 100,
            180, 32, 7, feature_fs_hz=fs, is_save=True,
        ))

    assert results[0] == results[1]
    result = results[1]
    counts = np.asarray(result["spike_counts"])[0]
    assert counts.sum() == 2 + include_singleton
    assert counts[0] == counts[90] == 1
    assert counts[45] == include_singleton
    assert sum(result["occupancy_samples"]) == 202 + include_singleton
    assert sum(result["occupancy_time_s"]) == pytest.approx(2.02 + include_singleton * 0.01)
    assert result["metadata"]["feature_fs_hz"] == pytest.approx(100.0)
    np.testing.assert_allclose(result["metadata"]["valid_pose_intervals_s"], expected_support)


def test_tuning_discards_epoch_fragments_without_a_pose_sample(tmp_path):
    kilosort_dir = tmp_path / "kilosort"
    kilosort_dir.mkdir()
    (kilosort_dir / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n")
    np.save(kilosort_dir / "spike_clusters.npy", np.ones(3, dtype=int))
    probe_dir = tmp_path / "data/probeA"
    probe_dir.mkdir(parents=True)
    np.save(probe_dir / "adc_spike_time.npy", [100.003, 100.5, 109.5])
    camera_times = np.arange(1001) / 100
    frame_ids = np.r_[0, np.arange(900, 1001)]
    feature, fs = tuning_curve_utils.make_head_direction_tsd(
        camera_times, frame_ids, np.ones(len(frame_ids)),
    )

    result = tuning_curve_utils.tuning_curve(
        tmp_path, kilosort_dir, "A", [[0.002, 10.01]], feature, 100,
        180, 4, 0, feature_fs_hz=fs,
    )

    assert sum(result["spike_counts"][0]) == 1
    np.testing.assert_allclose(result["metadata"]["valid_pose_intervals_s"], [[8.995, 10.005]])


@pytest.mark.parametrize("rayleigh_p, expected_class", [
    (.009, 3), (.01, 3), (np.nextafter(.01, np.inf), 1), (.03, 1),
])
@pytest.mark.parametrize("kappa_cutoff", [.075, 1000.0])
def test_tuning_classifies_rayleigh_at_inclusive_point_zero_one(
    tmp_path, monkeypatch, rayleigh_p, expected_class, kappa_cutoff,
):
    kilosort_dir = tmp_path / "kilosort"
    kilosort_dir.mkdir()
    (kilosort_dir / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n1\tgood\n")
    np.save(kilosort_dir / "spike_clusters.npy", np.ones(2, dtype=int))
    probe_dir = tmp_path / "data/probeA"
    probe_dir.mkdir(parents=True)
    np.save(probe_dir / "adc_spike_time.npy", [100.1, 100.5])
    frame_ids = np.arange(1001)
    feature, fs = tuning_curve_utils.make_head_direction_tsd(
        frame_ids / 100, frame_ids, frame_ids * 360 / 1000 % 360,
    )
    monkeypatch.setattr(tuning_curve_utils, "rayleigh_test", lambda *_: (1., rayleigh_p))
    monkeypatch.setattr(kilosort_utils, "_compute_shuffle_r_numba", None)
    monkeypatch.setattr(kilosort_utils, "_compute_shuffle_r_numpy", lambda *_: np.zeros(99))
    save_path = tmp_path / "tuning_curves.tc"

    result = tuning_curve_utils.tuning_curve(
        tmp_path, kilosort_dir, "A", [[0, 10]], feature, 100,
        180, 99, 0, feature_fs_hz=fs, is_save=True, save_path=save_path,
        kappa_cutoff=kappa_cutoff,
    )

    if expected_class == 3 and kappa_cutoff == 1000.0:
        expected_class = 2
    assert result["metadata"]["classification"]["rayleigh_alpha"] == .01
    assert result["metadata"]["classification"]["kappa_cutoff"] == kappa_cutoff
    assert result["unit_data"]["hd_class"] == [expected_class]
    assert result["unit_data"]["rayleigh_significant"] == [expected_class in (2, 3)]
    assert result["unit_data"]["shuffle_p"] == [.01]
    assert json.loads(save_path.read_text())["unit_data"] == result["unit_data"]


@pytest.mark.parametrize("hd_class, expected_ids", [
    (2, [1]), (1, [2, 3, 4]), (0, [5]), ((1, 2), [1, 2, 3, 4]),
])
def test_hd_loader_reclassifies_saved_point_zero_five_results(tmp_path, hd_class, expected_ids):
    from Utils.direction_comparison import load_hd_profiles

    path = tmp_path / "tuning_curves.tc"
    data = {
        "metadata": {"classification": {"rayleigh_alpha": .05}},
        "unit_id": list(range(1, 9)),
        "unit_data": {
            "hd_class": [2, 2, 2, 1, 0, None, None, None],
            "rayleigh_p": [.01, np.nextafter(.01, np.inf), .03, .001, .2, None, .001, None],
            "shuffle_p": [.01, .005, .005, .02, .2, .001, None, None],
        },
        "spike_counts": [[1] * 180] * 8,
        "firing_rate_hz": [[1] * 180] * 8,
        "occupancy_time_s": [1] * 180,
        "angle_bin_edges_deg": list(range(0, 361, 2)),
    }
    path.write_text(json.dumps(data))

    profiles = load_hd_profiles(path, hd_class=hd_class)

    assert profiles.index.tolist() == [("A", unit_id) for unit_id in expected_ids]
    assert set(tmp_path.iterdir()) == {path}
    assert profiles.attrs["unit_info"] == {
        ("A", unit): {"hd_class": 2 if unit == 1 else 0 if unit == 5 else 1}
        for unit in expected_ids
    }
    assert json.loads(path.read_text()) == data


@pytest.mark.parametrize("smoothing_deg", [0, 90])
def test_hd_loader_reuses_native_class_three_before_display_processing(tmp_path, smoothing_deg):
    from Utils.direction_comparison import hd_pick, load_hd_profiles

    path = tmp_path / "tuning_curves.tc"
    data = _saved_hd_curves()
    path.write_text(json.dumps(data))

    all_profiles = load_hd_profiles(path, smoothing_deg=smoothing_deg)
    selected = hd_pick(all_profiles)

    assert selected.index.tolist() == [("A", 21)]
    assert hd_pick(all_profiles, 2).index.tolist() == [("A", 12)]
    assert hd_pick(all_profiles, (2, 3)).index.tolist() == [("A", 21), ("A", 12)]
    assert load_hd_profiles(path, hd_class=3).index.tolist() == selected.index.tolist()
    assert json.loads(path.read_text()) == data
    assert set(tmp_path.iterdir()) == {path}
