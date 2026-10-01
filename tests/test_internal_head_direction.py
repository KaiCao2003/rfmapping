from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

import decode_internal_head_direction as decoder


@pytest.fixture
def directional_population():
    support = nap.IntervalSet(start=0.0, end=4.0)
    frame_times = np.arange(400) / 100 + 0.005
    feature = nap.Tsd(
        t=frame_times, d=np.repeat([45.0, 135.0, 225.0, 315.0], 100),
        time_support=support,
    )
    spikes = nap.TsGroup({
        unit: nap.Ts(t=second + np.arange(20) / 20 + 0.025, time_support=support)
        for second, unit in enumerate([11, 22, 33, 44])
    }, time_support=support)
    curves, training = decoder.fit_tuning_curves(spikes, feature, support, 100.0, 4)
    return spikes, curves, training


def test_pynapple_decodes_population_preferred_direction(directional_population):
    spikes, curves, epochs = directional_population
    frame_times = np.arange(40) / 10 + 0.05

    angles, qc = decoder.decode_frames(spikes, curves, epochs, frame_times, 0.2)

    np.testing.assert_array_equal(angles, np.repeat([45.0, 135.0, 225.0, 315.0], 10))
    assert qc == {"decoded_bins": 20, "valid_bins": 20}


def test_decoding_chunks_preserve_bin_assignment(directional_population):
    spikes, curves, epochs = directional_population
    frame_times = np.arange(400) / 100

    whole, whole_qc = decoder.decode_frames(
        spikes, curves, epochs, frame_times, 0.1, chunk_bins=100,
    )
    chunked, chunk_qc = decoder.decode_frames(
        spikes, curves, epochs, frame_times, 0.1, chunk_bins=3,
    )

    np.testing.assert_array_equal(chunked, whole)
    assert chunk_qc == whole_qc


def test_gap_partial_and_silent_bins_stay_null(directional_population):
    _, curves, _ = directional_population
    epochs = nap.IntervalSet(start=[0.0, 1.0], end=[0.25, 1.35])
    spikes = nap.TsGroup({
        unit: nap.Ts(t=[0.025, 0.175, 1.025, 1.225], time_support=epochs)
        for unit in [11, 22, 33, 44]
    }, time_support=epochs)
    frame_times = np.array([
        -0.01, 0.01, 0.05, 0.1, 0.15, 0.2, 0.24, 0.25, 0.5,
        1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.34, 1.35,
    ])

    angles, qc = decoder.decode_frames(
        spikes, curves, epochs, frame_times, 0.1, chunk_bins=1,
    )

    np.testing.assert_array_equal(np.isfinite(angles), [
        False, True, True, True, True, False, False, False, False,
        True, True, False, False, True, True, False, False, False,
    ])
    assert qc == {"decoded_bins": 5, "valid_bins": 4}


def test_exact_count_boundary_uses_the_following_direction(directional_population):
    _, curves, _ = directional_population
    epochs = nap.IntervalSet(start=0.0, end=1.0)
    spikes = nap.TsGroup({
        unit: nap.Ts(t=times, time_support=epochs)
        for unit, times in [(11, [0.625]), (22, [0.525]), (33, []), (44, [])]
    }, time_support=epochs)

    angles, _ = decoder.decode_frames(
        spikes, curves, epochs, np.array([0.599999999, 0.6, 0.699999999, 0.7]), 0.1,
    )

    np.testing.assert_array_equal(angles, [135.0, 45.0, 45.0, np.nan])


def test_unvisited_angles_are_excluded_from_decoder(directional_population):
    spikes, _, _ = directional_population
    support = nap.IntervalSet(start=0.0, end=2.0)
    feature = nap.Tsd(
        t=np.arange(200) / 100 + 0.005, d=np.repeat([45.0, 225.0], 100),
        time_support=support,
    )

    curves, epochs = decoder.fit_tuning_curves(spikes, feature, support, 100.0, 4)
    angles, _ = decoder.decode_frames(
        spikes, curves, epochs, np.arange(20) / 10 + 0.05, 0.1,
    )

    np.testing.assert_array_equal(curves.coords[curves.dims[1]].values, [45.0, 225.0])
    assert np.all(np.asarray(curves.attrs["occupancy"]) > 0)
    np.testing.assert_array_equal(angles, np.repeat([45.0, 225.0], 10))


def test_output_preserves_frames_and_uses_internal_direction():
    payload = decoder.build_payload(np.array([0, 3, 5]), np.array([45.0, np.nan, 315.0]))

    assert json.loads(json.dumps(payload, allow_nan=False)) == {
        "internal_direction": {
            "frames": [0, 3, 5], "head_direction_deg": [45.0, None, 315.0],
        },
    }


@pytest.mark.parametrize(
    "options,expected,mode",
    [({}, [11], "saved_hd_class"), ({"all_good": True}, [11, 22, 33], "all_good"),
     ({"units": [33, 11, 11]}, [11, 33], "explicit")],
)
def test_spike_selection_and_adc_origin(tmp_path, options, expected, mode):
    kilosort = tmp_path / "kilosort/ProbeA/kilosort_4"
    kilosort.mkdir(parents=True)
    pd.DataFrame({
        "cluster_id": [11, 22, 33, 44], "KSLabel": ["good", "good", "good", "mua"],
    }).to_csv(kilosort / "cluster_KSLabel.tsv", sep="\t", index=False)
    np.save(kilosort / "spike_clusters.npy", np.array([22, 11, 33, 44, 11]))
    (tmp_path / "data/probeA").mkdir(parents=True)
    np.save(tmp_path / "data/probeA/adc_spike_time.npy", 100 + np.arange(5) / 10 + 0.05)
    if mode == "saved_hd_class":
        tc_path = tmp_path / "data/tuning_curves/ProbeA/tuning_curves.tc"
        tc_path.parent.mkdir(parents=True)
        tc_path.write_text(json.dumps({
            "unit_id": [11, 22, 33, 44], "unit_data": {"hd_class": [3, 1, None, 3]},
        }))

    spikes, selection = decoder.load_spikes(
        tmp_path, kilosort, "A", 100.0, nap.IntervalSet(start=0.0, end=1.0), **options,
    )

    assert list(spikes.keys()) == expected
    np.testing.assert_allclose(spikes[11].index.to_numpy(), [0.15, 0.45])
    assert selection["mode"] == mode
    assert selection["unit_ids"] == expected
    with pytest.raises(ValueError, match="good KSLabel"):
        decoder.load_spikes(
            tmp_path, kilosort, "A", 100.0, nap.IntervalSet(start=0.0, end=1.0), units=[44],
        )


def test_motive_trailing_frame_is_preserved_but_unsynced(tmp_path):
    path = tmp_path / "head_direction.json"
    path.write_text(json.dumps({"hp4": {
        "frames": [0, 1, 2, 3, 4], "head_direction_deg": [0, 90, 180, 270, 45],
    }}))

    frames, synced, feature, fs = decoder.load_head_direction(
        path, "hp4", np.array([0.05, 0.15, 0.25, 0.35]),
    )

    np.testing.assert_array_equal(frames, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(synced, [True, True, True, True, False])
    np.testing.assert_array_equal(feature.values, [0, 90, 180, 270])
    assert fs == pytest.approx(10.0)


def test_basler_sparse_hd_and_camera_pause_support(tmp_path):
    path = tmp_path / "head_direction.json"
    path.write_text(json.dumps({"hp4": {"frames": [0, 2, 3, 4, 5], "hd": [-10, None, 90, 180, 270]}}))
    times = np.array([0.05, 0.15, 0.25, 1.05, 1.15, 1.25])

    frames, synced, feature, fs = decoder.load_head_direction(path, "hp4", times, basler=True)

    np.testing.assert_array_equal(frames, [0, 2, 3, 4, 5])
    assert synced.all()
    np.testing.assert_array_equal(feature.values, [350, 90, 180, 270])
    np.testing.assert_allclose(feature.time_support.values, [[0.0, 0.1], [1.0, 1.3]], atol=1e-9)
    assert fs == pytest.approx(10.0)
    with pytest.raises(ValueError, match="Motive requires consecutive frames"):
        decoder.load_head_direction(path, "hp4", times)


def test_combined_probes_preserve_overlapping_ids_counts_and_support(tmp_path):
    support = nap.IntervalSet(start=[0.0, 0.7], end=[0.5, 1.0])
    kilosort_dirs = {}
    for probe, times in [("A", [0.1, 0.2, 0.8]), ("B", [0.15, 0.75, 0.85, 0.95])]:
        kilosort = tmp_path / "kilosort" / f"Probe{probe}" / "kilosort_4"
        kilosort.mkdir(parents=True)
        pd.DataFrame({"cluster_id": [11], "KSLabel": ["good"]}).to_csv(
            kilosort / "cluster_KSLabel.tsv", sep="\t", index=False,
        )
        np.save(kilosort / "spike_clusters.npy", np.full(len(times), 11))
        spike_dir = tmp_path / "data" / f"probe{probe}"
        spike_dir.mkdir(parents=True)
        np.save(spike_dir / "adc_spike_time.npy", 100 + np.array(times))
        kilosort_dirs[probe] = kilosort

    spikes, metadata = decoder.load_population(
        tmp_path, kilosort_dirs, 100.0, support, all_good=True,
    )

    assert list(spikes.keys()) == [0, 1]
    np.testing.assert_array_equal(spikes.time_support.values, support.values)
    np.testing.assert_allclose(spikes[0].index.to_numpy(), [0.1, 0.2, 0.8])
    np.testing.assert_allclose(spikes[1].index.to_numpy(), [0.15, 0.75, 0.85, 0.95])
    np.testing.assert_array_equal(spikes.count(ep=support).values, [[2, 1], [1, 3]])
    assert metadata["decoder_unit_map"] == [
        {"decoder_unit_id": 0, "probe": "A", "unit_id": 11},
        {"decoder_unit_id": 1, "probe": "B", "unit_id": 11},
    ]
    assert metadata["unit_selection"]["per_probe"] == {
        probe: {"mode": "all_good", "unit_ids": [11]} for probe in ["A", "B"]
    }


@pytest.mark.parametrize("options", [
    ["--probe", "A", "A"],
    ["--probe", "A", "B", "--kilosort-dir", "/tmp/ks"],
    ["--probe", "A", "B", "--units", "11"],
])
def test_cli_rejects_ambiguous_or_duplicate_multi_probe_selection(options):
    with pytest.raises(SystemExit):
        decoder.parse_args(["/tmp/session", *options])


@pytest.mark.parametrize("probes", [["A"], ["A", "B"]])
def test_run_defaults_and_population_metadata(tmp_path, monkeypatch, probes):
    frame_times = np.arange(400) / 100 + 0.005
    data_dir = tmp_path / "data"
    (data_dir / "processed").mkdir(parents=True)
    (data_dir / "session_info.json").write_text(json.dumps({"session_info": {}}))
    (data_dir / "processed/head_direction.json").write_text(json.dumps({"hp4": {
        "frames": list(range(400)),
        "head_direction_deg": np.repeat([45.0, 135.0, 225.0, 315.0], 100).tolist(),
    }}))
    pd.DataFrame({"interval_type": ["baseline"], "start": [0.0], "end": [4.0]}).to_csv(
        data_dir / "interval_table.csv", index=False,
    )
    for probe in probes:
        kilosort = tmp_path / "kilosort" / f"Probe{probe}" / "kilosort_4"
        kilosort.mkdir(parents=True)
        pd.DataFrame({"cluster_id": [11, 22, 33, 44], "KSLabel": ["good"] * 4}).to_csv(
            kilosort / "cluster_KSLabel.tsv", sep="\t", index=False,
        )
        np.save(kilosort / "spike_clusters.npy", np.repeat([11, 22, 33, 44], 20))
        spike_dir = data_dir / f"probe{probe}"
        spike_dir.mkdir()
        np.save(spike_dir / "adc_spike_time.npy", 100 + np.arange(80) / 20 + 0.025)
        tc_path = data_dir / "tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
        tc_path.parent.mkdir(parents=True)
        tc_path.write_text(json.dumps({
            "unit_id": [11, 22, 33, 44], "unit_data": {"hd_class": [3, 3, 3, 3]},
            "metadata": {"probe": probe},
        }))
    monkeypatch.setattr(decoder, "get_exposure_timestamps", lambda *args, **kwargs: (
        frame_times, 100.0, {"source": "test_clock"},
    ))
    options = [] if probes == ["A"] else ["--probe", *probes]
    args = decoder.parse_args([str(tmp_path), *options])
    assert args.probe == probes

    output = decoder.run(args)

    payload = json.loads(output.read_text())
    assert list(payload) == ["internal_direction"]
    assert payload["internal_direction"]["frames"] == list(range(400))
    assert all(value is not None for value in payload["internal_direction"]["head_direction_deg"])
    metadata = json.loads(output.with_suffix(".metadata.json").read_text())
    assert metadata["angle_bins"] == 60
    assert metadata["bin_size_s"] == 0.1
    np.testing.assert_allclose(metadata["training_intervals_s"], [[0.0, 4.0]], atol=1e-8)
    assert metadata["valid_frame_count"] == 400
    if len(probes) == 1:
        assert metadata["probe"] == "A"
        assert metadata["unit_selection"]["unit_ids"] == [11, 22, 33, 44]
        assert metadata["unit_selection"]["mode"] == "saved_hd_class"
        assert "decoder_unit_map" not in metadata
    else:
        assert metadata["probes"] == probes
        assert set(metadata["kilosort_dirs"]) == set(probes)
        assert metadata["unit_selection"]["mode"] == "combined_probes"
        for probe in probes:
            selected = metadata["unit_selection"]["per_probe"][probe]
            assert selected["mode"] == "saved_hd_class"
            assert selected["hd_class"] == 3
            assert selected["unit_ids"] == [11, 22, 33, 44]
            assert selected["source_metadata"] == {"probe": probe}
        assert metadata["decoder_unit_map"] == [
            {"decoder_unit_id": index, "probe": probe, "unit_id": unit}
            for index, (probe, unit) in enumerate(
                (probe, unit) for probe in probes for unit in [11, 22, 33, 44]
            )
        ]
