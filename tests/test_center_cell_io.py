"""Saved-input contracts for center-cell clocks, identity and explicit exports."""

import json

import numpy as np
import pandas as pd
import pytest

from EBB.center_cell_io import (
    load_center_analysis, load_center_session, load_center_tc, load_motive_center_session,
    load_lfp, make_center_tc, save_center_analysis, save_lfp,
)
from EBB.center_cells import CenterCellConfig, analyze_center_cells, prepare_center_behavior


@pytest.fixture
def saved_session(tmp_path):
    session = tmp_path / "m22" / "261008" / "261008_11"
    directory = session / "data" / "spatial_cells" / "ProbeA" / "baseline"
    kilosort = session / "kilosort" / "ProbeA" / "kilosort_11"
    spike_path = session / "data" / "probeA" / "adc_spike_time.npy"
    for path in (directory, kilosort, spike_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    times = np.array([1., 2., 3.])
    frames = np.array([100, 101, 102])
    np.savez_compressed(
        directory / "restored_maps.npz", frame_times=times, frame_ids=frames,
        trajectory_x_cm=np.array([np.nan, 12., 20.]),
        trajectory_y_cm=np.array([np.nan, 10., 20.]),
        unit_ids=np.array([0, 9, 12]),
    )
    np.savez_compressed(
        directory / "position_correction.npz", frame_times=times, frame_ids=frames,
        ebc_hd_deg=np.array([360., 270., 50.]), hd_deg=np.array([360., -90., 45.]),
        position_held=np.array([False, True, False]),
        initial_outside=np.array([True, False, False]),
    )
    (kilosort / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n5\tmua\n9\tgood\n12\tgood\n"
    )
    # Include absolute-clock endpoints, out-of-trial spikes, a rejected unit,
    # unsorted input times, and a good unit without any in-trial spikes.
    np.save(spike_path, np.array([100.5, 101., 101.75, 102.5, 103., 103.5, 102.25]))
    np.save(kilosort / "spike_clusters.npy", np.array([0, 0, 9, 0, 9, 12, 5]))
    audit = {
        "status": "complete", "probe_name": "A", "phase_key": "baseline",
        "heading_convention": "physical north-zero clockwise heading; clockwise egocentric bearings",
        "heading_calibration": "image_unit_vector_scaled_by_cm_per_pixel_xy",
        "arena_size_cm": [40., 40.],
        "source": {"adc_time_origin_s": 100., "kilosort_dir": str(kilosort),
                   "spike_times_path": str(spike_path)},
    }
    (directory / "audit.json").write_text(json.dumps(audit))
    return session, directory, audit


def test_saved_clock_rebasing_preserves_unit_zero_and_silent_good_units(saved_session):
    session, directory, _ = saved_session
    before = {str(p.relative_to(session)): p.read_bytes() for p in session.rglob("*") if p.is_file()}
    result = load_center_session(session)
    assert set(result["spikes_by_unit"]) == {0, 9, 12}
    np.testing.assert_allclose(result["spikes_by_unit"][0], [1., 2.5])
    np.testing.assert_allclose(result["spikes_by_unit"][9], [1.75, 3.])
    assert result["spikes_by_unit"][12].shape == (0,)
    np.testing.assert_allclose(result["times_s"], [1., 2., 3.])
    np.testing.assert_array_equal(result["hd_cw_deg"], [360., 270., 50.])
    np.testing.assert_array_equal(result["raw_hd_cw_deg"], [360., -90., 45.])
    np.testing.assert_array_equal(result["valid_mask"], [False, False, True])
    assert np.isnan(result["xy_cm_y_down"][0]).all()
    np.testing.assert_array_equal(result["saved_unit_ids"], [0, 9, 12])
    assert result["tracking_selection"]["excluded_fraction"] == pytest.approx(2 / 3)
    assert len(result["provenance"]) == 6
    assert all(len(item["sha256"]) == 64 for item in result["provenance"])
    after = {str(p.relative_to(session)): p.read_bytes() for p in session.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize("field", ["frame_times", "frame_ids"])
def test_geometry_and_heading_must_use_identical_saved_frames(saved_session, field):
    session, directory, _ = saved_session
    path = directory / "position_correction.npz"
    with np.load(path, allow_pickle=False) as saved:
        values = {key: saved[key].copy() for key in saved.files}
    values[field][1] += 1
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="frame clocks differ"):
        load_center_session(session)


@pytest.mark.parametrize("field", ["kilosort_dir", "spike_times_path"])
def test_stale_recording_or_probe_provenance_is_rejected(saved_session, field):
    session, directory, audit = saved_session
    if field == "kilosort_dir":
        audit["source"][field] = str(session / "kilosort" / "ProbeB" / "kilosort_11")
    else:
        audit["source"][field] = str(session.parent / "261008_2" / "data" / "probeA" / "adc_spike_time.npy")
    (directory / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="provenance"):
        load_center_session(session)


@pytest.mark.parametrize("changes", [
    {"heading_convention": "north-zero counterclockwise heading"},
    {"heading_calibration": "uncalibrated_pixel_angles"},
])
def test_uncalibrated_or_opposite_heading_convention_is_rejected(saved_session, changes):
    session, directory, audit = saved_session
    audit.update(changes)
    (directory / "audit.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="heading"):
        load_center_session(session)


def test_missing_saved_inputs_never_trigger_generation(tmp_path, saved_session):
    missing_session = tmp_path / "absent_recording"
    with pytest.raises(FileNotFoundError):
        load_center_session(missing_session)
    assert not missing_session.exists()
    session, directory, _ = saved_session
    missing = directory / "position_correction.npz"
    missing.unlink()
    with pytest.raises(FileNotFoundError):
        load_center_session(session)
    assert not missing.exists()


@pytest.mark.parametrize("name", ["theta.npz", "theta_without_suffix"])
def test_lfp_round_trip_uses_exact_requested_path_and_refuses_overwrite(tmp_path, name):
    path = tmp_path / "exports" / name
    lfp = {"times_s": np.array([1., 1.001, 1.002]),
           "voltage_uv": np.array([-20., 0., 20.]), "sample_rate_hz": 1000.,
           "metadata": {"clock": "ADC-relative seconds", "channel": 0}}
    save_lfp(path, lfp)
    assert path.is_file()
    assert not path.with_name(path.name + ".npz").exists()
    loaded = load_lfp(path)
    np.testing.assert_array_equal(loaded["times_s"], lfp["times_s"])
    np.testing.assert_array_equal(loaded["voltage_uv"], lfp["voltage_uv"])
    assert loaded["sample_rate_hz"] == 1000.
    assert loaded["metadata"] == lfp["metadata"]
    original = path.read_bytes()
    updated = {**lfp, "voltage_uv": lfp["voltage_uv"] * 2}
    with pytest.raises(FileExistsError):
        save_lfp(path, updated)
    assert path.read_bytes() == original
    save_lfp(path, updated, overwrite=True)
    np.testing.assert_array_equal(load_lfp(path)["voltage_uv"], updated["voltage_uv"])


def test_lfp_reader_does_not_create_missing_export(tmp_path):
    path = tmp_path / "not_exported.npz"
    with pytest.raises(FileNotFoundError):
        load_lfp(path)
    assert not path.exists()


def test_motive_reader_preserves_saved_geometry_native_yaw_and_silent_units(tmp_path, monkeypatch):
    from Utils import camera_timing

    session = tmp_path / "m19/260827/260827_9"
    data_dir = session / "data"
    geometry_path = data_dir / "spatial_cells/ProbeA/baseline/inner/egocentric_rate_map.rfmap"
    pose_path = data_dir / "processed/filtered.csv"
    kilosort = session / "kilosort/ProbeA/kilosort_9"
    spike_path = data_dir / "probeA/adc_spike_time.npy"
    for directory in (geometry_path.parent, pose_path.parent, kilosort, spike_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    geometry = dict(boundaryType="circle", boundaryCenterRaw=[10., 20.],
                    cmPerPositionUnit=2., boundaryDiameterCm=20., positionSource=str(pose_path))
    with geometry_path.open("wb") as stream:
        np.savez_compressed(stream, metadata=np.frombuffer(json.dumps(geometry).encode(), dtype=np.uint8))
    columns = pd.MultiIndex.from_tuples([
        ("Frame", "", "", "Frame"), ("Time", "", "", "Time"),
        *(("hp4", "id", kind, axis) for kind in ("Rotation", "Position") for axis in "XYZ"),
    ])
    pose = np.column_stack((
        np.arange(5), np.arange(5), np.zeros((5, 2)), [-90., 0., 90., 180., np.nan],
        [10., 11., 10., 20., np.nan], [21., 20., 20., 30., 21.], np.zeros(5),
    ))
    pd.DataFrame(pose, columns=columns).to_csv(pose_path, index=False)
    (data_dir / "session_info.json").write_text('{"session_info": {}}')
    (data_dir / "sync_data.json").write_text("{}")
    (data_dir / "interval_table.csv").write_text("start,end,interval_type\n1,5,baseline\n")
    (kilosort / "cluster_KSLabel.tsv").write_text("cluster_id\tKSLabel\n0\tgood\n5\tmua\n9\tgood\n")
    np.save(spike_path, [101., 102.5, 104.5, 106., 103.])
    np.save(kilosort / "spike_clusters.npy", [0, 0, 0, 9, 5])
    monkeypatch.setattr(
        camera_timing, "get_exposure_timestamps",
        lambda *args, **kwargs: (np.arange(1., 6.), 100., {"source_path": str(data_dir / "sync_data.json")}),
    )
    before = {str(path): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    result = load_motive_center_session(session)
    np.testing.assert_allclose(result["times_s"], np.arange(1., 6.))
    np.testing.assert_allclose(result["xy_cm_y_down"][:3], [[10., 8.], [12., 10.], [10., 10.]])
    np.testing.assert_allclose(result["hd_cw_deg"][:4], [180., 90., 0., 270.])
    np.testing.assert_allclose(result["native_hd_ccw_deg"][:4], [-90., 0., 90., 180.])
    np.testing.assert_array_equal(result["valid_mask"], [True, True, True, False, False])
    np.testing.assert_allclose(result["spikes_by_unit"][0], [1., 2.5, 4.5])
    assert set(result["spikes_by_unit"]) == {0, 9}
    assert result["spikes_by_unit"][9].size == 0
    assert result["arena_shape"] == "circle"
    assert result["geometry"]["center_raw"] == [10., 20.]
    after = {str(path): path.read_bytes() for path in session.rglob("*") if path.is_file()}
    assert after == before


def test_motive_reader_requires_saved_geometry_without_generating_it(tmp_path):
    session = tmp_path / "260827_9"
    with pytest.raises(FileNotFoundError):
        load_motive_center_session(session)
    assert not session.exists()


@pytest.fixture
def center_result():
    times = np.arange(0., 60.05, .05)
    phase = times * .7
    radius = 8 + 6 * np.sin(times * 1.37)
    xy = 20 + radius[:, None] * np.c_[np.cos(phase), np.sin(phase)]
    config = CenterCellConfig(position_smoothing_s=0., angle_bin_deg=90., angle_smoothing_deg=90.,
                              min_spatial_coverage_exclusive=0., min_spikes_exclusive=5)
    behavior = prepare_center_behavior(times, xy, times * 51 % 360, 40., config=config)
    result = analyze_center_cells(behavior, {0: times[::3], "001": times[::5], "silent": []},
                                  n_shuffles=3, seed=17)
    result["summary"]["cb_is_cell"] = pd.array([True, False, pd.NA], dtype="boolean")
    return result


def test_center_save_load_preserves_identity_nullable_classes_and_complete_evidence(tmp_path, center_result):
    directory = tmp_path / "center"
    provenance = [{"path": "/saved/input.npz", "sha256": "unchanged"}]
    save_center_analysis(directory, center_result, provenance=provenance)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    loaded = load_center_analysis(directory)
    pd.testing.assert_frame_equal(loaded["summary"], center_result["summary"])
    assert loaded["unit_ids"] == [0, "001", "silent"]
    assert loaded["config"] == center_result["config"]
    assert loaded["metadata"] == center_result["metadata"]
    assert loaded["provenance"] == provenance
    for unit, curves in center_result["tuning"].items():
        for feature, fields in curves.items():
            assert loaded["tuning"][unit][feature].keys() == fields.keys()
            for name, values in fields.items():
                np.testing.assert_array_equal(loaded["tuning"][unit][feature][name], values)
    for statistic, values in center_result["nulls"].items():
        np.testing.assert_array_equal(loaded["nulls"][statistic], values)
        np.testing.assert_allclose(loaded["thresholds"][statistic], center_result["thresholds"][statistic])
    np.testing.assert_array_equal(loaded["shuffle_offsets_s"], center_result["shuffle_offsets_s"])
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    assert set(before) == {"center_cells.csv", "center_tuning_and_nulls.npz", "manifest.json",
                           "cb_tuning.tc", "cd_tuning.tc"}


@pytest.mark.parametrize("feature,edges_name", [("cb", "angle_bin_edges_deg"),
                                               ("cd", "distance_bin_edges_cm")])
def test_center_tuning_uses_hd_fields_with_native_coordinates_and_shuffle_evidence(
    tmp_path, center_result, feature, edges_name,
):
    save_center_analysis(tmp_path, center_result, provenance=[])
    path = tmp_path / f"{feature}_tuning.tc"
    data = load_center_tc(path)
    json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(f"Nonstandard JSON constant: {value}"))
    assert data == make_center_tc(center_result, feature)
    assert data["unit_id"] == [0, "001", "silent"]
    reference = center_result["tuning"][0][feature]
    np.testing.assert_array_equal(data[edges_name], reference["edges"])
    np.testing.assert_array_equal(data["occupancy_time_s"], reference["occupancy_s"])
    np.testing.assert_array_equal(data["spike_counts"][0], reference["spike_counts"])
    np.testing.assert_allclose(np.asarray(data["firing_rate_hz"][0], float), reference["rate_hz"], equal_nan=True)
    assert set(data["shuffle_statistics"]) == {key for key in center_result["nulls"] if key.startswith(feature + "_")}
    assert data["unit_data"][f"{feature}_is_cell"][-1] is None
    assert "hd_class" not in data["unit_data"]
    if feature == "cb":
        assert data["unit_data"]["cb_is_cell"] == [True, False, None]
        assert reference["edges"][-1] == 360.
    else:
        assert data["metadata"]["coordinate_units"] == "cm"


def test_center_exports_require_explicit_overwrite_and_reject_mixed_theta_directory(tmp_path, center_result):
    save_center_analysis(tmp_path, center_result, provenance=[])
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        save_center_analysis(tmp_path, center_result, provenance=[])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    save_center_analysis(tmp_path, center_result, provenance=[], overwrite=True)
    (tmp_path / "theta_cells.csv").write_text("old theta results\n")
    with pytest.raises(ValueError, match="center-only"):
        save_center_analysis(tmp_path, center_result, provenance=[], overwrite=True)


def test_center_reader_does_not_create_missing_exports(tmp_path):
    missing = tmp_path / "not_generated"
    with pytest.raises(FileNotFoundError):
        load_center_analysis(missing)
    with pytest.raises(FileNotFoundError):
        load_center_tc(missing / "cb_tuning.tc")
    assert not missing.exists()


def test_previous_center_exports_load_without_writing_new_tuning_files(tmp_path, center_result):
    result = {**center_result, "summary": center_result["summary"].iloc[:1].copy(), "unit_ids": [0],
              "tuning": {0: center_result["tuning"][0]},
              "nulls": {key: value[:1] for key, value in center_result["nulls"].items()},
              "shuffle_offsets_s": center_result["shuffle_offsets_s"][:1]}
    result["summary"]["unit_id"] = result["summary"]["unit_id"].astype(int)
    result["summary"] = result["summary"].set_index("unit_id", drop=False)
    save_center_analysis(tmp_path, result, provenance=[])
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["schema_version"], manifest["unit_ids"]
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "cb_tuning.tc").unlink()
    (tmp_path / "cd_tuning.tc").unlink()
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    pd.testing.assert_frame_equal(load_center_analysis(tmp_path)["summary"], result["summary"])
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
