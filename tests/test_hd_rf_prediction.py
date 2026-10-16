import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from Utils.hd_rf_prediction import fit_hd_rf_model, predict_hd_rf, wrap_deg


def test_fixed_allocentric_example_and_circular_seam():
    hd = np.arange(0., 360., 30.)
    model = fit_hd_rf_model(hd, wrap_deg(40. - hd), kappas=(1., 4., 16.))
    prediction = predict_hd_rf(model, np.array([10., 20., 0., 360., np.nan]))
    np.testing.assert_allclose(prediction["fixed_allocentric_rf_ego_deg"][:2], [30., 20.])
    assert model["fixed_allocentric"]["beta_deg"] == pytest.approx(40.)
    np.testing.assert_allclose(prediction["rf_ego_deg"][2:4], [40., 40.], atol=1e-8)
    assert all(np.isnan(values[-1]) for values in prediction.values())
    assert model["nested_loo_metrics"]["n"] == len(hd)
    assert json.loads(json.dumps(model, allow_nan=False))["kappa"] == model["kappa"]


def test_circular_response_uses_short_path_and_does_not_force_constant_allo():
    hd = np.arange(0., 360., 30.)
    rf = wrap_deg(179. + 5. * np.sin(np.deg2rad(hd)))
    model = fit_hd_rf_model(hd, rf, kappas=(0., 1., 4.))
    prediction = predict_hd_rf(model, np.array([0., 90., 180., 270.]))
    assert np.all(abs(prediction["rf_ego_deg"]) > 170.)
    assert abs(wrap_deg(prediction["rf_allo_deg"][1] - prediction["rf_allo_deg"][0])) > 80.
    assert model["fixed_allocentric"]["forced"] is False


def test_outer_holdout_does_not_leak_response_into_its_prediction():
    hd = np.arange(0., 360., 45.)
    rf = wrap_deg(130. - hd + np.array([5., -3., 2., 1., 8., -4., 6., 2.]))
    first = fit_hd_rf_model(hd, rf, kappas=(.5, 2., 8.))
    rf[3] += 100.
    second = fit_hd_rf_model(hd, rf, kappas=(.5, 2., 8.))
    assert first["nested_loo_rf_ego_deg"][3] == pytest.approx(second["nested_loo_rf_ego_deg"][3])
    assert first["nested_loo_kappa"][3] == second["nested_loo_kappa"][3]


def test_support_measures_detect_sparse_heading_and_keep_shape():
    model = fit_hd_rf_model([0., 10., 20., 30.], [10., 20., 30., 40.], kappas=(1.,))
    result = predict_hd_rf(model, np.array([[0., 180.], [10., np.nan]]))
    assert result["rf_ego_deg"].shape == (2, 2)
    np.testing.assert_allclose(result["nearest_training_hd_distance_deg"][0], [0., 150.])
    assert np.all((result["resultant_length"][0] >= 0) & (result["resultant_length"][0] <= 1))


@pytest.mark.parametrize("hd,rf", [([0., 1., 2.], [2., 3., 4.]),
                                  ([0., 1., 2., 3.], [1., 2., np.nan, 4.])])
def test_fit_requires_eligible_pairs(hd, rf):
    with pytest.raises(ValueError, match="paired finite"):
        fit_hd_rf_model(hd, rf)


@pytest.mark.parametrize("write_cache", [False, True])
def test_generated_profiles_use_current_heading_and_preserve_saved_cohort_file(
    tmp_path, monkeypatch, write_cache,
):
    from Utils import camera_timing
    from Utils.hd_rf_prediction import generate_heading_profiles

    session = tmp_path / "260827_12"
    tc_path = session / "data/tuning_curves/ProbeA/tuning_curves.tc"
    tc_path.parent.mkdir(parents=True)
    ks = session / "kilosort/ProbeA/kilosort_12"
    ks.mkdir(parents=True)
    metadata = dict(probe="A", kilosort_dir=str(ks), epoch_intervals_s=[[0., 3.9]],
                    ttl_qc=dict(camera_input_channel=1, camera_ttl_active_high=False))
    tc_path.write_text(json.dumps(dict(metadata=metadata, unit_data=dict(hd_class=[3]))))
    before = tc_path.read_bytes()
    (session / "data/session_info.json").write_text(json.dumps(dict(session_info={})))
    (session / "data/probeA").mkdir()
    np.save(ks / "spike_clusters.npy", [7, 7, 7, 7])
    np.save(session / "data/probeA/adc_spike_time.npy", [100.2, 100.4, 100.6, 100.8])
    heading = session / "head_direction.json"
    heading.write_text(json.dumps(dict(hp4=dict(frames=list(range(40)), hd=[90.] * 20 + [270.] * 20))))
    monkeypatch.setattr(camera_timing, "get_exposure_timestamps",
                        lambda *a, **kw: (np.arange(40) / 10., 100., {}))
    cache = tmp_path / "profiles.json" if write_cache else None
    files_before = set(tmp_path.rglob("*"))

    profiles, provenance = generate_heading_profiles(tc_path, heading, [7], cache)

    assert profiles.loc[7].idxmax() == 90.
    assert tc_path.read_bytes() == before
    expected_rates = np.full((1, 30), np.nan)
    expected_rates[0, [7, 22]] = [2., 0.]
    np.testing.assert_allclose(profiles, expected_rates, rtol=1e-14, atol=0)
    if write_cache:
        saved = json.loads(cache.read_text())
        expected_counts = np.zeros((1, 30), dtype=int)
        expected_counts[0, 7] = 4
        expected_occupancy = np.zeros(30)
        expected_occupancy[[7, 22]] = 2.
        assert saved["spike_counts"] == expected_counts.tolist()
        np.testing.assert_allclose(saved["occupancy_time_s"], expected_occupancy, rtol=1e-14, atol=0)
        assert saved["unit_id"] == [7]
        assert saved["angle_bin_edges_deg"] == np.arange(0., 361., 12.).tolist()
        np.testing.assert_array_equal(np.asarray(saved["firing_rate_hz"], dtype=float), profiles)
    else:
        assert set(tmp_path.rglob("*")) == files_before
    assert provenance["hd_heading_source"] == str(heading)
    assert provenance["dropped_unsynchronized_frames"] == 0


def test_heading_profile_calculation_is_in_memory_and_preserves_missing_bins(monkeypatch):
    import pynapple as nap
    from Utils.hd_rf_prediction import calculate_heading_profiles

    counts = np.zeros((30, 2))
    counts[[3, 12]] = [[4., 6.], [8., 0.]]
    occupancy_samples = np.zeros(30)
    occupancy_samples[[3, 12]] = [20., 40.]
    curves = xr.DataArray(counts, dims=("angle", "unit"), coords={"unit": [9, 7]},
                          attrs={"occupancy": occupancy_samples})
    spikes, feature, epochs = object(), object(), object()

    def compute(**settings):
        assert settings == dict(data=spikes, features=feature, bins=30, epochs=epochs,
                                range=(0., 360.), return_counts=True, fs=10.)
        return curves

    def fail_io(*args, **kwargs):
        pytest.fail("The in-memory heading calculation must not open files")

    monkeypatch.setattr(nap, "compute_tuning_curves", compute)
    monkeypatch.setattr(Path, "open", fail_io)
    profiles, actual_counts, occupancy = calculate_heading_profiles(
        spikes, feature, epochs, feature_fs_hz=10.,
    )
    assert profiles.index.tolist() == [9, 7]
    np.testing.assert_array_equal(profiles.columns, np.arange(6., 360., 12.))
    np.testing.assert_array_equal(actual_counts, counts.T)
    np.testing.assert_array_equal(occupancy, occupancy_samples / 10.)
    np.testing.assert_array_equal(profiles.iloc[:, [3, 12]], [[2., 2.], [3., 0.]])
    assert np.isnan(profiles.drop(columns=[42., 150.])).all().all()


def test_heading_profile_writer_saves_supplied_values_without_calculating(tmp_path, monkeypatch):
    from Utils import hd_rf_prediction as prediction

    def fail_calculation(*args, **kwargs):
        pytest.fail("Writing a saved heading profile must not rerun calculation")

    monkeypatch.setattr(prediction, "calculate_heading_profiles", fail_calculation)
    values = np.full((1, 30), np.nan)
    values[0, 7] = 2.
    table = pd.DataFrame(values, index=[7], columns=np.arange(6., 360., 12.))
    counts = np.zeros((1, 30))
    counts[0, 7] = 4.
    occupancy = np.zeros(30)
    occupancy[7] = 2.
    path = tmp_path / "profiles.json"
    prediction.save_heading_profiles(path, table, counts, occupancy, metadata={"source": "test"},
                                     epoch_intervals_s=[[0., 2.]], feature_fs_hz=10.)
    saved = json.loads(path.read_text())
    assert saved["metadata"] == {"source": "test"}
    assert saved["epoch_intervals_s"] == [[0., 2.]]
    assert saved["feature_fs_hz"] == 10.
    assert saved["spike_counts"] == counts.astype(int).tolist()
    np.testing.assert_array_equal(np.asarray(saved["firing_rate_hz"], dtype=float), values)
