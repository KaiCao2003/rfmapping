import json

import numpy as np
import pytest

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


def test_json_profiles_use_current_heading_and_preserve_saved_cohort_file(tmp_path, monkeypatch):
    from Utils import tuning_curve_utils
    from Utils.hd_rf_prediction import _json_hd_profiles

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
    monkeypatch.setattr(tuning_curve_utils, "get_exposure_timestamps",
                        lambda *a, **kw: (np.arange(40) / 10., 100., {}))
    cache = tmp_path / "profiles.json"

    profiles, provenance = _json_hd_profiles(tc_path, heading, [7], cache)

    assert profiles.loc[7].idxmax() == 90.
    assert tc_path.read_bytes() == before
    assert json.loads(cache.read_text())["spike_counts"][0][7] == 4
    assert provenance["hd_heading_source"] == str(heading)
    assert provenance["dropped_unsynchronized_frames"] == 0
