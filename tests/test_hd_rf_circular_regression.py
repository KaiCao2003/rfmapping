import json

import numpy as np
import pytest

from Utils.hd_rf_circular_regression import (
    fisher_lee_correlation, fit_circular_conversion, predict_circular_conversion,
)


def test_uniform_reflection_has_negative_correlation_and_exact_conversion():
    hd = np.arange(0., 360., 30.)
    rf = (40. - hd + 180.) % 360. - 180.
    model = fit_circular_conversion(hd, rf, n_permutations=999)
    assert model["rho"] == pytest.approx(-1.)
    assert model["beta_deg"] == pytest.approx(40.)
    assert model["mae_deg"] == pytest.approx(0., abs=1e-12)
    assert model["permutation_p"] <= .01
    assert model["conversion_slope"] == -1
    np.testing.assert_allclose(model["observed_rf_allo_deg"], 40.)
    assert json.loads(json.dumps(model, allow_nan=False))["n"] == 12


def test_positive_association_is_not_forced_negative_and_rotation_is_invariant():
    hd = np.array([3., 18., 49., 91., 144., 222., 301.])
    rf = hd + 37.
    model = fit_circular_conversion(hd, rf, n_permutations=99)
    assert model["rho"] == pytest.approx(1.)
    assert model["mae_deg"] > 30.
    assert fisher_lee_correlation(hd + 178., rf - 191.) == pytest.approx(1.)
    assert fisher_lee_correlation(hd, -rf) == pytest.approx(-1.)


def test_prediction_crosses_seam_and_preserves_nonfinite_headings():
    prediction = predict_circular_conversion(np.array([[0., 360., 179.], [181., -179., np.nan]]), 5.)
    np.testing.assert_allclose(prediction["rf_ego_deg"][0], [5., 5., -174.])
    np.testing.assert_allclose(prediction["rf_ego_deg"][1, :2], [-176., -176.])
    np.testing.assert_allclose(prediction["rf_allo_deg"][np.isfinite(prediction["rf_allo_deg"])], 5.)
    assert np.isnan(prediction["rf_ego_deg"][1, 2])
    assert np.isnan(prediction["rf_allo_deg"][1, 2])


def test_permutations_repeat_and_outcome_selection_has_no_p_value():
    hd = np.arange(0., 360., 45.)
    rf = 119. - hd + np.array([7., -8., 2., 15., -12., 4., 3., -6.])
    first = fit_circular_conversion(hd, rf, n_permutations=999, random_seed=14)
    second = fit_circular_conversion(hd, rf, n_permutations=999, random_seed=14)
    assert first == second
    selected = fit_circular_conversion(hd, rf, selected_on_sum=True)
    assert selected["rho"] == first["rho"]
    assert selected["beta_deg"] == first["beta_deg"]
    assert selected["permutation_p"] is None
    assert selected["n_permutations"] == 0
    json.dumps(selected, allow_nan=False)


def test_undefined_correlation_and_phase_remain_explicit_and_json_safe():
    constant = fit_circular_conversion([0., 0., 0.], [1., 2., 3.])
    assert constant["rho"] is None
    assert constant["permutation_p"] is None
    uniform_sum = fit_circular_conversion([0., 90., 180., 270.], [0., 0., 0., 0.])
    assert uniform_sum["beta_deg"] is None
    assert uniform_sum["mae_deg"] is None
    assert uniform_sum["fitted_rf_ego_deg"] is None
    json.dumps(uniform_sum, allow_nan=False)


def test_invalid_pairs_are_not_silently_dropped():
    with pytest.raises(ValueError, match="finite"):
        fit_circular_conversion([0., 30., 60.], [2., np.nan, 40.])
    with pytest.raises(ValueError, match="three aligned"):
        fit_circular_conversion([0., 30.], [2., 40.])
