import json

import numpy as np
import pytest

from Utils.hd_rf_circular_regression import (
    fisher_lee_correlation, fit_circular_conversion, predict_circular_conversion,
)


def circular_difference(first, second):
    return (np.asarray(first) - np.asarray(second) + 180.) % 360. - 180.


def test_uniform_reflection_has_negative_correlation_and_exact_prediction():
    hd = np.arange(0., 360., 30.)
    rf = (40. - hd + 180.) % 360. - 180.
    model = fit_circular_conversion(hd, rf, n_permutations=999)
    assert model["method"] == "first_harmonic_circular_regression"
    assert "beta_deg" not in model and "conversion_slope" not in model
    assert model["rho"] == pytest.approx(-1.)
    assert model["mae_deg"] == pytest.approx(0., abs=1e-12)
    assert model["permutation_p"] <= .01
    prediction = predict_circular_conversion(hd, model)
    np.testing.assert_allclose(circular_difference(prediction["rf_ego_deg"], rf), 0., atol=1e-12)
    np.testing.assert_allclose(prediction["rf_allo_deg"], 40., atol=1e-12)
    assert json.loads(json.dumps(model, allow_nan=False))["n"] == 12


def test_positive_relation_is_fit_exactly_without_fixing_the_sum():
    hd = np.array([3., 18., 49., 91., 144., 222., 301.])
    rf = hd + 37.
    model = fit_circular_conversion(hd, rf, n_permutations=99)
    assert model["rho"] == pytest.approx(1.)
    assert model["mae_deg"] == pytest.approx(0., abs=1e-12)
    prediction = predict_circular_conversion(hd, model)
    np.testing.assert_allclose(circular_difference(prediction["rf_ego_deg"], rf), 0., atol=1e-12)
    np.testing.assert_allclose(prediction["rf_allo_deg"], (2 * hd + 37.) % 360., atol=1e-12)
    assert np.ptp(prediction["rf_allo_deg"]) > 180.
    assert fisher_lee_correlation(hd + 178., rf - 191.) == pytest.approx(1.)
    assert fisher_lee_correlation(hd, -rf) == pytest.approx(-1.)


def test_nonlinear_relation_uses_both_components_and_allo_is_only_derived():
    hd = np.arange(0., 360., 15.)
    radians = np.deg2rad(hd)
    rf = np.rad2deg(np.arctan2(.45 * np.sin(radians) + .2,
                              .7 * np.cos(radians) + .4))
    model = fit_circular_conversion(hd, rf, n_permutations=0)
    prediction = predict_circular_conversion(hd, model)
    assert model["mae_deg"] < 15.
    assert np.ptp(prediction["rf_allo_deg"]) > 180.
    # Non-uniform angular increments distinguish this from a fixed +/-1 slope.
    steps = circular_difference(prediction["rf_ego_deg"][1:], prediction["rf_ego_deg"][:-1])
    assert np.ptp(steps) > 5.
    np.testing.assert_allclose(prediction["rf_allo_deg"],
                               (hd + prediction["rf_ego_deg"]) % 360.)


def test_prediction_crosses_seam_is_periodic_and_preserves_shape_and_gaps():
    hd = np.arange(0., 360., 30.)
    model = fit_circular_conversion(hd, 5. - hd, n_permutations=0)
    headings = np.array([[0., 360., 179., np.inf], [181., -179., np.nan, -np.inf]])
    prediction = predict_circular_conversion(headings, model)
    assert prediction["rf_ego_deg"].shape == headings.shape
    assert prediction["rf_allo_deg"].shape == headings.shape
    np.testing.assert_allclose(prediction["rf_ego_deg"][0, :3], [5., 5., -174.])
    np.testing.assert_allclose(prediction["rf_ego_deg"][1, :2], [-176., -176.])
    np.testing.assert_allclose(prediction["rf_allo_deg"][np.isfinite(headings)], 5., atol=1e-12)
    assert np.isnan(prediction["rf_ego_deg"][~np.isfinite(headings)]).all()
    assert np.isnan(prediction["rf_allo_deg"][~np.isfinite(headings)]).all()
    scalar = predict_circular_conversion(np.array(360.), model)
    assert scalar["rf_ego_deg"].shape == ()
    assert scalar["rf_ego_deg"] == pytest.approx(5.)
    empty = predict_circular_conversion(np.empty((0, 2)), model)
    assert empty["rf_ego_deg"].shape == (0, 2)


def test_zero_resultant_is_undefined_instead_of_an_arbitrary_direction():
    zero = {"cos_coefficients": [0., 0., 0.], "sin_coefficients": [0., 0., 0.]}
    prediction = predict_circular_conversion([0., 90., np.nan], zero)
    assert np.isnan(prediction["rf_ego_deg"]).all()
    assert np.isnan(prediction["rf_allo_deg"]).all()
    # Opposite responses at HD=0 cancel; predictions at the other HDs exist.
    model = fit_circular_conversion([0., 0., 120., 120., 240., 240.],
                                    [0., 180., 30., 30., 60., 60.], n_permutations=0)
    assert model["fitted_rf_ego_deg"][:2] == [None, None]
    np.testing.assert_allclose(model["fitted_rf_ego_deg"][2:], [30., 30., 60., 60.], atol=1e-12)
    assert model["mae_deg"] is None
    json.dumps(model, allow_nan=False)


def test_permutations_repeat_and_outcome_selection_has_no_p_value():
    hd = np.arange(0., 360., 45.)
    rf = 119. - hd + np.array([7., -8., 2., 15., -12., 4., 3., -6.])
    first = fit_circular_conversion(hd, rf, n_permutations=999, random_seed=14)
    second = fit_circular_conversion(hd, rf, n_permutations=999, random_seed=14)
    assert first == second
    selected = fit_circular_conversion(hd, rf, selected_on_sum=True)
    assert selected["rho"] == first["rho"]
    assert selected["cos_coefficients"] == first["cos_coefficients"]
    assert selected["sin_coefficients"] == first["sin_coefficients"]
    assert selected["permutation_p"] is None
    assert selected["n_permutations"] == 0
    json.dumps(selected, allow_nan=False)


def test_undefined_correlation_does_not_prevent_defined_rf_predictions():
    constant_hd = fit_circular_conversion([0., 0., 0.], [1., 2., 3.])
    assert constant_hd["rho"] is None
    assert constant_hd["permutation_p"] is None
    np.testing.assert_allclose(constant_hd["fitted_rf_ego_deg"], 2.)
    constant_rf = fit_circular_conversion([0., 90., 180., 270.], [0., 0., 0., 0.])
    assert constant_rf["rho"] is None
    assert constant_rf["mae_deg"] == pytest.approx(0., abs=1e-12)
    np.testing.assert_allclose(constant_rf["fitted_rf_ego_deg"], 0., atol=1e-12)
    json.dumps(constant_rf, allow_nan=False)


def test_invalid_pairs_are_not_silently_dropped():
    with pytest.raises(ValueError, match="finite"):
        fit_circular_conversion([0., 30., 60.], [2., np.nan, 40.])
    with pytest.raises(ValueError, match="three aligned"):
        fit_circular_conversion([0., 30.], [2., 40.])
