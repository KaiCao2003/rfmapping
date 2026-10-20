"""Gaussian fits use saved RF sectors without changing native observations."""

from dataclasses import replace

import numpy as np
import pytest

from RF.rflocate import RFResult, asrfmap
from RF.rflocate.fitting import fit_rf_gaussians


X_POSITIONS = np.arange(-170., 180., 20.)
Y_POSITIONS = np.arange(-36., 37., 12.)
GRID_SHAPE = (Y_POSITIONS.size, X_POSITIONS.size)


def _gaussian(amplitude, x_center, y_center, x_width, y_width):
    x, y = np.meshgrid(X_POSITIONS, Y_POSITIONS)
    return amplitude * np.exp(
        -.5 * (((x - x_center) / x_width) ** 2
               + ((y - y_center) / y_width) ** 2)
    )


def _response(values, *, unit_id=7):
    return replace(
        asrfmap(values, end_time=.2), unit_id=unit_id,
        x_positions=X_POSITIONS.copy(), y_positions=Y_POSITIONS.copy(),
        metadata={"responseUnits": "Hz",
                  "responseNormalization": "presentation_count_time"},
    )


def _result(response, mask, polarity, *, unit_ids=None, wrap_x=False):
    masks = np.asarray(mask, dtype=np.uint8)
    if masks.ndim == 2:
        masks = masks[np.newaxis]
    centers = np.zeros_like(masks)
    for row, unit_mask in enumerate(masks):
        occupied = np.flatnonzero(unit_mask)
        if occupied.size:
            # A valid legacy result may have one center for several sectors.
            centers[row].flat[occupied[0]] = 1
    return RFResult(
        mask_2d=masks, center_2d=centers,
        unit_ids=np.array([response.unit_id] if unit_ids is None else unit_ids),
        manifest={
            "source_path": str(response.source_path),
            "grid_shape": list(GRID_SHAPE),
            "storage_shape": list(masks.shape),
            "x_positions": response.x_positions.tolist(),
            "y_positions": response.y_positions.tolist(),
            "time_range_s": list(response.time_window_s),
            "response_units": "Hz",
            "parameters": {
                "alternative": "greater" if polarity == "excitatory" else "less",
                "wrap_x": wrap_x,
            },
        },
        cache_key=f"fixture-{polarity}",
    )


def _two_peaks(*, negative):
    values = 6 + _gaussian(5, -70, -12, 22, 16) + _gaussian(4, 90, 12, 24, 18)
    excitatory = np.zeros(GRID_SHAPE, dtype=np.uint8)
    excitatory[1:4, 4:7] = 1
    excitatory[3:6, 12:15] = 1
    inhibitory = np.zeros_like(excitatory)
    if negative:
        values -= _gaussian(2, 10, 12, 25, 15)
        inhibitory[3:6, 8:11] = 1
    response = _response(values)
    return response, _result(response, excitatory, "excitatory"), _result(
        response, inhibitory, "inhibitory",
    )


@pytest.mark.parametrize("negative", [False, True])
def test_component_counts_follow_final_sectors_not_legacy_centers(negative):
    response, excitatory, inhibitory = _two_peaks(negative=negative)
    assert excitatory.center_2d.sum() == 1
    fitted = fit_rf_gaussians(
        response, excitatory=excitatory, inhibitory=inhibitory, wrap_x=False,
    )
    assert fitted["component_counts"] == {"excitatory": 2, "inhibitory": int(negative)}
    components = fitted["components"]
    assert [item["sign"] for item in components] == [1, 1] + ([-1] if negative else [])
    positive_centers = sorted(
        (item["x_deg"], item["y_deg"])
        for item in components if item["polarity"] == "excitatory"
    )
    np.testing.assert_allclose(positive_centers, [[-70, -12], [90, 12]], atol=1.)
    assert fitted["training_rmse_hz"] < .02


def test_sectors_use_four_connectivity_and_only_horizontal_wrap():
    response = _response(np.full(GRID_SHAPE, 6.))
    mask = np.zeros(GRID_SHAPE, dtype=np.uint8)
    mask[2, [0, -1]] = 1
    mask[4, 7] = mask[5, 8] = 1  # Diagonal contact does not merge sectors.
    mask[0, 12] = mask[-1, 12] = 1
    for wrap_x, expected_count in ((False, 6), (True, 5)):
        fitted = fit_rf_gaussians(
            response,
            excitatory=_result(response, mask, "excitatory", wrap_x=wrap_x),
            inhibitory=_result(response, np.zeros_like(mask), "inhibitory", wrap_x=wrap_x),
            wrap_x=wrap_x,
        )
        assert fitted["component_counts"] == {"excitatory": expected_count, "inhibitory": 0}
        labels = fitted["detection_labels"]["excitatory"]
        assert bool(labels[2, 0] == labels[2, -1]) == wrap_x
        assert labels[4, 7] != labels[5, 8]
        assert labels[0, 12] != labels[-1, 12]


def test_wrapped_seam_gaussian_recovers_one_center_and_native_response():
    x, y = np.meshgrid(X_POSITIONS, Y_POSITIONS)
    center_x = 178.
    offset_x = (x - center_x + 180) % 360 - 180
    wrapped_x = sum(np.exp(-.5 * ((offset_x + 360 * shift) / 24.) ** 2)
                    for shift in (-2, -1, 0, 1, 2))
    values = 4 + 7 * wrapped_x * np.exp(-.5 * (y / 15.) ** 2)
    response = _response(values)
    mask = (np.abs(offset_x) < 35) & (np.abs(y) < 18)
    fitted = fit_rf_gaussians(
        response, excitatory=_result(response, mask, "excitatory", wrap_x=True),
        inhibitory=_result(response, np.zeros_like(mask), "inhibitory", wrap_x=True),
        wrap_x=True,
    )
    assert fitted["component_counts"] == {"excitatory": 1, "inhibitory": 0}
    component = fitted["components"][0]
    center_error = (component["x_deg"] - center_x + 180) % 360 - 180
    assert abs(center_error) < .1
    np.testing.assert_allclose(
        [component["amplitude_hz"], component["y_deg"], component["sigma_x_deg"], component["sigma_y_deg"]],
        [7, 0, 24, 15], atol=.1,
    )
    np.testing.assert_allclose(fitted["fitted_response_hz"], values, atol=.01)
    assert fitted["fitted_offset_hz"] == pytest.approx(4, abs=.01)
    assert fitted["success"]


def test_no_detected_sectors_fits_only_the_observed_mean():
    values = np.linspace(0., 12., np.prod(GRID_SHAPE)).reshape(GRID_SHAPE)
    values[-1, -3:] = np.nan
    valid = np.isfinite(values)
    response = _response(values)
    empty = np.zeros(GRID_SHAPE, dtype=np.uint8)
    fitted = fit_rf_gaussians(
        response, excitatory=_result(response, empty, "excitatory"),
        inhibitory=_result(response, empty, "inhibitory"), wrap_x=False,
    )
    assert fitted["component_counts"] == {"excitatory": 0, "inhibitory": 0}
    assert fitted["components"] == []
    assert fitted["component_maps_hz"].shape == (0, *GRID_SHAPE)
    assert fitted["fitted_offset_hz"] == pytest.approx(np.nanmean(values))
    np.testing.assert_allclose(fitted["fitted_response_hz"][valid], np.nanmean(values))
    np.testing.assert_array_equal(fitted["net_excitatory_hz"][valid], 0)
    np.testing.assert_array_equal(fitted["net_inhibitory_hz"][valid], 0)
    for key in ("fitted_response_hz", "net_excitatory_hz", "net_inhibitory_hz", "residual_hz"):
        np.testing.assert_array_equal(np.isfinite(fitted[key]), valid)
    assert fitted["training_rmse_hz"] == pytest.approx(np.nanstd(values))


def test_inhibitory_only_sector_does_not_add_a_positive_component():
    values = 6 - _gaussian(2, 10, 12, 25, 15)
    response = _response(values)
    empty = np.zeros(GRID_SHAPE, dtype=np.uint8)
    inhibitory_mask = empty.copy()
    inhibitory_mask[3:6, 8:11] = 1
    fitted = fit_rf_gaussians(
        response, excitatory=_result(response, empty, "excitatory"),
        inhibitory=_result(response, inhibitory_mask, "inhibitory"), wrap_x=False,
    )
    assert fitted["component_counts"] == {"excitatory": 0, "inhibitory": 1}
    assert [item["sign"] for item in fitted["components"]] == [-1]
    center = fitted["components"][0]
    np.testing.assert_allclose([center["x_deg"], center["y_deg"]], [10, 12], atol=1.)
    assert fitted["training_rmse_hz"] < .02


def test_continuous_signed_outputs_reconstruct_fit_without_mutating_inputs():
    response, excitatory, inhibitory = _two_peaks(negative=True)
    before_response = response.spike_counts.copy()
    before_masks = [item.mask_2d.copy() for item in (excitatory, inhibitory)]
    before_centers = [item.center_2d.copy() for item in (excitatory, inhibitory)]
    before_manifests = [dict(item.manifest) for item in (excitatory, inhibitory)]
    fitted = fit_rf_gaussians(
        response, excitatory=excitatory, inhibitory=inhibitory, wrap_x=False,
    )
    prediction = fitted["fitted_response_hz"]
    positive = fitted["net_excitatory_hz"]
    negative = fitted["net_inhibitory_hz"]
    parts = fitted["component_maps_hz"]
    signs = np.array([item["sign"] for item in fitted["components"]])
    assert prediction.shape == positive.shape == negative.shape == GRID_SHAPE
    assert parts.shape == (3, *GRID_SHAPE)
    assert np.all(parts >= 0)
    np.testing.assert_allclose(prediction, fitted["fitted_offset_hz"] + positive - negative)
    np.testing.assert_allclose(prediction, fitted["fitted_offset_hz"] + (parts * signs[:, None, None]).sum(axis=0))
    np.testing.assert_allclose(fitted["residual_hz"], response.to_2d_array() - prediction)
    assert not np.any((positive > 0) & (negative > 0))
    assert np.any((positive > 0) & (excitatory.mask_2d[0] == 0))
    assert np.unique(positive).size > 10  # Native continuous values, not a mask.
    np.testing.assert_array_equal(response.spike_counts, before_response)
    assert response._rf_result_cache == {}
    for index, result in enumerate((excitatory, inhibitory)):
        np.testing.assert_array_equal(result.mask_2d, before_masks[index])
        np.testing.assert_array_equal(result.center_2d, before_centers[index])
        assert result.manifest == before_manifests[index]


def test_each_detection_selects_the_requested_unit_id_despite_different_row_orders():
    response, other_exc, other_inh = _two_peaks(negative=True)
    empty = np.zeros(GRID_SHAPE, dtype=np.uint8)
    excitatory = _result(
        response, np.stack([other_exc.mask_2d[0], empty]), "excitatory", unit_ids=[99, 7],
    )
    inhibitory = _result(
        response, np.stack([empty, other_inh.mask_2d[0]]), "inhibitory", unit_ids=[7, 99],
    )
    fitted = fit_rf_gaussians(
        response, excitatory=excitatory, inhibitory=inhibitory, wrap_x=False,
    )
    assert fitted["unit_id"] == 7
    assert fitted["component_counts"] == {"excitatory": 0, "inhibitory": 0}
    assert fitted["fitted_offset_hz"] == pytest.approx(response.to_2d_array().mean())


def test_missing_bins_are_excluded_from_loss_and_preserved_as_missing():
    values = 6 + _gaussian(5, -70, -12, 22, 16)
    mask = np.zeros(GRID_SHAPE, dtype=np.uint8)
    mask[1:4, 4:7] = 1
    values[2, 5] = np.nan
    values[-1, -3:] = np.nan
    response = _response(values)
    fitted = fit_rf_gaussians(
        response, excitatory=_result(response, mask, "excitatory"),
        inhibitory=_result(response, np.zeros_like(mask), "inhibitory"), wrap_x=False,
    )
    valid = np.isfinite(values)
    np.testing.assert_array_equal(fitted["valid_mask"], valid)
    for key in ("fitted_response_hz", "net_excitatory_hz", "net_inhibitory_hz", "residual_hz"):
        np.testing.assert_array_equal(np.isfinite(fitted[key]), valid)
    np.testing.assert_array_equal(response.to_2d_array(), values)
    np.testing.assert_array_equal(np.isfinite(fitted["component_maps_hz"][0]), valid)
    np.testing.assert_allclose(fitted["fitted_response_hz"][valid], values[valid], atol=.02)


@pytest.mark.parametrize("mismatch", ["window", "polarity", "grid"])
def test_fitting_rejects_detection_from_another_window_polarity_or_grid(mismatch):
    response, excitatory, inhibitory = _two_peaks(negative=False)
    manifest = dict(excitatory.manifest)
    if mismatch == "window":
        manifest["time_range_s"] = [.1, .3]
    elif mismatch == "polarity":
        manifest["parameters"] = dict(manifest["parameters"], alternative="less")
    else:
        manifest["x_positions"] = (response.x_positions + 1).tolist()
    wrong = replace(excitatory, manifest=manifest)
    with pytest.raises(ValueError):
        fit_rf_gaussians(response, excitatory=wrong, inhibitory=inhibitory, wrap_x=False)


def test_fitting_requires_explicit_time_reduction():
    response, excitatory, inhibitory = _two_peaks(negative=False)
    multiple_times = replace(
        response, spike_counts=np.repeat(response.spike_counts, 2, axis=2),
        time_bin_edges_s=np.array([0., .1, .2]),
    )
    with pytest.raises(ValueError):
        fit_rf_gaussians(multiple_times, excitatory=excitatory, inhibitory=inhibitory, wrap_x=False)
