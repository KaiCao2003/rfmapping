"""Pure, descriptive Gaussian fits sized by separately detected RF regions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ._detector import _label_clusters
from ._validation import _EDGE_ATOL_S, _bool_value
from .models import RFMap
from .results import RFResult, _rf_result_for_map

__all__ = ["fit_rf_gaussians"]

_COMPONENT_PARAMETERS = ("amplitude_hz", "x_deg", "y_deg", "sigma_x_deg", "sigma_y_deg")


def _detection_mask(result, response, polarity, wrap_x):
    mask = _rf_result_for_map(result, response).astype(bool)
    manifest = result.manifest
    if "time_range_s" in manifest and not np.allclose(
        manifest["time_range_s"], response.time_window_s, rtol=0, atol=_EDGE_ATOL_S,
    ):
        raise ValueError(f"{polarity} detection response window does not match the RFMap")
    if manifest.get("response_units", "Hz") != "Hz":
        raise ValueError(f"{polarity} detection response_units must be Hz")
    parameters = manifest.get("parameters", {})
    alternative = parameters.get("alternative", manifest.get("alternative"))
    expected = "greater" if polarity == "excitatory" else "less"
    if alternative is not None and alternative != expected:
        raise ValueError(f"{polarity} detection has the wrong response polarity")
    sigma = parameters.get("sigma", manifest.get("sigma"))
    if sigma is not None and ((float(sigma) > 0) != (polarity == "excitatory") or float(sigma) == 0):
        raise ValueError(f"{polarity} detection has the wrong sigma polarity")
    saved_wrap = parameters.get("wrap_x", manifest.get("wrap_x"))
    if saved_wrap is not None and saved_wrap != wrap_x:
        raise ValueError(f"{polarity} detection wrap_x does not match the requested connectivity")
    return mask


def _component_seed(values, component, x_grid, y_grid, baseline, sign, period, wrap_x):
    """Unwrap around a weighted medoid before estimating a seam-crossing width."""
    selected = component & np.isfinite(values)
    if not selected.any():
        raise ValueError("a detected component has no finite response bins for initialization")
    x, y = x_grid[selected], y_grid[selected]
    effect = np.maximum(sign * (values[selected] - baseline), 0)
    weights = effect if effect.any() else np.ones(effect.size)
    dx = x[:, None] - x[None, :]
    if wrap_x:
        dx = (dx + period / 2) % period - period / 2
    dy = y[:, None] - y[None, :]
    anchor = x[np.argmin((dx * dx + dy * dy) @ weights)]
    local_x = anchor + (x - anchor + period / 2) % period - period / 2 if wrap_x else x
    center_x = np.average(local_x, weights=weights)
    center_y = np.average(y, weights=weights)
    sigma_x = np.sqrt(np.average((local_x - center_x) ** 2, weights=weights))
    sigma_y = np.sqrt(np.average((y - center_y) ** 2, weights=weights))
    return [float(effect.max()), float(center_x), float(center_y), float(sigma_x), float(sigma_y)]


def fit_rf_gaussians(
    response: RFMap, *, excitatory: RFResult, inhibitory: RFResult, wrap_x: bool,
) -> dict[str, Any]:
    """Fit one signed Gaussian per connected region of each supplied RF mask.

    ``response`` is one uncollapsed, regular 2-D grid of mean firing rates in
    Hz with one response-window bin. Regions use the detector's four-neighbor
    connectivity and optional horizontal seam. Saved center markers do not
    determine component counts. The function neither detects nor loads/saves.

    All components and the free offset are jointly fitted to finite measured
    bins by unweighted least squares. Missing bins remain NaN in continuous
    outputs. Components are nonnegative magnitudes ordered excitatory first,
    then inhibitory; their descriptors carry the signs. ``net_excitatory_hz``
    and ``net_inhibitory_hz`` split the signed sum around the fitted offset,
    with no magnitude threshold. These descriptive fits establish neither
    statistical significance nor the number of physiological RF mechanisms.
    """
    if not isinstance(response, RFMap):
        raise TypeError("response must be an RFMap")
    if response.n_time_bins != 1 or response.duration_s <= 0:
        raise ValueError("response must contain exactly one positive-duration response window")
    if response.metadata.get("responseUnits") != "Hz":
        raise ValueError("response must contain mean firing rates in Hz")
    if min(response.n_y, response.n_x) < 2 or "spatialAggregation" in response.metadata:
        raise ValueError("response must be an uncollapsed 2-D spatial grid")
    wrap_x = _bool_value(wrap_x, "wrap_x")
    x, y = np.asarray(response.x_positions), np.asarray(response.y_positions)
    for name, positions in (("x", x), ("y", y)):
        steps = np.diff(positions)
        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=0, atol=1e-10):
            raise ValueError(f"response {name} positions must form an increasing regular grid")
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    period = dx * x.size
    x_lower, x_upper = float(x[0] - dx / 2), float(x[-1] + dx / 2)
    values = np.asarray(response.to_2d_array(), dtype=float)
    valid = np.isfinite(values)
    if np.isinf(values).any() or not valid.any() or np.any(values[valid] < 0):
        raise ValueError("response must contain finite nonnegative Hz values or missing NaN bins")
    observed = values[valid]
    baseline = float(np.median(observed))
    span = float(np.ptp(observed))
    maximum = float(observed.max())
    x_grid, y_grid = np.meshgrid(x, y)
    masks, labels, counts, descriptors, seeds = {}, {}, {}, [], []
    for polarity, result, sign in (("excitatory", excitatory, 1), ("inhibitory", inhibitory, -1)):
        mask = _detection_mask(result, response, polarity, wrap_x)
        labeled, masses = _label_clusters(np.ones(values.shape), mask, wrap_x=wrap_x)
        masks[polarity], labels[polarity], counts[polarity] = mask, labeled, len(masses)
        for label in range(1, len(masses) + 1):
            seed = _component_seed(values, labeled == label, x_grid, y_grid, baseline, sign, period, wrap_x)
            seed[0] = max(seed[0], 0.1 * span)
            if wrap_x:
                seed[1] = (seed[1] - x_lower) % period + x_lower
            seeds.append(seed)
            descriptors.append({"polarity": polarity, "sign": sign, "source_label": label})
    n_components = len(seeds)
    signs = np.asarray([item["sign"] for item in descriptors], dtype=float)
    lower = np.array([0.0] + [0.0, x_lower, float(y[0]), dx / 2, dy / 2] * n_components)
    upper = np.array([maximum] + [3 * span, x_upper, float(y[-1]), period / 4, dy * y.size] * n_components)
    initial = np.clip(np.array([baseline] + [value for seed in seeds for value in seed]), lower, upper)

    def evaluate(parameters, *, jacobian=False):
        fitted = np.full(values.shape, parameters[0])
        parts = np.empty((n_components, *values.shape))
        derivative = np.ones((valid.sum(), parameters.size)) if jacobian else None
        for index, sign in enumerate(signs):
            amplitude, center_x, center_y, sigma_x, sigma_y = parameters[1 + 5 * index:6 + 5 * index]
            offset_x = x_grid - center_x
            if wrap_x:
                offset_x = (offset_x + period / 2) % period - period / 2
                images = offset_x[None] + period * np.arange(-2, 3)[:, None, None]
            else:
                images = offset_x[None]
            gx_images = np.exp(-0.5 * (images / sigma_x) ** 2)
            gx = gx_images.sum(axis=0)
            offset_y = y_grid - center_y
            gy = np.exp(-0.5 * (offset_y / sigma_y) ** 2)
            shape = gx * gy
            parts[index] = amplitude * shape
            fitted += sign * parts[index]
            if jacobian:
                columns = (
                    shape,
                    amplitude * (gx_images * images / sigma_x ** 2).sum(axis=0) * gy,
                    parts[index] * offset_y / sigma_y ** 2,
                    amplitude * (gx_images * images ** 2 / sigma_x ** 3).sum(axis=0) * gy,
                    parts[index] * offset_y ** 2 / sigma_y ** 3,
                )
                derivative[:, 1 + 5 * index:6 + 5 * index] = sign * np.stack(columns, axis=-1)[valid]
        return (fitted, parts, derivative) if jacobian else (fitted, parts)

    starts = []
    if n_components == 0 or span == 0:
        parameters = initial.copy()
        parameters[0] = float(observed.mean())
        success, message = True, "Analytic constant fit"
    else:
        # Fixed bounds (for example a two-column grid's horizontal width) are
        # removed from the optimizer, without removing any detected component.
        free = upper > lower
        parameter_scale = np.array([max(maximum, 1)] + [max(span, 1), period / 6, dy * y.size / 3,
                                   period / 12, dy * y.size / 3] * n_components)
        fits = []
        for width_scale in (1.0, 1.5, 2.0):
            seed = initial.copy()
            seed[4::5] *= width_scale
            seed[5::5] *= width_scale
            seed = np.clip(seed, lower, upper)

            def expand(free_parameters):
                parameters = seed.copy()
                parameters[free] = free_parameters
                return parameters

            fit = least_squares(
                lambda p: evaluate(expand(p))[0][valid] - observed,
                seed[free], bounds=(lower[free], upper[free]), x_scale=parameter_scale[free],
                jac=lambda p: evaluate(expand(p), jacobian=True)[2][:, free],
                max_nfev=1500, ftol=1e-9, xtol=1e-9, gtol=1e-9,
            )
            fits.append((fit, expand(fit.x)))
            starts.append({"width_scale": width_scale, "sse_hz2": float(2 * fit.cost),
                           "success": bool(fit.success), "nfev": int(fit.nfev)})
        best, parameters = min(fits, key=lambda pair: pair[0].cost)
        success, message = bool(best.success), str(best.message)

    fitted, parts = evaluate(parameters)
    residual = values - fitted
    error = float(np.sum(residual[valid] ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    net = fitted - parameters[0]
    net_excitatory, net_inhibitory = np.maximum(net, 0), np.maximum(-net, 0)
    for array in (fitted, net_excitatory, net_inhibitory, residual):
        array[~valid] = np.nan
    parts[:, ~valid] = np.nan
    for index, descriptor in enumerate(descriptors):
        descriptor.update(zip(_COMPONENT_PARAMETERS, map(float, parameters[1 + 5 * index:6 + 5 * index])))
    parameter_names = ["fitted_offset_hz"] + [f"component_{index}.{name}"
        for index in range(n_components) for name in _COMPONENT_PARAMETERS]
    at_lower = np.isclose(parameters, lower, rtol=0, atol=1e-4)
    at_upper = np.isclose(parameters, upper, rtol=0, atol=1e-4)
    boundaries = [{"parameter": name, "value": float(value),
                   "bound": "fixed" if lower[index] == upper[index] else "lower" if at_lower[index] else "upper"}
                  for index, (name, value) in enumerate(zip(parameter_names, parameters))
                  if at_lower[index] or at_upper[index]]
    return {
        "unit_id": response.unit_id, "source_path": str(response.source_path),
        "time_range_s": list(response.time_window_s), "response_units": "Hz",
        "x_positions": x.copy(), "y_positions": y.copy(), "wrap_x": wrap_x,
        "x_period_deg": period if wrap_x else None, "valid_mask": valid.copy(),
        "fitted_response_hz": fitted, "net_excitatory_hz": net_excitatory,
        "net_inhibitory_hz": net_inhibitory, "residual_hz": residual,
        "component_maps_hz": parts, "components": descriptors, "component_counts": counts,
        "detection_masks": masks, "detection_labels": labels,
        "detection_cache_keys": {"excitatory": excitatory.cache_key, "inhibitory": inhibitory.cache_key},
        "fitted_offset_hz": float(parameters[0]), "parameters": parameters,
        "initial_parameters": initial, "parameter_names": parameter_names,
        "parameter_bounds": {"lower": lower, "upper": upper},
        "success": success, "message": message, "starts": starts,
        "training_rmse_hz": float(np.sqrt(error / observed.size)),
        "training_r_squared": 1 - error / total if total > 0 else float("nan"),
        "boundary_limited_parameters": boundaries,
    }
