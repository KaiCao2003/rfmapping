"""Signed circular association and a fixed reverse-phase HD/RF conversion."""

import numpy as np


def _paired_radians(hd_deg, rf_ego_deg):
    hd = np.asarray(hd_deg, dtype=float)
    rf = np.asarray(rf_ego_deg, dtype=float)
    if hd.ndim != 1 or rf.shape != hd.shape or hd.size < 3:
        raise ValueError("At least three aligned one-dimensional angle pairs are required")
    if not np.isfinite(hd).all() or not np.isfinite(rf).all():
        raise ValueError("HD and RF angles must be finite")
    return np.deg2rad(hd % 360.), np.deg2rad(rf % 360.)


def _fisher_lee_components(hd, rf):
    pairs = np.triu_indices(hd.size, k=1)
    hd_sine = np.sin(hd[pairs[0]] - hd[pairs[1]])
    rf_sine_matrix = np.sin(rf[:, None] - rf[None, :])
    rf_sine = rf_sine_matrix[pairs]
    hd_norm, rf_norm = np.linalg.norm(hd_sine), np.linalg.norm(rf_sine)
    # Identical or exactly antipodal angles have no defined T-linear correlation.
    tolerance = 8 * np.finfo(float).eps * np.sqrt(hd_sine.size)
    if min(hd_norm, rf_norm) <= tolerance:
        return None, pairs, hd_sine, rf_sine_matrix, None
    denominator = hd_norm * rf_norm
    rho = float(np.clip(np.dot(hd_sine, rf_sine) / denominator, -1., 1.))
    return rho, pairs, hd_sine, rf_sine_matrix, denominator


def fisher_lee_correlation(hd_deg, rf_ego_deg):
    """Return signed Fisher–Lee circular correlation, or None if undefined.

    This uses pairwise sine differences, so no estimated mean direction is
    needed. A rotated reflection has rho=-1, including uniform angle samples.
    Fisher & Lee (1983), https://doi.org/10.1093/biomet/70.2.327;
    https://circstat.github.io/pycircstat2/reference/correlation/.
    """
    hd, rf = _paired_radians(hd_deg, rf_ego_deg)
    return _fisher_lee_components(hd, rf)[0]


def predict_circular_conversion(hd_deg, beta_deg):
    """Apply RFego=wrap180(beta-HD); return RFallo on [0, 360).

    The slope -1 is the declared coordinate conversion, not an estimated
    regression coefficient. Nonfinite headings remain NaN in both outputs.
    """
    hd = np.asarray(hd_deg, dtype=float)
    beta = np.nan if beta_deg is None else float(beta_deg)
    rf_ego = (beta - hd + 180.) % 360. - 180.
    rf_allo = (hd + rf_ego) % 360.
    return {"rf_ego_deg": rf_ego, "rf_allo_deg": rf_allo}


def fit_circular_conversion(hd_deg, rf_ego_deg, *, selected_on_sum=False,
                            n_permutations=10_000, random_seed=1):
    """Estimate the reverse-phase offset and independently test association.

    beta=circmean(HD+RFego) minimizes mean cosine loss for the fixed slope -1.
    MAE is an in-sample circular error, not a held-out prediction score.
    The two-sided permutation test shuffles unit pairings and compares |rho|.
    Outcome-selected samples (scheme 3) receive descriptive results only;
    testing those would require repeating their selection within each shuffle.
    All returned values are JSON serializable, including undefined statistics.
    """
    hd, rf = _paired_radians(hd_deg, rf_ego_deg)
    if n_permutations < 0 or int(n_permutations) != n_permutations:
        raise ValueError("n_permutations must be a nonnegative integer")
    rho, pairs, hd_sine, rf_sine_matrix, denominator = _fisher_lee_components(hd, rf)
    phase = np.mean(np.exp(1j * (hd + rf)))
    resultant = min(float(abs(phase)), 1.)
    beta = float(np.rad2deg(np.angle(phase)) % 360.) if resultant > 1e-12 else None
    prediction = predict_circular_conversion(hd_deg, beta)
    residual = (np.asarray(rf_ego_deg) - prediction["rf_ego_deg"] + 180.) % 360. - 180.
    mae = float(np.mean(abs(residual))) if beta is not None else None

    p_value = None
    performed = 0
    if not selected_on_sum and rho is not None and n_permutations:
        rng = np.random.default_rng(random_seed)
        exceedances = 0
        for start in range(0, int(n_permutations), 256):
            count = min(256, int(n_permutations) - start)
            order = rng.permuted(np.broadcast_to(np.arange(hd.size), (count, hd.size)), axis=1)
            shuffled_sine = rf_sine_matrix[order[:, pairs[0]], order[:, pairs[1]]]
            shuffled_rho = (shuffled_sine @ hd_sine) / denominator
            exceedances += int(np.sum(abs(shuffled_rho) >= abs(rho) - 1e-14))
        performed = int(n_permutations)
        p_value = (exceedances + 1.) / (performed + 1.)

    return {
        "method": "reversed_circular_phase",
        "conversion_slope": -1,
        "n": int(hd.size),
        "beta_deg": beta,
        "rho": rho,
        "correlation_method": "Fisher-Lee signed circular-circular",
        "permutation_p": p_value,
        "permutation_alternative": "two-sided absolute rho",
        "n_permutations": performed,
        "random_seed": int(random_seed),
        "mae_deg": mae,
        "fit_resultant_length": resultant,
        "selection_note": (
            "Selected on HD+RF: descriptive only; no permutation p-value"
            if selected_on_sum else "Cohort not selected on HD+RF association"
        ),
        "fitted_rf_ego_deg": prediction["rf_ego_deg"].tolist() if beta is not None else None,
        "observed_rf_allo_deg": ((np.asarray(hd_deg) + np.asarray(rf_ego_deg)) % 360.).tolist(),
    }
