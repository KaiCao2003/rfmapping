"""Across-neuron circular HD-preference to RF-azimuth association.

Predictions evaluated on behavioral HD are a model-derived hypothesis. They
are not measurements of a neuron's RF changing within a behavioral recording.
The fit uses native saved classifications through the existing HD loader.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.direction_comparison import (
    hd_pick, load_hd_profiles, recording_profiles, resample_profiles, rf_pick, rf_profiles,
    select_rf_profiles, tcRange,
)
from Utils.json_tools import read_formatted_json
from Utils.rflocate import load_rf, load_rfmap, rf_result_path
from Utils.statistic_utils import fisher_lee_correlation, fisher_lee_statistics


DEFAULT_KAPPAS = (0., .125, .25, .5, 1., 2., 4., 8., 16., 32., 64.)
METHOD_REFERENCE = "https://doi.org/10.1111/j.1467-9469.2012.00809.x"


def wrap_deg(angle):
    """Signed shortest angles, in [-180, 180)."""
    return (np.asarray(angle, dtype=float) + 180.) % 360. - 180.


def _kernel_prediction(hd, rf, query, kappa, *, leave_out=False):
    difference = np.deg2rad(np.asarray(query).reshape(-1, 1) - hd)
    weights = np.exp(kappa * (np.cos(difference) - 1.))
    if leave_out:
        np.fill_diagonal(weights, 0.)
    total = weights.sum(axis=1)
    mean = weights @ np.exp(1j * np.deg2rad(rf)) / total
    angles = np.rad2deg(np.angle(mean))
    angles[np.abs(mean) < 1e-12] = np.nan  # Opposite responses have no mean direction.
    return angles, np.abs(mean), total ** 2 / np.sum(weights ** 2, axis=1)


def circular_metrics(observed, predicted):
    error = wrap_deg(np.asarray(predicted) - np.asarray(observed))
    valid = np.isfinite(error)
    return dict(n=int(valid.sum()), total_n=int(error.size),
                mean_absolute_error_deg=float(np.mean(abs(error[valid]))),
                circular_loss=float(np.mean(1. - np.cos(np.deg2rad(error[valid])))))


def _select_kappa(hd, rf, kappas):
    predictions = [_kernel_prediction(hd, rf, hd, k, leave_out=True)[0] for k in kappas]
    # An undefined prediction is maximally penalized, rather than discarded.
    losses = [float(np.mean(np.where(np.isfinite(p),
                                    1. - np.cos(np.deg2rad(p - rf)), 2.)))
              for p in predictions]
    selected = int(np.argmin(losses))
    return float(kappas[selected]), predictions[selected], losses


def fit_hd_rf_model(hd_deg, rf_ego_deg, *, kappas=DEFAULT_KAPPAS):
    """Fit a periodic circular-response kernel with nested leave-one-neuron-out QC.

    f(h) = arg(sum_i exp(kappa * (cos(h-H_i)-1)) * exp(1j*RF_i)).
    Inner leave-one-out circular loss chooses concentration. Outer held-out
    neurons evaluate the entire bandwidth-selection procedure independently.
    """
    hd, rf = np.asarray(hd_deg, dtype=float), np.asarray(rf_ego_deg, dtype=float)
    kappas = np.asarray(kappas, dtype=float)
    if hd.ndim != 1 or hd.shape != rf.shape or len(hd) < 4 or not np.isfinite([hd, rf]).all():
        raise ValueError("Use at least four paired finite one-dimensional HD/RF angles.")
    if kappas.ndim != 1 or not len(kappas) or not np.isfinite(kappas).all() or np.any(kappas < 0):
        raise ValueError("Kernel concentrations must be finite and nonnegative.")
    hd, rf = hd % 360., wrap_deg(rf)
    kappa, loo, losses = _select_kappa(hd, rf, kappas)
    nested, nested_kappa = [], []
    for i in range(len(hd)):
        keep = np.arange(len(hd)) != i
        selected, _, _ = _select_kappa(hd[keep], rf[keep], kappas)
        predicted = _kernel_prediction(hd[keep], rf[keep], hd[i:i + 1], selected)[0]
        nested.append(float(predicted[0]))
        nested_kappa.append(selected)
    rf_vectors = np.exp(1j * np.deg2rad(rf))
    allo_vectors = np.exp(1j * np.deg2rad(hd + rf))
    constant_rf = float(np.rad2deg(np.angle(rf_vectors.sum())))
    beta = float(np.rad2deg(np.angle(allo_vectors.sum())) % 360.)
    constant_loo = np.rad2deg(np.angle(rf_vectors.sum() - rf_vectors))
    fixed_loo = wrap_deg(np.rad2deg(np.angle(allo_vectors.sum() - allo_vectors)) - hd)
    fitted = _kernel_prediction(hd, rf, hd, kappa)[0]
    nested_metrics = circular_metrics(rf, nested)
    return dict(
        method="von_mises_kernel_circular_response", schema_version=1,
        reference=METHOD_REFERENCE, kappa=kappa,
        hd_preferred_deg=hd.tolist(), rf_ego_deg=rf.tolist(),
        angle_convention="HD and RF horizontal angles positive clockwise; RFallo=wrap360(HD+RFego)",
        coordinate_note="HD and RF horizontal angles positive clockwise; RFallo=wrap360(HD+RFego). World zero belongs to the fitted HD source session.",
        interpretation="Across-neuron association evaluated on behavior; not measured within-neuron RF motion.",
        bandwidth_selection="minimum leave-one-neuron-out mean 1-cos(error); lower kappa wins exact ties",
        kappa_candidates=kappas.tolist(), candidate_loo_circular_loss=losses,
        training_metrics=circular_metrics(rf, fitted),
        selected_kappa_loo_metrics=circular_metrics(rf, loo),
        nested_loo_metrics=nested_metrics,
        validation=dict(method="nested leave-one-neuron-out", circular_mae_deg=nested_metrics["mean_absolute_error_deg"],
                        circular_loss=nested_metrics["circular_loss"], n=nested_metrics["n"]),
        loo_rf_ego_deg=np.asarray(loo).tolist(), nested_loo_rf_ego_deg=nested,
        nested_loo_kappa=nested_kappa,
        constant_rf=dict(rf_ego_deg=constant_rf, loo_metrics=circular_metrics(rf, constant_loo)),
        fixed_allocentric=dict(beta_deg=beta, resultant_length=float(abs(allo_vectors.mean())),
                               loo_metrics=circular_metrics(rf, fixed_loo),
                               equation="RFego=wrap180(beta-HD); RFallo=beta", forced=False),
    )


def predict_hd_rf(model, hd_deg):
    """Predict RF angles and descriptive support measures at arbitrary HD values.

    Resultant length is local response agreement, not a confidence interval.
    Effective sample size and nearest-HD distance reveal sparsely sampled HDs.
    Invalid behavioral HD stays NaN in every output.
    """
    query = np.asarray(hd_deg, dtype=float)
    flat = query.reshape(-1)
    valid = np.isfinite(flat)
    keys = ("rf_ego_deg", "rf_allo_deg", "resultant_length", "effective_sample_size",
            "nearest_training_hd_distance_deg", "fixed_allocentric_rf_ego_deg")
    result = {key: np.full(flat.shape, np.nan) for key in keys}
    hd, rf = np.asarray(model["hd_preferred_deg"]), np.asarray(model["rf_ego_deg"])
    predicted, concentration, effective = _kernel_prediction(hd, rf, flat[valid], model["kappa"])
    result["rf_ego_deg"][valid] = wrap_deg(predicted)
    result["rf_allo_deg"][valid] = (flat[valid] + predicted) % 360.
    result["resultant_length"][valid] = concentration
    result["effective_sample_size"][valid] = effective
    result["nearest_training_hd_distance_deg"][valid] = np.min(
        abs(wrap_deg(flat[valid, None] - hd)), axis=1,
    )
    result["fixed_allocentric_rf_ego_deg"][valid] = wrap_deg(model["fixed_allocentric"]["beta_deg"] - flat[valid])
    return {key: value.reshape(query.shape) for key, value in result.items()}


def _json_hd_profiles(hd_path, heading_path, unit_ids, cache_path):
    """Recompute only directional counts/rates; retain the saved Class 3 cohort."""
    import pynapple as nap
    from Utils.tuning_curve_utils import get_exposure_timestamps, make_head_direction_tsd

    session = Path(hd_path).parents[3]
    metadata = read_formatted_json(hd_path)["metadata"]
    heading = read_formatted_json(heading_path)[metadata.get("headplate", "hp4")]
    frames = np.asarray(heading["frames"], dtype=int)
    angles = np.asarray(heading["hd"] if "hd" in heading else heading["head_direction_deg"], dtype=float)
    qc = metadata["ttl_qc"]
    exposures, origin, timing = get_exposure_timestamps(
        read_formatted_json(session / "data/session_info.json")["session_info"], session / "data",
        camera_input_channel=qc["camera_input_channel"],
        camera_ttl_active_high=qc["camera_ttl_active_high"],
    )
    synced = (frames >= 0) & (frames < len(exposures))
    feature, fs = make_head_direction_tsd(exposures, frames[synced], angles[synced])
    epochs = nap.IntervalSet(np.asarray(metadata["epoch_intervals_s"])).intersect(feature.time_support)
    kilosort = Path(metadata["kilosort_dir"])
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    probe = metadata["probe"]
    times = np.load(session / "data" / f"probe{probe}/adc_spike_time.npy", mmap_mode="r").reshape(-1)
    spikes = nap.TsGroup({int(unit): nap.Ts(t=np.asarray(times[clusters == unit]) - origin)
                         for unit in unit_ids})
    curves = nap.compute_tuning_curves(data=spikes, features=feature, bins=30, epochs=epochs,
                                      range=(0., 360.), return_counts=True, fs=fs)
    curves = curves.transpose("unit", *[dim for dim in curves.dims if dim != "unit"])
    counts = np.asarray(curves.values, dtype=float)
    occupancy = np.asarray(curves.attrs["occupancy"]).reshape(-1) / fs
    rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan), where=occupancy > 0)
    ids = curves.coords["unit"].values.astype(int)
    profiles = pd.DataFrame(rates, index=ids, columns=np.arange(6., 360., 12.))
    provenance = dict(hd_heading_source=str(heading_path),
                      hd_heading_sha256=hashlib.sha256(Path(heading_path).read_bytes()).hexdigest(),
                      hd_profile_method="Pynapple tuning counts / occupancy from current JSON; 30 bins; no smoothing",
                      classification_source="Cohort selected by existing update_hd_classification on saved TC; no new classification or shuffles",
                      dropped_unsynchronized_frames=int((~synced).sum()), camera_timing=timing,
                      heading_profile_sourcecache=str(cache_path) if cache_path is not None else None)
    if cache_path is not None:
        cache = dict(metadata=provenance, unit_id=ids.tolist(),
                     angle_bin_edges_deg=np.arange(0., 361., 12.).tolist(),
                     spike_counts=counts.astype(int).tolist(), occupancy_time_s=occupancy.tolist(),
                     firing_rate_hz=np.where(np.isfinite(rates), rates, None).tolist(),
                     epoch_intervals_s=metadata["epoch_intervals_s"], feature_fs_hz=fs)
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2, allow_nan=False) + "\n")
    return profiles, provenance


def load_hd_rf_pairs(hd_path, rf_path, *, mouse="m19", date=260827, probe="A",
                     hd_heading_source=None, heading_profile_cache=None):
    """Match the paired notebook's Class 3, detected-2D RF, and bin-quality selection."""
    recording = dict(mouse=mouse, date=date, probe=probe)
    hd = load_hd_profiles(hd_path, probe=probe, bins=30, smoothing_deg=0)
    hd = resample_profiles(hd, range=tcRange(False), bins=30)
    hd = recording_profiles(hd, mouse=mouse, date=date, label="HD")
    hd = hd_pick(hd, hd_class=3)
    heading_provenance = {}
    if hd_heading_source is not None:
        profiles, heading_provenance = _json_hd_profiles(
            hd_path, hd_heading_source, hd.index.get_level_values("unit_id"), heading_profile_cache,
        )
        hd.loc[:, :] = profiles.loc[hd.index.get_level_values("unit_id")].to_numpy()
    raw_maps = load_rfmap(rf_path)
    count_maps = raw_maps.sum(0., .2, show_progress=False).sum_to_1d(axis="x")
    counts = rf_profiles(count_maps, probe=probe)
    counts = select_rf_profiles(counts, load_rf(rf_result_path(rf_path)))
    detected = recording_profiles(counts, mouse=mouse, date=date)
    count_profiles = rf_pick(detected, max_zero_bins=2)
    count_profiles = resample_profiles(count_profiles, range=tcRange(True), bins=30, fill_value=0)
    rate_maps = raw_maps.to_firing_rate(reconstruct_presentations=True).mean_rate(0., .2, show_progress=False).sum_to_1d(axis="x")
    rates = rf_profiles(rate_maps, probe=probe)
    rf = recording_profiles(
        resample_profiles(rates, range=tcRange(True), bins=30, fill_value=0), mouse=mouse, date=date,
    ).loc[count_profiles.index]
    shared = hd.index.intersection(rf.index)
    rows = []
    for key in shared:
        h, r = hd.loc[key], rf.loc[key]
        if not (np.isfinite(h).any() and np.isfinite(r).any() and h.max() > h.min() and r.max() > r.min()):
            continue
        rows.append(dict(zip(hd.index.names, key), hd_preferred_deg=float(h.idxmax() % 360.),
                         rf_ego_deg=float(wrap_deg(r.idxmax())),
                         rf_count_peak_deg=float(wrap_deg(count_profiles.loc[key].idxmax())),
                         hd_peak_ties=int((h == h.max()).sum()), rf_peak_ties=int((r == r.max()).sum())))
    pairs = pd.DataFrame(rows)
    provenance = dict(recording=recording, hd_path=str(hd_path), rf_path=str(rf_path),
                      hd_source=str(hd_path), rf_source=str(rf_path), selected_units=pairs.unit_id.tolist(),
                      hd_class=3, hd_bins=30, hd_smoothing_deg=0, rf_window_s=[0., .2],
                      rf_detection="excitatory 2d", rf_smoothing_bins=0,
                      rf_max_zero_bins=2,
                      peak_method="argmax on the paired notebook's 30-bin profiles; first source-order maximum for ties",
                      rf_projection="presentation-exposure-normalized response-window Hz, sum over elevation, then periodic interpolation to 30 bins",
                      rf_response_normalization="load_rfmap().to_firing_rate(reconstruct_presentations=True).mean_rate(0, 0.2).sum_to_1d(axis='x').to_1d_array(axis='x')",
                      rf_peaks_changed_by_rate_normalization=int(np.count_nonzero(
                          wrap_deg(pairs.rf_ego_deg - pairs.rf_count_peak_deg))),
                      hd_class3_units=len(hd), rf_detected_units=len(detected),
                      rf_quality_units=len(rf), shared_units=len(shared), finite_nonflat_pairs=len(pairs),
                      sources_sha256={str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                      for path in (hd_path, rf_path, Path(rf_path).with_suffix(".npz"))})
    provenance.update(heading_provenance)
    return pairs, provenance


def save_fit_artifacts(output, pairs, model):
    """Save the portable model, source pairs, and a dense prediction grid."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output / "training_pairs.csv", index=False)
    (output / "model.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    grid = np.arange(0., 360.01, .25)
    predictions = predict_hd_rf(model, grid)
    pd.DataFrame(dict(hd_deg=grid, **predictions)).to_csv(output / "prediction_grid.csv", index=False)
    plot_fit_diagnostics(output / "fit_diagnostics.png", pairs, model, grid, predictions)
    return output


def plot_fit_diagnostics(path, pairs, model, grid, predictions):
    """Show the unconstrained fit alongside the fixed-allocentric comparison."""
    from matplotlib import pyplot as plt
    from Utils.plotting import LIGHT_PLOT_STYLE

    def circular_line(ax, x, y, **style):
        plotted = np.asarray(y).copy()
        plotted[np.r_[False, abs(np.diff(plotted)) > 180.]] = np.nan
        ax.plot(x, plotted, **style)

    style = LIGHT_PLOT_STYLE | {"figure.facecolor": "white", "axes.facecolor": "white",
                               "savefig.facecolor": "white", "savefig.transparent": False,
                               "text.color": "#202934", "axes.labelcolor": "#202934",
                               "xtick.color": "#202934", "ytick.color": "#202934"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        beta = model["fixed_allocentric"]["beta_deg"]
        circular_line(axes[0], grid, predictions["rf_ego_deg"], color="#1667b1", lw=2,
                      label="Flexible circular fit")
        circular_line(axes[0], grid, predictions["fixed_allocentric_rf_ego_deg"],
                      color="#888888", lw=1.5, ls="--", label="Fixed-allo comparison")
        axes[0].scatter(pairs.hd_preferred_deg, pairs.rf_ego_deg, s=42, color="#16374e", zorder=3)
        previous = None
        for (hd, rf), group in pairs.groupby(["hd_preferred_deg", "rf_ego_deg"], sort=True):
            nearby = previous is not None and hd - previous[0] < 20 and abs(rf - previous[1]) < 20
            right_edge = hd > 340
            axes[0].annotate("/".join(map(str, group.unit_id)), (hd, rf),
                             xytext=(-4 if right_edge else 4, -14 if nearby else 4),
                             ha="right" if right_edge else "left",
                             textcoords="offset points", fontsize=7)
            previous = hd, rf
        axes[0].set(ylabel="RF egocentric horizontal angle (deg)", ylim=(-180, 180),
                    yticks=np.arange(-180, 181, 90), title="Measured pairs and circular prediction")
        circular_line(axes[1], grid, predictions["rf_allo_deg"], color="#1667b1", lw=2,
                      label="HD + predicted RF ego")
        axes[1].scatter(pairs.hd_preferred_deg, (pairs.hd_preferred_deg + pairs.rf_ego_deg) % 360.,
                        s=42, color="#16374e", zorder=3)
        axes[1].axhline(beta, color="#888888", ls="--", lw=1.5, label=f"Fixed allo: {beta:.1f}°")
        axes[1].set(ylabel="RF allocentric angle (deg)", ylim=(0, 360), yticks=np.arange(0, 361, 90),
                    title="Allocentric interpretation")
        for ax in axes:
            ax.set(xlim=(0, 360), xticks=np.arange(0, 361, 90), xlabel="HD preferred direction / behavioral HD (deg)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#e3e7ed", lw=.6)
            ax.legend(fontsize=8, frameon=True, facecolor="white", framealpha=1.)
        fig.suptitle(f"m19 · {len(pairs)} Class 3 + detected RF pairs · periodic circular regression",
                     x=.065, ha="left", fontsize=15, fontweight="bold")
        kernel_mae = model["validation"]["circular_mae_deg"]
        fixed_mae = model["fixed_allocentric"]["loo_metrics"]["mean_absolute_error_deg"]
        constant_mae = model["constant_rf"]["loo_metrics"]["mean_absolute_error_deg"]
        note = (f"Kernel κ={model['kappa']:g}; nested leave-one-neuron-out MAE {kernel_mae:.1f}°.  "
                f"Fixed-allo LOO MAE {fixed_mae:.1f}°; constant-RF LOO MAE {constant_mae:.1f}°.\n"
                "Across-neuron association evaluated on behavioral HD; not measured within-neuron RF motion.\n"
                "RF: 0–200 ms response-window Hz summed over elevation; existing paired-analysis bin quality filters.")
        fig.text(.065, .03, note, fontsize=8.5, color="#344454")
        fig.subplots_adjust(left=.065, right=.98, top=.85, bottom=.25, wspace=.24)
        fig.savefig(path, dpi=180, facecolor="white", transparent=False)
        plt.close(fig)


def predict_circular_conversion(hd_deg, model):
    """Predict RFego from HD; derive RFallo=wrap360(HD+RFego).

    Coefficients predict the cosine and sine of RFego on the basis
    [1, cos(HD), sin(HD)]. Shapes and gaps in HD are preserved. A zero
    predicted vector has no direction and stays NaN in both outputs.
    """
    hd = np.asarray(hd_deg, dtype=float)
    finite = np.isfinite(hd)
    radians = np.deg2rad(np.where(finite, hd, np.nan) % 360.)
    design = np.stack((np.ones(hd.shape), np.cos(radians), np.sin(radians)), axis=-1)
    predicted_cos = design @ np.asarray(model["cos_coefficients"], dtype=float)
    predicted_sin = design @ np.asarray(model["sin_coefficients"], dtype=float)
    # Unit-vector responses can cancel to zero up to trigonometric/OLS roundoff.
    defined = finite & (np.hypot(predicted_cos, predicted_sin) > 8 * np.finfo(float).eps)
    angle = (np.rad2deg(np.arctan2(predicted_sin, predicted_cos)) + 180.) % 360. - 180.
    rf_ego = np.where(defined, angle, np.nan)
    rf_allo = (np.where(finite, hd, np.nan) % 360. + rf_ego) % 360.
    return {"rf_ego_deg": rf_ego, "rf_allo_deg": rf_allo}


def fit_circular_conversion(hd_deg, rf_ego_deg, *, selected_on_sum=False,
                            n_permutations=10_000, random_seed=1):
    """Fit RFego directly from HD and independently test association.

    Separate least squares fits of cos(RFego) and sin(RFego) use the basis
    [1, cos(HD), sin(HD)]; atan2 gives their predicted direction. Neither
    the direction of association nor a constant HD+RFego sum is imposed.
    This is the order-1 trigonometric circular-circular regression described
    at https://search.r-project.org/CRAN/refmans/CircStats/html/circ.reg.html.
    MAE is an in-sample circular error, not a held-out prediction score.
    The two-sided permutation test shuffles unit pairings and compares |rho|.
    Outcome-selected samples (scheme 3) receive descriptive results only;
    testing those would require repeating their selection within each shuffle.
    All returned values are JSON serializable, including undefined statistics.
    """
    statistics = fisher_lee_statistics(
        hd_deg, rf_ego_deg, n_permutations=n_permutations,
        random_seed=random_seed, test=not selected_on_sum,
    )
    hd = np.deg2rad(np.asarray(hd_deg, dtype=float) % 360.)
    rf = np.deg2rad(np.asarray(rf_ego_deg, dtype=float) % 360.)
    design = np.column_stack((np.ones(hd.size), np.cos(hd), np.sin(hd)))
    coefficients = np.linalg.lstsq(design, np.column_stack((np.cos(rf), np.sin(rf))), rcond=None)[0]
    model = {
        "method": "first_harmonic_circular_regression",
        "cos_coefficients": coefficients[:, 0].tolist(),
        "sin_coefficients": coefficients[:, 1].tolist(),
    }
    prediction = predict_circular_conversion(hd_deg, model)
    residual = (np.asarray(rf_ego_deg) - prediction["rf_ego_deg"] + 180.) % 360. - 180.
    mae = float(np.mean(abs(residual))) if np.isfinite(residual).all() else None

    return {
        **model,
        "n": int(hd.size),
        **statistics,
        "correlation_method": "Fisher-Lee signed circular-circular",
        "permutation_alternative": "two-sided absolute rho",
        "random_seed": int(random_seed),
        "mae_deg": mae,
        "selection_note": (
            "Selected on HD+RF: descriptive only; no permutation p-value"
            if selected_on_sum else "Cohort not selected on HD+RF association"
        ),
        "fitted_rf_ego_deg": [float(value) if np.isfinite(value) else None
                              for value in prediction["rf_ego_deg"]],
        "observed_rf_allo_deg": ((np.asarray(hd_deg) + np.asarray(rf_ego_deg)) % 360.).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/senzailab/Kai/#Recording"))
    parser.add_argument("--output", type=Path, default=Path("output/hd_rf_prediction/m19_260827"))
    parser.add_argument("--hd-file", type=Path)
    parser.add_argument("--rf-file", type=Path)
    parser.add_argument("--heading-json", type=Path,
                        help="Recompute HD profiles from this JSON while preserving the saved Class 3 cohort")
    args = parser.parse_args(argv)
    recording = args.root / "m19/260827"
    hd = args.hd_file or recording / "260827_9/data/tuning_curves/ProbeA/tuning_curves.tc"
    rf = args.rf_file or recording / "260827_2/data/rfmapping/good/-100_400_1ms/ProbeA/regular_unitsSpikeCounts_260827_2.rfmap"
    pairs, provenance = load_hd_rf_pairs(
        hd, rf, hd_heading_source=args.heading_json,
        heading_profile_cache=args.output / "json_hd_profiles.json" if args.heading_json else None,
    )
    model = fit_hd_rf_model(pairs.hd_preferred_deg, pairs.rf_ego_deg)
    model["provenance"] = provenance
    save_fit_artifacts(args.output, pairs, model)
    print(json.dumps({key: model[key] for key in ("kappa", "nested_loo_metrics", "fixed_allocentric")}, indent=2))
    return model


if __name__ == "__main__":
    main()
