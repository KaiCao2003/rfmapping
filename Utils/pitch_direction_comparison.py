"""Signed pitch versus circular HD comparisons for the existing paired sessions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
from matplotlib import pyplot as plt
from scipy.stats import linregress

from Utils.json_tools import read_formatted_json
from Utils.kilosort_utils import _locate_spike_arrays
from Utils.plotting import LIGHT_PLOT_STYLE, plot_keyed_heatmap
from Utils.rfmap import load_rf_maps
from Utils.tuning_curve_utils import get_exposure_timestamps, make_head_direction_tsd, update_hd_classification


PAIRED_SESSIONS = (("m14", "260609", 1, "A"), ("m15", "260630", 1, "A"),
                   ("m15", "260630", 1, "B"), ("m19", "260827", 9, "A"),
                   ("m20", "260921", 9, "A"))


def cache_name(mouse, date, session, probe):
    return f"{mouse}_{date}_{session}_Probe{probe}_pitch_direction.npz"


def build_paired_tuning(recording_root, output_dir, mouse, date, session, probe):
    """Reuse saved HD classes and epochs; recompute both features on equal support."""
    session_dir = Path(recording_root) / mouse / str(date) / f"{date}_{session}"
    data_dir = session_dir / "data"
    source_tc = data_dir / "tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
    saved = json.loads(source_tc.read_text())
    update_hd_classification(saved)
    selected = np.asarray(saved["unit_data"]["hd_class"]) == 3
    unit_ids = np.asarray(saved["unit_id"])[selected]
    metadata = saved["metadata"]
    pitch_path = data_dir / "processed" / "pitch_direction.json"
    hd_path = data_dir / "processed" / "head_direction.json"
    pitch_data = json.loads(pitch_path.read_text())["hp4"]
    hd_data = json.loads(hd_path.read_text())["hp4"]
    frames = np.asarray(pitch_data["frames"], dtype=int)
    np.testing.assert_array_equal(frames, hd_data["frames"])
    pitch = np.asarray(pitch_data["pitch_direction_deg"], dtype=float)
    hd = np.asarray(hd_data["head_direction_deg"], dtype=float)
    session_info = read_formatted_json(data_dir / "session_info.json")["session_info"]
    times, origin, timing_qc = get_exposure_timestamps(session_info, data_dir)
    if len(frames) == len(times) + 1:
        frames, pitch, hd = frames[:-1], pitch[:-1], hd[:-1]
    if len(frames) != len(times):
        raise ValueError(f"{session_dir.name}: {len(frames)} pose frames vs {len(times)} exposure pulses")
    valid = np.isfinite(hd) & np.isfinite(pitch)
    hd_tsd, fs = make_head_direction_tsd(times, frames[valid], hd[valid])
    # Reuse the HD timing/gap contract, retaining the signed pitch values.
    pitch_tsd = nap.Tsd(t=hd_tsd.index.to_numpy(), d=pitch[valid], time_support=hd_tsd.time_support)
    epochs = nap.IntervalSet(np.asarray(metadata["epoch_intervals_s"]))
    support = epochs.intersect(hd_tsd.time_support)
    feature_times = hd_tsd.index.to_numpy()
    occupied = np.searchsorted(feature_times, support.end, side="right") > np.searchsorted(feature_times, support.start)
    support = support[occupied]
    spike_times = np.load(data_dir / f"probe{probe}" / "adc_spike_time.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(Path(metadata["kilosort_dir"]) / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    spikes = nap.TsGroup({uid: nap.Ts(t=values - origin) for uid, _, values in
                         _locate_spike_arrays(unit_ids, clusters, spike_times)})
    arrays = {"unit_id": unit_ids}
    for feature, tsd, limits in (("hd", hd_tsd, (0, 360)), ("pitch", pitch_tsd, (-90, 90))):
        counts = nap.compute_tuning_curves(data=spikes, features=tsd, bins=30, epochs=support,
                                          range=limits, return_counts=True, fs=fs)
        dim = next(d for d in counts.dims if d != "unit")
        counts = counts.transpose("unit", dim).sel(unit=unit_ids)
        arrays[f"{feature}_counts"] = counts.values
        arrays[f"{feature}_occupancy_s"] = np.asarray(counts.attrs["occupancy"]) / fs
        arrays[f"{feature}_edges"] = np.asarray(counts.attrs["bin_edges"][0])
    np.testing.assert_allclose(arrays["hd_occupancy_s"].sum(), arrays["pitch_occupancy_s"].sum())
    np.testing.assert_array_equal(arrays["hd_counts"].sum(axis=1), arrays["pitch_counts"].sum(axis=1))
    out = Path(output_dir) / cache_name(mouse, date, session, probe)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    report = {
        "mouse": mouse, "date": str(date), "session": session, "probe": probe,
        "units": len(unit_ids), "frames": int(valid.sum()), "feature_fs_hz": fs,
        "occupancy_s": float(arrays["pitch_occupancy_s"].sum()),
        "class_selection": "existing HD Class 3; no pitch classification", "class_source": str(source_tc),
        "hd_source": str(hd_path), "pitch_source": str(pitch_path), "timing": timing_qc,
        "epoch_intervals_s": metadata["epoch_intervals_s"], "valid_pose_intervals_s": support.values.tolist(),
        "output": str(out),
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_pair(output_dir, mouse, date, session, probe, *, min_occupancy_s=1.0):
    """Load equal-support curves; mask bins with inadequate sampling before peaks."""
    path = Path(output_dir) / cache_name(mouse, date, session, probe)
    with np.load(path) as data:
        index = pd.MultiIndex.from_tuples([(mouse, str(date), probe, int(uid)) for uid in data["unit_id"]],
                                         names=["mouse", "date", "probe", "unit_id"])
        curves = []
        for name in ("hd", "pitch"):
            occupancy = data[f"{name}_occupancy_s"]
            counts = data[f"{name}_counts"]
            rates = np.divide(counts, occupancy, out=np.full(counts.shape, np.nan), where=occupancy >= min_occupancy_s)
            edges = data[f"{name}_edges"]
            table = pd.DataFrame(rates, index=index, columns=(edges[:-1] + edges[1:]) / 2)
            table.attrs["label"] = f"{mouse} {date} Probe {probe} {name.upper()}"
            curves.append(table)
    return tuple(curves)


def preferred_angles(hd, pitch):
    def peak(table):
        values = table.to_numpy()
        peaks = table.columns.to_numpy()[np.argmax(np.where(np.isfinite(values), values, -np.inf), axis=1)]
        peaks[~np.any(np.isfinite(values) & (values > 0), axis=1)] = np.nan
        return peaks
    return pd.DataFrame({"hd_peak_deg": peak(hd), "pitch_peak_deg": peak(pitch)}, index=hd.index)


def plot_pair(hd, pitch, *, sort_by="hd", label="", figsize=(9, 8),
              cmap="viridis", vmin=None, vmax=None, colorbar_label="Response"):
    """Plot supplied curve values without circular pitch wrapping or rescaling."""
    peaks = preferred_angles(hd, pitch)
    order = peaks.sort_values(f"{sort_by}_peak_deg", kind="stable").index.tolist()
    figures = []
    for name, table, ticks in (("HD", hd, [0, 90, 180, 270, 360]), ("Pitch", pitch, [-90, -45, 0, 45, 90])):
        fig, ax = plot_keyed_heatmap(dict(zip(table.index, table.to_numpy())), order,
                                    column_order=np.arange(len(table.columns)), angle_centers=table.columns.to_numpy(),
                                    xticks=ticks, xticklabels=ticks, xlabel=f"{name} (°)",
                                    title=f"{label} | {name} | sorted by {sort_by} peak", figsize=figsize, show=False,
                                    cmap=cmap, vmin=vmin, vmax=vmax, colorbar_label=colorbar_label)
        # Retain full unit labels in returned peak table without overlapping 100s of ticks.
        step = max(1, len(order) // 20)
        ax.set_yticks(np.arange(0, len(order), step), labels=[
            f"{order[i][0]} {order[i][2]}:{order[i][3]}" for i in range(0, len(order), step)
        ])
        figures.append(fig)
    return peaks, figures


def circular_linear_association(hd_deg, pitch_deg, *, n_permutations=10000, seed=1):
    """Multiple R for pitch ~ 1 + sin(HD) + cos(HD), with unit-label permutations."""
    hd = np.asarray(hd_deg, dtype=float)
    pitch = np.asarray(pitch_deg, dtype=float)
    keep = np.isfinite(hd) & np.isfinite(pitch)
    hd, pitch = hd[keep], pitch[keep]
    result = {"n": len(hd), "r": np.nan, "p": np.nan}
    if len(hd) < 4 or np.ptp(pitch) == 0:
        return result
    design = np.column_stack((np.ones(len(hd)), np.sin(np.deg2rad(hd)), np.cos(np.deg2rad(hd))))
    pinv = np.linalg.pinv(design)
    centered = pitch - pitch.mean()
    variance = centered @ centered
    def r2(values):
        residual = values - design @ (pinv @ values)
        return max(0.0, 1 - (residual @ residual) / variance)
    observed = r2(pitch)
    rng = np.random.default_rng(seed)
    shuffled = np.asarray([r2(rng.permutation(pitch)) for _ in range(n_permutations)])
    result.update(r=float(np.sqrt(observed)), p=float((1 + np.sum(shuffled >= observed)) / (n_permutations + 1)))
    return result


def plot_peak_comparison(peaks, *, label=""):
    stats = circular_linear_association(peaks.hd_peak_deg, peaks.pitch_peak_deg)
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.scatter(peaks.hd_peak_deg, peaks.pitch_peak_deg, s=26, alpha=.75)
        ax.set(xlim=(0, 360), ylim=(-90, 90), xticks=[0, 90, 180, 270, 360], yticks=[-90, -45, 0, 45, 90],
               xlabel="Preferred HD (°; clockwise)", ylabel="Preferred pitch (°; up positive)",
               title=f"{label}\nCircular–linear R={stats['r']:.3f}, permutation p={stats['p']:.4g}, n={stats['n']}")
        ax.grid(alpha=.35)
    return stats, fig


def _linear_profile_peaks(table):
    values = table.to_numpy()
    peaks = table.columns.to_numpy(dtype=float)[
        np.argmax(np.where(np.isfinite(values), values, -np.inf), axis=1)
    ]
    peaks[~np.any(np.isfinite(values) & (values > 0), axis=1)] = np.nan
    return peaks


def rf_vertical_profiles(source, *, mouse, date, probe, window=(0.0, 0.2),
                         rf_type="2d", y_to_elevation_sign=-1):
    """Return native elevation profiles and the distinct 2-D maximum-bin elevation.

    Rates are occupancy-normalized before summing [start, stop) and horizontal
    positions. The stimulus PTB coordinate has positive Y downward, so elevation
    is -yPositions. No circular transform, smoothing, or baseline subtraction.
    """
    source = Path(source)
    maps = load_rf_maps(source, unit_firing_rate=True).sum(*window, show_progress=False)
    elevation = y_to_elevation_sign * maps[0].y_positions
    order = np.argsort(elevation)
    index = pd.MultiIndex.from_tuples([(mouse, str(date), probe, int(uid)) for uid in maps.unit_ids],
                                     names=["mouse", "date", "probe", "unit_id"])
    profiles = pd.DataFrame(maps.to_1d_array(axis="y")[:, order], index=index, columns=elevation[order])
    max_2d = []
    for rf_map in maps:
        matrix = rf_map.to_2d_array()[order]
        if not np.any(np.isfinite(matrix) & (matrix > 0)):
            max_2d.append(np.nan)
        else:
            row, _ = np.unravel_index(np.nanargmax(matrix), matrix.shape)
            max_2d.append(elevation[order[row]])
    peaks = pd.DataFrame({"rf_vertical_profile_peak_deg": _linear_profile_peaks(profiles),
                          "rf_2d_max_elevation_deg": max_2d}, index=index)
    if rf_type == "2d":
        with np.load(source.with_suffix(".npz"), allow_pickle=False) as result:
            detected = result["unit_ids"][np.any(result["mask_2d"], axis=(1, 2))]
        keep = index.get_level_values("unit_id").isin(detected)
        profiles, peaks = profiles.loc[keep], peaks.loc[keep]
    elif rf_type is not None:
        raise ValueError("rf_type must be '2d' or None")
    return profiles, peaks


def load_pitch_rf_pair(cache_dir, rf_source, *, mouse, date, pitch_session, probe,
                       min_occupancy_s=1.0, rf_type="2d", rf_peak_method="vertical_profile",
                       y_to_elevation_sign=-1, window=(0.0, 0.2)):
    """Match cached Class 3 pitch units to RF; native bins determine both peaks."""
    _, pitch = load_pair(cache_dir, mouse, date, pitch_session, probe, min_occupancy_s=min_occupancy_s)
    rf, peaks = rf_vertical_profiles(rf_source, mouse=mouse, date=date, probe=probe,
                                     window=window, rf_type=rf_type,
                                     y_to_elevation_sign=y_to_elevation_sign)
    shared = pitch.index.intersection(rf.index)
    pitch, rf, peaks = pitch.loc[shared], rf.loc[shared], peaks.loc[shared].copy()
    peaks["pitch_peak_deg"] = _linear_profile_peaks(pitch)
    choices = {"vertical_profile": "rf_vertical_profile_peak_deg", "max_bin_2d": "rf_2d_max_elevation_deg"}
    peaks["rf_peak_deg"] = peaks[choices[rf_peak_method]]
    # A maximum requires a positive measured response in both features.
    eligible = peaks[["pitch_peak_deg", "rf_peak_deg"]].notna().all(axis=1)
    pitch, rf, peaks = pitch.loc[eligible], rf.loc[eligible], peaks.loc[eligible]
    # Native y grids differ across mice (9x10° vs 7x12°). Only heatmap display
    # uses a common 1° grid; peaks and regressions above retain the native bins.
    grid = np.arange(-45., 46.)
    display_rf = pd.DataFrame([
        np.interp(grid, rf.columns.to_numpy(dtype=float), row, left=np.nan, right=np.nan)
        for row in rf.to_numpy()
    ], index=rf.index, columns=grid)
    return pitch, display_rf, peaks


def plot_pitch_rf_pair(pitch, rf, peaks, *, sort_by="pitch", label="", figsize=(9, 8),
                       cmap="viridis", vmin=None, vmax=None, colorbar_label="Response"):
    """Plot supplied values on bounded linear axes with one shared unit order."""
    if sort_by not in ("pitch", "rf"):
        raise ValueError("sort_by must be 'pitch' or 'rf'")
    order = peaks.sort_values(f"{sort_by}_peak_deg", kind="stable").index.tolist()
    figures = []
    for name, table, ticks in (("Pitch", pitch, [-90, -45, 0, 45, 90]),
                               ("RF elevation", rf, [-45, -30, -15, 0, 15, 30, 45])):
        fig, ax = plot_keyed_heatmap(dict(zip(table.index, table.to_numpy())), order,
                                    column_order=np.arange(len(table.columns)), angle_centers=table.columns.to_numpy(),
                                    xticks=ticks, xticklabels=ticks, xlabel=f"{name} (°; up positive)",
                                    title=f"{label}\n{name} | sorted by {sort_by} peak", figsize=figsize, show=False,
                                    cmap=cmap, vmin=vmin, vmax=vmax, colorbar_label=colorbar_label)
        step = max(1, len(order) // 20)
        ax.set_yticks(np.arange(0, len(order), step), labels=[
            f"{order[i][0]} {order[i][2]}:{order[i][3]}" for i in range(0, len(order), step)
        ])
        figures.append(fig)
    return figures


def pitch_rf_regression(peaks):
    """Ordinary linear regression of RF elevation on pitch; both are bounded."""
    finite = peaks[["pitch_peak_deg", "rf_peak_deg"]].dropna()
    result = dict(n=len(finite), slope=np.nan, intercept=np.nan, r=np.nan, r_squared=np.nan, p=np.nan)
    if len(finite) >= 3 and finite.pitch_peak_deg.nunique() > 1:
        fit = linregress(finite.pitch_peak_deg, finite.rf_peak_deg)
        result.update(slope=float(fit.slope), intercept=float(fit.intercept), r=float(fit.rvalue),
                      r_squared=float(fit.rvalue ** 2), p=float(fit.pvalue))
    return result


def plot_pitch_rf_regression(peaks, *, label="", rf_peak_method="vertical_profile"):
    stats = pitch_rf_regression(peaks)
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.scatter(peaks.pitch_peak_deg, peaks.rf_peak_deg, s=30, alpha=.8)
        if np.isfinite(stats["slope"]):
            x = np.array([peaks.pitch_peak_deg.min(), peaks.pitch_peak_deg.max()])
            ax.plot(x, stats["intercept"] + stats["slope"] * x, color="black", lw=1.5, label="OLS fit")
            ax.legend()
        method = "vertical profile peak" if rf_peak_method == "vertical_profile" else "2-D maximum-bin elevation"
        ax.set(xlim=(-90, 90), ylim=(-45, 45), xticks=[-90, -45, 0, 45, 90], yticks=[-40, -20, 0, 20, 40],
               xlabel="Preferred pitch (°; up positive)", ylabel=f"RF {method} (°; up positive)",
               title=f"{label}\nn={stats['n']}, slope={stats['slope']:.3f}, R²={stats['r_squared']:.3f}, p={stats['p']:.4g}")
        ax.grid(alpha=.35)
    return stats, fig
