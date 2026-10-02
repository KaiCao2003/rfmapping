"""Session geometry and RF elevation / EBC distance comparisons."""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.stats import false_discovery_control, rankdata

import spatial_cell_analysis as spatial
from Utils.json_tools import read_formatted_json
from Utils.rfmap import load_rf_maps
from Utils.tuning_curve_utils import get_exposure_timestamps, make_head_direction_tsd, tuning_curve


BANDS12 = (("0-8", None, 8), ("8-16", 8, 16), ("16-", 16, None))
THETA_DEG = np.arange(0., 360., 6.)
# Basler bounds in pixels: left, right, top, bottom.
boundary_old = (370., 920., 210., 760.)
boundary_new = (225., 720., 280., 785.)
BASLER_BOUNDS_PX = boundary_old
BASLER_SIZE_CM = 41.


def frame_counts(spike_times, spike_clusters, good_ids, times):
    selected = (spike_times >= times[0]) & (spike_times <= times[-1])
    spikes, clusters = spike_times[selected], spike_clusters[selected]
    units = np.intersect1d(good_ids, np.unique(clusters))
    selected = np.isin(clusters, units)
    unit_indices = np.searchsorted(units, clusters[selected])
    frames = spatial.nearest_frame_indices(spikes[selected], times)
    counts = np.bincount(
        unit_indices * len(times) + frames, minlength=len(units) * len(times),
    ).reshape(len(units), len(times)).astype(np.int32)
    return units, counts


def load_basler_session(session, *, bounds_px, size_cm, probe="A", phase="baseline"):
    """Load Basler position in cm using explicit left/right/top/bottom pixel bounds."""
    pose, times, spikes, clusters, good, source = spatial.load_data(
        session_dir=session, probe=probe, phase=phase, basler_output=True, optihub2_output=False,
    )
    units, counts = frame_counts(spikes, clusters, good, times)
    # Basler image Y points down; use Cartesian cm and north-zero CCW HD.
    left, right, top, bottom = bounds_px
    xy = np.c_[(pose.center_x - left) * size_cm / (right - left),
               (bottom - pose.center_y) * size_cm / (bottom - top)]
    return dict(session=Path(session), probe=probe, times=times, xy=xy,
                frame_ids=pose.frame.to_numpy(dtype=int),
                hd=pose.hd_deg.to_numpy() % 360, unit_ids=units, counts=counts, source=source,
                dt=float(np.median(np.diff(times))), phase=phase)


def rebuild_basler_hd(data, *, shuffles=1000, seed=0):
    session, probe = data["session"], data["probe"]
    path = session / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
    session_info = read_formatted_json(session / "data/session_info.json")["session_info"]
    exposures, _, _ = get_exposure_timestamps(
        session_info, session / "data", camera_ttl_active_high=False,
    )
    hd_tsd, feature_fs_hz = make_head_direction_tsd(exposures, data["frame_ids"], data["hd"])
    tuning_curve(
        base_dir=session, kilosort_dir=Path(data["source"]["kilosort_dir"]), probe_name=probe,
        interval_pairs=np.array([data["source"]["selected_interval_s"]]),
        HD_tsd=hd_tsd,
        adc_time_origin_s=data["source"]["adc_time_origin_s"], num_of_bins_in_hd=180,
        num_shuffle=shuffles, shuffle_seed=seed, is_save=True, save_path=path,
        feature_fs_hz=feature_fs_hz,
        timestamp_reference=data["source"]["camera_timing"]["timestamp_reference"],
        metadata={"epoch": data["phase"], "pose_source": data["source"]["pose_path"],
                  "ttl_qc": data["source"]["camera_timing"], "frame_mapping": "zero_based_csv_frame"},
    )
    return path


def load_motive_session(session, *, probe="A", phase="baseline"):
    session = Path(session)
    directory = session / "data"
    pose = pd.read_csv(directory / "processed/filtered.csv", header=[0, 1, 2, 3])
    xy = pose.xs("Position", level=2, axis=1).to_numpy(float)[:, :2]
    frames = pose.iloc[:, 0].to_numpy(int)
    heading = read_formatted_json(directory / "processed/head_direction.json")["hp4"]
    hd = pd.Series(heading["head_direction_deg"], index=heading["frames"]).reindex(frames).to_numpy()
    info = read_formatted_json(directory / "session_info.json")["session_info"]
    exposures, origin, timing = get_exposure_timestamps(session_info=info, data_dir=directory, camera_ttl_active_high=True)
    times = exposures[frames]
    intervals = pd.read_csv(directory / "interval_table.csv")
    interval = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0].to_numpy(float)
    selected = (times >= interval[0]) & (times <= interval[1]) & np.isfinite(xy).all(axis=1) & np.isfinite(hd)
    xy, hd, times = xy[selected], hd[selected] % 360, times[selected]
    center = (xy.min(axis=0) + xy.max(axis=0)) / 2
    radius = np.linalg.norm(xy - center, axis=1).max()
    scale = 15 / radius
    kilosort = next((session / "kilosort" / f"Probe{probe}").glob("kilosort_*"))
    spikes = np.load(directory / f"probe{probe}/adc_spike_time.npy", mmap_mode="r").ravel() - origin
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").ravel()
    units, counts = frame_counts(spikes, clusters, spatial.read_good_unit_ids(kilosort), times)
    return dict(session=session, probe=probe, times=times, xy=(xy - center) * scale,
                hd=hd, unit_ids=units, counts=counts, dt=float(np.median(np.diff(times))),
                phase=phase, interval=interval, center_raw=center, cm_per_unit=scale,
                source={"camera_timing": timing, "position_path": str(directory / "processed/filtered.csv")})


def circle_distances(xy, hd_deg, radius_cm, theta_deg=THETA_DEG):
    """Forward intersections in Motive XY; saved HD is world Z yaw, CCW from +X."""
    angle = np.deg2rad(hd_deg[:, None] + theta_deg)
    projection = xy[:, 0, None] * np.cos(angle) + xy[:, 1, None] * np.sin(angle)
    discriminant = projection**2 + radius_cm**2 - np.sum(xy**2, axis=1)[:, None]
    # At the enclosing circle, roundoff can put a boundary distance just below 0.
    return np.maximum(-projection + np.sqrt(np.maximum(discriminant, 0)), 0)


def rectangle_distances(xy, hd_deg, theta_deg=THETA_DEG, size_cm=BASLER_SIZE_CM):
    return spatial.d(theta_deg, xy[:, 0], size_cm - xy[:, 1], hd_deg,
                     bounds=(0, size_cm, 0, size_cm))


def ray_histogram_matrix(distances, distance_edges):
    """Sparse angle/distance-by-frame projection, shared by units and controls."""
    n_frames, n_angles = distances.shape
    n_dist = len(distance_edges) - 1
    # Right-closed distance bins put exactly 30 cm in the outer (0, 30] band.
    bins = np.maximum(np.searchsorted(distance_edges, distances, side="left") - 1, 0)
    valid = (distances >= distance_edges[0]) & (distances <= distance_edges[-1])
    frame, angle = np.nonzero(valid)
    rows = angle * n_dist + bins[frame, angle]
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, frame)), shape=(n_angles * n_dist, n_frames))


def rates_from_counts(counts, occupancy, *, sigma=5, mask_unvisited=True):
    smoothed = gaussian_filter(np.asarray(counts, dtype=float), (0, sigma, sigma), mode=("nearest", "wrap", "nearest"))
    occ = gaussian_filter(occupancy, sigma, mode=("wrap", "nearest"))
    rates = np.divide(smoothed, occ, out=np.full_like(smoothed, np.nan), where=occ > 0)
    if mask_unvisited:
        rates[:, occupancy == 0] = np.nan
    return rates


def compute_circle_ebc(data, *, sigma=5):
    results = {}
    for name, radius, bands in (("inner", 15., ()), ("outer", 30., (("0-30", None, 30), ("30-60", 30, 60)))):
        distances = circle_distances(data["xy"], data["hd"], radius)
        edges = np.arange(0., 2 * radius + .1, 1.5)
        shape = (len(THETA_DEG), len(edges) - 1)
        projection = ray_histogram_matrix(distances, edges)
        occupancy = (np.asarray(projection.sum(axis=1)).ravel() * data["dt"]).reshape(shape)
        spike_maps = (projection @ data["counts"].T).T.reshape(-1, *shape)
        rates = rates_from_counts(spike_maps, occupancy, sigma=sigma)
        maps = {"distance_edges": edges, "theta_edges": np.arange(0, 361, 6)}
        metadata = {"boundaryType": "circle", "boundaryDiameterCm": 2 * radius,
                    "boundaryCenterRaw": data["center_raw"].tolist(), "cmPerPositionUnit": data["cm_per_unit"],
                    "positionSource": data["source"]["position_path"],
                    "angleConvention": "egocentric CCW; Motive XY heading is Z yaw from +X",
                    "smoothingSigmaBins": sigma, "unvisitedBins": "masked_after_smoothing"}
        path = data["session"] / "data/spatial_cells" / f"Probe{data['probe']}" / data["phase"] / name / "egocentric_rate_map.rfmap"
        spatial.save_egocentric_rfmap(path, rates, maps, data["unit_ids"].tolist(), data["interval"].tolist(), metadata=metadata)
        paths = {"full": path}
        for suffix, lower, upper in bands:
            target = path.with_name(f"{path.stem}_{suffix}.rfmap")
            spatial.save_egocentric_rfmap(target, rates, maps, data["unit_ids"].tolist(), data["interval"].tolist(),
                                         distance_band_cm=(lower, upper), metadata=metadata)
            paths[suffix] = target
        results[name] = dict(paths=paths, occupancy=occupancy, distance_edges=edges,
                             actual_distance_range=(float(distances.min()), float(distances.max())))
        print(name, "distance range (cm):", results[name]["actual_distance_range"], "units:", len(data["unit_ids"]))
    return results


def plot_circle_coverage(data, results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    axes[0].plot(*data["xy"][::10].T, color="0.6", lw=.4)
    for radius, color in ((15, "tab:blue"), (30, "tab:orange")):
        axes[0].add_patch(Circle((0, 0), radius, fill=False, color=color, label=f"Diameter {2 * radius} cm"))
    axes[0].set(xlim=(-32, 32), ylim=(-32, 32), xlabel="Motive X (cm)", ylabel="Motive Y (cm)", title="Session 9 trajectory")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    for ax, name in zip(axes[1:], ("inner", "outer")):
        item = results[name]
        image = ax.pcolormesh(item["distance_edges"], np.arange(0, 361, 6),
                              np.ma.masked_equal(item["occupancy"], 0), shading="flat")
        ax.set(xlabel="Boundary distance (cm)", ylabel="Egocentric bearing (°)", title=f"{name}: occupancy")
        fig.colorbar(image, ax=ax, label="Seconds")
    return fig


def band_mask(centers, lower, upper):
    selected = np.ones(len(centers), dtype=bool)
    if lower is not None:
        selected &= centers > lower
    if upper is not None:
        selected &= centers <= upper
    return selected


def ebc_distance_peaks(path, *, probe="A", bands=BANDS12):
    maps = load_rf_maps(path)
    values_by_unit = maps.to_2d_array()
    rows = []
    for name, lower, upper in (*bands, ("all", None, None)):
        selected = band_mask(maps[0].x_positions, lower, upper)
        distances = maps[0].x_positions[selected]
        for unit, values in zip(maps.unit_ids, values_by_unit[:, :, selected]):
            if not np.any(np.isfinite(values) & (values > 0)):
                continue
            angle, distance = np.unravel_index(np.nanargmax(values), values.shape)
            rows.append((probe, unit, name, distances[distance], maps[0].y_positions[angle], values[angle, distance]))
    return pd.DataFrame(rows, columns=["probe", "unit_id", "band", "d_peak_cm", "bearing_deg", "peak_response_hz"])


def row_spearman(x, y):
    """Last-axis rank correlation; constant ranks have undefined correlation."""
    xr, yr = rankdata(x, axis=-1), rankdata(y, axis=-1)
    xr = xr - xr.mean(axis=-1, keepdims=True)
    yr = yr - yr.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(xr*xr, axis=-1) * np.sum(yr*yr, axis=-1))
    return np.divide(np.sum(xr*yr, axis=-1), norm, out=np.full(np.shape(norm), np.nan), where=norm > 0)


def bootstrap_bounds(draws, *, limit=1.):
    """Conservative percentile bounds when discrete resamples have zero variance."""
    finite = np.isfinite(draws)
    if not finite.any():
        return np.array([np.nan, np.nan]), 0.
    # Undefined resamples can span the full parameter range; dropping them can
    # misleadingly exclude zero when almost every unit has the same distance.
    bounds = [np.quantile(np.where(finite, draws, -limit), .025),
              np.quantile(np.where(finite, draws, limit), .975)]
    return np.asarray(bounds), finite.mean()


def distance_correlations(centers, peaks, *, permutations=10_000, bootstraps=2000, seed=1):
    distances = peaks.pivot(index=["probe", "unit_id"], columns="band", values="d_peak_cm")
    data = centers.join(distances, how="inner").dropna(subset=[b[0] for b in BANDS12])
    rng = np.random.default_rng(seed)
    x = data.rf_y_deg.to_numpy()
    samples = rng.integers(0, len(data), size=(bootstraps, len(data)))
    x_ranks = rankdata(x)
    permutations_indices = np.array([rng.permutation(len(x)) for _ in range(permutations)])
    permuted_ranks = x_ranks[permutations_indices]
    permuted_ranks -= permuted_ranks.mean(axis=-1, keepdims=True)
    permuted_squares = np.sum(permuted_ranks * permuted_ranks, axis=-1)
    rows, bootstrap = [], {}
    for band in [b[0] for b in BANDS12] + ["all"]:
        y = data[band].to_numpy()
        rho = float(row_spearman(x, y))
        y_ranks = rankdata(y)
        y_ranks -= y_ranks.mean()
        norm = np.sqrt(permuted_squares * np.sum(y_ranks * y_ranks))
        null = np.divide(
            np.sum(permuted_ranks * y_ranks, axis=-1), norm,
            out=np.full(permutations, np.nan), where=norm > 0,
        )
        draws = row_spearman(x[samples], y[samples])
        bootstrap[band] = draws
        ci, valid_fraction = bootstrap_bounds(draws)
        p = (1 + np.count_nonzero(np.abs(null) >= abs(rho))) / (permutations + 1) if np.isfinite(rho) else np.nan
        rows.append((band, len(data), rho, p, *ci, valid_fraction, len(np.unique(y))))
    stats = pd.DataFrame(rows, columns=["band", "n", "rho", "p", "ci_low", "ci_high",
                                       "bootstrap_defined_fraction", "unique_distances"]).set_index("band")
    stats["q"] = np.nan
    valid = stats.loc[[b[0] for b in BANDS12], "p"].dropna()
    stats.loc[valid.index, "q"] = false_discovery_control(valid.to_numpy())
    differences = []
    for first, second in combinations([b[0] for b in BANDS12], 2):
        delta = bootstrap[first] - bootstrap[second]
        ci, valid_fraction = bootstrap_bounds(delta, limit=2.)
        differences.append((first, second, stats.loc[first, "rho"] - stats.loc[second, "rho"], *ci, valid_fraction))
    differences = pd.DataFrame(differences, columns=["first", "second", "rho_difference", "ci_low", "ci_high", "bootstrap_defined_fraction"])
    return data, stats, differences


def plot_distance_correlations(data, stats):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for ax, band in zip(axes, [b[0] for b in BANDS12] + ["all"]):
        groups = data.groupby(["rf_y_deg", band]).size()
        x, y = np.array(groups.index.tolist()).T
        ax.scatter(x, y, alpha=.7, s=28 * groups)
        for (x, y), count in groups.items():
            if count > 1:
                ax.annotate(str(count), (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
        median = data.groupby("rf_y_deg")[band].median()
        ax.plot(median.index, median, color="0.35", ls=":", label="Median at each RF y")
        s = stats.loc[band]
        ax.set(xlabel="RF center y (°)", ylabel="EBC peak distance (cm)",
               title=f"{band} cm | n={int(s['n'])}\nSpearman ρ={s.rho:.2f}, p={s.p:.3g}")
        ax.legend(fontsize=7)
    return fig
