"""Pure center-bearing, center-distance and head-direction analysis.

Methods follow Long et al., Nature Communications 16, 356 (2025),
doi:10.1038/s41467-024-54699-9. The shuffled reference population here is
the supplied session's eligible units, rather than the paper's entire cohort.
Nothing in this module reads, generates or saves recording files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d

from Utils.theta_analysis import phase_locking, phase_precession, theta_rhythmicity


@dataclass(frozen=True)
class CenterCellConfig:
    """Paper parameters, plus explicit handling of incomplete camera tracking."""

    speed_threshold_cm_s: float = 2.5
    min_spikes_exclusive: int = 100
    min_spatial_coverage_exclusive: float = 0.80
    position_smoothing_s: float = 0.400
    spatial_bin_cm: float = 2.5
    angle_bin_deg: float = 3.0
    angle_smoothing_deg: float = 15.0
    distance_bin_cm: float = 2.0
    distance_stability_smoothing_cm: float = 6.0
    shuffle_min_shift_s: float = 20.0
    percentile: float = 99.0
    max_gap_factor: float = 1.5
    center_tolerance_cm: float = 1e-9
    min_correlation_bins: int = 3


STATISTICS = ("hd_mvl", "hd_stability", "cb_mvl", "cb_stability", "cd_r2", "cd_stability")


def _config(config):
    if config is None:
        config = CenterCellConfig()
    elif isinstance(config, Mapping):
        config = CenterCellConfig(**config)
    if not isinstance(config, CenterCellConfig):
        raise TypeError("config must be CenterCellConfig or a mapping of its fields")
    positive = (config.spatial_bin_cm, config.angle_bin_deg, config.angle_smoothing_deg,
                config.distance_bin_cm, config.distance_stability_smoothing_cm,
                config.shuffle_min_shift_s, config.max_gap_factor)
    if not np.isfinite(positive).all() or min(positive) <= 0:
        raise ValueError("Bin widths, smoothing widths, gap factor and minimum shift must be positive")
    if config.max_gap_factor < 1 or config.min_correlation_bins < 3:
        raise ValueError("max_gap_factor must be >= 1 and min_correlation_bins >= 3")
    if not 0 <= config.min_spatial_coverage_exclusive <= 1 or not 0 < config.percentile < 100:
        raise ValueError("Coverage must be within [0, 1] and percentile within (0, 100)")
    if (not np.isfinite([config.position_smoothing_s, config.speed_threshold_cm_s,
                         config.center_tolerance_cm]).all()
            or min(config.position_smoothing_s, config.speed_threshold_cm_s,
                   config.center_tolerance_cm, config.min_spikes_exclusive) < 0):
        raise ValueError("Smoothing duration, speed threshold, tolerance and spike limit must be nonnegative")
    if not np.isclose(360 / config.angle_bin_deg, round(360 / config.angle_bin_deg)):
        raise ValueError("angle_bin_deg must divide 360")
    _kernel_size(config.angle_smoothing_deg, config.angle_bin_deg)
    _kernel_size(config.distance_stability_smoothing_cm, config.distance_bin_cm)
    return config


def _kernel_size(width, bin_width):
    size = int(round(width / bin_width))
    if size < 1 or size % 2 != 1 or not np.isclose(size * bin_width, width):
        raise ValueError("Smoothing width must be an odd integer number of bins")
    return size


def _segments(valid, times, max_gap):
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(times[indices]) > max_gap)) + 1
    return np.split(indices, cuts)


def _smooth_positions(times, xy, valid, width_s, max_gap):
    result = np.full_like(xy, np.nan, dtype=float)
    speed = np.full(len(times), np.nan)
    for indices in _segments(valid, times, max_gap):
        segment_times, segment_xy = times[indices], xy[indices]
        if width_s == 0:
            smoothed = segment_xy.copy()
        else:
            tolerance = max(float(np.spacing(np.abs(segment_times).max())) * 8, width_s * 1e-12)
            left = np.searchsorted(segment_times, segment_times - width_s / 2 - tolerance, side="left")
            right = np.searchsorted(segment_times, segment_times + width_s / 2 + tolerance, side="right")
            cumulative = np.vstack([np.zeros(2), np.cumsum(segment_xy, axis=0)])
            smoothed = (cumulative[right] - cumulative[left]) / (right - left)[:, None]
        result[indices] = smoothed
        if len(indices) >= 2:
            velocity = np.gradient(smoothed, segment_times, axis=0)
            speed[indices] = np.linalg.norm(velocity, axis=1)
    return result, speed


def _edges(extent, width, *, clip_last=False):
    count = max(1, int(np.ceil(extent / width)))
    result = np.arange(count + 1, dtype=float) * width
    if clip_last:
        result[-1] = extent
    return result


def prepare_center_behavior(times, xy_cm_y_down, hd_cw_deg, arena_size_cm, *,
                            valid_mask=None, config=None, arena_shape="rectangle"):
    """Prepare physical tracking without file I/O or changes to native angles.

    ``times`` and later spike timestamps must share the same clock. Positions
    are calibrated cm from the arena's top-left; heading is north-zero CW.
    CB is CW from the head (0 front, +90 right), the repository convention.
    ``valid_mask`` can explicitly exclude held/interpolated tracking samples.
    For a circle, both arena dimensions equal its diameter. Coverage uses only
    spatial bins whose centers lie inside the circle, not its bounding square.

    Position smoothing and finite-difference speed never cross a missing sample
    or a gap > 1.5 median frame intervals. Exposure is nearest-frame support,
    capped to half a median interval at each side of a gap and clipped to the
    first/last supplied timestamps. Thus gaps add neither occupancy nor spikes.
    This finite-difference speed estimate is an explicit adaptation, not the
    Kalman/RTS speed-score estimator described elsewhere in the paper.
    """
    cfg = _config(config)
    times = np.asarray(times, dtype=float)
    xy = np.asarray(xy_cm_y_down, dtype=float)
    heading = np.asarray(hd_cw_deg, dtype=float)
    arena = np.asarray(arena_size_cm, dtype=float)
    if arena.ndim == 0:
        arena = np.repeat(arena, 2)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("times must contain at least two finite, strictly increasing timestamps")
    if xy.shape != (len(times), 2) or heading.shape != (len(times),):
        raise ValueError("xy must have shape (frames, 2), and heading must have shape (frames,)")
    if arena.shape != (2,) or not np.isfinite(arena).all() or np.any(arena <= 0):
        raise ValueError("arena_size_cm must contain positive finite width and height")
    if arena_shape not in ("rectangle", "circle"):
        raise ValueError("arena_shape must be rectangle or circle")
    if arena_shape == "circle" and not np.isclose(arena[0], arena[1]):
        raise ValueError("A circular arena requires equal width and height")
    supplied_valid = np.ones(len(times), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if supplied_valid.shape != (len(times),):
        raise ValueError("valid_mask must have one value per tracking frame")
    valid_position = supplied_valid & np.isfinite(xy).all(axis=1) & ((xy >= 0) & (xy <= arena)).all(axis=1)
    if arena_shape == "circle":
        valid_position &= np.linalg.norm(xy - arena / 2, axis=1) <= arena[0] / 2 + cfg.center_tolerance_cm
    dt = float(np.median(np.diff(times)))
    max_gap = cfg.max_gap_factor * dt
    smoothed_xy, speed = _smooth_positions(times, xy, valid_position, cfg.position_smoothing_s, max_gap)
    delta = arena / 2 - smoothed_xy
    distance = np.linalg.norm(delta, axis=1)
    bearing = (np.degrees(np.arctan2(delta[:, 0], -delta[:, 1])) - heading) % 360
    bearing[(distance <= cfg.center_tolerance_cm) | ~np.isfinite(heading)] = np.nan
    intervals = np.diff(times)
    support_half = np.where(intervals <= max_gap, intervals / 2, dt / 2)
    left = times - np.r_[0., support_half]
    right = times + np.r_[support_half, 0.]
    duration = right - left
    moving = valid_position & np.isfinite(speed) & (speed > cfg.speed_threshold_cm_s)
    x_edges = _edges(arena[0], cfg.spatial_bin_cm, clip_last=True)
    y_edges = _edges(arena[1], cfg.spatial_bin_cm, clip_last=True)
    spatial_occupancy = np.histogram2d(smoothed_xy[moving, 0], smoothed_xy[moving, 1],
                                       bins=[x_edges, y_edges], weights=duration[moving])[0]
    spatial_bin_mask = np.ones(spatial_occupancy.shape, dtype=bool)
    if arena_shape == "circle":
        x = (x_edges[:-1] + x_edges[1:]) / 2 - arena[0] / 2
        y = (y_edges[:-1] + y_edges[1:]) / 2 - arena[1] / 2
        spatial_bin_mask = x[:, None]**2 + y[None, :]**2 <= (arena[0] / 2)**2
    coverage = float(np.mean(spatial_occupancy[spatial_bin_mask] > 0))
    return {
        "times_s": times.copy(), "xy_cm_y_down": xy.copy(),
        "smoothed_xy_cm_y_down": smoothed_xy, "raw_hd_cw_deg": heading.copy(),
        "hd_deg": heading % 360, "cb_deg": bearing, "cd_cm": distance,
        "speed_cm_s": speed, "valid_position_mask": valid_position,
        "input_valid_mask": supplied_valid.copy(), "moving_mask": moving,
        "exposure_left_s": left, "exposure_right_s": right,
        "exposure_s": duration, "median_frame_interval_s": dt,
        "arena_size_cm": arena.copy(), "center_xy_cm": arena / 2,
        "arena_shape": arena_shape,
        "max_center_distance_cm": float(arena[0] / 2 if arena_shape == "circle" else np.linalg.norm(arena / 2)),
        "trial_interval_s": np.array([times[0], times[-1]]),
        "spatial_occupancy_s": spatial_occupancy,
        "spatial_bin_mask": spatial_bin_mask,
        "spatial_x_edges_cm": x_edges, "spatial_y_edges_cm": y_edges,
        "spatial_coverage": coverage, "config": asdict(cfg),
        "metadata": {
            "coordinate_system": "physical cm from arena top-left; y increases downward",
            "heading_convention": "north-zero clockwise; raw input retained",
            "cb_convention": "clockwise from head: 0 front, +90 right, 180 back, 270 left",
            "cb_paper_sign_conversion": "paper CCW CB = (-cb_deg) % 360",
            "speed_estimator": "finite difference of segment-wise 400 ms boxcar positions (configurable)",
            "position_boxcar_edges": "truncated and renormalized within contiguous valid segments",
            "occupancy_selection": "same strictly speed-thresholded support used for spikes",
            "spatial_coverage_bins": "bin centers inside circle" if arena_shape == "circle" else "all rectangular bins",
            "gap_policy": "no smoothing or spike assignment across missing support; exposure capped at gaps",
        },
    }


def assign_spikes_to_behavior(behavior, spike_times):
    """Return frame indices, or -1 outside valid tracked exposure.

    This performs no speed filtering. Callers can use the same mapping to
    select speed-eligible spike timestamps for independent theta calculations.
    Exact midpoint ties select the earlier frame.
    """
    spikes = np.asarray(spike_times, dtype=float)
    if spikes.ndim != 1 or not np.isfinite(spikes).all():
        raise ValueError("spike_times must be a finite one-dimensional array")
    times = behavior["times_s"]
    right = np.minimum(np.searchsorted(times, spikes), len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(times[right] - spikes) < np.abs(times[left] - spikes), right, left)
    valid = (behavior["valid_position_mask"][indices]
             & (spikes >= behavior["exposure_left_s"][indices])
             & (spikes <= behavior["exposure_right_s"][indices]))
    return np.where(valid, indices, -1)


def select_moving_spikes(behavior, spike_times):
    """Select original spike timestamps in valid exposure at speed > threshold."""
    spikes = np.asarray(spike_times, dtype=float)
    indices = assign_spikes_to_behavior(behavior, spikes)
    accepted = (indices >= 0) & behavior["moving_mask"][np.maximum(indices, 0)]
    return spikes[accepted]


def behavior_intervals(behavior, selected_mask):
    """Merge selected frame exposure in seconds without filling tracking gaps."""
    intervals = []
    selected = selected_mask & (behavior["exposure_s"] > 0)
    for left, right in zip(behavior["exposure_left_s"][selected],
                           behavior["exposure_right_s"][selected]):
        if intervals and left <= intervals[-1][1] + 1e-9:
            intervals[-1][1] = right
        else:
            intervals.append([left, right])
    return np.asarray(intervals, dtype=float).reshape(-1, 2)


def analyze_center_theta(behavior, spikes_by_unit, theta):
    """Calculate movement-conditioned theta results for all supplied units.

    Classification uses the prepared behavior's spike/coverage thresholds.
    Below-coverage sessions retain descriptive measurements and NA labels.
    Spike ACG pairs can span speed dips within continuous tracked support;
    phase occupancy uses only the selected movement intervals.
    """
    config = behavior["config"]
    min_spikes = config["min_spikes_exclusive"]
    coverage_pass = behavior["spatial_coverage"] > config["min_spatial_coverage_exclusive"]
    moving_intervals = behavior_intervals(behavior, behavior["moving_mask"])
    tracking_intervals = behavior_intervals(behavior, behavior["valid_position_mask"])
    unit_results, rows = {}, []
    for unit, raw_spikes in spikes_by_unit.items():
        moving = select_moving_spikes(behavior, raw_spikes)
        row = dict(
            unit_id=unit, n_moving_spikes=len(moving),
            paper_eligible=len(moving) > min_spikes and coverage_pass,
            theta_index=np.nan, is_theta_rhythmic=pd.NA,
            mean_vector_length=np.nan, preferred_phase_rad=np.nan, rayleigh_p=np.nan,
            is_phase_locked=pd.NA, precession_candidate=pd.NA,
            phase_precession_index=np.nan, peak_frequency_per_cycle=np.nan,
            rhythmicity_status="not_assessed", phase_locking_status="not_assessed",
            precession_status="not_assessed", theta_status="insufficient_spikes",
        )
        rows.append(row)
        if len(moving) <= min_spikes:
            continue

        rhythm = theta_rhythmicity(moving, tracking_intervals)
        lock = phase_locking(moving, theta, exposure_intervals_s=moving_intervals)
        precession = phase_precession(moving, theta)
        unit_results[unit] = dict(
            rhythmicity=rhythm, phase_locking=lock, phase_precession=precession,
        )
        row.update(
            theta_index=rhythm["theta_index"], peak_frequency_hz=rhythm["peak_frequency_hz"],
            mean_vector_length=lock["mean_vector_length"],
            preferred_phase_rad=lock["preferred_phase_rad"], rayleigh_p=lock["rayleigh_p"],
            phase_precession_index=precession["phase_precession_index"],
            peak_frequency_per_cycle=precession["peak_frequency_per_cycle"],
            n_phase_spikes=lock["n_spikes"],
            rhythmicity_status="computed" if np.isfinite(rhythm["theta_index"]) else "no_acg_pairs",
            phase_locking_status="computed" if lock["n_spikes"] > min_spikes else "insufficient_phase_spikes",
            precession_status="computed" if precession["eligible"] else "insufficient_phase_acg_support",
            theta_status="computed" if coverage_pass else "descriptive_only_insufficient_spatial_coverage",
        )
        if row["paper_eligible"] and np.isfinite(rhythm["theta_index"]):
            row["is_theta_rhythmic"] = rhythm["is_theta_rhythmic"]
        if row["paper_eligible"] and lock["n_spikes"] > min_spikes:
            row["is_phase_locked"] = lock["is_phase_locked"]
            if precession["eligible"]:
                row["precession_candidate"] = precession["precession_candidate"]

    summary = pd.DataFrame(rows).set_index("unit_id", drop=False)
    for name in ("is_theta_rhythmic", "is_phase_locked", "precession_candidate"):
        summary[name] = pd.array(summary[name], dtype="boolean")
    return dict(
        summary=summary, units=unit_results,
        metadata=dict(
            filter_parameters=theta["parameters"], phase_qc=theta["phase_qc"],
            movement_support=f"same gap-aware speed >{config['speed_threshold_cm_s']:g}cm/s exposure as center tuning",
            precession_support="moving spikes paired within continuous LFP runs",
            acg_support="movement-filtered spikes paired across speed dips, within continuous tracked support",
            classification_scope=(
                f"NA if <={min_spikes} movement spikes or spatial coverage "
                f"<={config['min_spatial_coverage_exclusive']:.0%}"
            ),
            phase_alpha=.01, exact_published_numerical_replication=False,
        ),
    )


def circular_shift_spikes(spike_times, trial_interval_s, shift_s):
    """Shift raw timestamps in clock time, retaining pauses and camera gaps."""
    spikes = np.asarray(spike_times, dtype=float)
    start, stop = np.asarray(trial_interval_s, dtype=float)
    if not np.isfinite([start, stop, shift_s]).all() or stop <= start:
        raise ValueError("Circular shifts require a finite positive trial duration")
    if spikes.ndim != 1 or not np.isfinite(spikes).all() or np.any((spikes < start) | (spikes > stop)):
        raise ValueError("Only finite spike timestamps within the trial may be shifted")
    return start + (spikes - start + shift_s) % (stop - start)


def _smooth_rates(rate, size, *, circular):
    """Smooth rates, retaining unvisited bins as missing, never as measured zero."""
    rate = np.asarray(rate, dtype=float)
    finite = np.isfinite(rate)
    kernel = np.ones(size, dtype=float)
    mode = "wrap" if circular else "constant"
    numerator = convolve1d(np.where(finite, rate, 0.), kernel, mode=mode, cval=0.)
    denominator = convolve1d(finite.astype(float), kernel, mode=mode, cval=0.)
    return np.divide(numerator, denominator, out=np.full_like(rate, np.nan), where=finite & (denominator > 0))


def _smooth_angular_counts(counts, occupancy, size):
    """Match author hdstat.m: circularly smooth counts and occupancy separately."""
    kernel = np.ones(size, dtype=float) / size
    numerator = convolve1d(np.asarray(counts, dtype=float), kernel, mode="wrap")
    denominator = convolve1d(np.asarray(occupancy, dtype=float), kernel, mode="wrap")
    rates = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                      where=(occupancy > 0) & (denominator > 0))
    return rates, numerator, denominator


def _correlation(first, second, minimum):
    common = np.isfinite(first) & np.isfinite(second)
    if common.sum() < minimum:
        return np.nan
    a, b = first[common], second[common]
    a, b = a - np.mean(a), b - np.mean(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denominator, -1, 1)) if denominator > 0 else np.nan


def _mvl(rate, centers):
    if not np.isfinite(rate).all() or np.sum(rate) <= 0:
        return np.nan
    return float(np.abs(np.sum(rate * np.exp(1j * np.deg2rad(centers)))) / np.sum(rate))


def _linear_fit(centers, rates):
    valid = np.isfinite(rates)
    if valid.sum() < 3:
        return np.nan, np.nan, np.nan
    x, y = centers[valid], rates[valid]
    xc, yc = x - np.mean(x), y - np.mean(y)
    xx, yy = xc @ xc, yc @ yc
    if xx <= 0 or yy <= 0:
        return np.nan, np.nan, np.nan
    slope = float(xc @ yc / xx)
    intercept = float(np.mean(y) - slope * np.mean(x))
    r2 = float(np.clip((xc @ yc)**2 / (xx * yy), 0., 1.))
    return slope, intercept, r2


def _prepare_features(behavior, cfg):
    midpoint = np.mean(behavior["trial_interval_s"])
    exposure = behavior["exposure_s"]
    first_exposure = np.maximum(0., np.minimum(behavior["exposure_right_s"], midpoint)
                                - behavior["exposure_left_s"])
    first_exposure = np.minimum(first_exposure, exposure)
    half_exposure = np.stack([first_exposure, exposure - first_exposure])
    features = {}
    for feature, key in (("hd", "hd_deg"), ("cb", "cb_deg"), ("cd", "cd_cm")):
        angular = feature != "cd"
        edges = (_edges(360., cfg.angle_bin_deg) if angular else
                 _edges(behavior["max_center_distance_cm"], cfg.distance_bin_cm))
        values = behavior[key]
        valid = behavior["moving_mask"] & np.isfinite(values)
        bins = np.full(len(values), -1, dtype=int)
        bins[valid] = np.searchsorted(edges, values[valid], side="right") - 1
        bins[valid & (values == edges[-1])] = len(edges) - 2
        valid &= (bins >= 0) & (bins < len(edges) - 1)
        bins[~valid] = -1
        occupancy = np.bincount(bins[valid], weights=exposure[valid], minlength=len(edges) - 1)
        halves = np.stack([np.bincount(bins[valid], weights=weights[valid], minlength=len(edges) - 1)
                           for weights in half_exposure])
        features[feature] = {
            "edges": edges, "centers": (edges[:-1] + edges[1:]) / 2,
            "bins": bins, "occupancy_s": occupancy, "half_occupancy_s": halves,
            "smoothing_bins": _kernel_size(cfg.angle_smoothing_deg, cfg.angle_bin_deg) if angular
            else _kernel_size(cfg.distance_stability_smoothing_cm, cfg.distance_bin_cm),
            "angular": angular,
        }
    return features


def _tuning_statistics(behavior, features, spikes, cfg, *, retain_tuning):
    frame_indices = assign_spikes_to_behavior(behavior, spikes)
    accepted = frame_indices >= 0
    frame_indices, accepted_spikes = frame_indices[accepted], spikes[accepted]
    first_half = accepted_spikes < np.mean(behavior["trial_interval_s"])
    statistics, tunings = {}, {}
    for name, feature in features.items():
        spike_bins = feature["bins"][frame_indices]
        valid = spike_bins >= 0
        n_bins = len(feature["centers"])
        counts = np.bincount(spike_bins[valid], minlength=n_bins)
        halves = np.stack([np.bincount(spike_bins[valid & half], minlength=n_bins)
                           for half in (first_half, ~first_half)])
        occupancy, half_occupancy = feature["occupancy_s"], feature["half_occupancy_s"]
        rate = np.divide(counts, occupancy, out=np.full(n_bins, np.nan), where=occupancy > 0)
        half_rates = np.divide(halves, half_occupancy, out=np.full((2, n_bins), np.nan), where=half_occupancy > 0)
        if feature["angular"]:
            smoothed, smoothed_counts, smoothed_occupancy = _smooth_angular_counts(
                counts, occupancy, feature["smoothing_bins"],
            )
            half_smoothed = np.stack([
                _smooth_angular_counts(c, o, feature["smoothing_bins"])[0]
                for c, o in zip(halves, half_occupancy)
            ])
        else:
            smoothed = _smooth_rates(rate, feature["smoothing_bins"], circular=False)
            half_smoothed = np.stack([_smooth_rates(r, feature["smoothing_bins"], circular=False)
                                      for r in half_rates])
        statistics[f"{name}_stability"] = _correlation(*half_smoothed, cfg.min_correlation_bins)
        statistics[f"{name}_half_common_bins"] = int(np.isfinite(half_smoothed).all(axis=0).sum())
        if feature["angular"]:
            statistics[f"{name}_mvl"] = _mvl(smoothed, feature["centers"])
            statistics[f"{name}_preferred_deg"] = (float(feature["centers"][np.nanargmax(smoothed)])
                                                       if np.isfinite(smoothed).any() and np.nansum(smoothed) > 0 else np.nan)
        else:
            slope, intercept, r2 = _linear_fit(feature["centers"], rate)
            statistics.update(cd_slope_hz_per_cm=slope, cd_intercept_hz=intercept, cd_r2=r2)
        if retain_tuning:
            tunings[name] = {
                "edges": feature["edges"].copy(), "centers": feature["centers"].copy(),
                "spike_counts": counts, "occupancy_s": occupancy.copy(), "rate_hz": rate,
                "smoothed_rate_hz": smoothed, "half_spike_counts": halves,
                "half_occupancy_s": half_occupancy.copy(), "half_rate_hz": half_rates,
                "half_smoothed_rate_hz": half_smoothed,
            }
            if feature["angular"]:
                tunings[name].update(smoothed_spike_counts=smoothed_counts,
                                     smoothed_occupancy_s=smoothed_occupancy)
    return statistics, tunings


def analyze_center_cells(behavior, spikes_by_unit, *, n_shuffles=100, seed=0):
    """Calculate observed tuning, pooled timestamp-shift nulls and classifications.

    All supplied unit identities remain in the output. An unassessable class is
    ``pd.NA`` with a status explaining why; it is never silently called negative.
    Strict >100 moving spikes, >80% spatial coverage, and full pre-smoothing
    angular coverage are separate gates. CD requires at least three visited
    distance bins. Split-half correlations use bins sampled in both halves.
    Angular smoothing follows author hdstat.m: smooth spike counts and occupancy
    separately before division (15 degrees, circular). Raw rates are also retained.
    Missing bins remain NaN through smoothing. CD R²/slope use UNSMOOTHED
    rates; its 6 cm boxcar is used only for split-half stability and display.

    Each accepted unit's complete in-trial spike timestamps are shifted before
    applying tracking/speed selection. Shifts are uniform over
    [20 s, trial duration - 20 s], so a trial must be longer than 40 s.
    Reference nulls pool eligible supplied units separately for every statistic.
    """
    if not isinstance(spikes_by_unit, Mapping):
        raise TypeError("spikes_by_unit must map unit identities to raw spike timestamps")
    if not isinstance(n_shuffles, (int, np.integer)) or n_shuffles < 0:
        raise ValueError("n_shuffles must be a nonnegative integer")
    cfg = _config(behavior["config"])
    features = _prepare_features(behavior, cfg)
    unit_ids = list(spikes_by_unit)
    start, stop = behavior["trial_interval_s"]
    duration = stop - start
    can_shuffle = n_shuffles > 0 and duration > 2 * cfg.shuffle_min_shift_s
    rng = np.random.default_rng(seed)
    nulls = {stat: np.full((len(unit_ids), n_shuffles), np.nan) for stat in STATISTICS}
    offsets = np.full((len(unit_ids), n_shuffles), np.nan)
    rows, tuning, eligibility = [], {}, []
    spatial_ok = behavior["spatial_coverage"] > cfg.min_spatial_coverage_exclusive
    feature_coverage = {name: float(np.mean(feature["occupancy_s"] > 0)) for name, feature in features.items()}
    feature_ok = {name: bool(np.all(feature["occupancy_s"] > 0)) if feature["angular"]
                  else bool(np.count_nonzero(feature["occupancy_s"] > 0) >= 3)
                  for name, feature in features.items()}
    for index, unit_id in enumerate(unit_ids):
        raw = np.asarray(spikes_by_unit[unit_id], dtype=float)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError(f"Unit {unit_id!r}: spike timestamps must be a finite one-dimensional array")
        spikes = raw[(raw >= start) & (raw <= stop)]
        n_moving = len(select_moving_spikes(behavior, spikes))
        eligible = n_moving > cfg.min_spikes_exclusive and spatial_ok
        eligibility.append(eligible)
        statistics, tuning[unit_id] = _tuning_statistics(behavior, features, spikes, cfg, retain_tuning=True)
        rows.append({"unit_id": unit_id, "n_spikes_total": len(raw), "n_spikes_in_trial": len(spikes),
                     "n_speed_spikes": n_moving, "spatial_coverage": behavior["spatial_coverage"],
                     "spatial_coverage_pass": spatial_ok,
                     **{f"{name}_coverage": value for name, value in feature_coverage.items()},
                     **{f"{name}_coverage_pass": value for name, value in feature_ok.items()}, **statistics})
        if not can_shuffle or not eligible:
            continue
        offsets[index] = rng.uniform(cfg.shuffle_min_shift_s, duration - cfg.shuffle_min_shift_s, n_shuffles)
        for shuffle_index, shift in enumerate(offsets[index]):
            shifted = circular_shift_spikes(spikes, behavior["trial_interval_s"], shift)
            shuffled_statistics, _ = _tuning_statistics(behavior, features, shifted, cfg, retain_tuning=False)
            for stat in STATISTICS:
                if feature_ok[stat.split("_")[0]]:
                    nulls[stat][index, shuffle_index] = shuffled_statistics[stat]
    thresholds = {stat: float(np.percentile(values[np.isfinite(values)], cfg.percentile))
                  if np.isfinite(values).any() else np.nan for stat, values in nulls.items()}
    for index, row in enumerate(rows):
        for feature, strength in (("hd", "hd_mvl"), ("cb", "cb_mvl"), ("cd", "cd_r2")):
            stability = f"{feature}_stability"
            status = []
            if row["n_speed_spikes"] <= cfg.min_spikes_exclusive:
                status.append("insufficient_spikes")
            if not spatial_ok:
                status.append("insufficient_spatial_coverage")
            if not feature_ok[feature]:
                status.append("incomplete_angular_coverage" if feature != "cd" else "insufficient_distance_bins")
            if not np.isfinite(row[strength]):
                status.append("undefined_tuning_strength")
            if not np.isfinite(row[stability]):
                status.append("undefined_split_half_stability")
            if n_shuffles == 0:
                status.append("shuffle_not_run")
            elif not can_shuffle:
                status.append("insufficient_trial_duration")
            elif not np.isfinite([thresholds[strength], thresholds[stability]]).all():
                status.append("unavailable_null_distribution")
            row[f"{feature}_is_cell"] = (bool(row[strength] > thresholds[strength] and row[stability] > thresholds[stability])
                                            if not status else pd.NA)
            row[f"{feature}_status"] = ";".join(status) if status else "classified"
            row[f"{feature}_strength_threshold"] = thresholds[strength]
            row[f"{feature}_stability_threshold"] = thresholds[stability]
    summary = pd.DataFrame(rows)
    if not rows:
        summary = pd.DataFrame(columns=["unit_id", "n_spikes_total", "n_spikes_in_trial", "n_speed_spikes",
                                        *STATISTICS, "hd_is_cell", "cb_is_cell", "cd_is_cell"])
    for feature in ("hd", "cb", "cd"):
        summary[f"{feature}_is_cell"] = pd.array(summary[f"{feature}_is_cell"], dtype="boolean")
    summary = summary.set_index("unit_id", drop=False)
    return {
        "summary": summary, "tuning": tuning, "nulls": nulls, "thresholds": thresholds,
        "unit_ids": unit_ids, "shuffle_offsets_s": offsets,
        "config": asdict(cfg), "metadata": {
            "paper_doi": "10.1038/s41467-024-54699-9", "n_shuffles": int(n_shuffles), "seed": seed,
            "null_scope": "session_local_pooled_eligible_units",
            "null_scope_adaptation": "The paper pools all recorded eligible cohort units; this run pools only supplied eligible session units.",
            "pooled_eligible_unit_ids": [unit for unit, eligible in zip(unit_ids, eligibility) if eligible],
            "null_sample_counts": {stat: int(np.isfinite(values).sum()) for stat, values in nulls.items()},
            "shuffle_method": "continuous raw spike-time circular shift, then reapply tracking and speed selection",
            "split_half_boundary_s": float((start + stop) / 2),
            "split_half_missing_bins": "correlation on bins occupied in both halves; no filled missing bins",
            "boxcar_missing_bins": "normalize finite neighbours, preserve original unvisited bins as NaN",
            "angular_smoothing": "author hdstat.m: circular 15 degree boxcar on counts and occupancy separately, then divide; no duplicated closing angular bin",
            "cd_fit": "unweighted ordinary least squares on visited unsmoothed 2 cm rate bins",
            "classification": "strictly greater than pooled percentile for both tuning strength and split-half stability",
            "spike_quality_adaptation": "Caller supplies good units; paper L-ratio <1 criterion is not inferred from Kilosort labels.",
        },
    }


def classify_center_cohort(results_by_session):
    """Return copied session classifications using shared pooled null thresholds.

    ``results_by_session`` maps explicit session identifiers to outputs of
    :func:`analyze_center_cells` using the same configuration. No recordings are
    loaded and no shuffles are regenerated. Pooled samples already exclude each
    session's ineligible units/features. Original summaries are unchanged.

    Returns ``results_by_session``, a combined ``summary`` with session/unit
    MultiIndex, ``thresholds``, and ``metadata``. These are session-unit
    observations: the same neuron may occur in multiple jointly sorted sessions;
    population percentages must not claim independent-neuron inference.
    """
    if not isinstance(results_by_session, Mapping) or not results_by_session:
        raise ValueError("results_by_session must be a nonempty mapping")
    first = next(iter(results_by_session.values()))
    cfg = _config(first["config"])
    for session_id, result in results_by_session.items():
        if result["config"] != first["config"]:
            raise ValueError(f"Session {session_id!r} has different analysis configuration; do not pool incompatible nulls")
    pooled = {stat: np.concatenate([np.asarray(result["nulls"][stat], dtype=float).ravel()
                                    for result in results_by_session.values()]) for stat in STATISTICS}
    thresholds = {stat: float(np.percentile(values[np.isfinite(values)], cfg.percentile))
                  if np.isfinite(values).any() else np.nan for stat, values in pooled.items()}
    metadata = {
        "null_scope": "cohort_pooled_eligible_session_units",
        "session_ids": list(results_by_session), "percentile": cfg.percentile,
        "null_sample_counts": {stat: int(np.isfinite(values).sum()) for stat, values in pooled.items()},
        "session_shuffle_counts": {session_id: result["metadata"]["n_shuffles"]
                                    for session_id, result in results_by_session.items()},
        "observation_unit": "session-unit; repeated sessions may contain the same jointly sorted neuron",
        "interpretation": "Descriptive pooled classifications; session-unit rows are not assumed independent neurons.",
    }
    classified, tables = {}, []
    for session_id, result in results_by_session.items():
        summary = result["summary"].copy(deep=True)
        for unit_id, row in summary.iterrows():
            for feature, strength in (("hd", "hd_mvl"), ("cb", "cb_mvl"), ("cd", "cd_r2")):
                stability = f"{feature}_stability"
                # Retain every biological/sampling gate. Only the null reference
                # changes; a missing local distribution may now have cohort data.
                status = [value for value in row[f"{feature}_status"].split(";")
                          if value not in ("classified", "unavailable_null_distribution", "")]
                if not np.isfinite([thresholds[strength], thresholds[stability]]).all():
                    status.append("unavailable_null_distribution")
                summary.at[unit_id, f"{feature}_is_cell"] = (
                    bool(row[strength] > thresholds[strength] and row[stability] > thresholds[stability])
                    if not status else pd.NA
                )
                summary.at[unit_id, f"{feature}_status"] = ";".join(status) if status else "classified"
                summary.at[unit_id, f"{feature}_strength_threshold"] = thresholds[strength]
                summary.at[unit_id, f"{feature}_stability_threshold"] = thresholds[stability]
        classified[session_id] = {
            **result, "summary": summary, "thresholds": dict(thresholds),
            "metadata": {**result["metadata"], **metadata,
                         "session_local_thresholds": dict(result["thresholds"])},
        }
        table = summary.copy()
        table.insert(0, "session_id", session_id)
        tables.append(table.reset_index(drop=True))
    combined = pd.concat(tables, ignore_index=True).set_index(["session_id", "unit_id"], drop=False)
    return {"results_by_session": classified, "summary": combined, "thresholds": thresholds,
            "metadata": metadata}
