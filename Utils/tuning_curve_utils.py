import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.optimize import brentq
from scipy.special import i0e, i1e
from tqdm import tqdm

from Utils.json_tools import read_formatted_json


HD_RAW_BIN_COUNT = 180
HD_CLASSIFICATION_BIN_COUNT = 30
RAYLEIGH_ALPHA = 0.01
SHUFFLE_ALPHA = 0.01
HD_KAPPA_CUTOFF = 0.075


def _get_adc_folder(session_info: dict) -> Path:
    return (
        Path(session_info["base_path"])
        / session_info["record_nodes"]
        / session_info["experiment_id"]
        / session_info["recording_name"]
        / "continuous"
        / session_info["continuous_ADC_folder"]
    )


def _get_adc_time_origin(session_info: dict) -> float:
    return float(np.load(_get_adc_folder(session_info) / "timestamps.npy", mmap_mode="r")[0])


def _read_exposure_timestamps(
    session_info, camera_input_channel, camera_ttl_threshold, camera_ttl_active_high,
):
    """Read complete pulse midpoints, carrying the signal state across ADC chunks."""
    adc_folder = _get_adc_folder(session_info)
    source_path = adc_folder / "continuous.dat"
    timestamps = np.load(adc_folder / "timestamps.npy", mmap_mode="r")
    adc_origin = float(timestamps[0])
    channel_count = int(session_info["ADC_input_channel"])
    if not 0 <= camera_input_channel < channel_count:
        raise ValueError(f"Camera channel must be between 0 and {channel_count - 1}.")
    if source_path.stat().st_size != len(timestamps) * channel_count * 2:
        raise ValueError("ADC continuous.dat size does not match timestamps/channels.")

    rise_parts, fall_parts = [], []
    previous_active = False
    # Mapping each chunk separately bounds resident memory for long recordings.
    chunk_size = 1_000_000
    for start in range(0, len(timestamps), chunk_size):
        count = min(chunk_size, len(timestamps) - start)
        signal = np.memmap(
            source_path, dtype=np.int16, mode="r",
            offset=start * channel_count * 2, shape=(count, channel_count),
        )
        active = (
            signal[:, camera_input_channel] >= camera_ttl_threshold
            if camera_ttl_active_high else
            signal[:, camera_input_channel] < camera_ttl_threshold
        )
        if start == 0:
            previous_active = bool(active[0])
        state = np.empty(count + 1, dtype=bool)
        state[0] = previous_active
        state[1:] = active
        changes = np.flatnonzero(state[1:] != state[:-1])
        rise_parts.append(timestamps[start + changes[active[changes]]])
        fall_parts.append(timestamps[start + changes[~active[changes]]])
        previous_active = bool(active[-1])
        del signal

    rise_times = np.concatenate(rise_parts)
    fall_times = np.concatenate(fall_parts)
    if not len(rise_times) or not len(fall_times):
        raise ValueError("At least two complete camera TTL pulses are required.")
    # Basler stays low outside acquisition. Those unbounded intervals are not frames.
    fall_times = fall_times[fall_times > rise_times[0]]
    if len(fall_times):
        rise_times = rise_times[rise_times < fall_times[-1]]
    if len(rise_times) < 2 or not len(fall_times):
        raise ValueError("At least two complete camera TTL pulses are required.")
    exposure_timestamps = (rise_times + fall_times) / 2 - adc_origin
    return exposure_timestamps, adc_origin, source_path


def get_exposure_timestamps(
    session_info: dict,
    data_dir: str | Path,
    *,
    camera_input_channel: int = 1,
    camera_ttl_threshold: int | float = 14000,
    camera_ttl_active_high: bool = True,
) -> tuple[np.ndarray, float, dict]:
    """Read saved times or ADC pulse midpoints, in seconds relative to ADC origin."""
    data_dir = Path(data_dir)
    source_path = data_dir / "camera_frame_times.npy"
    if source_path.is_file():
        ADC_time_origin_s = _get_adc_time_origin(session_info)
        exposure_timestamps = np.load(source_path) - ADC_time_origin_s
        timestamp_reference = "saved_camera_frame_times"
    else:
        source_path = data_dir / "sync_data.json"
        sync_data = read_formatted_json(source_path) if source_path.is_file() else {}
        if "exposure_sampling_number_list_mid" in sync_data:
            exposure_timestamps = np.asarray(
                sync_data["exposure_sampling_number_list_mid"], dtype=float
            )
            if exposure_timestamps.size < 2:
                raise ValueError(f"No complete camera timing data in {source_path}.")
            if "exposure_sampling_number_list_mid_raw" in sync_data:
                ADC_time_origin_s = float(
                    sync_data["exposure_sampling_number_list_mid_raw"][0]
                    - exposure_timestamps[0]
                )
            else:
                ADC_time_origin_s = _get_adc_time_origin(session_info)
            timestamp_reference = "saved_exposure_midpoint"
        else:
            exposure_timestamps, ADC_time_origin_s, source_path = _read_exposure_timestamps(
                session_info, camera_input_channel, camera_ttl_threshold,
                camera_ttl_active_high,
            )
            timestamp_reference = "adc_exposure_midpoint"

    exposure_periods = np.diff(exposure_timestamps)
    if not exposure_periods.size or not np.all(exposure_periods > 0):
        raise ValueError(f"Camera times must contain at least two increasing timestamps: {source_path}")
    median_period_s = float(np.median(exposure_periods))
    ttl_qc = {
        "ttl_pulse_count": int(len(exposure_timestamps)),
        "first_exposure_s": float(exposure_timestamps[0]),
        "last_exposure_s": float(exposure_timestamps[-1]),
        "median_period_s": median_period_s,
        "measured_rate_hz": float(1 / np.mean(exposure_periods)),
        "source_path": str(source_path),
        "timestamp_reference": timestamp_reference,
    }
    if timestamp_reference == "adc_exposure_midpoint":
        ttl_qc.update({
            "camera_input_channel": int(camera_input_channel),
            "camera_ttl_threshold": float(camera_ttl_threshold),
            "camera_ttl_active_high": bool(camera_ttl_active_high),
        })
    return exposure_timestamps, ADC_time_origin_s, ttl_qc


def mean_resultant_length(angles_deg: np.ndarray) -> float:
    angles_deg = np.asarray(angles_deg, dtype=float)
    angles_deg = angles_deg[np.isfinite(angles_deg)]
    if not angles_deg.size:
        return np.nan

    mean_vector = np.mean(np.exp(1j * np.deg2rad(angles_deg)))
    return float(np.clip(np.abs(mean_vector), 0.0, 1.0))


def rayleigh_test(
    spike_counts: np.ndarray,
    occupancy_time_s: np.ndarray,
    angle_centers_deg: np.ndarray,
) -> tuple[float, float]:
    spike_counts = np.asarray(spike_counts, dtype=float)
    occupancy_time_s = np.asarray(occupancy_time_s, dtype=float)
    angle_centers_deg = np.asarray(angle_centers_deg, dtype=float)
    valid = (
        np.isfinite(spike_counts)
        & np.isfinite(occupancy_time_s)
        & np.isfinite(angle_centers_deg)
        & (occupancy_time_s > 0)
    )
    spike_counts = spike_counts[valid]
    occupancy_time_s = occupancy_time_s[valid]
    angle_centers_deg = angle_centers_deg[valid]
    total_spikes = float(np.sum(spike_counts))
    if total_spikes == 0 or len(spike_counts) < 3:
        return np.nan, np.nan

    angles_rad = np.deg2rad(angle_centers_deg)
    design = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
    occupancy_weights = occupancy_time_s / np.sum(occupancy_time_s)
    design_mean = occupancy_weights @ design
    centered_design = design - design_mean

    # Score test for cos/sin modulation in a Poisson model with log occupancy
    # as offset. With uniform occupancy this reduces to the Rayleigh score test.
    score = spike_counts @ centered_design
    information = total_spikes * (
        centered_design.T @ (centered_design * occupancy_weights[:, None])
    )
    if np.linalg.matrix_rank(information) < 2:
        return np.nan, np.nan

    rayleigh_score = float(score @ np.linalg.solve(information, score))
    rayleigh_p = float(np.exp(-rayleigh_score / 2))
    return rayleigh_score, rayleigh_p


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def von_mises_kappa(rates, angles_rad):
    """Moment-matched concentration of a complete occupancy-corrected rate curve.

    Solve I1(kappa)/I0(kappa) = R without smoothing or baseline removal.
    Unmeasured directions and zero firing leave concentration undefined.
    """
    rates = np.asarray(rates, dtype=float)
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        return np.nan
    total = np.sum(rates)
    if total == 0:
        return np.nan
    resultant = float(np.clip(abs(np.sum(rates * np.exp(1j * angles_rad))) / total, 0, 1))
    if resultant < np.finfo(float).eps:
        return 0.0
    if resultant == 1:
        return np.inf
    return brentq(lambda kappa: i1e(kappa) / i0e(kappa) - resultant,
                  0, 1 / (1 - resultant), xtol=1e-12)


def update_hd_classification(tuning_curves, kappa_cutoff=HD_KAPPA_CUTOFF):
    """Update classes and concentration in place, preserving saved significance tests.

    Kappa uses the raw 180-bin rates, independently of the 30-bin significance
    tests. An unbounded concentration passes the cutoff and is stored as null
    in JSON; kappa_pass distinguishes it from an unavailable concentration.
    """
    unit_data = tuning_curves["unit_data"]
    rates = np.asarray(tuning_curves["firing_rate_hz"], dtype=float)
    occupancy = np.asarray(tuning_curves["occupancy_time_s"], dtype=float)
    edges = np.asarray(tuning_curves["angle_bin_edges_deg"], dtype=float)
    angles_rad = np.deg2rad((edges[:-1] + edges[1:]) / 2)
    complete_coverage = np.all(np.isfinite(occupancy) & (occupancy > 0))
    kappas = [von_mises_kappa(curve, angles_rad) if complete_coverage else np.nan
              for curve in rates]
    kappa_pass = [None if np.isnan(kappa) else bool(kappa >= kappa_cutoff)
                  for kappa in kappas]
    classes = []
    for rayleigh_p, shuffle_p, passes_kappa in zip(
        unit_data["rayleigh_p"], unit_data["shuffle_p"], kappa_pass, strict=True,
    ):
        if (rayleigh_p is None or shuffle_p is None
                or not np.isfinite(rayleigh_p) or not np.isfinite(shuffle_p)):
            classes.append(None)
            continue
        significance_count = int(rayleigh_p <= RAYLEIGH_ALPHA) + int(shuffle_p <= SHUFFLE_ALPHA)
        classes.append(3 if significance_count == 2 and passes_kappa else significance_count)
    unit_data.update({
        "hd_class": classes,
        "von_mises_kappa": [_json_float(kappa) for kappa in kappas],
        "kappa_pass": kappa_pass,
    })
    tuning_curves["metadata"]["classification"].update({
        "method": "occupancy_adjusted_rayleigh_circular_shift_von_mises_v2",
        "class_0": "neither significance test passed",
        "class_1": "exactly one significance test passed",
        "class_2": "rayleigh and shuffle significant; kappa cutoff not passed or unavailable",
        "class_3": "rayleigh and shuffle significant; von Mises kappa >= cutoff",
        "class_null": "one or both significance tests unavailable",
        "kappa_cutoff": float(kappa_cutoff),
        "kappa_method": "I1(kappa)/I0(kappa) = abs(sum(rate * exp(1j * theta))) / sum(rate)",
        "kappa_num_angle_bins": HD_RAW_BIN_COUNT,
        "kappa_rate_processing": "occupancy corrected; no smoothing or baseline subtraction",
        "kappa_null": "unavailable when kappa_pass is null; unbounded concentration when kappa_pass is true",
    })
    return tuning_curves


def save_hd_unit_lists(tuning_curves, directory):
    """Save exact class-1, class-2, and class-3 IDs next to the tuning curves."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(tuning_curves["unit_id"], dtype=np.int64)
    classes = np.asarray(tuning_curves["unit_data"]["hd_class"], dtype=object)
    for label in (1, 2, 3):
        np.save(directory / f"hd_cells_{label}.npy", np.sort(ids[classes == label]))


def make_head_direction_tsd(
    exposure_timestamps: np.ndarray,
    frame_indices: np.ndarray,
    angles_deg: np.ndarray,
) -> tuple[nap.Tsd, float]:
    """Keep valid pose runs and the sampling rate of the complete camera clock.

    Each retained pose contributes one nominal camera period to occupancy.
    Missing poses and camera pauses split the support; half-frame boundaries
    retain the observed duration of singleton poses without bridging those gaps.
    """
    exposure_timestamps = np.asarray(exposure_timestamps, dtype=float)
    frame_indices = np.asarray(frame_indices)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if (
        exposure_timestamps.ndim != 1
        or exposure_timestamps.size < 2
        or not np.all(np.isfinite(exposure_timestamps))
        or not np.all(np.diff(exposure_timestamps) > 0)
    ):
        raise ValueError("Camera timestamps must be finite, increasing, and contain at least two frames.")
    if (
        frame_indices.ndim != 1
        or frame_indices.dtype.kind not in "iu"
        or angles_deg.shape != frame_indices.shape
        or not np.all(frame_indices >= 0)
        or not np.all(frame_indices < len(exposure_timestamps))
        or not np.all(np.diff(frame_indices) > 0)
    ):
        raise ValueError("Pose frame IDs must be increasing integers within the camera timestamp range.")

    frame_period_s = float(np.median(np.diff(exposure_timestamps)))
    valid = np.isfinite(angles_deg)
    frame_indices = frame_indices[valid]
    angles_deg = angles_deg[valid] % 360
    if not frame_indices.size:
        raise ValueError("At least one finite head-direction sample is required.")
    times = exposure_timestamps[frame_indices]
    # A camera acquisition pause can leave consecutive frame IDs but a long
    # time gap. It must be excluded just like a missing pose estimate.
    breaks = (np.diff(frame_indices) > 1) | (np.diff(times) > 1.5 * frame_period_s)
    run_starts = np.r_[0, np.flatnonzero(breaks) + 1]
    run_ends = np.r_[np.flatnonzero(breaks), len(times) - 1]
    support = nap.IntervalSet(
        start=times[run_starts] - frame_period_s / 2,
        end=times[run_ends] + frame_period_s / 2,
    )
    return (
        nap.Tsd(t=times, d=angles_deg, time_support=support),
        1.0 / frame_period_s,
    )


def tuning_curve(
    base_dir,
    kilosort_dir,
    probe_name,
    interval_pairs,
    HD_tsd,
    adc_time_origin_s,
    num_of_bins_in_hd,
    num_shuffle,
    shuffle_seed,
    *,
    is_save: bool = False,
    save_path: str | Path | None = None,
    metadata: dict | None = None,
    timestamp_reference: str = "saved_camera_timestamps",
    feature_fs_hz: float | None = None,
    classification_bins: int = HD_CLASSIFICATION_BIN_COUNT,
    kappa_cutoff: float = HD_KAPPA_CUTOFF,
) -> dict:
    from Utils.kilosort_utils import (
        _compress_times_to_epoch_clock,
        _compute_shuffle_r_numba,
        _compute_shuffle_r_numpy,
        _flatten_feature_segments,
        convert_time_list_to_nap_tsd,
    )

    if num_of_bins_in_hd != HD_RAW_BIN_COUNT:
        raise ValueError(
            f"The GUI tuning-curve contract requires exactly {HD_RAW_BIN_COUNT} angle bins."
        )
    classification_bins = min(num_of_bins_in_hd, classification_bins)
    if classification_bins < 1 or num_of_bins_in_hd % classification_bins:
        raise ValueError("Classification bins must be positive and divide the raw angle bins.")

    base_dir = Path(base_dir)
    kilosort_dir = Path(kilosort_dir)
    cluster_KSLabel = pd.read_csv(kilosort_dir / "cluster_KSLabel.tsv", sep="\t")
    good_unit_ids = (
        cluster_KSLabel.loc[
            cluster_KSLabel["KSLabel"].astype(str).str.lower() == "good",
            "cluster_id",
        ]
        .astype(int)
        .to_numpy()
    )

    spike_times = np.load(
        base_dir / "data" / f"probe{probe_name}" / "adc_spike_time.npy",
        mmap_mode="r",
    ).reshape(-1)
    spike_clusters = np.load(
        kilosort_dir / "spike_clusters.npy", mmap_mode="r"
    ).reshape(-1)
    assert spike_times.shape == spike_clusters.shape

    good_spike_mask = np.isin(spike_clusters, good_unit_ids)
    good_spike_times = np.asarray(spike_times[good_spike_mask], dtype=float)
    good_spike_times -= float(adc_time_origin_s)
    good_spike_clusters = np.asarray(spike_clusters[good_spike_mask], dtype=np.int64)
    del good_spike_mask, spike_times, spike_clusters
    cluster_order = np.argsort(good_spike_clusters, kind="stable")
    sorted_clusters = good_spike_clusters[cluster_order]
    sorted_spike_times = good_spike_times[cluster_order]
    del cluster_order, good_spike_clusters, good_spike_times
    cluster_ids, cluster_starts, cluster_counts = np.unique(
        sorted_clusters,
        return_index=True,
        return_counts=True,
    )
    spikes_dict = {
        int(cluster_id): nap.Ts(
            t=sorted_spike_times[start : start + count],
            time_units="s",
        )
        for cluster_id, start, count in zip(cluster_ids, cluster_starts, cluster_counts)
    }
    del sorted_clusters, sorted_spike_times
    tsgroup = nap.TsGroup(spikes_dict)
    time_support = convert_time_list_to_nap_tsd(interval_pairs, return_in_list=False)
    time_support = time_support.intersect(HD_tsd.time_support)
    feature_times = HD_tsd.index.to_numpy()
    first_samples = np.searchsorted(feature_times, time_support.start, side="left")
    last_samples = np.searchsorted(feature_times, time_support.end, side="right")
    # A requested epoch can clip only the edge of a frame's support without
    # retaining its sample. Such fragments have no occupancy or shuffle values.
    occupied_intervals = last_samples > first_samples
    if not np.any(occupied_intervals):
        raise ValueError("Selected epochs contain no valid head-direction samples.")
    time_support = time_support[occupied_intervals]
    HD_tsd = HD_tsd.restrict(time_support)

    hd_spike_counts = nap.compute_tuning_curves(
        data=tsgroup,
        features=HD_tsd,
        bins=num_of_bins_in_hd,
        epochs=time_support,
        range=(0, 360),
        return_counts=True,
        fs=feature_fs_hz,
    )
    angle_dims = [dim for dim in hd_spike_counts.dims if dim != "unit"]
    if "unit" not in hd_spike_counts.dims or len(angle_dims) != 1:
        raise ValueError(
            "Head-direction tuning counts must have exactly unit and angle dimensions."
        )
    angle_dim = angle_dims[0]
    hd_spike_counts = hd_spike_counts.transpose("unit", angle_dim)
    raw_unit_ids = np.asarray(hd_spike_counts.coords["unit"].values, dtype=float)
    if (
        raw_unit_ids.ndim != 1
        or not raw_unit_ids.size
        or not np.all(np.isfinite(raw_unit_ids) & (raw_unit_ids >= 0))
        or not np.allclose(raw_unit_ids, np.rint(raw_unit_ids))
    ):
        raise ValueError("Tuning-curve unit IDs must be non-negative integers.")
    unit_ids = np.rint(raw_unit_ids).astype(np.int64)
    if len(np.unique(unit_ids)) != len(unit_ids):
        raise ValueError("Tuning-curve unit IDs must be unique.")
    angle_bin_edges_deg = np.asarray(hd_spike_counts.attrs["bin_edges"][0], dtype=float)
    expected_edges_deg = np.linspace(0.0, 360.0, HD_RAW_BIN_COUNT + 1)
    if (
        angle_bin_edges_deg.shape != expected_edges_deg.shape
        or not np.all(np.isfinite(angle_bin_edges_deg))
        or not np.allclose(angle_bin_edges_deg, expected_edges_deg, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("Tuning-curve angle bins must span 0–360° in 180 equal bins.")
    raw_occupancy_samples = np.asarray(
        hd_spike_counts.attrs["occupancy"], dtype=float
    ).reshape(-1)
    feature_fs_hz = float(hd_spike_counts.attrs["fs"])
    if (
        raw_occupancy_samples.size != HD_RAW_BIN_COUNT
        or not np.all(np.isfinite(raw_occupancy_samples) & (raw_occupancy_samples >= 0))
        or not np.allclose(raw_occupancy_samples, np.rint(raw_occupancy_samples))
    ):
        raise ValueError(
            "Tuning-curve occupancy samples must be 180 non-negative integers."
        )
    if not np.isfinite(feature_fs_hz) or feature_fs_hz <= 0:
        raise ValueError(
            "Tuning-curve feature sampling rate must be finite and positive."
        )
    occupancy_samples = np.rint(raw_occupancy_samples).astype(np.int64)
    if not np.any(occupancy_samples > 0):
        raise ValueError(
            "Tuning-curve occupancy must contain at least one occupied bin."
        )
    occupancy_time_s = occupancy_samples / feature_fs_hz
    raw_spike_counts = np.asarray(hd_spike_counts.values, dtype=float)
    expected_counts_shape = (len(unit_ids), HD_RAW_BIN_COUNT)
    if (
        raw_spike_counts.shape != expected_counts_shape
        or not np.all(np.isfinite(raw_spike_counts) & (raw_spike_counts >= 0))
        or not np.allclose(raw_spike_counts, np.rint(raw_spike_counts))
    ):
        raise ValueError(
            "Tuning-curve spike counts must be a unit-by-180 matrix of non-negative integers."
        )
    spike_counts = np.rint(raw_spike_counts).astype(np.int64)
    if np.any(spike_counts[:, occupancy_samples == 0] != 0):
        raise ValueError("Zero-occupancy angle bins must have zero spike counts.")
    firing_rates = np.full(spike_counts.shape, np.nan, dtype=float)
    np.divide(
        spike_counts,
        occupancy_time_s,
        out=firing_rates,
        where=occupancy_time_s > 0,
    )

    # Rebin counts and occupancy together; saved curves retain their raw resolution.
    bins_per_classification_bin = num_of_bins_in_hd // classification_bins
    classification_counts = spike_counts.reshape(
        len(unit_ids), classification_bins, bins_per_classification_bin
    ).sum(axis=2)
    classification_occupancy_samples = occupancy_samples.reshape(
        classification_bins, bins_per_classification_bin
    ).sum(axis=1)
    classification_occupancy_time_s = classification_occupancy_samples / feature_fs_hz
    classification_edges_deg = angle_bin_edges_deg[::bins_per_classification_bin]
    classification_centers_deg = (
        classification_edges_deg[:-1] + classification_edges_deg[1:]
    ) / 2
    classification_rates = np.full(classification_counts.shape, np.nan, dtype=float)
    np.divide(
        classification_counts,
        classification_occupancy_time_s,
        out=classification_rates,
        where=classification_occupancy_time_s > 0,
    )

    ep_starts = np.asarray(time_support.start, dtype=float)
    ep_ends = np.asarray(time_support.end, dtype=float)
    ep_lengths = ep_ends - ep_starts
    cum_lengths = np.concatenate([[0.0], np.cumsum(ep_lengths)])
    total_valid_len = float(cum_lengths[-1])
    feature_times, feature_values, feature_segment_starts, feature_segment_ends = (
        _flatten_feature_segments(
            HD_tsd,
            ep_starts,
            ep_ends,
        )
    )
    angles_rad = np.deg2rad(classification_centers_deg)
    cos_angles = np.cos(angles_rad)
    sin_angles = np.sin(angles_rad)
    shuffle_occupancy_samples = classification_occupancy_samples.copy()
    shuffle_occupancy_samples[shuffle_occupancy_samples == 0] = 1
    compute_shuffle_r = _compute_shuffle_r_numba or _compute_shuffle_r_numpy
    rng = np.random.default_rng(shuffle_seed)

    firing_rate_hz = []
    unit_data = {
        "rate_mvl": [],
        "spike_angle_mrl": [],
        "rayleigh_score": [],
        "rayleigh_p": [],
        "rayleigh_significant": [],
        "shuffle_p": [],
        "shuffle_significant": [],
    }
    with tqdm(total=len(unit_ids), desc="Classifying HD cells", unit="unit") as pbar:
        for unit_index, unit_id in enumerate(unit_ids):
            firing_rate_hz.append(
                [
                    float(value) if np.isfinite(value) else None
                    for value in firing_rates[unit_index]
                ]
            )
            unit_rates = classification_rates[unit_index]
            valid_rates = np.isfinite(unit_rates)
            rate_sum = float(np.sum(unit_rates[valid_rates]))
            rate_mvl = np.nan
            if rate_sum > 0:
                rate_mvl = float(
                    np.clip(
                        np.abs(
                            np.sum(
                                unit_rates[valid_rates]
                                * np.exp(1j * angles_rad[valid_rates])
                            )
                        )
                        / rate_sum,
                        0.0,
                        1.0,
                    )
                )

            if not np.isfinite(rate_mvl):
                for values in unit_data.values():
                    values.append(None)
                pbar.update(1)
                continue

            unit_spikes = tsgroup[int(unit_id)].restrict(time_support)
            spike_angles = unit_spikes.value_from(HD_tsd).values
            spike_angle_mrl = mean_resultant_length(spike_angles)
            rayleigh_score, rayleigh_p = rayleigh_test(
                classification_counts[unit_index],
                classification_occupancy_time_s,
                classification_centers_deg,
            )
            rayleigh_significant = (
                bool(rayleigh_p <= RAYLEIGH_ALPHA) if np.isfinite(rayleigh_p) else None
            )

            compressed_times = _compress_times_to_epoch_clock(
                unit_spikes.index.to_numpy(),
                ep_starts,
                ep_ends,
                cum_lengths,
            )
            shifts = rng.uniform(0, total_valid_len, size=num_shuffle)
            shuffle_R = compute_shuffle_r(
                compressed_times,
                shifts,
                total_valid_len,
                ep_starts,
                cum_lengths,
                feature_times,
                feature_values,
                feature_segment_starts,
                feature_segment_ends,
                classification_edges_deg,
                shuffle_occupancy_samples,
                cos_angles,
                sin_angles,
            )
            finite_shuffle_R = shuffle_R[np.isfinite(shuffle_R)]
            if num_shuffle:
                assert finite_shuffle_R.size == num_shuffle, (
                    f"Shuffle failed for unit {unit_id}: "
                    f"{finite_shuffle_R.size}/{num_shuffle} finite values."
                )
            shuffle_p = np.nan
            if np.isfinite(rate_mvl) and finite_shuffle_R.size:
                shuffle_p = (1 + np.count_nonzero(finite_shuffle_R >= rate_mvl)) / (
                    len(finite_shuffle_R) + 1
                )
            shuffle_significant = (
                bool(shuffle_p <= SHUFFLE_ALPHA) if np.isfinite(shuffle_p) else None
            )

            unit_data["rate_mvl"].append(_json_float(rate_mvl))
            unit_data["spike_angle_mrl"].append(_json_float(spike_angle_mrl))
            unit_data["rayleigh_score"].append(_json_float(rayleigh_score))
            unit_data["rayleigh_p"].append(_json_float(rayleigh_p))
            unit_data["rayleigh_significant"].append(rayleigh_significant)
            unit_data["shuffle_p"].append(_json_float(shuffle_p))
            unit_data["shuffle_significant"].append(shuffle_significant)
            pbar.update(1)

    output_metadata = {
        "session": base_dir.name,
        "probe": probe_name,
        "kilosort_dir": str(kilosort_dir),
        "timebase": "open_ephys_adc_t0_relative_seconds",
        "adc_time_origin_raw_s": float(adc_time_origin_s),
        "timestamp_reference": timestamp_reference,
        "angle_convention_note": (
            "head_direction_deg follows the input JSON angle convention; "
            "tuning-curve generation only applies modulo 360."
        ),
        "num_angle_bins": int(num_of_bins_in_hd),
        "feature_fs_hz": feature_fs_hz,
        "epoch_intervals_s": np.asarray(interval_pairs, dtype=float).tolist(),
        "valid_pose_intervals_s": time_support.values.tolist(),
        "classification": {
            "method": "occupancy_adjusted_rayleigh_or_circular_shift_v1",
            "num_angle_bins": int(classification_bins),
            "class_0": "neither significant",
            "class_1": "exactly one significant",
            "class_2": "rayleigh and shuffle significant",
            "class_null": "one or both significance tests unavailable",
            "rayleigh_alpha": RAYLEIGH_ALPHA,
            "rayleigh_test": (
                "Poisson cos/sin score test with log occupancy-time offset; "
                "chi-square with 2 degrees of freedom"
            ),
            "shuffle_alpha": SHUFFLE_ALPHA,
            "num_shuffle": int(num_shuffle),
            "shuffle_seed": int(shuffle_seed),
        },
    }
    if metadata:
        conflicting = sorted(output_metadata.keys() & metadata.keys())
        if conflicting:
            raise ValueError(
                "metadata cannot override computed tuning-curve fields: "
                + ", ".join(conflicting)
            )
        output_metadata.update(metadata)

    tuning_curves = {
        "metadata": output_metadata,
        "angle_bin_edges_deg": angle_bin_edges_deg.tolist(),
        "occupancy_samples": occupancy_samples.astype(int).tolist(),
        "occupancy_time_s": occupancy_time_s.tolist(),
        "unit_id": unit_ids.astype(int).tolist(),
        "spike_counts": spike_counts.astype(int).tolist(),
        "firing_rate_hz": firing_rate_hz,
        "unit_data": unit_data,
    }
    update_hd_classification(tuning_curves, kappa_cutoff=kappa_cutoff)

    if is_save:
        if save_path is None:
            save_path = (
                base_dir
                / "data"
                / "tuning_curves"
                / f"Probe{probe_name}"
                / "tuning_curves.tc"
            )
        else:
            save_path = Path(save_path)

        serialized_tuning_curves = json.dumps(tuning_curves, indent=2, allow_nan=False)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(temporary_path, "w") as file:
            file.write(serialized_tuning_curves)
        temporary_path.replace(save_path)
        save_hd_unit_lists(tuning_curves, save_path.parent)

    return tuning_curves
