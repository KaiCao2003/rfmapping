import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
from tqdm import tqdm

from Utils.json_tools import read_formatted_json


HD_RAW_BIN_COUNT = 180
RAYLEIGH_ALPHA = 0.05
SHUFFLE_ALPHA = 0.01


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


def save_hd_unit_lists(tuning_curves, directory):
    """Save exact class-1 and class-2 IDs next to the tuning curves."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(tuning_curves["unit_id"], dtype=np.int64)
    classes = np.asarray(tuning_curves["unit_data"]["hd_class"], dtype=object)
    for label in (1, 2):
        np.save(directory / f"hd_cells_{label}.npy", np.sort(ids[classes == label]))


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

    hd_spike_counts = nap.compute_tuning_curves(
        data=tsgroup,
        features=HD_tsd,
        bins=num_of_bins_in_hd,
        epochs=time_support,
        range=(0, 360),
        return_counts=True,
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
    angle_centers_deg = hd_spike_counts.coords[angle_dim].values.astype(float)
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
    angles_rad = np.deg2rad(angle_centers_deg)
    cos_angles = np.cos(angles_rad)
    sin_angles = np.sin(angles_rad)
    shuffle_occupancy_samples = occupancy_samples.copy()
    shuffle_occupancy_samples[shuffle_occupancy_samples == 0] = 1
    compute_shuffle_r = _compute_shuffle_r_numba or _compute_shuffle_r_numpy
    rng = np.random.default_rng(shuffle_seed)

    firing_rate_hz = []
    unit_data = {
        "hd_class": [],
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
            unit_rates = firing_rates[unit_index]
            firing_rate_hz.append(
                [float(value) if np.isfinite(value) else None for value in unit_rates]
            )
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
                spike_counts[unit_index],
                occupancy_time_s,
                angle_centers_deg,
            )
            rayleigh_significant = (
                bool(rayleigh_p < RAYLEIGH_ALPHA) if np.isfinite(rayleigh_p) else None
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
                angle_bin_edges_deg,
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

            hd_class = None
            if rayleigh_significant is not None and shuffle_significant is not None:
                hd_class = (
                    2
                    if rayleigh_significant and shuffle_significant
                    else 1
                    if rayleigh_significant or shuffle_significant
                    else 0
                )

            unit_data["hd_class"].append(hd_class)
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
            "head_direction_deg must be calibrated to GUI convention: 0 degrees up, "
            "positive counter-clockwise. This notebook only applies modulo 360."
        ),
        "num_angle_bins": int(num_of_bins_in_hd),
        "feature_fs_hz": feature_fs_hz,
        "epoch_intervals_s": np.asarray(interval_pairs, dtype=float).tolist(),
        "classification": {
            "method": "occupancy_adjusted_rayleigh_or_circular_shift_v1",
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
        saved_tuning_curves = json.loads(serialized_tuning_curves)
        assert tuple(saved_tuning_curves) == (
            "metadata",
            "angle_bin_edges_deg",
            "occupancy_samples",
            "occupancy_time_s",
            "unit_id",
            "spike_counts",
            "firing_rate_hz",
            "unit_data",
        )
        assert len(saved_tuning_curves["angle_bin_edges_deg"]) == num_of_bins_in_hd + 1
        assert len(saved_tuning_curves["occupancy_samples"]) == num_of_bins_in_hd
        assert len(saved_tuning_curves["occupancy_time_s"]) == num_of_bins_in_hd
        saved_unit_ids = saved_tuning_curves["unit_id"]
        saved_counts = saved_tuning_curves["spike_counts"]
        saved_rates = saved_tuning_curves["firing_rate_hz"]
        saved_unit_data = saved_tuning_curves["unit_data"]
        num_units = len(unit_ids)
        assert len(saved_unit_ids) == len(saved_counts) == len(saved_rates) == num_units
        assert saved_unit_ids
        assert all(type(unit_id) is int and unit_id >= 0 for unit_id in saved_unit_ids)
        assert len(set(saved_unit_ids)) == num_units
        assert all(
            type(value) is int and value >= 0
            for value in saved_tuning_curves["occupancy_samples"]
        )
        assert all(
            type(value) in (int, float) and np.isfinite(value) and value >= 0
            for value in saved_tuning_curves["occupancy_time_s"]
        )
        assert any(value > 0 for value in saved_tuning_curves["occupancy_time_s"])
        assert all(
            (samples == 0) == (occupied_s == 0)
            for samples, occupied_s in zip(
                saved_tuning_curves["occupancy_samples"],
                saved_tuning_curves["occupancy_time_s"],
            )
        )
        assert np.allclose(
            saved_tuning_curves["occupancy_samples"],
            np.asarray(saved_tuning_curves["occupancy_time_s"]) * feature_fs_hz,
        )
        assert all(len(row) == num_of_bins_in_hd for row in saved_counts)
        assert all(len(row) == num_of_bins_in_hd for row in saved_rates)
        assert all(
            type(count) is int and count >= 0 for row in saved_counts for count in row
        )
        for count_row, rate_row in zip(saved_counts, saved_rates):
            for count, rate, occupied_s in zip(
                count_row,
                rate_row,
                saved_tuning_curves["occupancy_time_s"],
            ):
                if occupied_s == 0:
                    assert count == 0 and rate is None
                else:
                    assert type(rate) in (int, float) and np.isfinite(rate)
                    assert np.isclose(rate, count / occupied_s)

        expected_unit_data_keys = (
            "hd_class",
            "rate_mvl",
            "spike_angle_mrl",
            "rayleigh_score",
            "rayleigh_p",
            "rayleigh_significant",
            "shuffle_p",
            "shuffle_significant",
        )
        assert tuple(saved_unit_data) == expected_unit_data_keys
        assert all(
            len(saved_unit_data[key]) == num_units for key in expected_unit_data_keys
        )
        assert all(
            value is None or (type(value) is int and value in {0, 1, 2})
            for value in saved_unit_data["hd_class"]
        )
        for key in (
            "rate_mvl",
            "spike_angle_mrl",
            "rayleigh_score",
            "rayleigh_p",
            "shuffle_p",
        ):
            assert all(
                value is None or (type(value) in (int, float) and np.isfinite(value))
                for value in saved_unit_data[key]
            )
        assert all(
            value is None or 0 <= value <= 1
            for key in ("rate_mvl", "spike_angle_mrl", "rayleigh_p", "shuffle_p")
            for value in saved_unit_data[key]
        )
        assert all(
            value is None or value >= 0 for value in saved_unit_data["rayleigh_score"]
        )
        assert all(
            (score is None) == (p_value is None)
            for score, p_value in zip(
                saved_unit_data["rayleigh_score"],
                saved_unit_data["rayleigh_p"],
            )
        )
        for key in ("rayleigh_significant", "shuffle_significant"):
            assert all(
                value is None or type(value) is bool for value in saved_unit_data[key]
            )
        for (
            hd_class,
            rayleigh_p,
            rayleigh_significant,
            shuffle_p,
            shuffle_significant,
        ) in zip(
            saved_unit_data["hd_class"],
            saved_unit_data["rayleigh_p"],
            saved_unit_data["rayleigh_significant"],
            saved_unit_data["shuffle_p"],
            saved_unit_data["shuffle_significant"],
        ):
            assert rayleigh_significant == (
                None if rayleigh_p is None else rayleigh_p < RAYLEIGH_ALPHA
            )
            assert shuffle_significant == (
                None if shuffle_p is None else shuffle_p <= SHUFFLE_ALPHA
            )
            expected_hd_class = (
                None
                if rayleigh_significant is None or shuffle_significant is None
                else 2
                if rayleigh_significant and shuffle_significant
                else 1
                if rayleigh_significant or shuffle_significant
                else 0
            )
            assert hd_class == expected_hd_class

        save_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = save_path.with_suffix(save_path.suffix + ".tmp")
        with open(temporary_path, "w") as file:
            file.write(serialized_tuning_curves)
        temporary_path.replace(save_path)
        save_hd_unit_lists(tuning_curves, save_path.parent)

    return tuning_curves
