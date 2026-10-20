"""Shared camera timestamps and valid heading time support."""

from pathlib import Path

import numpy as np
import pynapple as nap

from Utils.json_tools import read_formatted_json


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


