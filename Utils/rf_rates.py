"""Presentation-based RF rates and the legacy generator's trial exposure."""

from pathlib import Path
import re

import numpy as np


RATE_NORMALIZATION = "presentation_count_time"
_SESSION_NAME = re.compile(r"(?P<date>\d{6})_\d+")


def counts_to_rates(counts, presentation_counts, time_bin_edges_s):
    """Divide pooled counts by presentations times each lag-bin width."""
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(presentation_counts)[..., None] * np.diff(time_bin_edges_s)
    return np.divide(counts, exposure, out=np.zeros_like(counts), where=exposure > 0)


def aggregate_rate(rates, time_bin_edges_s, start, stop):
    """Average bin rates over [start, stop), retaining the time dimension."""
    rates = np.asarray(rates)
    widths = np.diff(time_bin_edges_s)[start:stop]
    duration = np.sum(widths)
    if duration == 0:
        return np.zeros((*rates.shape[:-1], 1), dtype=float)
    return (rates[..., start:stop] * widths).sum(axis=-1, keepdims=True) / duration


def _spatial_matrix(values, shape, name):
    matrix = np.asarray(values, dtype=float)
    # MATLAB JSON squeezes singleton spatial dimensions.
    if matrix.ndim < 2 and 1 in shape and matrix.size == np.prod(shape):
        matrix = matrix.reshape(shape)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have spatial shape {shape}, got {matrix.shape}")
    return matrix


def resolve_presentation_counts(raw, source_path, x_positions, y_positions):
    """Read saved presentation counts or reconstruct the contributing trials.

    Historical RF files use the 960-pixel, 360-degree screen and +1 rotation
    convention of RFmapping.m. Presentation counts come from the contributing
    trials, independently of the saved cumulative display time.
    """
    shape = (len(y_positions), len(x_positions))
    if "stimulusPresentationCounts" in raw:
        counts = _spatial_matrix(raw["stimulusPresentationCounts"], shape,
                                 "stimulusPresentationCounts")
        if (not np.all(np.isfinite(counts)) or np.any(counts < 0)
                or np.any(counts != np.floor(counts))):
            raise ValueError("stimulusPresentationCounts must contain non-negative integers")
        return counts, {"method": "stimulusPresentationCounts"}

    source_path = Path(source_path)
    stem = source_path.stem
    if "free_moving" in stem:
        raise ValueError(
            "Legacy free-moving RF geometry requires saved stimulusPresentationCounts: "
            f"{source_path}"
        )
    session = next((parent for parent in source_path.parents
                    if _SESSION_NAME.fullmatch(parent.name)), None)
    if session is None:
        raise ValueError(
            "Correct RF Hz requires stimulusPresentationCounts or a source inside "
            f"its DATE_SESSION recording directory: {source_path}"
        )
    date = _SESSION_NAME.fullmatch(session.name).group("date")
    trials_path = session / f"{date}.mat"
    onsets_path = session / "data/on_list_times.npy"
    # Reuse MAT struct decoding without loading spikes or enforcing shuffle blocks.
    from Utils.rf_trials import _trial_records

    trials = _trial_records(trials_path)
    edges = np.asarray(np.load(onsets_path, allow_pickle=False), dtype=float).squeeze()
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("RF presentation reconstruction requires increasing onset boundaries")
    trial_count = len(edges) - 1
    if trial_count > len(trials):
        raise ValueError("RF onset intervals exceed the trial table length")
    # Reproduce the trials actually pooled by historical RFmapping_core, including
    # its N-onsets case.
    trials = trials[:trial_count]
    positions_x = np.asarray([trial["Square_PositionX"] for trial in trials], dtype=float)
    positions_y = np.asarray([trial["Square_PositionY"] for trial in trials], dtype=float)
    luminance = np.asarray([trial["Square_Luminance"] for trial in trials], dtype=float)
    selected_luminance = raw.get("lum", 0.5 if stem.endswith("_gray") else
                                 0.0 if stem.endswith("_off") else 1.0)
    selected = luminance == selected_luminance

    is_bar = raw.get("isVerticalBar", "vertical_bar_pooled" in stem)
    is_moving = raw.get("isBackgroundMoving", stem.startswith("egocentric_"))
    is_rotation = raw.get("isRotation", stem.startswith("rotation_"))
    is_allocentric = raw.get("isAllocentricPixelBins", stem.startswith("allocentric_pixelbins_"))
    if not (is_bar or is_moving or is_rotation or is_allocentric or stem.startswith("regular_")):
        raise ValueError(f"RF source has no supported trial-geometry declaration: {source_path.name}")

    counts = np.zeros(shape, dtype=float)
    x_positions = np.asarray(x_positions, dtype=float)
    y_positions = np.asarray(y_positions, dtype=float)
    if not (is_bar or is_moving or is_rotation or is_allocentric):
        for trial_index in np.flatnonzero(selected):
            covered = ((y_positions[:, None] == positions_y[trial_index])
                       & (x_positions[None, :] == positions_x[trial_index]))
            counts += covered
    else:
        screen_width = int(raw.get("screenWidthPix", 960))
        screen_deg = float(raw.get("screenDeg", raw.get("screenWidthDeg", 360)))
        pixels_per_degree = screen_width / screen_deg
        sizes = np.asarray([trial["Square_Size"] for trial in trials], dtype=float)
        if is_bar:
            if shape[0] != 1 or np.any(positions_y != 0):
                raise ValueError("Vertical-bar presentation counts require a singleton Y=0 axis")
            native_x = np.arange(screen_width)
            pixels_per_bin = screen_width // shape[1]
        elif is_moving or is_rotation:
            native_x = np.arange(screen_width)
            pixels_per_bin = screen_width // shape[1]
            offset_sign = float(raw.get("rotationOffsetSign", 1))
            if is_moving:
                offsets = np.asarray([trial["BackgroundRotation_XOffset_Pix"]
                                      for trial in trials], dtype=float)
            else:
                offsets = np.asarray(np.load(session / "data/hd_trials_times.npy",
                                             allow_pickle=False), dtype=float).ravel(order="F")
                if len(offsets) < trial_count:
                    raise ValueError("Rotation offsets are shorter than the analyzed trial sequence")
                offsets = offsets[:trial_count]
            positions_x = np.mod((positions_x + screen_deg / 2) * pixels_per_degree
                                 + offset_sign * offsets, screen_width)
        else:
            native_count = int(round(float(raw.get("total_deg", 360)) * pixels_per_degree))
            native_x = -float(raw.get("total_deg", 360)) / 2 + np.arange(native_count) / pixels_per_degree
            pixels_per_bin = native_count // shape[1]

        if pixels_per_bin < 1 or len(native_x) != pixels_per_bin * shape[1]:
            raise ValueError("RF spatial bins do not divide the native screen grid")
        for trial_index in np.flatnonzero(selected):
            if is_bar:
                width = int(np.ceil(sizes[trial_index] * pixels_per_degree))
                center = (screen_width + 1) / 2 + positions_x[trial_index] * pixels_per_degree
                first = int(np.floor(center - (width - 1) / 2 + 0.5)) - 1
                if first < 0 or first + width > screen_width:
                    raise ValueError("Vertical-bar footprint extends beyond the recorded screen")
                covered_native = (native_x >= first) & (native_x < first + width)
            elif is_moving or is_rotation:
                dx = (native_x - positions_x[trial_index] + screen_width / 2) % screen_width - screen_width / 2
                width = sizes[0] * pixels_per_degree
                covered_native = (dx >= -width / 2) & (dx < width / 2)
            else:
                dx = native_x - positions_x[trial_index]
                covered_native = (dx >= -sizes[0] / 2) & (dx < sizes[0] / 2)
            # A trial contributes once when any native pixel covers the final bin.
            covered_x = covered_native.reshape(shape[1], pixels_per_bin).any(axis=1)
            covered = ((y_positions[:, None] == positions_y[trial_index]) & covered_x[None, :])
            counts += covered
    return counts, {
        "method": "raw_trial_footprint",
        "trials_mat": str(trials_path),
        "onsets_npy": str(onsets_path),
        "analyzed_trials": trial_count,
    }
