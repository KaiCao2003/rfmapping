"""Strict raw-trial loading for regular sparse-noise RF maps.

The pooled ``regular_unitsSpikeCounts`` JSON files do not retain the trial
axis needed by permutation-based RF detection.  This module reconstructs
that axis from the original MATLAB trial table and aligned Kilosort spikes.
It intentionally supports only the regular (one position per trial) mapping
mode; transformed pixel-bin, rotation, and egocentric maps require different
label semantics and are rejected.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from Utils.rfmap import RFMapList

__all__ = ["load_regular_rf_trials"]


_SESSION_NAME = re.compile(r"^(?P<date>\d{6})_(?P<recording>\d+)$")
_REGULAR_JSON_NAME = re.compile(r"^regular_unitsSpikeCounts_.+\.json$")
_SYNTHETIC_EDGE_INTERVAL_S = 0.1
_EDGE_ATOL_S = 1e-9


def _regular_source_paths(session: str | Path, probe: str) -> dict[str, Path]:
    """Resolve and validate the four canonical raw files for one session."""

    session_path = Path(session).expanduser()
    if not session_path.is_dir():
        raise FileNotFoundError(
            f"regular RF session directory does not exist: {session_path}"
        )

    session_match = _SESSION_NAME.fullmatch(session_path.name)
    if session_match is None:
        raise ValueError(
            "regular RF session directory name must be DATE_RECORDING "
            f"(for example 260630_3), got {session_path.name!r}"
        )

    if not isinstance(probe, str):
        raise ValueError("probe must be A, B, ProbeA, or ProbeB")
    probe_match = re.fullmatch(r"(?i)(?:probe)?([ab])", probe.strip())
    if probe_match is None:
        raise ValueError("probe must be A, B, ProbeA, or ProbeB")
    probe_letter = probe_match.group(1).upper()

    date = session_match.group("date")
    recording = session_match.group("recording")
    paths = {
        "trials_mat": session_path / f"{date}.mat",
        "onsets_npy": session_path / "data" / "on_list_times.npy",
        "spike_times_npy": (
            session_path
            / "data"
            / f"probe{probe_letter}"
            / "adc_spike_time.npy"
        ),
        "spike_clusters_npy": (
            session_path
            / "kilosort"
            / f"Probe{probe_letter}"
            / f"kilosort_{recording}"
            / "spike_clusters.npy"
        ),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"regular RF source {label} does not exist: {path}"
            )
    return paths


def _finite_number(value: Any, label: str) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a numeric scalar")
    scalar = array.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)) or not isinstance(scalar, Real):
        raise ValueError(f"{label} must be a numeric scalar")
    parsed = float(scalar)
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def _integer_scalar(value: Any, label: str) -> int:
    parsed = _finite_number(value, label)
    if not parsed.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(parsed)


def _text_scalar(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{label} must be a text scalar")
    scalar = array.reshape(-1)[0]
    if not isinstance(scalar, str):
        raise ValueError(f"{label} must be a text scalar")
    return scalar.strip()


def _trial_records(path: Path) -> list[Mapping[str, Any]]:
    try:
        payload = loadmat(path, simplify_cells=True)
    except Exception as exc:
        raise ValueError(f"unable to read regular RF trials MAT file {path}: {exc}") from exc

    if "trials" not in payload:
        raise ValueError(f"regular RF MAT file has no 'trials' variable: {path}")
    raw_trials = payload["trials"]
    if isinstance(raw_trials, Mapping):
        candidates: list[Any] = [raw_trials]
    elif isinstance(raw_trials, (list, tuple)):
        candidates = list(raw_trials)
    elif isinstance(raw_trials, np.ndarray):
        candidates = list(raw_trials.reshape(-1))
    else:
        raise ValueError("MAT 'trials' must be a non-empty struct array")

    if not candidates:
        raise ValueError("MAT 'trials' must be a non-empty struct array")
    if not all(isinstance(record, Mapping) for record in candidates):
        raise ValueError("every MAT trial must be a struct")
    return candidates  # type: ignore[return-value]


def _load_1d_npy(path: Path, label: str) -> NDArray[Any]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"unable to read {label} from {path}: {exc}") from exc
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional NPY array")
    return array


def _validate_finite_sorted_times(times: NDArray[Any], label: str) -> None:
    if (
        np.issubdtype(times.dtype, np.bool_)
        or not np.issubdtype(times.dtype, np.number)
        or np.issubdtype(times.dtype, np.complexfloating)
    ):
        raise ValueError(f"{label} must contain real numeric timestamps")
    if times.size == 0:
        raise ValueError(f"{label} must not be empty")

    chunk_size = 1_000_000
    for start in range(0, times.size, chunk_size):
        chunk = np.asarray(times[start : start + chunk_size])
        if not np.all(np.isfinite(chunk)):
            raise ValueError(f"{label} must contain only finite timestamps")
        if start and float(chunk[0]) < float(times[start - 1]):
            raise ValueError(f"{label} must be sorted in non-decreasing order")
        if chunk.size > 1 and np.any(chunk[1:] < chunk[:-1]):
            raise ValueError(f"{label} must be sorted in non-decreasing order")


def _validate_rf_maps(
    rf_maps: RFMapList,
) -> tuple[NDArray[np.float64], NDArray[np.float64], tuple[float, float]]:
    if not isinstance(rf_maps, RFMapList):
        raise TypeError("rf_maps must be an RFMapList")
    if _REGULAR_JSON_NAME.fullmatch(rf_maps.source_path.name) is None:
        raise ValueError(
            "raw trial loading supports only regular_unitsSpikeCounts JSON data"
        )
    if not rf_maps:
        raise ValueError("rf_maps must contain at least one unit")

    first = rf_maps[0]
    if first.n_time_bins != 1:
        raise ValueError(
            "rf_maps must already be summed to exactly one response-window time bin"
        )
    x_positions = np.asarray(first.x_positions, dtype=float)
    y_positions = np.asarray(first.y_positions, dtype=float)
    time_edges = np.asarray(first.time_bin_edges_s, dtype=float)
    if time_edges.shape != (2,):
        raise ValueError("summed rf_maps must have exactly two time-bin edges")
    time_range_s = (float(time_edges[0]), float(time_edges[1]))
    if not all(math.isfinite(value) for value in time_range_s):
        raise ValueError("response-window time edges must be finite")
    if not time_range_s[0] < time_range_s[1]:
        raise ValueError("response-window time edges must be strictly increasing")

    for rf_map in rf_maps:
        if rf_map.n_time_bins != 1:
            raise ValueError(
                "every rf_map must have exactly one response-window time bin"
            )
        if not np.array_equal(rf_map.x_positions, x_positions):
            raise ValueError("all rf_maps must have identical x positions")
        if not np.array_equal(rf_map.y_positions, y_positions):
            raise ValueError("all rf_maps must have identical y positions")
        if not np.array_equal(rf_map.time_bin_edges_s, time_edges):
            raise ValueError("all rf_maps must have identical response-window edges")

    return x_positions, y_positions, time_range_s


def _raw_trial_geometry(
    trials: Sequence[Mapping[str, Any]],
    x_positions: NDArray[np.float64],
    y_positions: NDArray[np.float64],
) -> tuple[
    NDArray[np.int64],
    NDArray[np.int8],
    NDArray[np.int64],
]:
    required = {
        "Stimulus_Type",
        "Square_PositionX",
        "Square_PositionY",
        "Square_Luminance",
    }
    raw_x = np.empty(len(trials), dtype=float)
    raw_y = np.empty(len(trials), dtype=float)
    polarities = np.empty(len(trials), dtype=np.int8)

    transformed_fields = {
        "BackgroundRotation_XOffset_Pix",
        "HeadDirection_XOffset_Pix",
        "Rotation_XOffset_Pix",
    }
    for index, trial in enumerate(trials):
        missing = required.difference(trial)
        if missing:
            raise ValueError(
                f"trial {index} is missing required regular RF fields: "
                f"{sorted(missing)}"
            )
        unexpected = transformed_fields.intersection(trial)
        if unexpected:
            raise ValueError(
                f"trial {index} contains transformed-map fields: "
                f"{sorted(unexpected)}"
            )
        stimulus_type = _text_scalar(
            trial["Stimulus_Type"],
            f"trials[{index}].Stimulus_Type",
        )
        if stimulus_type != "Receptive Field Mapping":
            raise ValueError(
                f"trial {index} is not a regular receptive-field mapping trial"
            )
        raw_x[index] = _finite_number(
            trial["Square_PositionX"],
            f"trials[{index}].Square_PositionX",
        )
        raw_y[index] = _finite_number(
            trial["Square_PositionY"],
            f"trials[{index}].Square_PositionY",
        )
        luminance = _integer_scalar(
            trial["Square_Luminance"],
            f"trials[{index}].Square_Luminance",
        )
        if luminance not in {0, 1}:
            raise ValueError("Square_Luminance must contain exactly OFF=0 or ON=1")
        polarities[index] = luminance

    raw_x_positions = np.unique(raw_x)
    raw_y_positions = np.unique(raw_y)
    if not np.array_equal(raw_x_positions, x_positions):
        raise ValueError("raw trial x positions do not match pooled RF x positions")
    if not np.array_equal(raw_y_positions, y_positions):
        raise ValueError("raw trial y positions do not match pooled RF y positions")
    if not np.array_equal(np.unique(polarities), np.asarray([0, 1])):
        raise ValueError("regular RF trials must contain both OFF=0 and ON=1")

    x_indices = np.searchsorted(x_positions, raw_x)
    y_indices = np.searchsorted(y_positions, raw_y)
    position_ids = (
        y_indices * x_positions.size + x_indices
    ).astype(np.int64, copy=False)

    n_positions = x_positions.size * y_positions.size
    block_size = n_positions * 2
    if len(trials) % block_size:
        raise ValueError(
            "regular RF trial count is not divisible by the complete "
            "position-by-polarity factorial block size"
        )
    n_blocks = len(trials) // block_size
    strata = np.repeat(np.arange(n_blocks, dtype=np.int64), block_size)
    expected = np.arange(block_size, dtype=np.int64)
    encoded = position_ids + polarities.astype(np.int64) * n_positions
    for block in range(n_blocks):
        block_values = np.sort(encoded[strata == block])
        if not np.array_equal(block_values, expected):
            raise ValueError(
                f"trial block {block} is not a complete one-repeat "
                "position-by-polarity factorial"
            )

    return position_ids, polarities, strata


def _aligned_onsets(
    paths: Mapping[str, Path],
    n_trials: int,
) -> NDArray[np.float64]:
    raw_edges = _load_1d_npy(paths["onsets_npy"], "stimulus onset edges")
    _validate_finite_sorted_times(raw_edges, "stimulus onset edges")
    if raw_edges.size != n_trials + 1:
        raise ValueError(
            "on_list_times must contain one onset per trial plus one synthetic "
            f"terminal edge; got {raw_edges.size} values for {n_trials} trials"
        )
    if np.any(np.asarray(raw_edges[1:]) <= np.asarray(raw_edges[:-1])):
        raise ValueError("stimulus onset edges must be strictly increasing")

    terminal_interval = float(raw_edges[-1] - raw_edges[-2])
    if not math.isclose(
        terminal_interval,
        _SYNTHETIC_EDGE_INTERVAL_S,
        rel_tol=0.0,
        abs_tol=_EDGE_ATOL_S,
    ):
        raise ValueError(
            "final on_list_times value does not match the pipeline's synthetic "
            "+0.1 s terminal edge"
        )
    return np.asarray(raw_edges[:-1], dtype=float)


def _unit_trial_counts(
    paths: Mapping[str, Path],
    unit_ids: NDArray[np.int64],
    selected_onsets: NDArray[np.float64],
    time_range_s: tuple[float, float],
) -> NDArray[np.int64]:
    spike_times = _load_1d_npy(paths["spike_times_npy"], "aligned spike times")
    spike_clusters = _load_1d_npy(
        paths["spike_clusters_npy"],
        "spike cluster IDs",
    )
    if spike_times.shape != spike_clusters.shape:
        raise ValueError("spike times and spike cluster IDs must have equal lengths")
    _validate_finite_sorted_times(spike_times, "aligned spike times")
    if not np.issubdtype(spike_clusters.dtype, np.integer) or np.issubdtype(
        spike_clusters.dtype, np.bool_
    ):
        raise ValueError("spike cluster IDs must have an integer dtype")
    if spike_clusters.size == 0 or np.any(spike_clusters < 0):
        raise ValueError("spike cluster IDs must be non-empty and non-negative")

    relevant = np.isin(spike_clusters, unit_ids)
    relevant_clusters = np.asarray(spike_clusters[relevant], dtype=np.int64)
    relevant_times = np.asarray(spike_times[relevant], dtype=float)
    order = np.argsort(relevant_clusters, kind="stable")
    sorted_clusters = relevant_clusters[order]
    sorted_times = relevant_times[order]

    starts = selected_onsets + time_range_s[0]
    stops = selected_onsets + time_range_s[1]
    responses = np.empty((unit_ids.size, selected_onsets.size), dtype=np.int64)
    for unit_index, unit_id in enumerate(unit_ids):
        first = int(np.searchsorted(sorted_clusters, unit_id, side="left"))
        last = int(np.searchsorted(sorted_clusters, unit_id, side="right"))
        if first == last:
            raise ValueError(
                f"RFMap unit_id {int(unit_id)} is absent from spike_clusters.npy"
            )
        unit_times = sorted_times[first:last]
        left = np.searchsorted(unit_times, starts, side="left")
        right = np.searchsorted(unit_times, stops, side="left")
        responses[unit_index] = right - left
    return responses


def _validate_pooled_counts(
    rf_maps: RFMapList,
    responses: NDArray[np.int64],
    position_ids: NDArray[np.int64],
) -> None:
    n_positions = rf_maps[0].n_y * rf_maps[0].n_x
    presentation_counts = np.bincount(
        position_ids,
        minlength=n_positions,
    ).reshape(rf_maps[0].n_y, rf_maps[0].n_x)

    for unit_index, rf_map in enumerate(rf_maps):
        pooled_raw = np.bincount(
            position_ids,
            weights=responses[unit_index],
            minlength=n_positions,
        ).reshape(rf_map.n_y, rf_map.n_x)
        pooled_json = np.asarray(rf_map.to_2d_array())
        if not np.array_equal(pooled_raw, pooled_json):
            max_difference = float(np.max(np.abs(pooled_raw - pooled_json)))
            raise ValueError(
                "raw trial counts do not match pooled regular RF JSON for "
                f"unit_id {rf_map.unit_id}; maximum absolute difference is "
                f"{max_difference:g}"
            )
        if (
            rf_map.presentation_counts is not None
            and not np.array_equal(rf_map.presentation_counts, presentation_counts)
        ):
            raise ValueError(
                "raw trial presentation counts do not match pooled regular RF JSON"
            )


def load_regular_rf_trials(
    session: str | Path,
    probe: str,
    rf_maps: RFMapList,
    *,
    polarity: str = "on",
    validate_pooled: bool = True,
) -> dict[str, Any]:
    """Load trial responses aligned to an already-summed regular RFMapList.

    The sole RFMap time bin defines the half-open response window.  ON and OFF
    trials are selected independently, and `(x, y)` is encoded as one joint
    position label.  The returned strata are empirically verified repeat-block
    indices.  Existing pooled JSON files contain ON responses only, so OFF
    loading requires ``validate_pooled=False``.
    """

    paths = _regular_source_paths(session, probe)
    x_positions, y_positions, time_range_s = _validate_rf_maps(rf_maps)

    if not isinstance(polarity, str):
        raise ValueError("polarity must be 'on' or 'off'")
    normalized_polarity = polarity.strip().lower()
    if normalized_polarity not in {"on", "off"}:
        raise ValueError("polarity must be 'on' or 'off'")
    if not isinstance(validate_pooled, (bool, np.bool_)):
        raise TypeError("validate_pooled must be bool")
    if normalized_polarity == "off" and validate_pooled:
        raise ValueError(
            "regular pooled JSON contains ON counts only; use "
            "validate_pooled=False when loading OFF trials"
        )

    trials = _trial_records(paths["trials_mat"])
    onsets = _aligned_onsets(paths, len(trials))
    all_position_ids, polarities, all_strata = _raw_trial_geometry(
        trials,
        x_positions,
        y_positions,
    )

    polarity_value = 1 if normalized_polarity == "on" else 0
    selected = polarities == polarity_value
    selected_position_ids = all_position_ids[selected]
    selected_strata = all_strata[selected]
    selected_onsets = onsets[selected]
    if not np.any(selected):
        raise ValueError(f"regular RF trials contain no {normalized_polarity} trials")

    unit_ids = np.asarray(rf_maps.unit_ids, dtype=np.int64)
    responses = _unit_trial_counts(
        paths,
        unit_ids,
        selected_onsets,
        time_range_s,
    )
    if validate_pooled:
        _validate_pooled_counts(
            rf_maps,
            responses,
            selected_position_ids,
        )

    provenance = {
        "kind": "regular_sparse_noise_raw",
        "trials_mat": str(paths["trials_mat"]),
        "onsets_npy": str(paths["onsets_npy"]),
        "spike_times_npy": str(paths["spike_times_npy"]),
        "spike_clusters_npy": str(paths["spike_clusters_npy"]),
        "pooled_rf_json": str(rf_maps.source_path),
        "synthetic_terminal_edge_excluded": True,
        "n_all_trials": len(trials),
        "n_repeat_blocks": int(np.unique(all_strata).size),
    }
    arrays = {
        "responses": np.array(responses, copy=True),
        "position_ids": np.array(selected_position_ids, copy=True),
        "stratum_ids": np.array(selected_strata, copy=True),
        "unit_ids": np.array(unit_ids, copy=True),
        "x_positions": np.array(x_positions, copy=True),
        "y_positions": np.array(y_positions, copy=True),
    }
    for array in arrays.values():
        array.setflags(write=False)
    return {
        **arrays,
        "shape": (int(y_positions.size), int(x_positions.size)),
        "time_range_s": time_range_s,
        "polarity": normalized_polarity,
        "provenance": provenance,
    }
