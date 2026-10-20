"""Pure RF source/result loading and explicit result saving."""
from __future__ import annotations
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
import numpy as np
from Utils.json_tools import read_formatted_json
from .models import RFMapList, _make_rf_map
from ._validation import (
    _axis_name, _counts_are_numeric, _flat_list, _integer, _number,
    _presentation_matrix, _readonly_array, _spatial_axis,
)
from ._cache import read_rf_result, save_rf_result, rf_result_path
from .results import RFResult

__all__ = ["load_rfmap", "load_rf", "save_rf", "rf_result_path", "rf_tc_table", "save_rf_tc", "load_rf_tc"]

_STRUCTURAL_JSON_FIELDS = {
    "unitsSpikeCounts",
    "unitsSpikeCountsSize",
    "unitPool",
    "xPositions",
    "yPositions",
    "timeBinEdges",
    "stimulusPresentationCounts",
}


def _read_rf_source(source_path: Path) -> dict[str, Any]:
    """Read legacy JSON or MATLAB's indexed NPZ without expanding counts to lists."""

    with source_path.open("rb") as stream:
        indexed_npz = stream.read(4) == b"PK\x03\x04"
    if not indexed_npz:
        return read_formatted_json(source_path)

    with np.load(source_path, allow_pickle=False) as archive:
        raw = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        for name in (
            "unitPool", "xPositions", "yPositions", "timeBinEdges", "occupancyTimeSec",
        ):
            raw[name] = archive[name].tolist()
        if "stimulusPresentationCounts" in archive:
            raw["stimulusPresentationCounts"] = archive["stimulusPresentationCounts"].tolist()
        raw["unitsSpikeCounts"] = np.stack([
            archive[f"unit_{_integer(unit_id, 'unitPool value')}"]
            for unit_id in raw["unitPool"]
        ])
    return raw



def load_rf_maps(path: str | Path) -> RFMapList:
    """Load native values and missing bins without running analysis.

    Precomputed rate sources retain their declared Hz units. Convert counts
    with an explicit ``to_firing_rate()`` call after loading. JSON null becomes
    NaN, not a measured zero.
    """

    source_path = Path(path)
    raw = _read_rf_source(source_path)
    if not isinstance(raw, dict):
        raise ValueError("RF mapping JSON must contain an object at the top level")
    required = {
        "unitsSpikeCounts",
        "unitsSpikeCountsSize",
        "unitPool",
        "xPositions",
        "yPositions",
        "timeBinEdges",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Missing JSON keys: {', '.join(missing)}")

    size_values = _flat_list(raw["unitsSpikeCountsSize"], "unitsSpikeCountsSize")
    if len(size_values) != 4:
        raise ValueError("unitsSpikeCountsSize must contain four values")
    shape = tuple(
        _integer(value, "unitsSpikeCountsSize value") for value in size_values
    )
    if any(value <= 0 for value in shape):
        raise ValueError("unitsSpikeCountsSize values must be positive")
    n_units, n_y, n_x, n_time_bins = shape

    stored_rates = raw.get("responseUnits") == "Hz"
    if not _counts_are_numeric(raw["unitsSpikeCounts"], allow_null=True):
        raise ValueError(
            "unitsSpikeCounts contains a value that is not numeric "
            "(real numbers only; bool is invalid)"
        )
    try:
        spike_counts = np.asarray(
            raw["unitsSpikeCounts"], dtype=float if stored_rates else None,
        )
        if spike_counts.dtype == object:
            spike_counts = spike_counts.astype(float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Unable to parse unitsSpikeCounts: {exc}") from exc
    if spike_counts.shape != shape:
        raise ValueError(
            f"unitsSpikeCounts has shape {spike_counts.shape}, expected {shape}"
        )
    if np.any(np.isinf(spike_counts)) or np.any(spike_counts < 0):
        raise ValueError("unitsSpikeCounts values must be non-negative numbers or null")

    unit_pool = tuple(
        _integer(value, "unitPool value")
        for value in _flat_list(raw["unitPool"], "unitPool")
    )
    if len(unit_pool) != n_units:
        raise ValueError("unitPool length does not match unit count")
    if len(set(unit_pool)) != len(unit_pool):
        raise ValueError("unitPool must contain unique unit IDs")

    x_positions = _readonly_array(
        [
            _number(value, "xPositions value")
            for value in _flat_list(raw["xPositions"], "xPositions")
        ],
        dtype=float,
    )
    y_positions = _readonly_array(
        [
            _number(value, "yPositions value")
            for value in _spatial_axis(raw["yPositions"], "yPositions", n_y)
        ],
        dtype=float,
    )
    time_bin_edges_s = _readonly_array(
        [
            _number(value, "timeBinEdges value")
            for value in _flat_list(raw["timeBinEdges"], "timeBinEdges")
        ],
        dtype=float,
    )
    if len(x_positions) != n_x:
        raise ValueError("xPositions length does not match x dimension")
    if len(y_positions) != n_y:
        raise ValueError("yPositions length does not match y dimension")
    if len(time_bin_edges_s) != n_time_bins + 1:
        raise ValueError("timeBinEdges must contain nTimeBins + 1 edges")
    if not np.all(np.diff(time_bin_edges_s) > 0):
        raise ValueError("timeBinEdges must be strictly increasing")

    presentation_counts = None
    if "stimulusPresentationCounts" in raw:
        presentation_counts = _presentation_matrix(
            raw["stimulusPresentationCounts"],
            n_y,
            n_x,
        )

    occupancy_time_s = None
    unpresented = np.zeros((n_y, n_x), dtype=bool)
    if presentation_counts is not None:
        unpresented |= presentation_counts == 0
    if "occupancyTimeSec" in raw:
        occupancy_time_s = np.asarray(raw["occupancyTimeSec"], dtype=np.float64).reshape(
            n_y, n_x,
        )
        if not np.all(np.isfinite(occupancy_time_s)) or np.any(occupancy_time_s < 0):
            raise ValueError("occupancyTimeSec must contain finite, non-negative seconds")
        unpresented |= occupancy_time_s == 0
    absent_values = spike_counts[:, unpresented, :]
    if np.any(np.isfinite(absent_values) & (absent_values != 0)):
        raise ValueError("unpresented RF positions must have zero counts or missing rates")

    spike_counts.setflags(write=False)

    metadata = {
        key: deepcopy(value)
        for key, value in raw.items()
        if key not in _STRUCTURAL_JSON_FIELDS
    }
    if not stored_rates:
        metadata["responseUnits"] = "spike_count"
        metadata["responseNormalization"] = "none"
    if occupancy_time_s is not None:
        metadata["occupancyTimeSec"] = occupancy_time_s.tolist()
    maps = [
        _make_rf_map(
            unit_index=unit_index,
            unit_id=unit_id,
            spike_counts=spike_counts[unit_index],
            x_positions=x_positions,
            y_positions=y_positions,
            time_bin_edges_s=time_bin_edges_s,
            presentation_counts=presentation_counts,
            metadata=metadata,
            source_path=source_path,
        )
        for unit_index, unit_id in enumerate(unit_pool)
    ]
    return RFMapList(maps, source_path)



# Keep the old reader name as an alias to the same implementation.
load_rfmap = load_rf_maps
load_rf = read_rf_result


def save_rf(result: RFResult, path: str | Path) -> None:
    """Save a loaded or explicitly detected RFResult without computing it."""
    save_rf_result(path, mask_2d=result.mask_2d, center_2d=result.center_2d,
                   unit_ids=result.unit_ids, manifest=result.manifest,
                   cache_key=result.cache_key)


def rf_tc_table(maps: RFMapList, *, axis: str = "x"):
    """Convert prepared RF profiles to a native-coordinate table without I/O.

    Maps must already have one time bin and one collapsed spatial axis.
    Values, missing bins, unit order, and retained coordinates are written as
    supplied; this function performs no aggregation, selection, or detection.
    """
    import pandas as pd

    axis = _axis_name(axis)
    values = maps.to_1d_array(axis=axis)
    coordinates = getattr(maps[0], f"{axis}_positions")
    if any(not np.array_equal(getattr(item, f"{axis}_positions"), coordinates) for item in maps):
        raise ValueError("RF tuning curves must share one coordinate axis")
    table = pd.DataFrame(values, index=pd.Index(maps.unit_ids, name="unit_id"),
                         columns=coordinates)
    table.attrs["response_units"] = maps[0].metadata.get("responseUnits", "spike_count")
    return table


def save_rf_tc(table, path: str | Path) -> None:
    """Write an already prepared table without converting maps or generating curves."""
    import pandas as pd

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a prepared DataFrame; convert maps with rf_tc_table first")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unit_id", *table.columns])
        writer.writerows([unit_id, *row] for unit_id, row in zip(table.index, table.to_numpy(), strict=True))


def load_rf_tc(path: str | Path):
    """Read an existing RF profile CSV, indexed by unit ID with numeric coordinates.

    The exact path is required. Missing files raise ``FileNotFoundError``;
    no maps or detections are loaded or generated.
    """
    import pandas as pd

    profiles = pd.read_csv(path, index_col="unit_id", float_precision="round_trip")
    profiles.columns = profiles.columns.astype(float)
    return profiles
