from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def write_indexed_rfmap():
    """Write the indexed NPZ layout emitted by MATLAB's WriteRfmap."""

    def write(path: Path, payload: dict) -> Path:
        shared_names = (
            "unitPool", "xPositions", "yPositions", "timeBinEdges", "occupancyTimeSec",
        )
        metadata = {
            name: value for name, value in payload.items()
            if name not in {*shared_names, "unitsSpikeCounts"}
        }
        metadata.update(format="rfmap", formatVersion=2, storage="indexed_npz")
        arrays = {
            name: np.asarray(payload[name]) for name in shared_names
        }
        arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
        counts = np.asarray(payload["unitsSpikeCounts"], dtype=float)
        # ZIP entry order differs from unitPool; NPY arrays use MATLAB column order.
        for index in reversed(range(len(payload["unitPool"]))):
            arrays[f"unit_{payload['unitPool'][index]}"] = np.asfortranarray(counts[index])
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return path

    return write
