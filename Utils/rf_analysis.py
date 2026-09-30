"""Shared no-shuffle RF analysis for the notebook and MATLAB's Python caller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from Utils.rf_cache import rf_result_path
from Utils.rfmap import RFMapList, _nonnegative_integer, load_rf_maps

__all__ = ["rf_bin_qc", "analyze_rf_file"]


def rf_bin_qc(
    summed: RFMapList,
    *,
    max_missing_bins: int = 2,
    max_zero_bins: int = 2,
) -> dict[str, np.ndarray]:
    """Count missing and observed zero bins separately for each unit."""
    max_missing_bins = _nonnegative_integer(max_missing_bins, "max_missing_bins")
    max_zero_bins = _nonnegative_integer(max_zero_bins, "max_zero_bins")
    pooled = np.stack([rf_map.to_2d_array() for rf_map in summed])
    missing = ~np.isfinite(pooled)
    if summed[0].presentation_counts is not None:
        missing |= summed[0].presentation_counts[None, ...] <= 0
    if "occupancyTimeSec" in summed[0].metadata:
        occupancy = np.asarray(summed[0].metadata["occupancyTimeSec"]).reshape(pooled.shape[1:])
        missing |= occupancy[None, ...] <= 0
    zero = ~missing & (pooled == 0)
    missing_counts = missing.sum(axis=(1, 2))
    zero_counts = zero.sum(axis=(1, 2))
    valid_counts = (~missing & ~zero).sum(axis=(1, 2))
    return {
        "unit_ids": np.asarray(summed.unit_ids),
        "missing_bins": missing_counts,
        "zero_bins": zero_counts,
        "valid_bins": valid_counts,
        "keep": (
            (missing_counts <= max_missing_bins)
            & (zero_counts <= max_zero_bins)
            & (valid_counts > 0)
        ),
    }


def analyze_rf_file(
    path: str | Path,
    *,
    probe: str,
    time_range_s: tuple[float, float] = (0.0, 0.2),
    max_missing_bins: int = 2,
    max_zero_bins: int = 2,
    cluster_forming_z_2d: float = 1.8,
    cluster_forming_z_1d: float = 1.0,
    drop_bins: int = 2,
    wrap_x: bool = True,
    collapse_from_2d: bool = False,
    save_results: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Load one RF file, apply bin QC, and return its 2-D and 1-D detections.

    Counts are normalized by stimulus exposure through ``load_rf_maps``.
    Missing and zero responses are excluded from the no-shuffle mean and SD.
    Saved outputs use the source filename so separate stimulus maps do not
    overwrite each other's unit lists or QC summaries.
    """
    source_path = Path(path)
    raw = load_rf_maps(source_path)
    summed = raw.sum(*time_range_s, show_progress=show_progress)
    qc = rf_bin_qc(
        summed, max_missing_bins=max_missing_bins, max_zero_bins=max_zero_bins,
    )
    keep = qc["keep"]
    if not keep.any():
        raise ValueError(
            f"No units in Probe{probe} pass RF bin QC for {source_path} "
            f"(max_missing_bins={max_missing_bins}, max_zero_bins={max_zero_bins}) "
            "with at least one valid nonzero bin"
        )
    raw = RFMapList(
        [rf_map for rf_map, retained in zip(raw, keep, strict=True) if retained],
        raw.source_path,
    )
    summed = RFMapList(
        [rf_map for rf_map, retained in zip(summed, keep, strict=True) if retained],
        summed.source_path,
    )
    output_paths = {
        "result_2d": rf_result_path(source_path),
        "result_1d": source_path.with_name(f"{source_path.stem}_1d.npz"),
        "units_2d": source_path.with_name(f"{source_path.stem}_units_with_rf.npy"),
        "units_1d": source_path.with_name(f"{source_path.stem}_units_with_rf_1d.npy"),
        "summary": source_path.with_name(f"{source_path.stem}_analysis.json"),
    }
    options = dict(
        is_shuffle=False, exclude_zero_bins=True, drop_bins=drop_bins,
        wrap_x=wrap_x, show_progress=show_progress,
    )
    options_2d = dict(
        options, cluster_forming_z=cluster_forming_z_2d,
        result_path=output_paths["result_2d"] if save_results else None,
    )
    mask_2d = summed.rf_2d(**options_2d)
    center_2d = summed.rf_2d(is_center=True, **options_2d)
    options_1d = dict(
        options, cluster_forming_z=cluster_forming_z_1d,
        collapse_from_2d=collapse_from_2d,
        result_path=output_paths["result_1d"] if save_results else None,
    )
    mask_1d = summed.rf_1d(**options_1d)
    center_1d = summed.rf_1d(is_center=True, **options_1d)
    units_with_rf = [
        (probe, summed.unit_ids[index])
        for index in np.flatnonzero(mask_2d.any(axis=(1, 2)))
    ]
    units_with_rf_1d = [
        (probe, summed.unit_ids[index])
        for index in np.flatnonzero(mask_1d.any(axis=1))
    ]
    if save_results:
        np.save(output_paths["units_2d"], np.asarray(units_with_rf, dtype=str).reshape(-1, 2))
        np.save(output_paths["units_1d"], np.asarray(units_with_rf_1d, dtype=str).reshape(-1, 2))
        summary = {
            "source_path": str(source_path),
            "probe": probe,
            "parameters": {
                "time_range_s": list(time_range_s),
                "max_missing_bins": max_missing_bins,
                "max_zero_bins": max_zero_bins,
                "cluster_forming_z_2d": cluster_forming_z_2d,
                "cluster_forming_z_1d": cluster_forming_z_1d,
                "drop_bins": drop_bins,
                "wrap_x": wrap_x,
                "collapse_from_2d": collapse_from_2d,
                "is_shuffle": False,
                "exclude_zero_bins": True,
            },
            "bin_qc": {name: values.tolist() for name, values in qc.items()},
            "kept_unit_ids": list(summed.unit_ids),
            "units_with_rf": units_with_rf,
            "units_with_rf_1d": units_with_rf_1d,
            "output_paths": {name: str(value) for name, value in output_paths.items()},
        }
        output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "raw": raw,
        "summed": summed,
        "mask_2d": mask_2d,
        "center_2d": center_2d,
        "mask_1d": mask_1d,
        "center_1d": center_1d,
        "bin_qc": qc,
        "units_with_rf": units_with_rf,
        "units_with_rf_1d": units_with_rf_1d,
        "output_paths": output_paths,
    }
