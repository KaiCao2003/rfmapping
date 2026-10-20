"""Shared no-shuffle RF analysis for the notebook and MATLAB's Python caller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._cache import rf_result_path, save_rf_result
from .models import RFMapList
from .io import load_rfmap
from .detection import detect_rf
from ._validation import _nonnegative_integer

__all__ = ["rf_bin_qc", "analyze_rf_file", "save_rf_unit_lists"]


def rf_bin_qc(
    summed: RFMapList,
    *,
    max_zero_bins: int = 2,
) -> dict[str, np.ndarray]:
    """Apply one zero-bin limit, including positions loaded from null/NaN."""
    max_zero_bins = _nonnegative_integer(max_zero_bins, "max_zero_bins")
    pooled = np.stack([rf_map.to_2d_array() for rf_map in summed])
    zero = np.isnan(pooled) | (pooled == 0)
    zero_counts = zero.sum(axis=(1, 2))
    valid_counts = (~zero).sum(axis=(1, 2))
    return {
        "unit_ids": np.asarray(summed.unit_ids),
        "zero_bins": zero_counts,
        "valid_bins": valid_counts,
        "keep": (zero_counts <= max_zero_bins) & (valid_counts > 0),
    }


def analyze_rf_file(
    path: str | Path,
    *,
    probe: str,
    rf_type: str = "excitatory",
    time_range_s: tuple[float, float] = (0.0, 0.2),
    max_zero_bins: int = 2,
    cluster_forming_z_2d: float | None = None,
    cluster_forming_z_1d: float | None = None,
    drop_bins: int = 2,
    wrap_x: bool = True,
    collapse_from_2d: bool = False,
    save_results: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Load one RF file, apply bin QC, and return its 2-D and 1-D detections.

    Responses are explicitly converted to firing rates and averaged over the
    selected window. Loaded null/NaN bins remain missing and count toward QC.
    Saved outputs use the source filename so separate stimulus maps do not
    overwrite each other's unit lists or QC summaries. Inhibitory outputs add
    ``_inhibitory`` to the source stem to preserve excitatory results.
    Default thresholds are mean - 1.5 SD (2-D) and mean - 0.75 SD (1-D)
    for inhibitory RFs; excitatory defaults remain mean + 1.8 SD and + 1 SD.
    ``rf_type="both"`` loads, sums, and filters the source once, then returns
    a dictionary keyed by ``"excitatory"`` and ``"inhibitory"``. Explicit
    threshold overrides apply to both detections.
    """
    if rf_type not in ("excitatory", "inhibitory", "both"):
        raise ValueError("rf_type must be 'excitatory', 'inhibitory', or 'both'")
    rf_types = ("excitatory", "inhibitory") if rf_type == "both" else (rf_type,)
    source_path = Path(path)
    raw = load_rfmap(source_path).to_firing_rate(reconstruct_presentations=True)
    summed = raw.mean_rate(*time_range_s, show_progress=show_progress)
    qc = rf_bin_qc(summed, max_zero_bins=max_zero_bins)
    keep = qc["keep"]
    if not keep.any():
        raise ValueError(
            f"No units in Probe{probe} pass RF bin QC for {source_path} "
            f"(max_zero_bins={max_zero_bins}) "
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
    analyses = {}
    for detection_type in rf_types:
        z_2d = cluster_forming_z_2d
        if z_2d is None:
            z_2d = 1.5 if detection_type == "inhibitory" else 1.8
        z_1d = cluster_forming_z_1d
        if z_1d is None:
            z_1d = 0.75 if detection_type == "inhibitory" else 1.0
        output_source = source_path
        if detection_type == "inhibitory":
            output_source = source_path.with_name(f"{source_path.stem}_inhibitory{source_path.suffix}")
        output_paths = {
            "result_2d": rf_result_path(source_path, rf_type=detection_type),
            "result_1d": rf_result_path(source_path, dimension="1d", rf_type=detection_type),
            "units_2d": output_source.with_name(f"{output_source.stem}_units_with_rf.npy"),
            "units_1d": output_source.with_name(f"{output_source.stem}_units_with_rf_1d.npy"),
            "summary": output_source.with_name(f"{output_source.stem}_analysis.json"),
        }
        options = dict(
            is_shuffle=False, exclude_zero_bins=False, drop_bins=drop_bins,
            wrap_x=wrap_x, show_progress=show_progress, rf_type=detection_type,
        )
        options_2d = dict(
            options, cluster_forming_z=z_2d,
            result_path=output_paths["result_2d"] if save_results else None,
        )
        result_2d = detect_rf(summed, **options_2d)
        mask_2d = result_2d.mask_2d
        center_2d = result_2d.center_2d
        if collapse_from_2d:
            mask_1d = result_2d.project(axis="x")[:, 0, :]
            center_1d = result_2d.project(axis="x", center_only=True)[:, 0, :]
            if save_results:
                parent = result_2d
                manifest = dict(
                    parent["manifest"], collapse_axis="x",
                    grid_shape=[1, summed[0].n_x],
                    storage_shape=[len(summed), 1, summed[0].n_x],
                    y_positions=summed[0].y_positions[:1].tolist(),
                    projection_source_cache_key=parent["cache_key"],
                )
                projection_key = hashlib.sha256(
                    (parent["cache_key"] + ":projection:x").encode("utf-8")
                ).hexdigest()
                save_rf_result(
                    output_paths["result_1d"], mask_2d=mask_1d[:, None, :],
                    center_2d=center_1d[:, None, :], unit_ids=summed.unit_ids,
                    manifest=manifest, cache_key=projection_key,
                )
        else:
            options_1d = dict(
                options, cluster_forming_z=z_1d,
                result_path=output_paths["result_1d"] if save_results else None,
            )
            result_1d = detect_rf(summed, dimension="1d", axis="x", **options_1d)
            mask_1d = result_1d.mask_2d[:, 0, :]
            center_1d = result_1d.center_2d[:, 0, :]
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
                    "rf_type": detection_type,
                    "time_range_s": list(time_range_s),
                    "max_zero_bins": max_zero_bins,
                    "cluster_forming_z_2d": z_2d,
                    "cluster_forming_z_1d": z_1d,
                    "drop_bins": drop_bins,
                    "wrap_x": wrap_x,
                    "collapse_from_2d": collapse_from_2d,
                    "is_shuffle": False,
                    "exclude_zero_bins": False,
                },
                "bin_qc": {name: values.tolist() for name, values in qc.items()},
                "kept_unit_ids": list(summed.unit_ids),
                "units_with_rf": units_with_rf,
                "units_with_rf_1d": units_with_rf_1d,
                "output_paths": {name: str(value) for name, value in output_paths.items()},
            }
            output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")
        analyses[detection_type] = {
            "raw": raw,
            "summed": summed,
            "result_2d": result_2d,
            "mask_2d": mask_2d,
            "center_2d": center_2d,
            "mask_1d": mask_1d,
            "center_1d": center_1d,
            "bin_qc": qc,
            "units_with_rf": units_with_rf,
            "units_with_rf_1d": units_with_rf_1d,
            "output_paths": output_paths,
        }
    return analyses if rf_type == "both" else analyses[rf_type]


def save_rf_unit_lists(analyses_by_probe: dict[str, dict[str, Any]], output_dir: str | Path) -> None:
    """Save pooled unit lists while retaining probe identity in every row."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rf_type in ("excitatory", "inhibitory"):
        suffix = "_inhibitory" if rf_type == "inhibitory" else ""
        for name in ("units_with_rf", "units_with_rf_1d"):
            units = [
                pair
                for analyses in analyses_by_probe.values()
                for pair in analyses[rf_type][name]
            ]
            np.save(output_dir / f"{name}{suffix}.npy", np.asarray(units, dtype=str).reshape(-1, 2))
