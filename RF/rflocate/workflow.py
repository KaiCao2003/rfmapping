"""Shared no-shuffle RF analysis for the notebook and MATLAB's Python caller."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ._cache import rf_result_path
from .models import RFMapList
from .io import load_rfmap, save_rf
from .detection import detect_rf
from .results import RFResult
from ._validation import _nonnegative_integer

__all__ = [
    "DEFAULT_RF_SIGMAS", "rf_bin_qc", "rf_analysis_paths", "analyze_rf_maps", "analyze_rf_file", "save_rf_analysis",
    "save_rf_unit_lists",
]

DEFAULT_RF_SIGMAS = {"excitatory": (1.8, 1.0), "inhibitory": (-1.0, -0.75)}


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


def rf_analysis_paths(source_path: str | Path, rf_type: str) -> dict[str, Path]:
    """Keep separate detection, unit-list, and summary files for each polarity."""
    source_path = Path(source_path)
    output_source = source_path
    if rf_type == "inhibitory":
        output_source = source_path.with_name(f"{source_path.stem}_inhibitory{source_path.suffix}")
    return {
        "result_2d": rf_result_path(source_path, rf_type=rf_type),
        "result_1d": rf_result_path(source_path, dimension="1d", rf_type=rf_type),
        "units_2d": output_source.with_name(f"{output_source.stem}_units_with_rf.npy"),
        "units_1d": output_source.with_name(f"{output_source.stem}_units_with_rf_1d.npy"),
        "summary": output_source.with_name(f"{output_source.stem}_analysis.json"),
    }


def save_rf_analysis(
    analysis: dict[str, Any], *, dimensions: tuple[str, ...] = ("2d", "1d"),
) -> None:
    """Save both dimensions, or refresh 2-D while retaining saved 1-D provenance."""
    if dimensions not in (("2d",), ("2d", "1d")):
        raise ValueError("dimensions must be ('2d',) or ('2d', '1d')")
    output_paths = analysis["output_paths"]
    previous = {}
    if dimensions == ("2d",) and output_paths["summary"].exists():
        previous = json.loads(output_paths["summary"].read_text())
    for dimension in dimensions:
        save_rf(analysis[f"result_{dimension}"], output_paths[f"result_{dimension}"])
        units_key = "units_with_rf" if dimension == "2d" else "units_with_rf_1d"
        np.save(
            output_paths[f"units_{dimension}"],
            np.asarray(analysis[units_key], dtype=str).reshape(-1, 2),
        )
    summary = {
        "source_path": str(analysis["source_path"]),
        "probe": analysis["probe"],
        "parameters": dict(analysis["parameters"]),
        "saved_dimensions": list(dimensions),
        "bin_qc": {name: values.tolist() for name, values in analysis["bin_qc"].items()},
        "kept_unit_ids": list(analysis["summed"].unit_ids),
        "units_with_rf": analysis["units_with_rf"],
        "units_with_rf_1d": analysis["units_with_rf_1d"],
        "output_paths": {name: str(value) for name, value in output_paths.items()},
    }
    if dimensions == ("2d",):
        previous_parameters = previous.get("parameters", {})
        for key in ("sigma_1d", "cluster_forming_z_1d", "drop_bins_1d", "drop_bins", "collapse_from_2d"):
            old_key = "drop_bins" if key == "drop_bins_1d" and key not in previous_parameters else key
            if old_key in previous_parameters:
                summary["parameters"][key] = previous_parameters[old_key]
            else:
                summary["parameters"].pop(key, None)
        if "units_with_rf_1d" in previous:
            summary["units_with_rf_1d"] = previous["units_with_rf_1d"]
        else:
            summary.pop("units_with_rf_1d")
    output_paths["summary"].write_text(json.dumps(summary, indent=2) + "\n")


def analyze_rf_maps(
    maps: RFMapList,
    *,
    probe: str,
    drop_bins: int,
    reference_2d: str,
    drop_bins_2d: dict[str, int] | None = None,
    rf_type: str = "excitatory",
    rf_sigmas: dict[str, tuple[float, float]] | None = None,
    time_range_s: tuple[float, float] = (0.0, 0.2),
    max_zero_bins: int = 2,
    cluster_forming_z_2d: float | None = None,
    cluster_forming_z_1d: float | None = None,
    wrap_x: bool = True,
    collapse_from_2d: bool = False,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Calculate 2-D and 1-D detections from loaded maps without file I/O.

    Responses are explicitly converted to firing rates and averaged over the
    selected window. Loaded null/NaN bins remain missing and count toward QC.
    Count maps must carry presentation counts. Legacy exposure reconstruction
    belongs to the explicit file workflow, not this numerical operation.
    Default thresholds are mean - 1 SD (2-D) and mean - 0.75 SD (1-D)
    for inhibitory RFs; excitatory defaults remain mean + 1.8 SD and + 1 SD.
    ``rf_type="both"`` sums and filters the source once, then returns
    a dictionary keyed by ``"excitatory"`` and ``"inhibitory"``.
    ``rf_sigmas`` supplies signed (2-D, 1-D) pairs, as in the notebook.
    ``drop_bins`` and ``reference_2d`` must be chosen explicitly.
    ``drop_bins_2d`` optionally overrides the 2-D component cutoff by polarity;
    independent 1-D detection continues to use ``drop_bins``.
    ``reference_2d="bottom_row"`` uses the mean of the last spatial row and the
    SD of the whole spatial map. Independent 1-D detection is unchanged.
    Legacy unsigned threshold overrides apply to both detections.
    """
    if rf_type not in ("excitatory", "inhibitory", "both"):
        raise ValueError("rf_type must be 'excitatory', 'inhibitory', or 'both'")
    rf_types = ("excitatory", "inhibitory") if rf_type == "both" else (rf_type,)
    if rf_sigmas is None:
        rf_sigmas = DEFAULT_RF_SIGMAS
    source_path = maps.source_path
    raw = maps.to_firing_rate()
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
    response_1d = summed.sum_to_1d(axis="x", missing_as_zero=True)
    analyses = {}
    for detection_type in rf_types:
        sigma_2d, sigma_1d = rf_sigmas[detection_type]
        sign = -1 if detection_type == "inhibitory" else 1
        if cluster_forming_z_2d is not None:
            sigma_2d = sign * cluster_forming_z_2d
        if cluster_forming_z_1d is not None:
            sigma_1d = sign * cluster_forming_z_1d
        output_paths = rf_analysis_paths(source_path, detection_type)
        resolved_drop_bins_2d = (
            drop_bins if drop_bins_2d is None else drop_bins_2d[detection_type]
        )
        options = dict(
            is_shuffle=False, exclude_zero_bins=False,
            wrap_x=wrap_x, show_progress=show_progress,
        )
        result_2d = detect_rf(
            summed, reference=reference_2d, sigma=sigma_2d,
            drop_bins=resolved_drop_bins_2d, **options,
        )
        mask_2d = result_2d.mask_2d
        center_2d = result_2d.center_2d
        if collapse_from_2d:
            manifest = dict(
                result_2d.manifest, collapse_axis="x",
                grid_shape=[1, summed[0].n_x],
                storage_shape=[len(summed), 1, summed[0].n_x],
                y_positions=summed[0].y_positions[:1].tolist(),
                projection_source_cache_key=result_2d.cache_key,
            )
            projection_key = hashlib.sha256(
                (result_2d.cache_key + ":projection:x").encode("utf-8")
            ).hexdigest()
            result_1d = RFResult(
                mask_2d=result_2d.project(axis="x"),
                center_2d=result_2d.project(axis="x", center_only=True),
                unit_ids=summed.unit_ids, manifest=manifest, cache_key=projection_key,
            )
        else:
            result_1d = detect_rf(response_1d, sigma=sigma_1d, drop_bins=drop_bins, **options)
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
        analyses[detection_type] = {
            "source_path": source_path,
            "probe": probe,
            "parameters": {
                "rf_type": detection_type,
                "reference_2d": reference_2d,
                "time_range_s": list(time_range_s),
                "max_zero_bins": max_zero_bins,
                "cluster_forming_z_2d": abs(sigma_2d),
                "cluster_forming_z_1d": abs(sigma_1d),
                "sigma_2d": sigma_2d,
                "sigma_1d": sigma_1d,
                "drop_bins": drop_bins,
                "drop_bins_2d": resolved_drop_bins_2d,
                "drop_bins_1d": drop_bins,
                "wrap_x": wrap_x,
                "collapse_from_2d": collapse_from_2d,
                "is_shuffle": False,
                "exclude_zero_bins": False,
            },
            "raw": raw,
            "summed": summed,
            "response_1d": response_1d,
            "result_2d": result_2d,
            "result_1d": result_1d,
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


def analyze_rf_file(
    path: str | Path,
    *,
    probe: str,
    drop_bins: int,
    reference_2d: str,
    drop_bins_2d: dict[str, int] | None = None,
    rf_type: str = "excitatory",
    rf_sigmas: dict[str, tuple[float, float]] | None = None,
    time_range_s: tuple[float, float] = (0.0, 0.2),
    max_zero_bins: int = 2,
    cluster_forming_z_2d: float | None = None,
    cluster_forming_z_1d: float | None = None,
    wrap_x: bool = True,
    collapse_from_2d: bool = False,
    save_results: bool = True,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Explicit file workflow: read exposure, compute RFs, optionally save outputs.

    The notebook and MATLAB caller use this orchestration boundary. Loading
    raw maps and reconstructing legacy exposure happen before pure analysis.
    ``drop_bins`` and ``reference_2d`` are required keyword arguments.
    """
    counts = load_rfmap(path)
    maps = counts.to_firing_rate(reconstruct_presentations=True)
    result = analyze_rf_maps(
        maps, probe=probe, rf_type=rf_type, rf_sigmas=rf_sigmas,
        reference_2d=reference_2d,
        drop_bins_2d=drop_bins_2d,
        time_range_s=time_range_s, max_zero_bins=max_zero_bins,
        cluster_forming_z_2d=cluster_forming_z_2d,
        cluster_forming_z_1d=cluster_forming_z_1d,
        drop_bins=drop_bins, wrap_x=wrap_x, collapse_from_2d=collapse_from_2d,
        show_progress=show_progress,
    )
    if save_results:
        analyses = result.values() if rf_type == "both" else (result,)
        for analysis in analyses:
            save_rf_analysis(analysis)
    return result


def save_rf_unit_lists(
    analyses_by_probe: dict[str, dict[str, Any]], output_dir: str | Path,
    *, dimensions: tuple[str, ...] = ("2d", "1d"),
) -> None:
    """Save pooled unit lists while retaining probe identity in every row."""
    if not dimensions or any(dimension not in ("2d", "1d") for dimension in dimensions):
        raise ValueError("dimensions must contain '2d' and/or '1d'")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for rf_type in ("excitatory", "inhibitory"):
        suffix = "_inhibitory" if rf_type == "inhibitory" else ""
        for dimension in dimensions:
            name = "units_with_rf" if dimension == "2d" else "units_with_rf_1d"
            units = [
                pair
                for analyses in analyses_by_probe.values()
                for pair in analyses[rf_type][name]
            ]
            np.save(output_dir / f"{name}{suffix}.npy", np.asarray(units, dtype=str).reshape(-1, 2))
