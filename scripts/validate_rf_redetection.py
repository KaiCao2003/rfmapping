"""Independently validate the authorized global RF redetection, without writes.

Run remotely from the repository root with ``python -m
scripts.validate_rf_redetection REPORT.json``. The report contains ``sources``
with ``source_path``, ``probe``, and ``outputs`` keyed by excitatory/inhibitory;
each output has ``result_2d``, ``units_2d``, and ``summary`` paths. Its
``protected_sha256`` maps source/sidecar and preserved 1-D paths to pre-run
hashes. A source may supply a documented ``presentation_counts_path`` override,
also protected by its hash. All paths are absolute. JSON findings go to stdout;
failures exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from RF.rflocate import load_rf, load_rfmap


SETTINGS = {"excitatory": (2.0, 2), "inhibitory": (-1.0, 3)}
WINDOW = (0.0, 0.2)


def components(mask):
    """Flood-fill four neighbors, wrapping columns but never rows."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    for row, column in np.argwhere(mask):
        if seen[row, column]:
            continue
        pending = [(int(row), int(column))]
        seen[row, column] = True
        members = []
        while pending:
            y, x = pending.pop()
            members.append(y * width + x)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, (x - 1) % width), (y, (x + 1) % width)):
                if 0 <= yy < height and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    pending.append((yy, xx))
        yield np.asarray(members, dtype=np.int64)


def expected_masks(values, sigma, drop_bins):
    """Use only NumPy statistics and the independent flood-fill above."""
    masks = np.zeros(values.shape, dtype=bool)
    for index, observed in enumerate(values):
        spatial = np.where(np.isnan(observed), 0.0, observed)
        sd = spatial.std(ddof=0)
        if sd == 0:
            continue
        threshold = spatial.mean() + sigma * sd
        candidate = spatial >= threshold if sigma > 0 else spatial <= threshold
        for member_indices in components(candidate):
            if member_indices.size > drop_bins:
                masks[index].flat[member_indices] = True
    return masks


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_array(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}")
    require(np.array_equal(actual, expected), f"{name}: values differ")


def validate_result(record, polarity, response, qc, values):
    paths = record["outputs"][polarity]
    result = load_rf(paths["result_2d"])
    unit_ids = qc["unit_ids"][qc["keep"]]
    sigma, drop_bins = SETTINGS[polarity]
    expected = expected_masks(values, sigma, drop_bins)
    require_array(result.unit_ids, unit_ids, "result unit IDs/source order")
    require(result.mask_2d.shape == expected.shape, "result mask shape differs from native grid")
    mismatch = result.mask_2d.astype(bool) != expected
    require(not mismatch.any(), f"mask differs in {int(mismatch.sum())} bins; units "
            f"{unit_ids[mismatch.any(axis=(1, 2))][:10].tolist()}")
    require(not np.any(result.center_2d & ~result.mask_2d), "centers lie outside mask")
    region_counts = []
    for unit_id, mask, centers in zip(unit_ids, expected, result.center_2d, strict=True):
        regions = list(components(mask))
        for indices in regions:
            require(centers.ravel()[indices].sum() == 1,
                    f"unit {unit_id}: retained component does not have exactly one center")
        require(centers.sum() == len(regions), f"unit {unit_id}: center count differs from region count")
        region_counts.append(len(regions))

    manifest, parameters = result.manifest, result.manifest["parameters"]
    expected_parameters = {
        "reference": "global", "alternative": "greater" if sigma > 0 else "less",
        "cluster_forming_z": abs(sigma), "drop_bins": drop_bins,
        "is_shuffle": False, "exclude_zero_bins": False, "wrap_x": True,
    }
    for name, value in expected_parameters.items():
        require(parameters[name] == value, f"manifest {name}: {parameters[name]!r} != {value!r}")
    require(manifest["detector_algorithm"] == "pooled-spatial-z-v3-zero-filled", "wrong detector algorithm")
    require("baseline" not in manifest, "global result still contains bottom-row baseline metadata")
    require("collapse_axis" not in manifest, "2-D result is marked as collapsed")
    require(Path(manifest["source_path"]).resolve() == Path(record["source_path"]).resolve(), "manifest source differs")
    require(manifest["response_units"] == "Hz", "manifest response is not Hz")
    require(np.allclose(manifest["time_range_s"], WINDOW, rtol=0, atol=1e-12), "manifest window differs")
    require_array(manifest["grid_shape"], expected.shape[1:], "manifest grid shape")
    require_array(manifest["storage_shape"], expected.shape, "manifest storage shape")
    require_array(manifest["x_positions"], response[0].x_positions, "manifest x coordinates")
    require_array(manifest["y_positions"], response[0].y_positions, "manifest y coordinates")

    pairs = [[str(record["probe"]), str(unit_id)] for unit_id in unit_ids[expected.any(axis=(1, 2))]]
    expected_list = np.asarray(pairs, dtype=str).reshape(-1, 2)
    require_array(np.load(paths["units_2d"], allow_pickle=False), expected_list, "saved unit list")
    summary = json.loads(Path(paths["summary"]).read_text())
    require(Path(summary["source_path"]).resolve() == Path(record["source_path"]).resolve(), "summary source differs")
    require(str(summary["probe"]) == str(record["probe"]), "summary probe differs")
    require_array(summary["kept_unit_ids"], unit_ids, "summary retained unit IDs")
    require_array(np.asarray(summary["units_with_rf"], dtype=str).reshape(-1, 2), expected_list,
                  "summary detected unit list")
    for name, value in qc.items():
        require_array(summary["bin_qc"][name], value, f"summary QC {name}")
    expected_summary = {"rf_type": polarity, "reference_2d": "global", "sigma_2d": sigma,
                        "cluster_forming_z_2d": abs(sigma), "drop_bins_2d": drop_bins,
                        "max_zero_bins": 2, "wrap_x": True, "is_shuffle": False,
                        "exclude_zero_bins": False}
    for name, value in expected_summary.items():
        require(summary["parameters"][name] == value, f"summary parameter {name} differs")
    require(np.allclose(summary["parameters"]["time_range_s"], WINDOW, rtol=0, atol=1e-12),
            "summary response window differs")
    return {"polarity": polarity, "passed": True, "qc_units": len(unit_ids),
            "detected_units": len(pairs), "selected_bins": int(expected.sum()),
            "regions": sum(region_counts), "result_2d": paths["result_2d"]}


def validate_report(report):
    findings, errors = [], []
    sources = report["sources"]
    require(bool(sources), "report contains no completed sources")
    protected = report["protected_sha256"]
    for record in sources:
        source = record["source_path"]
        try:
            require(source in protected, "source is absent from protected_sha256")
            maps = load_rfmap(source)
            if "presentation_counts_path" in record:
                counts_path = record["presentation_counts_path"]
                require(counts_path in protected, "presentation override is absent from protected_sha256")
                rates = maps.to_firing_rate(presentation_counts=np.load(counts_path, allow_pickle=False))
            else:
                rates = maps.to_firing_rate(reconstruct_presentations=True)
            response = rates.mean_rate(*WINDOW, show_progress=False)
            values = response.to_2d_array()
            require(not np.isinf(values).any(), "source response contains infinity")
            zero = np.isnan(values) | (values == 0)
            zero_counts = zero.sum(axis=(1, 2))
            valid_counts = (~zero).sum(axis=(1, 2))
            keep = (zero_counts <= 2) & (valid_counts > 0)
            require(keep.any(), "completed source has no units passing QC")
            qc = {"unit_ids": np.asarray(response.unit_ids), "zero_bins": zero_counts,
                  "valid_bins": valid_counts, "keep": keep}
            for polarity in SETTINGS:
                try:
                    findings.append({"source_path": source, "probe": record["probe"],
                                     **validate_result(record, polarity, response, qc, values[keep])})
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    errors.append({"source_path": source, "polarity": polarity, "error": str(exc)})
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            errors.append({"source_path": source, "error": str(exc)})

    verified_hashes = 0
    for name, expected_hash in protected.items():
        try:
            with Path(name).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            require(actual == expected_hash, "protected SHA256 differs from pre-run hash")
            verified_hashes += 1
        except (OSError, ValueError) as exc:
            errors.append({"path": name, "error": str(exc)})
    return {"passed": not errors, "sources_checked": len(sources),
            "protected_hashes_verified": verified_hashes, "results": findings, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    validation = validate_report(report)
    print(json.dumps(validation, indent=2, allow_nan=False))
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
