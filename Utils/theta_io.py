"""Explicit persistence for theta summaries and the evidence used in plots.

The CSV is the per-unit table; NPZ retains numerical arrays and their dtypes.
The manifest retains scalar results, parameters, provenance, and array links.
Readers only load this saved output and never generate missing inputs.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


_FORMAT = "theta_analysis_v1"


def _store_arrays(value, arrays):
    if isinstance(value, np.ndarray):
        name = f"array_{len(arrays)}"
        arrays[name] = value
        return {"npz_array": name}
    if isinstance(value, dict):
        return {str(key): _store_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_store_arrays(item, arrays) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _restore_arrays(value, arrays):
    if isinstance(value, dict):
        if set(value) == {"npz_array"}:
            return arrays[value["npz_array"]].copy()
        return {key: _restore_arrays(item, arrays) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_arrays(item, arrays) for item in value]
    return value


def save_theta_analysis(directory, result, *, provenance, lfp_reference=None, overwrite=False):
    """Save independent theta results, optionally including prepared LFP QC.

    Full feature dictionaries are retained, including counts, occupancy, rates,
    spectra, classification gates, and numerical parameters. The optional LFP
    reference contains only its prepared spectrum and short trace, not raw LFP.
    """
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(directory)
    arrays = {}
    manifest = _store_arrays(dict(
        format=_FORMAT,
        unit_ids=result["summary"]["unit_id"].tolist(),
        summary_dtypes={name: str(dtype) for name, dtype in result["summary"].dtypes.items()},
        units=[dict(unit_id=unit, features=features) for unit, features in result["units"].items()],
        metadata=result["metadata"], provenance=provenance,
        lfp_reference=lfp_reference,
    ), arrays)
    manifest_text = json.dumps(manifest, indent=2) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(directory / "theta_cells.csv", index=False)
    np.savez_compressed(directory / "theta_arrays.npz", **arrays)
    (directory / "manifest.json").write_text(manifest_text, encoding="utf-8")
    return directory


def load_theta_analysis(directory):
    """Read a saved theta run, with no raw recording or CB/CD dependencies."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["format"] != _FORMAT:
        raise ValueError(f"Unsupported theta analysis format: {manifest['format']}")
    summary = pd.read_csv(
        directory / "theta_cells.csv", dtype=manifest["summary_dtypes"],
        float_precision="round_trip",
    )
    # CSV inference alone cannot preserve string IDs such as "001" or "NA".
    summary["unit_id"] = pd.Series(
        manifest["unit_ids"], index=summary.index, dtype=manifest["summary_dtypes"]["unit_id"],
    )
    summary = summary.set_index("unit_id", drop=False)
    with np.load(directory / "theta_arrays.npz", allow_pickle=False) as arrays:
        saved = _restore_arrays(manifest, arrays)
    return dict(
        summary=summary,
        units={entry["unit_id"]: entry["features"] for entry in saved["units"]},
        metadata=saved["metadata"], provenance=saved["provenance"],
        lfp_reference=saved["lfp_reference"],
    )
