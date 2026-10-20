"""Read saved center tuning and prepare CB/CD profiles without reclassification."""

from pathlib import Path

import numpy as np
import pandas as pd

from EBB.center_cell_io import load_center_tc
from Utils.direction_comparison import normalize_tc, recording_profiles, save_tc, tc_loader, tcRange
from Utils.tc_preparation import check_tc_outputs, prefix_tc


def _bin_count(native, bins):
    if bins is None:
        return native
    if (isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer))
            or bins < 1 or native % bins != 0):
        raise ValueError("bins must be a positive integer dividing the native bin count")
    return bins


def _center_profiles(data, feature, *, probe, bins, cells_only):
    if data["metadata"]["feature"] != feature:
        raise ValueError(f"Expected saved {feature.upper()} tuning")
    units = data["unit_id"]
    occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
    native = len(occupancy)
    bins = _bin_count(native, bins)
    edges_name = "angle_bin_edges_deg" if feature == "cb" else "distance_bin_edges_cm"
    edges = np.asarray(data[edges_name], dtype=float)
    counts = np.asarray(data["spike_counts"], dtype=float).reshape(len(units), native)
    if native:
        group = native // bins
        counts = counts.reshape(len(units), bins, group).sum(axis=2)
        occupancy = occupancy.reshape(bins, group).sum(axis=1)
        edges = edges[::group]
    rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan), where=occupancy > 0)
    index = (pd.Index(units, name="unit_id") if probe is None else
             pd.MultiIndex.from_product([[probe], units], names=["probe", "unit_id"]))
    table = pd.DataFrame(rates, index=index, columns=(edges[:-1] + edges[1:]) / 2)
    table.attrs = {
        "response_units": "Hz", "response_kind": feature.upper(),
        "coordinate_units": "degrees" if feature == "cb" else "cm",
        "coordinate_convention": data["metadata"]["coordinate_convention"],
        "unit_info": {key: {name: values[row] for name, values in data["unit_data"].items()}
                      for row, key in enumerate(index)},
    }
    if feature == "cb":
        table.attrs["range"] = tcRange(True)
    if cells_only:
        selected = np.array([value is True for value in data["unit_data"][f"{feature}_is_cell"]], dtype=bool)
        table = table.loc[selected].copy()
        table.attrs["unit_info"] = {key: table.attrs["unit_info"][key] for key in table.index}
    return table


def cb_profiles(data, *, probe="A", bins=None, cells_only=False):
    """Rebin loaded CB counts and exposure, retaining native clockwise centers."""
    return _center_profiles(data, "cb", probe=probe, bins=bins, cells_only=cells_only)


def cd_profiles(data, *, probe="A", bins=None, cells_only=False):
    """Rebin loaded CD counts and exposure on the original distance axis in cm."""
    return _center_profiles(data, "cd", probe=probe, bins=bins, cells_only=cells_only)


def _prep_center_tc(table, *, bins, angular):
    """Match HD comparison preprocessing after native rate CSV generation."""
    bins = _bin_count(table.shape[1], bins)
    if table.shape[1]:
        group = table.shape[1] // bins
        coordinates = table.columns.to_numpy(dtype=float)
        if angular:
            coordinates = np.unwrap(coordinates, period=360)
        coordinates = coordinates.reshape(bins, group).mean(axis=1)
        values = table.to_numpy().reshape(len(table), bins, group).sum(axis=2)
        prepared = pd.DataFrame(values, index=table.index, columns=coordinates)
        prepared.attrs = table.attrs.copy()
    else:
        prepared = table.copy()
    return normalize_tc(prepared)


def prep_cb_tc(table, *, bins=30):
    """Rebin and normalize loaded comparison rates; retain clockwise ordering.

    For occupancy-weighted native rebinning, call cb_profiles on the saved .tc
    before exporting. This step matches prep_hd_tc's sum-and-normalize display.
    """
    return _prep_center_tc(table, bins=bins, angular=True)


def prep_cd_tc(table, *, bins=None):
    """Normalize loaded distance rates; optional adjacent rebinning stays in cm."""
    return _prep_center_tc(table, bins=bins, angular=False)


def prepare_cb(data, *, mouse, date, probe="A", label="CB", bins=30, cells_only=False):
    """Prepare loaded CB evidence for comparisons with complete recording keys."""
    return recording_profiles(cb_profiles(data, probe=probe, bins=bins, cells_only=cells_only),
                              mouse=mouse, date=date, label=label)


def prepare_cd(data, *, mouse, date, probe="A", label="CD", bins=None, cells_only=False):
    """Prepare loaded CD evidence with recording keys and native distance units."""
    return recording_profiles(cd_profiles(data, probe=probe, bins=bins, cells_only=cells_only),
                              mouse=mouse, date=date, label=label)


def load_center_comparison(path, *, feature, label=None):
    """Read one saved numerical CB/CD CSV without source access or generation."""
    if feature not in ("cb", "cd"):
        raise ValueError("feature must be cb or cd")
    table = tc_loader(path, label=label, response_units="Hz",
                      range=tcRange(True) if feature == "cb" else None)
    table.attrs["response_kind"] = feature.upper()
    table.attrs["coordinate_units"] = "degrees" if feature == "cb" else "cm"
    return table


def generate_center_tc(source, output, *, feature, unit_prefix, bins=None,
                       cells_only=False, overwrite=False):
    """Explicitly export saved CB/CD counts and exposure as a comparison CSV."""
    if feature not in ("cb", "cd"):
        raise ValueError("feature must be cb or cd")
    output = Path(output)
    check_tc_outputs([output], inputs=[source], overwrite=overwrite)
    table = _center_profiles(load_center_tc(source), feature, probe=None,
                             bins=bins, cells_only=cells_only)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_tc(prefix_tc(table, unit_prefix), output)
    return output
