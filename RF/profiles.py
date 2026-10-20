"""Read saved RF curves and prepare loaded values through explicit transforms."""

from pathlib import Path

import numpy as np
import pandas as pd

from RF.rflocate import load_rf as load_detected_rf, load_rfmap, load_rf_tc, rf_result_path
from Utils.direction_comparison import normalize_tc, rebin_profiles, recording_profiles, tc_loader


def rf_pick(table, max_zero_bins=2):
    """Select by measured zero bins; call on native curves before resampling."""
    selected = (np.ones(len(table), dtype=bool) if max_zero_bins is None else
                table.eq(0).sum(axis=1) <= max_zero_bins)
    result = table.loc[selected].copy()
    if "unit_info" in table.attrs:
        info = table.attrs["unit_info"]
        result.attrs["unit_info"] = {key: info[key].copy() for key in result.index}
    return result


def rf_profiles(maps, *, axis="x", probe="A", range: list[int] | None = None):
    """Wrap prepared 1-D RF maps or a loaded TC CSV in a probe-keyed table.

    Select the time window and call ``sum_to_1d`` before this conversion when
    either axis still needs aggregation. Native angles, missing values, and
    unit order are retained; selection and smoothing are separate operations.
    """
    if isinstance(maps, pd.DataFrame):
        responses = maps.to_numpy()
        angles = maps.columns.to_numpy(dtype=float)
        unit_ids = maps.index
        response_units = maps.attrs.get("response_units", "spike_count")
    else:
        responses = maps.to_1d_array(axis=axis)
        angles = maps[0].x_positions if axis.strip().lower() == "x" else maps[0].y_positions
        unit_ids = maps.unit_ids
        response_units = maps[0].metadata.get("responseUnits", "spike_count")
    index = pd.MultiIndex.from_product([[probe], unit_ids], names=["probe", "unit_id"])
    profiles = pd.DataFrame(responses, index=index, columns=angles, copy=True)
    if range is not None:
        profiles.attrs["range"] = list(range)
    profiles.attrs["response_kind"] = "RF"
    profiles.attrs["response_units"] = response_units
    profiles.attrs["unit_info"] = {
        key: {} for key in profiles.index
    }
    return profiles


def load_rf_profiles(source, *, axis="x", probe="A", range: list[int] | None = None):
    """Read a TC CSV or prepared 1-D RF map, preserving native values and angles.

    Raw multi-bin or two-dimensional sources require explicit time aggregation
    and ``sum_to_1d`` followed by ``rf_profiles``. This loader does no detection,
    selection, smoothing, rate conversion, or angular resampling.
    """
    source = Path(source)
    profiles = load_rf_tc(source) if source.suffix.lower() == ".csv" else load_rfmap(source)
    return rf_profiles(profiles, axis=axis, probe=probe, range=range)


def select_rf_profiles(profiles, *detected_results):
    """Select one probe's native profiles by explicitly supplied saved results."""
    if (profiles.index.names not in (["probe", "unit_id"], ["source", "probe", "unit_id"],
                                    ["mouse", "date", "probe", "unit_id"])
            or len(profiles.index.droplevel("unit_id").unique()) > 1):
        raise ValueError("select RF units for one probe and recording before pooling")
    detected_ids = set()
    for result in detected_results:
        detected = np.any(result["mask_2d"], axis=(1, 2))
        detected_ids.update(result["unit_ids"][detected].tolist())
    # Localization may contain a QC subset in a different unit order.
    table = profiles.loc[profiles.index.get_level_values("unit_id").isin(detected_ids)].copy()
    table.attrs["unit_info"] = {
        key: profiles.attrs["unit_info"][key].copy() for key in table.index
    }
    return table


def load_rf_centers(source, *, probe="A", detected_rf_path=None):
    """Read every saved center, optionally from an explicitly selected sidecar.

    Multi-sector RFs repeat their (probe, unit_id), including centers sharing x.
    """
    source = Path(source)
    result = load_detected_rf(rf_result_path(source) if detected_rf_path is None else detected_rf_path)
    centers = result.center_2d
    unit_ids = result.unit_ids
    maps = load_rfmap(source)
    rows, y, x = np.nonzero(centers)
    index = pd.MultiIndex.from_arrays(
        [[probe] * len(rows), unit_ids[rows]], names=["probe", "unit_id"],
    )
    return pd.DataFrame({"rf_x_deg": maps[0].x_positions[x], "rf_y_deg": maps[0].y_positions[y]}, index=index)


def load_rf(path, *, mouse, date, probe="A", label="RF", axis="x", range: list[int] | None = None):
    """Read prepared RF curves with recording keys, preserving the native grid."""
    profiles = load_rf_profiles(path, axis=axis, probe=probe, range=range)
    return recording_profiles(profiles, mouse=mouse, date=date, label=label)


def load_rf_comparison(mouseid, base_dir, probe, date, sessionID, *,
                       rf_range=(-180, 180), rf_only=True):
    """Read an existing comparison CSV without opening sources or generating data."""
    session_dir = Path(base_dir) / mouseid / str(date) / f"{date}_{sessionID}"
    suffix = "_rfonly" if rf_only else ""
    path = session_dir / "data/tc_comparison" / f"rf_excitatory_x_2d{suffix}_Probe{probe}.csv"
    return tc_loader(path, label=f"{mouseid} {date} Probe{probe} RF{sessionID}",
                     range=rf_range, response_units="spike_count")


def prep_rf_tc(table, *, bins=30, max_zero_bins=2):
    """Select native zero bins, then rebin and normalize an already loaded table."""
    return normalize_tc(rebin_profiles(rf_pick(table, max_zero_bins=max_zero_bins), bins=bins))


