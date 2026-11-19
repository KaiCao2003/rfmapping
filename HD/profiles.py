"""Read saved HD data and prepare profiles as separate operations."""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

from Utils.direction_comparison import normalize_tc, profile_table, rebin_profiles, tc_loader, tcRange
from Utils.json_tools import read_formatted_json


def load_hd_tc(path):
    """Read saved tuning-curve JSON without classifying or transforming it."""
    return read_formatted_json(path)


def load_hd_comparison(mouseid, base_dir, probe, date, sessionID, *, hd_class=3):
    """Read saved Hz and attach exposure when the session TC matches the CSV."""
    session_dir = Path(base_dir) / mouseid / str(date) / f"{date}_{sessionID}"
    path = session_dir / "data/tc_comparison" / f"hd_class{hd_class}_Probe{probe}.csv"
    table = tc_loader(path, label=f"{mouseid} {date} Probe{probe} HD",
                      range=tcRange(False), response_units="Hz")
    source = session_dir / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
    if source.is_file():
        data = load_hd_tc(source)
        if len(data["occupancy_time_s"]) % len(table.columns) == 0:
            native = hd_profiles(data, probe=None, bins=len(table.columns))
            # Exported keys end with the source unit ID. Compare rates before
            # using exposure from a TC that may have been regenerated later.
            native.index = native.index.astype(str)
            ids = table.index.str.rsplit(":", n=1).str[-1]
            if (ids.isin(native.index).all() and table.columns.isin(native.columns).all()
                    and np.allclose(table, native.loc[ids, table.columns],
                                    rtol=1e-12, atol=1e-12, equal_nan=True)):
                table.attrs["hd_occupancy_s"] = native.attrs["hd_occupancy_s"]
    return table


def prep_hd_tc(table, *, bins=30):
    """Pool observed spike counts and exposure before normalizing each HD row."""
    if bins < 1 or table.shape[1] % bins:
        raise ValueError("HD comparison bins must be a positive divisor of the source bin count")
    if bins == table.shape[1]:
        return normalize_tc(table)
    occupancy = table.attrs.get("hd_occupancy_s", {})
    if not all(angle in occupancy for angle in table.columns):
        raise ValueError("Rebinning HD rates requires matching per-angle exposure from tuning_curves.tc")
    seconds = np.array([occupancy[angle] for angle in table.columns])
    counts = table.mul(seconds, axis="columns")
    counts.loc[:, seconds == 0] = 0
    pooled = rebin_profiles(counts, bins=bins)
    seconds = seconds.reshape(bins, -1).sum(axis=1)
    rates = pooled.div(np.where(seconds > 0, seconds, np.nan), axis="columns")
    rates.attrs["hd_occupancy_s"] = dict(zip(rates.columns, seconds, strict=True))
    return normalize_tc(rates)


def hd_pick(table, hd_class=3):
    """Copy units in one or more HD classes; None retains every class."""
    info = table.attrs["unit_info"]
    selected = (np.ones(len(table), dtype=bool) if hd_class is None else
                np.isin([info[key]["hd_class"] for key in table.index], hd_class))
    result = table.loc[selected].copy()
    result.attrs["unit_info"] = {key: info[key].copy() for key in result.index}
    return result


def hd_profiles(data, *, probe="A", bins=30, smoothing_deg=0, hd_class=None):
    """Rebin supplied counts and occupancy; None retains the native bin grid."""
    classes = data["unit_data"]["hd_class"]
    ids = np.asarray(data["unit_id"], dtype=int)
    counts = np.asarray(data["spike_counts"], dtype=float)
    occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
    n_native = len(occupancy)
    bins = n_native if bins is None else bins
    if smoothing_deg:
        counts = gaussian_filter1d(counts, smoothing_deg / (360 / n_native), axis=1, mode="wrap")
        occupancy = gaussian_filter1d(occupancy, smoothing_deg / (360 / n_native), mode="wrap")
    counts = counts.reshape(len(ids), bins, n_native // bins).sum(axis=2)
    occupancy = occupancy.reshape(bins, n_native // bins).sum(axis=1)
    rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan), where=occupancy > 0)
    edges = np.asarray(data.get("angle_bin_edges_deg", np.linspace(0, 360, n_native + 1)))
    edges = edges[::n_native // bins]
    angles = (edges[:-1] + edges[1:]) / 2
    profiles = profile_table(rates, ids, angles, probe=probe)
    profiles.attrs["response_units"] = "Hz"
    profiles.attrs["hd_occupancy_s"] = dict(zip(profiles.columns, occupancy, strict=True))
    profiles.attrs["unit_info"] = {
        key: {"hd_class": value} for key, value in zip(profiles.index, classes, strict=True)
    }
    return profiles if hd_class is None else hd_pick(profiles, hd_class)
