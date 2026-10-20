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
    """Read an existing comparison CSV; missing files are not generated here."""
    session_dir = Path(base_dir) / mouseid / str(date) / f"{date}_{sessionID}"
    path = session_dir / "data/tc_comparison" / f"hd_class{hd_class}_Probe{probe}.csv"
    return tc_loader(path, label=f"{mouseid} {date} Probe{probe} HD",
                     range=tcRange(False), response_units="Hz")


def prep_hd_tc(table, *, bins=30):
    """Rebin and normalize a supplied comparison table without reading or saving."""
    return normalize_tc(rebin_profiles(table, bins=bins))


def hd_pick(table, hd_class=3):
    """Copy units in one or more HD classes; None retains every class."""
    info = table.attrs["unit_info"]
    selected = (np.ones(len(table), dtype=bool) if hd_class is None else
                np.isin([info[key]["hd_class"] for key in table.index], hd_class))
    result = table.loc[selected].copy()
    result.attrs["unit_info"] = {key: info[key].copy() for key in result.index}
    return result


def hd_profiles(data, *, probe="A", bins=30, smoothing_deg=0, hd_class=None):
    """Rebin supplied counts and occupancy, retaining the saved HD classes."""
    classes = data["unit_data"]["hd_class"]
    ids = np.asarray(data["unit_id"], dtype=int)
    counts = np.asarray(data["spike_counts"], dtype=float)
    occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
    n_native = len(occupancy)
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
    profiles.attrs["unit_info"] = {
        key: {"hd_class": value} for key, value in zip(profiles.index, classes, strict=True)
    }
    return profiles if hd_class is None else hd_pick(profiles, hd_class)
