"""Prepare EBB angular profiles from already loaded maps."""

import numpy as np
import pandas as pd

from Utils.direction_comparison import profile_table, recording_profiles, resample_profiles, tcRange


def boundary_band_profiles(rates, occupancy, theta_edges, distance_edges, unit_keys, *, bins=30):
    """Average measured rate cells over distance and adjacent angular bins.

    Retain the saved rate-map convention used in the poster. This is an
    equal-cell mean, not pooled spikes/exposure. Zero occupancy stays missing.
    """
    rates = np.asarray(rates, dtype=float)
    native_bins = len(theta_edges) - 1
    if native_bins % bins:
        raise ValueError('Comparison bins must divide the native bearing bins')
    values = np.where(np.asarray(occupancy)[None] > 0, rates, np.nan)
    centers = (np.asarray(distance_edges[:-1]) + distance_edges[1:]) / 2
    first, second = np.searchsorted(centers, [8., 16.], side='right')
    bands = {'all': (0, len(centers)), '0-8': (0, first),
             '8-16': (first, second), '16-inf': (second, len(centers))}
    group = native_bins // bins
    angles = (np.asarray(theta_edges[:-1]) + theta_edges[1:]) / 2
    angles = angles.reshape(bins, group).mean(axis=1)
    results = {}
    for name, (start, stop) in bands.items():
        cells = values[:, :, start:stop].reshape(len(values), bins, group, stop-start)
        count = np.isfinite(cells).sum(axis=(2, 3))
        means = np.divide(np.nansum(cells, axis=(2, 3)), count,
                          out=np.full(count.shape, np.nan), where=count > 0)
        table = pd.DataFrame(means, index=unit_keys, columns=(angles + 180) % 360 - 180)
        table.attrs.update(label=f'EBB {name}', range=tcRange(True), response_units='Hz',
                           distance_bin_indices=[int(start), int(stop)],
                           distance_support_cm=[float(distance_edges[start]), float(distance_edges[stop])])
        results[name] = table
    return results


def ebc_profiles(maps, *, probe="A"):
    """Collapse supplied maps over distance, retaining native angles and NaNs."""
    angles = maps[0].y_positions
    projected = maps.sum_to_1d(axis="y")
    profiles = profile_table(projected.to_1d_array(axis="y"), maps.unit_ids, angles, probe=probe)
    profiles.attrs["response_units"] = maps[0].metadata.get("responseUnits", "spike_count")
    profiles.attrs["unit_info"] = {key: {} for key in profiles.index}
    return profiles


def prepare_ebc(maps, *, mouse, date, probe="A", label="EBC", range: list[int] | None = None):
    """Resample supplied EBB maps to the comparison grid and add recording keys."""
    profiles = ebc_profiles(maps, probe=probe)
    profiles = resample_profiles(profiles, range=tcRange(True) if range is None else range)
    table = recording_profiles(profiles, mouse=mouse, date=date)
    table.attrs["label"] = label
    return table
