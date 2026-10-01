"""Decode session C's spikes from its native, unsmoothed saved HD curves."""

import numpy as np
import pandas as pd
from scipy.special import xlogy

from Utils.json_tools import read_formatted_json
from Utils.tuning_curve_utils import update_hd_classification


def load_decoder_tuning_curves(tc_path, hd_class=3):
    """Select the current HD class and divide saved counts by occupancy seconds.

    Classification runs in memory using the repository's existing definition.
    The saved TC, native angular bins, and zero firing rates remain unchanged.
    """
    tc = read_formatted_json(tc_path)
    update_hd_classification(tc)
    selected = np.asarray(tc["unit_data"]["hd_class"], dtype=object) == hd_class
    occupancy = np.asarray(tc["occupancy_time_s"], dtype=float)
    if not np.all(np.isfinite(occupancy) & (occupancy > 0)):
        raise ValueError("Decoding requires positive saved occupancy in every native HD bin.")
    if not selected.any():
        raise ValueError(f"Session C has no HD Class {hd_class} units.")
    counts = np.asarray(tc["spike_counts"], dtype=float)[selected]
    edges = np.asarray(tc["angle_bin_edges_deg"], dtype=float)
    curves = pd.DataFrame(
        counts / occupancy,
        index=pd.Index(np.asarray(tc["unit_id"], dtype=int)[selected], name="unit_id"),
        columns=(edges[:-1] + edges[1:]) / 2,
    )
    curves.attrs["source"] = str(tc_path)
    curves.attrs["metadata"] = tc["metadata"]
    return curves


def decode_hd_frames(curves, spike_times, spike_clusters, frame_times, bin_size_s, epochs,
                     *, chunk_bins=128):
    """Return native MAP HD for each frame's complete half-open spike-count bin.

    All times are seconds relative to session C's ADC origin. Exact Poisson
    likelihoods retain zero rates: a spike at a zero-rate angle is impossible.
    Silent bins, impossible observations, and frames outside full bins are NaN.
    Camera pauses must be separate rows of ``epochs``; angles are never filled.
    """
    rates = curves.to_numpy(dtype=float)
    if not np.all(np.isfinite(rates) & (rates >= 0)):
        raise ValueError("Saved HD rates must be finite and nonnegative.")
    if bin_size_s <= 0 or chunk_bins < 1:
        raise ValueError("Decoding bin size and chunk size must be positive.")
    spike_times = np.asarray(spike_times, dtype=float).reshape(-1)
    spike_clusters = np.asarray(spike_clusters).reshape(-1)
    if spike_times.shape != spike_clusters.shape:
        raise ValueError("Spike times and cluster IDs must have the same shape.")
    frame_times = np.asarray(frame_times, dtype=float)
    units = curves.index.to_numpy(dtype=int)
    angles = curves.columns.to_numpy(dtype=float)

    selected = np.isin(spike_clusters, units)
    selected_clusters = spike_clusters[selected]
    selected_times = spike_times[selected]
    order = np.argsort(selected_clusters, kind="stable")
    selected_clusters, selected_times = selected_clusters[order], selected_times[order]
    starts = np.searchsorted(selected_clusters, units, side="left")
    ends = np.searchsorted(selected_clusters, units, side="right")
    unit_times = [np.sort(selected_times[start:end]) for start, end in zip(starts, ends, strict=True)]

    values = np.full(frame_times.shape, np.nan)
    qc = dict(decoded_bins=0, valid_bins=0, silent_bins=0, impossible_bins=0)
    rate_penalty = bin_size_s * rates.sum(axis=0)
    for start, end in np.asarray(epochs, dtype=float).reshape(-1, 2):
        n_bins = int(np.floor((end - start) / bin_size_s + 1e-9))
        for offset in range(0, n_bins, chunk_bins):
            count = min(chunk_bins, n_bins - offset)
            # Use the same 9-decimal clock precision as the existing decoder.
            edges = np.round(start + np.arange(offset, offset + count + 1) * bin_size_s, 9)
            counts = np.stack([
                np.diff(np.searchsorted(times, edges, side="left")) for times in unit_times
            ], axis=1)
            log_likelihood = xlogy(counts[:, None, :], rates.T[None, :, :]).sum(axis=2)
            log_likelihood -= rate_penalty
            silent = counts.sum(axis=1) == 0
            possible = np.isfinite(log_likelihood).any(axis=1)
            valid = ~silent & possible
            decoded = np.full(count, np.nan)
            decoded[valid] = angles[np.argmax(log_likelihood[valid], axis=1)]
            rows = (frame_times >= edges[0]) & (frame_times < edges[-1])
            indices = np.searchsorted(edges[1:], frame_times[rows], side="right")
            values[rows] = decoded[indices]
            qc["decoded_bins"] += count
            qc["valid_bins"] += int(valid.sum())
            qc["silent_bins"] += int(silent.sum())
            qc["impossible_bins"] += int((~silent & ~possible).sum())
    return values, qc
