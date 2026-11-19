"""Occupancy-normalized HD × boundary-bearing maps from loaded frame data."""

import numpy as np
from scipy.ndimage import gaussian_filter


def hd_boundary_joint_maps(projection, frame_counts, frame_times, hd_deg,
                           theta_edges, distance_edges, bands, *, hd_bins, smoothing_deg):
    """Condition each native boundary ray on HD and a native distance-bin band.

    A frame can contribute to several bearings, just as in the saved EBB maps.
    Rays are separate spatial conditions, not independent spike observations.
    Smooth counts and occupancy separately on both circular angular axes;
    retain raw counts/exposure and mask originally unvisited bins in the display.
    """
    counts = np.asarray(frame_counts, dtype=float)
    times = np.asarray(frame_times, dtype=float)
    headings = np.asarray(hd_deg, dtype=float)
    shape = (len(theta_edges) - 1, len(distance_edges) - 1)
    if counts.shape != times.shape or headings.shape != times.shape:
        raise ValueError('HD, spike counts and frame times must share the frame axis')
    if projection.shape != (int(np.prod(shape)), len(times)):
        raise ValueError('Boundary projection does not match the native bins and frames')
    dt = float(np.median(np.diff(times)))
    hd_edges = np.linspace(0, 360, hd_bins + 1)
    finite_hd = np.isfinite(headings)
    hd_index = np.full(len(headings), -1, dtype=int)
    hd_index[finite_hd] = np.searchsorted(hd_edges, headings[finite_hd] % 360, side='right') - 1
    joint_counts = np.zeros((hd_bins, *shape))
    joint_occupancy = np.zeros_like(joint_counts)
    for index in range(hd_bins):
        keep = hd_index == index
        subset = projection[:, keep]
        joint_counts[index] = (subset @ counts[keep]).reshape(shape)
        joint_occupancy[index] = (subset @ np.full(keep.sum(), dt)).reshape(shape)
    sigma = (smoothing_deg / (360 / hd_bins), smoothing_deg / np.diff(theta_edges)[0])
    results = {}
    for name, (start, stop) in bands.items():
        raw = joint_counts[:, :, start:stop].sum(axis=2)
        occupancy = joint_occupancy[:, :, start:stop].sum(axis=2)
        rates = np.divide(raw, occupancy, out=np.full_like(raw, np.nan), where=occupancy > 0)
        smoothed_counts = gaussian_filter(raw, sigma, mode='wrap')
        smoothed_occupancy = gaussian_filter(occupancy, sigma, mode='wrap')
        displayed = np.divide(smoothed_counts, smoothed_occupancy,
                              out=np.full_like(raw, np.nan), where=smoothed_occupancy > 0)
        displayed[occupancy == 0] = np.nan
        results[name] = dict(spike_counts=raw, occupancy_s=occupancy, raw_hz=rates, hz=displayed)
    return dict(hd_edges=hd_edges, theta_edges=np.asarray(theta_edges),
                spike_maps=joint_counts, occupancy_maps=joint_occupancy, bands=results,
                frame_period_s=dt, n_frames=int(finite_hd.sum()), smoothing_deg=smoothing_deg)
