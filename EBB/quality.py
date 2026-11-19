"""Descriptive boundary-map stability from loaded frame counts and geometry."""

import numpy as np

from .ebc_analysis import rates_from_counts


def map_correlation(first, second):
    """Pearson r over bins measured in both maps; constant maps are undefined."""
    shared = np.isfinite(first) & np.isfinite(second)
    a, b = first[shared], second[shared]
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def boundary_map_quality(projection, frame_counts, frame_times, held, shape, *, sigma):
    """Reconstruct full, temporal-half and unheld maps without reading files.

    ``projection`` is native bearing/distance bins × frames; ``frame_counts``
    is units × frames. Exposure uses the saved spatial pipeline's median frame
    period, including for each temporal subset. No stability/significance
    threshold is imposed. Unvisited bins remain missing in all statistics.
    """
    times = np.asarray(frame_times, dtype=float)
    held = np.asarray(held, dtype=bool)
    if (projection.shape != (int(np.prod(shape)), len(times))
            or frame_counts.shape[1] != len(times) or held.shape != times.shape):
        raise ValueError('Geometry, counts and held flags must share the frame axis')
    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Boundary-map quality requires increasing frame times')
    dt = float(np.median(np.diff(times)))
    middle = float(np.median(times))
    selections = [np.ones(len(times), dtype=bool), times <= middle, times > middle, ~held]
    occupancies, rates, mean_hz = [], [], []
    for index, keep in enumerate(selections):
        subset = projection[:, keep]
        occupancy = (subset @ np.full(keep.sum(), dt)).reshape(shape)
        counts = (subset @ frame_counts[:, keep].T).toarray().T.reshape((-1, *shape))
        smoothed = rates_from_counts(counts, occupancy, sigma=sigma, mask_unvisited=False)
        if index == 0:
            full_counts, display_rates = counts, smoothed.copy()
        smoothed[:, occupancy == 0] = np.nan
        occupancies.append(occupancy)
        rates.append(smoothed)
        totals = np.asarray(frame_counts[:, keep].sum(axis=1)).ravel()
        mean_hz.append(totals / (keep.sum() * dt) if keep.any() else np.full(len(totals), np.nan))

    records = []
    for row, full in enumerate(rates[0]):
        measured = full[np.isfinite(full)]
        q05, q95 = np.quantile(measured, [.05, .95]) if measured.size else (np.nan, np.nan)
        modulation = float((q95 - q05) / q95) if q95 > 0 else np.nan
        half_r = map_correlation(rates[1][row], rates[2][row])
        records.append(dict(
            half_r=half_r, unheld_r=map_correlation(full, rates[3][row]),
            modulation_q95_q05=modulation,
            descriptive_score=max(half_r, 0) * modulation,
            half_shared_bins=int((np.isfinite(rates[1][row]) & np.isfinite(rates[2][row])).sum()),
            n_spikes=int(frame_counts[row].sum()), mean_hz=float(mean_hz[0][row]),
            first_half_hz=float(mean_hz[1][row]), second_half_hz=float(mean_hz[2][row]),
        ))
    return dict(
        occupancy=occupancies[0], spike_maps=full_counts, display_rates=display_rates,
        rates=rates[0], half_rates=np.stack(rates[1:3]), unheld_rates=rates[3],
        half_occupancy=np.stack(occupancies[1:3]), unheld_occupancy=occupancies[3],
        metrics=records,
        summary=dict(frame_period_s=dt, split_time_s=middle, n_frames=len(times),
                     held_fraction=float(held.mean()),
                     coverage=float((occupancies[0] > 0).mean()),
                     unheld_coverage=float((occupancies[3] > 0).mean()),
                     half_coverage=[float((occ > 0).mean()) for occ in occupancies[1:3]]),
    )
