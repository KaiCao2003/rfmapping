"""Native presentation exposure for saved free-moving bar RF maps."""

import numpy as np


def free_moving_exposure(trials, onset_edges_s, position_xy, hd_offset_px, calibration,
                        *, n_x_bins, screen_width_px, screen_deg, luminance=1):
    """Reproduce RFmapping_core_fm geometry using already-loaded recording data.

    XY and screen radius must share the calibration's units. Heading offsets
    are calibrated screen pixels. Each ON trial contributes once to a coarse
    bin if any native screen pixel is covered, including across the wrap seam.
    Counts normalize peri-stimulus rates; inter-onset duration reproduces the
    separately saved occupancy, not the RF response-window exposure.
    """
    onsets = np.asarray(onset_edges_s, dtype=float).ravel()
    xy = np.asarray(position_xy, dtype=float)
    hd = np.asarray(hd_offset_px, dtype=float).ravel()
    n = len(onsets) - 1
    if n < 1 or len(trials) < n or xy.shape != (n, 2) or hd.shape != (n,):
        raise ValueError('Free-moving trials, onset intervals, XY and heading must align')
    if not np.all(np.isfinite(onsets)) or np.any(np.diff(onsets) <= 0):
        raise ValueError('Free-moving onset boundaries must be finite and increasing')
    if screen_width_px % n_x_bins:
        raise ValueError('RF bins must divide the native screen pixel grid')
    trials = trials[:n]
    if any(trial['Square_PositionY'] != 0 for trial in trials):
        raise ValueError('Free-moving bar exposure requires the singleton Y=0 row')
    stimulus = np.asarray([t['Square_PositionX'] for t in trials], dtype=float)
    width = np.asarray([t['Square_Size'] for t in trials], dtype=float)
    selected = np.asarray([t['Square_Luminance'] for t in trials]) == luminance
    px_per_deg = screen_width_px / screen_deg
    radius = calibration['screen_diamter'] / 2
    zero = calibration['screen_center']
    head = hd / px_per_deg + zero

    def bearing(stimulus_edge):
        angle = np.deg2rad(zero - stimulus_edge)
        world = np.rad2deg(np.arctan2(radius * np.sin(angle) - xy[:, 1],
                                     radius * np.cos(angle) - xy[:, 0]))
        return -((world - head + 180) % 360 - 180)

    left, right = bearing(stimulus - width / 2), bearing(stimulus + width / 2)
    ego_width = (right - left) % 360
    center_px = (left + ego_width / 2 + screen_deg / 2) * px_per_deg
    half_width_px = ego_width * px_per_deg / 2
    mask = np.zeros((n, n_x_bins), dtype=bool)
    pixels = np.arange(screen_width_px)
    # Bound temporary trial × pixel arrays without changing native binning.
    for start in range(0, n, 512):
        end = min(start + 512, n)
        dx = ((pixels - center_px[start:end, None] + screen_width_px / 2)
              % screen_width_px - screen_width_px / 2)
        covered = ((dx >= -half_width_px[start:end, None])
                   & (dx < half_width_px[start:end, None]) & selected[start:end, None])
        mask[start:end] = covered.reshape(end - start, n_x_bins, -1).any(axis=2)
    return dict(presentation_counts=mask.sum(axis=0),
                occupancy_s=np.diff(onsets) @ mask, trial_bin_mask=mask)
