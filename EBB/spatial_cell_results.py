"""Save complete spatial plotting inputs from the analyzed recording interval."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix, save_npz, vstack


def spatial_result_path(session_dir, *, probe_name="A", phase_key="baseline"):
    """Return the analysis output path shared by analysis and plotting."""
    return (Path(session_dir) / "data" / "spatial_cells" / f"Probe{probe_name}"
            / phase_key / "egocentric_rate_map.rfmap")


def collapse_distance_band(rate_maps, distance_edges, band):
    """Sum a distance band, retaining angle bins and missing versus zero responses.

    Bins are selected by their centers in (lower, upper]; None leaves that
    side open. This is a sum of the stored rates, not occupancy pooling.
    """
    lower, upper = band
    centers = (distance_edges[:-1] + distance_edges[1:]) / 2
    selected = np.ones(len(centers), dtype=bool)
    if lower is not None:
        selected &= centers > lower
    if upper is not None:
        selected &= centers <= upper
    contributing = np.asarray(rate_maps, dtype=np.float64)[:, :, selected]
    collapsed = np.nansum(contributing, axis=2, keepdims=True)
    collapsed[~np.isfinite(contributing).any(axis=2, keepdims=True)] = np.nan
    start = distance_edges[0] if lower is None else lower
    end = distance_edges[-1] if upper is None else upper
    # An open-ended band beyond the grid retains an empty physical extent.
    return collapsed, np.array([start, max(start, end)]), centers[selected]


def save_spatial_plot_inputs(directory, pose, times, spikes_by_unit, unit_ids,
                            session_maps, rate_maps, metadata):
    from EBB.spatial_cell_analysis import count_spikes_by_frame, prepare_2d_projection

    directory = Path(directory)
    counts = vstack([csr_matrix(count_spikes_by_frame(spikes_by_unit[u], times))
                     for u in unit_ids], format="csr")
    dt = float(np.median(np.diff(times)))
    geometry = session_maps["geometry_metadata"]
    left, right, top, bottom = geometry["boundaryPx"]
    width, height = geometry["arenaSizeCm"]
    corrected = session_maps["corrected_positions_px"]
    x = (corrected[:, 0] - left) * width / (right - left)
    y = (corrected[:, 1] - top) * height / (bottom - top)
    xedges, yedges = np.linspace(0, width, 41), np.linspace(0, height, 41)
    projection = session_maps["spike_projection"]
    ego_shape = (len(session_maps["theta_edges"]) - 1, len(session_maps["distance_edges"]) - 1)
    ego_occupancy = (projection @ np.full(len(times), dt)).reshape(ego_shape)
    ego_spikes = (projection @ counts.T).toarray().T.reshape(len(unit_ids), *ego_shape)
    xy_projection = prepare_2d_projection(x, y, xedges, yedges)
    xy_occupancy = (xy_projection @ np.full(len(times), dt)).reshape(40, 40)
    xy_spikes = (xy_projection @ counts.T).toarray().T.reshape(len(unit_ids), 40, 40)
    xy_denominator = gaussian_filter(xy_occupancy, 1.5, mode="nearest")
    xy_numerator = gaussian_filter(xy_spikes.astype(float), (0, 1.5, 1.5), mode="nearest")
    xy_rates = np.divide(xy_numerator, xy_denominator[None],
                         out=np.full_like(xy_numerator, np.nan), where=xy_denominator[None] > 0)
    theta_edges = session_maps["theta_edges"]
    hd_bins = np.searchsorted(theta_edges, pose.hd_deg.to_numpy() % 360, side="right") - 1
    hd_entries = np.ones(len(times)), (hd_bins, np.arange(len(times)))
    hd_projection = csr_matrix(hd_entries, shape=(len(theta_edges) - 1, len(times)))
    hd_occupancy = np.asarray(hd_projection.sum(axis=1)).ravel() * dt
    hd_spikes = (hd_projection @ counts.T).toarray().T
    hd_denominator = gaussian_filter(hd_occupancy, 1.5, mode="wrap")
    hd_numerator = gaussian_filter(hd_spikes, (0, 1.5), mode=("nearest", "wrap"))
    hd_rates = np.divide(hd_numerator, hd_denominator[None],
                         out=np.full_like(hd_numerator, np.nan), where=hd_denominator[None] > 0)
    np.savez_compressed(directory / "restored_maps.npz", unit_ids=unit_ids,
                        theta_edges=theta_edges, distance_edges=session_maps["distance_edges"],
                        x_edges=xedges, y_edges=yedges, trajectory_x_cm=x, trajectory_y_cm=y,
                        frame_times=times, frame_ids=pose.frame.to_numpy(), egocentric_occupancy=ego_occupancy,
                        egocentric_spike_maps=ego_spikes, egocentric_rate_maps=rate_maps,
                        allocentric_occupancy=xy_occupancy, allocentric_spike_maps=xy_spikes,
                        allocentric_rate_maps=xy_rates, allocentric_tuning_curves=hd_rates)
    save_npz(directory / "frame_spike_counts.npz", counts)
    save_position_correction(directory, pose, times, session_maps)
    audit = dict(metadata, status="complete", unit_count=len(unit_ids),
                 bounds_px=geometry["boundaryPx"], arena_size_cm=geometry["arenaSizeCm"],
                 cm_per_pixel=geometry["cmPerPixelXY"], angle_sampling="bin_centers",
                 heading_convention=geometry["headingConvention"],
                 raw_heading_convention=geometry["rawHeadingConvention"],
                 heading_calibration=geometry["headingCalibration"],
                 position_correction_policy=geometry["positionCorrectionPolicy"],
                 held_position_frames=geometry["heldPositionCount"],
                 initial_outside_frames=geometry["initialOutsideCount"])
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def save_position_correction(directory, pose, times, session_maps):
    """Persist analyzed rays and tracking with the loader's full camera clock."""
    np.savez_compressed(Path(directory) / "position_correction.npz",
                        raw_xy_px=session_maps["raw_positions_px"],
                        corrected_xy_px=session_maps["corrected_positions_px"],
                        position_held=session_maps["position_held"],
                        initial_outside=session_maps["initial_outside"],
                        frame_ids=pose.frame.to_numpy(), frame_times=times,
                        hd_deg=pose.hd_deg.to_numpy(), ebc_hd_deg=session_maps["ebc_hd_deg"],
                        theta_deg=session_maps["theta_grid"][0], theta_d_cm=session_maps["theta_d_cm"],
                        exposure_times=pose.attrs["exposure_times"])
