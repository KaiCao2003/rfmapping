"""Explicit spatial smoothing of loaded, window-averaged RF rates."""

import numpy as np

from .models import RFMapList, _make_rf_map


def smooth_rf_maps(maps: RFMapList) -> RFMapList:
    """Apply one GUI smoothing pass after raw-map QC, without file I/O.

    Smooth counts and presentation exposure separately, then divide. This
    equals smoothing Hz directly only for uniform presentations. The 3x3
    1/2/1 kernel truncates at both boundaries; it does not wrap or reflect.
    Missing centers stay missing, while true zero rates remain valid.
    Input values, native coordinates, and presentation counts are unchanged.
    """
    output = []
    preprocessing = {
        "algorithm": "gui-weighted-spatial-smoothing-v1", "iterations": 1,
        "kernel_weights": [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        "boundaries": "truncate and renormalize finite neighbor weights; no wrap or reflection",
        "missing_centers": "remain missing", "zero_counts": "valid",
        "rate_formula": "S(counts) / (S(presentations) * duration_s)",
        "order": "raw-map QC -> GUI smoothing1 -> baseline mean/SD -> threshold -> component filter",
    }
    for item in maps:
        if (item.metadata.get("responseUnits") != "Hz"
                or item.metadata.get("responseNormalization") != "presentation_count_time"
                or item.presentation_counts is None):
            raise ValueError("smooth_rf_maps requires presentation-normalized Hz maps")
        if item.n_y < 2 or item.n_x < 2 or "spatialPreprocessing" in item.metadata:
            raise ValueError("smooth_rf_maps requires native 2-D maps not already smoothed")
        values = item.to_2d_array()
        exposure = item.presentation_counts
        valid = np.isfinite(values) & (exposure > 0)
        if "occupancyTimeSec" in item.metadata:
            occupancy = np.asarray(item.metadata["occupancyTimeSec"]).reshape(values.shape)
            valid &= occupancy > 0
        weighted = np.pad(np.where(valid, values * exposure, 0.), 1)
        presentations = np.pad(np.where(valid, exposure, 0.), 1)
        numerator, denominator = np.zeros_like(values), np.zeros_like(values)
        for dy, wy in enumerate((1, 2, 1)):
            for dx, wx in enumerate((1, 2, 1)):
                region = np.s_[dy:dy + item.n_y, dx:dx + item.n_x]
                numerator += wy * wx * weighted[region]
                denominator += wy * wx * presentations[region]
        smoothed = np.divide(numerator, denominator, out=np.full_like(values, np.nan),
                             where=valid & (denominator > 0))
        output.append(_make_rf_map(
            unit_index=item.unit_index, unit_id=item.unit_id,
            spike_counts=smoothed[..., None], x_positions=item.x_positions,
            y_positions=item.y_positions, time_bin_edges_s=item.time_bin_edges_s,
            presentation_counts=item.presentation_counts, source_path=item.source_path,
            metadata={**item.metadata, "spatialPreprocessing": preprocessing},
        ))
    return RFMapList(output, maps.source_path)
