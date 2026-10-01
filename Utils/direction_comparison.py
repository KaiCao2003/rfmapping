"""Shared unit-keyed angular profiles, peak sorting, and circular alignment."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from Utils.json_tools import read_formatted_json
from Utils.plotting import LIGHT_PLOT_STYLE, angular_ticks, plot_keyed_heatmap
from Utils.rfmap import load_rf_maps
from Utils.statistic_utils import rayleigh_uniformity
from Utils.tuning_curve_utils import update_hd_classification


def tcRange(is_ego: bool) -> list[int]:
    """Default signed ego or wrapped allo angle labels, covering a full circle."""
    return [-180, -90, 0, 90, 180] if is_ego else [180, 270, 0, 90, 180]


def _range_ticks(range):
    """Resolve two boundaries or circular tick labels to increasing degrees."""
    labels = np.asarray(range, dtype=float)
    if labels.ndim != 1 or len(labels) < 2 or not np.all(np.isfinite(labels)):
        raise ValueError("range must contain at least two finite degree values")
    ticks = labels if np.all(np.diff(labels) > 0) else angular_ticks(labels)
    if np.any(np.diff(ticks) <= 0) or ticks[-1] - ticks[0] > 360:
        raise ValueError("range must increase over at most 360 degrees")
    return ticks


def profile_coordinates(profiles):
    labels = profiles.attrs.get("range", tcRange(True))
    ticks = _range_ticks(labels)
    if len(labels) == 2:
        ticks = np.linspace(ticks[0], ticks[-1], 5)
        labels = ticks.tolist()
    return {"ticklabels": list(labels), "ticks": ticks}


def convert_profile_coordinates(profiles, *, range: list[int]):
    """Choose displayed labels without changing the source angles."""
    previous = _range_ticks(profiles.attrs.get("range", tcRange(True)))
    ticks = _range_ticks(range)
    if not np.array_equal(previous[[0, -1]], ticks[[0, -1]]):
        raise ValueError("display conversion must retain the angular extent; set range when loading")
    table = profiles.copy()
    table.attrs["range"] = list(range)
    return table


def gen_overlap_list(list1, list2):
    """Return sorted shared IDs; tuple keys keep probe identity."""
    return sorted(set(list1) & set(list2))


def profile_table(values, unit_ids, angles_deg, *, probe=None):
    """Wrap source bin centers to [-180, 180), retaining source order and direction."""
    index = (pd.Index(unit_ids, name="unit_id") if probe is None else
             pd.MultiIndex.from_product([[probe], unit_ids], names=["probe", "unit_id"]))
    angles = (np.asarray(angles_deg, dtype=float) + 180) % 360 - 180
    return pd.DataFrame(values, index=index, columns=angles)


def normalize_tc(tc):
    """Copy unit-by-bin curves, dividing each row by its finite maximum.

    Zero rows remain zero; missing bins remain NaN. Unit keys, angle order,
    and metadata are preserved. Apply this before passing curves to a plot.
    """
    result = tc.astype(float)
    values = result.where(np.isfinite(result))
    row_max = values.max(axis=1)
    result.iloc[:, :] = values.div(row_max.mask(row_max == 0, 1.0), axis=0)
    return result


def zscore_tc(tc):
    """Copy unit-by-bin curves as SD from each row's mean over finite bins.

    Bins have equal weight and SD uses ddof=0. Missing bins remain NaN;
    rows with zero SD (including a single measured bin) become all NaN.
    Unit keys, angle order, and metadata are preserved.
    """
    result = tc.astype(float)
    values = result.where(np.isfinite(result))
    mean = values.mean(axis=1)
    sd = values.std(axis=1, ddof=0)
    result.iloc[:, :] = values.sub(mean, axis=0).div(sd.where(sd > 0), axis=0)
    return result


def peak_angles(profiles, units, *, use_min=False):
    """Select the maximum or minimum bin; missing curves return NaN."""
    rows = profiles.reindex(units).to_numpy()
    present = np.any(np.isfinite(rows), axis=1)
    peaks = np.full(len(rows), np.nan)
    select_bin = np.nanargmin if use_min else np.nanargmax
    peaks[present] = np.asarray(profiles.columns)[select_bin(rows[present], axis=1)]
    return peaks


def sorted_by_peak(profiles, overlap):
    """Sort by peak then unit key; missing peaks last, ties use the first source bin."""
    units = sorted(overlap)
    peaks = peak_angles(profiles, units)
    return [units[i] for i in np.argsort(peaks, kind="stable")]


def paired_peak_angles(reference, matched, *, order=None, is_wrap=True,
                       reference_min=False, matched_min=False):
    """Select paired maxima/minima; order rearranges the full unit set only."""
    units = set(reference.index)
    if (not reference.index.is_unique or not matched.index.is_unique
            or units != set(matched.index)):
        raise ValueError("tables must have the same unique unit keys; select overlap before plotting")
    sort_by_reference = order is None
    if sort_by_reference:
        order = sorted(units)
    else:
        order = list(order)
        if len(order) != len(units) or set(order) != units:
            raise ValueError("order must be a permutation of all input unit keys")
    reference_peak = peak_angles(reference, order, use_min=reference_min)
    matched_peak = peak_angles(matched, order, use_min=matched_min)
    peak_sum = reference_peak + matched_peak
    if is_wrap:
        peak_sum = (peak_sum + 180) % 360 - 180
    table = pd.DataFrame({
        "reference_peak_deg": reference_peak, "matched_peak_deg": matched_peak,
        "peak_sum_deg": peak_sum,
    }, index=reference.loc[order].index)
    if sort_by_reference:
        table = table.sort_values("reference_peak_deg", kind="stable")
    table.attrs["is_wrap"] = is_wrap
    return table


def peak_direction_sums(hd_profiles, rf_profiles, *, order=None, is_wrap=True):
    """The same paired peaks as the heatmaps, labeled for the HD/RF summary."""
    return paired_peak_angles(hd_profiles, rf_profiles, order=order, is_wrap=is_wrap).rename(columns={
        "reference_peak_deg": "hd_peak_deg", "matched_peak_deg": "rf_peak_deg",
    })


def peak_sum_statistics(table):
    """Calculate histogram counts and circular concentration of paired peak sums."""
    is_wrap = table.attrs.get("is_wrap", True)
    limit = 180 if is_wrap else 360
    values = table.peak_sum_deg.to_numpy()
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=np.arange(-limit, limit + 1, 12))
    return {"counts": counts, "edges": edges, "statistics": rayleigh_uniformity(values)}


def plot_peak_direction_sums(summary, *, hd_label="HD", rf_label="RF", figsize=(6, 4)):
    """Render supplied peak-sum counts and circular statistics."""
    statistics = summary["statistics"]
    edges = summary["edges"]
    limit = edges[-1]
    p_label = f"{statistics['p']:.3g}" if np.isfinite(statistics["p"]) else "—"
    r_label = f"{statistics['r']:.2f}" if np.isfinite(statistics["r"]) else "—"
    with plt.rc_context(LIGHT_PLOT_STYLE):
        figure, axis = plt.subplots(figsize=figsize, layout="constrained")
        axis.stairs(summary["counts"], edges, fill=True, edgecolor="white")
        axis.set(title=f"{hd_label} + {rf_label}", xlabel="Peak sum (°)", ylabel="Units",
                 xlim=(-limit, limit), xticks=np.linspace(-limit, limit, 5))
        axis.yaxis.get_major_locator().set_params(integer=True)
        axis.margins(y=.3)
        axis.text(.98, .98, f"n = {statistics['n']} · R = {r_label}\nRayleigh p = {p_label}",
                  transform=axis.transAxes, ha="right", va="top", color="black",
                  bbox=dict(facecolor="white", edgecolor="none", alpha=1))
    return figure


def _interpolate_angles(angles, values, targets, range, *, is_wrap=True, fill_value=np.nan):
    """Interpolate in physical degrees without bridging a partial field's unseen arc."""
    ticks = _range_ticks(range)
    lower, upper = ticks[0], ticks[-1]
    if upper - lower == 360:
        if is_wrap:
            return np.interp(targets, angles, values, period=360)
        angles = (np.asarray(angles) + 180) % 360 - 180
        order = np.argsort(angles)
        return np.interp(targets, angles[order], values[order], left=fill_value, right=fill_value)
    angles = (np.asarray(angles) - lower) % 360 + lower
    order = np.argsort(angles)
    angles, values = angles[order], values[order]
    within = (angles >= lower) & (angles <= upper)
    angles, values = angles[within], values[within]
    if not len(angles):
        return np.full(len(targets), fill_value)
    # Edge bins cover half a bin beyond their centers; never bridge the unseen arc.
    angles = np.r_[lower, angles, upper]
    values = np.r_[values[0], values, values[-1]]
    targets = np.asarray(targets)
    query = (targets - lower) % 360 + lower
    result = np.interp(query, angles, values, left=fill_value, right=fill_value)
    if not is_wrap:
        result[(targets < -180) | (targets >= 180)] = fill_value
    return result


def _resample_profiles(profiles, centers):
    angle_range = profiles.attrs["range"]
    fill_value = 0. if profiles.attrs.get("response_kind") == "RF" else np.nan
    values = np.array([
        _interpolate_angles(profiles.columns, row, centers, angle_range, fill_value=fill_value)
        for row in profiles.to_numpy()
    ]).reshape(len(profiles), len(centers))
    table = pd.DataFrame(values, index=profiles.index, columns=centers)
    table.attrs = profiles.attrs.copy()
    return table


def align_profiles(profiles, units, offset_deg, *, is_wrap=True):
    """Add each unit's offset; unwrapped sums use a wider 12° grid without repetition."""
    relative_deg = np.arange(-180., 180., 12.) if is_wrap else np.arange(-360., 361., 12.)
    rows = profiles.reindex(units)
    angle_range = profiles.attrs.get("range", tcRange(True))
    fill_value = 0. if profiles.attrs.get("response_kind") == "RF" else np.nan
    shifted = np.array([
        _interpolate_angles(profiles.columns, row, relative_deg - offset,
                            angle_range, is_wrap=is_wrap, fill_value=fill_value)
        for row, offset in zip(rows.to_numpy(), offset_deg, strict=True)
    ]).reshape(len(units), len(relative_deg))
    result = pd.DataFrame(shifted, index=rows.index, columns=relative_deg)
    result.attrs = profiles.attrs.copy()
    result.attrs["range"] = tcRange(True) if is_wrap else [-360, -180, 0, 180, 360]
    if "unit_info" in profiles.attrs:
        result.attrs["unit_info"] = {
            key: profiles.attrs["unit_info"][key].copy()
            for key in result.index if key in profiles.attrs["unit_info"]
        }
    return result


def smooth_profiles(profiles, smoothing_bins=0):
    """Smooth angular neighbors, retaining gaps and avoiding partial-field wraparound."""
    if smoothing_bins == 0:
        return profiles
    ticks = _range_ticks(profiles.attrs.get("range", tcRange(True)))
    columns = np.argsort((np.asarray(profiles.columns) - ticks[0]) % 360 + ticks[0])
    values = profiles.to_numpy(dtype=float)[:, columns]
    finite = np.isfinite(values)
    mode = "wrap" if ticks[-1] - ticks[0] == 360 else "nearest"
    weights = gaussian_filter1d(finite.astype(float), smoothing_bins, axis=1, mode=mode)
    smoothed = gaussian_filter1d(np.where(finite, values, 0.), smoothing_bins, axis=1, mode=mode)
    np.divide(smoothed, weights, out=smoothed, where=weights > 0)
    smoothed[~finite] = np.nan
    result = profiles.copy()
    result.iloc[:, columns] = smoothed
    return result


def plot_profiles(profiles, order, name, *, axis=None, sort_name=None,
                  aligned=False, show=True, figsize=(10, 6), is_wrap=True,
                  reference_min=False, cmap="viridis", vmin=None, vmax=None,
                  colorbar_label="Response"):
    """Render supplied curve values without normalizing or standardizing them."""
    if (not profiles.index.is_unique or len(order) != len(profiles)
            or set(order) != set(profiles.index)):
        raise ValueError("order must be a permutation of all input unit keys")
    if axis is None:
        axis = {"ticklabels": tcRange(True)} if aligned else profile_coordinates(profiles)
        if aligned == "sum" and not is_wrap:
            axis = {"ticklabels": (-360, -180, 0, 180, 360)}
    ticks = axis["ticks"] if "ticks" in axis else angular_ticks(axis["ticklabels"])
    columns = np.asarray(profiles.columns)
    values = profiles.to_numpy()
    if aligned and is_wrap:
        # The -180° bin straddles the plot boundary; display its other half at +180°.
        columns = np.r_[columns, 180.]
        values = np.c_[values, values[:, 0]]
    elif not aligned:
        start = min(ticks[0], ticks[-1])
        columns = (columns - start) % 360 + start
    column_order = np.argsort(columns)
    xlabel = f"{name} direction (°)"
    suffix = ""
    reference_bin = "minimum" if reference_min else "peak"
    if aligned:
        xlabel = f"Angle from reference {reference_bin} (°)"
        suffix = f" | {reference_bin} aligned"
    if aligned == "sum":
        xlabel = f"{name} angle + {sort_name} {reference_bin} (°{'; wrapped' if is_wrap else ''})"
        suffix = f" | {reference_bin} sum"
    return plot_keyed_heatmap(
        dict(zip(profiles.index, values)), order,
        column_order=column_order, xticks=ticks,
        xticklabels=axis["ticklabels"], angle_centers=columns[column_order],
        xlabel=xlabel, title=f"{name} | sorted by {sort_name or name}{suffix}",
        show=show, figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar_label=colorbar_label,
    )


def prepare_comparison(reference, matched, *, mode="native", order=None, is_wrap=True,
                       reference_min=False, matched_min=False):
    """Calculate paired peaks, row order, and explicit angular transformations.

    Native keeps source angles. Aligned subtracts the reference direction from
    both tables. Sum centers the reference and adds its direction to the match.
    """
    if mode not in ("native", "aligned", "sum"):
        raise ValueError("mode must be 'native', 'aligned', or 'sum'")
    peaks = paired_peak_angles(reference, matched, order=order, is_wrap=is_wrap,
                               reference_min=reference_min, matched_min=matched_min)
    order = peaks.index.tolist()
    reference_peak = peaks.reference_peak_deg.fillna(0).to_numpy()
    offsets = np.zeros(len(order)) if mode == "native" else -reference_peak
    matched_offsets = reference_peak if mode == "sum" else offsets
    reference_panel = reference.loc[order]
    matched_panel = matched.loc[order]
    if mode != "native":
        reference_panel = align_profiles(reference, order, offsets)
        matched_panel = align_profiles(matched, order, matched_offsets,
                                       is_wrap=is_wrap if mode == "sum" else True)
    return {
        "reference": reference_panel, "matched": matched_panel,
        "order": order, "reference_peak_deg": peaks.reference_peak_deg.to_numpy(),
        "matched_peak_deg": peaks.matched_peak_deg.to_numpy(), "peak_sum_deg": peaks.peak_sum_deg.to_numpy(),
        "offset_deg": offsets, "matched_offset_deg": matched_offsets, "mode": mode,
        "peaks": peaks, "is_wrap": is_wrap, "reference_min": reference_min,
    }


def plot_comparison_heatmaps(comparison, *, sorted_by_axis=None, apply_to_axis=None,
                             save_dir=None, show=True, figsize=(10, 6), labels=None,
                             names=(0, 1), cmap="viridis", vmin=None, vmax=None,
                             colorbar_label="Response"):
    """Render the ordered and transformed tables from prepare_comparison."""
    reference, matched = comparison["reference"], comparison["matched"]
    labels = ((reference.attrs.get("label", "Reference"), matched.attrs.get("label", "Matched"))
              if labels is None else labels)
    mode = comparison["mode"]
    directory = Path(save_dir) if save_dir is not None else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    figures = []
    for name, profiles, axis, layout, label in (
        (names[0], reference, sorted_by_axis, mode != "native", labels[0]),
        (names[1], matched, apply_to_axis, "sum" if mode == "sum" else mode != "native", labels[1]),
    ):
        figure, axes = plot_profiles(
            profiles, comparison["order"], label, axis=axis, sort_name=labels[0], aligned=layout,
            show=False, figsize=figsize, is_wrap=comparison["is_wrap"] if layout == "sum" else True,
            reference_min=comparison["reference_min"], cmap=cmap, vmin=vmin, vmax=vmax,
            colorbar_label=colorbar_label,
        )
        if directory is not None:
            figure.savefig(directory / f"{names[0]}_{names[1]}_{name}_{mode}.png",
                           dpi=160, facecolor="white", transparent=False)
        figures.append((figure, axes))
        if show:
            plt.show()
            plt.close(figure)
    return figures


def compare_both_orders(profiles, first, second, *, mode="native", previous=None, is_wrap=True):
    """Calculate both reference orders without plotting or writing outputs."""
    return {
        reference: {**prepare_comparison(
            profiles[reference], profiles[matched], mode=mode, is_wrap=is_wrap,
            order=None if previous is None else previous[reference]["order"],
        ), "reference_name": reference, "matched_name": matched}
        for reference, matched in ((first, second), (second, first))
    }


def hd_pick(table, hd_class=3):
    """Copy units in one or more HD classes; None retains every class."""
    info = table.attrs["unit_info"]
    selected = (np.ones(len(table), dtype=bool) if hd_class is None else
                np.isin([info[key]["hd_class"] for key in table.index], hd_class))
    result = table.loc[selected].copy()
    result.attrs["unit_info"] = {key: info[key].copy() for key in result.index}
    return result


def rf_pick(table, max_zero_bins=2):
    """Copy RF units within the native zero-bin limit; None keeps every unit."""
    info = table.attrs["unit_info"]
    selected = (np.ones(len(table), dtype=bool) if max_zero_bins is None else
                np.asarray([info[key]["zero_bins"] for key in table.index]) <= max_zero_bins)
    result = table.loc[selected].copy()
    result.attrs["unit_info"] = {key: info[key].copy() for key in result.index}
    return result


def load_hd_profiles(path, *, probe="A", bins=30, smoothing_deg=0, hd_class=None):
    """Read all HD units and rebin rates; optionally select classes with hd_pick."""
    data = read_formatted_json(path)
    # Classify native rates before display rebinning or smoothing; never save here.
    update_hd_classification(data)
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
    profiles.attrs["unit_info"] = {
        key: {"hd_class": value} for key, value in zip(profiles.index, classes, strict=True)
    }
    return profiles if hd_class is None else hd_pick(profiles, hd_class)


def load_ebc_profiles(path, *, probe="A"):
    """Read every EBC unit with missing source responses represented by zero."""
    maps = load_rf_maps(path)
    angles = maps[0].y_positions
    profiles = profile_table(maps.to_1d_array(axis="y"), maps.unit_ids, angles, probe=probe)
    profiles.attrs["unit_info"] = {key: {} for key in profiles.index}
    return profiles


def load_rf_profiles(source, *, probe="A", window=(0.0, .2), smoothing_bins=0,
                     max_zero_bins=None, rf_type=None,
                     rf_detection="excitatory", range: list[int] | None = None):
    """Project RF units, optionally selecting saved localization and native x bins.

    Missing responses are zero. Count native zero bins before smoothing or
    resampling; None disables the zero-bin limit. rf_type selects saved '2d', '1d', or
    'either' detections; None retains all source units without reading localization.
    rf_detection selects excitatory/inhibitory files without changing responses.
    """
    if rf_type not in (None, "2d", "1d", "either"):
        raise ValueError("rf_type must be None, '2d', '1d', or 'either'")
    if rf_detection not in ("excitatory", "inhibitory"):
        raise ValueError("rf_detection must be 'excitatory' or 'inhibitory'")
    source = Path(source)
    maps = load_rf_maps(source, unit_firing_rate=False).sum(*window, show_progress=False)
    responses = maps.to_1d_array(axis="x").astype(float)
    responses = np.where(np.isnan(responses), 0., responses)
    angles = maps[0].x_positions
    profiles = profile_table(responses, maps.unit_ids, angles, probe=probe)
    if range is not None:
        profiles.attrs["range"] = list(range)
    profiles.attrs["response_kind"] = "RF"
    profiles.attrs["unit_info"] = {
        key: {"zero_bins": int(zero)}
        for key, zero in zip(profiles.index, profiles.eq(0).sum(axis=1), strict=True)
    }
    if rf_type is not None:
        detection_source = (source.with_name(f"{source.stem}_inhibitory{source.suffix}")
                            if rf_detection == "inhibitory" else source)
        dimensions = ("2d", "1d") if rf_type == "either" else (rf_type,)
        detected_ids = set()
        for dimension in dimensions:
            result_path = (detection_source.with_suffix(".npz") if dimension == "2d" else
                           detection_source.with_name(f"{detection_source.stem}_1d.npz"))
            with np.load(result_path, allow_pickle=False) as result:
                detected = np.any(result["mask_2d"], axis=(1, 2))
                detected_ids.update(result["unit_ids"][detected].tolist())
        # Localization may contain a QC subset in a different unit order.
        profiles = profiles.loc[profiles.index.get_level_values("unit_id").isin(detected_ids)].copy()
        profiles.attrs["unit_info"] = {
            key: profiles.attrs["unit_info"][key].copy() for key in profiles.index
        }
    if max_zero_bins is not None:
        profiles = rf_pick(profiles, max_zero_bins)
    return smooth_profiles(profiles, smoothing_bins)


def load_rf_centers(source, *, probe="A"):
    """Read localized RF centers for distance analyses, independently of curve loading."""
    source = Path(source)
    with np.load(source.with_suffix(".npz"), allow_pickle=False) as result:
        masks = result["mask_2d"]
        centers = result["center_2d"]
        unit_ids = result["unit_ids"]
    maps = load_rf_maps(source, unit_firing_rate=False)
    selected = np.any(masks, axis=(1, 2))
    _, y, x = np.nonzero(centers[selected])
    index = pd.MultiIndex.from_product([[probe], unit_ids[selected]], names=["probe", "unit_id"])
    return pd.DataFrame({"rf_x_deg": maps[0].x_positions[x], "rf_y_deg": maps[0].y_positions[y]}, index=index)


def _prepare_profiles(profiles, *, mouse, date, range: list[int]):
    """Put every reader on one unit index and 30-bin grid, retaining source order."""
    if profiles.index.names != ["probe", "unit_id"]:
        raise ValueError("profiles must be indexed by (probe, unit_id)")
    angles = np.asarray(profiles.columns, dtype=float)
    ticks = _range_ticks(range)
    edges = np.linspace(ticks[0], ticks[-1], 31)
    centers = ((edges[:-1] + edges[1:]) / 2 + 180) % 360 - 180
    angles = (angles + 180) % 360 - 180
    profiles = profiles.copy()
    profiles.attrs["range"] = list(range)
    if len(angles) == len(centers) and np.array_equal(np.sort(angles), np.sort(centers)):
        table = profiles.copy()
        table.columns = angles
    else:
        table = _resample_profiles(profiles, centers)
    table = pd.concat({(str(mouse), str(date)): table}, names=["mouse", "date"])
    unit_info = profiles.attrs.get("unit_info", {})
    table.attrs["unit_info"] = {
        (str(mouse), str(date), *key): unit_info[key].copy()
        for key in profiles.index if key in unit_info
    }
    table.attrs["range"] = list(range)
    return table


def load_tc(path, *, mouse, date, probe="A", label="HD", range: list[int] | None = None,
            bins=30, smoothing_deg=0, hd_class=None):
    """Read all HD units, optionally selecting classes, with full recording keys."""
    profiles = load_hd_profiles(
        Path(path), probe=probe, bins=bins, smoothing_deg=smoothing_deg, hd_class=hd_class,
    )
    table = _prepare_profiles(profiles, mouse=mouse, date=date,
                              range=tcRange(False) if range is None else range)
    table.attrs["label"] = label
    return table


def load_rf(path, *, mouse, date, probe="A", label="RF", range: list[int] | None = None,
            window=(0.0, .2), smoothing_bins=0, max_zero_bins=None,
            rf_type=None, rf_detection="excitatory"):
    """Read RF curves; rf_type selects saved detections, bin limits remain optional."""
    angle_range = tcRange(True) if range is None else range
    profiles = load_rf_profiles(
        Path(path), probe=probe, window=window, smoothing_bins=smoothing_bins,
        max_zero_bins=max_zero_bins, rf_type=rf_type, rf_detection=rf_detection, range=angle_range,
    )
    table = _prepare_profiles(profiles, mouse=mouse, date=date, range=angle_range)
    table.attrs["label"] = label
    return table


def load_ebc(path, *, mouse, date, probe="A", label="EBC", range: list[int] | None = None):
    """Read an EBC angular rate map into the same curve structure."""
    profiles = load_ebc_profiles(Path(path), probe=probe)
    table = _prepare_profiles(profiles, mouse=mouse, date=date,
                              range=tcRange(True) if range is None else range)
    table.attrs["label"] = label
    return table


def combine(*curves, label=None, range: list[int] | None = None):
    """Stack curve tables without averaging; reject duplicate unit keys.

    Matching grids retain the first input's bin order and display labels.
    Different grids require an explicit common range and are interpolated in
    physical degrees onto 30 bins, with partial fields left blank outside coverage.
    """
    if range is None:
        centers = curves[0].columns
        if any(set(curve.columns) != set(centers) for curve in curves[1:]):
            raise ValueError("different angular grids require an explicit common range in combine")
        aligned = [curve.reindex(columns=centers) for curve in curves]
    else:
        ticks = _range_ticks(range)
        edges = np.linspace(ticks[0], ticks[-1], 31)
        centers = ((edges[:-1] + edges[1:]) / 2 + 180) % 360 - 180
        aligned = [_resample_profiles(curve, centers) for curve in curves]
    table = pd.concat(aligned, verify_integrity=True)
    table.attrs = curves[0].attrs.copy()
    if range is not None:
        table.attrs["range"] = list(range)
    table.attrs["unit_info"] = {
        key: curve.attrs["unit_info"][key].copy()
        for curve in curves for key in curve.index if key in curve.attrs.get("unit_info", {})
    }
    if label is not None:
        table.attrs["label"] = label
    return table


class TuningCurveCollection:
    """Named 30-bin curves matched by mouse, date, probe, and unit ID.

    Prefixes name datasets; sessions from the same mouse and date share unit keys.
    """

    def __init__(self):
        self.profiles = {}

    def add(self, profiles, *, mouse, date, prefix, range: list[int] | None = None):
        """Register source-angle curves on the requested 30-bin angular range.

        Existing 30-bin profiles keep their source order for peak ties.
        Other angular grids are interpolated, retaining NaN gaps and partial coverage.
        Reusing a prefix replaces that dataset.
        """
        table = _prepare_profiles(profiles, mouse=mouse, date=date,
                                  range=tcRange(True) if range is None else range)
        self.profiles[prefix] = table
        return table

    def load_tc(self, path, *, mouse, date, probe, prefix, kind="HD", range: list[int] | None = None,
                **options):
        """Read path directly; mouse/date/probe identify units, prefix names the group.

        Build paths with ordinary f-strings at the call site. ``kind`` selects
        the reader; ``range`` supplies angular boundaries and display labels.
        """
        readers = {"HD": load_hd_profiles, "RF": load_rf_profiles, "EBC": load_ebc_profiles}
        kind = kind.upper()
        if kind not in readers:
            raise ValueError("kind must be HD, RF, or EBC")
        if kind == "RF":
            options["range"] = range
        profiles = readers[kind](Path(path), probe=probe, **options)
        table = _prepare_profiles(profiles, mouse=mouse, date=date,
                                  range=tcRange(False) if range is None else range)
        self.profiles[prefix] = table
        return table

    def combine(self, prefix, sources, *, range: list[int] | None = None):
        """Pool units in the first dataset's bin order; reject duplicate unit keys.

        This preserves tied peaks when sources share a bin order. Sources with
        another order use the first dataset's tie-breaking order after pooling.
        """
        table = combine(*(self.profiles[name] for name in sources), range=range)
        self.profiles[prefix] = table
        return table
