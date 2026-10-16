"""Shared unit-keyed angular profiles, peak sorting, and circular alignment."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from Utils.plotting import LIGHT_PLOT_STYLE, angular_ticks, plot_keyed_heatmap
from Utils.statistic_utils import rayleigh_uniformity


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
    if "response_units" in result.attrs:
        result.attrs["response_units"] = "normalized"
    return result


def minmax_profiles(table):
    """Scale each displayed unit; missing bins stay missing, constants become 0."""
    minimum = table.min(axis=1)
    span = table.max(axis=1) - minimum
    result = table.sub(minimum, axis=0).div(span.replace(0., 1.), axis=0)
    result.attrs = table.attrs.copy()
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
    if "response_units" in result.attrs:
        result.attrs["response_units"] = "zscore"
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
    """Sort paired maxima/minima, retaining reference row order for equal peaks.

    An explicit order rearranges the full unit set without sorting. Opaque unit
    labels do not determine tie order or seeded permutation inputs.
    """
    units = set(reference.index)
    if (not reference.index.is_unique or not matched.index.is_unique
            or units != set(matched.index)):
        raise ValueError("tables must have the same unique unit keys; select overlap before plotting")
    sort_by_reference = order is None
    if sort_by_reference:
        order = list(reference.index)
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


def _resample_profiles(profiles, centers, *, fill_value=np.nan):
    angle_range = profiles.attrs["range"]
    values = np.array([
        _interpolate_angles(profiles.columns, row, centers, angle_range, fill_value=fill_value)
        for row in profiles.to_numpy()
    ]).reshape(len(profiles), len(centers))
    table = pd.DataFrame(values, index=profiles.index, columns=centers)
    table.attrs = profiles.attrs.copy()
    return table


def align_profiles(profiles, units, offset_deg, *, bins=30, is_wrap=True, fill_value=np.nan):
    """Add each unit's offset; fill unmeasured angles only as explicitly requested.

    Unwrapped sums use twice the angular span at the same bin spacing.
    """
    relative_deg = ((np.arange(bins) - bins // 2) * (360.0 / bins) if is_wrap
                    else np.linspace(-360., 360., 2 * bins + 1))
    rows = profiles.reindex(units)
    angle_range = profiles.attrs.get("range", tcRange(True))
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


def plot_profiles(profiles, order=None, name=None, *, axis=None, sort_name=None,
                  aligned=False, show=True, figsize=(10, 6), is_wrap=True,
                  reference_min=False, cmap="viridis", vmin=None, vmax=None,
                  colorbar_label=None):
    """Render supplied curve values without normalizing or standardizing them."""
    order = profiles.index.tolist() if order is None else order
    name = profiles.attrs.get("label", "TC") if name is None else name
    if colorbar_label is None:
        colorbar_label = {
            "Hz": "Firing rate (Hz)", "spike_count": "Spike count",
            "normalized": "Normalized response", "zscore": "Z-score (SD)",
        }.get(profiles.attrs.get("response_units"), "Response")
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
    if aligned and is_wrap and columns[0] == -180:
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
    title = f"{name}{suffix}" if sort_name is None else f"{name} | sorted by {sort_name}{suffix}"
    return plot_keyed_heatmap(
        dict(zip(profiles.index, values)), order,
        column_order=column_order, xticks=ticks,
        xticklabels=axis["ticklabels"], angle_centers=columns[column_order],
        xlabel=xlabel, title=title,
        show=show, figsize=figsize, cmap=cmap, vmin=vmin, vmax=vmax,
        colorbar_label=colorbar_label,
    )


def prepare_comparison(reference, matched, *, mode="native", order=None, bins=30, is_wrap=True,
                       reference_min=False, matched_min=False,
                       reference_fill_value=np.nan, matched_fill_value=np.nan):
    """Calculate paired peaks, row order, and explicit angular transformations.

    Native keeps source angles. Aligned subtracts the reference direction from
    both tables. Sum centers the reference and adds its direction to the match.
    Padding outside each measured angular range is specified per panel.
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
        reference_panel = align_profiles(reference, order, offsets, bins=bins,
                                         fill_value=reference_fill_value)
        matched_panel = align_profiles(matched, order, matched_offsets, bins=bins,
                                       is_wrap=is_wrap if mode == "sum" else True,
                                       fill_value=matched_fill_value)
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
                             colorbar_label=None):
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


def rebin_profiles(profiles, *, bins):
    """Sum adjacent bins, preserving source order, unit identity, and metadata."""
    group_size = profiles.shape[1] // bins
    values = profiles.to_numpy().reshape(len(profiles), bins, group_size).sum(axis=2)
    # Unwrap before averaging centers across the +180/-180 degree boundary.
    angles = np.unwrap(profiles.columns.to_numpy(dtype=float), period=360)
    centers = angles.reshape(bins, group_size).mean(axis=1)
    result = pd.DataFrame(values, index=profiles.index, columns=(centers + 180) % 360 - 180)
    result.attrs = profiles.attrs.copy()
    return result


def resample_profiles(profiles, *, range: list[int], bins=30, fill_value=np.nan):
    """Interpolate to the requested grid, using fill_value outside measured angles."""
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    angles = np.asarray(profiles.columns, dtype=float)
    ticks = _range_ticks(range)
    edges = np.linspace(ticks[0], ticks[-1], bins + 1)
    centers = ((edges[:-1] + edges[1:]) / 2 + 180) % 360 - 180
    angles = (angles + 180) % 360 - 180
    profiles = profiles.copy()
    profiles.attrs.setdefault("range", list(range))
    if len(angles) == len(centers) and np.array_equal(np.sort(angles), np.sort(centers)):
        table = profiles.copy()
        table.columns = angles
    else:
        table = _resample_profiles(profiles, centers, fill_value=fill_value)
    table.attrs["range"] = list(range)
    return table


def recording_profiles(profiles, *, mouse, date, label=None):
    """Add recording keys and an optional label without changing curve values."""
    if profiles.index.names != ["probe", "unit_id"]:
        raise ValueError("profiles must be indexed by (probe, unit_id)")
    table = profiles.copy()
    table = pd.concat({(str(mouse), str(date)): table}, names=["mouse", "date"])
    table.attrs = profiles.attrs.copy()
    unit_info = profiles.attrs.get("unit_info", {})
    table.attrs["unit_info"] = {
        (str(mouse), str(date), *key): unit_info[key].copy()
        for key in profiles.index if key in unit_info
    }
    if label is not None:
        table.attrs["label"] = label
    return table


def _prepare_profiles(profiles, *, mouse, date, range: list[int]):
    """Resample a supplied table and add recording identity without file access."""
    return recording_profiles(resample_profiles(profiles, range=range), mouse=mouse, date=date)


def save_tc(table, path):
    """Write opaque unit IDs and numeric degree columns without altering values.

    Encode complete unit identity in the index before saving. DataFrame
    attributes are not stored in this plain numerical format.
    """
    if isinstance(table.index, pd.MultiIndex):
        raise ValueError("TC CSV requires a simple index containing complete unit identity")
    unit_ids = table.index.astype(str)
    if unit_ids.has_duplicates:
        raise ValueError("TC CSV unit IDs must be unique")
    angles = table.columns.to_numpy(dtype=float)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["unit_id", *angles])
        writer.writerows([unit_id, *row] for unit_id, row in zip(unit_ids, table.to_numpy(), strict=True))


def tc_loader(path, *, label=None, range: list[int] | None = None, response_units=None):
    """Read a numerical CSV with opaque unit IDs and degree columns.

    Identity comes only from ``unit_id`` in the file. Optional metadata describes
    the stored values; loading does not classify, select, or transform curves.
    """
    table = pd.read_csv(path, converters={"unit_id": str},
                        float_precision="round_trip").set_index("unit_id")
    table.columns = table.columns.astype(float)
    if label is not None:
        table.attrs["label"] = label
    if range is not None:
        table.attrs["range"] = list(range)
    if response_units is not None:
        table.attrs["response_units"] = response_units
    return table


def concat(*curves, label=None):
    """Stack curve tables without averaging; reject duplicate unit keys.

    Matching grids retain the first input's bin order and display labels.
    Resample differing grids explicitly before concatenation.
    """
    centers = curves[0].columns
    if any(set(curve.columns) != set(centers) for curve in curves[1:]):
        raise ValueError("different angular grids; call resample_profiles explicitly before concat")
    aligned = [curve.reindex(columns=centers) for curve in curves]
    table = pd.concat(aligned, verify_integrity=True)
    table.attrs = curves[0].attrs.copy()
    table.attrs.pop("source", None)
    for attribute in ("response_units", "response_kind"):
        if any(curve.attrs.get(attribute) != curves[0].attrs.get(attribute) for curve in curves[1:]):
            table.attrs.pop(attribute, None)
    if any("unit_info" in curve.attrs for curve in curves):
        table.attrs["unit_info"] = {
            key: curve.attrs["unit_info"][key].copy()
            for curve in curves for key in curve.index if key in curve.attrs.get("unit_info", {})
        }
    if label is not None:
        table.attrs["label"] = label
    else:
        labels = dict.fromkeys(curve.attrs["label"] for curve in curves if "label" in curve.attrs)
        if labels:
            table.attrs["label"] = " + ".join(labels)
    return table


combine = concat


class TuningCurveCollection:
    """Named TC tables matched by the unit keys stored in each table."""

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

    def load_tc(self, path, *, prefix, label=None, range: list[int] | None = None,
                response_units=None):
        """Read a TC CSV and register it under the supplied dataset name."""
        table = tc_loader(path, label=label, range=range, response_units=response_units)
        self.profiles[prefix] = table
        return table

    def combine(self, prefix, sources):
        """Pool units in the first dataset's bin order; reject duplicate unit keys.

        This preserves tied peaks when sources share a bin order. Sources with
        another order use the first dataset's tie-breaking order after pooling.
        """
        table = combine(*(self.profiles[name] for name in sources))
        self.profiles[prefix] = table
        return table
