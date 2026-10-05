# RF, HD, and EBC comparisons

## Load, filter, select shared units, scale, then plot

[tc_comparison_pairs.ipynb](../tc_comparison_pairs.ipynb) keeps each operation explicit.
Run project code on `hhw9l84` with `~/.virtualenvs/rfmapping`.

```python
from Utils.direction_comparison import (
    load_tc, hd_pick, rf_pick, concat,
    select_rf_profiles, resample_profiles,
    normalize_tc, zscore_tc, prepare_comparison, plot_comparison_heatmaps, tcRange,
)
from Utils.rflocate import load_rf, rf_result_path

hd_all = load_tc(hd_file)
rf_all = load_tc(rf_csv_file, kind="RF", range=tcRange(True))
rf_detected = load_rf(rf_result_path(rf_file))
rf_all = select_rf_profiles(rf_all, rf_detected)
ebc = load_tc(ebc_file, kind="EBC")

hd = hd_pick(hd_all, hd_class=3)
rf = rf_pick(rf_all, max_zero_bins=2)
units = hd.index.intersection(rf.index)

hd_normalized = normalize_tc(hd.loc[units])
rf_normalized = normalize_tc(rf.loc[units])
plot_options = dict(vmin=0, vmax=1, colorbar_label="Normalized response")
native = prepare_comparison(hd_normalized, rf_normalized, mode="native")
aligned = prepare_comparison(hd_normalized, rf_normalized, mode="aligned")
summed = prepare_comparison(hd_normalized, rf_normalized, mode="sum", is_wrap=True)
plot_comparison_heatmaps(native, **plot_options)
plot_comparison_heatmaps(aligned, **plot_options)
plot_comparison_heatmaps(summed, **plot_options)
```

RF CSV loading retains every saved unit, native angle, and missing value. Time
aggregation, spatial summation, saved-RF selection, smoothing, and resampling
are explicit operations. The notebooks choose the CSV path, generate it only
when missing, and then call `load_tc(path, kind="RF", ...)`. See
[RF 1-D CSV generation](rfmap.md#notebook-1-d-csvs) for the `rf_only` bool.
Filtering does not modify the full tables or write class-list files.

Paths are read literally. Build repeated layouts with ordinary `str.format` or
f-strings before loading. For standard `<mouse>/<date>/<date>_<session>/data/...`
paths, `load_tc` derives mouse/date identity from the directories and the probe
from a `ProbeA`/`ProbeB` directory or the generated RF CSV filename. Same-day HD
and RF sessions therefore retain matching unit keys without repeating metadata
in every load call. Explicit identity arguments remain compatibility overrides.

## Tables and filters

`load_tc` returns a DataFrame with:

- rows indexed by `(mouse, date, probe, unit_id)` for standard recording paths;
- native angular columns for RF; the existing HD/EBC loaders prepare 30 bins;
- `attrs["label"]` and `attrs["range"]` for plotting;
- `attrs["unit_info"]` keyed by unit identity, retaining HD classification or
  native RF zero-bin counts for later filtering.

Use `kind="RF"` for RF CSVs; the default `kind="HD"` reads saved HD `.tc` files.
`kind="EBC"` uses the existing EBC reader. Files outside the recording layout
retain `(probe, unit_id)` keys and their source identity; the loader does not
invent mouse or date values. `load_rf` and `load_ebc` remain available.

Every source uses the same structure: columns contain the degree array, and
each unit row contains its response array. Firing rates use Hz; the current RF
CSVs contain spike counts. `attrs["response_units"]` records that distinction.
For a CSV explicitly saved in Hz, pass `response_units="Hz"`; this labels the
stored numbers and does not convert them.

```python
from Utils.direction_comparison import concat, plot_profiles

m14_hd = load_tc(m14_path)
m15_hd = load_tc(m15_path)
m14_15 = concat(m14_hd, m15_hd)
plot_profiles(m14_15)
```

`concat` stacks units and retains their recording keys, degree coordinates,
values, and per-unit information. For generic files without recording keys,
source identity distinguishes units from different files when concatenated.
Matching degree grids preserve the first
input's order. Different grids need an explicit common `range`, as with the
existing `combine` name, which remains an alias. Plotting defaults to the
object's label and response units and keeps input unit order. Normalization
and z-scoring update the units label for subsequent plots.

For RF, `range` records degree boundaries or tick labels without changing its
native grid. Call `resample_profiles(table, range=..., bins=30)` explicitly to
wrap angles into [-180, 180) and interpolate to a chosen grid. The existing
HD/EBC loaders still use `range` to prepare their 30-bin grid.
`tcRange(True)` returns the full-circle ego labels
`[-180, -90, 0, 90, 180]`; `tcRange(False)` returns the equivalent allo labels
`[180, 270, 0, 90, 180]`. Both defaults use centers −174 through 174 degrees.
The m14 RF override `range=[-150, 150]` instead uses 30 bins with centers −145
through 145 degrees, spaced by 10 degrees. Its HD retains the full-circle grid.
Native RF columns retain their source angles; explicit resampling produces
signed columns even when display labels use [0, 360].
Partial angular ranges do not connect their two endpoints by circular
interpolation or smoothing. Missing source responses remain NaN. Explicit RF
resampling represents angles outside its recorded support by zero; generic
HD/EBC gaps remain NaN.

The HD reader used by `load_tc` rebins spike counts and occupancy to angular rates without smoothing.
Zero-response, missing-data, and unclassified rows remain available in the full
loaded table. HD classification reuses `update_hd_classification` on the native
rates before display rebinning or smoothing. Class 3 uses the existing Rayleigh,
shuffle, and von Mises κ criteria. Class 2 retains units passing both significance
tests without passing the κ cutoff; class 1 passes exactly one test; class 0 passes
neither. A missing p value makes the class unavailable.
`hd_pick(table)` defaults to class 3; `(2, 3)` selects both classes in source order
without duplicates. `hd_class=None` keeps all rows.

`Utils.direction_comparison.load_rf_profiles`, `load_rf`, and `load_tc(kind="RF")`
read a TC CSV or already prepared RF map with a single time bin and singleton
collapsed spatial axis. They reject raw maps requiring aggregation. `load_rf`
adds recording keys and a label; it does not wrap or resample angles. The CSV
path is read literally, without filename selection or file generation.

`Utils.rflocate.load_rf` reads a saved detection without computing or replacing it.
Choose files with `rf_result_path`, then pass the loaded results to
`select_rf_profiles`:

| Loaded results | Units retained | Localization files |
|---|---|---|
| No selection call | Every source unit | None required |
| 2D result | Units with a detected 2D RF | Same-name `.npz` |
| 1D result | Units with a detected 1D RF | Same stem plus `_1d.npz` |
| Both results | Units detected by 2D or 1D, including both | Both files |

Both result files store `mask_2d` and `unit_ids`; independent 1D masks have a
singleton spatial axis. Nonempty masks identify detections, matched by saved unit
ID because localization QC may remove or reorder units. The returned curves keep
their source values and order. A requested result file must exist; an absent file
raises `FileNotFoundError`. Use `rf_result_path(source, rf_type="inhibitory")`
for inhibitory results and `dimension="1d"` for independent 1D results.
Only a 2D result defines whole-row/column selection in `sum_to_1d(rf_only=True)`.
An absent unit ID is an error, so select the saved-result unit set explicitly
before projecting a source whose detection file contains only a QC subset.

`rf_pick(table, max_zero_bins=2)` permits at most two zero angular bins in
its native horizontal projection, before interpolation or smoothing. Missing
responses remain NaN and are distinct from measured zero bins; this zero-bin
filter does not impose a missing-bin limit. `None` disables the zero-bin limit.
The native zero count is retained in `attrs["unit_info"]`; interpolation does
not change QC membership.

`load_ebc` projects every unit in the saved angular rate map, retaining zero
responses and missing values. Source angles keep their direction and wrap to [-180, 180):
90 stays 90 and 270 becomes −90. Other full-circle grids, including EBC's 60 bins,
are periodically interpolated while retaining missing intervals. To switch
between ego and allo labels on an existing full-circle table, use
`convert_profile_coordinates(table, range=tcRange(False))`; this preserves its
values and angular extent. Changing the RF grid belongs in an explicit
`resample_profiles` call.

## Pooling and choosing rows

`combine(A, B, ...)` stacks rows without averaging and preserves per-unit filter
metadata. Tables sharing a grid retain the first table's bin order and display
settings. For different grids, supply an explicit common `range`: the paired
notebooks pool m14's partial RF grid and the other sessions on
`range=tcRange(True)`. m14 angles outside ±150° are zero in that pooled RF table.
The pooled curves use 12-degree bins, so peaks extracted from them can differ
from the native 10-degree m14 peaks; single-session plots use the native m14 grid.
Combining probes from the same m14 session instead retains `range=[-150, 150]`.
The function accepts an optional `label`, rejects duplicate unit keys, and
returns the same table structure for further filtering or combination.

```python
hd_all = combine(hd_m19, hd_m20, label="Pooled HD")
rf_all = combine(rf_m14, rf_m19, rf_m20, range=tcRange(True), label="Pooled RF")
hd = hd_pick(hd_all, hd_class=3)
rf = rf_pick(rf_all, max_zero_bins=2)
units = hd.index.intersection(rf.index)

# Optional numeric-ID selection; mouse/date/probe remain part of each key.
units = units[units.get_level_values("unit_id").isin([10, 20])]
hd_normalized = normalize_tc(hd.loc[units])
rf_normalized = normalize_tc(rf.loc[units])
pooled = prepare_comparison(hd_normalized, rf_normalized)
plot_comparison_heatmaps(pooled, **plot_options)
```

Unit identity excludes session because same-day sessions share spike sorting.
Different mice, dates and probes remain distinct even when numeric IDs match.
Repeated sessions of the same neurons can be compared directly without pooling.
For an inclusive numeric range, narrow `units` with a mask on its `unit_id` level
before `.loc`; the plotting functions do not implement row selection.

## Explicit normalization and z-scores

`normalize_tc(tc)` and `zscore_tc(tc)` return new floating-point DataFrames with
the same shape, row index, angular columns, ordering, and metadata. They operate
independently on each unit's measured angular bins. Missing values remain missing,
and measured zeros participate in the calculation. The input table is unchanged.

`normalize_tc(tc)` divides each row by its maximum, matching the previous
per-unit heatmap normalization. It is not min–max scaling. All-zero rows stay
zero, and entirely missing rows stay missing.

`zscore_tc(tc)` computes `(value - mean) / SD` within each row, using population
SD (`ddof=0`) and equal weight per finite angular bin. Nonconstant rows have
mean zero and SD one over their measured bins. Constant rows, rows with only one
measured bin, and entirely missing rows become NaN because their z-score is
undefined. This describes variation across the supplied tuning curve's angular
bins; it is neither a baseline significance test nor trial-to-trial variability.
Neither helper converts spike counts to firing rates.

For z-score heatmaps, supply an explicit symmetric color range shared by both
panels and a diverging colormap:

```python
import numpy as np

hd_z = zscore_tc(hd.loc[units])
rf_z = zscore_tc(rf.loc[units])
values = np.concatenate([hd_z.to_numpy().ravel(), rf_z.to_numpy().ravel()])
finite = values[np.isfinite(values)]
limit = float(np.max(np.abs(finite))) if finite.size else 1.0
z_options = dict(cmap="RdBu_r", vmin=-limit, vmax=limit, colorbar_label="Z-score (SD)")
z_native = prepare_comparison(hd_z, rf_z)
z_aligned = prepare_comparison(hd_z, rf_z, mode="aligned")
plot_comparison_heatmaps(z_native, **z_options)
plot_comparison_heatmaps(z_aligned, **z_options)
```

Scale the final TC before alignment and before the duplicated boundary bin used
for circular display. Alignment interpolation may reduce a row's sampled maximum
or SD; plotting does not normalize it again. This retains one scale across native,
aligned, and summed views. Keep the original tables for analyses that require
rates, counts, or information about constant curves.

## Plotting and statistics

`prepare_comparison(reference, matched, mode="native")` computes paired
peaks, row order, and transformed tables. Inputs must contain the same unique
unit-key set; an explicit `order` must be a permutation of those keys. The
caller selects the overlap before calculation.

- `mode="native"` retains source angles.
- `mode="aligned"` subtracts the reference peak from both curves.
- `mode="sum"` centers the reference and adds its peak to the matched angles.

For inhibitory comparisons, choose `reference_min=True` or `matched_min=True`
for the RF table while the HD table retains its maximum. Peak ties use the
first source bin. `is_wrap=True` wraps sums to [-180, 180); False retains direct
sums and uses a −360 through 360 degree matched panel.

The returned dictionary contains `reference`, `matched`, `order`, `peaks`, and
the offsets. `plot_comparison_heatmaps(prepared, ...)` renders those supplied
tables without peak calculation, selection, normalization, smoothing, or
alignment. It returns two `(figure, axes)` pairs. With `show=False`, the caller
closes returned figures. `compare_both_orders` only calculates the two reference
orders; its caller plots and writes the resulting tables explicitly.

Display options include `figsize`, `labels`, `show`, `cmap`, `vmin`, `vmax`,
`colorbar_label`, and `save_dir`. Use explicit common limits when comparing
colors. Figures and exports have opaque white backgrounds and dark text.

Use the calculated `peaks` table for direction statistics.
`compare_direction_angles` calculates circular correlation and Spearman Mantel
rho with permutation p values; `plot_direction_comparison` renders supplied
statistics. The paired notebooks use 10,000 permutations and seed 1, and skip
these tests with fewer than three finite pairs.

`peak_sum_statistics(peaks)` calculates histogram counts, n, R, and Rayleigh p;
`plot_peak_direction_sums(summary)` only renders those values. The hand-written
polar notebook histograms likewise calculate counts and the Rayleigh test before
plotting. Their presentation retains the chosen modulo-360° angles.
