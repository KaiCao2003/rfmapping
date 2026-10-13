# RF, HD, and EBC comparisons

## Load, filter, select shared units, scale, then plot

[tc_comparison_pairs.ipynb](../tc_comparison_pairs.ipynb) keeps each operation explicit.
Run project code on `hhw9l84` with `~/.virtualenvs/rfmapping`.

```python
from Utils.direction_comparison import (
    tc_loader, rf_pick, concat, resample_profiles,
    normalize_tc, zscore_tc, prepare_comparison, plot_comparison_heatmaps, tcRange,
)
hd = tc_loader(hd_class3_csv, label="HD", range=tcRange(False), response_units="Hz")
rf_native = tc_loader(rf_2d_csv, label="RF", response_units="spike_count")
rf_selected = rf_pick(rf_native, max_zero_bins=2)
rf = resample_profiles(rf_selected, range=tcRange(True), bins=30, fill_value=0)
units = hd.index.intersection(rf.index)

hd_normalized = normalize_tc(hd.loc[units])
rf_normalized = normalize_tc(rf.loc[units])
plot_options = dict(vmin=0, vmax=1, colorbar_label="Normalized response")
native = prepare_comparison(hd_normalized, rf_normalized, mode="native")
aligned = prepare_comparison(hd_normalized, rf_normalized, mode="aligned", matched_fill_value=0)
summed = prepare_comparison(hd_normalized, rf_normalized, mode="sum", is_wrap=True, matched_fill_value=0)
plot_comparison_heatmaps(native, **plot_options)
plot_comparison_heatmaps(aligned, **plot_options)
plot_comparison_heatmaps(summed, **plot_options)
```

`tc_comparison_pairs.ipynb` uses `Utils.tc_comparison.prep_hd_tc` and `prep_rf_tc`
with mouse, date, probe, and session IDs. These helpers prepare and read each
comparison CSV, then return normalized curves before shared-unit selection.
RF zero-bin filtering uses the native grid before resampling. The inhibitory
notebook calls the preparation functions and `tc_loader()` directly.
Preparation skips existing outputs before accessing source
data. Missing HD tables are generated from saved `.tc` data; missing RF tables
use saved native projections or prepare them from RF maps and existing detection
results. `tc_loader()` itself only reads CSVs.

`calculate_tc_statistics(hd, rf)` matches shared units in HD table order and
prepares both reference orders for native, aligned, and summed comparisons,
plus peak-angle and peak-sum statistics. `plot_tc_comparison(hd, rf, stats)`
plots those prepared results without repeating the statistics.

The HD tuning notebooks, HD regeneration script, and Basler rebuild pipeline
call `prepare_hd_tc()` after saving tuning curves. The RF notebook and MATLAB
bridge's canonical regular ON workflow call `prepare_rf_comparison()` after
detection. All use the same preparation functions as the paired notebooks.

On `hhw9l84`, `/home/kai/scripts/run_pipeline.sh` invokes the separate
`/home/kai/scripts/run_tuning_curves.py` entrypoint. After saving each probe's
`.tc`, that entrypoint runs this repository's HD exporter with
`--comparison-code-dir` (default `~/Developer/rfmapping`) as its working
directory. This keeps its preprocessing `Utils` separate from the comparison
package and covers both Motive and Basler sessions.

The published `scripts/run_pipeline.sh` instead invokes the included
`scripts.run_tuning_curves` module from the repository root. That module imports
`prepare_hd_tc` directly, so a fresh checkout includes the complete call chain.

## Prepare one comparison CSV

Run from `~/Developer/rfmapping` on `hhw9l84`. Output directories are created
as needed. These two commands prepare one HD table and one RF table:

```sh
~/.virtualenvs/rfmapping/bin/python -m scripts.export_comparison_tcs hd \
    '/mnt/senzailab/Kai/#Recording/m14/260609/260609_1/data/tuning_curves/ProbeA/tuning_curves.tc' \
    '/mnt/senzailab/Kai/#Recording/m14/260609/260609_1/data/tc_comparison/hd_class3_ProbeA.csv' \
    --bins 30 --hd-class 3 --unit-prefix m14:260609:A

~/.virtualenvs/rfmapping/bin/python -m scripts.export_comparison_tcs rf \
    '/mnt/senzailab/Kai/#Recording/m14/260609/260609_3/data/regular_unitsSpikeCounts_260609_3_ProbeA_1d_rfonly.csv' \
    '/mnt/senzailab/Kai/#Recording/m14/260609/260609_3/data/tc_comparison/rf_excitatory_x_2d_rfonly_ProbeA.csv' \
    --select-results '/mnt/senzailab/Kai/#Recording/m14/260609/260609_3/data/rfmapping/good/-100_400_1ms/ProbeA/regular_unitsSpikeCounts_260609_3.npz' \
    --unit-prefix m14:260609:A
```

Each invocation writes only its specified output. Paths and filenames are caller
choices; the exporter has no recording-directory convention. Existing outputs
are skipped unless `--overwrite` is supplied. Input files are never replaced.

The `hd` command rebins saved counts and occupancy to rates, with the requested
`--bins` and no smoothing. `--hd-class` optionally selects one class after native
classification. Omitting it keeps every class. The `rf` command copies an already
prepared x or y projection CSV. It preserves native angles and values, including
zeros and missing bins. It does not open `.rfmap` files, select a time window,
project maps, or compute detections. Prepare native projections upstream with
[RF 1-D CSV generation](rfmap.md#notebook-1-d-csvs).

`prepare_rf_comparison()` handles that upstream dependency when needed: it calls
`prepare_rf_projection()` to sum raw counts over the explicit response window
and project onto native x bins, then `prepare_rf_tc()` to select units and save
their paired identities. RF-only projections include whole rows touched by the
saved 2D RF. Preparation never reruns detection or converts counts to Hz.
Existing CSVs are reused; after changing the response window, classification,
or projection settings, request `overwrite=True` explicitly to regenerate.

`--select-results` optionally restricts RF rows to units with a nonempty saved
mask. Multiple supplied files select their union by numeric source unit ID;
output order follows the source CSV. Omitting the option retains all source rows.
Choosing a full projection or an RF-only projection is the caller's source-file
choice. The paired notebooks use explicitly listed files under each session's
`data/tc_comparison/` directory; exporting does not enumerate class, detection,
or projection combinations.

Every output contains only `unit_id` and numeric degree columns. A source unit
23 with `--unit-prefix m14:260609:A` is saved as `m14:260609:A:23`. Use the same
prefix when HD and RF recordings share the same sorted neurons, and distinct
prefixes for different mice, dates, or probes. The prefix is supplied explicitly;
no part of it is inferred from the path.

## Tables, selection, and resampling

`tc_loader(path)` reads only CSVs and returns a DataFrame with:

- a simple `unit_id` index containing the exact opaque strings in the file;
- the saved degree columns, values, missing bins, and row order;
- optional `label`, `range`, and `response_units` attributes supplied by the caller.

Loading performs no classification, detection, zero-bin counting, or resampling.
It has no `kind` switch, path-based identity, or assumed response units.
`response_units="Hz"` or `"spike_count"` labels stored values without converting
them. `save_tc(table, path)` writes the same plain format; it requires a simple,
unique `unit_id` index. DataFrame attributes are not saved in the CSV.

`rf_pick(table, max_zero_bins=2)` counts measured zero bins in the supplied
curves. Call it on native RF curves before interpolation or smoothing. Missing
bins remain NaN and do not count as zeros; the filter imposes no missing-bin
limit. `None` disables the zero-bin limit. Selection does not write any files.

`range` supplied to `tc_loader` only records plotting labels. To change the grid,
call `resample_profiles(table, range=..., bins=...)` explicitly. It wraps output
angles to [-180, 180) and interpolates onto the requested grid, preserving source
order when that grid already matches. `tcRange(True)` supplies full-circle ego
labels `[-180, -90, 0, 90, 180]`; `tcRange(False)` supplies equivalent allo labels
`[180, 270, 0, 90, 180]`. With `bins=30`, these grids have centers −174 through
174 degrees. The m14 RF choice `range=[-150, 150], bins=30` instead has centers
−145 through 145 degrees.

`fill_value` explicitly controls values outside measured angular support. It
defaults to NaN. The paired notebooks use `fill_value=0` for their RF comparison
policy, including m14 outside ±150° when pooling onto a full-circle grid.
Missing responses inside the measured support remain NaN. Partial angular
ranges do not connect their endpoints by circular interpolation or smoothing.
Full-circle source grids use periodic interpolation.

The source readers remain separate from `tc_loader`. `load_hd_profiles` reads
original HD `.tc` data, rebins counts and occupancy into rates, and classifies
native rates before rebinning or optional smoothing. Class 3 uses the existing
Rayleigh, shuffle, and von Mises κ criteria. Class 2 passes both significance
tests without passing κ; class 1 passes exactly one test; class 0 passes neither.
An unavailable p value leaves the class unavailable. `hd_pick` selects classes
from this source table; class metadata is not stored in comparison CSVs.

`load_rf_profiles` reads a native RF CSV or an already prepared RF map with a
single time bin and collapsed spatial axis. `load_rf` adds explicitly supplied
recording keys for existing source-reader workflows. `Utils.rflocate.load_rf`
reads a saved detection; `select_rf_profiles` selects its detected numeric unit
IDs from native source tables before opaque IDs are exported. Only a 2D result
defines whole-row/column selection in `sum_to_1d(rf_only=True)`; an independent
1D result can select units but cannot define that projection.

`load_ebc` retains the separate EBC preparation workflow: it projects saved
angular rate maps and prepares their comparison grid. Zero responses and
missing values retain their meaning. To switch ego and allo display labels on
an existing full-circle table, use
`convert_profile_coordinates(table, range=tcRange(False))`; this preserves its
values and angular extent.

## Pooling and choosing rows

`concat(A, B, ...)` only stacks rows on matching degree grids. It retains the
first table's bin order, preserves values and unit IDs, and rejects duplicate
IDs. It does not create identity from source paths or interpolate. `combine`
remains an alias with the same behavior. Resample differing grids explicitly
before pooling:

```python
hd_all = concat(hd_m14, hd_m19, label="Pooled HD")
rf_m14_native = tc_loader(m14_rf_csv, label="m14 RF", response_units="spike_count")
rf_m19_native = tc_loader(m19_rf_csv, label="m19 RF", response_units="spike_count")
rf_m14 = resample_profiles(
    rf_pick(rf_m14_native, max_zero_bins=2),
    range=tcRange(True), bins=30, fill_value=0,
)
rf_m19 = resample_profiles(
    rf_pick(rf_m19_native, max_zero_bins=2),
    range=tcRange(True), bins=30, fill_value=0,
)
rf_all = concat(rf_m14, rf_m19, label="Pooled RF")
units = hd_all.index.intersection(rf_all.index)
hd_normalized = normalize_tc(hd_all.loc[units])
rf_normalized = normalize_tc(rf_all.loc[units])
pooled = prepare_comparison(hd_normalized, rf_normalized)
plot_comparison_heatmaps(pooled, **plot_options)
```

For a chosen neuron subset, use complete saved IDs such as
`["m14:260609:A:10", "m14:260609:A:20"]` in `.loc`. The loader and plotting
functions do not parse those strings. Separate sessions of the same neurons
can be compared directly; pooling the same identity twice raises an error.
The pooled 12-degree grid can give different sampled peaks from a native
10-degree m14 RF grid, so retain native curves for analyses of native peaks.

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

By default, units with equal reference peaks retain their order in the reference
input table. Renaming opaque unit IDs therefore does not change the row order
fed to seeded permutation tests. Reordering the input rows can still change the
sampled permutations; keep input order and random seed fixed for exact repeats.

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
