# RF, HD, and EBC comparisons

## Load, filter, select shared units, scale, then plot

[tc_comparison_pairs.ipynb](../tc_comparison_pairs.ipynb) keeps each operation explicit.
Run project code on `hhw9l84` with `~/.virtualenvs/rfmapping`.

```python
from Utils.direction_comparison import (
    load_tc, load_rf, load_ebc, hd_pick, rf_pick, combine,
    normalize_tc, zscore_tc, plot_sort, plot_align, plot_sum, tcRange,
)

recording = dict(mouse="m19", date=260827, probe="A")
rf_range_overrides = {"m14": [-150, 150]}
hd_all = load_tc(hd_file, **recording, range=tcRange(False))
rf_all = load_rf(rf_file, **recording, rf_type="2d",
                 range=rf_range_overrides.get(recording["mouse"], tcRange(True)))
ebc = load_ebc(ebc_file, **recording, range=tcRange(True))

hd = hd_pick(hd_all, hd_class=3)
rf = rf_pick(rf_all, max_missing_bins=2, max_zero_bins=2)
units = hd.index.intersection(rf.index)

hd_normalized = normalize_tc(hd.loc[units])
rf_normalized = normalize_tc(rf.loc[units])
plot_options = dict(vmin=0, vmax=1, colorbar_label="Normalized response")
native = plot_sort(hd_normalized, rf_normalized, **plot_options)
aligned = plot_align(hd_normalized, rf_normalized, **plot_options)
summed = plot_sum(hd_normalized, rf_normalized, is_wrap=True, **plot_options)
```

Loaders retain all source units and apply no smoothing by default. Filtering is
optional and does not modify the full tables. Loading with `hd_class=3` or
`max_missing_bins=2, max_zero_bins=2` gives the same selection as the corresponding `hd_pick` or
`rf_pick` call after loading. Neither loading nor filtering writes class-list files.

Paths are read literally. Build repeated layouts with ordinary `str.format` or
f-strings before loading; directory suffixes such as `ProbeA_bk` do not change
biological probe identity.

## Tables and filters

The three public loaders return DataFrames with:

- rows indexed by `(mouse, date, probe, unit_id)`;
- 30 angular columns spanning the requested `range`, retaining source order
  when the source already has that grid;
- `attrs["label"]` and `attrs["range"]` for plotting;
- `attrs["unit_info"]` keyed by unit identity, retaining HD classification or
  native RF missing-bin and zero-bin counts for later filtering.

`range` is a list of degree boundaries or tick labels and sets the actual
30-bin angular grid. `tcRange(True)` returns the full-circle ego labels
`[-180, -90, 0, 90, 180]`; `tcRange(False)` returns the equivalent allo labels
`[180, 270, 0, 90, 180]`. Both defaults use centers −174 through 174 degrees.
The m14 RF override `range=[-150, 150]` instead uses 30 bins with centers −145
through 145 degrees, spaced by 10 degrees. Its HD retains the full-circle grid.
Internal angular columns stay in [-180, 180), including when labels use [0, 360].
Partial angular ranges do not connect their two endpoints by circular
interpolation or smoothing; unmeasured angles remain NaN.

`load_tc` rebins spike counts and occupancy to angular rates without smoothing.
Zero-response, missing-data, and unclassified rows remain available in the full
loaded table. HD classification reuses `update_hd_classification` on the native
rates before display rebinning or smoothing. Class 3 uses the existing Rayleigh,
shuffle, and von Mises κ criteria. Class 2 retains units passing both significance
tests without passing the κ cutoff; class 1 passes exactly one test; class 0 passes
neither. A missing p value makes the class unavailable.
`hd_pick(table)` defaults to class 3; `(2, 3)` selects both classes in source order
without duplicates. `hd_class=None` keeps all rows.

`load_rf` projects every RF source unit onto horizontal angle over the requested
`window`, default `(0.0, 0.2)`. Optional `rf_type` selects units from saved
localization results before applying bin limits:

| `rf_type` | Units retained | Localization files |
|---|---|---|
| `None` (default) | Every source unit | None required |
| `"2d"` | Units with a detected 2D RF | Same-name `.npz` |
| `"1d"` | Units with a detected 1D RF | Same stem plus `_1d.npz` |
| `"either"` | Units detected by 2D or 1D, including both | Both files |

Both result files store `mask_2d` and `unit_ids`; independent 1D masks have a
singleton spatial axis. Nonempty masks identify detections, matched by saved unit
ID because localization QC may remove or reorder units. The returned curves keep
their source values and order. A requested result file must exist; an absent file
raises `FileNotFoundError`. `load_rf_profiles` and the collection's RF reader
accept the same `rf_type` option. The pair notebook exposes it as one parameter.

`rf_pick(table, max_missing_bins=2, max_zero_bins=2)` permits
at most two missing angular bins in the native 1-D projection, before interpolation
or smoothing. A bin is missing when all contributing y bins are NaN or unmeasured;
measured zeros count separately toward `max_zero_bins`. Occupancy and presentation
metadata identify unmeasured positions when raw files store them as zero, so these
positions count only as missing. Both limits use saved native counts before
interpolation or smoothing. `None` disables the corresponding limit; the default
`max_zero_bins=None` keeps measured zeros. The pair notebook explicitly sets both
limits to 2, excluding curves with more than two measured zero bins, including
entirely zero curves.

`load_ebc` projects every unit in the saved angular rate map, retaining zero and
entirely missing curves. Source angles keep their direction and wrap to [-180, 180):
90 stays 90 and 270 becomes −90. Other full-circle grids, including EBC's 60 bins,
are periodically interpolated while retaining missing intervals. To switch
between ego and allo labels on an existing full-circle table, use
`convert_profile_coordinates(table, range=tcRange(False))`; this preserves its
values and angular extent. Changing the extent belongs in the loader's `range`.

## Pooling and choosing rows

`combine(A, B, ...)` stacks rows without averaging and preserves per-unit filter
metadata. Tables sharing a grid retain the first table's bin order and display
settings. For different grids, supply an explicit common `range`: the paired
notebooks pool m14's partial RF grid and the other sessions on
`range=tcRange(True)`. m14 angles outside ±150° remain NaN in that pooled table.
The pooled curves use 12-degree bins, so peaks extracted from them can differ
from the native 10-degree m14 peaks; single-session plots use the native m14 grid.
Combining probes from the same m14 session instead retains `range=[-150, 150]`.
The function accepts an optional `label`, rejects duplicate unit keys, and
returns the same table structure for further filtering or combination.

```python
hd_all = combine(hd_m19, hd_m20, label="Pooled HD")
rf_all = combine(rf_m14, rf_m19, rf_m20, range=tcRange(True), label="Pooled RF")
hd = hd_pick(hd_all, hd_class=3)
rf = rf_pick(rf_all, max_missing_bins=2, max_zero_bins=2)
units = hd.index.intersection(rf.index)

# Optional numeric-ID selection; mouse/date/probe remain part of each key.
units = units[units.get_level_values("unit_id").isin([10, 20])]
pooled = plot_sort(normalize_tc(hd.loc[units]), normalize_tc(rf.loc[units]), **plot_options)
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
z_native = plot_sort(hd_z, rf_z, **z_options)
z_aligned = plot_align(hd_z, rf_z, **z_options)
```

Scale the final TC before alignment and before the duplicated boundary bin used
for circular display. Alignment interpolation may reduce a row's sampled maximum
or SD; plotting does not normalize it again. This retains one scale across native,
aligned, and summed views. Keep the original tables for analyses that require
rates, counts, or information about constant curves.

## Plotting and statistics

`plot_sort(reference, matched)`, `plot_align(reference, matched)` and
`plot_sum(reference, matched)` require the same unit-key set in both inputs.
Input row order may differ. Each function computes peaks and sorts by the first
argument; swapping arguments changes the reference. A supplied `order` must contain
all current keys exactly once. Plotting never intersects, filters, adds, or smooths
rows; unequal sets raise an error so callers can correct the selection. Response
scaling is always explicit: these functions and `plot_profiles` render the
supplied values without normalizing or z-scoring them.

- `plot_sort` keeps source angles.
- `plot_align` subtracts the reference peak from both curves.
- `plot_sum` centers the reference at zero and adds its peak to the matched curve.

Display options include `figsize=(10, 5)`, `labels=("Before", "After")`,
`show=False`, `cmap`, `vmin`, `vmax`, and `colorbar_label`. The default colorbar
label is `"Response"`; raw rates can use `"Firing rate (Hz)"` with appropriate
limits. Set common limits explicitly when colors should be comparable between
panels. Figures and exports use opaque white backgrounds and dark text.
All rows are retained and labeled with their full unit keys. Empty selected tables display
“No units to plot.” Rows without measured peaks remain missing.

Peak sums are reference + matched. `is_wrap=True` wraps to [-180, 180);
False retains direct sums and gives the matched sum panel a −360 through 360 degree
range. Each call returns `order`, `peaks`, offsets, and two `(figure, axes)` pairs.
With `show=False`, the caller closes returned figures.

Use a result's `peaks` table and retain rows with both finite peaks for direction
statistics. `compare_direction_angles` gives circular correlation and Spearman
Mantel rho with permutation p values; `plot_direction_comparison` shows the same-unit
scatter and pairwise circular-distance density. The notebooks use 10,000 permutations,
seed 1, and skip these tests with fewer than three measured pairs.
`plot_peak_direction_sums` gives a compact histogram with n, R, and Rayleigh p for
circular concentration. The hand-written histograms in `tc_comparison_pairs.ipynb`
retain the user-selected modulo-360° presentation.

The name-based `compare_both_orders` helper remains available for the specialized
notebook and requires an already-selected mapping with equal unit sets. Each
notebook comparison cell constructs that mapping with explicitly normalized
curves. It accepts the same color-scale options.

## Saved inputs for specialized analyses

`hd_rf_ebc_comparison.ipynb` retains complete native profiles in `all_profiles`,
then applies `hd_pick` and `rf_pick` into separate `profiles` tables. Each pair
computes its own overlap and passes `.loc` subsets to comparison functions.
Curve pools use `combine` so unit metadata survives across probes. Its low-level
readers use `(probe, unit_id)` keys within the single configured mouse and date.

The RF-y distance analysis separately calls `load_rf_centers`, which reads saved
localization centers and unit IDs from the same-name `.npz` and retains localized
units. Run `locate_rf` first when those centers are needed. Curve loading does not
rerun localization or read trials.

The specialized notebook reads HD9's `.tc`, rebuilds HD12 from Basler frame IDs
and exposure timing, and assigns classes independently per session. Generation
saves class lists; reading existing curves does not rewrite them. EBC retains all
source-map units. Boundary reconstruction and distance calculations are unchanged.

## Boundaries and preferred distance


Session 12 retains its existing numerical maps and distance-bin centers. Its
three filenames are `egocentric_rate_map_0-8.rfmap`, `_8-16.rfmap`, and
`_16-.rfmap`; the full map is `egocentric_rate_map.rfmap`. The saved grid ends
at 28.99 cm. Bands include centers ≤8, >8 and ≤16, and >16 cm, respectively.
Distance peaks come from maxima in the full angle×distance map, not the
distance-summed curves.

Session 9 uses Motive XY and the saved world-Z yaw. The center is the midpoint
of the XY extrema; the farthest valid point sets the inner 15 cm radius.
The outer radius is 30 cm with the same position scale. Each boundary has its
own ray intersections, spike counts, and occupancy. Full maps are saved under
`data/spatial_cells/ProbeA/baseline/{inner,outer}/egocentric_rate_map.rfmap`.
The outer directory also contains `_0-30.rfmap` and `_30-60.rfmap` curves.
New distance bins are right-closed, so exactly 30 cm belongs to the near band;
the origin belongs to the first bin. Angular bins are 6°, distances 1.5 cm.
Unvisited cells stay NaN after smoothing. Center, scale, diameters, and angle
convention are recorded in map metadata. The notebook also saves occupancy
arrays and the actual coverage table.

The three RF-y correlations use the same units with all three valid bands.
They report Spearman rho, two-sided unit-pairing permutation p, FDR-adjusted q,
and paired unit-bootstrap intervals. Undefined correlations in bootstrap
resamples are allowed their full admissible range when computing conservative
interval bounds, instead of dropping them. A constant original distance has
undefined rho. Tables report defined-bootstrap fractions and unique distance
counts. Paired rho differences use the same resampled unit indices; their
95% intervals are exploratory, without a further multiple-comparison correction.

## Artifact controls and interpretation

`ebc_artifact_controls.ipynb` contains its data preparation, statistical-control,
and plotting functions in the analysis definition cells. The Basler arena is
41 cm square. Frames outside the measured bounds are excluded from geometry
after spike-to-frame assignment.
The full, untruncated 60-direction distance vector is used for matching.

Video generation has three entrances: `ebc_video_rectangle.py` for Basler
rectangle overlays, `ebc_video_circle.py` for calibrated circular-screen camera
overlays, and `ebc_video_sep.py` for separate world/EBC panels. See
[EBC video entrances](ebc_videos.md) for settings and CLI usage. The rectangle
and circle entrances save all good units as `save_path/<unit_id>.mp4`. The default
`audio_source = "continuous"` reads the real Open Ephys `continuous.dat` voltage
on each unit's dominant-template peak channel, after unwhitening and applying
`channel_map.npy`. A causal 300–6000 Hz bandpass and 48 kHz resampling produce
OE-style spike-monitor audio at the original playback speed. Other units and
background on that electrode remain audible; this is not isolated-unit audio.
Set `audio_source = "clicks"` for synthetic clicks at that unit's sorted spike
times instead. Both modes use the original exposure midpoints and shared OE
clock for video alignment, including missing pose frames and clock drift.
Distinct electrodes are read and filtered together in one sequential pass;
temporary channel audio is reused for units sharing an electrode. `audio_gain`
sets the continuous track's peak level with one constant gain for the entire
clip. `audio_gate_sigma = 3.` suppresses background below three times the robust
noise estimate. Short centered windows retain voltage events, with smooth gate
edges and no time shift; the original global gain is retained after gating.
Set it to `0` to hear the full bandpassed channel. The gate can suppress weak
spikes and does not isolate the selected unit from other units on its electrode.
The circle entrance uses preserved Motive XYZ, the fixed level reference,
camera registration, and the full audited frame clock for its recording.
Geometry is calculated before projection onto the raw video. Explicit sync
gaps retain their original frames with omitted geometry and silent audio.
The sep entrance shows saved inner/outer EBC heatmaps beside the calibrated
world view for one selected unit. It can reuse an existing synchronized JSON.
Rectangle `video_start_s` is relative to the Basler AVI start; circle export
uses the full recording, while sep uses its selected time range.
The `auto` encoder mode checks NVENC initialization and uses eight CPU
encoding threads if NVENC is unavailable, reporting the actual ffmpeg error.

- Behavior states use 2 cm position and 12° HD bins. Geometry matching requires
  RMS distance-profile difference ≤2 cm, position separation ≥10 cm, and HD
  separation ≥60°. Selection uses behavior only, with no state reused.
- Same-HD comparisons use the same 12° bin and ≥10 cm position separation.
  A second analysis also matches speed bins at 2.5, 5, and 10 cm/s.
- Firing rates receive equal common-occupancy weights within each pair.
  Paired 10 s time-block bootstrap resampling supplies rate-difference
  intervals; individual sparse-state scatter points remain descriptive.
- Five contiguous held-out folds estimate HD lookup rates from the remaining
  time, excluding a 5 s guard. There are 500 Poisson controls per unit, using
  the real trajectory and the same occupancy and smoothing as observations.
  This follows the HD-only control idea in [Peyrache et al. (2017)](https://doi.org/10.1038/s41467-017-01908-3).
  No GLM is fitted. Zero-spike null draws have undefined MRL; the conditional
  MRL test reports its number of defined null samples.
- Spatial information uses the occupancy-weighted mean of the smoothed rate
  map. Three-band MRL tests are FDR-adjusted across units and bands. Split-half,
  left/right position, movement >2.5 cm/s, and lighter-smoothing correlations
  use common visited support. These correlations are descriptive; angular
  bins are not independent statistical replicates.

The evidence table requires an HD-null MRL q<0.05 and temporal and spatial
correlations ≥0.5 **in the same band** for “仍有额外 EBC 证据”. Sampling support
requires ≥50 total spikes, ≥10 matched seconds and ≥20 weighted spikes on
each side of both matching comparisons, and ≥95% HD lookup coverage.
“采样不足” also covers significant effects lacking stable replication;
separate columns distinguish these reasons. “HD/采样偏差可解释” expresses
compatibility, not proof that HD is the only mechanism. This observational
fixed-arena analysis cannot by itself prove genuine causal EBC coding.

Set `unit_id` near the top to inspect another unit's matches and null plots.
Pair CSVs refer to the states stored in `behavior_states.npz` and
`speed_states.npz`. `control_arrays.npz` retains observed and expected maps,
null curve intervals, and all null MRL and information samples.
