# RFMap Python API

This guide covers the public Python RF analysis interface: loading a pooled
regular RF source, summing a response window, rebuilding aligned raw trials,
and returning a 2-D mask, its discrete center, or a 1-D projection.

Analysis code only needs these imports:

```python
from Utils.rfmap import RFMap, RFMapList, asrfmap, load_rf_maps
from Utils.rf_trials import load_regular_rf_trials
```

RF detection is exposed through `RFMap.rf_2d()`, `RFMap.rf_1d()`,
`RFMapList.rf_2d()`, and `RFMapList.rf_1d()`. The caller chooses between the
complete mask and its center with one boolean: `is_center=False` returns the
mask and `is_center=True` returns the center.
All four methods default to `is_shuffle=False, drop_bins=2`.

## Quick start

```python
from pathlib import Path

from Utils.rfmap import load_rf_maps
from Utils.rf_trials import load_regular_rf_trials

session = Path("/mnt/senzailab/Kai/#Recording/m15/260630/260630_3")
rf_source = (
    session
    / "data/rfmapping/good/-100_400_1ms/ProbeA"
    / "regular_unitsSpikeCounts_260630_3.json"
)

raw = load_rf_maps(rf_source, unit_firing_rate=False)
summed = raw.sum(0.0, 0.2, show_progress=True)
trials = load_regular_rf_trials(
    session,
    "A",
    summed,
    on=True,
    off=False,
)
result_path = rf_source.with_suffix(".npz")

options = {
    "is_shuffle": True,
    "cluster_forming_z": 1.5,
    "alpha": 0.05,
    "n_permutations": 10_000,
    "random_seed": 0,
    "wrap_x": True,
    "result_path": result_path,
    "show_progress": True,
}

mask_2d = summed.rf_2d(trials, **options)
center_2d = summed.rf_2d(trials, is_center=True, **options)
mask_x = summed.rf_1d(trials, axis="x", **options)
mask_y = summed.rf_1d(trials, axis="y", **options)
```

The first call computes the statistical mask and its center and writes both to
`result_path`. The later calls validate and reuse that result; changing
`is_center` or the projection axis does not rerun the permutation.

For an `RFMapList`, the four output shapes above are `(unit, y, x)`,
`(unit, y, x)`, `(unit, x)`, and `(unit, y)`. A single `RFMap` has no leading
unit axis.

## Source files and result files

`.rfmap` is a source-data extension, not a result or cache extension:

- Regular pooled sources can be legacy JSON (`.json` or `.rfmap`) or MATLAB's
  indexed NPZ (`.rfmap`). `load_rf_maps()` detects the format from file contents.
- A free-moving `.rfmap` is HDF5 and follows its own source schema.
- A reusable detection result ends in `.npz`.

Do not write a result over a `.rfmap` source. `result_path` deliberately
requires the distinct `.npz` suffix.

The result sidecar is versioned and contains both binary arrays needed by the
public API: the complete 2-D mask and the discrete 2-D center. It also records
the aligned trial inputs, unit/grid identity, response window, and statistical
parameters through a validation key. Reuse occurs only when that identity
matches. A missing, stale, or mismatched result is recomputed instead of being
silently accepted.

On disk, both arrays always use canonical `(unit, y, x)` shape. A single
`RFMap` still returns `(y, x)` through the public methods; the leading unit axis
is removed only when that view is requested.

This is best thought of as a persisted analysis result, not another RF source.
It solves the common sequence “generate the mask now, request its center
later”: the first call has already saved both, so the second call reads the
validated center without another permutation.

`result_path` is optional. Without it, repeated compatible calls on the same
in-memory `RFMap` or `RFMapList` still reuse their result. Supplying a path also
allows reuse after restarting Python or reloading the source.

Keep the semantic arguments the same when reusing a result. For example, if
the first call used `alpha=0.01`, pass `alpha=0.01` to the center and 1-D calls
as well. A different trial dictionary, response window, unit set, grid, or
statistical parameter describes a different result.

## Object model and shapes

One pooled regular RF source stores counts in this order:

```text
(unit, y, x, time)
```

`load_rf_maps()` returns an ordered `RFMapList`. Each item is one `RFMap` with
shape `(y, x, time)`.

By default, loading divides `unitsSpikeCounts` by the spatial
`occupancyTimeSec`, producing firing-rate values in Hz without changing any
array shape. Pass `unit_firing_rate=False` to keep raw spike counts. Raw mode
is required when `load_regular_rf_trials()` validates reconstructed counts.
The returned metadata describes the loaded values: `responseUnits` is
`spike_count` or `Hz`, and conversion to Hz sets `responseNormalization` to
`occupancyTimeSec`. Sources already stored in Hz retain their normalization.

Zero-occupancy positions remain NaN in rate maps and are excluded from
non-shuffle RF detection. One-dimensional projections sum observed values;
a projected bin with no observed values remains NaN.

`occupancyTimeSec` is the display time at each position: the sum of the
qualifying trials' show times, or `show_time * trial_count` when show time is
constant. This is the normalization denominator even when the selected
response window is longer or shorter than show time. Non-shuffle detection
uses the loaded values directly, without another division by presentation
counts. Loading raw counts therefore also leaves those counts unnormalized
when detecting a non-shuffle RF.

| Value | `RFMapList` shape | One `RFMap` shape | Meaning |
| --- | --- | --- | --- |
| `raw.shape` | `(unit, y, x, time)` | `(y, x, time)` | Pooled count timeline |
| `summed.shape` | `(unit, y, x, 1)` | `(y, x, 1)` | One response bin |
| `rf_2d(..., is_center=False)` | `(unit, y, x)` | `(y, x)` | Complete binary RF mask |
| `rf_2d(..., is_center=True)` | `(unit, y, x)` | `(y, x)` | At most one selected bin per unit |
| `rf_1d(..., axis="x")` | `(unit, x)` | `(x,)` | Collapse y |
| `rf_1d(..., axis="y")` | `(unit, y)` | `(y,)` | Collapse x |

Use `where(value)` to locate exact values on the native count axes. It follows
NumPy tuple ordering: `(y, x, time)` for one map and
`(unit, y, x, time)` for a list. The unit result contains list offsets, not
recorded unit IDs:

```python
import numpy as np

zero_locations = summed.where(0)
units_with_any_zero_bin = np.unique(zero_locations[0])
unit_ids_with_any_zero_bin = np.asarray(summed.unit_ids)[units_with_any_zero_bin]
```

List position, source index, and recorded unit ID are separate concepts:

```python
by_position = summed[5]
by_source_index = summed.by_index(5)
by_recorded_id = summed.by_unit_id(127)
```

Use `by_unit_id()` when the number comes from cluster labels or another data
source. Do not assume a recorded unit ID is a Python list index.

## Summing the response window

The RF methods accept exactly one time bin. If the source contains a timeline,
sum the desired half-open interval first:

```python
summed = raw.sum(0.0, 0.2, show_progress=True)
```

This includes bins in `[0.0, 0.2)` and returns another `RFMap` or `RFMapList`
with a singleton time axis. Both endpoints must match actual `timeBinEdges`
values, in seconds, within `1e-12` seconds. Equal endpoints produce a valid
zero-valued singleton time axis; reversed intervals are invalid.

For an `RFMapList`, `show_progress=True` displays one `Sum` update per unit.
Pass `show_progress=False` for silent execution. A single `RFMap.sum()` is one
array reduction and does not create a progress bar.

If the source already has one bin, use it directly:

```python
summed = raw if raw[0].n_time_bins == 1 else raw.sum(0.0, 0.2)
```

There is deliberately no `time_range` argument on `rf_2d()` or `rf_1d()`.
The singleton-bin object is the response window, which prevents pooled counts
and reconstructed trial responses from silently using different windows.

An `RFMap` is also callable as a shorthand for `sum()`. Omitted bounds use the
first or last available time edge:

```python
rf_map(0.0, 0.204)          # rf_map.sum(0.0, 0.204)
rf_map(None, 0.204)         # first available edge through 0.204
rf_map(later_s=0.204)       # same as the line above
rf_map(0.204)               # 0.204 through the last available edge
rf_map()                    # the complete available time window
```

Python does not allow an omitted argument before a comma, so
`rf_map(, 0.204)` is invalid syntax; use `None` or the `later_s` keyword.

## Subtracting two summed maps

Subtraction is defined between compatible `RFMap` objects that each already
contain exactly one time bin. Sum each source window first:

```python
response_minus_baseline = (
    rf_map.sum(0.1, 0.2) - rf_map.sum(0.0, 0.1)
)
```

The operation subtracts the stored values element by element, so negative
values are valid. The result keeps the left-hand map's time window; in this
example that is `[0.1, 0.2)`. Default-loaded maps therefore produce a firing
rate difference, while maps loaded with `unit_firing_rate=False` produce a raw
pooled spike-count difference.

## Loading regular trials

The pooled source has no trial axis, so it cannot support a label-permutation
test by itself. Rebuild matching regular sparse-noise trials from the raw
session:

```python
trials = load_regular_rf_trials(
    session,
    "A",
    summed,
    on=True,
    off=False,
)
```

The loader resolves and validates the session MAT file, stimulus onsets, probe
spike times, Kilosort cluster labels, and good-unit labels. It checks the grid,
unit IDs, response window, repeat structure, and pooled counts against
`summed`. It returns aligned trial responses, joint spatial labels,
exchangeability strata, positions, unit IDs, and provenance.
The onset file may contain exactly one timestamp per trial, or one additional
terminal boundary. Only the first N timestamps are used for N trials; the
terminal interval need not equal the stimulus duration. Provenance records
whether the extra boundary was excluded in `terminal_edge_excluded`.

Exactly one polarity flag must be true. `on=True, off=False` selects
`Square_Luminance == 1`; `on=False, off=True` selects
`Square_Luminance == 0`. The default is ON. The supplied pooled RF source must
match the selected polarity when `validate_pooled=True`.

Positions are shuffled as one joint `(x, y)` label. Responses are never
shuffled. The loader first filters the requested polarity, then uses repeat
blocks as strata so each permutation changes only the position-response
relationship described by the null hypothesis. Each repeat is validated
against the luminances actually present in the session, so ON-only, OFF-only,
and ON+OFF designs use different valid block sizes.

This loader is for regular one-position-per-trial sparse noise. Pixel-bin,
rotation, egocentric, or transformed maps need different label semantics and
must not be passed through it.

## RF method arguments

Both `rf_2d()` and `rf_1d()` accept the same statistical arguments; `rf_1d()`
also takes `axis="x"` or `axis="y"`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `is_center` | `False` | Return the complete mask; `True` returns its discrete center |
| `is_shuffle` | `False` | Run cluster-permutation significance testing when explicitly enabled |
| `drop_bins` | `2` | No-shuffle only: remove components of this size or smaller |
| `cluster_forming_z` | `1.5` | Select pixels allowed to form candidate clusters |
| `alpha` | `0.05` | Cluster-level significance cutoff |
| `n_permutations` | `10_000` | Number of shuffled null maps |
| `alternative` | `"greater"` | Test increased response; `"less"` tests decreased response |
| `wrap_x` | `True` | Treat the first and last x columns as adjacent |
| `fill_single_holes` | `False` | Fill eligible one-bin holes inside significant clusters |
| `min_hole_neighbors` | `3` | Required 4-connected significant neighbors for hole filling |
| `random_seed` | `0` | Reproducible permutation seed; `None` is nondeterministic |
| `batch_size` | `64` | Permutation chunk size; does not change result identity |
| `n_jobs` | `None` | Worker limit; does not change result identity |
| `result_path` | `None` | Optional validated `.npz` result |
| `show_progress` | `True` | Show applicable progress; use `False` for silent execution |

Use `wrap_x=True` only when the x grid is genuinely periodic, such as a
360-degree display. It changes both candidate connectivity and center distance
at the left/right seam.

`cluster_forming_z=1.5` is not a pixelwise `p < 0.05` threshold. It only
determines which neighboring pixels can enter candidate clusters.
Significance is determined at the cluster level by the shuffled maximum-mass
distribution.

## What `is_shuffle=True` tests

For every retained trial, the spike response stays fixed. Within each allowed
stratum, the joint spatial label is randomly permuted. Each permutation then:

1. Recomputes mean response by position.
2. Applies the same z transform and cluster-forming threshold.
3. Finds 4-connected clusters, respecting `wrap_x`.
4. Sums z values within each cluster.
5. Stores the largest cluster mass from that permutation.

The empirical cluster p-value uses the plus-one rule:

```text
(1 + number of shuffled maxima >= real cluster mass)
----------------------------------------------------
                  n_permutations + 1
```

Using the maximum cluster from every shuffle controls spatial family-wise
error within one unit. It does not additionally correct across units, probes,
stimulus polarities, or separately run analyses.

The returned mask contains clusters whose empirical p-value is no greater
than `alpha`. The 1-D result is a projection of this final 2-D mask, not an
independent 1-D significance test.

When `is_shuffle=True`, `drop_bins` is deliberately ignored. The permutation
result therefore remains identical to the existing significance procedure,
regardless of the supplied `drop_bins` value.

## What `is_shuffle=False` returns

No-shuffle mode is exploratory. It skips the null distribution and begins
with the cluster-forming candidate pixels. `drop_bins` then filters whole
candidate components:

```python
mask = summed.rf_2d(
    trials,
    is_shuffle=False,
    drop_bins=1,
    cluster_forming_z=1.5,
)
```

Components use 4-connectivity and respect `wrap_x`. Every component containing
`drop_bins` bins or fewer is removed:

- `drop_bins=0` keeps every candidate component.
- `drop_bins=1` removes isolated one-bin components.
- `drop_bins=2` removes one- and two-bin components.

The value must be a non-negative integer. Filtering is component-based; it
does not sort pixels globally and does not remove an arbitrary number of the
weakest bins. A retained no-shuffle mask is not permutation-significant and
must not be reported as such.

## Complete masks, centers, and 1-D projections

The normal output is the complete binary area:

```python
mask_2d = summed.rf_2d(trials, is_center=False)
mask_x = summed.rf_1d(trials, axis="x", is_center=False)
mask_y = summed.rf_1d(trials, axis="y", is_center=False)
```

Select the center through the same public methods:

```python
center_2d = summed.rf_2d(trials, is_center=True)
center_x = summed.rf_1d(trials, axis="x", is_center=True)
center_y = summed.rf_1d(trials, axis="y", is_center=True)
```

Mask calculation is required before center calculation. The implementation
therefore obtains both as one reusable result rather than running the detector
again for `is_center=True`. Units with an empty mask remain all zero. Every
non-empty mask contributes exactly one selected center bin.

For a non-empty RF, the response-effect weight within the final mask is:

```text
greater: w[y, x] = max(response_map[y, x] - null_mean_map[y, x], 0)
less:    w[y, x] = max(null_mean_map[y, x] - response_map[y, x], 0)
```

The center is the RF-positive bin `p` minimizing
`sum_q w[q] * distance(p, q)^2`. This discrete weighted medoid cannot fall
between bins or outside the final mask. With `wrap_x=True`, horizontal distance
is circular. A zero total weight falls back to equal RF-bin weights; remaining
ties prefer the higher local weight and then row-major order.

The 1-D methods use logical projection of the selected 2-D output. They never
run separate 1-D statistics.

## Batch workflow

An `RFMapList` runs one aligned batch and returns arrays with a leading unit
axis:

```python
masks = summed.rf_2d(
    trials,
    n_permutations=10_000,
    result_path=result_path,
)

for unit_id, mask in zip(summed.unit_ids, masks):
    print(unit_id, int(mask.sum()))
```

The trial loader and RF methods align responses by recorded unit ID. Output
array order follows `RFMapList.unit_ids`.

Keep probe identity separate because unit IDs can overlap between probes:

```python
results_by_probe = {}

for probe in ("A", "B"):
    source = (
        session
        / "data/rfmapping/good/-100_400_1ms"
        / f"Probe{probe}"
        / "regular_unitsSpikeCounts_260630_3.json"
    )
    raw = load_rf_maps(source, unit_firing_rate=False)
    summed = raw.sum(0.0, 0.2)
    trials = load_regular_rf_trials(session, probe, summed)
    results_by_probe[probe] = summed.rf_2d(
        trials,
        n_permutations=10_000,
        wrap_x=True,
        result_path=source.with_suffix(".npz"),
    )
```

## Plotting with physical positions

Array indices are not necessarily visual degrees. Use the positions attached
to an `RFMap` when setting plot extents:

```python
import matplotlib.pyplot as plt

unit = summed.by_unit_id(127)
unit_mask = unit.rf_2d(trials, wrap_x=True)

plt.imshow(
    unit_mask,
    origin="lower",
    aspect="auto",
    extent=(
        unit.x_positions[0],
        unit.x_positions[-1],
        unit.y_positions[0],
        unit.y_positions[-1],
    ),
)
plt.xlabel("x position")
plt.ylabel("y position")
```

## Constructing a map from an array

Use `asrfmap()` for a standalone array:

```python
import numpy as np

from Utils.rfmap import asrfmap

single_frame = asrfmap(
    np.zeros((7, 30)),
    start_time=0.0,
    end_time=0.2,
)
```

A 2-D input receives a singleton time axis. A 3-D input must use
`(y, x, time)` axis order. `asrfmap()` validates numeric values, timing, and
geometry and keeps stored arrays read-only.

An array-created map has no corresponding raw-session trial dictionary unless
the caller supplies matching trial data. Pooled spatial values alone cannot be
used to invent a valid permutation null.

## Timing caveat

In the regular session used by `locate_rf.ipynb`, stimuli are spaced about
100 ms apart. A `[0.0, 0.2)` response window therefore overlaps the next
stimulus. Negative bins can likewise overlap the previous stimulus.

The permutation test asks whether response is associated with the assigned
position under the chosen trial construction. It does not repair temporal
overlap or prove that every spike in a long window was caused only by the
current stimulus. Inspect the full `timeBinEdges` and timeline before giving a
causal interpretation to the detected RF.

## Common errors

### Detection requires exactly one time bin

Call `sum()` first, then build trials for that exact object:

```python
summed = raw.sum(0.0, 0.2)
trials = load_regular_rf_trials(session, "A", summed)
mask = summed.rf_2d(trials)
```

### A requested endpoint is not in `timeBinEdges`

Inspect the actual edges and choose exact values:

```python
print(raw[0].time_bin_edges_s)
```

The API does not snap an arbitrary time to the nearest bin.

### Trial geometry or response window does not match

Build trials from the exact `summed` object passed to `rf_2d()` or `rf_1d()`.
Do not reuse a trial dictionary made for another probe, time window, grid, or
unit set.

### Raw trial totals disagree with the pooled source

Treat this as a data-alignment failure. Check the selected session and probe,
MAT trial count, onset edges, spike-time array, cluster-label array, response
window, and unit IDs. Do not disable validation merely to obtain a mask.

### A result is not reused

Confirm that the path ends in `.npz` and that the same aligned trials,
unit set, grid, response window, and statistical parameters are being used.
`is_center` and `rf_1d()`'s axis choose a view of the stored result; they do not
make a new statistical result.

Never rename a result to `.rfmap`. The latter identifies source data and has a
different contract.

### Candidate pixels exist but no significant RF remains

This is a valid outcome. The cluster-forming threshold is deliberately
permissive; the shuffled maximum-cluster distribution decides significance.

### `is_shuffle=False` returns a mask

That mask is exploratory. It contains candidate components larger than
`drop_bins` and is not a permutation-significant RF.

### The output cannot be modified

Analysis arrays are read-only to prevent accidental mutation. Make an explicit
copy when a writable array is required:

```python
import numpy as np

writable = np.array(mask_2d, copy=True)
```

## Compact API reference

| API | Returns | Purpose |
| --- | --- | --- |
| `load_rf_maps(path, unit_firing_rate=True)` | `RFMapList` | Load firing rates (`count / occupancyTimeSec`) by default; pass `False` for raw counts |
| `asrfmap(array, ...)` | `RFMap` | Validate one standalone array |
| `rf_map.sum(start, end)` | `RFMap` | Sum a half-open response window |
| `rf_map(earlier_s=None, later_s=None)` | `RFMap` | Callable shorthand for `sum()`, with omitted bounds resolved to the available edges |
| `rf_maps.sum(start, end, show_progress=...)` | `RFMapList` | Sum the same window for all units |
| `left_rf_map - right_rf_map` | `RFMap` | Elementwise signed difference between compatible singleton-bin maps; keep the left time window |
| `load_regular_rf_trials(session, probe, summed, on=..., off=...)` | `dict` | Reconstruct aligned ON or OFF regular trials |
| `summed.rf_2d(trials, is_center=..., result_path=..., ...)` | read-only `uint8` array | Return the full 2-D mask or center |
| `summed.rf_1d(trials, axis=..., is_center=..., result_path=..., ...)` | read-only `uint8` array | Project the same 2-D mask or center |
| `rf_maps.by_index(index)` | `RFMap` | Select by original source unit index |
| `rf_maps.by_unit_id(unit_id)` | `RFMap` | Select by recorded unit or cluster ID |
| `rf_maps.to_4d_array()` | array | Stack pooled count timelines |
| `summed.to_2d_array()` | array | Return singleton-bin loaded values |
| `rf_map.where(value)` | index tuple | Locate matches as `(y, x, time)` |
| `rf_maps.where(value)` | index tuple | Locate matches as `(unit, y, x, time)` |

The authoritative statistical object is the final 2-D mask. Centers and 1-D
arrays are deterministic views of that mask and share the same validated
result.
