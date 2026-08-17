# RFMap Python API

This guide covers the Python RF analysis interface: loading pooled JSON maps,
summing a response window, rebuilding raw regular trials, and detecting a
spatial RF with a trial-label cluster permutation test.

The analysis-facing imports are intentionally small:

```python
from Utils.rfmap import RFMap, RFMapList, asrfmap, load_rf_maps
from Utils.rf_trials import load_regular_rf_trials
```

`load_regular_rf_trials()` returns an ordinary dictionary. `detect_rf()` also
returns an ordinary dictionary, so analysis code does not need configuration,
trial-container, or result-wrapper objects.

## Quick start

```python
from pathlib import Path

import numpy as np

from Utils.rfmap import load_rf_maps
from Utils.rf_trials import load_regular_rf_trials

session = Path("/mnt/senzailab/Kai/#Recording/m15/260630/260630_3")
rf_json = (
        session
        / "data/rfmapping/good/-100_400_1ms/ProbeA"
        / "regular_unitsSpikeCounts_260630_3.json"
)

raw = load_rf_maps(rf_json)
summed = raw.sum(0.0, 0.2, show_progress=True)
trials = load_regular_rf_trials(session, "A", summed)

result = summed._detect_rf(
    trials,
    is_shuffle=True,
    cluster_forming_z=1.5,
    alpha=0.05,
    n_permutations=10_000,
    random_seed=0,
    wrap_x=True,
    n_jobs=None,
    show_progress=True,
)

rf_2d = result["final_mask"]
rf_1d_x = np.any(rf_2d, axis=1).astype(np.uint8)
rf_1d_y = np.any(rf_2d, axis=2).astype(np.uint8)

print("2-D shape:", rf_2d.shape)
print("x projection shape:", rf_1d_x.shape)
print("workers used:", result["n_workers"])
```

For an `RFMapList`, the shapes above are `(unit, y, x)`, `(unit, x)`, and
`(unit, y)`. For one `RFMap`, omit the unit axis and collapse y with `axis=0`
or x with `axis=1`.

## Object model and shapes

One pooled RF JSON has count data in this order:

```text
(unit, y, x, time)
```

`load_rf_maps()` returns an ordered `RFMapList`. Each item is one `RFMap` with
shape `(y, x, time)`.

| Value | Shape | Meaning |
| --- | --- | --- |
| `raw.shape` | `(unit, y, x, time)` | Logical batch shape |
| `raw.to_4d_array()` | `(unit, y, x, time)` | Stacked pooled counts |
| `summed.shape` | `(unit, y, x, 1)` | One summed response bin per unit |
| `summed.to_2d_array()` | `(unit, y, x)` | Summed response counts |
| `result["final_mask"]` | `(unit, y, x)` | Final binary RF masks |
| `np.any(mask, axis=1)` | `(unit, x)` | Projection along x |
| `np.any(mask, axis=2)` | `(unit, y)` | Projection along y |

For one `RFMap`, the same arrays have no leading unit dimension.

Use `where(value)` to locate exact values on the object's native count axes.
It follows `numpy.where` tuple ordering: `(y, x, time)` for one `RFMap` and
`(unit, y, x, time)` for an `RFMapList`. The unit result contains list offsets,
not recorded unit IDs, and repeats an offset when several bins match:

```python
zero_locations = summed.where(0)
zero_unit_offsets = np.unique(zero_locations[0])
zero_unit_ids = np.asarray(summed.unit_ids)[zero_unit_offsets]
```

A summed map retains a singleton time axis, so the final index array returned
by `where()` contains zeros. Use `np.unique()` on the first array when the goal
is one entry per matching unit.

Unit list position and recorded unit ID are separate concepts:

```python
by_position = summed[5]
by_source_index = summed.by_index(5)
by_recorded_id = summed.by_unit_id(127)
```

Use `by_unit_id()` when the number comes from cluster labels or another data
source. Do not assume that a recorded unit ID is a Python list index.

## Summing the response window

RF detection accepts exactly one time bin. If the JSON contains a timeline,
sum the desired half-open interval first:

```python
summed = raw.sum(0.0, 0.2, show_progress=True)
```

The operation includes bins in `[0.0, 0.2)` and returns another `RFMap` or
`RFMapList` whose time axis has length one. Both endpoints must match actual
`timeBinEdges` values. Times are seconds.

For an `RFMapList`, `show_progress=True` displays the `Sum` bar once per unit.
Pass `show_progress=False` for silent execution. A single `RFMap.sum()` is one
array reduction and does not create a progress bar.

If the loaded map already has one bin, use it directly:

```python
summed = raw if raw[0].n_time_bins == 1 else raw.sum(0.0, 0.2)
```

There is deliberately no `time_range` argument on `detect_rf()`, `rf_2d()`, or
`rf_1d()`. The single-bin object is the response window, which prevents the
pooled map and reconstructed trial responses from silently using different
windows.

## Loading regular trials

The pooled JSON has no trial axis, so it cannot support a label-permutation
test by itself. Rebuild the matching regular sparse-noise trials from the raw
session:

```python
trials = load_regular_rf_trials(session, "A", summed)
```

The loader resolves and validates the session MAT file, stimulus onsets, probe
spike times, and Kilosort cluster labels. It checks the grid, unit IDs, response
window, repeat structure, and pooled counts against `summed`. It returns a
dictionary containing the aligned trial responses, joint spatial labels,
exchangeability strata, positions, unit IDs, and provenance.

Positions are shuffled as one joint `(x, y)` label. Responses are never
shuffled. The loader first filters the requested ON or OFF polarity, then uses
repeat blocks as strata so each permutation changes only the position-response
relationship that the null hypothesis is meant to destroy.

The loader is intentionally for regular one-position-per-trial sparse noise.
Pixel-bin, rotation, egocentric, or transformed maps need different label
semantics and must not be passed through this loader.

## Detecting an RF

`detect_rf()` exposes ordinary keyword arguments:

```python
result = summed._detect_rf(
    trials,
    is_shuffle=True,
    cluster_forming_z=1.5,
    alpha=0.05,
    n_permutations=10_000,
    random_seed=0,
    wrap_x=True,
    n_jobs=None,
    show_progress=True,
)
```

The important arguments are:

| Argument | Default | Meaning |
| --- | --- | --- |
| `is_shuffle` | `True` | Run the permutation significance test |
| `cluster_forming_z` | `1.5` | Select pixels allowed to form candidate clusters |
| `alpha` | `0.05` | Cluster-level significance cutoff |
| `n_permutations` | `10_000` | Number of shuffled null maps |
| `random_seed` | `0` | Reproducible permutation seed; `None` is nondeterministic |
| `wrap_x` | `True` | Treat the first and last x columns as adjacent |
| `n_jobs` | `None` | Auto: one worker for one unit; up to two across multiple units |
| `show_progress` | `True` | Show permutation progress; use `False` for silent execution |

Use `wrap_x=True` only when the x grid is genuinely periodic, such as the
360-degree regular RF display. It changes cluster connectivity at the left and
right borders.

`cluster_forming_z=1.5` is not a `p < 0.05` pixel threshold. It only determines
which neighboring pixels can enter a candidate cluster. Significance comes
from comparing a real cluster mass with the shuffled maximum-cluster-mass
distribution.

### What the shuffle tests

For every retained trial, the spike response stays fixed. Within each allowed
stratum, the joint spatial label is randomly permuted. Each permutation then:

1. Recomputes mean response by position.
2. Applies the same z transform and cluster-forming threshold.
3. Finds 4-connected clusters.
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
or separately analyzed stimulus conditions.

### Result dictionary

The final 2-D output is always:

```python
mask = result["final_mask"]
```

The dictionary also retains the statistical diagnostic arrays produced by the
detector, including the response and z maps, candidate and significant masks,
cluster labels and masses, empirical p-values, and null maxima. Use the keys
directly rather than wrapping the dictionary in another container.

The actual worker count is:

```python
result["n_workers"]
```

This makes automatic thread selection observable and makes performance reports
reproducible.

### `is_shuffle=False`

```python
exploratory = summed._detect_rf(
    trials,
    is_shuffle=False,
    cluster_forming_z=1.5,
)
candidate_mask = exploratory["final_mask"]
```

This skips the shuffled null and returns the cluster-forming candidate mask. It
is useful for exploration, but it is not a significant RF and has no valid
permutation p-value.

## Automatic threading

Permutation work is distributed across units for an `RFMapList`. A one-unit
`RFMap` can also split permutation chunks when `n_jobs > 1`. The default is
automatic:

```python
result = summed._detect_rf(trials, n_jobs=None)
print(result["n_workers"])
```

Automatic mode respects the CPUs available to the process and applies an
internal safety cap. It uses one worker for a single RFMap and up to two for an
RFMapList. This is deliberate: real 7 x 30 single-unit benchmarks made two
threads about 3--6% slower, while the multi-unit workload benefits from two.

For an explicit serial run:

```python
serial = summed._detect_rf(trials, n_jobs=1)
```

For an explicit upper bound:

```python
parallel = summed._detect_rf(trials, n_jobs=4)
```

With the same data, seed, and statistical arguments, serial and threaded runs
produce the same masks and null results. Parallelism changes scheduling, not
the permutation sequence.

## Getting 2-D and 1-D masks

If diagnostic statistics are needed, call `detect_rf()` once and reuse its
mask:

```python
result = summed._detect_rf(trials)
rf_2d = result["final_mask"]
rf_x = np.any(rf_2d, axis=1).astype(np.uint8)  # RFMapList: collapse y
rf_y = np.any(rf_2d, axis=2).astype(np.uint8)  # RFMapList: collapse x
```

For a single `RFMap`:

```python
unit = summed.by_unit_id(127)
unit_result = unit._detect_rf(trials)
unit_2d = unit_result["final_mask"]
unit_x = np.any(unit_2d, axis=0).astype(np.uint8)
unit_y = np.any(unit_2d, axis=1).astype(np.uint8)
```

If only the binary output is needed, convenience methods are available:

```python
rf_2d = summed.rf_2d(trials)
rf_x = summed.rf_1d(trials, axis="x")
rf_y = summed.rf_1d(trials, axis="y")
```

To return one discrete, response-weighted center bin per unit instead of the
complete RF area:

```python
center_2d = summed.rf_2d(
    trials,
    return_center=True,
    show_progress=True,
)
center_x = np.any(center_2d, axis=1).astype(np.uint8)
center_y = np.any(center_2d, axis=2).astype(np.uint8)
```

The detector always computes `final_mask` first. Units whose mask is empty stay
all zero and are skipped by the `Center` bar; if every unit is empty, no center
bar is created. For a non-empty RF, define the response-effect weight inside
the final mask as:

```text
greater: w[y, x] = max(response_map[y, x] - null_mean_map[y, x], 0)
less:    w[y, x] = max(null_mean_map[y, x] - response_map[y, x], 0)
```

The returned center is the RF-positive bin `p` minimizing
`sum_q w[q] * distance(p, q)^2`. This discrete weighted medoid cannot fall
between bins or outside `final_mask`. With `wrap_x=True`, horizontal distance
is circular. A zero total weight falls back to equal RF-bin weights; remaining
ties prefer the higher local weight and then row-major order.

The 1-D methods are logical projections of the final 2-D mask. They are not
independent 1-D significance tests.

Do not call `detect_rf()` and then call `rf_2d()` with the same inputs: the
second call runs the permutations again. Reuse `result["final_mask"]` instead.
Likewise, derive both 1-D projections from that stored mask when both are
needed.

## Batch workflow

An `RFMapList` runs one aligned batch and returns arrays with a leading unit
axis:

```python
result = summed._detect_rf(
    trials,
    n_permutations=10_000,
    n_jobs=None,
)

masks = result["final_mask"]
for unit_id, mask in zip(summed.unit_ids, masks):
    print(unit_id, int(mask.sum()))
```

The trial loader and detector align responses by recorded unit ID. The returned
array order is the `RFMapList` order.

Probe identity must remain separate because unit IDs can overlap between
probes:

```python
results_by_probe = {}

for probe in ("A", "B"):
    probe_json = (
            session
            / "data/rfmapping/good/-100_400_1ms"
            / f"Probe{probe}"
            / f"regular_unitsSpikeCounts_260630_3.json"
    )
    raw = load_rf_maps(probe_json)
    summed = raw.sum(0.0, 0.2)
    trials = load_regular_rf_trials(session, probe, summed)
    results_by_probe[probe] = summed._detect_rf(
        trials,
        n_permutations=10_000,
        wrap_x=True,
        n_jobs=None,
    )
```

## Plotting with physical positions

Array indices are not necessarily visual degrees. Use the positions attached to
the `RFMap` when setting plot extents:

```python
import matplotlib.pyplot as plt

unit = summed.by_unit_id(127)
unit_result = unit._detect_rf(trials, wrap_x=True)

plt.imshow(
    unit_result["final_mask"],
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

A 2-D input receives a singleton time axis. A 3-D input must use `(y, x, time)`
axis order. `asrfmap()` validates numeric values, timing, and geometry and keeps
the stored arrays read-only.

An array-created map has no corresponding raw-session trial dictionary unless
the caller supplies matching trial data. Pooled spatial values alone cannot be
used to invent a valid permutation null.

## Timing caveat

In the regular session used by `locate_rf.ipynb`, stimuli are spaced about
100 ms apart. A `[0.0, 0.2)` response window therefore overlaps the next
stimulus. Negative bins can likewise overlap the previous stimulus.

The permutation test answers whether response is associated with the assigned
position under the chosen trial construction. It does not repair temporal
overlap or prove that every spike in a long window was caused only by the
current stimulus. Inspect the full `timeBinEdges` and timeline before giving a
causal interpretation to the detected RF.

## Common errors

### Detection requires exactly one time bin

Call `sum()` first:

```python
summed = raw.sum(0.0, 0.2)
result = summed._detect_rf(
    load_regular_rf_trials(session, "A", summed)
)
```

### A requested endpoint is not in `timeBinEdges`

Inspect the actual edges and choose exact values:

```python
print(raw[0].time_bin_edges_s)
```

The API does not snap an arbitrary time to the nearest bin.

### Trial geometry or response window does not match

Build trials from the exact `summed` object passed to detection. Do not reuse a
trial dictionary made for another probe, time window, grid, or unit set.

### Raw trial totals disagree with pooled JSON

Treat this as a data-alignment failure. Check the selected session and probe,
MAT trial count, onset edges, spike-time array, cluster-label array, response
window, and unit IDs. Do not disable validation merely to obtain a mask.

### Invalid `n_jobs`

Use `None` for automatic selection or a positive integer. `n_jobs=1` is the
explicit serial setting.

### Candidate pixels exist but no significant RF remains

This is a valid outcome. The cluster-forming threshold is deliberately
permissive; the shuffled maximum-cluster distribution decides significance.

### `is_shuffle=False` returns a mask

That mask is exploratory. It is the candidate layer and must not be reported as
a permutation-significant RF.

### The output cannot be modified

Analysis arrays are read-only to prevent accidental mutation. Make an explicit
copy when an independent writable array is required:

```python
writable = np.array(result["final_mask"], copy=True)
```

## Compact API reference

| API | Returns | Purpose |
| --- | --- | --- |
| `load_rf_maps(path)` | `RFMapList` | Load and validate one pooled RF JSON |
| `asrfmap(array, ...)` | `RFMap` | Validate one standalone array |
| `rf_map.sum(start, end)` | `RFMap` | Sum a half-open response window |
| `rf_maps.sum(start, end, show_progress=...)` | `RFMapList` | Sum the same window for all units |
| `load_regular_rf_trials(session, probe, summed)` | `dict` | Reconstruct aligned raw regular trials |
| `summed.detect_rf(trials, ...)` | `dict` | Run cluster detection and retain statistics |
| `summed.rf_2d(trials, return_center=..., show_progress=..., ...)` | `uint8` array | Return the final 2-D mask or one center bin |
| `summed.rf_1d(trials, axis=..., return_center=..., show_progress=..., ...)` | `uint8` array | Project the same final 2-D mask or center |
| `rf_maps.by_index(index)` | `RFMap` | Select by original JSON unit index |
| `rf_maps.by_unit_id(unit_id)` | `RFMap` | Select by recorded unit or cluster ID |
| `rf_maps.to_4d_array()` | array | Stack pooled count timelines |
| `summed.to_2d_array()` | array | Return singleton-bin count maps |
| `rf_map.where(value)` | index tuple | Locate matches as `(y, x, time)` |
| `rf_maps.where(value)` | index tuple | Locate matches as `(unit, y, x, time)` |

The authoritative 2-D significance output is `result["final_mask"]`. Derive
downstream views from that one mask so 2-D and 1-D analyses cannot drift apart.
