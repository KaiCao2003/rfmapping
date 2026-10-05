# RF Python API

Use `Utils.rflocate` for RF response data, saved detections, and explicit
analysis. The API has three core concepts:

| Concept | Responsibility |
| --- | --- |
| `RFMap` / `RFMapList` | Response values, unit IDs, spatial coordinates, time bins, and explicit numeric transformations |
| `RFResult` | Detected masks, centers, unit IDs, and detection provenance |
| `detect_rf()` | Run a requested detection on prepared responses and return an `RFResult` |

Loading, transforming responses, and viewing an existing result do not run
RF detection. Complete recording analyses are separate workflow functions.

## Start with a source and an existing detection

```python
from pathlib import Path
from Utils.rflocate import RFMapList, load_rfmap, load_rf, rf_result_path

source = Path(
    "/mnt/senzailab/Kai/#Recording/m15/260630/260630_3"
    "/data/rfmapping/good/-100_400_1ms/ProbeA"
    "/regular_unitsSpikeCounts_260630_3.rfmap"
)

raw = load_rfmap(source)
rf = load_rf(rf_result_path(source))

# Saved detections may contain only units that passed analysis QC.
selected = RFMapList(
    [raw.by_unit_id(unit_id) for unit_id in rf.unit_ids],
    source_path=raw.source_path,
)

# Retain x and sum only whole rows containing the unit's detected 2-D RF.
x_responses = selected.sum_to_1d(axis="x", rf_only=True, detected_rf=rf)
x_profiles = x_responses.sum(0.0, 0.2).to_1d_array(axis="x")
```

`load_rfmap()` returns an `RFMapList` and preserves the source's values, units,
zeros, and missing bins. `load_rf()` only reads a saved detection; a missing or
malformed result raises an error. Neither loader chooses a response window,
reconstructs exposure, selects units, or detects RFs.
The example explicitly selects the units stored in the detection, in its saved
order. `sum_to_1d(rf_only=True)` requires a detection for every supplied unit;
it does not silently drop units absent from the result.

## Public entrypoints and code layout

The normal import surface is:

```python
from Utils.rflocate import (
    RFMap, RFMapList, RFResult, asrfmap,
    load_rfmap, load_rf, save_rf, rf_result_path, detect_rf,
    load_rf_tc, save_rf_tc,
)
```

| Module | Contents |
| --- | --- |
| `Utils.rflocate.models` | Response objects and numeric transformations |
| `Utils.rflocate.results` | Detected result object and deterministic mask views |
| `Utils.rflocate.io` | Source and detection file I/O |
| `Utils.rflocate.detection` | Explicit 2-D or independent 1-D detection |
| `Utils.rflocate.rates` | Presentation counts and response-rate conversion |
| `Utils.rflocate.trials` | Reconstruction of aligned regular-stimulus trials |
| `Utils.rflocate.plotting` | Plot data preparation, plots, and figure export |
| `Utils.rflocate.workflow` | Complete file analysis, QC, and result export |

Modules beginning with `_` are implementation details. The root API does not
import plotting or the complete recording workflow. Existing root notebooks
and the MATLAB bridge `locate_rf.py` retain their locations.

## Response objects and explicit transformations

A source has `(unit, y, x, time)` axes. Each item in its `RFMapList` is one
`RFMap` with `(y, x, time)` axes. Selecting units keeps their recorded IDs:

```python
unit = raw.by_unit_id(127)  # Recorded cluster ID.
first = raw[0]             # Current list position.
original = raw.by_index(5) # Original source index.
```

Unit IDs may overlap between probes. Keep session and probe identity alongside
the objects; a list offset is not a recorded unit ID.

### Time window and response units

```python
counts = raw.sum(0.0, 0.2)
rates = raw.to_firing_rate()
mean_rates = rates.mean_rate(0.0, 0.2)
```

`sum()` literally adds stored values, with no normalization. Its half-open
window is in seconds: `[0.0, 0.2)` includes 0–200 ms. Both endpoints must match
actual `timeBinEdges` within `1e-12` seconds; arbitrary endpoints are not
snapped to a bin. The result retains a singleton time axis. Equal endpoints
produce a zero-valued singleton; reversed intervals are invalid.

`to_firing_rate()` explicitly computes
`count / (stimulusPresentationCounts * bin_width_seconds)`. Presentation counts
are distinct from cumulative `occupancyTimeSec`. If counts are absent, supply
`presentation_counts=...` or explicitly use
`to_firing_rate(reconstruct_presentations=True)` to read the matching session
logs. Loading never reconstructs them. Precomputed Hz maps retain their source
values; legacy occupancy-normalized rates can be converted explicitly through
their original counts.

`mean_rate()` requires Hz input and calculates the time-weighted mean, including
unequal bin widths. Missing contributing values propagate through sums and
averages. Measured zero and missing response remain distinct.

Compatible singleton-bin maps support subtraction. The result retains the
left-hand time window and stored response units:

```python
count_difference = unit.sum(0.1, 0.2) - unit.sum(0.0, 0.1)
rate = unit.to_firing_rate()
rate_difference = rate.mean_rate(0.1, 0.2) - rate.mean_rate(0.0, 0.1)
```

### Spatial sum to 1-D

```python
x_all = raw.sum_to_1d(axis="x")
x_rf = selected.sum_to_1d(axis="x", rf_only=True, detected_rf=rf)
y_rf = selected.sum_to_1d(axis="y", rf_only=True, detected_rf=rf)
```

`axis` names the coordinate retained. For each unit:

| Call | Selection and reduction | Output shape |
| --- | --- | --- |
| `axis="x", rf_only=False` | Sum all rows | `(1, x, time)` |
| `axis="y", rf_only=False` | Sum all columns | `(y, 1, time)` |
| `axis="x", rf_only=True` | Sum whole rows containing any 2-D RF bin | `(1, x, time)` |
| `axis="y", rf_only=True` | Sum whole columns containing any 2-D RF bin | `(y, 1, time)` |

For a 3-by-2 map with RF only in its second row, the x result is that complete
row. Responses outside the mask within that selected row are included. Several
RF-containing rows are added together. This is a numeric response sum;
`rf.project()` below instead projects the binary mask.

Time bins, unit IDs, retained native coordinates, and response units remain
unchanged. Empty RFs produce missing responses and an empty selection. Missing
values in contributing rows or columns propagate. The output records selected
indices and original coordinates. Detection rows are matched by unit ID; new
results also validate source path and grid. Older results lack that provenance,
so the caller must supply the matching source.

`sum_to_1d()` never chooses a time window, detects, normalizes, or saves.
`to_1d_array()` only extracts an already prepared spatial/time singleton:

```python
profile = x_rf.sum(0.0, 0.2).to_1d_array(axis="x")
```

### Notebook 1-D CSVs

`locate_rf.ipynb` and the RF comparison notebooks use `rf_only: bool = True`.
Set it to `False` to sum all rows. Each notebook chooses its CSV filename and
generates the file only when it is missing, then reads that exact file.

Files live directly under the RF session's `data/` directory:

| Responses | Filename example |
| --- | --- |
| All rows | `regular_unitsSpikeCounts_260630_3_ProbeA_1d.csv` |
| Rows containing excitatory RF | `regular_unitsSpikeCounts_260630_3_ProbeA_1d_rfonly.csv` |
| Rows containing inhibitory RF | `regular_unitsSpikeCounts_260630_3_ProbeA_inhibitory_1d_rfonly.csv` |

Each CSV contains only `unit_id` and the TC values, with native positions as
column headers. The notebooks sum native response counts over their selected
time window, then over the chosen rows. RF-only generation uses the unit IDs
and masks in the 2-D result; an empty mask gives a missing TC. The saved 1-D
detection masks remain separate `.npz` files.

`save_rf_tc(prepared_maps, path)` writes already prepared singleton-time,
singleton-spatial maps. `load_rf_tc(path)` only reads the specified CSV into a
DataFrame indexed by unit ID. Neither function chooses filenames or runs
detection. Comparison notebooks use one call to read and wrap the CSV:

```python
from Utils.direction_comparison import load_tc

m14_rf = load_tc(path, kind="RF")
```

This returns the existing TC DataFrame with native degree columns and response
rows. Standard `<mouse>/<date>/<date>_<session>/data/...` paths supply
`(mouse, date, probe, unit_id)` keys. Probe identity comes from a `ProbeA`/`ProbeB`
directory or the generated CSV filename, including `_rfonly` and inhibitory
variants. Explicit identity arguments remain available for nonstandard inputs.
Files outside the recording layout keep `(probe, unit_id)` keys and their source
identity; `concat(first, second)` retains that source identity when pooling them.
`range`, when supplied for RF, only adds display labels. An existing CSV
is reused as-is; changing the time window or detection does not automatically
regenerate it. The loader never chooses `_rfonly` filenames.

`rf_only` also controls row selection in the locate notebook's mean-Hz
previews and figure exports. `collapse_from_2d` independently controls whether
the 1-D detection mask is projected from 2-D or detected separately.

### Constructing and extracting arrays

```python
import numpy as np
from Utils.rflocate import asrfmap

frame = asrfmap(np.zeros((7, 30)), start_time=0.0, end_time=0.2)
```

A 2-D array receives a singleton time axis. A 3-D input uses `(y, x, time)`.
`asrfmap()` validates values, timing, and geometry and preserves missing values.
Stored arrays are read-only; use `np.array(values, copy=True)` for a writable
copy. An array alone does not supply valid trial data for permutation testing.

| Operation | Purpose |
| --- | --- |
| `maps.to_4d_array()` | Stack responses as `(unit, y, x, time)` |
| `prepared.to_2d_array()` | Extract a singleton time bin |
| `prepared.to_1d_array(axis="x")` | Extract a spatial/time singleton without aggregation |
| `unit.where(value)` | Locate exact values as `(y, x, time)` indices |
| `maps.where(value)` | Locate exact values as `(unit, y, x, time)` indices; unit indices are list offsets |

## Detected result object

`load_rf()` and `detect_rf()` return the same `RFResult` type:

```python
masks = rf.mask_2d
centers = rf.center_2d
unit_ids = rf.unit_ids
metadata = rf.manifest

unit_id = int(rf.unit_ids[0])
unit_mask = rf.for_map(raw.by_unit_id(unit_id))
unit_center = rf.for_unit(unit_id, center_only=True)
x_mask = rf.project(axis="x")
y_center = rf.project(axis="y", center_only=True)
```

Masks and centers have canonical `(unit, y, x)` shape, including results for a
single `RFMap`. `for_unit()` selects by recorded ID and returns `(y, x)`;
`for_map()` additionally checks the available source/grid provenance.
`project(axis="x")` returns `(unit, 1, x)` and `project(axis="y")` returns
`(unit, y, 1)`. Projection is deterministic and cannot invoke detection.

An empty mask has an empty center. A nonempty mask has exactly one selected
center bin. `manifest`, `cache_key`, and `schema_version` describe the persisted
analysis and its identity. Existing dictionary-style access remains supported
for compatibility; new code uses attributes.

### Source files and detection files

`.rfmap` identifies source response data. Regular sources may be legacy JSON
or indexed NPZ; `load_rfmap()` checks the contents. Do not rename a binary
`.rfmap` to `.json`. Saved detections use a separate `.npz` contract containing
masks, centers, unit IDs, and a versioned manifest.

```python
from Utils.rflocate import save_rf
save_rf(rf, source.with_name("alternative_detection.npz"))
```

`save_rf()` writes an existing result. `load_rf()` never recomputes stale or
missing results. When detection is explicitly requested with `result_path`,
its persisted result can be reused only if the source/trials, unit set, grid,
response window, and statistical parameters match. A mismatch then causes
that requested detection to run again. Use distinct result paths to retain
alternative analyses; never write a detection over its `.rfmap` source.

## Explicit detection

Prepare the response window before calling the detector:

```python
from Utils.rflocate import detect_rf

response = raw.sum(0.0, 0.2)
rf = detect_rf(response, dimension="2d", cluster_forming_z=1.5)
rf_x = detect_rf(response, dimension="1d", axis="x", cluster_forming_z=1.0)
```

Detection requires exactly one time bin. `dimension="1d"` first sums numeric
responses along the other spatial axis, then detects independently on that
profile. It is different from projecting a saved 2-D detection. Its result
retains a singleton collapsed spatial axis and records that axis in its
manifest; it cannot substitute for the 2-D result in an RF-restricted response
sum.

### Reconstructed trials and permutation detection

Pooled response maps have no trial axis. Explicitly reconstruct matching trials
when requesting a label-permutation test:

```python
from Utils.rflocate.trials import load_regular_rf_trials

session = source.parents[5]
response = raw.sum(0.0, 0.2)
trials = load_regular_rf_trials(session, "A", response, on=True, off=False)
rf = detect_rf(
    response, trials,
    dimension="2d", is_shuffle=True, cluster_forming_z=1.5,
    n_permutations=10_000, random_seed=0, wrap_x=True,
    result_path=rf_result_path(source), show_progress=True,
)
```

The trial loader reads the session MAT file, stimulus onsets, probe spikes,
cluster labels, and good-unit labels. It validates the grid, unit IDs, window,
repeat structure, and pooled raw counts. Use a raw count response for that
validation. Exactly one of `on` and `off` is true; the pooled source must match
that polarity. ON-only, OFF-only, and ON+OFF designs have different valid repeat
block sizes.

The onset file can contain N trial timestamps or N+1 boundaries; only the first
N are used, and an excluded terminal boundary is recorded in provenance.
This reconstruction is for regular one-position-per-trial sparse noise;
pixel-bin, rotation, egocentric, and transformed maps need their own trial
semantics. Do not disable pooled validation to bypass mismatched inputs.

### Detection arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `dimension` | `"2d"` | 2-D detection or independent `"1d"` detection |
| `axis` | `"x"` | Retained coordinate for independent 1-D detection |
| `rf_type` | `"excitatory"` | Increased responses; `"inhibitory"` selects decreased responses |
| `is_shuffle` | `False` | Explicitly enable cluster-permutation testing |
| `drop_bins` | `2` | No-shuffle only: remove components of this size or smaller |
| `exclude_zero_bins` | `False` | No-shuffle only: exclude zero responses from spatial mean, SD, and candidate selection |
| `cluster_forming_z` | 2-D: `1.5`; 1-D: `1.0` excitatory / `0.75` inhibitory | Candidate threshold in spatial SD units |
| `alpha` | `0.05` | Cluster significance cutoff |
| `n_permutations` | `10_000` | Number of shuffled null maps |
| `alternative` | From `rf_type` | Compatibility override: `"greater"` or `"less"` |
| `wrap_x` | `True` | Connect first and last x bins; use only for periodic grids |
| `fill_single_holes` | `False` | Fill eligible one-bin holes in retained clusters |
| `min_hole_neighbors` | `3` | Required 4-connected neighbors for hole filling |
| `random_seed` | `0` | Permutation seed; `None` permits nondeterministic sampling |
| `batch_size` | `64` | Permutation chunk size; excluded from result identity |
| `n_jobs` | `None` | Worker limit; excluded from result identity |
| `result_path` | `None` | Optional validated `.npz` persistence for this explicit detection |
| `show_progress` | `False` | Show detection progress |

No-shuffle detection is exploratory: candidates cross the spatial mean plus
(or minus, for inhibitory RF) the chosen population SD threshold. Connected
components of `drop_bins` bins or fewer are removed. Thus zero keeps every
component, one removes isolated bins, and two removes one- and two-bin
components. A retained mask is not permutation-significant. With
`exclude_zero_bins=True`, zero bins retain their coordinates and cannot enter
the mask; independent 1-D applies this rule after spatial summation. Missing
values remain in the loaded object; the no-shuffle detector represents them
as zero at its analysis stage.

Permutation detection holds each trial response fixed and permutes the joint
`(x, y)` label within repeat strata. Each shuffled map uses the same transform,
threshold, connectivity, and cluster mass as the observed map. The largest
cluster mass per shuffle defines the null. A cluster p-value is
`(1 + count(shuffled maxima >= observed mass)) / (n_permutations + 1)`; clusters with
p-value no greater than `alpha` are retained. This controls spatial family-wise
error within a unit, not across units, probes, polarities, or separate analyses.
`drop_bins` is ignored for this test. The candidate z threshold itself is not
a pixelwise p-value. `exclude_zero_bins` is unsupported with shuffling.

Inhibitory detection identifies decreased spatial responses under the chosen
null; it does not compare against a prestimulus baseline. Center selection uses
nonnegative response-effect weights inside the final mask and chooses the
RF-positive bin minimizing weighted squared distance. Horizontal distance is
circular when `wrap_x=True`. Zero total weight uses equal weights; remaining
ties use higher local weight, then row-major order. The center always lies in
the detected mask.

A long response window can overlap adjacent stimuli (about 100 ms apart in
some recordings). The test does not remove that overlap or attribute each spike
exclusively to the current stimulus. Inspect full time edges and raw timing
before interpreting negative or late activity.

## Complete analyses and plotting

For the established recording workflow, use:

```python
from Utils.rflocate.workflow import analyze_rf_file, save_rf_unit_lists
from Utils.rflocate.plotting import rf_unit_plot_data, plot_rf_unit

analysis = analyze_rf_file(source, probe="A", rf_type="both", save_results=False)
data = rf_unit_plot_data(analysis["excitatory"], 127)
fig, axes = plot_rf_unit(data, title="Probe A · Unit 127")
```

`analyze_rf_file()` explicitly loads counts, converts to Hz, averages the
requested window, applies bin QC, detects, and optionally saves. This combined
behavior belongs to the workflow, not either loader. `max_zero_bins=2` counts
both zeros and missing positions, and at least one nonzero response must remain.
If every unit fails QC, the workflow raises an error. The defaults are mean +
1.8 SD / +1 SD for excitatory 2-D / 1-D and mean −1.5 SD / −0.75 SD for
inhibitory 2-D / 1-D. `collapse_from_2d=True` projects the 2-D result already
computed; otherwise 1-D detection is independent.

`locate_rf.ipynb` runs both RF types, with `rf_type` selecting plots.
`locate_rf.py` defaults to excitatory and accepts `--rf-type inhibitory` or
`--rf-type both`. Plot functions consume prepared data. Population smoothing
is an explicit notebook step. Figures and exports use opaque white backgrounds
and dark labels.

For a source `example.rfmap`, the workflow can save:

| File | Contents |
| --- | --- |
| `example.npz`, `example_1d.npz` | 2-D and 1-D masks/centers with retained unit IDs |
| `example_units_with_rf.npy`, `example_units_with_rf_1d.npy` | `(probe, unit_id)` rows |
| `example_analysis.json` | Parameters, QC counts, and selected units |

Inhibitory outputs append `_inhibitory` to the source stem. Unit lists are
pickle-free Unicode arrays with `(N, 2)` shape, including `(0, 2)` for no RFs.
The notebook also writes combined unit lists for both types. Set
`save_results=False` to avoid workflow output files.

The MATLAB bridge remains:

```text
RFmapping_core.m -> RFmapping_run_python.m -> locate_rf.py
    -> Utils.rflocate.workflow.analyze_rf_file()
```

`runRfDetection=false` in MATLAB generates sources without detection. With it
true, executable, script path, window, thresholds, QC, wrapping, and 1-D mode
come from the MATLAB parameters. The detector window must align with source
time edges. Python runs synchronously; failures raise
`RFmapping:PythonDetectionFailed`. See the repository README for acquisition
and MATLAB setup.

## Compatibility and migration

Existing imports remain available. New code has one recommended location:

| Existing name | Recommended API |
| --- | --- |
| `Utils.rfmap.RFMap`, `RFMapList`, `asrfmap` | `Utils.rflocate` |
| `Utils.rfmap.load_rf_maps(path)` | `Utils.rflocate.load_rfmap(path)` |
| `Utils.rf_cache.read_rf_result(path)` | `Utils.rflocate.load_rf(path)` |
| `summed.rf_2d(..., is_center=False)` | `detect_rf(summed, ...).mask_2d` |
| `summed.rf_2d(..., is_center=True)` | `detect_rf(summed, ...).center_2d` |
| `summed.rf_1d(..., collapse_from_2d=False)` | `detect_rf(summed, dimension="1d", axis=...)` |
| `summed.rf_1d(collapse_from_2d=True, detected_rf=rf)` | `rf.project(axis=...)` |
| `Utils.rf_analysis` | `Utils.rflocate.workflow` |
| `Utils.rf_plotting` | `Utils.rflocate.plotting` |
| `Utils.rf_trials`, `Utils.rf_rates` | `Utils.rflocate.trials`, `Utils.rflocate.rates` |

Legacy mask-returning methods keep their original array shapes. `RFResult`
always keeps the leading unit axis, and its projected masks keep the collapsed
spatial axis. The old loader's explicit `unit_firing_rate=True` option uses saved
presentation counts only; use `to_firing_rate()` in new code. Callable maps
remain shorthand for `sum()`, but explicit method calls make units clearer.

Comparison profiles belong to `Utils.direction_comparison`, separately from
saved RF detection. Its `load_rf()` reads already prepared response profiles;
`Utils.rflocate.load_rf()` reads saved detection results. In new analysis code,
make profile preparation explicit:

```python
from Utils.direction_comparison import (
    rf_profiles, select_rf_profiles, resample_profiles, recording_profiles,
)

prepared = raw.sum(0.0, 0.2).sum_to_1d(axis="x")
profiles = rf_profiles(prepared, probe="A")
profiles = select_rf_profiles(profiles, rf)
profiles = resample_profiles(profiles, range=[-180, 180], bins=30)
profiles = recording_profiles(profiles, mouse="m15", date=260630)
```

Omit selection or resampling when unwanted. For Hz profiles, replace the time
sum with `raw.to_firing_rate().mean_rate(...)`. Keep native values for scientific
statistics; display interpolation does not replace native peak coordinates.
