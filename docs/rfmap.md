# RF Python API

Use `RF.rflocate` for RF response data, saved detections, and explicit
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
from RF.rflocate import RFMapList, load_rfmap, load_rf, rf_result_path

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
from RF.rflocate import (
    RFMap, RFMapList, RFResult, asrfmap,
    load_rfmap, load_rf, save_rf, rf_result_path, detect_rf,
    load_rf_tc, rf_tc_table, save_rf_tc, fit_rf_gaussians,
)
```

| Module | Contents |
| --- | --- |
| `RF.rflocate.models` | Response objects and numeric transformations |
| `RF.rflocate.results` | Detected result object and deterministic mask views |
| `RF.rflocate.io` | Source and detection file I/O |
| `RF.rflocate.detection` | Detection from prepared 1-D or 2-D responses |
| `RF.rflocate.fitting` | Continuous signed Gaussian fits sized by supplied detection regions |
| `RF.rflocate.rates` | Presentation counts and response-rate conversion |
| `RF.rflocate.trials` | Reconstruction of aligned regular-stimulus trials |
| `RF.rflocate.plotting` | Plot data preparation, plots, and figure export |
| `RF.rflocate.workflow` | Complete file analysis, QC, and result export |
| `RF.profiles` | RF curve loading, explicit selection, and comparison preparation |
| `RF.tc_generation` | Pure TC calculations and explicit source-to-CSV generation |

Modules beginning with `_` are implementation details. The root API does not
import plotting or the complete recording workflow. Existing root notebooks
and the MATLAB bridge `locate_rf.py` retain their locations.

## Continuous Gaussian fits from existing detections

```python
from RF.rflocate import fit_rf_gaussians

response = raw.by_unit_id(295).to_firing_rate(
    reconstruct_presentations=True,
).mean_rate(0.0, 0.2)
fit = fit_rf_gaussians(
    response, excitatory=load_rf(ext_path), inhibitory=load_rf(inh_path),
    wrap_x=True,
)
continuous_hz = fit["fitted_response_hz"]
```

Each connected region in the supplied excitatory mask contributes one positive
Gaussian; each inhibitory region contributes one negative Gaussian. Connectivity
is four-neighbor with the explicitly requested horizontal wrap. Saved center
markers do not set the count. A polarity with no regions adds no components.

The model is `offset + sum(positive Gaussians) - sum(negative Gaussians)`.
Axis-aligned Gaussian amplitudes, centers, horizontal/vertical widths, and a
shared offset are jointly fitted to the original finite bin responses in Hz.
The horizontal kernels are periodic when `wrap_x=True`. No smoothing or mask
values are used as the fitting target. The region masks initialize components;
centers can move during fitting. Width bounds and any parameters reaching a
bound are returned for inspection. Fits are descriptive, not significance tests.

Every continuous output retains the native `(y, x)` grid and missing bins.
`net_excitatory_hz` and `net_inhibitory_hz` are the nonnegative positive and
negative parts of the fitted response relative to its fitted offset. Individual
component magnitudes are also returned in `component_maps_hz`. No final response
threshold is applied; choose a positive threshold explicitly when creating a
binary mask, because Gaussian tails generally remain nonzero. Loading, fitting,
and thresholding do not save or replace detections.

For an isolated export, run `python -m scripts.fit_detected_rf --help` remotely.
The script requires explicit source, detection files, units, response window,
horizontal wrapping, and a new output directory. It writes continuous NPZs,
JSON provenance, and comparison PNGs without invoking the detector.

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
values in contributing rows or columns propagate by default. Set
`missing_as_zero=True` to replace missing values with zero before spatial
summation, as the established independent 1-D detection does. The output records
selected indices and original coordinates. Detection rows are matched by unit ID; new
results also validate source path and grid. Older results lack that provenance,
so the caller must supply the matching source.

`sum_to_1d()` never chooses a time window, detects, normalizes, or saves.
`to_1d_array()` only extracts an already prepared spatial/time singleton:

```python
profile = x_rf.sum(0.0, 0.2).to_1d_array(axis="x")
```

### Notebook 1-D CSVs

The RF comparison notebooks use `rf_only: bool = True`. Set it to `False` to
use all rows. Generate a missing CSV through an explicit generation step;
readers never create it automatically. `locate_rf.ipynb` prepares its current response curves explicitly and
writes them in its save step. The two
`tc_comparison_pairs` notebooks read only prepared HD/RF CSVs with `load_tc`.
`scripts/export_comparison_tcs.py` writes one explicitly requested comparison
CSV from an HD `.tc` or an already prepared native RF CSV. The paired notebooks
use these outputs under `data/tc_comparison/`; see
[comparison preparation](hd_rf_ebc_comparison.md).

Files live directly under the RF session's `data/` directory:

| Responses | Filename example |
| --- | --- |
| All rows | `regular_unitsSpikeCounts_260630_3_ProbeA_1d.csv` |
| Rows containing excitatory RF | `regular_unitsSpikeCounts_260630_3_ProbeA_1d_rfonly.csv` |
| Rows containing inhibitory RF | `regular_unitsSpikeCounts_260630_3_ProbeA_inhibitory_1d_rfonly.csv` |

Each CSV contains only `unit_id` and the TC values, with native positions as
column headers. Preparation sums native response counts over the selected
time window, then over the chosen rows. RF-only generation uses the unit IDs
and masks in the 2-D result; an empty mask gives a missing TC. The saved 1-D
detection masks remain separate `.npz` files.

`rf_tc_table(prepared_maps, axis="x")` converts already prepared singleton-time,
singleton-spatial maps to a DataFrame without file I/O. `save_rf_tc(table, path)`
only writes that table; it does not accept maps or an `axis` argument.
`load_rf_tc(path)` only reads the specified CSV into a DataFrame indexed by unit
ID. None of these functions chooses a window or runs detection:

```python
table = rf_tc_table(raw.sum(0.0, 0.2).sum_to_1d(axis="x"))
save_rf_tc(table, native_csv)
loaded_table = load_rf_tc(native_csv)
```

Export the chosen projection with an explicit unit prefix before
reading it in the paired comparison notebook:

```python
from Utils.direction_comparison import tc_loader

m14_rf = tc_loader(comparison_csv, label="m14 RF", response_units="spike_count")
```

This reads native degree columns, response values, and the exact `unit_id`
strings in the file, such as `m14:260609:A:23`. The loader does not infer identity
or numerical units from paths. `range` only adds display labels. Select measured
zero-bin limits with `rf_pick` on the native table; use `resample_profiles`
explicitly if a common grid is needed before `concat`.

The exporter preserves the supplied RF CSV's projection and values. It can
select units using explicitly named saved detection files, but does not generate
native projections or choose `_rfonly` files. Existing comparison outputs require
`--overwrite` to replace them.

The locate notebook uses the same explicitly prepared 1-D response for its
independent detection and mean-Hz preview. Project an existing 2-D mask with
`rf.project(axis="x")` when that is the intended analysis.

### Constructing and extracting arrays

```python
import numpy as np
from RF.rflocate import asrfmap

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

An empty mask has no centers. New detections select exactly one center bin per
connected RF sector, using the detector's four-connectivity and `wrap_x` option.
Projection preserves these selected bins; centers that land in the same
projected bin merge. `manifest`, `cache_key`, and `schema_version` describe the
persisted analysis and its identity. Existing dictionary-style access remains
supported for compatibility; new code uses attributes.

The center algorithm is `component-response-weighted-medoid-v2`. Its result
identity differs from the previous one-center-per-unit algorithm. Loading a
saved result preserves its original centers and never silently migrates a
historical analysis. Explicitly compute and save a new result to replace it.

### Source files and detection files

`.rfmap` identifies source response data. Regular sources may be legacy JSON
or indexed NPZ; `load_rfmap()` checks the contents. Do not rename a binary
`.rfmap` to `.json`. Saved detections use a separate `.npz` contract containing
masks, centers, unit IDs, and a versioned manifest.

```python
from RF.rflocate import save_rf
save_rf(rf, source.with_name("alternative_detection.npz"))
```

`save_rf()` writes an existing result. `load_rf()` never recomputes stale or
missing results. `detect_rf()` accepts loaded inputs, calculates a new result,
and performs no file I/O; it has no `result_path` argument. Use distinct paths in
explicit `save_rf(result, path)` calls to retain
alternative analyses; never write a detection over its `.rfmap` source.

## Explicit detection

Prepare the response window before calling the detector. `drop_bins` is a
required keyword with no default. These no-shuffle examples explicitly choose
`2` to remove components containing one or two bins; use `0` to keep every
candidate component. The reference selection never supplies this value.

```python
from RF.rflocate import detect_rf

response = raw.sum(0.0, 0.2)
rf = detect_rf(response, sigma=1.8, drop_bins=2)
rf_inhibitory = detect_rf(response, sigma=-1.0, drop_bins=2)

response_x = response.sum_to_1d(axis="x", missing_as_zero=True)
rf_x = detect_rf(response_x, sigma=1.0, drop_bins=2)
rf_x_inhibitory = detect_rf(response_x, sigma=-0.75, drop_bins=2)
```

`sigma` specifies the threshold and direction: positive values select responses
at or above `mean + sigma * SD`; negative values select responses at or below
that threshold. Thus `sigma=1.8` detects increased responses and `sigma=-1`
detects decreased responses. No separate polarity or dimension argument is
needed.

Detection accepts a numeric 1-D vector, a numeric 2-D map, an `RFMap`, or an
`RFMapList`. Map objects require exactly one time bin. The input's spatial shape
determines whether detection is 1-D or 2-D; a singleton spatial axis is retained
in the result. Prepare a 1-D response with `sum_to_1d()` before detection.
Detection never sums a 2-D input implicitly.

```python
rf_vector = detect_rf(np.array([0., 0., 4., 5., 4., 0.]), sigma=1.0, drop_bins=2)
rf_array = detect_rf(response[0].to_2d_array(), sigma=1.8, drop_bins=2)
```

Numeric inputs describe a single unit on an index-based grid. Use response
objects to retain recorded unit IDs, native coordinates, and source provenance.
Independent 1-D detection differs from projecting a saved 2-D detection.
Results prepared with `sum_to_1d()` record the aggregation axis and provenance;
they cannot substitute for a 2-D result in an RF-restricted response sum. Native
singleton grids retain their original grid identity.

### Bottom-row mean with global SD

Use `reference="bottom_row"` to replace the global mean with the mean of the
last spatial array row, while retaining the SD of the whole 2-D response map:

```python
counts = load_rfmap(source)
response = counts.to_firing_rate(reconstruct_presentations=True).mean_rate(0.0, 0.2)
rf = detect_rf(response, reference="bottom_row", sigma=1.8, drop_bins=2)
rf_inhibitory = detect_rf(response, reference="bottom_row", sigma=-1.0, drop_bins=2)
```

For each unit, the threshold is `mean(response[-1, :]) + sigma * SD(response)`.
The population SD (`ddof=0`) is identical to the global reference's SD. The
baseline row is selected by array order (`-1`), not by the minimum Y coordinate.
Each unit has one threshold shared across all its columns. With the default
`exclude_zero_bins=False`, both references replace missing values with zero
and include zero responses in the mean and SD. Positive sigma selects values
at or above the threshold; negative sigma
selects values at or below it. Constant maps produce no detections.

The bottom-row reference requires an uncollapsed 2-D map with at least two
spatial rows but no repeat responses. It cannot be combined with shuffling or
`exclude_zero_bins=True`. It reuses the existing connected-component and center
rules. It is a descriptive threshold, not a permutation significance test.
`reference="global"` retains the existing spatial mean/SD threshold and remains
the default.

### Reconstructed trials and permutation detection

Pooled response maps have no trial axis. Explicitly reconstruct matching trials
when requesting a label-permutation test:

```python
from RF.rflocate.trials import load_regular_rf_trials

session = source.parents[5]
response = raw.sum(0.0, 0.2)
trials = load_regular_rf_trials(session, "A", response, on=True, off=False)
rf = detect_rf(
    response, trials,
    sigma=1.5, is_shuffle=True, drop_bins=0,  # Required; ignored by permutation detection.
    n_permutations=10_000, random_seed=0, wrap_x=True,
    show_progress=True,
)
save_rf(rf, rf_result_path(source))
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
| `reference` | `"global"` | Whole-map mean or `"bottom_row"` last-row mean; both use whole-map population SD |
| `sigma` | `1.5` | Signed threshold in baseline SD units; positive selects increased responses, negative selects decreased responses |
| `is_shuffle` | `False` | Explicitly enable cluster-permutation testing |
| `drop_bins` | Required; no default | No-shuffle only: remove components of this size or smaller; `0` keeps every candidate component |
| `exclude_zero_bins` | `False` | No-shuffle only: exclude zero responses from spatial mean, SD, and candidate selection |
| `alpha` | `0.05` | Cluster significance cutoff |
| `n_permutations` | `10_000` | Number of shuffled null maps |
| `wrap_x` | `True` | Connect first and last x bins; use only for periodic grids |
| `fill_single_holes` | `False` | Fill eligible one-bin holes in retained clusters |
| `min_hole_neighbors` | `3` | Required 4-connected neighbors for hole filling |
| `random_seed` | `0` | Permutation seed; `None` permits nondeterministic sampling |
| `batch_size` | `64` | Permutation chunk size; excluded from result identity |
| `n_jobs` | `None` | Worker limit; excluded from result identity |
| `show_progress` | `False` | Show detection progress |

Global no-shuffle detection is exploratory: candidates cross the spatial mean plus
the signed `sigma` times the population SD. Connected
components of `drop_bins` bins or fewer are removed. Thus zero keeps every
component, one removes isolated bins, and two removes one- and two-bin
components. A retained mask is not permutation-significant. With
`exclude_zero_bins=True`, zero bins retain their coordinates and cannot enter
the mask; 1-D detection applies this rule to the supplied profile. Missing
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
nonnegative response-effect weights separately inside each connected sector
of the final mask and chooses the sector's RF-positive bin minimizing weighted
squared distance to bins in that sector. Horizontal distance is
circular when `wrap_x=True`. Zero total weight uses equal weights; remaining
ties use higher local weight, then row-major order. The center always lies in
its own detected sector. Thus a unit with three sectors has three centers;
an empty unit has none.

A long response window can overlap adjacent stimuli (about 100 ms apart in
some recordings). The test does not remove that overlap or attribute each spike
exclusively to the current stimulus. Inspect full time edges and raw timing
before interpreting negative or late activity.

## Complete analyses and plotting

For the established recording workflow, use:

```python
from RF.rflocate.workflow import analyze_rf_maps, analyze_rf_file, save_rf_analysis
from RF.rflocate.plotting import rf_unit_plot_data, plot_rf_unit

analysis = analyze_rf_file(
    source, probe="A", rf_type="both", reference_2d="global", drop_bins=2, save_results=False,
)
data = rf_unit_plot_data(analysis["excitatory"], 127)
fig, axes = plot_rf_unit(data, title="Probe A · Unit 127")
```

For already loaded maps, the pure calculation is separate:

```python
maps = load_rfmap(source)
# Only legacy sources without stored exposure require this explicit file-reading step.
rates = maps.to_firing_rate(reconstruct_presentations=True)
analysis = analyze_rf_maps(rates, probe="A", rf_type="both", reference_2d="global", drop_bins=2)
for result in analysis.values():
    save_rf_analysis(result)
```

`analyze_rf_maps()` performs no file loading, exposure reconstruction, or saving.
It accepts normalized Hz maps or count maps carrying presentation counts.
Both analysis functions require explicit `reference_2d` and `drop_bins` keywords;
neither has a default. These examples choose `"global"` for whole-map mean/SD
and `2` to remove one- and two-bin components. Direct `detect_rf()` calls retain
their `reference="global"` default, including independent 1-D detection.
`analyze_rf_file()` is the explicit orchestration boundary: it loads counts,
resolves legacy exposure when needed, converts to Hz, averages the
requested window, applies bin QC, detects, and optionally saves. This combined
behavior belongs to the workflow, not either loader. `max_zero_bins=2` counts
both zeros and missing positions, and at least one nonzero response must remain.
If every unit fails QC, the workflow raises an error. The defaults are mean +
1.8 SD / +1 SD for excitatory 2-D / 1-D and mean −1 SD / −0.75 SD for
inhibitory 2-D / 1-D. `collapse_from_2d=True` projects the 2-D result already
computed; otherwise 1-D detection is independent. Pass `rf_sigmas={"excitatory":
(1.8, 1.0), "inhibitory": (-1.0, -0.75)}` to configure the same signed pairs as
the notebook.

Choose `reference_2d="bottom_row"` in `analyze_rf_maps()` or `analyze_rf_file()` to
use the last-row mean with the whole-map SD for each unit. No raw trial loading
is needed for this reference. Legacy count files still need their matching
session when presentation counts must be reconstructed during Hz conversion.
Independent 1-D detection keeps the global reference; `collapse_from_2d=True`
projects the selected 2-D result.

`locate_rf.ipynb` shows loading, Hz conversion, QC, 2-D detection, explicit
spatial summation, 1-D detection, previews, and saving as separate steps. Its
`rf_sigmas` setting provides signed thresholds for both 2-D and 1-D detection.
Its visible `reference_2d` setting selects `"global"` or `"bottom_row"` from the
same prepared 2-D response maps.
`locate_rf.py` defaults to both polarities and accepts `--rf-type excitatory` or
`--rf-type inhibitory` to select one. Both this CLI and
`scripts.redetect_rf_sessions` require `--reference-2d` and `--drop-bins`, with
no defaults. Choose `--reference-2d global` for whole-map mean/SD or
`--reference-2d bottom_row` for last-row mean with whole-map SD. For example,
`--reference-2d global --drop-bins 2` removes one- and two-bin components using
the global reference. The baseline choice does
not set this cutoff. Plot functions consume prepared data. Population smoothing
is an explicit notebook step. Figures and exports use opaque white backgrounds
and dark labels.

`--comparison-output-dir` with `--unit-prefix` explicitly generates the native
projection and comparison CSVs after detection. Add `--overwrite-comparison`
when replacing existing CSVs; ordinary comparison readers never regenerate
missing inputs.

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
    -> RF.rflocate.workflow.analyze_rf_file()
```

The helper is bundled in `matlab/Utils/`. Use the configured terminal paths:

```matlab
params.runRfDetection = true;
params.rfPythonExecutable = getenv('RFMAP_PYTHON');
params.rfPythonScript = fullfile(getenv('RFMAP_CODE_DIR'), 'locate_rf.py');
params.rfReference2d = 'global'; % Explicit choice; 'bottom_row' is also supported.
params.dropBins = 2; % Explicit choice: remove one- and two-bin components.
```

`runRfDetection=false` in MATLAB generates sources without detection. With it
true, executable, script path, window, thresholds, 2-D reference, component-size
cutoff, QC, wrapping, and 1-D mode
come from the MATLAB parameters. The detector window must align with source
time edges. Python runs synchronously; failures raise
`RFmapping:PythonDetectionFailed`. The caller uses POSIX shell quoting on
Linux/macOS; native Windows MATLAB callers should set `runRfDetection=false`
and invoke `locate_rf.py` separately. See the repository README for acquisition
and MATLAB setup.

## Compatibility and migration

RF helpers now live under `RF/`; the old `Utils.rf*` paths are no longer provided.
Compatibility module filenames remain available inside `RF/`. New code has one
recommended location:

| Existing name | Recommended API |
| --- | --- |
| `RF.rfmap.RFMap`, `RFMapList`, `asrfmap` | `RF.rflocate` |
| `RF.rfmap.load_rf_maps(path)` | `RF.rflocate.load_rfmap(path)` |
| `RF.rf_cache.read_rf_result(path)` | `RF.rflocate.load_rf(path)` |
| `summed.rf_2d(..., is_center=False)` | `detect_rf(summed, sigma=..., drop_bins=...).mask_2d` |
| `summed.rf_2d(..., is_center=True)` | `detect_rf(summed, sigma=..., drop_bins=...).center_2d` |
| `summed.rf_1d(..., collapse_from_2d=False)` | `detect_rf(summed.sum_to_1d(axis=..., missing_as_zero=True), sigma=..., drop_bins=...)` |
| `summed.rf_1d(collapse_from_2d=True, detected_rf=rf)` | `rf.project(axis=...)` |
| `RF.rf_analysis` | `RF.rflocate.workflow` |
| `RF.rf_plotting` | `RF.rflocate.plotting` |
| `RF.rf_trials`, `RF.rf_rates` | `RF.rflocate.trials`, `RF.rflocate.rates` |

Legacy mask-returning methods keep their original arguments and array shapes.
The CLI and MATLAB bridge also retain their existing options. For direct
`detect_rf()` calls, replace `cluster_forming_z` and `rf_type` with signed
`sigma`, supply `drop_bins` explicitly, and prepare 1-D inputs instead of passing `dimension` or
`axis`. `RFResult`
always keeps the leading unit axis, and its projected masks keep the collapsed
spatial axis. The old loader's `unit_firing_rate` argument has been removed;
replace it with `load_rfmap(path).to_firing_rate()`. Reconstruct missing legacy
exposure only through an explicit `to_firing_rate(reconstruct_presentations=True)`
step. Callable maps
remain shorthand for `sum()`, but explicit method calls make units clearer.

Legacy `RFMap.rf_1d()` and `rf_2d()` retain their explicit `result_path` cache
wrapper. That wrapper may read and save a matching detection; canonical
`detect_rf()` does not use it. New code calls `detect_rf()`, `save_rf()`, and
`load_rf()` separately.

RF comparison profiles belong to `RF.profiles`, separately from
saved RF detection. Its `load_rf()` reads already prepared response profiles;
`RF.rflocate.load_rf()` reads saved detection results. In new analysis code,
make profile preparation explicit:

```python
from RF.profiles import rf_profiles, select_rf_profiles
from Utils.direction_comparison import resample_profiles, recording_profiles

prepared = raw.sum(0.0, 0.2).sum_to_1d(axis="x")
profiles = rf_profiles(prepared, probe="A")
profiles = select_rf_profiles(profiles, rf)
profiles = resample_profiles(profiles, range=[-180, 180], bins=30)
profiles = recording_profiles(profiles, mouse="m15", date=260630)
```

Omit selection or resampling when unwanted. For Hz profiles, replace the time
sum with `raw.to_firing_rate().mean_rate(...)`. Keep native values for scientific
statistics; display interpolation does not replace native peak coordinates.
