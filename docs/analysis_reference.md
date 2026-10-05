# Analysis reference

This reference covers analysis after RF source generation: the Python RFMap
API, related notebooks, saved timing inputs, and optional dependencies. For the
raw-recording-to-`.rfmap` workflow, start with the [main guide](../README.md).
The Python/Tk, SwiftUI, and Web viewers now live in the sibling
`../../rfmapping_gui` repository and are not runtime dependencies of this package.

GLM notebooks, scripts, reports, and their tests live in the sibling
[`../../glm`](../../glm/README.md) directory. Shared timing, loading, and geometry
helpers remain here and are copied into that directory when needed.

## RF and tuning comparisons

The root RF and tuning-comparison notebooks use the public RF API and explicit
loading, selection, statistics, and plotting stages. See
[TC loading and comparisons](hd_rf_ebc_comparison.md).

Independent EBC/spatial, decoder, and video experiments remain local files
covered by `.gitignore`; a fresh clone does not include those workflows.

## Python RFMap API

`Utils.rflocate.load_rfmap()` loads one pooled RF source into an ordered `RFMapList`
containing one `RFMap` per recorded unit. Array-index lookup and recorded
unit-ID lookup are deliberately separate.

```python
from pathlib import Path

import numpy as np

from Utils.rflocate import asrfmap, load_rfmap, detect_rf
from Utils.rflocate.trials import load_regular_rf_trials

session = Path("/mnt/senzailab/Kai/#Recording/m15/260630/260630_3")
rf_source = (
    session
    / "data/rfmapping/good/-100_400_1ms/ProbeA"
    / "regular_unitsSpikeCounts_260630_3.json"
)

raw = load_rfmap(rf_source)
summed = raw.sum(0.0, 0.2, show_progress=True)
trials = load_regular_rf_trials(
    session,
    "A",
    summed,
    on=True,
    off=False,
)

result_path = rf_source.with_suffix(".npz")
rf = detect_rf(
    summed, trials,
    is_shuffle=True,
    cluster_forming_z=1.5,
    alpha=0.05,
    n_permutations=10_000,
    random_seed=0,
    wrap_x=True,
    result_path=result_path,
    show_progress=True,
)
rf_masks_2d = rf.mask_2d
rf_masks_x = rf.project(axis="x")
rf_centers_2d = rf.center_2d

unit_by_source_index = summed.by_index(5)
the_same_unit = summed.by_unit_id(unit_by_source_index.unit_id)

units_with_any_zero_bin = np.unique(summed.where(0)[0])
unit_ids_with_any_zero_bin = np.asarray(summed.unit_ids)[units_with_any_zero_bin]

array_map = asrfmap(np.zeros((7, 30)), start_time=0.0, end_time=0.2)
```

`load_rfmap()` preserves the source values and units, including raw counts
needed by `load_regular_rf_trials()` validation. `sum(start, stop)` always adds
stored values. To obtain Hz, explicitly call `raw.to_firing_rate()`, which uses
each position's presentation count and each time bin's width in seconds; then
use `mean_rate(start, stop)` for a time-weighted mean. New sources save
`stimulusPresentationCounts`. Historical sources can explicitly reconstruct it
with `to_firing_rate(reconstruct_presentations=True)`; loading does not read
session trials or onset boundaries. `occupancyTimeSec` remains display-time
metadata. Loaded null/NaN values stay missing, and missing contributing bins
propagate through numeric sums. No-shuffle detection applies its own missing-bin
policy only when detection is requested.

`detect_rf()` returns an `RFResult` and defaults to
`dimension="2d", is_shuffle=False, drop_bins=2`. Pass `is_shuffle=True`
explicitly for trial-label permutation, as in the example above.

`sum(earlier, later)` uses seconds and the half-open interval
`[earlier, later)`. Both values must resolve to actual `timeBinEdges` entries
within `1e-12` seconds. Equal edges produce a valid zero-valued singleton time
axis; reversed intervals are invalid.

`detect_rf()` operates on a single-bin prepared object. Shuffle detection
also requires matching trial data. A pooled source does not contain a trial
axis and is insufficient for label permutation. The regular-data loader
reconstructs per-trial responses from the authoritative MAT, onset, spike-time,
cluster, and good-unit files. Set exactly one of `on` and `off`: ON selects
`Square_Luminance == 1`, while OFF selects `Square_Luminance == 0`.
Permutations keep responses fixed and shuffle the joint `(x, y)` label within
verified exchangeability blocks after polarity filtering. Repeat blocks are
validated against the luminances actually present, including ON-only and
OFF-only sessions.

Candidate pixels use the configured cluster-forming z threshold. With shuffle
enabled, significance comes from the null distribution of the maximum
4-connected cluster mass. `rf.project(axis="x")` projects the final 2-D mask
without another test. Independent 1-D detection is an explicit
`detect_rf(summed, dimension="1d", axis="x", ...)` call. Correction is within a unit and does not correct across
units, polarities, or separately run analyses.

Batch work can show `Sum`, `Detecting RF`, and `Center` progress bars.
The result exposes its complete mask as `mask_2d` and one response-weighted RF
bin per nonempty mask as `center_2d`. Both have `(unit, y, x)` axes; projections
retain the unit and collapsed spatial axes. Detection is silent unless
`show_progress=True`.

`result_path` is an optional, versioned `.npz` result sidecar. One
successful run stores both the mask and center, together with enough input and
parameter identity to reject a stale result. `Utils.rflocate.load_rf()` reads
that saved result without running detection; accessing its center or projecting
its mask does not rerun the permutation. Do not use `.rfmap` for this:
`.rfmap` remains a raw-source extension. The current regular writer produces
indexed NPZ, while the current `RFmapping_fm.m` workflow produces JSON text
under the same extension. `load_rfmap()` reads both of these storage formats.
JSON sources may end in either `.json` or `.rfmap`. A separate HDF5 workflow
also exists; it uses a different contract and is not the output of the current
`RFmapping_fm.m` workflow described in the main guide.

With `is_shuffle=False`, `drop_bins=1` removes every 4-connected candidate
component containing one bin; in general, components with size less than or
equal to `drop_bins` are removed. `is_shuffle=True` retains the existing
permutation behavior and ignores `drop_bins`.

See [rfmap.md](rfmap.md) for the complete data contract, array shapes,
permutation semantics, and troubleshooting guide.

## Saved timing inputs for tuning comparisons

Tuning comparisons read `data/probeA/adc_spike_time.npy` (or `probeB`)
in Kilosort spike order. These files contain absolute timestamps in seconds;
the analysis subtracts the ADC origin once.

`get_exposure_timestamps(session_info, data_dir)` first reads existing
`data/camera_frame_times.npy` or
`sync_data.json["exposure_sampling_number_list_mid"]`. The NPY contains absolute
seconds; the JSON midpoint field already contains ADC-relative seconds,
despite its `sampling_number` name. If neither is available, the helper reads
complete exposure pulses from the configured raw ADC channel in chunks.
Basler opto-coupled output uses low pulses in this setup; OptiHub2 uses high
pulses. Callers select the camera polarity explicitly.
The source and raw-signal settings are recorded in `ttl_qc`.

For JSON with `exposure_sampling_number_list_mid_raw`, the ADC origin is the
difference between the first raw and relative midpoints. Otherwise only the
first ADC timestamp supplies the origin. Saved camera times retain their
upstream frame alignment and timing definition. Spike times must already be
saved; this path does not reconstruct spikes from sample indices.

Head-direction callers use `make_head_direction_tsd(exposure_timestamps,
frame_indices, angles_deg)` and pass its returned `feature_fs_hz` to
`tuning_curve()`. The helper uses the full camera sequence to determine the
sampling rate and valid pose intervals, excluding missing or nonfinite poses
and camera pauses. Observed spike counts and circular shuffles use those same
intervals. Occupancy retains the saved format's nominal frame-count/rate
definition, so tracking gaps no longer increase the duration assigned to each
observed frame. Previously saved tuning files need to be regenerated to use
this behavior.

## Install and validate

Python 3.12 or newer is required. Project code is run only on `hhw9l84` with
the existing remote virtualenv:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/pip install -e ".[test]"'

ssh hhw9l84 'cd ~/Developer/rfmapping && \
  PYTHONDONTWRITEBYTECODE=1 ~/.virtualenvs/rfmapping/bin/python -m pytest -q \
  tests/test_rf_package.py tests/test_rf_result_io.py \
  tests/test_rf_result_object.py'
```

The optional `analysis` dependency group covers plotting/tuning helpers:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/pip install -e ".[analysis,test]"'
```

The optional `waveform` group covers SpikeInterface extraction and probe plots:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/pip install -e ".[waveform,test]"'
```

## MATLAB pipeline

See the [main guide](../README.md) for the current MATLAB entry points,
authoritative remote source locations, required raw inputs, and `.rfmap`
generation steps.
