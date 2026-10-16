# Spatial-cell analysis and plotting

Run `spatial_cell_analysis.ipynb` to generate the four RFMaps and saved spatial
plotting arrays. Run `spatial_cell_plotting.ipynb` to export every saved good unit
as a ten-panel PNG and SVG, preview one unit, and display the population heatmap.
Use the remote `~/.virtualenvs/rfmapping` kernel on `hhw9l84`.

## Analysis inputs and geometry

Set the recording, probe, phase, arena bounds, angular/distance bins and
smoothing in the analysis notebook. `unit_ids = None` selects every Kilosort
`good` unit, including units with no spikes in this recording; a list selects a
subset.

The shared analysis loader reads pose columns `frame`, `center_x`, `center_y`,
`hd_deg`. Basler sessions containing a `.tak` file use `<date>_yolo.csv`, preserving
the Motive `<date>.csv`; other sessions use `<date>.csv`. A missing selected CSV
is an error, not permission to substitute another tracking source. Other inputs
are the interval table, Kilosort labels/clusters and
`data/probe<probe>/adc_spike_time.npy`. Raw Motive exports must first be processed
into the pose columns.

Select exactly one camera output:

```python
basler_output = True
optihub2_output = False
```

Basler opto-coupled `ExposureActive` uses complete low pulses; OptiHub2 uses high
pulses. The Basler preset assumes no additional `LineInverter` inversion. See
[Basler output levels](https://docs.baslerweb.com/line-status#opto-coupled-output-line)
and [Line Inverter](https://docs.baslerweb.com/line-inverter).
Camera type and polarity are explicit. Saved camera times are read when
available; otherwise the configured ADC channel is scanned in chunks for
exposure midpoints, excluding ADC-boundary idle intervals. Channels are
zero-based and thresholds use raw int16 ADC units. The CLI defaults to channel 6
for Basler, except configured legacy channel-1 sessions; OptiHub2 uses channel 1.
An explicit channel overrides this selection. Only OptiHub2 permits one trailing
Motive frame without a TTL.

The CLI boundary order is explicit `--bounds-px LEFT RIGHT TOP BOTTOM`, then a
session-specific configuration, then the selected `--boundary old|new` preset
(default `new`). The current analysis notebook detects the Basler arena from
video and passes its result as explicit bounds. Saved `audit.json` records the
actual bounds used; plotting reads those saved results and does not detect or
change the arena.

Rectangle geometry uses a 40 × 40 cm arena, scaling each pixel axis by its own
calibrated span. Heading direction vectors use those same X/Y scales before
converting back to an angle; the source pixel-plane HD is also retained. A
finite point outside the inclusive boundary holds the last
complete in-bound point until re-entry. This happens before phase selection.
Leading outside points remain missing until the first valid point; missing
points stay missing and do not erase the previous valid position. HD, frame IDs,
timestamps and frame spike assignments are unchanged. The source CSV is never
rewritten.

The analysis uses the first interval matching the phase, the nearest retained
pose frame for each spike (ties go to the earlier frame), and the median frame
interval for occupancy. HD is north-zero clockwise; egocentric bearing is
clockwise relative to the head: 0° forward, 90° right, ±180° back, −90° left.
Rays use saved bin centers (3°, 9°, …, 357° for the default 6° bins), with the
configured Gaussian smoothing. Maximum distance remains
`rig_size_cm / 2 * sqrt(2)`. Historical results predating the clockwise/bin-center
correction need explicit regeneration; reversing old arrays is insufficient.

The three named distance bands select bins by their centers. At 40 cm with
20 distance bins, their actual bin supports are 0–8.485, 8.485–15.556, and
15.556–28.284 cm. `distanceBinSupportCm` records these edges in each band
file; `requestedDistanceBandCm` retains the nominal 8/16 cm thresholds.
Band responses sum stored rates after smoothing; they do not pool occupancy.

## Saved results

Analysis and plotting share `EBB.spatial_cell_results.spatial_result_path`.
The standard directory is `session/data/spatial_cells/ProbeA/baseline/`, with
probe and phase taken from the selected configuration.

| Filename | Content |
| --- | --- |
| `egocentric_rate_map.rfmap` | Full angle × distance rate matrix |
| `egocentric_rate_map_0-8.rfmap` | 360° bearing curve, distance ≤8 cm |
| `egocentric_rate_map_8-16.rfmap` | 360° bearing curve, 8 < distance ≤16 cm |
| `egocentric_rate_map_16-.rfmap` | 360° bearing curve, distance >16 cm |
| `restored_maps.npz` | Unit IDs, axes, trajectory, raw occupancy/spike maps and rates |
| `frame_spike_counts.npz` | Sparse per-frame spike counts for all selected units |
| `position_correction.npz` | Original/corrected positions, hold masks, frame IDs/times, raw `hd_deg`, calibrated `ebc_hd_deg`, native `theta_deg`/`theta_d_cm`, and full `exposure_times`; overlay reads eight saved angle columns |
| `audit.json` | Source/timing provenance, actual geometry and completion status |

RFMaps use version-2 indexed NPZ storage under the `.rfmap` suffix. The full
array shape is `(unit, angle, distance, 1)`; each band is `(unit, angle, 1, 1)`.
They retain unit order, float64 rates, physical coordinates/bin edges and one
time bin spanning the analysis interval. `occupancyTimeSec` contains ones tagged
`normalization_identity_for_precomputed_rates`; measured occupancy is saved in
`restored_maps.npz`. Loading preserves finite rates and NaNs; missing bins are
not converted into measured zero rates.

Bearing curves sum saved smoothed rates over distance bins selected by their
centers. Centers exactly at 8 or 16 cm belong to the lower band. The default
20-bin grid gives 6, 5 and 9 bins in the three bands. These sums add no smoothing
or occupancy normalization. A contributing NaN is ignored when other distances
have data; an entirely missing or empty band remains NaN. The three band files
record their physical cm edges, included centers and
`responseAggregation = "sum_over_distance"`.

`run_analysis()` returns the four RFMap paths, full map first. Rerunning explicitly
replaces analysis outputs; use another `result_path` to retain a separate version.
The four RFMaps are saved after all unit calculations finish, then the plot inputs
and audit are saved. Plotting never generates missing analysis files.

## Plotting

Both the notebook and pipeline use `EBB.spatial_cell_plotting` to render all saved
good units, including zero-spike units, to:

```text
session/data/spatial_cells/ProbeA/all_units/unit_<id>.png
session/data/spatial_cells/ProbeA/all_units/unit_<id>.svg
```

The notebook's `unit_id` selects only its inline preview. The pipeline entry is:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/python -m scripts.plot_spatial_cells RESULT_DIR --workers 4'
```

`RESULT_DIR` is the standard `Probe<probe>/<phase>` directory. The CLI writes
`spatial_plot_manifest.json` beside the exported images. For an explicitly selected
historical/nonstandard result directory, pass `--output` and `--hd-tc` rather than
relying on the standard directory layout.

The ten panels show egocentric occupancy, spikes, rate, polar rate and bearing
curves; allocentric occupancy, spikes, rate, trajectory with spikes and HD tuning.
The EBC bearing panel uses the same distance sum as the band files and population
plot: the full saved distance range plus the three rings. The current saved maps end at 28.991 cm; the >16 cm curve therefore covers the saved 16–28.991 cm range. It does not select a single column at
the peak distance.

HD curves come from this recording's saved
`data/tuning_curves/Probe<probe>/tuning_curves.tc`. `HD.profiles.load_hd_tc` only
reads the file; `hd_profiles` rebins its loaded counts and occupancy to 30 angular
bins without smoothing or reclassification. Every saved HD class is retained.
Units absent from the HD file remain in the spatial export and their HD panel
says no saved TC is available. No older spatial HD curve is substituted. An absent
HD file is an error; generate it explicitly before plotting.

The optional single-unit RFMap preview supports `heatmap`, `polar` and `donut`.
EBC heatmaps and polar ticks display signed bearings from −180° to 180°: 0° forward, +90° right and −90° left. Stored angular bins stay unchanged. Polar views use saved cm edges and clockwise angles; `donut_hole_fraction` changes
only the display origin, not distances or data. `is_save` controls only this
extra single-panel PNG. The all-unit PNG/SVG export runs separately by default.
Every figure, axes and export uses an opaque white background and dark text.

The final population heatmap uses `Utils.plotting.plot_keyed_heatmap`, shared with
`hd_rf_comparison.ipynb`. It excludes entirely missing/all-zero units from this
summary only, records their IDs in `excluded_unit_ids`, collapses distance,
normalizes each retained curve by its maximum, and sorts by the native-bin peak
with unit ID as the tie-breaker. Its columns are `−180 → −90 → 0 → 90 → 180`,
without a sign inversion. Missing angular bins stay missing. Up to 12 y-axis
labels show recorded Kilosort IDs, while every retained row remains in the plot.
An empty population gets an explanatory message. `is_save_heatmap = True` saves
`Probe<probe>/all_units_angle_heatmap.png`; the default is `False`.

Trajectory displays break connections above 250 cm/s, across nonconsecutive
frame IDs, or at time gaps longer than 1.5 median frame intervals. They retain
positions, mark break endpoints and show the calibrated arena. This display rule
is separate from boundary hold and never changes saved arrays. Older arrays
without timestamps use a 5 cm step limit.
`scripts/export_spatial_trajectory_overviews.py` exports a recording-level
trajectory without unit-specific spike dots.

Historical batch/review directories are retained as evidence. They are not the
standard analysis or figure destination.

## Motive circular arenas

Both repository and production pipelines run `python -m scripts.run_cylinder_spatial`
from the repository checkout. The production standalone entry links to that same
script. It calls `EBB.ebc_analysis.load_motive_session` and `compute_circle_ebc`;
historical `Utils.ebc_*` imports forward to `EBB` rather than another calculation.

Motive input yaw remains counterclockwise from world +X. An egocentric bearing
turns clockwise from that heading: `world_ray = yaw - bearing`, with 0° forward,
+90° right and −90° left. Native saved bins retain their 3°, 9°, …, 357° center
samples; plots reorder these samples to −180°…180° without changing values.
The existing inner/outer model radii remain 15/30 cm; the Basler 40 cm square
calibration does not apply to circular arenas.

The loader preserves the original exposure timestamps. It accepts matching
zero-based continuous frames and the established single extra trailing Motive
frame. Larger count mismatches require the explicit `--ignore-hd-ttl` option,
which keeps the common prefix and records that choice in `cameraTiming`.
Pipelines copy `filtered.csv` and `head_direction.json` from the same completed
Motive analysis run. Running this entry explicitly writes the four circular
RFMaps under `data/spatial_cells/Probe<probe>/<phase>/inner/` and `outer/`.

The pipelines retain Motive HD and circular EBC analysis but do not generate
Motive overlay videos. Basler overlay generation remains enabled. Existing
Motive videos are retained.

## Verification

Run the relevant tests remotely:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  ~/.virtualenvs/rfmapping/bin/python -m pytest -q \
  tests/test_plotting.py tests/test_spatial_cell.py tests/test_spatial_pipeline_outputs.py'
```

Verification covers geometry/timing, RFMap values and missing bins, 8/16 cm
selection, saved plot inputs, notebook displays, all-unit export and white
backgrounds. Real-data review must also inspect rendered PNG/SVG figures and
check their unit inventory against the saved analysis; a successful export alone
does not establish visual correctness.
