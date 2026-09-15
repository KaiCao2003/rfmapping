# Spatial-cell analysis and plotting

Use `spatial_cell_analysis.ipynb` to analyze and save, then
`spatial_cell_plotting.ipynb` to read the saved files and plot. Their small Python
modules retain the reusable functions and command-line entry points.
The legacy grid/border scores and shuffle analyses are not part of this pipeline.
Existing geometry,
occupancy weighting, bin limits, smoothing and tuning calculations are preserved.

Configure the recording, arena bounds, bin sizes and smoothing in the analysis
notebook. Select exactly one camera output explicitly:

```python
basler_output = True
optihub2_output = False
```

These presets match this setup: Basler opto-coupled `ExposureActive` is
electrically active-low, while OptiHub2 uses active-high pulses. Basler's
logical exposure signal must not be confused with the measured pin voltage;
`LineInverter` can reverse that voltage again. The Basler preset assumes no
additional inversion. See [Basler Line Status](https://docs.baslerweb.com/line-status#opto-coupled-output-line)
and [Line Inverter](https://docs.baslerweb.com/line-inverter).
The code never infers camera type or polarity from the signal.

Run notebook kernels with `~/.virtualenvs/rfmapping` on `hhw9l84`.
The command-line entry points also remain available:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  ~/.virtualenvs/rfmapping/bin/python spatial_cell_analysis.py \
  --date 260831 --recording-number 2 --probe A --phase baseline --workers 4'

ssh hhw9l84 'cd ~/Developer/rfmapping && \
  MPLBACKEND=Agg ~/.virtualenvs/rfmapping/bin/python spatial_cell_plotting.py \
  "/mnt/senzailab/Kai/#Recording/m19/260831/260831_2/data/spatial_cells/ProbeA/baseline"'
```

Analysis defaults to `session/data/spatial_cells/ProbeA/baseline/` with the
configured probe and phase. `--output /path/to/new/results` selects another
output directory; it must be new or empty. `--workers N` limits parallel
analysis workers. `--recording-root /path/to/mouse` changes the mouse directory;
`--units 7 9` analyzes only those good units, otherwise all good units are saved.
An interrupted run has no `metadata.json`; rerun into a new directory. The
manifest is published only after all units finish. Existing results are never
overwritten by analysis.

Inputs must include a processed `session/<date>.csv` with `frame`, `center_x`,
`center_y`, and `hd_deg` columns, Kilosort good-unit labels and clusters, and saved
`data/probeA/adc_spike_time.npy`. Saved camera times (`camera_frame_times.npy`
or `sync_data.json["exposure_sampling_number_list_mid"]`) are read when available.
When they are absent, the configured ADC channel is scanned in chunks and the
midpoint of each complete exposure pulse is used, relative to the ADC origin.
The low intervals before/after Basler acquisition are excluded. This does not
write or replace timing files in the source recording. `camera_input_channel`
is zero-based; `camera_ttl_threshold` uses raw int16 ADC units.

Basler pose frame IDs index the exposure times directly, including gaps caused
by invalid pose estimates. OptiHub2 additionally permits one trailing Motive
frame without a TTL, matching `tuning_curves.ipynb`. Raw Motive CSV exports must
still be processed into the pose columns above before spatial analysis.

The plotting notebook previews a selected unit and exports `unit_ids` (all
saved units when `None`). The plotting command also reads only the result
directory. Use `--units 7 9` to
select units and `--output /path/to/figures` to redirect figures. By default all
manifest units are rendered under `results/plots/`, with the existing combined
figure and ten individual panels in PNG and SVG. Figures have opaque white
backgrounds and dark labels. Plotting one unit at a time bounds memory use.
The plotting script does not import the analysis script and does not compute
occupancy, smoothing, rate maps, preferred distance or tuning curves. It only
expands saved per-frame counts into spike positions for the trajectory plot.
Copying the entire result directory is sufficient to replot on another machine;
the source recording paths in the manifest are provenance, not plotting inputs.

## Result format, version 1

- `metadata.json`: schema name/version, source recording, probe/phase, unit IDs,
  input paths, selected interval, ADC origin, camera timing source, analysis
  parameters, units and axis conventions. `frame_dt_s` records the occupancy
  weight assigned to each retained pose frame.
- `session.npz`: frame times, angular/distance/x/y bin edges, trajectory in cm,
  and common egocentric/allocentric occupancy in seconds.
- `units/<id>.npz`: egocentric and allocentric spike/rate maps, both tuning
  curves, preferred egocentric distance in cm, and integer spike counts per
  frame. Rate maps and tuning curves are in Hz; arrays retain NaNs.

NPZ files use numeric arrays and load with `allow_pickle=False`. Shared arrays
are stored once. Per-frame spike counts reproduce trajectory spike positions
without storing repeated coordinates. Egocentric maps have axis order
`(angle, distance)`; allocentric maps have `(x, y)`. Angles are degrees,
distances are cm, and arena y increases downward. Smoothing parameters are in
bins. Intermediate ray geometry and smoothed occupancy used only during
analysis are not persisted; final maps and curves are stored for exact replotting.

The current calculation uses the first interval matching the requested phase,
retains spikes between the first and last selected pose timestamps, and assigns
spikes to the nearest retained frame (midpoint ties go to the earlier frame).
Each frame receives the median retained frame interval as its occupancy weight.
Egocentric rays use the left angular bin edges, with 0 forward and positive
angles toward the animal's left. The maximum saved boundary distance remains
`rig_size_cm / 2 * sqrt(2)`; rays beyond that limit do not enter the histograms.
These choices are preserved from `spatial_cell.py`.

`egocentric_rate_map.rfmap` stores the final plotted matrices using the existing
RF JSON layout `(unit, angle, distance, 1)`. It preserves unit order, coordinate
centers and bin edges, with distance in cm and angle in degrees. The single time
bin spans the selected analysis interval. `responseUnits: "Hz"` identifies the
already computed values; NaNs are stored as JSON `null`.

`Utils.rfmap.load_rf_maps()` reads these values unchanged, without occupancy
normalization or smoothing (also with `unit_firing_rate=False`). For example,
`load_rf_maps(path)[0].to_2d_array()` returns the exact first unit's plotted
matrix. A separate GUI reader must support the `Hz` marker to open these files;
the legacy count-only GUI reader does not yet do so.

## Verification

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  ~/.virtualenvs/rfmapping/bin/python -m pytest -q \
  tests/test_spatial_cell.py tests/test_tuning_curve_utils.py'
```

Tests cover saved times and raw ADC timing with both polarities, pulses crossing
chunk boundaries, Basler idle intervals, explicit output selection, Motive's
trailing frame, geometry, spike alignment, numeric NPZ round trips including NaNs,
shared-array storage, schema checks and interrupted runs. An integration test
runs analysis in a separate process, makes the raw recording path unavailable,
then renders all 22 PNG/SVG outputs per unit using only saved results. Export checks also
exercise dark global Matplotlib defaults to confirm opaque white figures and
axes with readable labels.

Both notebooks were also executed on `m19/260831/260831_2`: 91,447 complete
low pulses at 25 Hz match the video's 91,447 frames, with 91,443 valid pose
frames. All 81 good units were analyzed and saved; every exported RF matrix
matched its unit NPZ exactly. A selected real unit was previewed and all 22
PNG/SVG files were exported. Validation outputs were written to a separate
temporary directory, leaving the default result directory available.
