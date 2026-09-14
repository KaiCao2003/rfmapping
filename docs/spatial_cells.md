# Spatial-cell analysis and plotting

`spatial_cell_analysis.py` replaces the combined `spatial_cell.py` script.
The old analysis and plotting notebooks are removed; their grid/border scores
and shuffle analyses are not part of this pipeline. Existing geometry,
occupancy weighting, bin limits, smoothing and tuning calculations are preserved.

Configure arena bounds, bin sizes and smoothing at the top of
`spatial_cell_analysis.py`. The recording, probe and phase also accept command
line overrides. Run project Python only on `hhw9l84`:

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
output directory; it must not already exist. `--workers N` limits parallel
analysis workers. `--recording-root /path/to/mouse` changes the mouse directory;
`--units 7 9` analyzes only those good units, otherwise all good units are saved.
An interrupted run has no `metadata.json`; rerun into a new directory. The
manifest is published only after all units finish. Existing results are never
overwritten by analysis.

Inputs must include a processed `session/<date>.csv` with `frame`, `center_x`,
`center_y`, and `hd_deg` columns, Kilosort good-unit labels and clusters, saved
`data/probeA/adc_spike_time.npy`, and saved camera times (`camera_frame_times.npy`
or `sync_data.json["exposure_sampling_number_list_mid"]`). Raw Motive CSV exports
must be processed upstream first. This script does not generate missing timing.
The configured `m19/260831/260831_2` recording currently lacks saved camera times
(checked 2026-09-14); the example needs those upstream inputs before it can run.

The plotting command reads only the result directory. Use `--units 7 9` to
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

Egocentric boundary data is fully represented in these files. It is not
exported as `.rfmap`: the current regular viewer schema requires trial counts
and a time axis, while the free-moving HDF5 viewer supports its own Square/Bar
schemas. A boundary `.rfmap` requires a dedicated format and matching GUI
support in `../rfmapping_gui`; relabeling firing rates as trial counts would
change their scientific meaning.

## Verification

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && \
  PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
  ~/.virtualenvs/rfmapping/bin/python -m pytest -q \
  tests/test_spatial_cell.py tests/test_tuning_curve_utils.py'
```

Tests cover geometry, spike alignment, numeric NPZ round trips including NaNs,
shared-array storage, schema checks and interrupted runs. An integration test
runs analysis in a separate process, makes the raw recording path unavailable,
then renders all 22 PNG/SVG outputs per unit using only saved results. Export checks also
exercise dark global Matplotlib defaults to confirm opaque white figures and
axes with readable labels.
