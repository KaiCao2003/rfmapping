# Spatial-cell analysis and plotting

`spatial_cell_analysis.py` replaces the combined `spatial_cell.py` script.
The old analysis and plotting notebooks are removed; their grid/border scores
and shuffle analyses are not part of this pipeline. Existing geometry,
occupancy weighting, bin limits, smoothing and tuning calculations are preserved.

Configure the recording, probe, phase, arena and analysis parameters at the top
of `spatial_cell_analysis.py`. Run project Python only on `hhw9l84`:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping && ~/.virtualenvs/rfmapping/bin/python spatial_cell_analysis.py'
ssh hhw9l84 'cd ~/Developer/rfmapping && MPLBACKEND=Agg ~/.virtualenvs/rfmapping/bin/python spatial_cell_plotting.py /path/to/results'
```

Analysis defaults to `session/data/spatial_cells/ProbeA/baseline/` with the
configured probe and phase. `--output /path/to/new/results` selects another
output directory; it must not already exist. `--workers N` limits parallel
analysis workers. An interrupted run has no `metadata.json`; rerun into a new
directory. The manifest is written only after all units finish.

The plotting command reads only the result directory. Use `--units 7 9` to
select units and `--output /path/to/figures` to redirect figures. By default all
manifest units are rendered under `results/plots/`, with the existing combined
figure and ten individual panels in PNG and SVG. Figures have opaque white
backgrounds and dark labels. Plotting one unit at a time bounds memory use.

## Result format, version 1

- `metadata.json`: schema name/version, source recording, unit IDs, analysis
  parameters, units and axis conventions.
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

Egocentric boundary data is fully represented in these files. It is not
exported as `.rfmap`: the current regular viewer schema requires trial counts
and a time axis, while the free-moving HDF5 viewer supports its own Square/Bar
schemas. A boundary `.rfmap` requires a dedicated format and matching GUI
support in `../rfmapping_gui`; relabeling firing rates as trial counts would
change their scientific meaning.
