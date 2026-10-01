# EBC video entrances

Run these scripts on `hhw9l84` with `~/.virtualenvs/rfmapping/bin/python`.
The video entrances in the repository root are:

| Entrance | Display | Scope |
| --- | --- | --- |
| `ebc_video_rectangle.py` | Basler recording with rectangular-arena rays and distances | All good units; full recording by default |
| `ebc_video_circle.py` | Raw camera recording with calibrated circular-screen geometry and VS bearing | All good units; full recording |
| `ebc_video_sep.py` | Separate world/screen and saved EBC heatmap panels | One selected unit and time range |
| `ebc_tuning_video.py` | JSON HD, animal-centered 1D EBC curve, and preferred-bearing wall ray | Selected good units, or all units with saved EBC maps |
| `hd_rf_population_video.py` | JSON HD and predicted RF allocentric rays, plus scrolling angle traces | One population movie, without audio |

The renderers, synchronized-data export, and electrode-audio functions live
under `Utils`. They are libraries, without additional command-line entrances.
The analysis notebooks and `.rfmap` generation remain independent of video export.

## Population HD–RF prediction

`hd_rf_population_video.py` renders the three schemes from
`hd_rf_correlation_schemes.ipynb`: all Class 3 cells, Class 3 cells with 2D
RFs, and the latter cells in the three largest HD+RF polar bins. Default
sources are m19 session 11 TC and session 2 RF. The fitted conversion is
`RF_ego = wrap180(beta - HD)`, so `RF_allo = beta` by construction.

Three booleans at the top of the script choose the outputs:

```python
render_scheme_1 = True
render_scheme_2 = True
render_scheme_3 = True
```

Enabled schemes run concurrently in separate processes and produce
`scheme1.mp4`, `scheme2.mp4`, and/or `scheme3.mp4`. Disabled schemes are not
loaded or rendered. If all three switches are false, the script exits without
reading the recording or creating outputs. All videos are silent population
overlays, with blue HD and orange RF rays and a scrolling angle trace.

```sh
cd ~/Developer/rfmapping
~/.virtualenvs/rfmapping/bin/python hd_rf_population_video.py
```

The script defaults to the full recording (`duration_s = None`). A short run:

```sh
~/.virtualenvs/rfmapping/bin/python hd_rf_population_video.py \
  --start 20 --duration 10 \
  --output-dir output/hd_rf_population/m19_260827_11_three_schemes_test
```

GPU encoding is explicitly `h264_nvenc`; initialization failure stops the
script. Decoding and overlay drawing use the CPU. On hhw9l84, driver-matched
NVENC libraries are unpacked in the user's `.local/lib/rfmapping-nvenc`
directory, and the script sets its process library path. The system driver
is unchanged. Each encoder uses one overlay worker because the common
multiworker renderer assumes the original video height.

HD comes only from generated `head_direction.json`; the position CSV
contributes `front_x/front_y`. For this legacy session the JSON frame ID is
one-based, so subtract one before joining the zero-based video/position
frames. Camera TTL polarity and frame mapping come from saved TC metadata.
`--start` and `--duration` use AVI time; the trace uses ADC-relative measured
exposure times. Missing tracking stays as a gap. The fixed 12° orange sector
is a direction marker, not a confidence interval.

Each enabled scheme also saves a PNG, frame-angle CSV, and JSON source/timing
manifest. `render_manifest.json` is written after all requested outputs finish.
This is a population conversion applied to behavior. Its constant RF allo
is the fixed-slope model's definition, not an independent biological finding.

## Rectangle

Edit the recording, pixel bounds, arena size, and audio/video settings at the
top of `ebc_video_rectangle.py`, then run the file. Bounds use
`(left, right, top, bottom)` in original camera pixels. The selected geometry
travels with the recording data to renderer workers.

```sh
cd ~/Developer/rfmapping
~/.virtualenvs/rfmapping/bin/python ebc_video_rectangle.py
```

The CLI can override the session, probe, phase, and output directory:

```sh
~/.virtualenvs/rfmapping/bin/python ebc_video_rectangle.py \
  '/mnt/senzailab/Kai/#Recording/m20/260921/260921_11' \
  --probe A --phase baseline --output-dir /tmp/rectangle_videos
```

`video_start_s` and `video_duration_s` select an optional clip. The default
duration is `None`, which exports the full AVI. Each good unit receives its
own `<unit_id>.mp4`, with shared image rendering and cached electrode audio.

## Circle

Edit the settings at the top of `ebc_video_circle.py`. The recording, camera
registration, complete audited frame clock, VS registration, fixed level-pose
reference, and preserved Motive CSV must belong to the selected recording.
These calibrations are recording-specific.

```sh
~/.virtualenvs/rfmapping/bin/python ebc_video_circle.py
```

Use `--output-dir` to override the destination. This entrance retains every
decoded frame and exports every good unit. Explicit synchronization gaps
remain raw frames with omitted geometry and silent audio. Geometry uses
the original Motive XYZ/fused-yaw calculation before camera projection.

Rectangle and circle videos use continuous Open Ephys voltage from each
unit's peak electrode. Other units and background on that electrode remain
audible. Audio is cached and encoded once per distinct electrode.

## Separate panels (`sep`)

Export synchronized data and render the selected unit through one entrance:

```sh
~/.virtualenvs/rfmapping/bin/python ebc_video_sep.py \
  '/mnt/senzailab/Kai/#Recording/m20/260922/260922_3' \
  --unit 318 --probe A --start 500 --duration 6 \
  --output-dir /tmp/ebc_sep
```

This display requires the saved inner/outer EBC maps. It reuses their rates
and geometry; it does not fit a model or recompute EBC analysis. The output
includes `session_overlay.json`, snapshots, and `screen_ebc_preview.mp4`.
The figure and exported canvas have opaque white backgrounds.

## Animal-centered tuning

```sh
~/.virtualenvs/rfmapping/bin/python ebc_tuning_video.py \
  --date 260921 --session 11 \
  --base-dir '/mnt/senzailab/Kai/#Recording/m20' \
  --probe A --wall-config new --unit 2 --start 10 --duration 10 \
  --output-dir /tmp/ebc_tuning
```

`base_dir` is the mouse directory containing date folders. `probe` defaults
to A. `wall_config` selects the existing `old` or `new` Basler rectangle;
it must match the arena used for the saved EBC analysis. Repeat `--unit`
to select more units, or omit it to render all good units with saved maps.
Units without any positive finite tuning bin are listed and skipped because
their preferred-bearing ray is undefined.
Omit `--duration` for the full recording. The script uses the selected
phase's saved `egocentric_rate_map.rfmap`, without recomputing EBC analysis.

Head direction comes only from `data/processed/head_direction.json`;
CSV `front_x/front_y` provide the head position. Frame IDs join the two
sources exactly. HD is north-zero clockwise, while EBC bearing is positive
counterclockwise from the head. The curve sums saved Hz across distance
bins using `RFMap.to_1d_array('y')` and scales its radius independently for
each unit. Its maximum 1D bin defines the head-to-wall ray. Unvisited bins
remain gaps. Missing JSON HD/position or out-of-bounds positions are labeled.

The real Open Ephys peak-electrode voltage uses the existing 300–6000 Hz
band-pass, gain and noise-expansion settings. This is the existing spike-band
electrode soundtrack, not a low-frequency-only LFP trace or synthetic clicks.
The audio clock uses measured camera exposure timestamps; the camera ADC
channel is read from the saved tuning-curve metadata and can be overridden
with `--camera-input-channel`. Each output has an MP4, PNG preview, and JSON
manifest recording data sources, geometry, timing, and audio parameters.

An existing JSON export can be rendered without loading the raw recording:

```sh
~/.virtualenvs/rfmapping/bin/python ebc_video_sep.py \
  --input /path/to/session_overlay.json --start 500 --duration 6 \
  --output-dir /tmp/ebc_sep
```

`--fps` can override the display frame rate. With an existing JSON, omitted
start, duration, and FPS use its saved preview settings.
