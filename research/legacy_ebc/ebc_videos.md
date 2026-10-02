# EBC video entrances

Run these scripts on `hhw9l84` with `~/.virtualenvs/rfmapping/bin/python`.
The video entrances in the repository root are:

| Entrance | Display | Scope |
| --- | --- | --- |
| `ebc_video_rectangle.py` | Basler recording with rectangular-arena rays and distances | All good units; full recording by default |
| `ebc_video_circle.py` | Raw camera recording with calibrated circular-screen geometry and VS bearing | All good units; full recording |
| `ebc_video_sep.py` | Separate world/screen and saved EBC heatmap panels | One selected unit and time range |
| `ebc_tuning_video.py` | JSON HD, animal-centered 1D EBC curve, and preferred-bearing wall ray | Selected good units, or all units with saved EBC maps |
| `hd_rf_population_video.py` | Saved real HD or neural decoded HD, predicted RF rays, and angle traces | Three population schemes, without audio |

The renderers, synchronized-data export, and electrode-audio functions live
under `Utils`. They are libraries, without additional command-line entrances.
The analysis notebooks and `.rfmap` generation remain independent of video export.

## Population HD–RF prediction

`hd_rf_population_video.py` has two explicit stages. Session A supplies RF
maps and session B supplies saved HD tuning curves; their matched unit
directions fit the three relations. Session C supplies the video and either
real `head_direction.json` or its own saved TC and spikes for neural decoding. C may be A,
B, or a separate session. Defaults are m20 `260922_1` for RF, `260922_3` for
HD, and `260922_3` for C, Probe A.

The three schemes are all valid Class 3 HD/RF pairs, the pairs with detected
2D RFs, and the latter pairs in the three largest HD+RF histogram bins.
Current RF maps are explicitly detected in the run's `relation` directory;
another RF source's masks are never substituted. Each scheme fits RF
egocentric preferred direction directly from the paired unit directions.
The first-harmonic circular regression uses ordinary least squares twice:

```text
cos(RF_ego) ~ a0 + a1*cos(HD) + a2*sin(HD)
sin(RF_ego) ~ b0 + b1*cos(HD) + b2*sin(HD)
h = radians(current_HD_deg)
predicted_RF_ego_deg = degrees(atan2(b0 + b1*cos(h) + b2*sin(h),
                                   a0 + a1*cos(h) + a2*sin(h)))
predicted_RF_allo_deg = wrap360(current_HD_deg + predicted_RF_ego_deg)
```

Trigonometric calculations use radians; exported angles are degrees. At each
video frame, the model predicts RF ego from the selected real/decoded HD, then adds
that HD to obtain RF allo. Both coefficient vectors are estimated from the
unit pairs, so the resulting allocentric direction can vary with HD. This is
the order-1 circular–circular regression described by
[CircStats `circ.reg`](https://search.r-project.org/CRAN/refmans/CircStats/html/circ.reg.html).

The top-level settings choose A, B, C, HD mode, and enabled outputs:

```python
session_a = base_dir / "260922_1"  # RF
session_b = base_dir / "260922_3"  # HD
session_c = base_dir / "260922_3"  # Video
use_decoded_hd = False
render_scheme_1 = True
render_scheme_2 = True
render_scheme_3 = True
```

The script directly reads `data/tuning_curves/ProbeA/tuning_curves.tc` from
session B and, for decoding, from session C. It never generates TC or searches
for another source if the configured file is absent. `tc_is_clockwise=False`
records the current saved TC's documented counter-clockwise convention; HD
peaks and decoded angles receive the explicit clockwise sign conversion.
The real JSON HD is read directly without that conversion.

Enabled schemes run concurrently in separate processes and produce
`scheme1.mp4`, `scheme2.mp4`, and/or `scheme3.mp4`. Disabled schemes are not
loaded or rendered. If all three switches are false, the script exits without
reading the recording or creating outputs. All videos are silent population
overlays. The arena image shows blue HD and the orange derived RF allo ray.
The head-relative dial shows blue forward (0°) and orange predicted RF ego.
The scrolling trace shows blue HD on the left axis (0–360°) and orange
predicted RF ego on the right axis (−180–180°); the sidebar reports HD, RF ego,
and their derived sum separately.

The default is real HD. Select decoded HD for this run:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping &&
  ~/.virtualenvs/rfmapping/bin/python hd_rf_population_video.py --decoded-hd'
```

This renders the full recording (`duration_s = None`). Add `--start 20
--duration 10` for a short clip. Without `--decoded-hd`, the script uses real
JSON HD. `--video-session /path/to/C` changes C independently of the training
sessions. C provides its own Motive XYZ, head-direction JSON, and raw ADC
recording; decoding also requires its own saved TC and spikes. Its unit IDs are
selected independently of the A/B fit cohorts.

Outputs are always under C's `data/hd_rf`, by default in
`hd260922_3_rf260922_1_decoded` or `hd260922_3_rf260922_1_real`. `--output-name`
changes that subdirectory. The `relation` subdirectory records A/B sources,
unit pairs, detection, selection, and fitted models. `head_direction.csv`
records C's selected HD input.

GPU encoding is explicitly `h264_nvenc`; initialization failure stops the
script. Decoding and overlay drawing use the CPU. On hhw9l84, driver-matched
NVENC libraries are unpacked in the user's `.local/lib/rfmapping-nvenc`
directory, and the script sets its process library path. The system driver
is unchanged. Each encoder uses one overlay worker because the common
multiworker renderer assumes the original video height.

Real HD comes directly from C's `data/processed/head_direction.json`, entity
`hp4`, field `head_direction_deg`. Motive XYZ comes from C's
`data/processed/trimmed_input.csv`; it supplies position only, not HD. Decoded HD uses C's
native saved count/occupancy rates and a uniform-prior Poisson MAP decoder in
100 ms bins. Zero rates remain zero; no pseudocount, clipping, smoothing,
interpolation, or behavioral-angle replacement is applied. Silent or
impossible population bins remain NaN. TC training and decoding can overlap;
this is a fitted population estimate.

The script constructs the clock from C's raw ADC: Motive channel 1/high/14000,
Basler channel 2/low/2800, then unique nearest pulse pairing. The explicit
`video_exposure_stride` and `video_exposure_phase` settings map AVI frames to
Motive exposure ordinals; they default to 2 and 0 and are not inferred from
the AVI FPS. The regular mapping must agree with the decoded frame count.
Recordings with dropped video frames need explicit `video_motive_segments`
in the script. The existing m20 `260922_3` segments retain 41 unavailable
frames and approximately 16.67 ms absolute synchronization uncertainty. Those
frames retain their raw image without synchronized geometry. No precomputed
`camera_clock` files are required. `--start` is
measured elapsed time from the first video exposure; `--duration` is measured
ADC duration. Export uses the measured frame rate rather than the incorrect
AVI header rate. The fixed 12° orange sector is a direction marker, not a
confidence interval.

The cylinder overlay projects Motive XYZ and world-space ray endpoints into
the raw camera image, using the existing `Utils.ebc_camera` geometry. Shared
setup files live in `config/cylinder_camera_registration.json` and
`config/cylinder.calib`; `--camera-registration` and `--cylinder-calibration`
can select another setup. They are independent of A/B/C and can be reused
while camera placement and Motive world coordinates remain unchanged. The
registration was restored from the previously validated file, without a new
fit. JSON HD remains unchanged; `hd_world_zero_deg` specifies its zero in
Motive world XY for projection only.

Each enabled scheme also saves a PNG, frame-angle CSV, and JSON source/timing
manifest. `render_manifest.json` is written after all requested outputs finish.
The movie applies a relationship fitted across units' preferred angles to
behavioral HD. Its RF ray is a model prediction, not a measured instantaneous
RF. Reported angular fit error is in-sample. The signed Fisher–Lee correlation
and its permutation p-value describe association between the original unit
pairs; they do not measure prediction accuracy. Scheme 3 was selected using
HD+RF and remains descriptive, without a permutation p-value. A zero fitted
cosine/sine vector has no defined direction; its length is not a confidence
interval or probability.

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
