# EBC videos

The active EBC entries are `ebc_video_rectangle.py` and `ebc_video_circle.py`
(the latter is the cylinder entry). They share the same eight-ray calculation,
prepared overlay structure, renderer, encoder and electrode-audio batch export.
`hd_rf_population_video.py` is a separate workflow and was not redesigned here.

Run project code on `hhw9l84` in `~/Developer/rfmapping` with
`~/.virtualenvs/rfmapping/bin/python`:

```sh
cd ~/Developer/rfmapping
~/.virtualenvs/rfmapping/bin/python ebc_video_rectangle.py /path/to/rectangle.json
~/.virtualenvs/rfmapping/bin/python ebc_video_circle.py /path/to/cylinder.json
# Explicit short, silent preview:
~/.virtualenvs/rfmapping/bin/python ebc_video_circle.py /path/to/cylinder.json --start 500 --duration 2 --silent
```

`--start` and `--duration` are seconds on the output video timeline, using the
measured frame rate; start zero selects the first video frame. Without
`--duration`, export continues to the end of the recording. Without
`--silent`, each good unit receives synchronized electrode audio; channels
shared by multiple units are read and encoded once. Rendering is always an
explicit command, never a side effect of loading pose, calculating rays, or
preparing an overlay.

## Function boundaries

1. `load_basler_position` and `load_motive_position` return `PositionInfo`:
   zero-based video frame IDs, ADC-relative seconds, world XYZ in cm, HD in
   degrees CCW from world +X, validity and source metadata. Missing clock rows
   remain missing; position conversion never repairs synchronization.
2. `rectangle_boundary` and `cylinder_boundary` return `BoundaryInfo` with
   the same intersection interface, world-cm camera projection and drawing
   outline. Cylinder intersections are analytic; sampled outlines are only
   for drawing. Cylinder world origin and head Z are preserved.
3. `compute_ebc_rays(position, boundary)` returns the same `EBCInfo` for both
   shapes: eight bearings 0,45,...,315 degrees, exact endpoints in cm, distances
   and validity. This function has no rectangle/cylinder dispatch.
4. `prepare_overlay_info(position, boundary, ebc)` projects calculated results.
   `EBCOverlay.draw` draws only that prepared payload; it does not load files,
   select units, calculate EBC, normalize tuning or fit models.
5. `export_prepared_video` orchestrates encoding/audio and calls a separate
   `save_geometry_audit`. Output includes a preview, source/provenance JSON and
   `frame_geometry.csv`; synchronization gaps keep raw images and silent audio.

The source-specific workflow adapters load/normalize inputs and call the shared
calculation and preparation stages. Shared
functions contain no session paths, pixel bounds, physical arena dimensions,
HD registration values or camera-channel settings. Those belong in the config.
All config paths resolve relative to the config file. Output directories must
be supplied explicitly. The config is not written or changed by rendering.

## Rectangle config

This example is a schema example, not a calibration for an arbitrary recording.
Choose the actual bounds, dimensions, source columns and heading convention.
`heading_zero_deg` gives the source HD zero in world CCW degrees; north is 90.
`heading_clockwise` describes the source CSV, not the desired display sign.

```json
{
  "session": "/recordings/mouse/date/date_session",
  "video_path": "recording.avi",
  "pose_path": "pose.csv",
  "kilosort_dir": "kilosort/ProbeA/kilosort_session",
  "output_dir": "output/ebc",
  "probe": "A",
  "phase": "baseline",
  "boundary": {"bounds_px": [100, 500, 100, 500], "size_cm": 40},
  "pose": {
    "position_columns": ["center_x", "center_y"],
    "heading_column": "hd_deg",
    "heading_clockwise": true,
    "heading_zero_deg": 90,
    "frame_offset": 0
  },
  "camera_timing": {
    "camera_input_channel": 1,
    "camera_ttl_threshold": 14000,
    "camera_ttl_active_high": false
  },
  "video_workers": 2,
  "video_encoder": "auto"
}
```

`phase` is a label. To restrict geometry to an analysis interval, add explicit
`"interval_s": [start, stop]` in ADC-relative seconds. Source video frames are
retained outside that interval. Saved camera timing keeps its existing edge or
midpoint definition; camera settings govern raw extraction only when used.

## Cylinder config

Supply the matching raw Motive pose, physical calibration, camera projection,
level reference and audited clock. The clock directory contains
`video_clock_qc.json`, `video_adc_times.npy` and `video_motive_rows.npy`.
Its declared session, decoded frame count and unavailable ranges must agree.

```json
{
  "session": "/recordings/mouse/date/date_session",
  "video_path": "recording.avi",
  "pose_path": "processed/trimmed_input.csv",
  "kilosort_dir": "kilosort/ProbeA/kilosort_session",
  "output_dir": "output/ebc",
  "probe": "A",
  "headplate": "hp4",
  "calibration_path": "cylinder.calib",
  "registration_path": "camera_registration.json",
  "clock_dir": "full_video_clock",
  "hd_reference_path": "hd_level_reference.json",
  "hd_world_zero_deg": 0,
  "video_workers": 2,
  "video_encoder": "auto"
}
```

For an optional real VS marker, also supply `stimulus_mat_path`,
`stimulus_onsets_path`, `vs_zero_deg`, and `background_luminance`. The HD world
zero and the stimulus world zero are separate inputs. No offset is inferred.

Optional `audio` settings are `audio_source` (`continuous` or `clicks`),
`gain`, `audio_band_hz`, `audio_gate_sigma`, and `audio_expander_ratio`.
Continuous audio is electrode voltage, not an isolated sorted-unit waveform.

## Retired entries

The old `sep` and tuning entries and their historical source/tests are retained
under `research/legacy_ebc/`. They are not active root entrypoints. The old
cylinder renderer snapshot is also preserved there. Current population-video
helpers in `Utils.ebc_camera` remain for that separate workflow.
These are source snapshots for historical comparison; running them requires
their matching imports and dependencies. The current eight-ray display does
not reproduce the saved inner/outer heatmaps or unit tuning-curve view.

The config entries and their shared EBC modules were synchronized to
`hhw9l84:~/Developer/rfmapping` on 2026-10-02. That host's preceding source is
preserved under `research/legacy_ebc/remote-20261002.e2j0bI/`, with original
relative paths and `source.sha256`. The sync backup and verification logs are
under `.codex_tmp/ebc-sync-20261002.e2j0bI/`. The three EBC test files plus the
selected population/cylinder compatibility checks passed (44 tests), and both
entrypoints' `--help` commands passed. Verification used synthetic media in
temporary test directories; it did not render a recording or write session
outputs. The population entry, its analysis helpers and project dependencies
were left unchanged.

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
is unchanged. This separate entry continues to request one overlay worker.

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
