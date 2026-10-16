# EBC videos

The active Basler overlay replays saved spatial-analysis frame values through
`ebc_video_rectangle.py`. It selects eight existing native EBC rays; it does
not reload tracking, extract camera TTLs, calibrate headings, or calculate new
boundary intersections. Motive HD and circular EBC analysis remain enabled,
but the repository and production pipelines do not generate Motive overlays.
`hd_rf_population_video.py` is a separate workflow.

Run project code on `hhw9l84` in `~/Developer/rfmapping` with
`~/.virtualenvs/rfmapping/bin/python`:

```sh
cd ~/Developer/rfmapping
~/.virtualenvs/rfmapping/bin/python ebc_video_rectangle.py /path/to/rectangle.json
# Explicit short, silent preview from the same saved analysis:
~/.virtualenvs/rfmapping/bin/python ebc_video_rectangle.py /path/to/rectangle.json --start 500 --duration 2 --silent
```

`--start` and `--duration` are seconds on the source video timeline. Basler
keeps the source AVI frame rate; start zero selects its first frame. Without
`--duration`, export continues to the end of the recording. Without `--silent`,
each good unit receives synchronized electrode audio; channels shared by
multiple units are read and encoded once. Rendering is an explicit command.
Loading saved analysis never generates missing data or renders a video.

## Function boundaries

1. Spatial analysis saves `position_correction.npz` with analyzed frame IDs and
   times, raw and corrected positions, hold flags, raw and physical HD,
   `theta_deg`, `theta_d_cm`, and the full `exposure_times` camera clock.
   `audit.json` supplies the boundary, physical dimensions, selected phase,
   source paths, ADC time origin, and camera-timing provenance.
2. `EBB.ebc_workflow.load_saved_ebc_frames` reads those two files. It does not
   read the tracking CSV or derive any replacement values. Old analysis
   directories missing the saved ray or clock arrays require an explicit
   spatial-analysis generation step before video export.
3. `select_saved_ebc_rays` selects native indices
   `np.arange(8) * len(theta_deg) // 8`. For the 60-bin analysis, these are
   indices 0, 7, 15, 22, 30, 37, 45, 52, with signed angles
   3°, 45°, 93°, 135°, −177°, −135°, −87°, −45°. Distances are copied from those
   columns of `theta_d_cm`; labels show the actual saved angles. Bearings are
   clockwise from HD: 0° forward, +90° right, −90° left. Endpoints are reconstructed
   from the saved position, physical heading and distance for projection only;
   no boundary intersection is recalculated.
4. `prepare_overlay_info` projects the selected saved geometry into video
   pixels. `EBCOverlay.draw` draws that prepared payload; it does not load
   analysis, calculate EBC, normalize tuning or fit models.
5. `export_prepared_video` handles rendering, encoding and electrode audio,
   then `save_geometry_audit` writes the displayed frame values to
   `frame_geometry.csv`. Synchronization gaps retain raw images and silent
   audio. Output provenance identifies the analysis directory, geometry file
   and selected native angle indices.

The overlay therefore exposes the saved positions, physical headings, ray
distances and timing used by analysis. It does not display the EBC tuning map
or independently validate spike binning, occupancy, smoothing or statistics.
Those remain analysis outputs and require their corresponding checks.

Analysis holds a finite position outside the inclusive boundary at the last
complete in-bound location until tracking returns inside. It corrects the full
source sequence before phase selection. Leading outside positions remain
missing; missing source rows remain gaps. Holding position does not hold HD.
Analysis scales the pixel heading vector along X and Y using the same cm/pixel
factors as position, then saves the physical heading. Video reads these saved
results and does not repeat correction or heading calibration. The Basler
panel displays physical HD north-zero clockwise; internal `hd_world_deg`
uses world +X-zero CCW. Raw source CSV values are not rewritten.

## Production Basler pipeline

The repository `scripts/run_pipeline.sh` and production
`/home/kai/scripts/run_pipeline.sh` and `run_pipeline_2.sh` run spatial analysis,
plot its saved outputs, then render the Basler overlay. Boundary selection and
camera-channel settings belong to analysis. The video call receives
`--analysis-dir` and does not accept another boundary or camera-timing config.
The production `/home/kai/scripts/ebc_video.py` rectangle branch writes a small
rendering config and invokes the maintained `ebc_video_rectangle.py` entry.

Spatial analysis saves four RF maps plus `restored_maps.npz`,
`frame_spike_counts.npz`, `position_correction.npz` and `audit.json` in
`data/spatial_cells/Probe<probe>/<phase>/`.
`python -m scripts.plot_spatial_cells RESULT_DIR --workers N` reads saved
inputs and writes one 10-panel PNG/SVG pair per good unit under `all_units/`,
plus `spatial_plot_manifest.json`. The overlay reads the saved frame geometry
from that same analysis directory.

For m22 `261009_3`, the analysis boundary is left 230, right 740, top 335,
bottom 835, scaled independently along X/Y to the measured 40 × 40 cm interior.
A boundary or calibration change requires explicit analysis regeneration;
video export cannot apply its own override. Render the saved result with:

```sh
cd ~/Developer/rfmapping
mkdir -p /mnt/ssd4.1/ebc_video_tmp/m22_261009_3
TMPDIR=/mnt/ssd4.1/ebc_video_tmp/m22_261009_3 \
LD_LIBRARY_PATH=/home/kai/.local/lib/nvidia-video-595.91.07 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
~/.virtualenvs/rfmapping/bin/python /home/kai/scripts/ebc_video.py \
  '/mnt/senzailab/Kai/#Recording/m22/261009/261009_3' \
  --probe A --phase baseline --arena-type rectangle \
  --analysis-dir '/mnt/senzailab/Kai/#Recording/m22/261009/261009_3/data/spatial_cells/ProbeA/baseline' \
  --workers 48 --audio-workers 48 --video-encoder h264_nvenc \
  --output-dir '/mnt/senzailab/Kai/#Recording/m22/261009/261009_3/data/spatial_cells/videos_boundary_hold/ProbeA'
```

The output retains `overlay.mp4`, `overlay.png`, `overlay.json`,
`frame_geometry.csv`, every good-unit MP4, and `manifest.json`. Completed full
exports are indexed by `data/spatial_cells/video_outputs.json`. Existing video
outputs are retained. The 326-unit session has 198 distinct electrode channels;
full continuous-audio scratch needs about 101 GB, so use the dedicated SSD
`TMPDIR` rather than the smaller `/tmp`.

## Re-render the remaining Basler sessions

`scripts/reoverlay_basler_sessions.py` currently starts at m20 260921_11, then
runs m20 260918_11 and m19 260831_2, using 48 rendering workers and 48 audio
workers per session. The default is set by `START_FROM` in the script. It
reads each session's saved Probe A baseline analysis and replaces the full
overlay and good-unit movies under `data/spatial_cells/videos_boundary_hold/ProbeA/`.
It does not regenerate analysis or tracking.

Inspect the commands before starting:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping &&
  ~/.virtualenvs/rfmapping/bin/python scripts/reoverlay_basler_sessions.py --dry-run'
```

Start a batch that continues after SSH closes:

```sh
ssh hhw9l84 'cd ~/Developer/rfmapping &&
  nohup ~/.virtualenvs/rfmapping/bin/python -u scripts/reoverlay_basler_sessions.py \
    </dev/null >/dev/null 2>&1 &'
```

Normal runs append their start time, commands, progress, and errors to
`/mnt/ssd4.1/ebc_video_tmp/reoverlay_basler_sessions.log`. Both the batch and
its children write directly to this file, independently of the SSH output
pipe. Follow progress with:

```sh
ssh hhw9l84 'tail -f /mnt/ssd4.1/ebc_video_tmp/reoverlay_basler_sessions.log'
```

A nonblocking file lock prevents a second copy of this batch from starting
while the first is active; the lock releases when the batch exits. `--dry-run`
only prints commands and does not create logs, locks, or output directories.
Audio workers are used after the shared overlay finishes; they do not increase
the overlay frame rate.

Use `--start-from` to override the default; `--start-from m22/261009/261009_3`
selects all seven sessions. The batch reruns selected sessions in full,
including an incomplete session; it does not resume individual unit movies
within a session. Combine it with `--dry-run` to inspect the selected commands.
An existing active batch continues automatically, so no second launch is
needed while it is still running.

## Rectangle config

The config selects the saved analysis and rendering/audio options. Paths
resolve relative to the config file. The analysis audit supplies the probe,
phase, source paths and geometry; there are no `pose`, `boundary`,
`camera_timing` or independent `interval_s` settings for this saved-only path.
The saved analysis phase determines where geometry is available. Other source
video frames remain present without geometry. Supply `output_dir` explicitly.
This example assumes the config file is stored in the session directory.

```json
{
  "session": "/recordings/mouse/date/date_session",
  "video_path": "recording.avi",
  "analysis_dir": "data/spatial_cells/ProbeA/baseline",
  "output_dir": "data/spatial_cells/videos_boundary_hold/ProbeA",
  "video_workers": 2,
  "audio_workers": 4,
  "video_encoder": "auto"
}
```

Optional `audio` settings are `audio_source` (`continuous` or `clicks`),
`gain`, `audio_band_hz`, `audio_gate_sigma`, and `audio_expander_ratio`.
The production default is synchronized 300–6000 Hz electrode voltage with gain
0.35, gate 3 sigma and expander ratio 4; `clicks` uses sorted spikes. Audio
export reads its electrode/spike sources separately from saved EBC geometry.
Continuous audio is electrode voltage, not an isolated sorted-unit waveform.

## Historical Motive video workflows

`ebc_video_circle.py`, the older cylinder branch and screen/replay helpers are
retained historical or optional workflows. They are not invoked by the current
repository or production pipelines. Their pose/calibration/clock preparation
can calculate geometry separately, so they are not verification replays of the
canonical saved circular EBC analysis. No current Motive overlay is required.
Motive HD and circular EBC map generation remain active; see
[spatial-cell analysis](spatial_cells.md#motive-circular-arenas).

## Retired entries

The old `sep` and tuning entries and their historical source/tests are retained
under `research/legacy_ebc/`. They are not active root entrypoints. The old
cylinder renderer snapshot is also preserved there. Current population-video
helpers in `EBB.ebc_camera` remain for that separate workflow.
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
the raw camera image, using the existing `EBB.ebc_camera` geometry. Shared
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
