# Simple raw-video overlay

The `aligned_6s` preview is withdrawn as evidence of physical HD alignment.
It substituted an axis fitted to YOLO landmarks for the requested Motive HD.
That fitting path has been removed from the renderer. Its small residuals
describe agreement with those landmarks, not an independent HD measurement.

The 6-second unit 318 preview for `m20/260922_3` covers ADC-relative time
500–506 s, using a single camera view:

- Green: tracked headplate position and head direction.
- Cyan: front, left, back and right screen-wall distances in physical cm.
- Magenta: VS bearing line. Gray trials have no contrast and hide this line.
- Audio: real Probe A electrode 266 voltage, the peak electrode for unit 318.

## Motive bearing

Calculate the geometry before projecting it onto the camera image:

```text
Rc = R_intrinsic_XYZ(motive_rotation_xyz) @ R_reference.T
q = quaternion_xyzw(Rc)
fused_ccw = (2 * atan2(q.z, q.w)) % 360
video_hd = (fused_ccw + 44.7956488331414) % 360
theta = (44.7956488331414 - Square_PositionX) % 360
target_xy = wall_center_xy + wall_radius * (cos(theta), sin(theta))
bearing = wrap180(atan2(target_y - head_y, target_x - head_x) - video_hd)
```

Angles use degrees, with Motive +X at zero and counterclockwise positive.
The origin is tracked `hp4 Position`, without an eye offset. The wall center
and radius come directly from the session's `.calib`. The green HD arrow and
cyan wall intersections use the same `video_hd`. The magenta line joins the
projected head and projected wall target, at headplate Z. Changing HD changes
the relative bearing, not the world position of the VS target. Camera
perspective can change the apparent angle between lines; it is never used to
calculate the numeric bearing.

The user selected the previously approved XYZ/fused-yaw method and confirmed
that its level reference faced VS 0. Thus its world heading zero equals the
independently registered VS zero. The green arrow, bearing and wall rays use
this same world heading. The old 31-degree correction is not added. No HD
axis is fitted to video landmarks, and no extra rotation is applied after
projection. The original Euler-Z analysis files remain unchanged.

The fixed reference is the first 0–2 seconds of
`Take 2026-09-24 01.59.41 PM.csv`, previously confirmed to be level. Its hp4
rigid-body ID matches this recording. The quaternion is copied unchanged from
the approved reference; it is not recomputed from the present clip. Computation
uses CCW directly from the original XYZ data, independent of the Analysis
export configuration's clockwise display flag. In VS coordinates,
`HD_VS = (-fused_ccw) % 360`.
`frame_geometry.csv` records the raw/video HD, head coordinates, target and
bearing for each output frame. Raw data, calibration and saved analyses are
read only.

The coordinate correspondence reverses VS angles and includes its separately
measured zero: `Motive = (vs_zero - VS) % 360`. The reverse conversion, including
HD expressed in VS coordinates, is `VS = (vs_zero - Motive) % 360`. The saved
calibration's 2.5-degree angular zero is unchanged and is not added again.

The 44.79565-degree VS zero was fitted from three raw white-bar positions with
camera projection, wall geometry and sampling height fixed. Of six additional
positions, five yielded measurable centers (0.18–0.45-degree error): three also
passed edge checks, two had broader bands, and one was clipped and unscorable.
See `output/video_overlay_318/vs_alignment_check/alignment_check.json`.

Fused yaw means heading after removing tilt, rather than the horizontal
projection of the nose. Therefore the green arrow need not coincide with
the image-space headplate front/back line on every tilted frame. Arithmetic
checks verify the chosen definition, not anatomical accuracy. The fully
inverted pose is singular; the renderer does not fill or fit missing headings.

The geometry uses this session's calibrated circular screen at headplate height,
not the older inner/outer fitted EBC models. There are no tuning heatmaps, split
panels, side tables or interactive controls.

## Synchronization and audio

Within decoded AVI frames 29000–34000, Motive row is `2 * video_frame + 56`.
Timing uses measured exposure pulses rather than the AVI's nominal frame rate.
Screen-brightness validation has held-out correlation 0.9707 and uncertainty
±8.33 ms. Camera projection has held-out marker median error 1.59 px.
World pose comes from `data/processed/trimmed_input.csv`, indexed by its
original Motive frame ID. The session-root CSV now contains image-space
Basler poses and is not used as world coordinates.

The audio uses the same frame timeline and existing `ebc_video.py` helpers:
300–6000 Hz, gain 0.35, gate 3σ, expander 4:1. The source is continuous OE
voltage on electrode 266, not isolated unit waveforms or synthetic clicks.
The synchronized audio is copied unchanged into each display revision.

## Current video entrance

The raw-camera renderer is now a library in `Utils/ebc_camera.py`, called by
`ebc_video_circle.py`. Set that entrance's recording-specific registration,
full frame clock, VS registration, preserved Motive CSV, and fixed reference
paths before running. It exports the complete recording for every good unit
with synchronized electrode audio. See [EBC video entrances](ebc_videos.md).

```sh
cd ~/Developer/rfmapping
~/.virtualenvs/rfmapping/bin/python ebc_video_circle.py
```

The historical 6-second preview muxed its intermediate silent video with
`output/video_overlay_318/simple_overlay_ccw31_6s/unit318_audio_6s.m4a` by stream copy,
trimmed to the same 6-second duration.
The final `unit318_simple_with_audio.mp4`, PNGs and render manifest are in
the selected output directory. Audio provenance and its
recording-specific export script remain in `output/video_overlay_318/simple/`.
