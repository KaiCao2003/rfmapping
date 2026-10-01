"""Render a minimal raw-camera overlay with head, screen distances and VS bearing."""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from Utils.bearing import bearing
from Utils.ebc_video import _encoder_arguments, _select_encoder
from Utils.ebc_replay import nearest_rows, ray_circle_distance


BEARINGS_DEG = np.array([0., 90., 180., 270.])


def project(points, projection):
    points = np.asarray(points)
    homogeneous = np.concatenate((points, np.ones(points.shape[:-1] + (1,))), axis=-1)
    pixels = homogeneous @ projection.T
    return pixels[..., :2] / pixels[..., 2:3]


def boundary_endpoints(body_xyz, heading_deg, center_xy, radius):
    """Four head-relative rays to the calibrated screen, in raw Motive units."""
    angles = heading_deg + BEARINGS_DEG
    distances = ray_circle_distance(body_xyz[:2], angles, center_xy, radius)
    directions = np.c_[np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles)), np.zeros(4)]
    return body_xyz + distances[:, None] * directions, distances


def motive_vs_geometry(head_xyz, heading_deg, center_xy, radius, screen_deg, vs_zero_deg=0.):
    """Return the wall target and head-relative bearing in Motive XY.

    The caller supplies the HD used for this video. All positions and the radius
    use raw Motive units; positive angles turn from +X toward +Y. RF screen
    angles run in the opposite direction. The independently measured screen
    zero maps VS into Motive; it is not a head-direction correction.
    """
    world_angle = (vs_zero_deg - screen_deg) % 360
    angle = np.deg2rad(world_angle)
    target = np.r_[center_xy + radius * np.array([np.cos(angle), np.sin(angle)]), head_xyz[2]]
    relative_head = np.asarray(head_xyz[:2]) - center_xy
    beta, _ = bearing(*relative_head, radius, heading_deg, world_angle, world_angle)
    return target, beta


def load_motive_head(path):
    """Read preserved Motive position and intrinsic XYZ Euler angles (degrees)."""
    pose = pd.read_csv(path, header=[0, 1, 2, 3])
    head = pose["hp4"].droplevel(0, axis=1)  # Drop rigid-body ID, retaining type/axis.
    result = head["Position"][["X", "Y", "Z"]].copy()
    for axis in "XYZ":
        result[f"rotation_{axis.lower()}_deg"] = head["Rotation"][axis]
    result["hd_deg"] = head["Rotation"]["Z"] % 360
    result.index = pose.iloc[:, 0].to_numpy(dtype=int)
    return result


def fused_yaw(rotation_xyz_deg, reference_quaternion_xyzw):
    """CCW fused yaw relative to the fixed level pose; not yet a world azimuth.

    Matches Analysis.compute_fused_head_direction: Motive uses intrinsic XYZ,
    and the fixed body reference is removed on the right. Fully inverted
    orientations have no unique fused yaw and are left invalid.
    """
    rotation = Rotation.from_euler("XYZ", rotation_xyz_deg, degrees=True)
    reference = Rotation.from_quat(reference_quaternion_xyzw)
    quat = (rotation * reference.inv()).as_quat()
    heading = np.rad2deg(2 * np.arctan2(quat[:, 2], quat[:, 3])) % 360
    heading[np.hypot(quat[:, 2], quat[:, 3]) < 1e-12] = np.nan
    return heading


def render(session, replay_path, registration_path, video_times_path, output_dir,
           *, start=500., duration=6., fps=30., hd_ccw_offset_deg=0., vs_zero_deg=0.,
           motive_csv_path=None, hd_reference_path=None, hd_reference_world_deg=None,
           full_video=False, video_encoder="libx264", sync_valid=None):
    session, output_dir = Path(session), Path(output_dir)
    data = json.loads(Path(replay_path).read_text())
    registration = json.loads(Path(registration_path).read_text())
    projection = np.asarray(registration["world_to_video_projection_raw"], dtype=float)
    video_times = np.load(video_times_path)
    known = np.flatnonzero(np.isfinite(video_times))
    if full_video:
        # Only explicitly audited gaps may be missing; a local excerpt clock
        # must not silently become a full-recording clock.
        sync_valid = np.ones(len(video_times), dtype=bool) if sync_valid is None else np.asarray(sync_valid, dtype=bool)
        if not np.array_equal(sync_valid, np.isfinite(video_times)) or not np.all(np.diff(video_times[known]) > 0):
            raise ValueError("Full-video rendering requires a measured clock or an explicitly audited gap for every decoded AVI frame.")
        source_frames = np.arange(len(video_times))
        query_times = video_times
        adjacent = sync_valid[:-1] & sync_valid[1:]
        fps = 1 / np.median(np.diff(video_times)[adjacent])
        start, duration = float(video_times[0]), len(video_times) / fps
    else:
        query_times = start + np.arange(round(duration * fps)) / fps
        source_frames = known[nearest_rows(video_times[known], query_times)]
        if np.max(abs(video_times[source_frames] - query_times)) > 1 / fps:
            raise ValueError("The requested excerpt exceeds verified video synchronization.")
    raw_rows = np.load(Path(video_times_path).with_name("video_motive_rows.npy"))[source_frames]
    rows = np.where(np.isfinite(raw_rows) & np.isfinite(video_times[source_frames]), raw_rows, -1).astype(int)
    date = session.name.split("_")[0]
    calibration = json.loads((session / f"{date}.calib").read_text())
    # The session-root CSV now contains image-space YOLO poses. This preserved
    # Motive export keeps world XYZ and yaw paired in their original frame IDs.
    motive_csv_path = Path(motive_csv_path) if motive_csv_path else session / "data/processed/trimmed_input.csv"
    selected_pose = load_motive_head(motive_csv_path).reindex(rows)
    body = selected_pose[["X", "Y", "Z"]].to_numpy(float)
    heading = selected_pose["hd_deg"].to_numpy(float)
    reference = None
    if hd_reference_path is not None:
        if hd_ccw_offset_deg != 0 or hd_reference_world_deg is None:
            raise ValueError("Fused yaw requires the reference's world azimuth, without an extra Euler-Z offset.")
        reference = json.loads(Path(hd_reference_path).read_text())
        rotations = selected_pose[[f"rotation_{axis}_deg" for axis in "xyz"]].to_numpy(float)
        valid_rotation = np.isfinite(rotations).all(axis=1)
        relative_heading = np.full(len(rotations), np.nan)
        if valid_rotation.any():
            relative_heading[valid_rotation] = fused_yaw(rotations[valid_rotation], reference["reference_quaternion_xyzw"])
        video_heading = (relative_heading + hd_reference_world_deg) % 360
    else:
        if hd_reference_world_deg is not None:
            raise ValueError("A reference world azimuth requires --hd-reference.")
        relative_heading = np.full(len(heading), np.nan)
        video_heading = (heading + hd_ccw_offset_deg) % 360
    valid_pose = np.isfinite(body).all(axis=1) & np.isfinite(video_heading) & np.isfinite(video_times[source_frames])
    center_xy = np.asarray(calibration["screen_bottom_center"][:2])
    radius = calibration["screen_diamter"] / 2
    raw_units_per_cm = 10 * calibration["motive_units_per_real_mm"]
    trial_onsets = np.array([t["on_s"] for t in data["trials"]])
    unit_id = data["meta"].get("unit_id")
    label = f"Unit {unit_id}" if unit_id is not None else f"Probe {data['meta']['probe']}"
    basename = f"unit{unit_id}_simple" if unit_id is not None else "overlay"
    source_video = session / f"{date}.avi"
    width, source_height, height = 1280, 1024, 1088
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{basename}_silent.mp4"
    first, last = int(source_frames[0]), int(source_frames[-1])
    video_encoder = _select_encoder(video_encoder, width, height, fps)
    # Decode by sequential image index; the AVI header frame rate is incorrect.
    decoder_selection = [] if full_video else ["-vf", f"select=between(n\\,{first}\\,{last})"]
    decoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(source_video), *decoder_selection,
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], stdout=subprocess.PIPE)
    encoder = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an",
        *_encoder_arguments(video_encoder), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)
    next_source, image = first, None
    snapshots = []
    audit_path = output_dir / "frame_geometry.csv"
    audit_partial = output_dir / ".frame_geometry.partial.csv"
    audit_stream = None
    try:
        audit_stream = audit_partial.open("w", newline="")
        for index, (requested, row) in enumerate(zip(source_frames, rows)):
            while next_source <= requested:
                payload = decoder.stdout.read(width * source_height * 3)
                if len(payload) != width * source_height * 3:
                    raise RuntimeError(f"AVI ended before decoded frame {requested}.")
                image = Image.frombytes("RGB", (width, source_height), payload)
                next_source += 1
            time_s, raw_hd, body_xyz = float(video_times[requested]), heading[index], body[index]
            # Video-only heading is computed in Motive before all wall geometry.
            hd = video_heading[index]
            frame = image.copy()
            draw = ImageDraw.Draw(frame)
            if valid_pose[index]:
                position = project(body_xyz, projection)
                tip = project(body_xyz + 180 * np.array([np.cos(np.deg2rad(hd)), np.sin(np.deg2rad(hd)), 0.]), projection)
                heading_vector = tip - position
                endpoints, distances = boundary_endpoints(body_xyz, hd, center_xy, radius)
                endpoint_pixels = project(endpoints, projection)
                for endpoint, distance in zip(endpoint_pixels, distances):
                    if not np.isfinite(distance):
                        continue
                    draw.line([tuple(position), tuple(endpoint)], fill="#39d6e8", width=2)
                    x, y = endpoint
                    draw.ellipse((x-3, y-3, x+3, y+3), fill="#39d6e8")
                    # Put each short distance label along its ray, clear of the head.
                    location = position + .78 * (endpoint - position)
                    text = f"{distance / raw_units_per_cm:.1f}"
                    bounds = draw.textbbox(tuple(location), text, font=font, anchor="mm")
                    draw.rectangle((bounds[0]-3, bounds[1]-2, bounds[2]+3, bounds[3]+2), fill="white")
                    draw.text(tuple(location), text, font=font, anchor="mm", fill="#17212b")
            synced = np.isfinite(time_s)
            trial_index = int(np.searchsorted(trial_onsets, time_s, side="right") - 1) if synced else -1
            trial = data["trials"][trial_index] if trial_index >= 0 else None
            bearing_text = "VS off" if valid_pose[index] else ("Tracking unavailable" if synced else "Sync unavailable")
            visible = trial is not None and time_s < trial["off_s"] and trial["luminance"] != .5
            audit = dict(output_frame=index, video_frame=int(requested), motive_row=int(row),
                         adc_time_s=time_s, raw_hd_deg=float(raw_hd), video_hd_deg=float(hd),
                         fused_yaw_deg=float(relative_heading[index]) if reference else None,
                         video_hd_vs_deg=float((vs_zero_deg - hd) % 360),
                         head_x=float(body_xyz[0]), head_y=float(body_xyz[1]), head_z=float(body_xyz[2]),
                         trial_index=trial_index, vs_visible=visible if synced else None,
                         pose_valid=bool(valid_pose[index]), sync_valid=bool(synced),
                         vs_world_deg=None, target_x=None, target_y=None, bearing_deg=None)
            # Gray stimuli match the background and therefore have no visible bar.
            if visible and valid_pose[index]:
                target, beta = motive_vs_geometry(
                    body_xyz, hd, center_xy, radius, trial["screen_deg"], vs_zero_deg)
                target_pixel = project(target, projection)
                draw.line([tuple(position), tuple(target_pixel)], fill="#ff44d2", width=3)
                x, y = target_pixel
                draw.ellipse((x-5, y-5, x+5, y+5), fill="#ff44d2")
                bearing_text = f"Motive VS bearing {beta:+.1f} deg"
                audit.update(vs_visible=True, vs_world_deg=(vs_zero_deg - trial["screen_deg"]) % 360,
                             target_x=float(target[0]), target_y=float(target[1]), bearing_deg=float(beta))
            if index == 0:
                audit_writer = csv.DictWriter(audit_stream, fieldnames=list(audit), lineterminator="\n")
                audit_writer.writeheader()
            # Match pandas CSV's empty fields for missing pose and sync values.
            audit_writer.writerow({key: "" if pd.isna(value) else value for key, value in audit.items()})
            if valid_pose[index]:
                direction = heading_vector.copy()
                direction /= np.linalg.norm(direction)
                normal = np.array([-direction[1], direction[0]])
                draw.line([tuple(position), tuple(tip)], fill="#00f5b6", width=5)
                draw.polygon([tuple(tip), tuple(tip-15*direction+7*normal), tuple(tip-15*direction-7*normal)], fill="#00f5b6")
                x, y = position
                draw.ellipse((x-6, y-6, x+6, y+6), fill="#00f5b6", outline="black", width=2)
            output = Image.new("RGB", (width, height), "white")
            output.paste(frame, (0, 32))
            draw = ImageDraw.Draw(output)
            time_label = f"{time_s:.2f} s" if synced else f"Frame {requested}"
            draw.text((12, 5), f"{label}  |  {session.name}  |  {time_label}", font=font, fill="#17212b")
            draw.text((width-12, 5), bearing_text, font=font, anchor="rt", fill="#17212b")
            heading_label = "fused HD" if reference else "HD"
            draw.text((12, height-26), f"Green: head / {heading_label}    Cyan: screen-wall distance (cm)    Magenta: VS    Audio: OE electrode", font=font, fill="#17212b")
            if index in (0, len(query_times)//2, len(query_times)-1):
                snapshot = output_dir / f"{basename}_{index:04d}.png"
                output.save(snapshot)
                snapshots.append(str(snapshot))
            encoder.stdin.write(output.tobytes())
            if index % round(fps * 10) == 0:
                print(f"Rendered {index}/{len(query_times)} simple overlay frames", flush=True)
        if full_video and decoder.stdout.read(1):
            raise ValueError("The camera clock ends before the AVI; full rendering requires every decoded frame.")
        if full_video and decoder.wait():
            raise RuntimeError("AVI decoding failed.")
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("Video encoder failed.")
        audit_stream.close()
        audit_partial.replace(audit_path)
    finally:
        if audit_stream is not None:
            audit_stream.close()
        audit_partial.unlink(missing_ok=True)
        decoder.stdout.close()
        if decoder.poll() is None:
            decoder.terminate()
        decoder.wait()
        if not encoder.stdin.closed:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        if encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
    report = dict(video=str(path), snapshots=snapshots, unit_id=unit_id, frames=len(query_times),
                  fps=fps, duration_s=duration, requested_start_adc_s=start,
                  full_video=full_video, missing_pose_frames=int((~valid_pose).sum()),
                  unsynchronized_frames=int((~np.isfinite(video_times[source_frames])).sum()),
                  video_encoder=video_encoder,
                  first_decoded_video_frame=first, last_decoded_video_frame=last,
                  first_adc_time_s=float(video_times[first]), last_adc_time_s=float(video_times[last]),
                  registration=str(registration_path), source_video=str(source_video),
                  motive_pose_source=str(motive_csv_path),
                  hd_ccw_offset_deg=hd_ccw_offset_deg,
                  hd_reference=str(hd_reference_path) if reference else None,
                  hd_reference_quaternion_xyzw=reference["reference_quaternion_xyzw"] if reference else None,
                  hd_reference_world_deg=hd_reference_world_deg,
                  video_hd_definition=("(fused_yaw(R_intrinsic_XYZ @ R_reference.T) + hd_reference_world_deg) % 360"
                                       if reference else "(raw_euler_z + hd_ccw_offset_deg) % 360"),
                  offset_scope="Video only: HD correction precedes Motive geometry; no rotation after projection",
                  bearing_definition="wrap180(atan2(wall_target_y-head_y, wall_target_x-head_x)-video_hd)",
                  vs_world_angle_definition="(vs_zero_deg - Square_PositionX) % 360",
                  hd_in_vs_frame_definition="(vs_zero_deg - video_hd) % 360",
                  origin="Tracked hp4 Position in raw Motive XYZ; no eye offset",
                  wall_center_raw=center_xy.tolist(), wall_radius_raw=radius,
                  screen_zero_deg=vs_zero_deg,
                  vs_angle_source="Video-specific VS registration; source calibration angular zero is unchanged",
                  calibration_limit=("Fused yaw removes tilt and is not a projected nose direction; user confirmed reference faces VS0; VS-world zero is calibrated from visible bars"
                                     if reference else "Euler Z plus the supplied offset is an explicit heading convention, not independently verified anatomical HD"),
                  geometry="Four video-HD-relative rays to calibrated physical screen XY boundary at headplate Z",
                  ebc_bearings_deg=BEARINGS_DEG.tolist(), distance_units="physical cm",
                  tuning_heatmap=False, interactive=False, audio="Mux synchronized OE electrode audio after rendering")
    (output_dir / "render_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report
