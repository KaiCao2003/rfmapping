"""Render EBC overlays and export synchronized electrode-audio videos for good units."""

import argparse
import json
import subprocess
import tempfile
import time
import wave
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from fractions import Fraction
from functools import lru_cache
from multiprocessing import get_context, shared_memory
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, sosfilt, sosfilt_zi
from tqdm.auto import tqdm

from Utils.json_tools import read_formatted_json
from Utils.tuning_curve_utils import get_exposure_timestamps


# Edit these settings, then run this file in the IDE. No notebook state is used.
session = Path("/mnt/senzailab/Kai/#Recording/m20/260918/260918_9")
probe = "A"
phase = "baseline"
arena_type = "cylinder"  # Basler: "rectangle"; Motive session 9: "cylinder".
video_path = None if arena_type == "cylinder" else session / f"{session.name.split('_')[0]}.avi"
video_fps = 30.  # Motive animation only; AVI export retains the original frame rate.
save_path = session / "data/spatial_cells/videos"
video_start_s = 0.
video_duration_s = None  # Full Motive phase or full Basler AVI.
audio_gain = .35  # Fixed OE volume fraction (35%); no per-channel peak normalization.
audio_source = "continuous"  # Real OE voltage; "clicks" plays only sorted spike times.
audio_band_hz = (300., 6000.)  # Spike band; OE Audio Monitor itself uses 100–7000 Hz.
audio_gate_sigma = 3.  # Expand below this multiple of background noise; 0 disables it.
audio_expander_ratio = 4.  # Stronger suppression than OE's 1.2; 1 disables expansion.
video_workers = 8
audio_workers = 4
video_encoder = "auto"  # Prefer NVENC; use 8 CPU encoding threads if unavailable.

BASLER_BOUNDS_PX = (370., 920., 210., 760.)  # left, right, top, bottom
BASLER_SIZE_CM = 41.
RAY_DEG = np.arange(0., 360., 45.)
RAY_COLORS = ("#00a6d6", "#f28e2b", "#29b765", "#ee5971",
              "#ac82e8", "#c4a238", "#4dc7c2", "#da6db5")
HD_COLOR = "#f83ef1"
BOUNDARY_COLOR = "#41e5e5"
CYLINDER_RADII_CM = (15., 30.)  # Inner and outer boundaries in the same movie.
CYLINDER_COLORS = ("#2463ad", "#c26b18")


def load_video_data(session, *, probe="A", phase="baseline", arena_type="rectangle", fps=30.):
    """Load pose and exposure timing without computing statistical controls."""
    if arena_type == "cylinder":
        return _load_cylinder_data(session, probe=probe, phase=phase, fps=fps)
    if arena_type != "rectangle":
        raise ValueError("arena_type must be 'rectangle' or 'cylinder'.")
    session = Path(session)
    directory = session / "data"
    info = read_formatted_json(directory / "session_info.json")["session_info"]
    exposures, origin, timing = get_exposure_timestamps(
        info, directory, camera_ttl_active_high=False,
    )
    pose = pd.read_csv(session / f"{session.name.split('_')[0]}.csv",
                       usecols=["frame", "center_x", "center_y", "hd_deg"]).dropna()
    frames = pose.frame.to_numpy(dtype=int)
    if np.any((frames < 0) | (frames >= len(exposures))):
        raise ValueError("Pose frame IDs exceed the camera exposure timestamps.")
    times = exposures[frames]
    intervals = pd.read_csv(directory / "interval_table.csv")
    interval = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0].to_numpy(float)
    selected = (times >= interval[0]) & (times <= interval[1])
    pose = pose.loc[selected]
    left, right, top, bottom = BASLER_BOUNDS_PX
    xy = np.c_[(pose.center_x - left) * BASLER_SIZE_CM / (right - left),
               (bottom - pose.center_y) * BASLER_SIZE_CM / (bottom - top)]
    kilosort = next((session / "kilosort" / f"Probe{probe}").glob("kilosort_*"))
    return dict(session=session, probe=probe, phase=phase, arena_type="rectangle", xy=xy,
                hd=pose.hd_deg.to_numpy() % 360, frame_ids=frames[selected], times=times[selected],
                exposure_times=exposures, adc_time_origin_s=origin, camera_timing=timing,
                source=dict(kilosort_dir=str(kilosort), selected_interval_s=interval,
                            spike_times_path=str(directory / f"probe{probe}/adc_spike_time.npy")))


def _load_cylinder_data(session, *, probe="A", phase="baseline", fps=30.):
    """Motive XY and world-Z yaw, calibrated exactly as the circular EBC analysis."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motive animation fps must be positive.")
    session = Path(session)
    directory = session / "data"
    pose = pd.read_csv(directory / "processed/filtered.csv", header=[0, 1, 2, 3])
    frames = pose.iloc[:, 0].to_numpy(int)
    xy = pose.xs("Position", level=2, axis=1).to_numpy(float)[:, :2]
    heading = read_formatted_json(directory / "processed/head_direction.json")["hp4"]
    hd = pd.Series(heading["head_direction_deg"], index=heading["frames"]).reindex(frames).to_numpy()
    info = read_formatted_json(directory / "session_info.json")["session_info"]
    exposures, origin, timing = get_exposure_timestamps(info, directory, camera_ttl_active_high=True)
    if np.any((frames < 0) | (frames >= len(exposures))) or np.any(np.diff(frames) <= 0):
        raise ValueError("Motive frame IDs must increase within the exposure timestamp array.")
    times = exposures[frames]
    intervals = pd.read_csv(directory / "interval_table.csv")
    interval = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0].to_numpy(float)
    valid = ((times >= interval[0]) & (times <= interval[1])
             & np.isfinite(xy).all(axis=1) & np.isfinite(hd))
    if not valid.any():
        raise ValueError(f"No finite Motive XY/HD in phase {phase}.")
    # Fit once from every valid pose in the phase, before reducing the display
    # rate or selecting a clip. Inner/outer geometry then shares one cm scale.
    center = (xy[valid].min(axis=0) + xy[valid].max(axis=0)) / 2
    scale = CYLINDER_RADII_CM[0] / np.linalg.norm(xy[valid] - center, axis=1).max()
    frame_times = interval[0] + np.arange(int(np.ceil(np.diff(interval)[0] * fps))) / fps
    nearest = _nearest_pose_rows(times, frame_times)
    selected = valid[nearest] & (abs(times[nearest] - frame_times) <= 1.5 * np.median(np.diff(exposures)))
    rows = nearest[selected]
    kilosort = next((session / "kilosort" / f"Probe{probe}").glob("kilosort_*"))
    return dict(session=session, probe=probe, phase=phase, arena_type="cylinder",
                xy=(xy[rows] - center) * scale, hd=hd[rows] % 360,
                frame_ids=np.flatnonzero(selected), times=frame_times[selected], pose_frame_ids=frames[rows],
                exposure_times=exposures, adc_time_origin_s=origin, camera_timing=timing,
                fps=float(fps), total_frames=len(frame_times), video_time_origin_s=float(interval[0]),
                boundary_radii_cm=CYLINDER_RADII_CM, center_raw=center, cm_per_unit=scale,
                source=dict(kilosort_dir=str(kilosort), selected_interval_s=interval,
                            spike_times_path=str(directory / f"probe{probe}/adc_spike_time.npy")))


def _nearest_pose_rows(times, targets):
    """Nearest recorded pose; do not interpolate headings through missing tracking."""
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(targets - times[left]) <= abs(times[right] - targets), left, right)


def cylinder_geometry(data):
    """Forward ray-circle intersections, independently for both enclosing walls.

    Distances have shape (poses, boundaries, bearings). Motive yaw is CCW from
    +X; screen Y is inverted only when drawing, never in the EBC calculation.
    """
    xy, hd = np.asarray(data["xy"]), np.asarray(data["hd"])
    radii = np.asarray(data["boundary_radii_cm"])
    angles = np.deg2rad(hd[:, None] + RAY_DEG)
    direction = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    projection = np.sum(xy[:, None, :] * direction, axis=-1)
    norm2 = np.sum(xy * xy, axis=-1)
    valid = (np.isfinite(hd)[:, None] & np.isfinite(xy).all(axis=1)[:, None]
             & (norm2[:, None] <= radii[None, :]**2 + 1e-9))
    discriminant = projection[:, None, :]**2 + radii[None, :, None]**2 - norm2[:, None, None]
    distance = np.maximum(-projection[:, None, :] + np.sqrt(np.maximum(discriminant, 0)), 0)
    distance[~valid] = np.nan
    endpoints = xy[:, None, None, :] + distance[..., None] * direction[:, None, :, :]
    return endpoints, distance, valid


def overlay_geometry(data):
    """Invert the analysis calibration; distances are untruncated cm, not bins."""
    xy, hd = np.asarray(data["xy"]), np.asarray(data["hd"])
    left, right, top, bottom = BASLER_BOUNDS_PX
    pixels_per_cm = np.array([right - left, bottom - top]) / BASLER_SIZE_CM
    positions = xy * pixels_per_cm * [1, -1] + [left, bottom]
    valid = (np.isfinite(hd) & np.isfinite(xy).all(axis=1)
             & ((xy >= 0) & (xy <= BASLER_SIZE_CM)).all(axis=1))
    angles = np.deg2rad(hd[:, None] + RAY_DEG)
    directions = np.stack((-np.sin(angles), -np.cos(angles)), axis=-1)
    # Pixel displacement per cm along each ray; the nearest positive wall
    # intersection is its untruncated boundary distance in cm.
    step = directions * pixels_per_cm
    walls = np.where(step > 0, [right, bottom], [left, top])
    distances = np.min(np.divide(walls - positions[:, None, :], step,
                                 out=np.full_like(step, np.inf), where=np.abs(step) > 1e-12), axis=-1)
    distances[~valid] = np.nan
    endpoints = positions[:, None, :] + directions * distances[..., None] * pixels_per_cm
    return positions, endpoints, distances, valid


class _Overlay:
    def __init__(self, data, width, height, fps):
        self.data, self.width, self.height = data, width, height
        self.fps = fps
        self.positions, self.endpoints, self.distances, self.valid = overlay_geometry(data)
        self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        self.small = ImageFont.truetype("DejaVuSans.ttf", 16)
        self.heading = ImageFont.truetype("DejaVuSans.ttf", 26)
        self.size = (width + 370, height)
        self.template = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(self.template)
        x = width + 20
        # Rasterize fixed panel text once per worker, not once per video frame.
        for y, text, font in (
            (24, "EBC geometry", self.heading),
            (66, f"{data['session'].name} / {data['phase']}", self.font),
            (224, "Mouse position (cm)", self.font),
            (340, "HD: 0° up / north, positive CCW", self.small),
            (410, "8 EBC boundary rays", self.font),
            (446, "Bearing from HD        Distance", self.small),
            (882, "0° front · 90° left · 180° back", self.small),
            (910, "270° right · distances in cm", self.small),
            (958, "Aligned by zero-based CSV frame ID", self.small),
        ):
            draw.text((x, y), text, font=font, fill="#17212b")
        for y in (202, 386):
            draw.line((x, y, x + 330, y), fill="#cbd2d9", width=1)
        draw.line((x, 312, x + 28, 312), fill=HD_COLOR, width=5)
        for i, (angle, color) in enumerate(zip(RAY_DEG, RAY_COLORS)):
            y = 492 + i * 41
            draw.line((x, y + 10, x + 28, y + 10), fill=color, width=4)
            draw.text((x + 40, y), f"{angle:3.0f}°", font=self.font, fill="#17212b")

    @lru_cache(maxsize=256)
    def _label(self, text):
        bounds = self.small.getbbox(text, anchor="mm")
        tile = Image.new("RGB", (bounds[2] - bounds[0] + 8, bounds[3] - bounds[1] + 6), "white")
        ImageDraw.Draw(tile).text((4 - bounds[0], 3 - bounds[1]), text,
                                  font=self.small, fill="#17212b", anchor="mm")
        return tile, (bounds[0] - 4, bounds[1] - 3)

    def draw(self, frame, frame_id, row):
        canvas = self.template.copy()
        canvas.paste(frame, (0, 0))
        draw = ImageDraw.Draw(canvas)

        def label(point, text):
            tile, offset = self._label(text)
            canvas.paste(tile, (round(point[0] + offset[0]), round(point[1] + offset[1])))

        left, right, top, bottom = BASLER_BOUNDS_PX
        draw.rectangle((left, top, right, bottom), outline="#111111", width=6)
        draw.rectangle((left, top, right, bottom), outline=BOUNDARY_COLOR, width=3)
        label(((left + right) / 2, top - 55), f"Arena boundary: {BASLER_SIZE_CM:g} × {BASLER_SIZE_CM:g} cm")
        distances = np.full(len(RAY_DEG), np.nan)
        if row >= 0:
            position = self.positions[row]
            hd = self.data["hd"][row]
            distances = self.distances[row]
            if self.valid[row]:
                for angle, end, color in zip(RAY_DEG, self.endpoints[row], RAY_COLORS):
                    line = (tuple(position), tuple(end))
                    draw.line(line, fill="#111111", width=5)
                    draw.line(line, fill=color, width=3)
                    draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=color)
                    # Label the intersection just outside its wall, away from the mouse.
                    offset = np.array([0., 0.])
                    if abs(end[0] - left) < 1e-6:
                        offset[0] = -24
                    elif abs(end[0] - right) < 1e-6:
                        offset[0] = 24
                    if abs(end[1] - top) < 1e-6:
                        offset[1] = -18
                    elif abs(end[1] - bottom) < 1e-6:
                        offset[1] = 18
                    label(tuple(end + offset), f"{angle:g}°")
            direction = np.array([-np.sin(np.deg2rad(hd)), -np.cos(np.deg2rad(hd))])
            tip = position + 70 * direction
            normal = np.array([-direction[1], direction[0]])
            draw.line((tuple(position), tuple(tip)), fill="white", width=9)
            draw.line((tuple(position), tuple(tip)), fill=HD_COLOR, width=5)
            draw.polygon([tuple(tip), tuple(tip - 16 * direction + 8 * normal),
                          tuple(tip - 16 * direction - 8 * normal)], fill=HD_COLOR)
            label(tuple(tip + 26 * direction), f"HD {hd:.1f}°")
            x, y = position
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#ff4040", outline="white", width=2)

        x = self.width + 20
        draw.text((x, 108), f"AVI frame {frame_id}", font=self.small, fill="#17212b")
        draw.text((x, 136), f"Video: {frame_id / self.fps:.2f} s", font=self.small, fill="#17212b")
        adc_text = f"{self.data['times'][row]:.3f} s" if row >= 0 else "—"
        draw.text((x, 164), f"ADC time: {adc_text}", font=self.small, fill="#17212b")
        xy = self.data["xy"][row] if row >= 0 else [np.nan, np.nan]
        position_text = f"x {xy[0]:5.2f}    y {xy[1]:5.2f}" if row >= 0 else "No valid pose in selected phase"
        draw.text((x, 260), position_text, font=self.small, fill="#17212b")
        heading_text = f"HD: {self.data['hd'][row]:.1f}°" if row >= 0 else "HD: —"
        draw.text((x + 40, 297), heading_text, font=self.heading, fill="#17212b")
        for i, distance in enumerate(distances):
            y = 492 + i * 41
            value = f"{distance:5.2f} cm" if np.isfinite(distance) else "—"
            draw.text((x + 205, y), value, font=self.font, fill="#17212b")
        status = "Geometry valid" if row >= 0 and self.valid[row] else (
            "Outside arena: EBC omitted" if row >= 0 else "Pose missing: overlay omitted")
        draw.text((x, 840), status, font=self.small, fill="#17212b" if row >= 0 and self.valid[row] else "#b3261e")
        return canvas


class _CylinderOverlay(_Overlay):
    """One top-down view with both boundaries and paired distance columns."""

    def __init__(self, data, width, height, fps):
        self.data, self.width, self.height, self.fps = data, width, height, fps
        self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        self.small = ImageFont.truetype("DejaVuSans.ttf", 16)
        self.heading = ImageFont.truetype("DejaVuSans.ttf", 26)
        self.size = (width + 370, height)
        self.center = np.array([width / 2, height / 2])
        self.radii = np.asarray(data["boundary_radii_cm"])
        self.scale = (min(width, height) / 2 - 112) / self.radii[-1]
        self.positions = self.center + np.asarray(data["xy"]) * self.scale * [1, -1]
        endpoints, self.distances, self.boundary_valid = cylinder_geometry(data)
        self.endpoints = self.center + endpoints * self.scale * [1, -1]
        self.valid = self.boundary_valid.all(axis=1)
        self.template = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(self.template)
        draw.text((32, 24), "Cylinder EBC · Motive top view", font=self.heading, fill="#17212b")
        for index, (radius, color) in enumerate(zip(self.radii, CYLINDER_COLORS)):
            r = radius * self.scale
            cx, cy = self.center
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=4)
            name = ("Inner", "Outer")[index]
            draw.text((32 + index * 260, 66), f"{name}: diameter {2 * radius:g} cm",
                      font=self.font, fill=color)
        for cm in (-30, -15, 0, 15, 30):
            x = self.center[0] + cm * self.scale
            y = self.center[1] - cm * self.scale
            draw.text((x, height - 72), f"{cm:g}", anchor="mt", font=self.small, fill="#34404b")
            draw.text((self.center[0] - self.radii[-1] * self.scale - 62, y),
                      f"{cm:g}", anchor="mm", font=self.small, fill="#34404b")
        draw.text((width / 2, height - 40), "Motive X (cm)", anchor="mt", font=self.font, fill="#17212b")
        draw.text((32, height / 2), "Y (cm)", font=self.font, fill="#17212b")
        x = width + 20
        for y, text, font in (
            (24, "Two EBC boundaries", self.heading),
            (66, f"{data['session'].name} / {data['phase']}", self.font),
            (224, "Mouse position (cm)", self.font),
            (340, "HD: 0° = +X / right, positive CCW", self.small),
            (410, "8 rays × 2 boundaries", self.font),
            (884, "● inner hit   ○ outer hit", self.small),
            (914, "0° front · 90° left · 180° back", self.small),
            (958, "Motive pose · synchronized OE audio", self.small),
        ):
            draw.text((x, y), text, font=font, fill="#17212b")
        for y in (202, 386):
            draw.line((x, y, x + 330, y), fill="#cbd2d9", width=1)
        draw.line((x, 312, x + 28, 312), fill=HD_COLOR, width=5)
        draw.text((x, 453), "Bearing", font=self.small, fill="#17212b")
        draw.text((x + 122, 453), "Inner cm", font=self.small, fill=CYLINDER_COLORS[0])
        draw.text((x + 237, 453), "Outer cm", font=self.small, fill=CYLINDER_COLORS[1])
        for i, (angle, color) in enumerate(zip(RAY_DEG, RAY_COLORS)):
            y = 496 + i * 41
            draw.line((x, y + 10, x + 20, y + 10), fill=color, width=4)
            draw.text((x + 30, y), f"{angle:3.0f}°", font=self.font, fill="#17212b")

    def draw(self, frame, frame_id, row):
        canvas = self.template.copy()
        draw = ImageDraw.Draw(canvas)

        def label(point, text):
            tile, offset = self._label(text)
            canvas.paste(tile, (round(point[0] + offset[0]), round(point[1] + offset[1])))

        if row >= 0:
            position = self.positions[row]
            # Outer rays first; the thicker inner segment and distinct hit markers
            # keep both distances visible along each common bearing.
            for boundary in (1, 0):
                if not self.boundary_valid[row, boundary]:
                    continue
                for angle, end, color in zip(RAY_DEG, self.endpoints[row, boundary], RAY_COLORS):
                    draw.line((tuple(position), tuple(end)), fill=color, width=2 if boundary else 4)
                    x, y = end
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="white" if boundary else color,
                                 outline=color, width=2)
                    if boundary:
                        outward = (end - self.center) / np.linalg.norm(end - self.center)
                        label(end + 22 * outward, f"{angle:g}°")
            hd = self.data["hd"][row]
            direction = np.array([np.cos(np.deg2rad(hd)), -np.sin(np.deg2rad(hd))])
            tip = position + 70 * direction
            normal = np.array([-direction[1], direction[0]])
            draw.line((tuple(position), tuple(tip)), fill="white", width=9)
            draw.line((tuple(position), tuple(tip)), fill=HD_COLOR, width=5)
            draw.polygon([tuple(tip), tuple(tip - 16 * direction + 8 * normal),
                          tuple(tip - 16 * direction - 8 * normal)], fill=HD_COLOR)
            label(tip + 30 * direction, f"HD {hd:.1f}°")
            x, y = position
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill="#ff4040", outline="white", width=2)
        x = self.width + 20
        pose_frame = str(self.data["pose_frame_ids"][row]) if row >= 0 else "—"
        draw.text((x, 108), f"Motive frame {pose_frame}", font=self.small, fill="#17212b")
        draw.text((x, 136), f"Video: {frame_id / self.fps:.2f} s", font=self.small, fill="#17212b")
        adc_time = self.data["video_time_origin_s"] + frame_id / self.fps
        draw.text((x, 164), f"ADC time: {adc_time:.3f} s", font=self.small, fill="#17212b")
        xy = self.data["xy"][row] if row >= 0 else [np.nan, np.nan]
        text = f"x {xy[0]:5.2f}    y {xy[1]:5.2f}" if row >= 0 else "No valid Motive pose"
        draw.text((x, 260), text, font=self.small, fill="#17212b")
        text = f"HD: {self.data['hd'][row]:.1f}°" if row >= 0 else "HD: —"
        draw.text((x + 40, 297), text, font=self.heading, fill="#17212b")
        distances = self.distances[row] if row >= 0 else np.full((2, len(RAY_DEG)), np.nan)
        for boundary, column in enumerate((122, 237)):
            for i, distance in enumerate(distances[boundary]):
                text = f"{distance:5.2f}" if np.isfinite(distance) else "—"
                draw.text((x + column, 496 + i * 41), text, font=self.font, fill=CYLINDER_COLORS[boundary])
        status = "Both boundaries valid" if row >= 0 and self.valid[row] else (
            "Outside boundary: ray omitted" if row >= 0 else "Tracking gap: pose omitted")
        draw.text((x, 840), status, font=self.small, fill="#17212b" if row >= 0 and self.valid[row] else "#b3261e")
        return canvas


def _make_overlay(data, width, height, fps):
    cls = _CylinderOverlay if data.get("arena_type") == "cylinder" else _Overlay
    return cls(data, width, height, fps)


_render_state = None


def _initialize_renderer(data, width, height, fps, memory_name):
    global _render_state
    memory = shared_memory.SharedMemory(name=memory_name)
    overlay = _make_overlay(data, width, height, fps)
    input_bytes = 0 if data.get("arena_type") == "cylinder" else width * height * 3
    _render_state = memory, overlay, input_bytes, overlay.size[0] * height * 3


def _render_shared_frame(slot, frame_id, row):
    memory, overlay, input_bytes, output_bytes = _render_state
    offset = slot * (input_bytes + output_bytes)
    frame = None
    if input_bytes:
        pixels = np.ndarray((overlay.height, overlay.width, 3), dtype=np.uint8,
                            buffer=memory.buf, offset=offset)
        frame = Image.fromarray(pixels)
    canvas = overlay.draw(frame, frame_id, row)
    start = offset + input_bytes
    memory.buf[start:start + output_bytes] = canvas.tobytes()


class _SharedRenderer:
    """Bounded frame slots: decoded once, drawn in parallel, encoded in order."""

    def __init__(self, data, width, height, fps, workers):
        self.slots = workers * 2
        self.input_bytes = 0 if data.get("arena_type") == "cylinder" else width * height * 3
        self.size = (width + 370, height)
        self.output_bytes = self.size[0] * height * 3
        self.stride = self.input_bytes + self.output_bytes
        self.memory = shared_memory.SharedMemory(create=True, size=self.slots * self.stride)
        # No spike-count matrices are copied to drawing workers.
        pose = {key: data[key] for key in ("session", "phase", "times", "xy", "hd", "arena_type",
                                          "pose_frame_ids", "video_time_origin_s", "boundary_radii_cm")
                if key in data}
        try:
            self.pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize_renderer,
                initargs=(pose, width, height, fps, self.memory.name),
            )
        except BaseException:
            self.memory.close()
            self.memory.unlink()
            raise

    def read(self, stream, slot):
        start = slot * self.stride
        with self.memory.buf[start:start + self.input_bytes] as target:
            count = 0
            while count < len(target):
                received = stream.readinto(target[count:])
                if not received:
                    break
                count += received
        return count

    def write(self, stream, slot, preview=None):
        start = slot * self.stride + self.input_bytes
        with self.memory.buf[start:start + self.output_bytes] as pixels:
            stream.write(pixels)
            if preview is not None:
                Image.frombytes("RGB", self.size, bytes(pixels)).save(preview)

    def close(self):
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.memory.close()
            self.memory.unlink()


def _encoder_arguments(video_encoder):
    if video_encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    if video_encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "8"]
    raise ValueError("video_encoder must be 'h264_nvenc' or 'libx264'.")


def _select_encoder(requested, width, height, fps):
    """Check runtime initialization, not just ffmpeg's compiled encoder list."""
    if requested not in ("auto", "h264_nvenc", "libx264"):
        raise ValueError("video_encoder must be 'auto', 'h264_nvenc', or 'libx264'.")
    if requested == "libx264":
        return requested
    # Use the actual output geometry. The one-frame probe writes no video file
    # and catches missing driver libraries before decoding the recording.
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-f", "lavfi",
         "-i", f"color=c=white:s={width}x{height}:r={fps}", "-frames:v", "1", "-an",
         *_encoder_arguments("h264_nvenc"), "-pix_fmt", "yuv420p", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return "h264_nvenc"
    reason = result.stderr.strip() or f"ffmpeg exited with code {result.returncode}"
    if requested == "h264_nvenc":
        raise RuntimeError(f"NVENC initialization failed:\n{reason}\n"
                           "Use video_encoder='auto' for a multithreaded CPU fallback.")
    print(f"NVENC unavailable; using libx264 with 8 CPU threads.\n{reason}", flush=True)
    return "libx264"


def _ffmpeg_error(process, log, operation):
    """Read stderr from a temporary file, which cannot fill a pipe and deadlock."""
    process.wait()
    log.seek(0)
    detail = log.read().decode("utf-8", errors="replace").strip()
    return RuntimeError(f"{operation} failed (ffmpeg exit {process.returncode}):\n"
                        f"{detail or 'ffmpeg produced no error details.'}")


def export_ebc_overlay(data, video_path, output_path, *, start_s=0., duration_s=None,
                       workers=1, video_encoder="libx264"):
    """Export a Basler overlay or a Motive cylinder animation with both walls.

    Clip times start at AVI time zero or at the selected Motive phase start.
    Missing tracking stays in the movie as a labeled gap. Multiple workers
    share bounded frame buffers; the common picture is encoded only once.
    """
    cylinder = data.get("arena_type") == "cylinder"
    video_path = Path(video_path) if video_path is not None else None
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4" or (video_path is not None and video_path.resolve() == output_path.resolve()):
        raise ValueError("Choose a separate .mp4 output path.")
    if start_s < 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("start_s must be nonnegative and duration_s must be positive or None.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if cylinder:
        if video_path is not None:
            raise ValueError("Use video_path=None for the Motive cylinder animation.")
        fps, total = data["fps"], data["total_frames"]
        fps_text = str(fps)
        width, height = 1280, 1024
    else:
        if video_path is None:
            raise ValueError("A Basler AVI is required for arena_type='rectangle'.")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video_path)],
            check=True, capture_output=True, text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        fps_text = stream["avg_frame_rate"]
        fps = float(Fraction(fps_text))
        width, height, total = int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])
        if (width, height) != (1280, 1024):
            raise ValueError("Expected the original Basler frame size (1280 × 1024), without cropping/resizing.")
    frame_ids = np.asarray(data["frame_ids"], dtype=int)
    if len(frame_ids) != len(data["times"]) or np.any(np.diff(frame_ids) <= 0) or np.any((frame_ids < 0) | (frame_ids >= total)):
        raise ValueError("Pose frame IDs must be increasing zero-based indices into the video timeline.")
    first = int(round(start_s * fps))
    stop = total if duration_s is None else min(total, first + int(round(duration_s * fps)))
    if not 0 <= first < stop:
        raise ValueError("The requested clip contains no video frames.")
    video_encoder = _select_encoder(video_encoder, width + 370, height, fps_text)
    codec_args = _encoder_arguments(video_encoder)
    workers = min(workers, stop - first)
    rows = np.full(total, -1, dtype=int)
    rows[frame_ids] = np.arange(len(frame_ids))
    overlay = _make_overlay(data, width, height, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial.mp4")
    preview = output_path.with_suffix(".png")
    preview_frame = min(stop - 1, first + int(20 * fps))
    # AVI seeking can skip the initial GOP even for -ss 0. Decode in order
    # and trim by frame ordinal so that CSV frame IDs remain exact.
    decoder_command = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-filter_threads", "2",
        "-i", str(video_path), "-map", "0:v:0",
        "-vf", f"trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS",
        "-fps_mode", "passthrough", "-frames:v", str(stop - first),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "2", "pipe:1",
    ]
    decoder, encoder, renderer = None, None, None
    decoder_log, encoder_log = None, None
    pending = deque()
    started = time.monotonic()

    def report(done):
        if done % 1000 == 0 or done == stop - first:
            print(f"Overlay: {done:,}/{stop - first:,} frames ({done / (time.monotonic() - started):.1f} frames/s)", flush=True)

    def write_next():
        future, slot, frame_id = pending.popleft()
        future.result()
        renderer.write(encoder.stdin, slot, preview if frame_id in (first, preview_frame) else None)
        report(frame_id - first + 1)

    try:
        encoder_log = tempfile.TemporaryFile()
        if not cylinder:
            decoder_log = tempfile.TemporaryFile()
            decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=decoder_log)
        if workers > 1:
            renderer = _SharedRenderer(data, width, height, fps, workers)
        print(f"Overlay: {workers} drawing workers, encoder={video_encoder}", flush=True)
        encoder = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{overlay.size[0]}x{overlay.size[1]}", "-r", fps_text, "-i", "pipe:0", "-an",
             *codec_args,
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(partial)], stdin=subprocess.PIPE, stderr=encoder_log,
        )
        for frame_id in range(first, stop):
            if renderer is not None:
                slot = (frame_id - first) % renderer.slots
                if len(pending) == renderer.slots:
                    write_next()
            if not cylinder:
                if renderer is not None:
                    count = renderer.read(decoder.stdout, slot)
                else:
                    payload = decoder.stdout.read(width * height * 3)
                    count = len(payload)
                # A clean EOF handles the Basler AVI's one-frame header overcount.
                if not count and decoder.wait() == 0 and frame_id > first:
                    stop = frame_id
                    break
                if count != width * height * 3:
                    if decoder.wait() != 0:
                        raise _ffmpeg_error(decoder, decoder_log, "AVI decoding")
                    raise RuntimeError(f"AVI decode stopped at frame {frame_id}; expected {stop} frames.")
            if renderer is not None:
                future = renderer.pool.submit(_render_shared_frame, slot, frame_id, int(rows[frame_id]))
                pending.append((future, slot, frame_id))
            else:
                frame = None if cylinder else Image.frombytes("RGB", (width, height), payload)
                canvas = overlay.draw(frame, frame_id, int(rows[frame_id]))
                encoder.stdin.write(canvas.tobytes())
                if frame_id in (first, preview_frame):
                    canvas.save(preview)
                report(frame_id - first + 1)
        while pending:
            write_next()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise _ffmpeg_error(encoder, encoder_log, f"{video_encoder} encoding")
        if decoder is not None and decoder.wait() != 0:
            raise _ffmpeg_error(decoder, decoder_log, "AVI decoding")
        partial.replace(output_path)
    except BrokenPipeError as error:
        raise _ffmpeg_error(encoder, encoder_log, f"{video_encoder} encoding") from error
    finally:
        for process in (decoder, encoder):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
        if decoder is not None:
            decoder.stdout.close()
        if encoder is not None and not encoder.stdin.closed:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        try:
            if renderer is not None:
                renderer.close()
        finally:
            for log in (decoder_log, encoder_log):
                if log is not None:
                    log.close()
            partial.unlink(missing_ok=True)
    selected = rows[first:stop]
    available = selected >= 0
    geometry_info = (dict(boundary_radii_cm=list(data["boundary_radii_cm"]),
                          center_raw=np.asarray(data["center_raw"]).tolist(), cm_per_unit=data["cm_per_unit"],
                          angle_convention="Motive XY: HD CCW from +X; ray bearing CCW relative to HD")
                     if cylinder else dict(bounds_px=BASLER_BOUNDS_PX, arena_size_cm=BASLER_SIZE_CM,
                                           angle_convention="HD: north-zero CCW; ray bearing: CCW relative to HD"))
    summary = dict(video=str(output_path), source_video=str(video_path) if video_path is not None else None,
                   preview=str(preview), arena_type="cylinder" if cylinder else "rectangle",
                   session=str(data["session"]), phase=data["phase"],
                   **geometry_info,
                   ray_bearings_deg=RAY_DEG.tolist(),
                   frames=stop - first, fps=fps, container_frame_count=total,
                   drawing_workers=workers, video_encoder=video_encoder,
                   duration_s=(stop - first) / fps, source_first_frame=first, source_last_frame=stop - 1,
                   missing_pose_frames=int((~available).sum()),
                   outside_arena_frames=int((~overlay.valid[selected[available]]).sum()),
                   render_seconds=round(time.monotonic() - started, 2))
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def export_good_unit_videos(data, video_path, save_path, *, start_s=0., duration_s=None, gain=.35,
                            workers=8, audio_workers=4, video_encoder="auto",
                            audio_source="continuous", audio_band_hz=(300., 6000.), audio_gate_sigma=3.,
                            audio_expander_ratio=4.):
    """Save save_path/<unit_id>.mp4 for every good unit.

    Continuous audio is its main electrode's bandpassed voltage, including other
    units and background on that electrode. It is not an isolated sorted unit.
    Gain is the OE volume fraction, independent of clip/channel peak amplitude.
    The soft expander suppresses background and also attenuates weak spikes.
    """
    save_path = Path(save_path).expanduser()
    if workers < 1 or audio_workers < 1:
        raise ValueError("workers and audio_workers must be at least 1.")
    if not 0 < gain <= 1:
        raise ValueError("gain must be between 0 (exclusive) and 1.")
    if audio_source not in ("continuous", "clicks"):
        raise ValueError("audio_source must be 'continuous' or 'clicks'.")
    if not np.isfinite(audio_gate_sigma) or audio_gate_sigma < 0:
        raise ValueError("audio_gate_sigma must be nonnegative (0 disables the gate).")
    if not np.isfinite(audio_expander_ratio) or audio_expander_ratio < 1:
        raise ValueError("audio_expander_ratio must be at least 1 (1 disables expansion).")
    if audio_source == "continuous":
        # Resolve channels and raw input before doing the expensive video render.
        source = _open_ephys_source(data["session"], data["probe"])
        unit_channels = _good_unit_channels(data["source"]["kilosort_dir"])
        good_units = list(unit_channels)
    save_path.mkdir(parents=True, exist_ok=True)
    # Keep the shared picture on local scratch storage; do not read the same
    # video from a network recording directory once per unit.
    with tempfile.TemporaryDirectory(prefix="ebc_batch_") as directory:
        temporary = Path(directory)
        overlay = export_ebc_overlay(
            data, video_path, temporary / "overlay.mp4", start_s=start_s, duration_s=duration_s,
            workers=workers, video_encoder=video_encoder,
        )
        duration = overlay["frames"] / overlay["fps"]
        if audio_source == "continuous":
            # Invert the same exposure clock used for sorted spikes. Do not zero
            # the probe and ADC clocks independently: OE already synchronized them.
            clock = _video_audio_clock(data, overlay)
            channels = sorted(set(unit_channels.values()))
            print(f"Reading {len(channels)} electrode channels once for {len(good_units)} good units")
            tracks = _cache_continuous_audio(source, channels, clock, duration, temporary,
                                            band_hz=audio_band_hz)
            units_by_channel = {channel: [] for channel in channels}
            for unit, channel in unit_channels.items():
                units_by_channel[channel].append(unit)
        else:
            good_units, spike_seconds, _ = _load_audio_spikes(data, overlay)
        pool = ProcessPoolExecutor(max_workers=audio_workers, mp_context=get_context("spawn"))
        try:
            if audio_source == "continuous":
                futures = {pool.submit(
                    _export_channel_videos, overlay["video"], save_path,
                    units, data["probe"], tracks[channel], duration, gain, audio_gate_sigma,
                    audio_expander_ratio,
                ): len(units) for channel, units in units_by_channel.items()}
            else:
                futures = {pool.submit(
                    _mux_spike_audio, overlay["video"], save_path / f"{unit}.mp4",
                    int(unit), data["probe"], spike_seconds[int(unit)], duration, gain,
                ): 1 for unit in good_units}
            with tqdm(total=len(good_units), desc="Good-unit videos", unit="unit") as progress:
                for future in as_completed(futures):
                    future.result()
                    progress.update(futures[future])
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    videos = [save_path / f"{unit}.mp4" for unit in good_units]
    print(f"Saved {len(videos)} videos to {save_path.resolve()}")
    return videos


def _video_audio_clock(data, video_result):
    """Clip seconds → common OE seconds, for either AVI or real-time animation."""
    fps, first = video_result["fps"], video_result["source_first_frame"]
    origin = data["adc_time_origin_s"]
    if data.get("arena_type") == "cylinder":
        # The animation uses elapsed recording time. Do not compress a camera
        # pause into a single nominal frame interval as an AVI timeline would.
        seconds = np.array([0., video_result["frames"] / fps])
        return seconds, origin + data["video_time_origin_s"] + first / fps + seconds
    return ((np.arange(len(data["exposure_times"])) - first) / fps,
            np.asarray(data["exposure_times"]) + origin)


def _open_ephys_source(session, probe):
    """Read the raw session stream, not Kilosort's concatenated/preprocessed input."""
    info = read_formatted_json(Path(session) / "data/session_info.json")["session_info"]
    structure_path = Path(info["session_name"])
    structure = json.loads(structure_path.read_text())
    folder = info[f"continuous_probe_{probe}_folder"]
    stream = next(item for item in structure["continuous"] if item["folder_name"].rstrip("/") == folder)
    directory = structure_path.parent / "continuous" / folder
    timestamps = directory / "timestamps.npy"
    times = np.load(timestamps, mmap_mode="r")
    binary = directory / "continuous.dat"
    channels = int(stream["num_channels"])
    if times.dtype.kind != "f":
        raise ValueError("Audio requires synchronized OE timestamps.npy in seconds (OE 0.6+).")
    if binary.stat().st_size != len(times) * channels * 2:
        raise ValueError(f"{binary}: int16 sample count does not match timestamps and channel count.")
    return dict(binary=binary, timestamps=timestamps, sample_rate=float(stream["sample_rate"]),
                num_channels=channels, bit_volts=np.array([ch["bit_volts"] for ch in stream["channels"]]))


def _good_unit_channels(kilosort):
    """Map each good cluster to the peak channel of its dominant, unwhitened template."""
    kilosort = Path(kilosort)
    labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t")
    units = np.sort(labels.loc[labels.KSLabel == "good", "cluster_id"].to_numpy(int))
    if not len(units):
        raise ValueError(f"No good units in {kilosort}.")
    templates = np.load(kilosort / "templates.npy", mmap_mode="r")
    channel_map = np.load(kilosort / "channel_map.npy").ravel()
    whitening_inv = np.load(kilosort / "whitening_mat_inv.npy")
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").ravel()
    template_ids = np.load(kilosort / "spike_templates.npy", mmap_mode="r").ravel()
    counts = np.zeros((len(units), len(templates)), dtype=np.int64)
    original_ids = True
    # One bounded pass over all spikes; cluster IDs can differ from template IDs
    # after curation. Avoid scanning millions of spikes separately for every unit.
    for start in range(0, len(clusters), 1_000_000):
        cluster = clusters[start:start + 1_000_000]
        template = template_ids[start:start + 1_000_000]
        original_ids &= np.array_equal(cluster, template)
        rows = np.searchsorted(units, cluster)
        selected = units[np.minimum(rows, len(units) - 1)] == cluster
        codes = rows[selected] * len(templates) + template[selected]
        counts += np.bincount(codes, minlength=counts.size).reshape(counts.shape)
    dominant = counts.argmax(axis=1)
    absent = ~counts.any(axis=1)
    if absent.any():
        # Session slices retain all original KS labels/templates, even if a unit
        # never fired in this session. Original cluster IDs are template indices.
        if not original_ids or np.any((units[absent] < 0) | (units[absent] >= len(templates))):
            raise ValueError(f"No template assignment for silent curated units {units[absent].tolist()}.")
        dominant[absent] = units[absent]
    template_indices = np.unique(dominant)
    selected = np.asarray(templates[template_indices])
    # One matrix product avoids starting a threaded BLAS operation per unit.
    waveforms = (selected.reshape(-1, selected.shape[-1]) @ whitening_inv).reshape(selected.shape)
    peak_channels = channel_map[np.ptp(waveforms, axis=1).argmax(axis=1)]
    peaks = dict(zip(template_indices, peak_channels))
    return {int(unit): int(peaks[template]) for unit, template in zip(units, dominant)}


def _clock_times(seconds, clock):
    """Piecewise linear video→OE clock, extrapolating the final exposure period."""
    video_times, oe_times = clock
    left = np.clip(np.searchsorted(video_times, seconds, side="right") - 1, 0, len(video_times) - 2)
    fraction = (seconds - video_times[left]) / (video_times[left + 1] - video_times[left])
    return oe_times[left] + fraction * (oe_times[left + 1] - oe_times[left])


def _continuous_audio_blocks(source, channels, clock, duration_s, *, band_hz=(300., 6000.),
                              sample_rate=48_000, block_s=1.):
    """One sequential raw read, causal bandpass, and clock-aligned audio resampling.

    Filter state survives block boundaries. The sample rate changes, not playback
    speed: each audio sample looks up the corresponding synchronized OE time.
    """
    channels = np.asarray(channels, dtype=int)
    times = np.load(source["timestamps"], mmap_mode="r")
    raw = np.memmap(source["binary"], dtype="<i2", mode="r",
                    shape=(len(times), source["num_channels"]))
    sos = butter(2, band_hz, btype="bandpass", fs=source["sample_rate"], output="sos")
    volts = np.asarray(source["bit_volts"])[channels]
    # Warm up the high-pass filter using the actual preceding voltage.
    first_time = _clock_times(np.array([0.]), clock)[0]
    cursor = int(np.searchsorted(times, first_time - .25))
    state = None
    previous_times = np.empty(0)
    previous_voltage = np.empty((0, len(channels)))
    sample_count = int(round(duration_s * sample_rate))
    block_samples = max(1, int(round(block_s * sample_rate)))
    for start in range(0, sample_count, block_samples):
        seconds = np.arange(start, min(start + block_samples, sample_count)) / sample_rate
        targets = _clock_times(seconds, clock)
        stop = min(len(times), int(np.searchsorted(times, targets[-1], side="right")) + 1)
        output = np.zeros((len(targets), len(channels)), dtype=np.float32)
        if stop > cursor:
            voltage = raw[cursor:stop, channels].astype(np.float32) * volts
            if state is None:
                state = sosfilt_zi(sos)[:, :, None] * voltage[0]
            voltage, state = sosfilt(sos, voltage, axis=0, zi=state)
            block_times = np.r_[previous_times, times[cursor:stop]]
            voltage = np.vstack((previous_voltage, voltage))
            cursor = stop
        else:
            block_times, voltage = previous_times, previous_voltage
        if len(block_times) > 1:
            left = np.clip(np.searchsorted(block_times, targets, side="right") - 1,
                           0, len(block_times) - 2)
            fraction = (targets - block_times[left]) / (block_times[left + 1] - block_times[left])
            output[:] = voltage[left] + fraction[:, None] * (voltage[left + 1] - voltage[left])
            # Keep unavailable acquisition time silent instead of stretching
            # the first/last measured value into the missing part of a clip.
            output[(targets < block_times[0]) | (targets > block_times[-1])] = 0
        # Keep both endpoints of the final interpolation interval: after
        # upsampling the next target can still fall before its right endpoint.
        previous_times, previous_voltage = block_times[-2:].copy(), voltage[-2:].copy()
        yield output


def _cache_continuous_audio(source, channels, clock, duration_s, directory, *,
                            band_hz=(300., 6000.), sample_rate=48_000):
    """Cache each distinct electrode once on scratch; share it across its units."""
    tracks = {int(ch): dict(path=Path(directory) / f"channel_{ch}.f32", channel=int(ch),
                           sample_rate=sample_rate, band_hz=band_hz) for ch in channels}
    noise_levels = []
    with ExitStack() as stack:
        files = [stack.enter_context(tracks[ch]["path"].open("wb")) for ch in channels]
        blocks = _continuous_audio_blocks(source, channels, clock, duration_s,
                                          band_hz=band_hz, sample_rate=sample_rate)
        for block in tqdm(blocks, total=int(np.ceil(duration_s)), desc="OE voltage audio", unit="s"):
            # Estimate background from sparse samples, robust to large spikes.
            sample = block[::16]
            noise_levels.append(np.median(abs(sample - np.median(sample, axis=0)), axis=0) / .67448975)
            for column, (ch, stream) in enumerate(zip(channels, files)):
                stream.write(block[:, column].astype("<f4").tobytes())
    noise_levels = np.asarray(noise_levels)
    for column, ch in enumerate(channels):
        # A short movement artifact must not set the gate for the entire clip.
        positive = noise_levels[:, column][noise_levels[:, column] > 0]
        tracks[ch]["noise_sigma"] = float(np.median(positive)) if len(positive) else 0.
    return tracks


@njit(cache=True, nogil=True)
def _expand_audio(voltage, threshold, ratio, envelope, level):
    """In-place downward expansion, retaining detector/gain state across blocks."""
    # OE 1.1 AudioNode.cpp uses these per-sample coefficients, not millisecond
    # time constants. Our threshold is in uV; OE's slider acts after volume gain.
    decay, smoothing = np.exp(-4.), np.exp(-1.)
    for i in range(len(voltage)):
        amplitude = abs(voltage[i]) + 1.e-29
        envelope = amplitude + decay * max(envelope - amplitude, 0.)
        target = min(envelope / threshold, 1.) ** (ratio - 1.)
        level = target + smoothing * (level - target)
        voltage[i] *= level
    return envelope, level


def _continuous_pcm(track, gain, gate_sigma=3., expander_ratio=4., *, block_s=1.):
    """Fixed voltage gain and soft expansion, without threshold-triggered windows.

    Gain/limiting follow OE AudioNode; the noise-relative threshold and stronger
    default ratio are ours. OE uses a ratio of 1.2 and an absolute slider value.
    Reference: https://github.com/open-ephys/plugin-GUI/blob/v1.1.0/Source/Processors/AudioNode/AudioNode.cpp
    """
    # OE normalized output = uV * volume_fraction / (32767 * .02).
    # Express the same conversion directly in int16 PCM units.
    scale = gain / .02
    threshold = gate_sigma * track["noise_sigma"]
    envelope, level = 0., 1.
    source = np.memmap(track["path"], dtype="<f4", mode="r")
    block_samples = max(1, round(block_s * track["sample_rate"]))
    for start in range(0, len(source), block_samples):
        block = np.clip(source[start:start + block_samples], -1000., 1000.)
        if threshold > 0 and expander_ratio > 1:
            envelope, level = _expand_audio(block, threshold, expander_ratio, envelope, level)
        # Reserve headroom and prevent int16 wraparound at high volume.
        yield np.rint(np.clip(block * scale, -.95 * 32767, .95 * 32767)).astype("<i2").tobytes()


def _mux_continuous_audio(video_path, output_path, unit_id, probe, track, duration, gain, gate_sigma=3.,
                          expander_ratio=4.):
    title = _continuous_audio_title(unit_id, probe, track, gain, gate_sigma, expander_ratio)
    _mux_pcm(video_path, output_path, title, duration,
             _continuous_pcm(track, gain, gate_sigma, expander_ratio), track["sample_rate"])


def _continuous_audio_title(unit_id, probe, track, gain, gate_sigma, expander_ratio):
    low, high = track["band_hz"]
    return (f"Probe {probe} unit {unit_id}; electrode {track['channel']} (zero-based); "
             f"continuous voltage {low:g}-{high:g} Hz; OE volume {gain:g}; "
             f"expander {expander_ratio:g}:1 below {gate_sigma:g} x noise sigma")


def _export_channel_videos(video_path, save_path, units, probe, track, duration, gain, gate_sigma,
                           expander_ratio):
    # Units sharing an electrode have identical audio. Encode it once on local
    # scratch, then copy both media streams while retaining each unit's label.
    audio_path = Path(track["path"]).with_suffix(".m4a")
    _mux_pcm(None, audio_path, f"Electrode {track['channel']}", duration,
             _continuous_pcm(track, gain, gate_sigma, expander_ratio), track["sample_rate"])
    for unit in units:
        title = _continuous_audio_title(unit, probe, track, gain, gate_sigma, expander_ratio)
        _mux_encoded_audio(video_path, audio_path, Path(save_path) / f"{unit}.mp4", title, duration)


def _mux_encoded_audio(video_path, audio_path, output_path, title, duration):
    partial = output_path.with_name(f".{output_path.stem}.partial.mp4")
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-t", f"{duration:.9f}",
             "-movflags", "+faststart", "-metadata:s:a:0", f"title={title}",
             "-metadata:s:a:0", f"handler_name={title}", str(partial)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"{output_path.name} audio export failed (ffmpeg exit {result.returncode}):\n"
                               f"{result.stderr.strip()}")
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)


def _load_audio_spikes(data, video_result):
    """Read spikes once for the batch, reusing the pose loader's exposure clock."""
    origin = data["adc_time_origin_s"]
    source = data["source"]
    kilosort = Path(source["kilosort_dir"])
    labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t")
    units = labels.loc[labels.KSLabel == "good", "cluster_id"].astype(int).tolist()
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").ravel()
    times = np.load(source["spike_times_path"], mmap_mode="r").ravel()
    start, end = source["selected_interval_s"]
    selected = np.isin(clusters, units) & (times >= start + origin) & (times <= end + origin)
    cluster_ids = clusters[selected]
    clock = _video_audio_clock(data, video_result)
    seconds = _clock_times(np.asarray(times[selected], dtype=float), (clock[1], clock[0]))
    in_clip = (seconds >= 0) & (seconds < video_result["frames"] / video_result["fps"])
    cluster_ids, seconds = cluster_ids[in_clip], seconds[in_clip]
    order = np.argsort(cluster_ids, kind="stable")
    cluster_ids, seconds = cluster_ids[order], seconds[order]
    # Keep every good label, including units with no spikes in this clip.
    left = np.searchsorted(cluster_ids, units, side="left")
    right = np.searchsorted(cluster_ids, units, side="right")
    grouped = {int(unit): seconds[a:b] for unit, a, b in zip(units, left, right)}
    return units, grouped, data["camera_timing"]


def spike_video_seconds(spike_times, exposure_times, fps):
    """Map ADC-relative spike times to fractional original AVI frame positions.

    Every camera exposure is used, including frames with missing pose or outside
    the arena. Interpolation preserves sub-frame timing and camera-clock drift.
    The first exposure midpoint is video time zero; end frames extrapolate by
    their adjacent exposure period instead of clamping spikes onto one instant.
    """
    left = np.clip(np.searchsorted(exposure_times, spike_times, side="right") - 1,
                   0, len(exposure_times) - 2)
    fraction = ((spike_times - exposure_times[left])
                / (exposure_times[left + 1] - exposure_times[left]))
    return (left + fraction) / fps


def _click_blocks(samples, click, sample_count, block_samples):
    for start in range(0, sample_count, block_samples):
        stop = min(start + block_samples, sample_count)
        block = np.zeros(stop - start, dtype=np.float32)
        # Include the tail of a click that began before this audio block.
        first = np.searchsorted(samples, start - len(click) + 1)
        last = np.searchsorted(samples, stop)
        for sample in samples[first:last]:
            offset = int(sample) - start
            a, b = max(offset, 0), min(offset + len(click), len(block))
            block[a:b] += click[a - offset:b - offset]
        yield block


def _spike_pcm(spike_seconds, duration_s, sample_rate, gain):
    """Prepare click gain and stream PCM blocks without an intermediate WAV."""
    if not 0 < gain <= 1:
        raise ValueError("gain must be between 0 (exclusive) and 1.")
    sample_count = int(round(duration_s * sample_rate))
    spike_seconds = np.asarray(spike_seconds, dtype=float)
    selected = spike_seconds[(spike_seconds >= 0) & (spike_seconds < duration_s)]
    # A sub-sample spike at the end belongs to the final audio sample.
    samples = np.minimum(np.rint(selected * sample_rate).astype(np.int64), sample_count - 1)
    samples.sort()
    t = np.arange(int(.004 * sample_rate)) / sample_rate
    click = np.exp(-t / .0007) * np.cos(2 * np.pi * 1800 * t)
    click -= click.mean()
    click = (click / np.max(np.abs(click))).astype(np.float32)
    block_samples = sample_rate * 10
    # Two streaming passes give one global gain without clipping dense bursts.
    peak = max(float(np.max(np.abs(block)))
               for block in _click_blocks(samples, click, sample_count, block_samples))
    applied_gain = min(gain, .95 / peak) if peak else gain

    def blocks():
        for block in _click_blocks(samples, click, sample_count, block_samples):
            block *= applied_gain * 32767
            np.rint(block, out=block)
            yield block.astype("<i2").tobytes()

    info = dict(spike_count=len(samples), sample_rate_hz=sample_rate,
                requested_gain=gain, applied_gain=applied_gain,
                sound="4 ms exponentially decaying 1.8 kHz biphasic click")
    return info, blocks()


def write_spike_clicks(path, spike_seconds, duration_s, *, sample_rate=48_000, gain=.35):
    """Write short biphasic clicks with constant per-spike gain, in bounded RAM."""
    info, blocks = _spike_pcm(spike_seconds, duration_s, sample_rate, gain)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        for block in blocks:
            stream.writeframes(block)
    return info


def _mux_spike_audio(video_path, output_path, unit_id, probe, spike_seconds, duration, gain):
    audio_info, blocks = _spike_pcm(spike_seconds, duration, 48_000, gain)
    _mux_pcm(video_path, output_path, f"Probe {probe} unit {unit_id} spike clicks", duration, blocks, 48_000)
    return audio_info


def _mux_pcm(video_path, output_path, title, duration, blocks, sample_rate):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    video_input = [] if video_path is None else ["-i", str(video_path)]
    stream_map = [] if video_path is None else ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    # Pipe PCM directly to ffmpeg to keep memory bounded; an optional video
    # input is copied without decoding or re-encoding its frames.
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *video_input,
             "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
             *stream_map, "-c:a", "aac", "-b:a", "192k",
             "-threads", "1", "-t", f"{duration:.9f}", "-movflags", "+faststart",
             "-metadata:s:a:0", f"title={title}", str(partial)],
            stdin=subprocess.PIPE, stderr=log,
        )
        try:
            for block in blocks:
                process.stdin.write(block)
            process.stdin.close()
            if process.wait() != 0:
                raise _ffmpeg_error(process, log, f"{output_path.name} audio export")
            partial.replace(output_path)
        except BrokenPipeError as error:
            raise _ffmpeg_error(process, log, f"{output_path.name} audio export") from error
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            partial.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, nargs="?")
    parser.add_argument("--probe", default=probe)
    parser.add_argument("--phase", default=phase)
    parser.add_argument(
        "--arena-type", choices=("rectangle", "cylinder"), default=arena_type
    )
    args = parser.parse_args(argv)

    selected_session = session if args.session is None else args.session
    selected_video = video_path if args.session is None else (
        None
        if args.arena_type == "cylinder"
        else selected_session / f"{selected_session.name.split('_')[0]}.avi"
    )
    selected_save_path = (
        save_path
        if args.session is None
        else selected_session / "data/spatial_cells/videos"
    )
    video_data = load_video_data(
        selected_session,
        probe=args.probe,
        phase=args.phase,
        arena_type=args.arena_type,
        fps=video_fps,
    )
    unit_videos = export_good_unit_videos(
        video_data, selected_video, selected_save_path,
        start_s=video_start_s, duration_s=video_duration_s, gain=audio_gain,
        workers=video_workers, audio_workers=audio_workers, video_encoder=video_encoder,
        audio_source=audio_source, audio_band_hz=audio_band_hz,
        audio_gate_sigma=audio_gate_sigma, audio_expander_ratio=audio_expander_ratio,
    )
    return unit_videos


if __name__ == "__main__":
    # The guard prevents spawned workers from starting the batch again.
    main()
