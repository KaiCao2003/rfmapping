"""Fit HD/RF relations from sessions A/B, then overlay session C.

Set the three render_scheme_* booleans below, then run this file with
~/.virtualenvs/rfmapping/bin/python. Each enabled scheme gets one silent MP4.
For a short check: python hd_rf_population_video.py --start 20 --duration 10

Session A supplies RF maps; session B supplies HD tuning curves. Session C
supplies the video, head_direction.json, and its own TC/spikes for decoding.
Set use_decoded_hd below. The default is tracked HD; --decoded-hd selects
neural HD for a run. Motive XYZ and rays are projected into the camera image
with the shared cylinder setup. The script reads C's raw ADC clock directly.
Outputs belong to session C's data/hd_rf directory.
"""

import argparse
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from Utils import ebc_video as video
from Utils import ebc_camera as camera
from Utils.ebc_replay import ray_circle_distance
from Utils.hd_rf_circular_regression import fit_circular_conversion, predict_circular_conversion
from Utils.hd_rf_schemes import load_peak_pairs, select_schemes
from Utils.hd_rf_video_data import load_population_video_data
from Utils.hd_rf_video_decode import decode_hd_frames, load_decoder_tuning_curves
from Utils.rfmap import load_rf_maps


# Set these three switches to choose the output videos.
render_scheme_1 = True  # All HD Class 3 cells.
render_scheme_2 = True  # HD Class 3 cells with a saved 2D RF.
render_scheme_3 = True  # Scheme 2 cells in the three largest HD+RF bins.

base_dir = Path("/mnt/senzailab/Kai/#Recording/m20/260922")
session_a = base_dir / "260922_1"  # RF
session_b = base_dir / "260922_3"  # HD
session_c = base_dir / "260922_3"  # Video and real/decoded HD
probe = "A"
use_decoded_hd = False
tc_is_clockwise = False  # The saved TC metadata documents counter-clockwise HD.
decode_bin_s = .1
tc_directory = Path("data/tuning_curves")
phase = "baseline"
start_s = 0.
duration_s = None  # None renders the full recording; seconds select a clip.
trace_window_s = 12.
# Raw ADC camera inputs; video frame f uses Motive exposure 2*f + phase.
motive_clock = dict(camera_input_channel=1, camera_ttl_threshold=14000,
                    camera_ttl_active_high=True)
basler_clock = dict(camera_input_channel=2, camera_ttl_threshold=2800,
                    camera_ttl_active_high=False)
video_exposure_stride = 2
video_exposure_phase = 0
# Explicitly measured dropped-frame mapping for this recording only.
# Tuples are (first video frame, last video frame, Motive exposure offset).
video_motive_segments = {
    ("m20", "260922_3"): [(0, 351, 0), (380, 72982, 56), (72996, 84813, 104)],
}
setup_dir = Path(__file__).resolve().parent / "config"
camera_registration_path = setup_dir / "cylinder_camera_registration.json"
cylinder_calibration_path = setup_dir / "cylinder.calib"
hd_world_zero_deg = 0.  # Saved clockwise HD zero in Motive's XY plane, CCW from +X.
# User-scoped NVENC libraries matching hhw9l84's installed driver.
# Set to None on a host whose system already provides libnvidia-encode.so.1.
nvenc_library_dir = Path.home() / ".local/lib/rfmapping-nvenc/595.71.05/runtime/usr/lib/x86_64-linux-gnu"


HD_COLOR = "#1686cc"
RF_COLOR = "#e88813"
TEXT_COLOR = "#17212b"
SCHEME_LABELS = {"scheme1": "All HD Class 3", "scheme2": "Class 3 + 2D RF",
                 "scheme3": "2D RF + top 3 sum bins"}


def enabled_schemes():
    return [f"scheme{i}" for i, enabled in enumerate(
        (render_scheme_1, render_scheme_2, render_scheme_3), start=1) if enabled]


def configure_nvenc():
    if nvenc_library_dir is not None:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if str(nvenc_library_dir) not in current.split(os.pathsep):
            os.environ["LD_LIBRARY_PATH"] = str(nvenc_library_dir) + (os.pathsep + current if current else "")


def screen_direction(angle_deg):
    radians = np.deg2rad(angle_deg)
    return np.stack((np.sin(radians), -np.cos(radians)), axis=-1)


def wrapped_segments(times, angles, max_gap_s=None):
    """Separate the 0/360 seam and missing samples rather than draw false jumps."""
    times, angles = np.asarray(times), np.asarray(angles)
    finite = np.isfinite(times) & np.isfinite(angles)
    separated = (np.abs(np.diff(angles)) > 180) | ~finite[:-1] | ~finite[1:]
    if max_gap_s is not None:
        separated |= np.diff(times) > max_gap_s
    breaks = np.flatnonzero(separated) + 1
    return [(times[part], angles[part]) for part in np.split(np.arange(len(times)), breaks)
            if len(part) > 1 and finite[part].all()]


class PopulationOverlay:
    def __init__(self, data, width, height, fps):
        self.data, self.width, self.height, self.fps = data, width, height, fps
        self.size = (width + 370, height + 300)
        self.cylinder = data.get("arena_type") == "cylinder"
        if self.cylinder:
            self.xyz = np.asarray(data["xyz"], dtype=float)
            self.projection = np.asarray(data["projection"], dtype=float)
            self.positions = camera.project(self.xyz, self.projection)
            self.cylinder_center = np.asarray(data["cylinder_center_raw"], dtype=float)
            self.cylinder_radius = data["cylinder_radius_raw"]
            self.raw_units_per_cm = data["raw_units_per_cm"]
            self.hd_world_zero_deg = data.get("hd_world_zero_deg", 0.)
            angles = np.linspace(0., 2 * np.pi, 129)
            self.wall_xy = (self.cylinder_center + self.cylinder_radius
                            * np.c_[np.cos(angles), np.sin(angles)])
        else:
            left, right, top, bottom = data["bounds_px"]
            scale = np.array([right - left, bottom - top]) / data["arena_size_cm"]
            self.positions = data["xy"] * scale * [1, -1] + [left, bottom]
        model = data["model"]
        self.prediction = predict_circular_conversion(data["hd"], model)
        self.rf_ego = np.asarray(self.prediction["rf_ego_deg"])
        self.rf_allo = np.asarray(self.prediction["rf_allo_deg"])
        self.valid = np.isfinite(self.positions).all(axis=1) & np.isfinite(data["hd"])
        if self.cylinder:
            self.valid &= np.isfinite(self.xyz).all(axis=1)
        self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        self.small = ImageFont.truetype("DejaVuSans.ttf", 16)
        self.title = ImageFont.truetype("DejaVuSans.ttf", 27)
        self.large = ImageFont.truetype("DejaVuSans.ttf", 31)
        self.window_s = data["trace_window_s"]
        self.max_gap_s = 1.5 * data["frame_period_s"]
        self.template = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(self.template)
        x = width + 20
        rho = "undefined" if model["rho_circular"] is None else f"{model['rho_circular']:.3f}"
        fit_error = "undefined" if model["fit_mae_deg"] is None else f"{model['fit_mae_deg']:.1f}°"
        for y, text, font in (
            (24, f"HD → RF · Scheme {model['scheme'][-1]}", self.title),
            (74, f"{data['session'].name} / Probe {data['probe']}", self.font),
            (111, SCHEME_LABELS[model["scheme"]], self.font),
            (172, f"{data['hd_mode'].capitalize()} head direction", self.small),
            (267, "Predicted RF egocentric angle", self.small),
            (362, "Derived sum: HD + RF ego", self.small),
            (471, "RF ego = f(HD)", self.font),
            (505, "Fitted circular regression", self.small),
            (737, "HD: source coordinate, clockwise", self.small),
            (768, "RF ego: 0° front, +90° right", self.small),
            (817, "Camera arrows: projected Motive XY" if self.cylinder
             else "Video arrow: HD + fitted RF ego", self.small),
            (847, "One population movie · no audio", self.small),
            (891, f"Cells: {model['n']} · circular ρ: {rho}", self.font),
            (925, f"Fitted angular error: {fit_error}", self.small),
            (959, "Selected on HD + RF; descriptive" if model['selected_on_sum']
             else "Class 3 · native 2D RF maximum", self.small),
        ):
            draw.text((x, y), text, font=font, fill=TEXT_COLOR)
        for y in (148, 450, 800):
            draw.line((x, y, x + 328, y), fill="#dce2e7", width=1)
        draw.text((32, height + 18), "Continuous HD and fitted RF ego", font=self.font, fill=TEXT_COLOR)
        draw.line((1060, height + 33, 1100, height + 33), fill=HD_COLOR, width=4)
        draw.text((1110, height + 20), "HD (left)", font=self.font, fill=TEXT_COLOR)
        draw.line((1210, height + 33, 1250, height + 33), fill=RF_COLOR, width=4)
        draw.text((1260, height + 20), "RF ego (right)", font=self.font, fill=TEXT_COLOR)

    @staticmethod
    def arrow(draw, start, angle, length, color):
        direction = screen_direction(angle)
        end = start + length * direction
        PopulationOverlay.projected_arrow(draw, start, end, color)

    @staticmethod
    def projected_arrow(draw, start, end, color):
        direction = np.asarray(end) - start
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length == 0:
            return
        direction = direction / length
        normal = np.array([-direction[1], direction[0]])
        draw.line([tuple(start), tuple(end)], fill="white", width=9)
        draw.line([tuple(start), tuple(end)], fill=color, width=5)
        draw.polygon([tuple(end), tuple(end - 16 * direction + 8 * normal),
                      tuple(end - 16 * direction - 8 * normal)], fill=color)

    def cylinder_geometry(self, row):
        """Construct Motive XY rays at head height, then project their endpoints."""
        head = self.xyz[row]
        hd_ccw = self.hd_world_zero_deg - self.data["hd"][row]
        wall_endpoints, distances = camera.boundary_endpoints(
            head, hd_ccw, self.cylinder_center, self.cylinder_radius,
        )
        heading = np.deg2rad(hd_ccw)
        hd_tip = head + 5 * self.raw_units_per_cm * np.array([np.cos(heading), np.sin(heading), 0.])
        rf_ccw = self.hd_world_zero_deg - self.rf_allo[row]
        angles = np.deg2rad([rf_ccw, rf_ccw - 6., rf_ccw + 6.])
        rf_distances = ray_circle_distance(
            head[:2], np.rad2deg(angles), self.cylinder_center, self.cylinder_radius,
        )
        rf_points = head + rf_distances[:, None] * np.c_[np.cos(angles), np.sin(angles), np.zeros(3)]
        wall = np.c_[self.wall_xy, np.full(len(self.wall_xy), head[2])]
        return dict(
            wall_pixels=camera.project(wall, self.projection),
            wall_endpoints_px=camera.project(wall_endpoints, self.projection),
            wall_distances_raw=distances, hd_tip_px=camera.project(hd_tip, self.projection),
            rf_points_px=camera.project(rf_points, self.projection),
        )

    def draw_cylinder(self, frame, row):
        geometry = self.cylinder_geometry(row)
        draw = ImageDraw.Draw(frame)
        head = self.positions[row]
        cyan = "#39d6e8"
        draw.line([tuple(point) for point in geometry["wall_pixels"]], fill=cyan, width=2)
        for endpoint, distance in zip(geometry["wall_endpoints_px"],
                                      geometry["wall_distances_raw"], strict=True):
            if not np.isfinite(endpoint).all():
                continue
            draw.line([tuple(head), tuple(endpoint)], fill=cyan, width=2)
            x, y = endpoint
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=cyan)
            draw.text(tuple(head + .78 * (endpoint - head)),
                      f"{distance / self.raw_units_per_cm:.1f}", font=self.small,
                      fill=cyan, stroke_width=2, stroke_fill=TEXT_COLOR, anchor="mm")
        self.projected_arrow(draw, head, geometry["hd_tip_px"], HD_COLOR)
        rf_points = geometry["rf_points_px"]
        if np.isfinite(rf_points).all():
            draw.line([tuple(head), tuple(rf_points[1]), tuple(rf_points[2]), tuple(head)],
                      fill=RF_COLOR, width=2)
            self.projected_arrow(draw, head, rf_points[0], RF_COLOR)
            draw.text(tuple(rf_points[0]), "RF", font=self.font, fill=RF_COLOR,
                      stroke_width=2, stroke_fill="white", anchor="mm")
        draw.ellipse((head[0] - 5, head[1] - 5, head[0] + 5, head[1] + 5),
                     fill="white", outline=TEXT_COLOR, width=2)

    def timeline(self, draw, now):
        left, right, top, bottom = 90, self.size[0] - 100, self.height + 75, self.height + 240
        start = now - self.window_s
        for angle in (0, 90, 180, 270, 360):
            y = bottom - angle / 360 * (bottom - top)
            draw.line((left, y, right, y), fill="#e4e8eb", width=1)
            draw.text((left - 12, y), f"{angle}°", font=self.small, fill=HD_COLOR, anchor="rm")
            draw.text((right + 12, y), f"{angle - 180}°", font=self.small, fill=RF_COLOR, anchor="lm")
        tick_s = self.window_s / 4
        for time in np.arange(np.ceil(start / tick_s) * tick_s, now + .001, tick_s):
            x = left + (time - start) / self.window_s * (right - left)
            draw.line((x, top, x, bottom), fill="#edf0f2", width=1)
            draw.text((x, bottom + 17), f"{time:.1f}", font=self.small, fill=TEXT_COLOR, anchor="mt")
        times = np.asarray(self.data["times"])
        lower, upper = np.searchsorted(times, [start, now], side="right")
        # Separate axes retain HD on [0, 360) and signed ego on [-180, 180).
        for angles, color in ((self.data["hd"], HD_COLOR), (self.rf_ego + 180., RF_COLOR)):
            for t, a in wrapped_segments(times[lower:upper], angles[lower:upper], self.max_gap_s):
                xs = left + (t - start) / self.window_s * (right - left)
                ys = bottom - a / 360 * (bottom - top)
                draw.line(list(zip(xs, ys)), fill=color, width=3)
        draw.line((right, top - 8, right, bottom), fill="#485661", width=2)
        draw.text((right, top - 12), "now", font=self.small, fill=TEXT_COLOR, anchor="rb")
        draw.text(((left + right) / 2, self.height + 283), "Time from Open Ephys ADC start (s)", font=self.small, fill=TEXT_COLOR, anchor="mm")

    def draw(self, frame, frame_id, row):
        canvas = self.template.copy()
        canvas.paste(frame, (0, 0))
        draw = ImageDraw.Draw(canvas)
        x = self.width + 20
        # The complete camera clock remains available when tracking has gaps.
        if (frame_id >= len(self.data["exposure_times"])
                or not np.isfinite(self.data["exposure_times"][frame_id])):
            draw.text((x, 210), "Camera timing unavailable", font=self.font, fill="#b3261e")
            return canvas
        now = float(self.data["exposure_times"][frame_id])
        self.timeline(draw, now)
        if row < 0 or not self.valid[row]:
            draw.text((x, 210), "HD / position unavailable", font=self.font, fill="#b3261e")
            return canvas
        hd, ego, allo = self.data["hd"][row], self.rf_ego[row], self.rf_allo[row]
        head = self.positions[row]
        if self.cylinder:
            camera_frame = frame.copy()
            self.draw_cylinder(camera_frame, row)
            canvas.paste(camera_frame, (0, 0))
        else:
            self.arrow(draw, head, hd, 78, HD_COLOR)
            if np.isfinite(allo):
                tip = head + 180 * screen_direction(allo)
                edges = head + 180 * screen_direction(np.array([allo - 6, allo + 6]))
                draw.line([tuple(head), tuple(edges[0]), tuple(edges[1]), tuple(head)], fill=RF_COLOR, width=2)
                self.arrow(draw, head, allo, 180, RF_COLOR)
                draw.text(tuple(tip + 12 * screen_direction(allo)), "RF", font=self.font,
                          fill=RF_COLOR, stroke_width=2, stroke_fill="white", anchor="mm")
            draw.ellipse((head[0] - 5, head[1] - 5, head[0] + 5, head[1] + 5),
                         fill="white", outline=TEXT_COLOR, width=2)
        draw.text((32, 24), f"{self.data['session'].name} · ADC {now:.2f} s", font=self.font,
                  fill="white", stroke_width=2, stroke_fill="#17212b")
        for y, value, color in ((206, hd, HD_COLOR), (301, ego, RF_COLOR), (396, allo, RF_COLOR)):
            draw.text((x, y), f"{value:.1f}°" if np.isfinite(value) else "undefined", font=self.large, fill=color)
        center, radius = np.array([x + 164, 642.]), 78
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="#cbd4da", width=2)
        draw.text((center[0], center[1] - radius - 16), "Forward · 0°", font=self.small, fill=TEXT_COLOR, anchor="mm")
        self.arrow(draw, center, 0., 62, HD_COLOR)
        if np.isfinite(ego):
            self.arrow(draw, center, ego, 75, RF_COLOR)
        return canvas


def fit_models(hd_file, rf_file, names, output_dir, *, mouse, date, probe,
               hd_is_clockwise):
    """Fit only A/B; session C does not enter unit pairing or RF selection."""
    directory = Path(output_dir) / "relation"
    directory.mkdir(parents=True, exist_ok=True)
    detection_path = directory / "rf_detection.npz"
    load_rf_maps(rf_file).sum(0., .2, show_progress=False).rf_2d(
        is_shuffle=False, cluster_forming_z=1.8, drop_bins=2, wrap_x=True,
        result_path=detection_path, show_progress=False,
    )
    pairs, provenance = load_peak_pairs(
        hd_file, rf_file, rf_detection_path=detection_path,
        hd_is_clockwise=hd_is_clockwise, mouse=mouse, date=date, probe=probe,
    )
    schemes, selection = select_schemes(pairs)
    pairs.to_csv(directory / "all_pairs.csv", index=False)
    (directory / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    models = []
    for name in names:
        selected = schemes[name]
        result = fit_circular_conversion(
            selected.hd_preferred_deg, selected.rf_ego_deg,
            selected_on_sum=name == "scheme3",
        )
        model = dict(result, scheme=name, rho_circular=result["rho"],
                     fit_mae_deg=result["mae_deg"], selected_on_sum=name == "scheme3",
                     selected_units=selected.unit_id.astype(int).tolist(), provenance=provenance)
        path = directory / f"{name}_conversion.json"
        payload = (json.dumps(model, indent=2, allow_nan=False) + "\n").encode()
        path.write_bytes(payload)
        selected.to_csv(directory / f"{name}.csv", index=False)
        model.update(model_path=str(path.resolve()), model_sha256=hashlib.sha256(payload).hexdigest())
        models.append(model)
    return models


def select_hd(data, tc_file, *, use_decoded_hd, bin_size_s):
    """Use C's tracked HD or decode from C's own saved TC and spikes."""
    data = dict(data)
    data["hd_mode"] = "decoded" if use_decoded_hd else "real"
    source = dict(mode=data["hd_mode"], session=str(data["session"]), probe=data["probe"],
                  angle_convention="HD clockwise in the source coordinate")
    if use_decoded_hd:
        curves = load_decoder_tuning_curves(tc_file)
        metadata = curves.attrs["metadata"]
        if metadata["session"] != data["session"].name:
            raise ValueError("The decoding TC must belong to session C.")
        if abs(metadata["adc_time_origin_raw_s"] - data["adc_time_origin_s"]) > 1e-8:
            raise ValueError("Session C's TC and video clock must have the same ADC origin.")
        times_path = data["session"] / "data" / f"probe{data['probe']}/adc_spike_time.npy"
        clusters_path = Path(metadata["kilosort_dir"]) / "spike_clusters.npy"
        times = np.load(times_path, mmap_mode="r").reshape(-1) - data["adc_time_origin_s"]
        clusters = np.load(clusters_path, mmap_mode="r").reshape(-1)
        data["hd"], qc = decode_hd_frames(
            curves, times, clusters, data["times"], bin_size_s, data["decoding_intervals_s"],
        )
        if not tc_is_clockwise:
            data["hd"] = (-data["hd"]) % 360.
        data["hd_path"] = str(tc_file)
        source.update(tc=str(tc_file), spike_times=str(times_path), spike_clusters=str(clusters_path),
                      unit_ids=curves.index.astype(int).tolist(), bin_size_s=bin_size_s,
                      method="Poisson MAP with uniform prior; native unsmoothed Hz; no pseudocount",
                      tc_sha256=hashlib.sha256(Path(tc_file).read_bytes()).hexdigest(),
                      evaluation="TC training and decoding may overlap; fitted population estimate",
                      **qc)
        print(f"Decoded C: {len(curves)} units; {np.isfinite(data['hd']).sum():,} finite video frames.", flush=True)
    else:
        source.update(json=str(data["hd_path"]), method="saved head_direction.json")
    data["hd_source"] = source
    return data


def frame_window(data, start_s, duration_s):
    """Choose AVI ordinals using measured time elapsed from the first frame."""
    clock = np.asarray(data["exposure_times"])
    frames = np.flatnonzero(np.isfinite(clock))
    times = clock[frames]
    start = times[0] + start_s
    first_index = np.searchsorted(times, start)
    if first_index == len(frames):
        raise ValueError("The requested start is after the measured video clock.")
    first = int(frames[first_index])
    stop = data["video_frame_count"]
    if duration_s is not None:
        stop_index = np.searchsorted(times, start + duration_s)
        if stop_index < len(frames):
            stop = int(frames[stop_index])
    return first, stop


def render_scheme(data, model, *, output_dir, start_s, duration_s, trace_window_s):
    """One spawned process: CPU drawing/decoding and strict GPU NVENC encoding."""
    configure_nvenc()
    data = dict(data, model=model, trace_window_s=trace_window_s)
    name = model["scheme"]
    output_dir = Path(output_dir)
    destination = output_dir / f"{name}.mp4"
    first_frame, stop_frame = frame_window(data, start_s, duration_s)
    result = video.export_ebc_overlay(data, data["video_path"], destination,
                                      start_s=start_s, duration_s=duration_s,
                                      workers=1, overlay_type=PopulationOverlay, video_encoder="h264_nvenc",
                                      frame_rate=1. / data["frame_period_s"],
                                      first_frame=first_frame, stop_frame=stop_frame)
    first, last = result["source_first_frame"], result["source_last_frame"]
    selected = (data["frame_ids"] >= first) & (data["frame_ids"] <= last)
    predicted = predict_circular_conversion(data["hd"][selected], model)
    table = pd.DataFrame({"video_frame": data["frame_ids"][selected], "adc_time_s": data["times"][selected],
                          "hd_deg": data["hd"][selected], "rf_ego_deg": predicted["rf_ego_deg"],
                          "rf_allo_deg": predicted["rf_allo_deg"]})
    table.to_csv(output_dir / f"{name}_frame_angles.csv", index=False)
    result.pop("ray_bearings_deg", None)
    result.pop("outside_arena_frames", None)
    result.update(scheme=name, model=model["model_path"], model_kind=model["method"],
                  model_sha256=model["model_sha256"],
                  cos_coefficients=model["cos_coefficients"], sin_coefficients=model["sin_coefficients"],
                  hd_source=data["hd_source"], hd_mode=data["hd_mode"], fit_provenance=model["provenance"],
                  n_cells=model["n"], circular_rho=model["rho_circular"],
                  fitted_angular_error_deg=model["fit_mae_deg"],
                  audio=False, trace_window_s=trace_window_s,
                  angle_convention="HD clockwise in source coordinate; RF ego clockwise/right positive; RF allo=wrap360(HD+RF ego)",
                  interpretation="RF ego=f(HD) fitted from paired unit directions; RF allo is the derived sum HD+RF ego",
                  trace_angles="HD on left axis [0, 360); fitted RF ego on right axis [-180, 180)",
                  camera_timing=data["camera_timing"],
                  frame_mapping=data["frame_mapping"], video_clock=data["video_clock"],
                  position_source=str(data["position_source"]),
                  encoding="GPU NVENC", drawing="CPU", width=1650, height=1324)
    if data.get("arena_type") == "cylinder":
        result.update(arena_type="cylinder", hd_world_zero_deg=data["hd_world_zero_deg"],
                      camera_registration=str(data["camera_registration_path"]),
                      camera_registration_sha256=data["camera_registration_sha256"],
                      cylinder_calibration=str(data["cylinder_calibration_path"]),
                      cylinder_calibration_sha256=data["cylinder_calibration_sha256"])
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {name}: {destination}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rf-session", type=Path, default=session_a)
    parser.add_argument("--hd-session", type=Path, default=session_b)
    parser.add_argument("--video-session", type=Path, default=session_c)
    parser.add_argument("--probe", default=probe)
    parser.add_argument("--camera-registration", type=Path, default=camera_registration_path)
    parser.add_argument("--cylinder-calibration", type=Path, default=cylinder_calibration_path)
    parser.add_argument("--phase", default=phase)
    parser.add_argument("--decoded-hd", action=argparse.BooleanOptionalAction, default=use_decoded_hd)
    parser.add_argument("--decode-bin", type=float, default=decode_bin_s)
    parser.add_argument("--start", type=float, default=start_s)
    parser.add_argument("--duration", type=float, default=duration_s)
    parser.add_argument("--trace-window", type=float, default=trace_window_s)
    parser.add_argument("--output-name", help="Subdirectory within session C's data/hd_rf.")
    args = parser.parse_args(argv)
    names = enabled_schemes()
    if not names:
        print("All three render switches are False; no videos selected.")
        return []
    if args.trace_window <= 0:
        raise ValueError("--trace-window must be positive.")
    if args.start < 0 or (args.duration is not None and args.duration <= 0) or args.decode_bin <= 0:
        raise ValueError("Start must be nonnegative; duration and decoding bin size must be positive.")
    mode = "decoded" if args.decoded_hd else "real"
    output_name = args.output_name or f"hd{args.hd_session.name}_rf{args.rf_session.name}_{mode}"
    if output_name in (".", "..") or Path(output_name).name != output_name:
        raise ValueError("--output-name must be a single directory name.")
    output_dir = args.video_session / "data/hd_rf" / output_name
    configure_nvenc()
    # Fail before reading the recording if GPU encoding cannot initialize.
    video._select_encoder("h264_nvenc", 1650, 1324, "25")
    hd_file = args.hd_session / tc_directory / f"Probe{args.probe}/tuning_curves.tc"
    rf_file = (args.rf_session / "data/rfmapping/good/-100_400_1ms" / f"Probe{args.probe}"
               / f"regular_unitsSpikeCounts_{args.rf_session.name}.rfmap")
    print(f"1. Fit relation: HD {args.hd_session.name}; RF {args.rf_session.name}.", flush=True)
    models = fit_models(
        hd_file, rf_file, names, output_dir, mouse=args.hd_session.parents[1].name,
        date=args.hd_session.parent.name, probe=args.probe, hd_is_clockwise=tc_is_clockwise,
    )
    print(f"2. Apply to C {args.video_session.name}: {mode} HD.", flush=True)
    data = load_population_video_data(
        args.video_session, probe=args.probe, phase=args.phase,
        motive_clock=motive_clock, basler_clock=basler_clock,
        video_stride=video_exposure_stride, video_phase=video_exposure_phase,
        video_segments=video_motive_segments.get((args.video_session.parents[1].name,
                                                 args.video_session.name)),
        camera_registration_path=args.camera_registration,
        cylinder_calibration_path=args.cylinder_calibration,
    )
    data["hd_world_zero_deg"] = hd_world_zero_deg
    data = select_hd(
        data, args.video_session / tc_directory / f"Probe{args.probe}/tuning_curves.tc",
        use_decoded_hd=args.decoded_hd, bin_size_s=args.decode_bin,
    )
    pd.DataFrame(dict(video_frame=data["frame_ids"], adc_time_s=data["times"],
                      hd_deg=data["hd"])).to_csv(output_dir / "head_direction.csv", index=False)
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.unlink(missing_ok=True)
    results = []
    print(f"Rendering {', '.join(names)} concurrently with GPU NVENC; no audio.", flush=True)
    with ProcessPoolExecutor(max_workers=len(models), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(render_scheme, data, model, output_dir=output_dir,
                               start_s=args.start, duration_s=args.duration,
                               trace_window_s=args.trace_window) for model in models]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result["scheme"])
    manifest_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
