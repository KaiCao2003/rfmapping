"""Render the three HD/RF schemes concurrently on hhw9l84 using NVENC.

Set the three render_scheme_* booleans below, then run this file with
~/.virtualenvs/rfmapping/bin/python. Each enabled scheme gets one silent MP4.
For a short check: python hd_rf_population_video.py --start 20 --duration 10

HD comes from generated JSON. Each scheme fits RF ego=f(HD) from the paired
unit directions. The world-frame arrow uses the derived sum HD+RF ego.
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
from Utils.hd_rf_circular_regression import predict_circular_conversion
from Utils.hd_rf_video_data import load_scheme_video_data


# Set these three switches to choose the output videos.
render_scheme_1 = True  # All HD Class 3 cells.
render_scheme_2 = True  # HD Class 3 cells with a saved 2D RF.
render_scheme_3 = True  # Scheme 2 cells in the three largest HD+RF bins.

base_dir = Path("/mnt/senzailab/Kai/#Recording/m19")
date = "260827"
session_id = "11"
probe = "A"
wall_config = "old"
phase = "baseline"
start_s = 0.
duration_s = None  # None renders the full recording; seconds select a clip.
trace_window_s = 12.
project_dir = Path(__file__).resolve().parent
model_dir = project_dir / "output/hd_rf_schemes/m19_260827_hd11_rf2"
output_dir = project_dir / "output/hd_rf_population/m19_260827_11_three_schemes"
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
        left, right, top, bottom = data["bounds_px"]
        scale = np.array([right - left, bottom - top]) / data["arena_size_cm"]
        self.positions = data["xy"] * scale * [1, -1] + [left, bottom]
        model = data["model"]
        self.prediction = predict_circular_conversion(data["hd"], model)
        self.rf_ego = np.asarray(self.prediction["rf_ego_deg"])
        self.rf_allo = np.asarray(self.prediction["rf_allo_deg"])
        self.valid = np.isfinite(self.positions).all(axis=1) & np.isfinite(data["hd"])
        self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        self.small = ImageFont.truetype("DejaVuSans.ttf", 16)
        self.title = ImageFont.truetype("DejaVuSans.ttf", 27)
        self.large = ImageFont.truetype("DejaVuSans.ttf", 31)
        self.window_s = data["trace_window_s"]
        self.template = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(self.template)
        x = width + 20
        for y, text, font in (
            (24, f"HD → RF · Scheme {model['scheme'][-1]}", self.title),
            (74, f"{data['session'].name} / Probe {data['probe']}", self.font),
            (111, SCHEME_LABELS[model["scheme"]], self.font),
            (172, "Current head direction", self.small),
            (267, "Predicted RF egocentric angle", self.small),
            (362, "Derived sum: HD + RF ego", self.small),
            (471, "RF ego = f(HD)", self.font),
            (505, "Fitted circular regression", self.small),
            (737, "HD: 0° up, positive clockwise", self.small),
            (768, "RF ego: 0° front, +90° right", self.small),
            (817, "Video arrow: HD + fitted RF ego", self.small),
            (847, "One population movie · no audio", self.small),
            (891, f"Cells: {model['n']} · circular ρ: {model['rho_circular']:.3f}", self.font),
            (925, f"Fitted angular error: {model['fit_mae_deg']:.1f}°", self.small),
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
        normal = np.array([-direction[1], direction[0]])
        draw.line([tuple(start), tuple(end)], fill="white", width=9)
        draw.line([tuple(start), tuple(end)], fill=color, width=5)
        draw.polygon([tuple(end), tuple(end - 16 * direction + 8 * normal),
                      tuple(end - 16 * direction - 8 * normal)], fill=color)

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
            for t, a in wrapped_segments(times[lower:upper], angles[lower:upper], 1.5 / self.fps):
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
        if frame_id >= len(self.data["exposure_times"]):
            draw.text((x, 210), "Camera timing unavailable", font=self.font, fill="#b3261e")
            return canvas
        now = float(self.data["exposure_times"][frame_id])
        self.timeline(draw, now)
        draw.text((32, 24), f"{self.data['session'].name} · {frame_id / self.fps:.2f} s", font=self.font,
                  fill="white", stroke_width=2, stroke_fill="#17212b")
        if row < 0 or not self.valid[row]:
            draw.text((x, 210), "HD / position unavailable", font=self.font, fill="#b3261e")
            return canvas
        hd, ego, allo = self.data["hd"][row], self.rf_ego[row], self.rf_allo[row]
        head = self.positions[row]
        self.arrow(draw, head, hd, 78, HD_COLOR)
        if np.isfinite(allo):
            # A narrow sector makes the predicted visual-field bearing visible.
            tip = head + 180 * screen_direction(allo)
            edges = head + 180 * screen_direction(np.array([allo - 6, allo + 6]))
            draw.line([tuple(head), tuple(edges[0]), tuple(edges[1]), tuple(head)], fill=RF_COLOR, width=2)
            self.arrow(draw, head, allo, 180, RF_COLOR)
            draw.text(tuple(tip + 12 * screen_direction(allo)), "RF", font=self.font,
                      fill=RF_COLOR, stroke_width=2, stroke_fill="white", anchor="mm")
        draw.ellipse((head[0] - 5, head[1] - 5, head[0] + 5, head[1] + 5), fill="white", outline=TEXT_COLOR, width=2)
        for y, value, color in ((206, hd, HD_COLOR), (301, ego, RF_COLOR), (396, allo, RF_COLOR)):
            draw.text((x, y), f"{value:.1f}°" if np.isfinite(value) else "undefined", font=self.large, fill=color)
        center, radius = np.array([x + 164, 642.]), 78
        draw.ellipse((center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), outline="#cbd4da", width=2)
        draw.text((center[0], center[1] - radius - 16), "Forward · 0°", font=self.small, fill=TEXT_COLOR, anchor="mm")
        self.arrow(draw, center, 0., 62, HD_COLOR)
        if np.isfinite(ego):
            self.arrow(draw, center, ego, 75, RF_COLOR)
        return canvas


def load_models(directory, names, session):
    """Read only enabled schemes and reject a different session's world frame."""
    models = []
    for name in names:
        path = Path(directory) / f"{name}_conversion.json"
        payload = path.read_bytes()
        model = json.loads(payload)
        if model["method"] != "first_harmonic_circular_regression" or model["scheme"] != name:
            raise ValueError(f"{path}: expected the {name} fitted circular regression.")
        if Path(model["provenance"]["hd_source"]).parents[3].resolve() != Path(session).resolve():
            raise ValueError(f"{path}: HD source session does not match the video session.")
        model.update(model_path=str(path.resolve()), model_sha256=hashlib.sha256(payload).hexdigest())
        models.append(model)
    return models


def render_scheme(data, model, *, output_dir, start_s, duration_s, trace_window_s):
    """One spawned process: CPU drawing/decoding and strict GPU NVENC encoding."""
    configure_nvenc()
    data = dict(data, model=model, trace_window_s=trace_window_s)
    name = model["scheme"]
    output_dir = Path(output_dir)
    destination = output_dir / f"{name}.mp4"
    result = video.export_ebc_overlay(data, data["video_path"], destination,
                                      start_s=start_s, duration_s=duration_s,
                                      workers=1, overlay_type=PopulationOverlay, video_encoder="h264_nvenc")
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
                  hd_source=str(data["hd_path"]), fit_provenance=model["provenance"],
                  n_cells=model["n"], circular_rho=model["rho_circular"],
                  fitted_angular_error_deg=model["fit_mae_deg"],
                  audio=False, trace_window_s=trace_window_s,
                  angle_convention="HD clockwise from image north; RF ego clockwise/right positive; RF allo=wrap360(HD+RF ego)",
                  interpretation="RF ego=f(HD) fitted from paired unit directions; RF allo is the derived sum HD+RF ego",
                  trace_angles="HD on left axis [0, 360); fitted RF ego on right axis [-180, 180)",
                  camera_timing=data["camera_timing"],
                  frame_mapping=data["frame_mapping"],
                  position_source=str(data["position_source"]),
                  encoding="GPU NVENC", drawing="CPU", width=1650, height=1324)
    destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {name}: {destination}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--date", default=date)
    parser.add_argument("--session", default=session_id)
    parser.add_argument("--probe", default=probe)
    parser.add_argument("--wall-config", choices=("old", "new"), default=wall_config)
    parser.add_argument("--phase", default=phase)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument("--start", type=float, default=start_s)
    parser.add_argument("--duration", type=float, default=duration_s)
    parser.add_argument("--trace-window", type=float, default=trace_window_s)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    args = parser.parse_args(argv)
    names = enabled_schemes()
    if not names:
        print("All three render switches are False; no videos selected.")
        return []
    if args.trace_window <= 0:
        raise ValueError("--trace-window must be positive.")
    session = args.base_dir / args.date / f"{args.date}_{args.session}"
    models = load_models(args.model_dir, names, session)
    configure_nvenc()
    # Fail before reading the recording if GPU encoding cannot initialize.
    video._select_encoder("h264_nvenc", 1650, 1324, "25")
    data = load_scheme_video_data(session, probe=args.probe, phase=args.phase, wall_config=args.wall_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "render_manifest.json"
    manifest_path.unlink(missing_ok=True)
    results = []
    print(f"Rendering {', '.join(names)} concurrently with GPU NVENC; no audio.", flush=True)
    with ProcessPoolExecutor(max_workers=len(models), mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(render_scheme, data, model, output_dir=args.output_dir,
                               start_s=args.start, duration_s=args.duration,
                               trace_window_s=args.trace_window) for model in models]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: result["scheme"])
    manifest_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    main()
