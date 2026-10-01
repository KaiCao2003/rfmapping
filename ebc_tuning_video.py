"""Overlay saved EBC tuning, JSON head direction, and real electrode audio.

Run on hhw9l84. Example (one 10-second clip):
  python ebc_tuning_video.py --date 260921 --session 11 \
      --base-dir '/mnt/senzailab/Kai/#Recording/m20' --wall-config new \
      --unit 2 --start 10 --duration 10

Omitting --unit exports every good unit with a saved EBC map. Clip times use
the AVI timeline. HD is north-zero clockwise; saved EBC bearings are CCW
relative to the head. CSV front_x/front_y supply position only, never HD.
"""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from Utils import ebc_video as video
from Utils.ebc_analysis import BASLER_SIZE_CM, boundary_new, boundary_old
from Utils.json_tools import read_formatted_json
from Utils.rfmap import load_rf_maps
from Utils.tuning_curve_utils import get_exposure_timestamps


WALL_CONFIGS = {"old": boundary_old, "new": boundary_new}
TUNING_COLOR = "#19b6c6"
PEAK_COLOR = "#ffbe32"


def load_json_pose(session, probe, phase, wall_config, camera_input_channel=None):
    """Join JSON HD to camera head positions by exact zero-based frame ID."""
    directory = session / "data"
    hd_path = directory / "processed/head_direction.json"
    heading = read_formatted_json(hd_path)["hp4"]
    # Basler and Motive JSON use these two established field names. This
    # rectangle entrance requires Basler image coordinates and a Basler AVI.
    hd_key = "hd" if "hd" in heading else "head_direction_deg"
    hd = pd.Series(heading[hd_key], index=heading["frames"], name="hd")
    pose = pd.read_csv(session / f"{session.name.split('_')[0]}.csv",
                       usecols=["frame", "front_x", "front_y"]).set_index("frame")
    pose = pose.join(hd, how="inner").dropna().sort_index()
    tc_path = directory / "tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
    ttl_qc = read_formatted_json(tc_path)["metadata"].get("ttl_qc", {})
    channel = ttl_qc.get("camera_input_channel", 1) if camera_input_channel is None else camera_input_channel
    exposures, origin, timing = get_exposure_timestamps(
        read_formatted_json(directory / "session_info.json")["session_info"], directory,
        camera_input_channel=channel, camera_ttl_active_high=False,
    )
    frames = pose.index.to_numpy(dtype=int)
    if np.any(frames < 0) or np.any(np.diff(frames) <= 0):
        raise ValueError("Head frame IDs must be unique zero-based camera exposure indices.")
    # Some Basler recordings contain a final tracked frame without a camera TTL.
    # Keep it out of synchronized geometry rather than inventing its timestamp.
    synchronized = frames < len(exposures)
    dropped = int((~synchronized).sum())
    pose, frames = pose.loc[synchronized], frames[synchronized]
    times = exposures[frames]
    intervals = pd.read_csv(directory / "interval_table.csv")
    interval = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0].to_numpy(float)
    selected = (times >= interval[0]) & (times <= interval[1])
    pose = pose.loc[selected]
    bounds = WALL_CONFIGS[wall_config]
    left, right, top, bottom = bounds
    xy = np.c_[(pose.front_x - left) * BASLER_SIZE_CM / (right - left),
               (bottom - pose.front_y) * BASLER_SIZE_CM / (bottom - top)]
    kilosort = next((session / "kilosort" / f"Probe{probe}").glob("kilosort_*"))
    return dict(session=session, probe=probe, phase=phase, xy=xy,
                hd=pose.hd.to_numpy() % 360, frame_ids=frames[selected], times=times[selected],
                bounds_px=bounds, arena_size_cm=BASLER_SIZE_CM,
                exposure_times=exposures, adc_time_origin_s=origin, camera_timing=timing,
                dropped_unsynchronized_pose_frames=dropped,
                hd_path=str(hd_path), position_columns=["front_x", "front_y"],
                kilosort_dir=str(kilosort))


class TuningOverlay:
    """Animal-centered 1D rate projection, with its maximum ray to the wall."""

    def __init__(self, data, width, height, fps):
        self.data, self.width, self.height, self.fps = data, width, height, fps
        self.size = (width + 370, height)
        self.angles = np.asarray(data["tuning_angles_deg"])
        rates = np.asarray(data["tuning_rate"])
        self.peak_index = int(np.nanargmax(rates))
        self.peak = float(self.angles[self.peak_index])
        self.radii_cm = 7. * rates / np.nanmax(rates)
        self.positions, self.endpoints, self.distances, self.valid = video.overlay_geometry(
            data, bearings_deg=[self.peak], hd_is_clockwise=True,
        )
        left, right, top, bottom = data["bounds_px"]
        self.pixels_per_cm = np.array([right - left, bottom - top]) / data["arena_size_cm"]
        self.font = ImageFont.truetype("DejaVuSans.ttf", 20)
        self.small = ImageFont.truetype("DejaVuSans.ttf", 16)
        self.title = ImageFont.truetype("DejaVuSans.ttf", 26)
        self.template = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(self.template)
        x = width + 22
        for y, text, font in (
            (24, "EBC tuning overlay", self.title),
            (70, f"{data['session'].name} / unit {data['unit_id']}", self.font),
            (108, data["phase"], self.small),
            (280, "HD from head_direction.json", self.small),
            (310, "HD: 0° up, positive clockwise", self.small),
            (380, "EBC 1D tuning", self.font),
            (418, "Saved rates summed over distance", self.small),
            (448, "Radius: 0 to unit maximum", self.small),
            (478, "Maximum radius: 7 cm", self.small),
            (570, f"Peak EBC bin: {self.peak:g}°", self.font),
            (608, "Ego: 0° front, 90° left", self.small),
            (730, "Real Open Ephys electrode audio", self.small),
            (760, "Band-pass: 300–6000 Hz", self.small),
            (790, "Aligned to camera exposure times", self.small),
        ):
            draw.text((x, y), text, font=font, fill="#17212b")
        for y, color in ((257, video.HD_COLOR), (356, TUNING_COLOR), (546, PEAK_COLOR)):
            draw.line((x, y, x + 45, y), fill=color, width=5)

    def draw(self, frame, frame_id, row):
        canvas = self.template.copy()
        canvas.paste(frame, (0, 0))
        draw = ImageDraw.Draw(canvas)
        left, right, top, bottom = self.data["bounds_px"]
        draw.rectangle((left, top, right, bottom), outline="#111111", width=6)
        draw.rectangle((left, top, right, bottom), outline=video.BOUNDARY_COLOR, width=3)
        x = self.width + 22
        draw.text((x, 160), f"AVI frame {frame_id} / {frame_id / self.fps:.2f} s", font=self.small, fill="#17212b")
        if row < 0:
            draw.text((x, 202), "No JSON HD / head position", font=self.small, fill="#b3261e")
            return canvas
        head = self.positions[row]
        hd = float(self.data["hd"][row])
        angle = np.deg2rad(self.angles - hd)
        directions = np.c_[-np.sin(angle), -np.cos(angle)]
        curve = head + directions * self.radii_cm[:, None] * self.pixels_per_cm
        # Draw only adjacent finite bins, preserving unvisited angular gaps.
        for i in range(len(curve)):
            segment = curve[[i, (i + 1) % len(curve)]]
            if np.isfinite(segment).all():
                points = [tuple(point) for point in segment]
                draw.line(points, fill="#10232c", width=7)
                draw.line(points, fill=TUNING_COLOR, width=4)
        if self.valid[row]:
            wall = self.endpoints[row, 0]
            draw.line([tuple(head), tuple(wall)], fill="#111111", width=7)
            draw.line([tuple(head), tuple(wall)], fill=PEAK_COLOR, width=4)
            draw.ellipse((wall[0] - 6, wall[1] - 6, wall[0] + 6, wall[1] + 6), fill=PEAK_COLOR, outline="#111111", width=2)
        direction = np.array([np.sin(np.deg2rad(hd)), -np.cos(np.deg2rad(hd))])
        tip = head + 64 * direction
        normal = np.array([-direction[1], direction[0]])
        draw.line([tuple(head), tuple(tip)], fill="white", width=9)
        draw.line([tuple(head), tuple(tip)], fill=video.HD_COLOR, width=5)
        draw.polygon([tuple(tip), tuple(tip - 14 * direction + 7 * normal),
                      tuple(tip - 14 * direction - 7 * normal)], fill=video.HD_COLOR)
        draw.ellipse((head[0] - 5, head[1] - 5, head[0] + 5, head[1] + 5), fill="white", outline="#111111", width=2)
        draw.text((x, 202), f"HD {hd:.1f}° / ADC {self.data['times'][row]:.3f} s", font=self.small, fill="#17212b")
        if self.valid[row]:
            draw.text((x, 648), f"Head → wall: {self.distances[row, 0]:.2f} cm", font=self.font, fill="#17212b")
        else:
            draw.text((x, 648), "Outside wall bounds: ray omitted", font=self.small, fill="#b3261e")
        return canvas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="YYMMDD")
    parser.add_argument("--session", required=True, help="Recording number, e.g. 11")
    parser.add_argument("--base-dir", "--base_dir", type=Path, required=True, help="Mouse directory containing date folders")
    parser.add_argument("--probe", default="A")
    parser.add_argument("--wall-config", "--wall_config", choices=WALL_CONFIGS, required=True)
    parser.add_argument("--unit", type=int, action="append", help="Repeat to select units; default: all good units with EBC maps")
    parser.add_argument("--phase", default="baseline")
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--duration", type=float, help="Seconds; default: full AVI")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--camera-input-channel", type=int, help="Override the saved tuning-curve TTL channel")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    session = args.base_dir / args.date / f"{args.date}_{args.session}"
    output = args.output_dir or session / "data/spatial_cells/videos/ebc_tuning" / f"Probe{args.probe}" / args.wall_config
    output.mkdir(parents=True, exist_ok=True)
    data = load_json_pose(session, args.probe, args.phase, args.wall_config, args.camera_input_channel)
    map_path = session / "data/spatial_cells" / f"Probe{args.probe}" / args.phase / "egocentric_rate_map.rfmap"
    maps = load_rf_maps(map_path)
    channels = video._good_unit_channels(Path(data["kilosort_dir"]))
    curves = {item.unit_id: item.to_1d_array(axis="y") for item in maps}
    eligible = sorted(unit for unit in set(channels) & set(curves)
                      if np.any(np.isfinite(curves[unit]) & (curves[unit] > 0)))
    units = eligible if args.unit is None else args.unit
    if not units or not set(units) <= set(eligible):
        raise ValueError("Select good units with positive finite tuning in the saved EBC map.")
    if args.unit is None:
        skipped = sorted((set(channels) & set(curves)) - set(eligible))
        if skipped:
            print(f"Omitting units without positive finite EBC tuning: {skipped}", flush=True)
    source = video._open_ephys_source(session, args.probe)
    source_video = session / f"{args.date}.avi"
    results = []
    with tempfile.TemporaryDirectory(prefix="ebc_tuning_") as temporary:
        tracks = None
        for unit in units:
            rf_map = maps[maps.unit_ids.index(unit)]
            rates = curves[unit]
            unit_data = dict(data, unit_id=unit, tuning_angles_deg=rf_map.y_positions, tuning_rate=rates)
            silent = Path(temporary) / f"{unit}.mp4"
            rendered = video.export_ebc_overlay(
                unit_data, source_video, silent, start_s=args.start, duration_s=args.duration,
                workers=args.workers, video_encoder="auto", overlay_type=TuningOverlay,
            )
            if tracks is None:
                clock = video._video_audio_clock(data, rendered)
                tracks = video._cache_continuous_audio(
                    source, sorted({channels[u] for u in units}), clock, rendered["duration_s"], temporary,
                    band_hz=(300., 6000.),
                )
            target = output / f"{unit}.mp4"
            track = tracks[channels[unit]]
            video._mux_continuous_audio(silent, target, unit, args.probe, track, rendered["duration_s"], .35, 3., 4.)
            preview = output / f"{unit}.png"
            Path(rendered["preview"]).replace(preview)
            peak_index = int(np.nanargmax(rates))
            rendered.update(video=str(target), preview=str(preview), probe=args.probe, unit_id=unit,
                            wall_config=args.wall_config, hd_source=data["hd_path"],
                            position_columns=data["position_columns"], ebc_source=str(map_path),
                            angle_convention="HD north-zero CW; EBC CCW relative to head",
                            peak_ebc_bin_deg=float(rf_map.y_positions[peak_index]),
                            tuning_projection="sum of saved Hz over distance bins; radius normalized per unit",
                            ray_bearings_deg=[float(rf_map.y_positions[peak_index])],
                            camera_timing=data["camera_timing"],
                            audio=dict(source=str(source["binary"]), channel_zero_based=channels[unit],
                                       band_hz=[300., 6000.], sample_rate_hz=48000,
                                       gain=.35, gate_sigma=3., expander_ratio=4.,
                                       noise_sigma_uv=track["noise_sigma"]))
            target.with_suffix(".json").write_text(json.dumps(rendered, indent=2) + "\n")
            results.append(rendered)
            print(f"Saved {target}", flush=True)
    return results


if __name__ == "__main__":
    main()
