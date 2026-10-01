"""Render calibrated screen, tracking, stimulus, and saved EBC geometry.

Run this script on hhw9l84 with the rfmapping virtual environment. The input
is the version-1 JSON exported for the synchronized screen/EBC viewer.
"""

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


TEXT = "#17212b"
SCREEN = "#657581"
MOUSE = "#cf3849"
EYE = "#008e96"
VS = "#903db2"
BOUNDARY_COLORS = ("#2463ad", "#c26b18")


def circle_distances(position, heading_deg, center, radius, bearings_deg):
    """Forward ray distances in world cm; the observer must be inside."""
    relative = np.asarray(position, dtype=float) - center
    angles = np.deg2rad(heading_deg + np.asarray(bearings_deg))
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    if not np.isfinite(relative).all() or np.dot(relative, relative) > radius**2 + 1e-8:
        return np.full(len(angles), np.nan), directions
    projection = directions @ relative
    distances = -projection + np.sqrt(np.maximum(
        projection**2 + radius**2 - np.dot(relative, relative), 0,
    ))
    return np.maximum(distances, 0), directions


def signed_bearing(position, heading_deg, points):
    """CCW relative bearings, with 0 forward and +90 to the animal's left."""
    delta = np.asarray(points) - position
    angle = np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0])) - heading_deg
    return (angle + 180) % 360 - 180


class ScreenEBCFigure:
    """Reuse static axes, heatmaps, and the time strip while drawing frames."""

    def __init__(self, data, start_s, end_s, *, width=1600, height=900):
        self.data = data
        self.meta = data["meta"]
        self.screen = data["screen"]
        self.models = data["ebc"]
        self.trials = data["trials"]
        self.trial_onsets = np.asarray([trial["on_s"] for trial in self.trials])
        self.frames = {
            name: np.asarray(values, dtype=bool if name == "valid" else float)
            for name, values in data["frames"].items()
        }
        self.times = self.frames["time_s"]
        self.start_s, self.end_s = start_s, end_s
        self.curve_bearings = np.linspace(-180, 180, 241)
        self.ray_bearings = np.arange(0., 360., 45.)
        self.artists = []

        plt.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
            "savefig.facecolor": "white", "savefig.transparent": False,
            "text.color": TEXT, "axes.labelcolor": TEXT,
            "xtick.color": TEXT, "ytick.color": TEXT, "axes.edgecolor": "#82909b",
            "font.size": 10, "font.family": "DejaVu Sans",
        })
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        self.fig.suptitle(
            f"{self.meta['session']}  ·  Probe {self.meta['probe']}  ·  Unit {self.meta['unit_id']}",
            x=.045, y=.975, ha="left", fontsize=18, fontweight="semibold",
        )
        self.fig.text(.045, .932,
                      "World: calibrated cm  ·  EBC: saved model distances; headplate origin",
                      fontsize=11)
        grid = self.fig.add_gridspec(2, 3, left=.045, right=.975, bottom=.12, top=.89,
                                    width_ratios=(1.8, 1, 1), height_ratios=(1, .14),
                                    wspace=.32, hspace=.35)
        self.world = self.fig.add_subplot(grid[0, 0])
        self.maps = [self.fig.add_subplot(grid[0, 1]), self.fig.add_subplot(grid[0, 2])]
        self.timeline = self.fig.add_subplot(grid[1, :])
        self._setup_world()
        self._setup_maps()
        self._setup_timeline()
        self.status = self._animated(self.fig.text(.045, .048, "", fontsize=11))
        self.fig.text(.045, .019,
                      "CCW bearing: 0° forward, +90° left  ·  Tracking gaps retain elapsed time",
                      fontsize=9, color="#4f606d")
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _animated(self, artist):
        artist.set_animated(True)
        self.artists.append(artist)
        return artist

    def _setup_world(self):
        ax = self.world
        center = np.asarray(self.screen["center_cm"], dtype=float)
        radius = self.screen["radius_cm"]
        limits = [np.r_[center - radius, center + radius]]
        ax.add_patch(Circle(center, radius, fill=False, color=SCREEN, lw=2, label="Screen"))
        for i, model in enumerate(self.models):
            point = np.asarray(model["center_cm"], dtype=float)
            r = model["radius_cm"]
            limits.append(np.r_[point - r, point + r])
            ax.add_patch(Circle(point, r, fill=False, color=BOUNDARY_COLORS[i], lw=1.3,
                                ls="--", label=f"{model['name'].title()} EBC model"))
            ax.plot(*point, marker="+", ms=5, color=BOUNDARY_COLORS[i])
        limits = np.asarray(limits)
        low = limits[:, :2].min(axis=0)
        high = limits[:, 2:].max(axis=0)
        midpoint = (low + high) / 2
        extent = (high - low).max() * .54
        ax.set(xlim=(midpoint[0] - extent, midpoint[0] + extent),
               ylim=(midpoint[1] - extent, midpoint[1] + extent),
               xlabel="Screen-world X (cm)", ylabel="Screen-world Y (cm)",
               title="Calibrated screen and current pose", aspect="equal")
        ax.axhline(0, color="#edf0f2", lw=.8, zorder=0)
        ax.axvline(0, color="#edf0f2", lw=.8, zorder=0)
        ax.legend(loc="lower left", fontsize=8, framealpha=1, facecolor="white",
                  edgecolor="#dce2e6")
        self.trace, = ax.plot([], [], color=MOUSE, alpha=.3, lw=1.1)
        self._animated(self.trace)
        self.model_rays = []
        for color in BOUNDARY_COLORS:
            rays = LineCollection([], colors=color, linewidths=.75, alpha=.35)
            ax.add_collection(rays)
            self.model_rays.append(self._animated(rays))
        self.vs_arc, = ax.plot([], [], color=VS, lw=6, solid_capstyle="round")
        self._animated(self.vs_arc)
        self.vs_rays = LineCollection([], colors=VS, linewidths=1.5, linestyles="dashed")
        ax.add_collection(self.vs_rays)
        self._animated(self.vs_rays)
        self.eye_connector, = ax.plot([], [], color=EYE, lw=1)
        self._animated(self.eye_connector)
        self.mouse, = ax.plot([], [], "o", color=MOUSE, ms=7, mec="white", mew=1)
        self.eye, = ax.plot([], [], "o", color=EYE, ms=4, mec="white", mew=.6)
        self._animated(self.mouse)
        self._animated(self.eye)
        self.heading = FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>",
                                       mutation_scale=16, color=MOUSE, lw=1.8)
        ax.add_patch(self.heading)
        self._animated(self.heading)
        self.world_label = self._animated(ax.text(.02, .98, "", transform=ax.transAxes,
                                                  va="top", fontsize=9,
                                                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95}))

    def _setup_maps(self):
        finite_rates = np.concatenate([
            np.asarray(model["rate_hz"], dtype=float).ravel() for model in self.models
        ])
        finite_rates = finite_rates[np.isfinite(finite_rates)]
        vmax = max(float(finite_rates.max()) if len(finite_rates) else 0., .01)
        norm = Normalize(0, vmax)
        cmap = matplotlib.colormaps["viridis"].copy()
        cmap.set_bad("#edf0f2")
        self.curves, self.bearing_lines = [], []
        for i, (ax, model) in enumerate(zip(self.maps, self.models)):
            heatmap = ax.pcolormesh(model["distance_edges_cm"], model["bearing_edges_deg"],
                                   np.ma.masked_invalid(np.asarray(model["rate_hz"], dtype=float)),
                                   cmap=cmap, norm=norm, shading="flat", rasterized=True)
            ax.set(xlim=(model["distance_edges_cm"][0], model["distance_edges_cm"][-1]),
                   ylim=(-180, 180), yticks=(-180, -90, 0, 90, 180),
                   xlabel="EBC distance (model cm)",
                   ylabel="Egocentric bearing (°)" if i == 0 else "",
                   title=f"{model['name'].title()} EBC model")
            ax.set_title(f"{model['name'].title()} EBC model", pad=26)
            ax.grid(axis="y", color="white", alpha=.2, lw=.7)
            # A black outline keeps the current geometry visible over any rate.
            outline, = ax.plot([], [], color="#142129", lw=3.2)
            curve, = ax.plot([], [], color="white", lw=1.6)
            self.curves.append((self._animated(outline), self._animated(curve)))
            lines = [self._animated(ax.axhline(0, color="#e866e8", lw=1.2, ls="--"))
                     for _ in range(2)]
            self.bearing_lines.append(lines)
            ax.text(.5, 1.005, "White: current boundary distance", transform=ax.transAxes,
                    ha="center", fontsize=8, color="#4f606d")
        colorbar_ax = self.fig.add_axes([.623, .253, .352, .015])
        self.fig.colorbar(heatmap, cax=colorbar_ax, orientation="horizontal", label="Firing rate (Hz)")
        colorbar_ax.xaxis.set_ticks_position("bottom")
        colorbar_ax.tick_params(labelsize=8, pad=2)

    def _setup_timeline(self):
        ax = self.timeline
        spikes = np.asarray(self.data["spikes_s"], dtype=float)
        spikes = spikes[(spikes >= self.start_s) & (spikes <= self.end_s)]
        ax.vlines(spikes, .25, .9, color="#273b4a", linewidth=.65)
        polygons, colors = [], []
        for trial in self.trials:
            if trial["off_s"] <= self.start_s or trial["on_s"] >= self.end_s:
                continue
            left, right = max(trial["on_s"], self.start_s), min(trial["off_s"], self.end_s)
            polygons.append([(left, -.35), (right, -.35), (right, -.02), (left, -.02)])
            luminance = float(trial["luminance"])
            gray = min(max(luminance / 255 if luminance > 1 else luminance, 0), 1)
            colors.append((gray, gray, gray, 1))
        ax.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors="#9aabb7", linewidths=.25))
        ax.set(xlim=(self.start_s, self.end_s), ylim=(-.45, 1), yticks=[-.18, .57],
               yticklabels=["VS", "Spikes"], xlabel="Seconds from ADC start")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        self.time_cursor = self._animated(ax.axvline(self.start_s, color=MOUSE, lw=1.5))

    def row_at(self, time_s):
        right = min(np.searchsorted(self.times, time_s), len(self.times) - 1)
        left = max(right - 1, 0)
        return left if abs(self.times[left] - time_s) <= abs(self.times[right] - time_s) else right

    def _trial_at(self, time_s):
        index = int(np.searchsorted(self.trial_onsets, time_s, side="right") - 1)
        if index < 0 or index >= len(self.trials):
            return None, index
        trial = self.trials[index]
        return (trial if trial["on_s"] <= time_s < trial["off_s"] else None), index

    def draw(self, time_s):
        row = self.row_at(time_s)
        frame = self.frames
        valid = bool(frame["valid"][row])
        position = np.array([frame["x_cm"][row], frame["y_cm"][row]])
        eye = np.array([frame["eye_x_cm"][row], frame["eye_y_cm"][row]])
        heading = frame["heading_deg"][row]
        trial, trial_index = self._trial_at(time_s)
        self.time_cursor.set_xdata([time_s, time_s])

        trace_start = np.searchsorted(self.times, time_s - 3)
        trace_x = frame["x_cm"][trace_start:row + 1].copy()
        trace_y = frame["y_cm"][trace_start:row + 1].copy()
        trace_valid = frame["valid"][trace_start:row + 1]
        trace_x[~trace_valid], trace_y[~trace_valid] = np.nan, np.nan
        self.trace.set_data(trace_x, trace_y)
        self.mouse.set_data([position[0]] if valid else [], [position[1]] if valid else [])
        self.eye.set_data([eye[0]] if valid else [], [eye[1]] if valid else [])
        self.eye_connector.set_data([position[0], eye[0]] if valid else [],
                                    [position[1], eye[1]] if valid else [])
        self.heading.set_visible(valid)
        if valid:
            direction = np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading))])
            self.heading.set_positions(position, position + 7 * direction)

        for i, model in enumerate(self.models):
            if valid:
                distances, _ = circle_distances(position, heading, model["center_cm"],
                                                model["radius_cm"], self.curve_bearings)
                values = distances * model["model_cm_per_world_cm"]
                for curve in self.curves[i]:
                    curve.set_data(values, self.curve_bearings)
                ray_distances, directions = circle_distances(position, heading, model["center_cm"],
                                                             model["radius_cm"], self.ray_bearings)
                endpoints = position + ray_distances[:, None] * directions
                self.model_rays[i].set_segments(np.stack((np.broadcast_to(position, endpoints.shape), endpoints), axis=1))
            else:
                for curve in self.curves[i]:
                    curve.set_data([], [])
                self.model_rays[i].set_segments([])

        edges = None
        if trial is not None:
            center = trial["center_world_deg"]
            half_width = trial["width_deg"] / 2
            angles = np.deg2rad(np.linspace(center - half_width, center + half_width, 61))
            arc = np.asarray(self.screen["center_cm"]) + self.screen["radius_cm"] * np.column_stack((np.cos(angles), np.sin(angles)))
            self.vs_arc.set_data(arc[:, 0], arc[:, 1])
            luminance = float(trial["luminance"])
            is_gray = 0 < luminance < 255 if luminance > 1 else 0 < luminance < 1
            self.vs_arc.set_color("#969696" if is_gray else VS)
            self.vs_arc.set_linestyle("--" if is_gray else "-")
            self.vs_arc.set_alpha(.6 if is_gray else 1)
            self.vs_rays.set_alpha(.35 if is_gray else .85)
            if valid:
                endpoints = arc[[0, -1]]
                self.vs_rays.set_segments(np.stack((np.broadcast_to(eye, endpoints.shape), endpoints), axis=1))
                edges = signed_bearing(position, heading, endpoints)
            else:
                self.vs_rays.set_segments([])
            vs_text = f"VS {trial['screen_deg']:+.1f}° · width {trial['width_deg']:g}° · luminance {luminance:g}"
            if is_gray:
                vs_text += "\nGray trial: assigned location (no contrast)"
        else:
            self.vs_arc.set_data([], [])
            self.vs_rays.set_segments([])
            vs_text = "VS off"
        for lines in self.bearing_lines:
            for j, line in enumerate(lines):
                line.set_visible(edges is not None)
                if edges is not None:
                    line.set_ydata([edges[j], edges[j]])
                    line.set_alpha(.35 if is_gray else 1)

        if valid:
            self.world_label.set_text(
                f"Headplate ({position[0]:+.1f}, {position[1]:+.1f}) cm\n"
                f"HD {heading % 360:.1f}°  ·  Red: headplate  ·  Teal: eye\n{vs_text}")
            tracking = f"Motive frame {int(frame['frame_id'][row])}"
        else:
            self.world_label.set_text(f"Tracking gap: pose and bearings omitted\n{vs_text}")
            tracking = "Tracking gap"
        self.status.set_text(
            f"t = {time_s:.3f} s  ·  {tracking}  ·  Trial {trial_index + 1 if trial_index >= 0 else '—'}"
            "  ·  Purple dashed: VS bearing at headplate; faint on gray trials")
        self.fig.canvas.restore_region(self.background)
        for artist in self.artists:
            self.fig.draw_artist(artist)
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3]

    def save_snapshot(self, time_s, path):
        from PIL import Image

        Image.fromarray(self.draw(time_s)).save(path)

    def close(self):
        plt.close(self.fig)


def render(input_path, output_dir, *, start_s=None, duration_s=None, fps=None):
    with Path(input_path).open() as stream:
        data = json.load(stream)
    meta = data["meta"]
    fps = float(meta["fps"] if fps is None else fps)
    start_s = float(meta["preview_start_s"] if start_s is None else start_s)
    duration_s = float(meta["preview_duration_s"] if duration_s is None else duration_s)
    if fps <= 0 or duration_s <= 0:
        raise ValueError("FPS and duration must be positive.")
    times = np.asarray(data["frames"]["time_s"])
    if start_s < times[0] or start_s + duration_s > times[-1] + 1 / meta["fps"]:
        raise ValueError("The requested clip must lie inside the exported frame interval.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = ScreenEBCFigure(data, start_s, start_s + duration_s)
    frame_count = int(round(duration_s * fps))
    snapshots = []
    for number, offset in enumerate((0, duration_s / 2, (frame_count - 1) / fps)):
        path = output_dir / f"screen_ebc_snapshot_{number + 1}.png"
        figure.save_snapshot(start_s + offset, path)
        snapshots.append(str(path))
    output_path = output_dir / "screen_ebc_preview.mp4"
    width, height = figure.fig.canvas.get_width_height()
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-threads", "8", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(output_path)]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            try:
                for frame_index in range(frame_count):
                    rgb = figure.draw(start_s + frame_index / fps)
                    encoder.stdin.write(rgb.tobytes())
                    if frame_index % max(1, round(fps * 5)) == 0:
                        print(f"Rendered {frame_index}/{frame_count} frames", flush=True)
            finally:
                encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("ffmpeg failed while encoding the screen/EBC preview.")
    finally:
        figure.close()
    result = {"video": str(output_path), "snapshots": snapshots, "frames": frame_count,
              "fps": fps, "start_s": start_s, "duration_s": frame_count / fps,
              "input": str(Path(input_path).resolve()), "unit_id": meta["unit_id"],
              "geometry_note": meta["geometry_note"], "eye_model": meta["eye_model"]}
    (output_dir / "screen_ebc_preview.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result

