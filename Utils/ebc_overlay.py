"""Prepare overlay coordinates separately from drawing recorded video frames."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Utils.ebc_geometry import project_points


@dataclass(frozen=True)
class OverlayInfo:
    positions_px: np.ndarray
    endpoints_px: np.ndarray
    heading_tips_px: np.ndarray
    distances_cm: np.ndarray
    bearings_deg: np.ndarray
    xyz_cm: np.ndarray
    hd_deg: np.ndarray
    times_s: np.ndarray
    valid: np.ndarray
    synchronized: np.ndarray
    pose_valid: np.ndarray
    outside_boundary: np.ndarray
    boundary_paths_px: tuple
    boundary_label: str
    target_px: np.ndarray | None = None
    target_bearing_deg: np.ndarray | None = None


def prepare_overlay_info(position, boundary, ebc, *, heading_length_cm=5., target_xyz_cm=None):
    """Project already calculated rays; no loading, selection or EBC calculation."""
    radians = np.deg2rad(position.hd_deg)
    direction = np.c_[np.cos(radians), np.sin(radians), np.zeros(len(radians))]
    points = project_points(position.xyz_cm, boundary.projection)
    endpoints = project_points(ebc.endpoints_cm, boundary.projection)
    tips = project_points(position.xyz_cm + heading_length_cm * direction, boundary.projection)
    valid = ebc.valid & np.isfinite(points).all(axis=1) & np.isfinite(tips).all(axis=1)
    valid &= np.isfinite(endpoints).all(axis=(1, 2))
    target_px, target_bearing = None, None
    if target_xyz_cm is not None:
        target_xyz_cm = np.asarray(target_xyz_cm)
        target_px = project_points(target_xyz_cm, boundary.projection)
        delta = target_xyz_cm[:, :2] - position.xyz_cm[:, :2]
        target_bearing = (np.rad2deg(np.arctan2(delta[:, 1], delta[:, 0])) - position.hd_deg + 180.) % 360. - 180.
        target_bearing[~valid | ~np.isfinite(target_px).all(axis=1)] = np.nan
    return OverlayInfo(
        points, endpoints, tips, ebc.distances_cm, ebc.bearings_deg,
        position.xyz_cm, position.hd_deg, position.times_s, valid,
        np.isfinite(position.times_s),
        np.isfinite(position.xyz_cm).all(axis=1) & np.isfinite(position.hd_deg),
        position.valid & ~ebc.valid,
        tuple(project_points(path, boundary.projection) for path in boundary.outline_xyz_cm),
        boundary.metadata.get("label", "Arena boundary"), target_px, target_bearing,
    )


class EBCOverlay:
    """Draw one prepared OverlayInfo, independently of pose source and arena shape."""

    colors = ("#007ea8", "#d16b12", "#16854d", "#c83055",
              "#8051c0", "#9c7a00", "#168d91", "#b64191")

    def __init__(self, data, width, height, fps):
        self.info = data["overlay_info"]
        self.valid = self.info.valid
        self.width, self.height, self.fps = width, height, fps
        self.size = (width + 370, max(height, 760))
        self.font = ImageFont.truetype("DejaVuSans.ttf", 18)
        self.small = ImageFont.truetype("DejaVuSans.ttf", 15)
        self.title = ImageFont.truetype("DejaVuSans.ttf", 24)
        self.template = Image.new("RGB", self.size, "white")
        draw = ImageDraw.Draw(self.template)
        x = width + 18
        draw.text((x, 20), "EBC boundary rays", font=self.title, fill="#17212b")
        draw.text((x, 58), f"{data['session'].name} / {data['phase']}", font=self.small, fill="#17212b")
        draw.text((x, 87), self.info.boundary_label, font=self.small, fill="#17212b")
        draw.text((x, 260), "Bearing from HD       Distance", font=self.font, fill="#17212b")
        for i, (bearing, color) in enumerate(zip(self.info.bearings_deg, self.colors)):
            y = 298 + 34 * i
            draw.line((x, y + 10, x + 22, y + 10), fill=color, width=4)
            draw.text((x + 34, y), f"{bearing:g}°", font=self.font, fill="#17212b")
        draw.text((x, 616), "0° front / 90° left / 180° back", font=self.small, fill="#17212b")
        draw.text((x, 644), "HD: world +X zero, positive CCW", font=self.small, fill="#17212b")

    def draw(self, frame, frame_id, row):
        canvas = self.template.copy()
        canvas.paste(frame, (0, 0))
        # Keep annotations clipped to the original image, away from the sidebar.
        picture = frame.copy()
        draw = ImageDraw.Draw(picture)
        info = self.info
        valid = row >= 0 and info.valid[row]
        synchronized = row >= 0 and info.synchronized[row]
        if valid:
            for path in info.boundary_paths_px:
                if np.isfinite(path).all():
                    draw.line([tuple(point) for point in path], fill="#168d91", width=2)
            origin = info.positions_px[row]
            for bearing, end, color in zip(info.bearings_deg, info.endpoints_px[row], self.colors):
                draw.line((tuple(origin), tuple(end)), fill=color, width=3)
                draw.ellipse((end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4), fill=color)
                draw.text(tuple(end), f"{bearing:g}°", font=self.small, fill=color,
                          stroke_width=2, stroke_fill="white")
            tip = info.heading_tips_px[row]
            draw.line((tuple(origin), tuple(tip)), fill="white", width=8)
            draw.line((tuple(origin), tuple(tip)), fill="#df2697", width=4)
            draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill="#df2697")
            if info.target_px is not None and np.isfinite(info.target_bearing_deg[row]):
                draw.line((tuple(origin), tuple(info.target_px[row])), fill="#d68b00", width=4)
            canvas.paste(picture, (0, 0))
        draw = ImageDraw.Draw(canvas)
        x = self.width + 18
        draw.text((x, 126), f"AVI frame {frame_id}", font=self.small, fill="#17212b")
        adc = f"{info.times_s[row]:.3f} s" if synchronized else "unavailable"
        draw.text((x, 152), f"ADC time: {adc}", font=self.small, fill="#17212b")
        if valid:
            xyz = info.xyz_cm[row]
            draw.text((x, 186), f"Position: {xyz[0]:.2f}, {xyz[1]:.2f} cm", font=self.small, fill="#17212b")
            draw.text((x, 218), f"HD: {info.hd_deg[row]:.1f}°", font=self.font, fill="#17212b")
        for i in range(len(info.bearings_deg)):
            value = f"{info.distances_cm[row, i]:.2f} cm" if valid else "—"
            draw.text((x + 195, 298 + 34 * i), value, font=self.font, fill="#17212b")
        if info.target_bearing_deg is not None:
            bearing = f"{info.target_bearing_deg[row]:.1f}°" if valid and np.isfinite(info.target_bearing_deg[row]) else "—"
            draw.text((x, 585), f"VS bearing: {bearing}", font=self.font, fill="#17212b")
        status = "Geometry valid" if valid else ("Geometry unavailable" if synchronized else "Sync unavailable")
        draw.text((x, 686), status, font=self.small, fill="#17212b" if valid else "#b3261e")
        return canvas
