"""Export synchronized screen, Motive, VS and saved EBC data for one unit."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from Utils.tuning_curve_utils import get_exposure_timestamps


def nearest_rows(times, targets):
    right = np.clip(np.searchsorted(times, targets), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    return np.where(abs(targets - times[left]) <= abs(targets - times[right]), left, right)


def legacy_eye_xy(position, heading_deg, calibration):
    """Preserve the explicitly selected planar FM eye model, in raw units."""
    if calibration["fm_eye_position_model"] != "xy_offset_rotated_by_calibrated_hd":
        raise ValueError("This exporter supports the calibrated planar FM eye model.")
    angle = np.deg2rad(heading_deg - calibration["screen_center"])
    x, y = calibration["hp_to_eye_center"][:2]
    return position + np.column_stack((x * np.cos(angle) - y * np.sin(angle),
                                      x * np.sin(angle) + y * np.cos(angle)))


def saved_boundary_in_world(metadata, calibration):
    """Invert the saved EBC transform before applying the screen calibration."""
    real_scale = 1 / (10 * calibration["motive_units_per_real_mm"])
    model_scale = metadata["cmPerPositionUnit"]
    center = ((np.asarray(metadata["boundaryCenterRaw"])
               - calibration["screen_bottom_center"][:2]) * real_scale)
    return dict(center_cm=center.tolist(),
                radius_cm=metadata["boundaryDiameterCm"] / 2 / model_scale * real_scale,
                model_cm_per_world_cm=model_scale / real_scale)


def ray_circle_distance(position, direction_deg, center, radius):
    """Forward intersections for origins inside the circle; outside is missing."""
    position = np.asarray(position, dtype=float) - center
    angle = np.deg2rad(direction_deg)
    direction = np.stack((np.cos(angle), np.sin(angle)), axis=-1)
    projection = np.sum(position * direction, axis=-1)
    norm2 = np.sum(position * position, axis=-1)
    distance = -projection + np.sqrt(np.maximum(projection**2 + radius**2 - norm2, 0))
    return np.where(norm2 <= radius**2 + 1e-9, np.maximum(distance, 0), np.nan)


def _json_array(values, decimals=6):
    values = np.asarray(values)
    if values.dtype.kind == "f":
        rounded = np.round(values, decimals).astype(object)
        rounded[~np.isfinite(values)] = None
        return rounded.tolist()
    return values.tolist()


def signed_ebc_display(rate):
    """Center cells on the source's 0, 6, ... ray samples and wrap the seam.

    compute_circle_ebc samples each saved bin's lower edge. Keeping its original
    center labels would displace the heatmap by half a bin against exact rays.
    The duplicated seam row allows clipping at -180/+180 without a missing strip.
    """
    rows = len(rate)
    step = 360 / rows
    signed = np.roll(rate, -(rows // 2), axis=0)
    return np.arange(rows + 2) * step - 180 - step / 2, np.concatenate((signed, signed[:1]))


def export_session(session, unit_id, output, *, probe="A", fps=30., preview_start=None):
    session, output = Path(session), Path(output)
    directory, date = session / "data", session.name.split("_")[0]
    calib_path = session / f"{date}.calib"
    calibration = json.loads(calib_path.read_text())
    info = json.loads((directory / "session_info.json").read_text())["session_info"]
    exposures, origin, timing = get_exposure_timestamps(
        info, directory, camera_ttl_active_high=True,
    )
    pose_path = directory / "processed/filtered.csv"
    pose = pd.read_csv(pose_path, header=[0, 1, 2, 3])
    frame_ids = pose.iloc[:, 0].to_numpy(dtype=int)
    if np.any(np.diff(frame_ids) <= 0) or frame_ids.min() < 0 or frame_ids.max() >= len(exposures):
        raise ValueError("Motive frame IDs do not map to the camera TTL clock.")
    raw_body = pose.xs("Position", level=2, axis=1).to_numpy(float)[:, :2]
    heading = json.loads((directory / "processed/head_direction.json").read_text())["hp4"]
    hd = pd.Series(heading["head_direction_deg"], index=heading["frames"]).reindex(frame_ids).to_numpy()
    raw_csv = pd.read_csv(session / f"{date}.csv", header=[2, 3, 6, 7], skip_blank_lines=False)
    raw_positions = raw_csv["Rigid Body"]["hp4"]["Position"][["X", "Y"]].to_numpy(float)
    raw_frames = raw_csv.iloc[:, 0].to_numpy(dtype=int)
    raw_lookup = pd.DataFrame(raw_positions, index=raw_frames).reindex(frame_ids).to_numpy()
    raw_eye = legacy_eye_xy(raw_lookup, hd, calibration)
    real_scale = 1 / (calibration["motive_units_per_real_mm"] * 10)
    screen_center = np.asarray(calibration["screen_bottom_center"][:2])
    body = (raw_body - screen_center) * real_scale
    eye = (raw_eye - screen_center) * real_scale
    screen_radius = calibration["screen_diamter"] / 2 * real_scale
    source_times = exposures[frame_ids]

    maps = []
    for name in ("inner", "outer"):
        path = directory / "spatial_cells" / f"Probe{probe}" / "baseline" / name / "egocentric_rate_map.rfmap"
        with np.load(path, allow_pickle=False) as z:
            if unit_id not in z["unitPool"]:
                raise ValueError(f"Unit {unit_id} is absent from {path}.")
            metadata = json.loads(z["metadata"].tobytes())
            rate = z[f"unit_{unit_id}"][..., 0]
            interval = np.asarray(z["timeBinEdges"])
        if metadata["angleConvention"] != "egocentric CCW; Motive XY heading is Z yaw from +X":
            raise ValueError("Saved EBC angle convention differs from the supported Motive convention.")
        y_edges = np.asarray(metadata["yBinEdges"])
        if not np.allclose(y_edges, np.linspace(0, 360, len(y_edges))):
            raise ValueError("Saved EBC requires uniform full-circle angle bins.")
        display_edges, display_rate = signed_ebc_display(rate)
        maps.append(dict(name=name, **saved_boundary_in_world(metadata, calibration),
                         distance_edges_cm=metadata["xBinEdges"],
                         bearing_edges_deg=display_edges.tolist(),
                         rate_hz=_json_array(display_rate),
                         ray_sample_location="saved_bin_lower_edge",
                         source_path=str(path), source_metadata=metadata))

    trials = loadmat(session / f"{date}.mat", simplify_cells=True)["trials"]
    edges = np.load(directory / "on_list_times.npy") - origin
    if len(edges) != len(trials) + 1 or not np.all(np.diff(edges) > 0):
        raise ValueError("Trial count and increasing N+1 onset boundaries must match.")
    trial_data = [dict(on_s=float(edges[i]), off_s=float(edges[i + 1]),
                       center_world_deg=float(calibration["screen_center"] - trial["Square_PositionX"]),
                       width_deg=float(trial["Square_Size"]), luminance=float(trial["Square_Luminance"]),
                       screen_deg=float(trial["Square_PositionX"])) for i, trial in enumerate(trials)]

    start, stop = max(source_times[0], interval[0]), min(source_times[-1], interval[-1])
    target_times = start + np.arange(int(np.floor((stop - start) * fps)) + 1) / fps
    rows = nearest_rows(source_times, target_times)
    valid = ((abs(source_times[rows] - target_times) <= 1.5 * np.median(np.diff(exposures)))
             & np.isfinite(body[rows]).all(axis=1) & np.isfinite(eye[rows]).all(axis=1)
             & np.isfinite(hd[rows]))
    trial_indices = np.searchsorted(edges, target_times, side="right") - 1
    trial_indices[(trial_indices < 0) | (trial_indices >= len(trials))] = -1
    view_body, view_eye, view_hd = body[rows].copy(), eye[rows].copy(), hd[rows].copy()
    view_body[~valid], view_eye[~valid], view_hd[~valid] = np.nan, np.nan, np.nan

    kilosort = session / "kilosort" / f"Probe{probe}" / f"kilosort_{session.name.split('_')[-1]}"
    labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t").set_index("cluster_id")
    if labels.loc[unit_id, "KSLabel"] != "good":
        raise ValueError(f"Unit {unit_id} is not labeled good.")
    spike_times = np.load(directory / f"probe{probe}/adc_spike_time.npy", mmap_mode="r").ravel()
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").ravel()
    if len(spike_times) != len(clusters):
        raise ValueError("Spike times and cluster IDs differ in length.")
    spikes = np.sort(spike_times[clusters == unit_id] - origin)
    spikes = spikes[(spikes >= start) & (spikes <= stop)]
    preview_start = float(edges[0] if preview_start is None else preview_start)
    preview_duration = min(60., stop - preview_start)
    if not start <= preview_start < stop or preview_duration <= 0:
        raise ValueError("Preview start must be inside the tracking interval.")

    frames = dict(time_s=_json_array(target_times), pose_time_s=_json_array(source_times[rows]),
                  frame_id=_json_array(frame_ids[rows]), valid=valid.tolist(),
                  x_cm=_json_array(view_body[:, 0]), y_cm=_json_array(view_body[:, 1]),
                  eye_x_cm=_json_array(view_eye[:, 0]), eye_y_cm=_json_array(view_eye[:, 1]),
                  heading_deg=_json_array(view_hd), trial_index=trial_indices.tolist())
    payload = dict(format_version=1,
                   meta=dict(session=f"{session.parent.parent.name}/{session.name}", unit_id=unit_id,
                             probe=probe, fps=fps, time_reference="Seconds from ADC start",
                             adc_origin_s=origin, preview_start_s=preview_start,
                             preview_duration_s=preview_duration,
                             geometry_note="Screen uses calibration; inner/outer retain fitted EBC geometry.",
                             angle_note="CCW positive; 0 forward; +90 left",
                             bearing_display_note="EBC cells are centered on the original 6-degree ray samples, correcting the saved labels' 3-degree offset. The seam row repeats; rate values are unchanged.",
                             stimulus_note="Trials occupy onset-to-next-onset intervals. Gray trials have no bar contrast; their position is an assigned location only.",
                             timing_note="Live pose uses high camera TTL midpoints, matching EBC. Legacy RF trial-pose caches use low midpoints about 4.16 ms later and are not used here.",
                             eye_model=calibration["fm_eye_position_model"],
                             body_source=str(pose_path), eye_position_source=str(session / f"{date}.csv"),
                             calibration_path=str(calib_path),
                             calibration_sha256=hashlib.sha256(calib_path.read_bytes()).hexdigest()),
                   screen=dict(center_cm=[0., 0.], radius_cm=screen_radius,
                               zero_deg=calibration["screen_center"]),
                   frames=frames, trials=trial_data, spikes_s=_json_array(spikes), ebc=maps)

    # Independent endpoint check: convert VS edge bearings back to world rays.
    active = np.flatnonzero(valid & (trial_indices >= 0))
    centers = np.array([trial_data[i]["center_world_deg"] for i in trial_indices[active]])
    widths = np.array([trial_data[i]["width_deg"] for i in trial_indices[active]])
    world_angles = centers[:, None] + widths[:, None] * np.array([-.5, .5])
    endpoints = screen_radius * np.stack((np.cos(np.deg2rad(world_angles)),
                                         np.sin(np.deg2rad(world_angles))), axis=-1)
    delta = endpoints - view_eye[active, None, :]
    ego = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0])) - view_hd[active, None]
    recovered_angles = view_hd[active, None] + ego
    distances = ray_circle_distance(view_eye[active, None, :], recovered_angles, np.zeros(2), screen_radius)
    recovered = view_eye[active, None, :] + distances[..., None] * np.stack(
        (np.cos(np.deg2rad(recovered_angles)), np.sin(np.deg2rad(recovered_angles))), axis=-1)
    endpoint_error = np.linalg.norm(recovered - endpoints, axis=-1)
    geometry_errors = []
    for item in maps:
        model = item["source_metadata"]
        transformed = (body - item["center_cm"]) * item["model_cm_per_world_cm"]
        direct = (raw_body - model["boundaryCenterRaw"]) * model["cmPerPositionUnit"]
        geometry_errors.append(float(np.max(np.abs(transformed - direct))))
    qc = dict(unit_id=unit_id, camera=timing, pose_rows=len(pose), trials=len(trials),
              display_frames=len(target_times), missing_display_frames=int((~valid).sum()),
              display_pose_max_offset_s=float(np.max(abs(source_times[rows[valid]] - target_times[valid]))),
              trial_onsets_outside_tracking=int(((edges[:-1] < source_times[0]) | (edges[:-1] > source_times[-1])).sum()),
              spikes=len(spikes), preview_spikes=int(((spikes >= preview_start) & (spikes < preview_start + preview_duration)).sum()),
              screen_diameter_cm=2 * screen_radius,
              vs_ray_endpoint_max_error_cm=float(np.nanmax(endpoint_error)),
              eye_outside_screen_frames=int((np.linalg.norm(eye, axis=1) >= screen_radius).sum()),
              ebc_transform_max_error_model_cm=max(geometry_errors),
              first_time_s=float(start), last_time_s=float(stop),
              preview_start_s=preview_start, preview_duration_s=preview_duration)
    if qc["eye_outside_screen_frames"] or not np.isfinite(endpoint_error).all() or qc["vs_ray_endpoint_max_error_cm"] > 1e-8:
        raise ValueError("VS screen/eye geometry failed the endpoint check.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    output.with_suffix(".qc.json").write_text(json.dumps(qc, indent=2, allow_nan=False) + "\n")
    print(json.dumps(qc, indent=2))
    return payload, qc
