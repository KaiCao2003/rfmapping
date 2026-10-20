"""Explicit saved-input readers and output writers for the CB/CD notebook.

Calculations live in center_cells and Utils.theta_analysis. Reading never
regenerates tracking, spike times, or spatial maps.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def file_evidence(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def _load_center_spikes(kilosort_dir, spike_path, times, origin):
    """Keep every good unit, including units silent within the selected interval."""
    labels = pd.read_csv(kilosort_dir / "cluster_KSLabel.tsv", sep="\t")
    units = labels.loc[labels.KSLabel == "good", "cluster_id"].astype(int).to_numpy()
    if len(np.unique(units)) != len(units):
        raise ValueError("Duplicate good-unit IDs.")
    clusters = np.load(kilosort_dir / "spike_clusters.npy", mmap_mode="r").ravel()
    spikes = np.load(spike_path, mmap_mode="r").ravel()
    if spikes.dtype.kind != "f" or spikes.shape != clusters.shape:
        raise ValueError("Expected synchronized floating spike seconds aligned with clusters.")
    # Restrict on the original clock before subtraction to bound memory.
    select = (spikes >= times[0] + origin) & (spikes <= times[-1] + origin) & np.isin(clusters, units)
    selected_times = np.asarray(spikes[select], dtype=float) - origin
    selected_clusters = np.asarray(clusters[select], dtype=int)
    order = np.argsort(selected_clusters, kind="stable")
    selected_times, selected_clusters = selected_times[order], selected_clusters[order]
    ids, starts, lengths = np.unique(selected_clusters, return_index=True, return_counts=True)
    grouped = {int(u): np.sort(selected_times[a:a + n]) for u, a, n in zip(ids, starts, lengths)}
    return {int(u): grouped.get(int(u), np.empty(0)) for u in units}


def load_center_session(session_dir, *, probe="A", phase="baseline"):
    """Read saved rectangular geometry and original spike seconds, all good units.

    Both output clocks are ADC-relative seconds. Saved image XY has Y down;
    ebc_hd_deg is the calibrated north-zero clockwise heading. Missing saved
    inputs raise; no YOLO, timing export, or spatial generation is triggered.
    """
    session_dir = Path(session_dir)
    directory = session_dir / "data" / "spatial_cells" / f"Probe{probe}" / phase
    audit_path = directory / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") not in ("complete", "verified"):
        raise ValueError(f"Incomplete saved spatial input: {audit_path}")
    if audit.get("probe_name", probe) != probe or audit.get("phase_key", phase) != phase:
        raise ValueError("Saved probe/phase differs from the requested input.")
    if audit.get("heading_convention") != "physical north-zero clockwise heading; clockwise egocentric bearings":
        raise ValueError("Input must declare calibrated clockwise heading.")
    if audit.get("heading_calibration") != "image_unit_vector_scaled_by_cm_per_pixel_xy":
        raise ValueError("Input must carry the saved physical heading calibration.")
    with np.load(directory / "restored_maps.npz", allow_pickle=False) as arrays:
        times = arrays["frame_times"].astype(float)
        frames = arrays["frame_ids"].copy()
        xy = np.column_stack((arrays["trajectory_x_cm"], arrays["trajectory_y_cm"]))
        saved_unit_ids = arrays["unit_ids"].copy()
    with np.load(directory / "position_correction.npz", allow_pickle=False) as arrays:
        if not np.array_equal(frames, arrays["frame_ids"]) or not np.array_equal(times, arrays["frame_times"]):
            raise ValueError("Saved geometry and trajectory frame clocks differ.")
        heading = arrays["ebc_hd_deg"].copy()
        raw_heading = arrays["hd_deg"].copy()
        held = arrays["position_held"].astype(bool)
        initial = arrays["initial_outside"].astype(bool)
    source = audit["source"]
    origin = float(source["adc_time_origin_s"])
    kilosort_dir = Path(source["kilosort_dir"])
    spike_path = Path(source["spike_times_path"])
    # Reject stale provenance pointing to a different recording/probe.
    if not kilosort_dir.is_relative_to(session_dir / "kilosort" / f"Probe{probe}"):
        raise ValueError("Kilosort provenance is outside the selected session/probe.")
    if spike_path != session_dir / "data" / f"probe{probe}" / "adc_spike_time.npy":
        raise ValueError("Spike timestamp provenance differs from selected session/probe.")
    by_unit = _load_center_spikes(kilosort_dir, spike_path, times, origin)
    paths = [audit_path, directory / "restored_maps.npz", directory / "position_correction.npz",
             kilosort_dir / "cluster_KSLabel.tsv", kilosort_dir / "spike_clusters.npy", spike_path]
    return dict(session_dir=session_dir, probe=probe, phase=phase, times_s=times,
                xy_cm_y_down=xy, hd_cw_deg=heading, raw_hd_cw_deg=raw_heading,
                arena_size_cm=np.asarray(audit["arena_size_cm"], dtype=float),
                valid_mask=~held & ~initial, spikes_by_unit=by_unit,
                adc_time_origin_s=origin, audit=audit, saved_unit_ids=saved_unit_ids,
                tracking_selection=dict(policy="exclude held and initial-outside frames",
                                        held_frames=int(held.sum()), initial_outside_frames=int(initial.sum()),
                                        excluded_fraction=float(np.mean(held | initial)),
                                        adaptation="Slides retain held positions; this analysis excludes them for speed estimation."),
                provenance=[file_evidence(path) for path in paths])


def load_motive_center_session(session_dir, *, probe="A", phase="baseline"):
    """Read Motive tracking and the saved inner-circle model, without refitting it.

    The indexed NPZ map supplies geometry only, never EBC firing rates or labels.
    Native Motive yaw is CCW from +X; the explicit analysis conversion is
    north-zero CW heading and Y-down positions in cm.
    """
    from Utils.camera_timing import get_exposure_timestamps
    from Utils.json_tools import read_formatted_json
    from Utils.motive_pose import matched_motive_frame_count, motive_z_yaw_degrees

    session_dir = Path(session_dir)
    directory = session_dir / "data"
    geometry_path = directory / "spatial_cells" / f"Probe{probe}" / phase / "inner/egocentric_rate_map.rfmap"
    with np.load(geometry_path, allow_pickle=False) as saved:
        geometry = json.loads(saved["metadata"].tobytes().decode("utf-8"))
    if geometry["boundaryType"] != "circle":
        raise ValueError("Motive center analysis requires a saved circular arena model.")
    center = np.asarray(geometry["boundaryCenterRaw"], dtype=float)
    scale = float(geometry["cmPerPositionUnit"])
    diameter = float(geometry["boundaryDiameterCm"])
    if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite([scale, diameter]).all() or min(scale, diameter) <= 0:
        raise ValueError("Saved circle center and centimetre scale must be finite and valid.")
    pose_path = directory / "processed/filtered.csv"
    if Path(geometry["positionSource"]) != pose_path:
        raise ValueError("Saved circle geometry belongs to a different tracking source.")
    pose = pd.read_csv(pose_path, header=[0, 1, 2, 3])
    frames = pose.iloc[:, 0].to_numpy(int)
    xy = pose.xs("Position", level=2, axis=1).to_numpy(float)[:, :2]
    rotation = pose.xs("Rotation", level=2, axis=1).to_numpy(float)
    heading = motive_z_yaw_degrees(rotation)
    info_path = directory / "session_info.json"
    info = read_formatted_json(info_path)["session_info"]
    exposures, origin, timing = get_exposure_timestamps(info, directory, camera_ttl_active_high=True)
    count = matched_motive_frame_count(frames, exposures, timing)
    times = exposures[frames[:count]]
    interval_path = directory / "interval_table.csv"
    intervals = pd.read_csv(interval_path)
    start, stop = intervals.loc[intervals.interval_type == phase, ["start", "end"]].iloc[0]
    selected = (times >= start) & (times <= stop)
    times = times[selected]
    xy = (xy[:count][selected] - center) * scale
    radius = diameter / 2
    heading = heading[:count][selected]
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(heading) & (np.linalg.norm(xy, axis=1) <= radius + 1e-9)
    session_number = session_dir.name.rsplit("_", 1)[1]
    kilosort_dir = session_dir / "kilosort" / f"Probe{probe}" / f"kilosort_{session_number}"
    spike_path = directory / f"probe{probe}" / "adc_spike_time.npy"
    by_unit = _load_center_spikes(kilosort_dir, spike_path, times, origin)
    paths = [geometry_path, pose_path, interval_path, info_path, Path(timing["source_path"]),
             kilosort_dir / "cluster_KSLabel.tsv", kilosort_dir / "spike_clusters.npy", spike_path]
    return dict(
        session_dir=session_dir, probe=probe, phase=phase, times_s=times,
        xy_cm_y_down=np.column_stack((xy[:, 0] + radius, radius - xy[:, 1])),
        hd_cw_deg=(90. - heading) % 360,
        native_hd_ccw_deg=rotation[:count, 2][selected].copy(),
        arena_size_cm=np.repeat(diameter, 2), arena_shape="circle",
        valid_mask=valid, spikes_by_unit=by_unit, adc_time_origin_s=origin,
        geometry=dict(center_raw=center.tolist(), cm_per_position_unit=scale,
                      diameter_cm=diameter, source=str(geometry_path),
                      interpretation="saved trajectory-derived inner-circle model; not a new physical calibration"),
        tracking_selection=dict(policy="exclude nonfinite pose and samples outside saved circle",
                                excluded_fraction=float(np.mean(~valid)), camera_timing=timing),
        provenance=[file_evidence(path) for path in paths],
    )


def generate_lfp_from_open_ephys(session_dir, *, probe, channel, interval_s,
                                  adc_time_origin_s, target_hz=1000., chunk_s=30.):
    """Explicitly extract one raw electrode and anti-alias downsample in memory.

    Requires the synchronized, unfiltered raw OE stream. Caller chooses a
    zero-based physical channel and verifies acquisition supports low frequencies.
    An AP-only/high-pass recording cannot recover theta. No files are written.
    """
    from scipy.signal import resample_poly
    from Utils.json_tools import read_formatted_json
    from Utils.si_utils import read_open_ephys_probe

    session_dir = Path(session_dir)
    info = read_formatted_json(session_dir / "data/session_info.json")["session_info"]
    recording = read_open_ephys_probe(info, probe, load_sync_timestamps=True)
    raw_file = Path(recording.get_annotation("open_ephys_raw_file"))
    if not raw_file.is_relative_to(session_dir):
        raise ValueError("Raw stream metadata points outside the selected recording session.")
    fs = float(recording.get_sampling_frequency())
    q = int(round(fs / target_hz))
    if q < 1 or not np.isclose(fs / q, target_hz):
        raise ValueError("Choose a target rate that is an integer divisor of raw sample rate.")
    if channel not in recording.channel_ids:
        raise ValueError(f"Raw channel {channel} is absent.")
    raw_times = recording.get_times()
    begin, end = np.asarray(interval_s, dtype=float) + adc_time_origin_s
    start, stop = np.searchsorted(raw_times, [begin, end])
    start = max(0, start // q * q)
    stop = min(len(raw_times), stop)
    if stop - start < fs * 4:
        raise ValueError("Less than four seconds of LFP overlap.")
    # Sampling drift is allowed, gaps and nonmonotonic clocks are not. Check in
    # bounded blocks instead of allocating a recording-length difference array.
    for a in range(start, stop - 1, 1_000_000):
        dt = np.diff(raw_times[a:min(stop, a + 1_000_001)])
        if not np.isfinite(dt).all() or np.any(dt <= 0) or np.any(np.abs(dt * fs - 1) > .05):
            raise ValueError("Raw LFP clock has gaps/nonuniform timing; extract continuous epochs separately.")
    step = max(q, int(chunk_s * fs) // q * q)
    pad = int(fs) // q * q
    output = []
    indices = []
    for a in range(start, stop, step):
        b = min(a + step, stop)
        lo, hi = max(start, a - pad), min(stop, b + pad)
        trace = recording.get_traces(start_frame=lo, end_frame=hi,
                                     channel_ids=[channel], return_in_uV=True).ravel()
        trace = resample_poly(trace.astype(float), 1, q)
        take = np.arange(a, b, q)
        output.append(trace[(take - lo) // q])
        indices.append(take)
    indices = np.concatenate(indices)
    return dict(times_s=np.asarray(raw_times[indices]) - adc_time_origin_s,
                voltage_uv=np.concatenate(output), sample_rate_hz=fs / q,
                metadata=dict(session_dir=str(session_dir), probe=probe, channel=int(channel),
                              clock="ADC-relative seconds", adc_time_origin_s=float(adc_time_origin_s),
                              raw_sample_rate_hz=fs, sample_rate_hz=fs / q,
                              anti_alias="scipy.signal.resample_poly; 1-second block overlap",
                              amplitude_units="uV", source=str(raw_file),
                              session_info=file_evidence(session_dir / "data/session_info.json")))


def save_lfp(path, lfp, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, times_s=lfp["times_s"], voltage_uv=lfp["voltage_uv"],
                            sample_rate_hz=lfp["sample_rate_hz"], metadata=json.dumps(lfp["metadata"]))


def load_lfp(path):
    """Read only a previously saved explicit LFP export."""
    with np.load(path, allow_pickle=False) as arrays:
        return dict(times_s=arrays["times_s"].copy(), voltage_uv=arrays["voltage_uv"].copy(),
                    sample_rate_hz=float(arrays["sample_rate_hz"]), metadata=json.loads(str(arrays["metadata"])))


def save_center_analysis(directory, result, *, provenance, theta_results=None, overwrite=False):
    """Persist tables, raw tuning evidence, shuffled statistics, and provenance."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(directory)
    if overwrite and theta_results is None and any((directory / name).exists() for name in ("theta_cells.csv", "theta_arrays.npz")):
        raise ValueError("Choose a fresh run directory to avoid mixing old theta outputs with a center-only run.")
    directory.mkdir(parents=True, exist_ok=True)
    result["summary"].to_csv(directory / "center_cells.csv", index=False)
    arrays = dict(unit_ids=result["summary"]["unit_id"].to_numpy())
    for name, values in result["nulls"].items():
        arrays["shuffle_" + name] = values
    arrays["shuffle_offsets_s"] = result["shuffle_offsets_s"]
    for unit, curves in result["tuning"].items():
        for feature, values in curves.items():
            for key, value in values.items():
                if isinstance(value, np.ndarray):
                    arrays[f"unit_{unit}_{feature}_{key}"] = value
    np.savez_compressed(directory / "center_tuning_and_nulls.npz", **arrays)
    if theta_results is not None:
        theta_results["summary"].to_csv(directory / "theta_cells.csv", index=False)
        theta_arrays = {}
        for unit, features in theta_results["units"].items():
            for feature, values in features.items():
                for key, value in values.items():
                    if isinstance(value, np.ndarray):
                        theta_arrays[f"unit_{unit}_{feature}_{key}"] = value
        np.savez_compressed(directory / "theta_arrays.npz", **theta_arrays)
    manifest = dict(paper="https://doi.org/10.1038/s41467-024-54699-9",
                    config=result["config"], metadata=result["metadata"],
                    thresholds=result["thresholds"], provenance=provenance,
                    theta_metadata=None if theta_results is None else theta_results.get("metadata"))
    if theta_results is not None:
        manifest["theta_parameters"] = {
            str(unit): {feature: values.get("parameters", {}) for feature, values in features.items()}
            for unit, features in theta_results["units"].items()
        }
    def encode(value):
        if hasattr(value, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    (directory / "manifest.json").write_text(json.dumps(manifest, default=encode, indent=2) + "\n")
    return directory
