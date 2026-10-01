"""Decode neural head direction with Pynapple on hhw9l84.

Train on the first baseline by default, then decode the camera-covered recording.
The output has the tracking JSON schema, with ``internal_direction`` as its key.
Angles inherit the input convention; no clockwise/counterclockwise flip is made.
This is a fitted population estimate, not a held-out accuracy evaluation.

Example (run with ~/.virtualenvs/rfmapping/bin/python on hhw9l84)::

    decode_internal_head_direction.py /path/to/session --probe A

See --help for unit selection, training/decoding intervals, and Basler input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap

from Utils.json_tools import read_formatted_json
from Utils.load_files import get_interval_pairs
from Utils.tuning_curve_utils import get_exposure_timestamps, make_head_direction_tsd


def load_head_direction(path, headplate, exposure_times, *, basler=False):
    source = read_formatted_json(path)[headplate]
    frames = np.asarray(source["frames"])
    angle_key = "head_direction_deg" if "head_direction_deg" in source else "hd"
    angles = np.asarray(source[angle_key], dtype=float)
    if (
        frames.ndim != 1 or not frames.size or frames.dtype.kind not in "iu"
        or angles.shape != frames.shape or np.any(frames < 0)
        or not np.all(np.diff(frames) > 0)
    ):
        raise ValueError("HD frames must be increasing nonnegative integers paired with angles.")
    if not basler and (
        not np.array_equal(frames, np.arange(len(frames)))
        or len(frames) not in (len(exposure_times), len(exposure_times) + 1)
    ):
        raise ValueError("Motive requires consecutive frames matching TTLs (one extra trailing frame is allowed).")
    if basler and frames[-1] >= len(exposure_times):
        raise ValueError("Basler frame IDs exceed the camera TTL range.")
    synced = frames < len(exposure_times)
    feature, fs = make_head_direction_tsd(exposure_times, frames[synced], angles[synced])
    return frames, synced, feature, fs


def load_spikes(session_dir, kilosort_dir, probe, origin, support, *, hd_class=3,
                all_good=False, units=None):
    labels = pd.read_csv(kilosort_dir / "cluster_KSLabel.tsv", sep="\t")
    good = labels.loc[labels["KSLabel"].str.lower() == "good", "cluster_id"].to_numpy(dtype=int)
    selection = {"mode": "all_good"}
    if units is not None:
        selected = np.unique(units)
        if not np.isin(selected, good).all():
            raise ValueError("Every requested unit must have a good KSLabel in the selected probe.")
        selection = {"mode": "explicit"}
    elif all_good:
        selected = np.sort(good)
    else:
        tc_path = session_dir / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
        tc = read_formatted_json(tc_path)
        hd_ids = np.asarray(tc["unit_id"])[np.asarray(tc["unit_data"]["hd_class"], dtype=object) == hd_class]
        selected = np.intersect1d(good, hd_ids)
        selection = {"mode": "saved_hd_class", "hd_class": hd_class,
                     "source": str(tc_path), "source_metadata": tc.get("metadata", {})}
    if not len(selected):
        raise ValueError("No good units match the requested selection.")

    times = np.load(session_dir / "data" / f"probe{probe}" / "adc_spike_time.npy", mmap_mode="r").reshape(-1)
    clusters = np.load(kilosort_dir / "spike_clusters.npy", mmap_mode="r").reshape(-1)
    if times.shape != clusters.shape:
        raise ValueError("adc_spike_time.npy and spike_clusters.npy must have the same length.")
    keep = np.isin(clusters, selected)
    selected_times = np.asarray(times[keep], dtype=float) - origin
    selected_clusters = clusters[keep]
    order = np.argsort(selected_clusters, kind="stable")
    selected_clusters, selected_times = selected_clusters[order], selected_times[order]
    left = np.searchsorted(selected_clusters, selected, side="left")
    right = np.searchsorted(selected_clusters, selected, side="right")
    spikes = nap.TsGroup({
        int(unit): nap.Ts(t=selected_times[start:end], time_support=support)
        for unit, start, end in zip(selected, left, right)
    }, time_support=support)
    selection["unit_ids"] = selected.tolist()
    return spikes, selection


def fit_tuning_curves(spikes, feature, train_epochs, feature_fs, bins):
    train_epochs = train_epochs.intersect(feature.time_support)
    times = feature.index.to_numpy()
    occupied = (np.searchsorted(times, train_epochs.end, side="right")
                > np.searchsorted(times, train_epochs.start, side="left"))
    train_epochs = train_epochs[occupied]
    if not len(train_epochs) or not np.any(spikes.count(ep=train_epochs).values):
        raise ValueError("Training intervals contain no valid HD samples or selected-unit spikes.")
    curves = nap.compute_tuning_curves(
        data=spikes, features=feature, bins=bins, range=(0.0, 360.0),
        epochs=train_epochs, fs=feature_fs,
    )
    angle_dim = next(dim for dim in curves.dims if dim != "unit")
    curves = curves.transpose("unit", angle_dim)
    occupancy = np.asarray(curves.attrs["occupancy"])
    # Pynapple's nansum likelihood otherwise lets unvisited (NaN) angles win.
    visited = (occupancy > 0) & np.isfinite(curves.values).all(axis=0)
    if np.count_nonzero(visited) < 2:
        raise ValueError("Training must cover at least two finite angle bins.")
    curves = curves.isel({angle_dim: visited})
    curves.attrs["occupancy"] = occupancy[visited]
    return curves, train_epochs


def decode_frames(spikes, curves, epochs, frame_times, bin_size, *, chunk_bins=None):
    """Assign each frame its full, half-open spike-count bin's MAP angle.

    Decode independently across camera pauses. Terminal partial bins and silent
    population bins stay NaN; never interpolate circular angles or extend an epoch.
    """
    values = np.full(len(frame_times), np.nan)
    # decode_bayes tiles time x angle x unit arrays; bound that working set.
    if chunk_bins is None:
        chunk_bins = max(1, min(600, 2_000_000 // curves.size))
    total_bins = valid_bins = 0
    for start, end in epochs.values:
        n_bins = int(np.floor((end - start) / bin_size + 1e-9))
        for offset in range(0, n_bins, chunk_bins):
            count = min(chunk_bins, n_bins - offset)
            chunk_start = start + offset * bin_size
            chunk_end = start + (offset + count) * bin_size
            chunk = nap.IntervalSet(start=chunk_start, end=chunk_end)
            decoded, posterior = nap.decode_bayes(
                tuning_curves=curves, data=spikes, epochs=chunk,
                bin_size=bin_size, uniform_prior=True,
            )
            counts = spikes.count(bin_size, ep=chunk).values.sum(axis=1)
            angles = np.asarray(decoded.values, dtype=float).copy()
            angles[(counts == 0) | ~np.isfinite(posterior.values).all(axis=1)] = np.nan
            first, last = np.searchsorted(frame_times, np.round([chunk_start, chunk_end], 9), side="left")
            # Counting rounds boundaries to 9 decimals; match that half-open rule.
            right_edges = np.round(decoded.index.to_numpy() + bin_size / 2, 9)
            indices = np.searchsorted(right_edges, frame_times[first:last], side="right")
            in_bin = indices < len(angles)
            rows = np.arange(first, last)[in_bin]
            values[rows] = angles[indices[in_bin]] % 360.0
            total_bins += len(angles)
            valid_bins += int(np.isfinite(angles).sum())
    return values, {"decoded_bins": total_bins, "valid_bins": valid_bins}


def build_payload(frames, angles):
    return {"internal_direction": {
        "frames": frames.tolist(),
        "head_direction_deg": [float(value) if np.isfinite(value) else None for value in angles],
    }}


def run(args):
    session_dir = args.session_dir.resolve()
    data_dir = session_dir / "data"
    hd_path = (args.head_direction or data_dir / "processed/head_direction.json").resolve()
    output = (args.output or data_dir / "processed/internal_head_direction.json").resolve()
    metadata_path = output.with_suffix(".metadata.json")
    if output == hd_path or metadata_path == hd_path:
        raise ValueError("The output must not overwrite the tracking head-direction input.")
    for path in (output, metadata_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; use --overwrite to replace decoder outputs.")
    kilosort_dir = args.kilosort_dir
    if kilosort_dir is None:
        matches = sorted((session_dir / "kilosort" / f"Probe{args.probe}").glob("kilosort_*"))
        if len(matches) != 1:
            raise ValueError("Specify --kilosort-dir when the probe does not have exactly one kilosort directory.")
        kilosort_dir = matches[0]
    session_info = read_formatted_json(data_dir / "session_info.json")["session_info"]
    exposure_times, origin, timing_qc = get_exposure_timestamps(
        session_info, data_dir,
        camera_input_channel=args.camera_channel if args.camera_channel is not None else (6 if args.basler else 1),
        camera_ttl_active_high=not args.basler,
    )
    frames, synced, feature, fs = load_head_direction(hd_path, args.headplate, exposure_times, basler=args.basler)
    # Neural decoding does not require a valid tracked angle outside training.
    camera_clock, _ = make_head_direction_tsd(
        exposure_times, np.arange(len(exposure_times)), np.zeros(len(exposure_times)),
    )
    spikes, selection = load_spikes(
        session_dir, kilosort_dir, args.probe, origin, camera_clock.time_support,
        hd_class=args.hd_class, all_good=args.all_good, units=args.units,
    )
    if args.train_interval is None:
        pairs = get_interval_pairs(pd.read_csv(data_dir / "interval_table.csv"), phase_key=args.train_phase)
        if not pairs:
            raise ValueError(f"No {args.train_phase!r} interval found; supply --train-interval START END.")
        train_pair = pairs[0]
    else:
        train_pair = args.train_interval
    curves, train_epochs = fit_tuning_curves(
        spikes, feature, nap.IntervalSet(start=train_pair[0], end=train_pair[1]), fs, args.bins,
    )
    decode_epochs = camera_clock.time_support
    if args.decode_interval is not None:
        decode_epochs = decode_epochs.intersect(nap.IntervalSet(
            start=args.decode_interval[0], end=args.decode_interval[1],
        ))
    angles = np.full(len(frames), np.nan)
    angles[synced], decode_qc = decode_frames(
        spikes, curves, decode_epochs, exposure_times[frames[synced]], args.bin_size,
    )
    if not np.any(np.isfinite(angles)):
        raise ValueError("No frame could be decoded; check intervals, bin size, and selected-unit spikes.")
    metadata = {
        "method": "pynapple.decode_bayes", "pynapple_version": nap.__version__,
        "session_dir": str(session_dir), "probe": args.probe, "kilosort_dir": str(kilosort_dir.resolve()),
        "head_direction_source": str(hd_path), "headplate": args.headplate,
        "angle_convention": "same as input head direction, degrees modulo 360",
        "timebase": "open_ephys_adc_t0_relative_seconds", "adc_time_origin_raw_s": origin,
        "unit_selection": selection, "training_intervals_s": train_epochs.values.tolist(),
        "decoding_intervals_s": decode_epochs.values.tolist(), "bin_size_s": args.bin_size,
        "angle_bins": args.bins, "trained_angle_centers_deg": curves.coords[curves.dims[1]].values.tolist(),
        "uniform_prior": True, "feature_fs_hz": fs, "timing_qc": timing_qc,
        "frame_count": len(frames), "valid_frame_count": int(np.isfinite(angles).sum()),
        "null_policy": "unsynced, outside decoded full bins, silent population, or invalid posterior",
        "evaluation": "Training and decoding may overlap; this export is not held-out validation.",
        **decode_qc,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_payload(frames, angles), allow_nan=False, indent=2) + "\n")
    metadata_path.write_text(json.dumps(metadata, allow_nan=False, indent=2) + "\n")
    print(f"Saved {output}: {metadata['valid_frame_count']}/{len(frames)} frames, {len(spikes)} units.")
    print(f"Decoder settings and provenance: {metadata_path}")
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--probe", default="A")
    parser.add_argument("--headplate", default="hp4", help="Input JSON entity; output is always internal_direction.")
    parser.add_argument("--head-direction", type=Path, help="Default: SESSION/data/processed/head_direction.json")
    parser.add_argument("--kilosort-dir", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--hd-class", type=int, choices=(0, 1, 2, 3), default=3, help="Read this saved hd_class (default: 3).")
    selection.add_argument("--all-good", action="store_true", help="Use all good units without reading saved HD classes.")
    selection.add_argument("--units", type=int, nargs="+", help="Explicit good unit IDs for this probe.")
    parser.add_argument("--train-phase", default="baseline", help="Use the first matching interval (default: baseline).")
    parser.add_argument("--train-interval", type=float, nargs=2, metavar=("START", "END"), help="Override training interval, ADC-relative seconds.")
    parser.add_argument("--decode-interval", type=float, nargs=2, metavar=("START", "END"), help="Limit decoding, ADC-relative seconds; other frames are null.")
    parser.add_argument("--bins", type=int, default=60, help="Number of HD tuning bins over 0..360 degrees (default: 60).")
    parser.add_argument("--bin-size", type=float, default=0.1, help="Spike-count bin duration in seconds (default: 0.1).")
    parser.add_argument("--basler", action="store_true", help="Allow sparse frames/hd input; default to camera channel 6, active low.")
    parser.add_argument("--camera-channel", type=int, help="Zero-based ADC channel override (Motive: 1; Basler: 6).")
    parser.add_argument("--output", type=Path, help="Default: SESSION/data/processed/internal_head_direction.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing decoder output and its metadata sidecar.")
    args = parser.parse_args(argv)
    if not np.isfinite(args.bin_size) or args.bin_size <= 0 or args.bins < 2:
        parser.error("--bin-size must be finite and positive; --bins must be at least 2.")
    for pair in (args.train_interval, args.decode_interval):
        if pair is not None and (not np.isfinite(pair).all() or pair[1] <= pair[0]):
            parser.error("Intervals require finite START < END, in ADC-relative seconds.")
    return args


if __name__ == "__main__":
    run(parse_args())
