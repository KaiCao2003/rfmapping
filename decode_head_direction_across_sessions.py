"""Train one Pynapple population decoder across sessions and apply it unchanged.

Run on hhw9l84 with ~/.virtualenvs/rfmapping/bin/python. Sessions must share
the same jointly sorted probe/unit identities. Training uses all valid tracked
camera coverage; decoding uses the target camera clock without target angles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pynapple as nap
import xarray as xr

from decode_internal_head_direction import (
    build_payload, decode_frames, load_head_direction, load_spikes,
)
from Utils.json_tools import read_formatted_json
from Utils.tuning_curve_utils import get_exposure_timestamps, make_head_direction_tsd


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_kilosort(session, probe):
    matches = sorted((session / "kilosort" / f"Probe{probe}").glob("kilosort_*"))
    if len(matches) != 1:
        raise ValueError(f"{session}: Probe {probe} must have exactly one kilosort directory.")
    return matches[0]


def select_training_units(sessions, probe, hd_class):
    """Use the training HD-class union, restricted to good IDs in every training session."""
    hd_union = np.array([], dtype=int)
    common_good = None
    provenance = []
    for session in sessions:
        kilosort = find_kilosort(session, probe)
        labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t")
        good = labels.loc[labels["KSLabel"].str.lower() == "good", "cluster_id"].to_numpy(dtype=int)
        common_good = good if common_good is None else np.intersect1d(common_good, good)
        source = session / "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc"
        tc = read_formatted_json(source)
        hd_ids = np.asarray(tc["unit_id"], dtype=int)[
            np.asarray(tc["unit_data"]["hd_class"], dtype=object) == hd_class
        ]
        hd_union = np.union1d(hd_union, hd_ids)
        provenance.append({
            "session_dir": str(session), "source": str(source),
            "source_sha256": file_sha256(source),
            "source_metadata": tc.get("metadata", {}),
            "saved_hd_class": hd_class, "matching_unit_ids": hd_ids.tolist(),
            "good_unit_ids": np.sort(good).tolist(), "kilosort_dir": str(kilosort.resolve()),
        })
    selected = np.intersect1d(hd_union, common_good)
    if not len(selected):
        raise ValueError("No good units match the union of training-session HD classes.")
    return selected, {
        "mode": "training_saved_hd_class_union_intersect_training_good_ids",
        "unit_ids": selected.tolist(), "training_sources": provenance,
        "classification_policy": "Use classes as saved; classifiers may differ between training sessions.",
        "target_classification_used": False,
    }


def load_session_clock(session, headplate, camera_channel, *, training, clock_override=None):
    data_dir = session / "data"
    allowed_trailing_frames = 1
    if clock_override is None:
        info = read_formatted_json(data_dir / "session_info.json")["session_info"]
        exposures, origin, timing_qc = get_exposure_timestamps(
            info, data_dir, camera_input_channel=camera_channel, camera_ttl_active_high=True,
        )
    else:
        if training:
            raise ValueError("Camera clock overrides are supported only for decoding targets.")
        with np.load(clock_override, allow_pickle=False) as stored:
            exposures = stored["exposure_times_s"]
            origin = float(stored["adc_time_origin_s"])
            timing_qc = json.loads(str(stored["metadata_json"].item()))
        allowed_trailing_frames = int(timing_qc["allowed_trailing_frames"])
        timing_qc.update(clock_override_path=str(clock_override), clock_override_sha256=file_sha256(clock_override))
    clock, _ = make_head_direction_tsd(exposures, np.arange(len(exposures)), np.zeros(len(exposures)))
    hd_path = data_dir / "processed/head_direction.json"
    result = {
        "exposures": exposures, "origin": origin, "clock": clock,
        "timing_qc": timing_qc, "head_direction_source": str(hd_path),
    }
    if training:
        frames, synced, feature, fs = load_head_direction(hd_path, headplate, exposures)
        result.update(frames=frames, synced=synced, feature=feature, fs=fs)
    else:
        # Only frame identity comes from target tracking. Target angles never enter the model.
        frames = np.asarray(read_formatted_json(hd_path)[headplate]["frames"])
        if (
            frames.ndim != 1 or frames.dtype.kind not in "iu" or not len(frames)
            or not np.array_equal(frames, np.arange(len(frames)))
            or (len(frames) - len(exposures) not in (0, 1) if clock_override is None
                else len(frames) - len(exposures) != allowed_trailing_frames)
        ):
            raise ValueError("Motive frames must match camera TTLs plus the explicitly permitted trailing frames.")
        result.update(frames=frames, synced=frames < len(exposures))
    return result


def session_training_counts(spikes, feature, feature_fs, bins):
    epochs = feature.time_support.intersect(spikes.time_support)
    times = feature.index.to_numpy()
    occupied = (np.searchsorted(times, epochs.end, side="right")
                > np.searchsorted(times, epochs.start, side="left"))
    epochs = epochs[occupied]
    if not len(epochs):
        raise ValueError("Training session has no valid tracked camera coverage.")
    counts = nap.compute_tuning_curves(
        data=spikes, features=feature, bins=bins, range=(0.0, 360.0),
        epochs=epochs, fs=feature_fs, return_counts=True,
        feature_names=["head_direction_deg"],
    ).transpose("unit", "head_direction_deg")
    return counts, epochs


def pool_training_counts(counts_per_session, sampling_rates):
    """Sum spikes and dwell time separately, including differing camera sample rates."""
    first = counts_per_session[0]
    spike_counts = np.zeros_like(first.values)
    occupancy_seconds = np.zeros(first.shape[1], dtype=float)
    for counts, fs in zip(counts_per_session, sampling_rates):
        if not counts.coords.equals(first.coords):
            raise ValueError("All training sessions must use the same unit IDs and angle bins.")
        spike_counts += counts.values
        occupancy_seconds += np.asarray(counts.attrs["occupancy"], dtype=float) / fs
    visited = occupancy_seconds > 0
    if visited.sum() < 2 or spike_counts.sum() == 0:
        raise ValueError("Training must contain selected-unit spikes and at least two visited angle bins.")
    rates = spike_counts[:, visited] / occupancy_seconds[visited]
    curves = xr.DataArray(
        rates, dims=("unit", "head_direction_deg"),
        coords={"unit": first.coords["unit"].values,
                "head_direction_deg": first.coords["head_direction_deg"].values[visited]},
        attrs={"occupancy": occupancy_seconds[visited], "occupancy_units": "seconds"},
    )
    arrays = {
        "unit_ids": first.coords["unit"].values,
        "angle_centers_deg": first.coords["head_direction_deg"].values,
        "angle_bin_edges_deg": np.asarray(first.attrs["bin_edges"][0]),
        "spike_counts": spike_counts, "occupancy_seconds": occupancy_seconds,
        "visited_angle_bins": visited, "tuning_curves_hz": rates,
    }
    return curves, arrays


def save_model(path, arrays, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    metadata["model_sha256"] = file_sha256(path)
    path.with_suffix(".metadata.json").write_text(json.dumps(metadata, allow_nan=False, indent=2) + "\n")


def load_model(path):
    with np.load(path, allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in stored.files}
    visited = arrays["visited_angle_bins"]
    curves = xr.DataArray(
        arrays["tuning_curves_hz"], dims=("unit", "head_direction_deg"),
        coords={"unit": arrays["unit_ids"],
                "head_direction_deg": arrays["angle_centers_deg"][visited]},
        attrs={"occupancy": arrays["occupancy_seconds"][visited], "occupancy_units": "seconds"},
    )
    return curves


def fit_model(args, sessions, units, selection):
    counts_per_session, sampling_rates, training = [], [], []
    for session in sessions:
        print(f"Training on {session}", flush=True)
        loaded = load_session_clock(session, args.headplate, args.camera_channel, training=True)
        spikes, _ = load_spikes(
            session, find_kilosort(session, args.probe), args.probe,
            loaded["origin"], loaded["clock"].time_support, units=units,
        )
        counts, epochs = session_training_counts(spikes, loaded["feature"], loaded["fs"], args.bins)
        counts_per_session.append(counts)
        sampling_rates.append(loaded["fs"])
        training.append({
            "session_dir": str(session), "head_direction_source": loaded["head_direction_source"],
            "adc_time_origin_raw_s": loaded["origin"], "feature_fs_hz": loaded["fs"],
            "training_intervals_s": epochs.values.tolist(),
            "training_duration_s": float(np.sum(epochs.end - epochs.start)),
            "feature_sample_count": len(loaded["feature"].restrict(epochs)),
            "occupancy_seconds": (np.asarray(counts.attrs["occupancy"]) / loaded["fs"]).tolist(),
            "training_spike_counts_per_unit": counts.values.sum(axis=1).astype(int).tolist(),
            "timing_qc": loaded["timing_qc"],
        })
    _, arrays = pool_training_counts(counts_per_session, sampling_rates)
    metadata = {
        "method": "pynapple.decode_bayes", "pynapple_version": nap.__version__,
        "training_sessions": training, "probe": args.probe, "headplate": args.headplate,
        "unit_selection": selection, "angle_bins": args.bins, "bin_size_s": args.bin_size,
        "trained_angle_centers_deg": arrays["angle_centers_deg"][arrays["visited_angle_bins"]].tolist(),
        "uniform_prior": True, "timebase": "open_ephys_adc_t0_relative_seconds_per_session",
        "angle_convention": "same as training head direction, degrees modulo 360",
        "pooling": "sum spike counts / sum per-session occupancy seconds; no averaging of rates",
        "training_scope": "full valid tracked camera coverage in training sessions only",
        "unit_identity_requirement": "same probe and jointly sorted cluster identities across all sessions",
    }
    save_model(args.model_output.resolve(), arrays, metadata)
    return metadata


def decode_session(args, session, model_metadata):
    model_path = args.model_output.resolve()
    # Reload the saved artifact for each target: neither target selects units or fits rates.
    curves = load_model(model_path)
    loaded = load_session_clock(
        session, args.headplate, args.camera_channel, training=False,
        clock_override=args.camera_clock.get(session),
    )
    units = curves.coords["unit"].values
    kilosort = find_kilosort(session, args.probe)
    spikes, _ = load_spikes(
        session, kilosort, args.probe, loaded["origin"], loaded["clock"].time_support, units=units,
    )
    frames, synced = loaded["frames"], loaded["synced"]
    angles = np.full(len(frames), np.nan)
    print(f"Decoding {session} with {len(units)} fixed units", flush=True)
    angles[synced], qc = decode_frames(
        spikes, curves, loaded["clock"].time_support,
        loaded["exposures"][frames[synced]], args.bin_size,
    )
    if not np.isfinite(angles).any():
        raise ValueError(f"{session}: no frame could be decoded.")
    spike_counts = np.array([len(spikes[int(unit)]) for unit in units], dtype=int)
    metadata = {
        "method": "pynapple.decode_bayes", "pynapple_version": nap.__version__,
        "session_dir": str(session), "probe": args.probe, "kilosort_dir": str(kilosort.resolve()),
        "model_path": str(model_path), "model_sha256": model_metadata["model_sha256"],
        "model_metadata_path": str(model_path.with_suffix(".metadata.json")),
        "training_session_dirs": [item["session_dir"] for item in model_metadata["training_sessions"]],
        "unit_ids": units.tolist(), "target_classification_used": False, "target_refit": False,
        "zero_spike_unit_ids": units[spike_counts == 0].tolist(),
        "spike_counts_per_unit": spike_counts.tolist(),
        "head_direction_source": loaded["head_direction_source"], "target_tracking_use": "frame IDs only",
        "headplate": args.headplate, "angle_convention": model_metadata["angle_convention"],
        "timebase": "open_ephys_adc_t0_relative_seconds", "adc_time_origin_raw_s": loaded["origin"],
        "decoding_intervals_s": loaded["clock"].time_support.values.tolist(),
        "bin_size_s": args.bin_size, "angle_bins": args.bins, "uniform_prior": True,
        "timing_qc": loaded["timing_qc"], "frame_count": len(frames),
        "valid_frame_count": int(np.isfinite(angles).sum()),
        "null_frame_count": int((~np.isfinite(angles)).sum()),
        "null_policy": "unsynced, outside decoded full bins, silent population, or invalid posterior",
        "evaluation": "Disjoint training sessions; no target angles or target HD classes used for fitting.",
        **qc,
    }
    output = session / "data/processed/internal_head_direction.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_payload(frames, angles), allow_nan=False, indent=2) + "\n")
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, allow_nan=False, indent=2) + "\n")
    print(f"Saved {output}: {metadata['valid_frame_count']}/{len(frames)} frames", flush=True)
    return output


def run(args):
    training = [path.resolve() for path in args.train_session]
    targets = [path.resolve() for path in args.decode_session]
    if len(set(training)) != len(training) or len(set(targets)) != len(targets):
        raise ValueError("Session lists must not contain duplicates.")
    if set(training) & set(targets):
        raise ValueError("Training and decoding sessions must be disjoint.")
    if not set(args.camera_clock).issubset(targets):
        raise ValueError("--camera-clock sessions must be listed in --decode-session.")
    outputs = [args.model_output.resolve(), args.model_output.resolve().with_suffix(".metadata.json")]
    for session in targets:
        output = session / "data/processed/internal_head_direction.json"
        outputs.extend([output, output.with_suffix(".metadata.json")])
    if not args.overwrite:
        for path in outputs:
            if path.exists():
                raise FileExistsError(f"{path} exists; use --overwrite to replace decoder outputs.")
    units, selection = select_training_units(training, args.probe, args.hd_class)
    print(f"Selected {len(units)} training units on Probe {args.probe}", flush=True)
    metadata = fit_model(args, training, units, selection)
    return [decode_session(args, session, metadata) for session in targets]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-session", type=Path, nargs="+", required=True)
    parser.add_argument("--decode-session", type=Path, nargs="+", required=True)
    parser.add_argument("--probe", default="A", help="One jointly sorted probe (default: A).")
    parser.add_argument("--headplate", default="hp4")
    parser.add_argument("--camera-channel", type=int, default=1, help="Motive ADC channel, active high (default: 1).")
    parser.add_argument("--camera-clock", action="append", default=[], metavar="SESSION=NPZ",
                        help="Explicit audited target clock NPZ; records its permitted trailing-frame count.")
    parser.add_argument("--hd-class", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--bin-size", type=float, default=0.1)
    parser.add_argument("--model-output", type=Path, required=True, help="Reusable .npz decoder with metadata sidecar.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    overrides = {}
    for value in args.camera_clock:
        if "=" not in value:
            parser.error("--camera-clock requires SESSION=NPZ.")
        session, clock_path = value.split("=", 1)
        key = Path(session).resolve()
        if key in overrides:
            parser.error("--camera-clock requires distinct session paths.")
        overrides[key] = Path(clock_path).resolve()
    args.camera_clock = overrides
    if args.model_output.suffix != ".npz":
        parser.error("--model-output must end in .npz.")
    if args.bins < 2 or not np.isfinite(args.bin_size) or args.bin_size <= 0:
        parser.error("--bins must be at least 2 and --bin-size must be finite and positive.")
    return args


if __name__ == "__main__":
    run(parse_args())
