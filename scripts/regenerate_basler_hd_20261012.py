"""Recompute audited Basler HD sessions, then publish verified outputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from Utils.json_tools import read_formatted_json
from Utils import kilosort_utils
from HD.tc_generation import generate_hd_tc
from Utils.camera_timing import get_exposure_timestamps, make_head_direction_tsd
from Utils.statistic_utils import rayleigh_test
from HD.tuning_curve_utils import tuning_curve


SESSIONS = {"m19_260831_2": 1, "m20_260918_11": 1, "m20_260921_11": 6,
            "m22_261008_11": 6}
RECORDINGS = Path("/mnt/senzailab/Kai/#Recording")
DEFAULT_OUTPUT = Path("output/session_comparisons_20261012/final_definition_fix/hd")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def verify_curves(result, hd_tsd, inputs):
    """Check occupancy, raw spike assignment, rates, tests and saved classes."""
    intervals = np.asarray(result["metadata"]["valid_pose_intervals_s"])
    times = hd_tsd.index.to_numpy()
    angles = hd_tsd.values
    selected = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        selected |= (times >= start) & (times <= end)
    edges = np.asarray(result["angle_bin_edges_deg"])
    occupancy = np.histogram(angles[selected], edges)[0]
    np.testing.assert_array_equal(result["occupancy_samples"], occupancy)
    occupancy_s = occupancy / result["metadata"]["feature_fs_hz"]
    np.testing.assert_array_equal(result["occupancy_time_s"], occupancy_s)
    ids = np.asarray(result["unit_id"])
    counts = np.asarray(result["spike_counts"])
    spike_times = np.load(inputs["spike_times"], mmap_mode="r").ravel()
    clusters = np.load(inputs["spike_clusters"], mmap_mode="r").ravel()
    valid = np.isin(clusters, ids)
    spike_times = np.asarray(spike_times[valid]) - result["metadata"]["adc_time_origin_raw_s"]
    clusters = np.asarray(clusters[valid])
    expected = np.zeros_like(counts)
    for start, end in intervals:
        use = (spike_times >= start) & (spike_times <= end)
        frame_use = (times >= start) & (times <= end)
        frame_times, frame_angles = times[frame_use], angles[frame_use]
        spikes = spike_times[use]
        right = np.searchsorted(frame_times, spikes).clip(0, len(frame_times) - 1)
        left = (right - 1).clip(0)
        closest = np.where(abs(spikes - frame_times[left]) < abs(spikes - frame_times[right]), left, right)
        bins = np.searchsorted(edges, frame_angles[closest], side="right") - 1
        rows = np.searchsorted(ids, clusters[use])
        np.add.at(expected, (rows, bins), 1)
    np.testing.assert_array_equal(counts, expected)
    rates = np.divide(counts, occupancy_s, out=np.full_like(counts, np.nan, dtype=float), where=occupancy_s > 0)
    np.testing.assert_allclose(np.asarray(result["firing_rate_hz"], dtype=float), rates, rtol=0, atol=0, equal_nan=True)
    class_counts = counts.reshape(len(ids), 30, 6).sum(axis=2)
    class_occupancy = occupancy_s.reshape(30, 6).sum(axis=1)
    classes = result["unit_data"]
    checked = 0
    for index, row in enumerate(class_counts):
        if classes["rayleigh_p"][index] is None:
            continue
        score, p = rayleigh_test(row, class_occupancy, np.arange(6, 360, 12))
        np.testing.assert_allclose([score, p], [classes["rayleigh_score"][index], classes["rayleigh_p"][index]], rtol=1e-12, atol=1e-12)
        shuffled = classes["shuffle_p"][index]
        assert shuffled is not None and .0 < shuffled <= 1
        np.testing.assert_allclose(shuffled * 1001, round(shuffled * 1001), atol=1e-9)
        significance = int(p <= .01) + int(shuffled <= .01)
        expected_class = 3 if significance == 2 and classes["kappa_pass"][index] else significance
        assert classes["hd_class"][index] == expected_class
        checked += 1
    return {"raw_counts_and_occupancy_exact": True, "rates_exact": True,
            "rayleigh_rechecked_units": checked, "shuffle_count": 1000,
            "shuffle_p_resolution_and_class_rules_checked": True}


def run(cohort, output_root, publish):
    mouse, date, number = cohort.split("_")
    session = RECORDINGS / mouse / date / f"{date}_{number}"
    output = output_root / cohort
    output.mkdir(parents=True, exist_ok=True)
    target = session / "data/tuning_curves/ProbeA"
    kilosort = next((session / "kilosort/ProbeA").glob("kilosort_*"))
    inputs = {"pose": session / f"{date}.csv", "spike_times": session / "data/probeA/adc_spike_time.npy",
              "spike_clusters": kilosort / "spike_clusters.npy", "unit_labels": kilosort / "cluster_KSLabel.tsv",
              "session_info": session / "data/session_info.json", "interval_table": session / "data/interval_table.csv"}
    hashes = {key: {"path": str(path), "sha256": sha(path)} for key, path in inputs.items()}
    info = read_formatted_json(inputs["session_info"])["session_info"]
    exposures, origin, timing = get_exposure_timestamps(
        session_info=info, data_dir=session / "data", camera_input_channel=SESSIONS[cohort],
        camera_ttl_active_high=False,
    )
    pose = pd.read_csv(inputs["pose"], usecols=["frame", "hd_deg"])
    frames = pose.frame.to_numpy(int)
    angles = pose.hd_deg.to_numpy(float) % 360
    trailing = int(frames[-1] == len(exposures))
    if trailing:
        frames, angles = frames[:-1], angles[:-1]
    hd, fs = make_head_direction_tsd(exposures, frames, angles)
    intervals = pd.read_csv(inputs["interval_table"])
    interval = intervals.loc[intervals.interval_type == "baseline", ["start", "end"]].iloc[[0]].to_numpy(float)
    np.savez_compressed(output / "hd_inputs.npz", frame_ids=frames, hd_cw_deg=angles, exposure_times_adc_s=exposures,
                        requested_intervals_s=interval, adc_time_origin_raw_s=origin)
    provenance = {"source_files": hashes, "heading_sha256": array_sha(angles), "frame_ids_sha256": array_sha(frames),
                  "exposure_times_sha256": array_sha(exposures), "timing_source": timing,
                  "camera_input_channel": SESSIONS[cohort], "trailing_frame_without_ttl_removed": trailing,
                  "generator_sha256": sha("Utils/tuning_curve_utils.py"), "script_sha256": sha(__file__),
                  "shuffle_module_sha256": sha("Utils/kilosort_utils.py"),
                  "shuffle_backend": "numba" if kilosort_utils._compute_shuffle_r_numba is not None else "numpy"}
    save_json(output / "input_provenance.json", provenance)
    print(f"START {cohort}: {len(frames)} pose rows, channel {SESSIONS[cohort]}, {fs:.6f} Hz; 180 bins, 30 classification bins, 1000 shuffles seed0", flush=True)
    result = tuning_curve(
        base_dir=session, kilosort_dir=kilosort, probe_name="A", interval_pairs=interval,
        HD_tsd=hd, adc_time_origin_s=origin, num_of_bins_in_hd=180, classification_bins=30,
        num_shuffle=1000, shuffle_seed=0, kappa_cutoff=.1, is_save=True,
        save_path=output / "tuning_curves.tc", feature_fs_hz=fs,
        timestamp_reference=timing["timestamp_reference"],
        metadata={"epoch": "baseline", "head_direction_is_clockwise": True,
                  "head_direction_zero": "north", "head_direction_source": str(inputs["pose"]),
                  "head_direction_convention": "north-zero clockwise; 90 degrees right",
                  "source_provenance": provenance, "classification_recomputed": True,
                  "analysis_run": "session_comparisons_20261012_final_definition_fix"},
    )
    validation = verify_curves(result, hd, inputs)
    for key, path in inputs.items():
        assert sha(path) == hashes[key]["sha256"], f"Input changed while computing: {path}"
    csv_name = "hd_class3_ProbeA.csv"
    generate_hd_tc(output / "tuning_curves.tc", output / csv_name, bins=30, hd_class=3,
                  unit_prefix=f"{mouse}:{date}:A", overwrite=True)
    labels = np.asarray(result["unit_data"]["hd_class"], dtype=object)
    ids = np.asarray(result["unit_id"])
    for value in (1, 2, 3):
        np.testing.assert_array_equal(np.load(output / f"hd_cells_{value}.npy"), np.sort(ids[labels == value]))
    prepared = pd.read_csv(output / csv_name, index_col=0)
    assert len(prepared) == sum(labels == 3) and prepared.shape[1] == 30
    records = []
    if publish:
        destinations = {name: target / name for name in ["tuning_curves.tc", "hd_cells_1.npy", "hd_cells_2.npy", "hd_cells_3.npy"]}
        destinations[csv_name] = session / "data/tc_comparison" / csv_name
        backup = session / "data/tuning_curves/superseded_pre_final_definition_fix_20261012/ProbeA"
        for name, dest in destinations.items():
            if dest.exists():
                old = backup / name
                old.parent.mkdir(parents=True, exist_ok=True)
                if not old.exists():
                    shutil.copy2(dest, old)
                assert sha(old) == sha(dest), f"Pre-existing backup is from a different publication: {old}"
                records.append({"path": str(dest), "previous_sha256": sha(dest), "backup": str(old)})
            else:
                records.append({"path": str(dest), "previous_sha256": None, "backup": None})
        for (name, dest), record in zip(destinations.items(), records, strict=True):
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name(dest.name + ".definition_fix_pending")
            shutil.copyfile(output / name, temporary)
            assert sha(temporary) == sha(output / name)
            temporary.replace(dest)
            record["sha256"] = sha(dest)
    receipt = {"cohort": cohort, "status": "published" if publish else "validated_staged",
               "completed_utc": datetime.now(timezone.utc).isoformat(), "unit_count": len(ids),
               "class_counts": {str(label): int(sum(labels == label)) for label in (0, 1, 2, 3, None)},
               "validation": validation, "provenance": provenance, "published": records,
               "tuning_curves_sha256": sha(output / "tuning_curves.tc"),
               "classification": result["metadata"]["classification"]}
    save_json(output / "receipt.json", receipt)
    print(f"DONE {cohort}: {receipt['class_counts']} {receipt['status']}", flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=SESSIONS, action="append")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    receipts = []
    for cohort in args.session or SESSIONS:
        receipt_path = args.output_root / cohort / "receipt.json"
        if args.publish and receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert receipt["status"] == "published"
            assert all(sha(item["path"]) == item["sha256"] for item in receipt["published"])
            print(f"Already published and hash-verified: {cohort}", flush=True)
        else:
            receipt = run(cohort, args.output_root, args.publish)
        receipts.append(receipt)
        name = f"manifest_{cohort}.json" if args.session else "manifest.json"
        save_json(args.output_root / name, receipts)


if __name__ == "__main__":
    main()
