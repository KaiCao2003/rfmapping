"""Recompute Probe A HD tuning curves for the selected clockwise-HD sessions.

Run on hhw9l84 with the rfmapping virtualenv. --apply overwrites only the
tuning_curves.tc and hd_cells_{1,2}.npy outputs for sessions that can be run.
Missing preprocessing caches are generated in /tmp, not in recording folders.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from Utils.Sessions import Session
from Utils.json_tools import read_formatted_json, write_formatted_json
from Utils.load_files import get_interval_pairs
from Utils.recording import gen_recording_interval_table
from Utils.tuning_curve_utils import get_exposure_timestamps, make_head_direction_tsd, tuning_curve


SESSIONS = {
    "m14": ["260609_1", "260615_1"],
    "m15": [
        "260624_1", "260625_1", "260625_4", "260630_1", "260630_3",
        "260630_4", "260630_5", "260630_extra", "260714_1", "260714_2",
        "260716_1", "260716_2",
    ],
    "m19": [
        "260820_1", "260820_2", "260820_3", "260820_4", "260821_1",
        "260821_2", "260821_3", "260821_4", "260821_6", "260824_1",
        "260824_2", "260824_3", "260824_4", "260827_2", "260827_4",
        "260827_5", "260827_6", "260827_7", "260827_9", "260827_10",
    ],
    "m20": [
        "260918_2", "260918_4", "260918_5", "260918_6", "260918_7",
        "260918_9", "260918_10", "260921_2", "260921_4", "260921_5",
        "260921_6", "260921_7", "260921_9", "260921_10", "260922_1",
    ],
}
EXCLUDED = {
    "m20/260921_10": "Motive and camera TTL frame counts differ by 34,929",
    "m20/260922_1": "only 93 HD frames (0.77 s); no classifiable HD units",
}
PROBE = "A"
HEADPLATE = "hp4"
PHASE = "baseline"
BIN_COUNT = 180
SHUFFLES = 1000
SHUFFLE_SEED = 0
CAMERA_INPUT_CHANNEL = 1


class NoBaselineInterval(Exception):
    pass


def targets(recording_root: Path):
    for mouse, names in SESSIONS.items():
        for name in names:
            yield f"{mouse}/{name}", recording_root / mouse / name.split("_")[0] / name


def probe_files(session_dir: Path) -> Path | None:
    kilosort_dirs = sorted((session_dir / "kilosort" / f"Probe{PROBE}").glob("kilosort_*"))
    if not kilosort_dirs:
        return None
    kilosort_dir = kilosort_dirs[0]
    spike_times = session_dir / "data" / f"probe{PROBE}" / "adc_spike_time.npy"
    if not all((
        spike_times.is_file(),
        (kilosort_dir / "spike_clusters.npy").is_file(),
        (kilosort_dir / "cluster_KSLabel.tsv").is_file(),
    )):
        return None
    return kilosort_dir


def load_session_info(session_dir: Path, data_dir: Path, file_names: dict) -> dict:
    path = data_dir / f"{file_names['session_info_filename']}.json"
    if path.is_file():
        return read_formatted_json(path)["session_info"]
    return Session(session_dir).get_session_info()


def first_baseline_interval(
    data_dir: Path, session_info: dict, file_names: dict,
) -> np.ndarray:
    interval_path = data_dir / f"{file_names['interval_table_filename']}.csv"
    if interval_path.is_file():
        interval_table = pd.read_csv(interval_path)
    else:
        with TemporaryDirectory(prefix="tc-hd-cache-") as scratch:
            scratch_data = Path(scratch) / "data"
            scratch_data.mkdir()
            write_formatted_json(
                ["session_info", "multi_recording"],
                [session_info, ["True"]],
                filename=file_names["session_info_filename"],
                path=scratch_data,
            )
            sync_path = data_dir / f"{file_names['sync_data_filename']}.json"
            if sync_path.is_file():
                shutil.copy2(sync_path, scratch_data / sync_path.name)
            success, message = gen_recording_interval_table(
                base_dir=scratch,
                multi_recording=True,
                camera_input_channel=CAMERA_INPUT_CHANNEL,
            )
            if not success:
                raise NoBaselineInterval(message)
            interval_table = pd.read_csv(
                scratch_data / f"{file_names['interval_table_filename']}.csv"
            )
    intervals = np.asarray(get_interval_pairs(interval_table, phase_key=PHASE), dtype=float)
    if not len(intervals):
        raise NoBaselineInterval("No baseline interval in interval_table.csv")
    return intervals[[0]]


def run_session(session_dir: Path, kilosort_dir: Path, file_names: dict) -> tuple[int, int]:
    data_dir = session_dir / "data"
    session_info = load_session_info(session_dir, data_dir, file_names)
    interval_pairs = first_baseline_interval(data_dir, session_info, file_names)

    hd_path = data_dir / "processed" / f"{file_names['head_direction_filename']}.json"
    hd_content = read_formatted_json(hd_path)[HEADPLATE]
    hd = np.asarray(hd_content["head_direction_deg"], dtype=float) % 360
    hd_frames = np.asarray(hd_content["frames"], dtype=int)
    if not np.array_equal(hd_frames, np.arange(len(hd_frames))):
        raise ValueError("Motive frame IDs must be zero-based and continuous")

    exposure_timestamps, adc_time_origin_s, ttl_qc = get_exposure_timestamps(
        session_info=session_info,
        data_dir=data_dir,
    )
    if len(hd_frames) == len(exposure_timestamps) + 1:
        print(f"Dropping trailing Motive frame {hd_frames[-1]}", flush=True)
        hd, hd_frames = hd[:-1], hd_frames[:-1]
    elif len(hd_frames) != len(exposure_timestamps):
        raise ValueError(
            f"Motive frame/TTL count mismatch: {len(hd_frames)} frames vs "
            f"{len(exposure_timestamps)} exposure pulses"
        )

    hd_tsd, feature_fs_hz = make_head_direction_tsd(exposure_timestamps, hd_frames, hd)
    ttl_qc["motive_frame_count"] = int(len(hd_frames))
    result = tuning_curve(
        base_dir=session_dir,
        kilosort_dir=kilosort_dir,
        probe_name=PROBE,
        interval_pairs=interval_pairs,
        HD_tsd=hd_tsd,
        adc_time_origin_s=adc_time_origin_s,
        num_of_bins_in_hd=BIN_COUNT,
        num_shuffle=SHUFFLES,
        shuffle_seed=SHUFFLE_SEED,
        is_save=True,
        feature_fs_hz=feature_fs_hz,
        timestamp_reference=ttl_qc["timestamp_reference"],
        metadata={
            "epoch": PHASE,
            "headplate": HEADPLATE,
            "head_direction_is_clockwise": True,
            "head_direction_source": str(hd_path),
            "ttl_qc": ttl_qc,
        },
    )
    return len(result["unit_id"]), len(hd_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording-root", type=Path,
        default=Path("/mnt/senzailab/Kai/#Recording"),
    )
    parser.add_argument("--session", action="append", help="Only run mouse/session, repeatable")
    parser.add_argument("--apply", action="store_true", help="Overwrite tuning curves and HD cell lists")
    parser.add_argument(
        "--log-file", type=Path,
        default=Path("output/regenerate_tuning_curves_from_hd.log"),
        help="Append per-session results and tracebacks here when applying",
    )
    args = parser.parse_args()

    logger = logging.getLogger("regenerate_tuning_curves_from_hd")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    if args.apply:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.addHandler(logging.FileHandler(args.log_file, mode="a"))
        logger.info("=== %s ===", datetime.now().astimezone().isoformat(timespec="seconds"))
        logger.info("Log file: %s", args.log_file)

    selected = list(targets(args.recording_root))
    if args.session:
        requested = set(args.session)
        unknown = requested - {label for label, _ in selected}
        if unknown:
            parser.error(f"Not in the selected HD batch: {', '.join(sorted(unknown))}")
        selected = [(label, path) for label, path in selected if label in requested]

    file_names = read_formatted_json(Path(__file__).resolve().parents[1] / "file_names.json")
    saved = skipped = failed = 0
    for label, session_dir in selected:
        if label in EXCLUDED:
            logger.info("SKIP %s: %s", label, EXCLUDED[label])
            skipped += 1
            continue
        kilosort_dir = probe_files(session_dir)
        if kilosort_dir is None:
            logger.info("SKIP %s: no complete Probe A spike files", label)
            skipped += 1
            continue
        if not args.apply:
            logger.info("CANDIDATE %s: %s", label, session_dir)
            continue
        logger.info("START %s", label)
        try:
            units, frames = run_session(session_dir, kilosort_dir, file_names)
        except NoBaselineInterval as error:
            logger.info("SKIP %s: %s", label, error)
            skipped += 1
            continue
        except Exception as error:
            logger.exception("FAILED %s: %s: %s", label, type(error).__name__, error)
            failed += 1
            continue
        logger.info("SAVED %s: %s units, %s HD frames", label, units, frames)
        saved += 1

    if args.apply:
        logger.info("Complete: %s saved, %s skipped, %s failed", saved, skipped, failed)
    else:
        logger.info("Dry run: %s candidates, %s skipped or excluded", len(selected) - skipped, skipped)
        logger.info("Use --apply to overwrite tuning_curves.tc and hd_cells_1/2.npy.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
