import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from EBB.spatial_cell_analysis import basler_session_defaults
from Utils.json_tools import read_formatted_json
from Utils.load_files import get_interval_pairs
from Utils.motive_pose import matched_motive_frame_count
from Utils.recording import gen_recording_interval_table
from Utils.camera_timing import get_exposure_timestamps, make_head_direction_tsd
from HD.tuning_curve_utils import tuning_curve


parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=Path)
parser.add_argument("date")
parser.add_argument("session_id")
parser.add_argument("probe_list")
parser.add_argument("--headplate-name", default="hp4")
parser.add_argument("--basler", action="store_true")
parser.add_argument("--camera-input-channel", type=int,
                    help="Override the camera TTL channel selected for this session.")
parser.add_argument("--ignore-hd-ttl", action="store_true",
                    help="Front-align mismatched Motive frames/TTLs to the shorter count.")
parser.add_argument("--comparison-code-dir", type=Path,
                    default=Path(__file__).resolve().parents[1])
args = parser.parse_args()

session_dir = args.base_dir / args.date / f"{args.date}_{args.session_id}"
data_dir = session_dir / "data"
session_key = f"{args.base_dir.name}_{args.date}_{args.session_id}"
camera_input_channel = args.camera_input_channel
if camera_input_channel is None:
    camera_input_channel = basler_session_defaults(session_key)[1] if args.basler else 1

interval_result = gen_recording_interval_table(
    base_dir=str(session_dir),
    multi_recording=True,
    camera_input_channel=camera_input_channel,
    # Basler inter-frame lows last ~15 ms; require ~100 ms to count as a pause.
    _recording_interval_min_len=3000 if args.basler else 300,
)
if interval_result is not None:
    interval_success, interval_message = interval_result
    print(interval_message)
    if not interval_success:
        raise RuntimeError(interval_message)

file_names = read_formatted_json(Path(__file__).resolve().parents[1] / "file_names.json")
session_info = read_formatted_json(
    data_dir / f"{file_names['session_info_filename']}.json"
)["session_info"]
interval_table = pd.read_csv(
    data_dir / f"{file_names['interval_table_filename']}.csv"
)
interval_pairs_all = np.asarray(
    get_interval_pairs(interval_table, phase_key="baseline"),
    dtype=float,
)
interval_pairs = interval_pairs_all[[0]]

hd_content = read_formatted_json(
    data_dir / "processed" / f"{file_names['head_direction_filename']}.json"
)
headplate_data = hd_content[args.headplate_name]
hd_raw = (
    headplate_data.get("head_direction_deg", headplate_data.get("hd"))
    if args.basler else headplate_data["head_direction_deg"]
)
hd = np.asarray(hd_raw, dtype=float) % 360
hd_frames = np.asarray(headplate_data["frames"], dtype=int)

exposure_timestamps, adc_time_origin_s, ttl_qc = get_exposure_timestamps(
    session_info=session_info,
    data_dir=data_dir,
    camera_input_channel=camera_input_channel,
    camera_ttl_active_high=not args.basler,
)

if args.basler:
    if not (
        np.all(hd_frames >= 0)
        and np.all(np.diff(hd_frames) > 0)
        and np.all(hd_frames < len(exposure_timestamps))
    ):
        raise ValueError(
            "Basler pose frame IDs must be increasing, zero-based, and within "
            "the saved camera timestamp range."
        )
    ttl_qc["basler_frame_count"] = int(len(hd_frames))
    ttl_qc["frame_timestamp_mapping"] = (
        "zero_based_pose_frame_id_to_saved_camera_timestamp"
    )
else:
    assert len(hd) == len(hd_frames)
    count = matched_motive_frame_count(
        hd_frames, exposure_timestamps, ttl_qc, ignore_hd_ttl=args.ignore_hd_ttl,
    )
    hd, hd_frames = hd[:count], hd_frames[:count]
    exposure_timestamps = exposure_timestamps[:count]

HD_tsd, feature_fs_hz = make_head_direction_tsd(exposure_timestamps, hd_frames, hd)

for probe_name in args.probe_list:
    kilosort_dir = (
        session_dir
        / "kilosort"
        / f"Probe{probe_name}"
        / f"kilosort_{args.session_id}"
    )
    save_path = (
        data_dir
        / "tuning_curves"
        / f"Probe{probe_name}"
        / "tuning_curves.tc"
    )
    tuning_curve(
        base_dir=session_dir,
        kilosort_dir=kilosort_dir,
        probe_name=probe_name,
        interval_pairs=interval_pairs,
        HD_tsd=HD_tsd,
        adc_time_origin_s=adc_time_origin_s,
        num_of_bins_in_hd=180,
        num_shuffle=1000,
        shuffle_seed=0,
        is_save=True,
        save_path=save_path,
        feature_fs_hz=feature_fs_hz,
        metadata={
            "epoch": "baseline",
            "headplate": args.headplate_name,
            "ttl_qc": ttl_qc,
        },
    )
    print(f"Saved tuning curves: {save_path}")
    # The comparison exporter uses the repository's own Utils in a separate process.
    subprocess.run(
        [sys.executable, "-m", "scripts.export_comparison_tcs", "hd", str(save_path),
         str(data_dir / "tc_comparison" / f"hd_class3_Probe{probe_name}.csv"),
         "--bins", "30", "--hd-class", "3",
         "--unit-prefix", f"{args.base_dir.name}:{args.date}:{probe_name}"],
        cwd=args.comparison_code_dir, check=True,
    )
