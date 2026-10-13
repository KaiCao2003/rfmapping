import argparse
from pathlib import Path

from .canonical_unit_artifacts import generate_canonical_unit_artifacts
from .make_analyzer_for_sigui import make_analyzer_for_sigui
from .split_kilosort_by_recording import split_kilosort_by_recording

parser = argparse.ArgumentParser()
parser.add_argument("--mouse-id", required=True)
parser.add_argument("--date", required=True)
parser.add_argument("--sessions", nargs="+", type=int, required=True)
parser.add_argument("--local-raw-base", type=Path, default=Path("/mnt/ssd4.1"))
parser.add_argument("--recording-root", type=Path,
                    default=Path("/mnt/senzailab/Kai/#Recording"))
parser.add_argument("--analyzer-output-dir", type=Path,
                    default=Path(__file__).resolve().parent / "sorting_analyzer")
args = parser.parse_args()

sessions: list[int] = args.sessions
session_tag = (
    "".join(str(session_id) for session_id in sessions)
    if all(1 <= session_id <= 9 for session_id in sessions)
    else "sessions-" + "-".join(str(session_id) for session_id in sessions)
)
data_base_folder = args.local_raw_base
date = args.date
remote_output = args.recording_root / args.mouse_id / date
session_name = f"{date}_{session_tag}"
session_folder = data_base_folder / "Data" / session_name

probe_list = sorted(
    probe_json.parents[1].name.removeprefix("Probe")
    for probe_json in session_folder.glob("Probe*/concat/probe.json")
    if (probe_json.parents[1] / "concat/traces_cached_seg0.raw").is_file()
    and (probe_json.parents[1] / "kilosort/spike_times.npy").is_file()
)

print(f"Detected probes: {probe_list}")

make_analyzer_for_sigui(
    probe_list=probe_list,
    session_name=session_name,
    data_base_folder=data_base_folder,
    analyzer_output_dir=args.analyzer_output_dir,
)

split_kilosort_by_recording(
    probe_list=probe_list,
    sessions=sessions,
    data_base_folder=data_base_folder,
    remote_output=remote_output,
    date=date,
)

generate_canonical_unit_artifacts(
    data_base_folder=data_base_folder,
    remote_output=remote_output,
    date=date,
    sessions=sessions,
    probe_list=probe_list,
)
