import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--mouse-id", required=True)
parser.add_argument("--date", required=True)
parser.add_argument("--sessions", nargs="+", type=int, required=True)
parser.add_argument("--local-raw-base", type=Path, default=Path("/mnt/ssd4.1"))
parser.add_argument("--recording-root", type=Path,
                    default=Path("/mnt/senzailab/Kai/#Recording"))
parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "preprocessing/pipeline/config_result.yaml")
args = parser.parse_args()

mouseID = args.mouse_id
date = args.date
session_list: list[int] = args.sessions
session_tag = (
    "".join(str(session_id) for session_id in session_list)
    if all(1 <= session_id <= 9 for session_id in session_list)
    else "sessions-" + "-".join(str(session_id) for session_id in session_list)
)

run_name = f"{date}_{session_tag}"
local_output = args.local_raw_base / "Data"
remote_output = args.recording_root / mouseID / date / run_name / "pipeline"

session_paths = "\n".join(
    [f'  - {json.dumps(str(args.local_raw_base / f"{date}_{i}"))}' for i in session_list]
)

content = f'''run_name: "{run_name}"

session_paths:
{session_paths}

local_output: {json.dumps(str(local_output))}
# remote_output: {json.dumps(str(remote_output))}

per_shank: False
copy_mode: "newer"
target_fs: 1250.0
verbose: True
overwrite: False

job_kwargs:
  n_jobs: 4
  chunk_duration: "2s"
  progress_bar: True
  mp_context: "fork"
'''

config_result = args.output
config_result.parent.mkdir(parents=True, exist_ok=True)
config_result.write_text(content, encoding="utf-8")
print(config_result)
