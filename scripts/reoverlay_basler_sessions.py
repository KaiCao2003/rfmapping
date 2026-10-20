"""Re-render Basler sessions from START_FROM using saved EBC data on hhw9l84."""

import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys


RECORDING_ROOT = Path("/mnt/senzailab/Kai/#Recording")
SESSIONS = [
    "m22/261009/261009_3",
    "m22/261009/261009_4",
    "m22/261008/261008_11",
    "m21/261006/261006_11",
    "m20/260921/260921_11",
    "m20/260918/260918_11",
    "m19/260831/260831_2",
]
START_FROM = "m20/260921/260921_11"
PROBE = "A"
PHASE = "baseline"
VIDEO_WORKERS = 48
AUDIO_WORKERS = 48
VIDEO_ENCODER = "auto"
WITH_UNIT_AUDIO = True  # False: one shared overlay per session, without unit movies.
SCRATCH_DIR = Path("/mnt/ssd4.1/ebc_video_tmp")
LOG_PATH = SCRATCH_DIR / "reoverlay_basler_sessions.log"
LOCK_PATH = SCRATCH_DIR / "reoverlay_basler_sessions.lock"


def run_sessions(sessions, dry_run=False):
    code_dir = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(TMPDIR=str(SCRATCH_DIR), OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    nvidia_lib = str(Path.home() / ".local/lib/nvidia-video-595.91.07")
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [nvidia_lib, env.get("LD_LIBRARY_PATH")]))
    # Sessions run sequentially; each renderer/audio batch uses its own workers.
    for index, name in enumerate(sessions, 1):
        session = RECORDING_ROOT / name
        output_name = "videos_boundary_hold" if WITH_UNIT_AUDIO else "videos_saved_analysis"
        command = [
            sys.executable, "-m", "scripts.ebc_video", str(session),
            "--probe", PROBE, "--phase", PHASE,
            "--analysis-dir", str(session / "data/spatial_cells" / f"Probe{PROBE}" / PHASE),
            "--output-dir", str(session / "data/spatial_cells" / output_name / f"Probe{PROBE}"),
            "--workers", str(VIDEO_WORKERS), "--audio-workers", str(AUDIO_WORKERS),
            "--video-encoder", VIDEO_ENCODER,
        ]
        if not WITH_UNIT_AUDIO:
            command.append("--silent")
        # No --duration: overwrite the full recording, including every good-unit movie.
        print(f"\n[{index}/{len(sessions)}] {name}\n{shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=code_dir, env=env, check=True)

    print("Commands listed." if dry_run else "All session overlays completed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without rendering.")
    parser.add_argument("--start-from", choices=SESSIONS, default=START_FROM,
                        help=f"Start at this session, including it (default: {START_FROM}).")
    args = parser.parse_args()
    sessions = SESSIONS[SESSIONS.index(args.start_from):]
    if args.dry_run:
        run_sessions(sessions, dry_run=True)
        return

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"An overlay batch is already running. Follow {LOG_PATH}") from None
        with LOG_PATH.open("a", buffering=1) as log:
            # Children inherit a real file, so closing SSH cannot break progress output.
            os.dup2(log.fileno(), sys.stdout.fileno())
            os.dup2(log.fileno(), sys.stderr.fileno())
            print(f"\nStarted {datetime.now().isoformat(timespec='seconds')}, pid={os.getpid()}, "
                  f"video_workers={VIDEO_WORKERS}, audio_workers={AUDIO_WORKERS}", flush=True)
            run_sessions(sessions)


if __name__ == "__main__":
    main()
