"""Build pitch/HD comparisons using the original paired notebook's HD epochs.

Run only on hhw9l84 after ../Analysis/scripts/generate_pitch_direction_json.py.
Writes separate caches; existing head_direction JSONs, HD curves, and classes
are read-only. Pitch is signed elevation, not circular yaw.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

from Utils.pitch_direction_comparison import PAIRED_SESSIONS, build_paired_tuning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=Path("/mnt/senzailab/Kai/#Recording"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/pitch_direction"))
    args = parser.parse_args()
    reports = []
    for mouse, date, session, probe in PAIRED_SESSIONS:
        started = perf_counter()
        report = build_paired_tuning(args.recording_root, args.output_dir, mouse, date, session, probe)
        report["elapsed_s"] = round(perf_counter() - started, 3)
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.output_dir / "tuning_report.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
