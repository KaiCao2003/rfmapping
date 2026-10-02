"""Render separate screen/animal and saved EBC panels for one unit.

Run on hhw9l84. Supply either a recording session (exports synchronized JSON)
or --input with an existing export. Fitting and saved EBC maps are reused.
"""

import argparse
from pathlib import Path

from Utils.ebc_replay import export_session
from Utils.ebc_screen import render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, nargs="?")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--unit", type=int)
    parser.add_argument("--probe", default="A")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=float)
    args = parser.parse_args(argv)
    if (args.session is None) == (args.input is None):
        parser.error("Supply a session or --input.")
    if args.session is not None and args.unit is None:
        parser.error("--unit is required when exporting from a session.")

    input_path = args.input
    if args.session is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        input_path = args.output_dir / "session_overlay.json"
        export_session(args.session, args.unit, input_path, probe=args.probe,
                       fps=30. if args.fps is None else args.fps, preview_start=args.start)
    return render(input_path, args.output_dir, start_s=args.start,
                  duration_s=args.duration, fps=args.fps)


if __name__ == "__main__":
    main()
