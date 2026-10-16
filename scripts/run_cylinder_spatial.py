"""Generate circular-arena spatial-cell maps from Motive tracking."""

import argparse
from pathlib import Path
import sys

# Resolve the checkout even when the production entry is a symlink to this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from EBB.ebc_analysis import compute_circle_ebc, load_motive_session


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--probe", choices=("A", "B"), required=True)
    parser.add_argument("--phase", default="baseline")
    parser.add_argument("--ignore-hd-ttl", action="store_true",
                        help="Front-align mismatched Motive frames/TTLs to the shorter count.")
    args = parser.parse_args(argv)

    results = compute_circle_ebc(
        load_motive_session(
            args.session,
            probe=args.probe,
            phase=args.phase,
            ignore_hd_ttl=args.ignore_hd_ttl,
        )
    )
    for boundary in results.values():
        for path in boundary["paths"].values():
            print(f"Saved cylinder spatial map: {path}")


if __name__ == "__main__":
    main()
