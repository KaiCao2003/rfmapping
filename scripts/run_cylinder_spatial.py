"""Generate circular-arena spatial-cell maps from Motive tracking."""

import argparse
from pathlib import Path

from Utils.ebc_analysis import compute_circle_ebc, load_motive_session


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("session", type=Path)
parser.add_argument("--probe", choices=("A", "B"), required=True)
parser.add_argument("--phase", default="baseline")
args = parser.parse_args()

results = compute_circle_ebc(
    load_motive_session(
        args.session,
        probe=args.probe,
        phase=args.phase,
    )
)
for boundary in results.values():
    for path in boundary["paths"].values():
        print(f"Saved cylinder spatial map: {path}")
