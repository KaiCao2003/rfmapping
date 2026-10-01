"""Export rectangular-arena Basler overlays and all good-unit videos.

Run on hhw9l84 with ~/.virtualenvs/rfmapping/bin/python. Edit the settings below.
"""

import argparse
from pathlib import Path

from Utils import ebc_video as video


# Edit these settings, then run this file in the IDE. No notebook state is used.
session = Path("/mnt/senzailab/Kai/#Recording/m20/260921/260921_11")
probe = "A"
phase = "baseline"
video_path = session / f"{session.name.split('_')[0]}.avi"
save_path = session / "data/spatial_cells/videos"
video_start_s = 0.
video_duration_s = None  # Full Basler AVI.
audio_gain = .35  # Fixed OE volume fraction (35%); no per-channel peak normalization.
audio_source = "continuous"  # Real OE voltage; "clicks" plays only sorted spike times.
audio_band_hz = (300., 6000.)  # Spike band; OE Audio Monitor itself uses 100–7000 Hz.
audio_gate_sigma = 3.  # Expand below this multiple of background noise; 0 disables it.
audio_expander_ratio = 4.  # Stronger suppression than OE's 1.2; 1 disables expansion.
video_workers = 8
audio_workers = 4
video_encoder = "auto"  # Prefer NVENC; use 8 CPU encoding threads if unavailable.
BASLER_BOUNDS_PX = (370., 920., 210., 760.)  # left, right, top, bottom
BASLER_SIZE_CM = 41.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, nargs="?")
    parser.add_argument("--probe", default=probe)
    parser.add_argument("--phase", default=phase)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    selected_session = session if args.session is None else args.session
    selected_video = video_path if args.session is None else selected_session / f"{selected_session.name.split('_')[0]}.avi"
    selected_save_path = args.output_dir or (
        save_path if args.session is None else selected_session / "data/spatial_cells/videos"
    )
    data = video.load_video_data(
        selected_session, probe=args.probe, phase=args.phase,
        bounds_px=BASLER_BOUNDS_PX, arena_size_cm=BASLER_SIZE_CM,
    )
    return video.export_good_unit_videos(
        data, selected_video, selected_save_path,
        start_s=video_start_s, duration_s=video_duration_s, gain=audio_gain,
        workers=video_workers, audio_workers=audio_workers, video_encoder=video_encoder,
        audio_source=audio_source, audio_band_hz=audio_band_hz,
        audio_gate_sigma=audio_gate_sigma, audio_expander_ratio=audio_expander_ratio,
    )


if __name__ == "__main__":
    main()
