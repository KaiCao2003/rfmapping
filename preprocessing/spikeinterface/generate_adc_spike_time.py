import argparse
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("base_dir", type=Path)
parser.add_argument("date")
parser.add_argument("session_id")
parser.add_argument("probe_list")
parser.add_argument("--convert-to-zero", action="store_true")
args = parser.parse_args()

session_dir = args.base_dir / args.date / f"{args.date}_{args.session_id}"
is_convert_to_zero: bool = args.convert_to_zero

for probe in args.probe_list:
    probe = probe.upper()
    print("==============================")
    print(f"Generating Probe{probe} ADC spike time...")

    timestamps_file = next(
        (session_dir / args.date).glob(
            f"Record Node */experiment*/recording*/continuous/"
            f"OneBox-*.Probe{probe}/timestamps.npy"
        )
    )
    probe_timestamps = np.load(timestamps_file, mmap_mode="r")
    if is_convert_to_zero:
        probe_timestamps = probe_timestamps - probe_timestamps[0]

    spike_times_file = (
        session_dir
        / "kilosort"
        / f"Probe{probe}"
        / f"kilosort_{args.session_id}"
        / "spike_times.npy"
    )
    spike_times = np.load(spike_times_file, mmap_mode="r")
    adc_spike_time = probe_timestamps[spike_times]

    output_file = session_dir / "data" / f"probe{probe}" / "adc_spike_time.npy"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, adc_spike_time)
    print(f"Probe{probe} ADC spike time saved to {output_file}")
