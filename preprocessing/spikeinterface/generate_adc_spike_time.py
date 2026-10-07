import argparse
from pathlib import Path

import numpy as np


def check_shared_clock(timestamps_file: Path) -> float:
    events_dir = timestamps_file.parents[2] / "events"
    probe_stream = timestamps_file.parent.name
    adc_stream = probe_stream.split(".")[0] + ".OneBox-ADC"
    probe_events = np.load(events_dir / probe_stream / "TTL" / "timestamps.npy")
    adc_events = np.load(events_dir / adc_stream / "TTL" / "timestamps.npy")
    probe_states = np.load(events_dir / probe_stream / "TTL" / "states.npy")
    adc_states = np.load(events_dir / adc_stream / "TTL" / "states.npy")

    if (probe_events.ndim != 1 or probe_events.size == 0
            or probe_events.shape != adc_events.shape
            or probe_states.shape != probe_events.shape
            or adc_states.shape != adc_events.shape):
        raise ValueError("Probe and ADC TTL timestamps/states must have matching nonempty vector shapes.")
    if not np.array_equal(probe_states, adc_states):
        raise ValueError("Probe and ADC TTL polarities differ; raw events do not correspond.")

    # Compare acquisition timestamps before changing either stream's origin.
    max_difference_sec = float(np.max(np.abs(probe_events - adc_events)))
    if not np.isfinite(max_difference_sec) or max_difference_sec > 0.001:
        raise ValueError(
            f"Raw Probe/ADC TTL timestamps differ by {max_difference_sec * 1000:.6f} ms; "
            "the shared-clock limit is 1 ms."
        )
    return max_difference_sec


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
    max_ttl_difference_sec = check_shared_clock(timestamps_file)
    print(f"Raw Probe{probe}/ADC TTL max difference: {max_ttl_difference_sec * 1000:.6f} ms")
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
