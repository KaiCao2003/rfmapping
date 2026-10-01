"""Render the full cylinder-camera recording for every good unit, with real audio.

Run on hhw9l84 with ~/.virtualenvs/rfmapping/bin/python. Edit the settings below.
The shared picture uses preserved Motive XYZ and a fixed level-reference JSON.
The reference faces VS0. With h = fused_yaw(R_XYZ @ R_reference.T):
    HD_world = (h + VS_zero) % 360
    VS_world = (VS_zero - Square_PositionX) % 360
    bearing = wrap180(atan2(target_y - head_y, target_x - head_x) - HD_world)
Positive bearing is CCW/left; geometry precedes camera projection. This writes
videos only, without recomputing saved RF/HD/EBC analyses. Video calibration and
frame synchronization belong to this recording and must match a new session.
"""

import argparse
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from tqdm.auto import tqdm

from Utils.ebc_video import _cache_continuous_audio, _export_channel_videos, _good_unit_channels, _open_ephys_source
from Utils.ebc_camera import render


# Edit these settings, then run the file. All decoded frames and good units are used.
session = Path("/mnt/senzailab/Kai/#Recording/m20/260922/260922_3")
probe = "A"
alignment_dir = Path(__file__).resolve().parent / "output/video_overlay_318"
registration_path = alignment_dir / "camera_registration.json"
clock_dir = alignment_dir / "full_video_clock"
vs_alignment_path = alignment_dir / "vs_alignment_check/alignment_check.json"
hd_reference_path = alignment_dir / "fused_yaw_6s/hp4_level_reference_20260924.json"
motive_csv_path = session / "data/processed/trimmed_input.csv"
kilosort_dir = session / "kilosort" / f"Probe{probe}" / "kilosort_3"
save_path = session / "data/spatial_cells/videos"

audio_gain = .35
audio_band_hz = (300., 6000.)
audio_gate_sigma = 3.
audio_expander_ratio = 4.
audio_workers = 4
video_encoder = "auto"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=save_path)
    output_dir = parser.parse_args(argv).output_dir
    clock_qc_path = clock_dir / "video_clock_qc.json"
    clock_qc = json.loads(clock_qc_path.read_text())
    if Path(clock_qc["session"]).resolve() != session.resolve():
        raise ValueError("Select the verified camera clock and registration for this session.")
    adc_origin = clock_qc["adc_time_origin_s"]
    video_times_path = clock_dir / "video_adc_times.npy"
    video_times = np.load(video_times_path)
    sync_valid = np.ones(len(video_times), dtype=bool)
    for first, last in clock_qc.get("uncertain_video_frame_ranges_inclusive", []):
        sync_valid[first:last + 1] = False
    if not np.array_equal(np.isfinite(video_times), sync_valid):
        raise ValueError("Missing camera timestamps must match the explicitly audited synchronization gaps.")

    date = session.name.split("_")[0]
    mat_path = session / f"{date}.mat"
    onset_path = session / "data/on_list_times.npy"
    trials = loadmat(mat_path, simplify_cells=True)["trials"]
    edges = np.load(onset_path) - adc_origin
    if len(edges) != len(trials) + 1 or not np.all(np.diff(edges) > 0):
        raise ValueError("MAT trials must match the increasing N+1 OE onset boundaries.")
    trial_data = [dict(on_s=float(edges[i]), off_s=float(edges[i + 1]),
                       screen_deg=float(trial["Square_PositionX"]),
                       luminance=float(trial["Square_Luminance"]))
                  for i, trial in enumerate(trials)]
    # VS registration was measured from visible bars, without fitting HD.
    # The user-confirmed level reference faces VS0, with no extra HD offset.
    vs_zero_deg = json.loads(vs_alignment_path.read_text())["beta_deg"]
    source = _open_ephys_source(session, probe)
    unit_channels = _good_unit_channels(kilosort_dir)
    units_by_channel = {}
    for unit, channel in unit_channels.items():
        units_by_channel.setdefault(channel, []).append(unit)

    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "stimulus_trials.json"
    replay_path.write_text(json.dumps(dict(meta=dict(probe=probe), trials=trial_data)) + "\n")
    print(f"Rendering the complete AVI for {len(unit_channels)} good units on Probe {probe}", flush=True)
    with tempfile.TemporaryDirectory(prefix="cylinder_audio_") as directory:
        temporary = Path(directory)
        overlay = render(session, replay_path, registration_path, video_times_path, temporary,
                         full_video=True, vs_zero_deg=vs_zero_deg, motive_csv_path=motive_csv_path,
                         hd_reference_path=hd_reference_path, hd_reference_world_deg=vs_zero_deg,
                         video_encoder=video_encoder, sync_valid=sync_valid)
        # Keep the shared picture local while every unit muxes it. Persist one
        # copy and the audit artifacts, with no scratch paths in saved manifests.
        scratch_video = Path(overlay["video"])
        saved_video = output_dir / scratch_video.name
        shutil.copyfile(scratch_video, saved_video)
        overlay["video"] = str(saved_video)
        overlay["snapshots"] = [str(shutil.move(path, output_dir / Path(path).name))
                                for path in overlay["snapshots"]]
        shutil.move(temporary / "frame_geometry.csv", output_dir / "frame_geometry.csv")
        (output_dir / "render_manifest.json").write_text(json.dumps(overlay, indent=2) + "\n")
        frames, fps, duration = overlay["frames"], overlay["fps"], overlay["duration_s"]
        # Each recorded image has its measured OE time. The final image lasts one
        # output frame. Audited unknown timestamps remain NaN: the audio helper
        # mutes intervals touching those nodes instead of guessing exposure times.
        audio_clock = (np.arange(frames + 1) / fps,
                       np.r_[video_times, video_times[-1] + 1 / fps] + adc_origin)
        channels = sorted(units_by_channel)
        tracks = _cache_continuous_audio(source, channels, audio_clock, duration, directory,
                                         band_hz=audio_band_hz)
        # Same electrode -> same real voltage. Encode its audio once, then mux
        # the shared picture for each unit, as in the rectangle exporter.
        with ProcessPoolExecutor(max_workers=audio_workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_export_channel_videos, scratch_video, output_dir, units,
                                   probe, tracks[channel], duration, audio_gain,
                                   audio_gate_sigma, audio_expander_ratio): len(units)
                       for channel, units in units_by_channel.items()}
            with tqdm(total=len(unit_channels), desc="Good-unit videos", unit="unit") as progress:
                for future in as_completed(futures):
                    future.result()
                    progress.update(futures[future])
        audio_info = dict(binary=str(source["binary"]), timestamps=str(source["timestamps"]),
                          unit_electrodes_zero_based=unit_channels, band_hz=audio_band_hz,
                          gain=audio_gain, gate_sigma=audio_gate_sigma,
                          expander_ratio=audio_expander_ratio,
                          noise_sigma_uv={ch: track["noise_sigma"] for ch, track in tracks.items()},
                          sample_rate_hz=48_000, first_oe_time_s=float(audio_clock[1][0]),
                          end_oe_time_s=float(audio_clock[1][-1]))

    manifest = dict(overlay, probe=probe, units=list(unit_channels),
                    videos=[str(output_dir / f"{unit}.mp4") for unit in unit_channels],
                    stimulus_mat=str(mat_path), stimulus_onsets=str(onset_path),
                    clock_qc=str(clock_qc_path), adc_time_origin_s=adc_origin,
                    vs_registration=str(vs_alignment_path), hd_reference_faces_vs_zero=True,
                    audio=audio_info, kilosort_dir=str(kilosort_dir))
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(unit_channels)} good-unit videos to {output_dir}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
