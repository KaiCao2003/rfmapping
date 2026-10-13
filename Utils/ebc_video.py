"""Render EBC overlays and export synchronized electrode-audio videos for good units."""

import json
import shutil
import subprocess
import tempfile
import time
import wave
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from fractions import Fraction
from multiprocessing import get_context, shared_memory
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from PIL import Image
from scipy.signal import butter, sosfilt, sosfilt_zi
from tqdm.auto import tqdm

from Utils.json_tools import read_formatted_json
from Utils.ebc_overlay import EBCOverlay as _Overlay


RAY_DEG = np.arange(0., 360., 45.)


_render_state = None


def _initialize_renderer(data, width, height, fps, memory_name, overlay_type=_Overlay):
    global _render_state
    memory = shared_memory.SharedMemory(name=memory_name)
    overlay = overlay_type(data, width, height, fps)
    input_bytes = width * height * 3
    _render_state = memory, overlay, input_bytes, overlay.size[0] * overlay.size[1] * 3


def _render_shared_frame(slot, frame_id, row):
    memory, overlay, input_bytes, output_bytes = _render_state
    offset = slot * (input_bytes + output_bytes)
    pixels = np.ndarray((overlay.height, overlay.width, 3), dtype=np.uint8,
                        buffer=memory.buf, offset=offset)
    frame = Image.fromarray(pixels)
    canvas = overlay.draw(frame, frame_id, row)
    start = offset + input_bytes
    memory.buf[start:start + output_bytes] = canvas.tobytes()


class _SharedRenderer:
    """Bounded frame slots: decoded once, drawn in parallel, encoded in order."""

    def __init__(self, data, width, height, fps, workers, overlay_type=_Overlay):
        self.slots = workers * 2
        self.input_bytes = width * height * 3
        self.size = overlay_type(data, width, height, fps).size
        self.output_bytes = self.size[0] * self.size[1] * 3
        self.stride = self.input_bytes + self.output_bytes
        self.memory = shared_memory.SharedMemory(create=True, size=self.slots * self.stride)
        # Only prepared drawing data enter workers; geometry is calculated once.
        pose = {key: value for key, value in data.items() if key not in ("ebc_info", "source", "provenance")}
        try:
            self.pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"),
                initializer=_initialize_renderer,
                initargs=(pose, width, height, fps, self.memory.name, overlay_type),
            )
        except BaseException:
            self.memory.close()
            self.memory.unlink()
            raise

    def read(self, stream, slot):
        start = slot * self.stride
        with self.memory.buf[start:start + self.input_bytes] as target:
            count = 0
            while count < len(target):
                received = stream.readinto(target[count:])
                if not received:
                    break
                count += received
        return count

    def write(self, stream, slot, preview=None):
        start = slot * self.stride + self.input_bytes
        with self.memory.buf[start:start + self.output_bytes] as pixels:
            stream.write(pixels)
            if preview is not None:
                Image.frombytes("RGB", self.size, bytes(pixels)).save(preview)

    def close(self):
        try:
            self.pool.shutdown(wait=True, cancel_futures=True)
        finally:
            self.memory.close()
            self.memory.unlink()


def _encoder_arguments(video_encoder):
    if video_encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-b:v", "0"]
    if video_encoder == "libx264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-threads", "8"]
    raise ValueError("video_encoder must be 'h264_nvenc' or 'libx264'.")


def _select_encoder(requested, width, height, fps):
    """Check runtime initialization, not just ffmpeg's compiled encoder list."""
    if requested not in ("auto", "h264_nvenc", "libx264"):
        raise ValueError("video_encoder must be 'auto', 'h264_nvenc', or 'libx264'.")
    if requested == "libx264":
        return requested
    # Use the actual output geometry. The one-frame probe writes no video file
    # and catches missing driver libraries before decoding the recording.
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-filter_threads", "1", "-f", "lavfi",
         "-i", f"color=c=white:s={width}x{height}:r={fps}", "-frames:v", "1", "-an",
         *_encoder_arguments("h264_nvenc"), "-pix_fmt", "yuv420p", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        return "h264_nvenc"
    reason = result.stderr.strip() or f"ffmpeg exited with code {result.returncode}"
    if requested == "h264_nvenc":
        raise RuntimeError(f"NVENC initialization failed:\n{reason}\n"
                           "Use video_encoder='auto' for a multithreaded CPU fallback.")
    print(f"NVENC unavailable; using libx264 with 8 CPU threads.\n{reason}", flush=True)
    return "libx264"


def _ffmpeg_error(process, log, operation):
    """Read stderr from a temporary file, which cannot fill a pipe and deadlock."""
    process.wait()
    log.seek(0)
    detail = log.read().decode("utf-8", errors="replace").strip()
    return RuntimeError(f"{operation} failed (ffmpeg exit {process.returncode}):\n"
                        f"{detail or 'ffmpeg produced no error details.'}")


def export_ebc_overlay(data, video_path, output_path, *, start_s=0., duration_s=None,
                       workers=1, video_encoder="libx264", overlay_type=_Overlay,
                       frame_rate=None, first_frame=None, stop_frame=None):
    """Encode prepared overlay data on the recorded video.

    Clip times start at AVI time zero.
    Explicit frame bounds and frame rate support an independently measured
    camera clock when the AVI header does not describe the recording clock.
    Missing tracking stays in the movie as a labeled gap. Multiple workers
    share bounded frame buffers; the common picture is encoded only once.
    """
    if video_path is None:
        raise ValueError("A source video is required.")
    video_path = Path(video_path)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4" or video_path.resolve() == output_path.resolve():
        raise ValueError("Choose a separate .mp4 output path.")
    if start_s < 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("start_s must be nonnegative and duration_s must be positive or None.")
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video_path)],
        check=True, capture_output=True, text=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    fps_text = stream["avg_frame_rate"] if frame_rate is None else str(frame_rate)
    fps = float(Fraction(fps_text))
    width, height, total = int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])
    frame_ids = np.asarray(data["frame_ids"], dtype=int)
    if len(frame_ids) != len(data["times"]) or np.any(np.diff(frame_ids) <= 0) or np.any(frame_ids < 0):
        raise ValueError("Pose frame IDs must be increasing zero-based video indices.")
    expected_count = data.get("video_frame_count")
    if expected_count is not None and total not in (expected_count, expected_count + 1):
        raise ValueError("AVI frame count disagrees with the audited full-video clock.")
    first = int(round(start_s * fps)) if first_frame is None else int(first_frame)
    stop = total if duration_s is None else min(total, first + int(round(duration_s * fps)))
    if stop_frame is not None:
        stop = int(stop_frame)
    if not 0 <= first < stop <= total:
        raise ValueError("The requested clip contains no video frames.")
    overlay = overlay_type(data, width, height, fps)
    video_encoder = _select_encoder(video_encoder, *overlay.size, fps_text)
    codec_args = _encoder_arguments(video_encoder)
    workers = min(workers, stop - first)
    rows = np.full(total, -1, dtype=int)
    within_video = frame_ids < total
    rows[frame_ids[within_video]] = np.flatnonzero(within_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.stem}.partial.mp4")
    preview = output_path.with_suffix(".png")
    preview_frame = min(stop - 1, first + int(20 * fps))
    # AVI seeking can skip the initial GOP even for -ss 0. Decode in order
    # and trim by frame ordinal so that CSV frame IDs remain exact.
    decoder_command = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "4", "-filter_threads", "2",
        "-i", str(video_path), "-map", "0:v:0",
        "-vf", f"trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS",
        "-fps_mode", "passthrough", "-frames:v", str(stop - first),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "2", "pipe:1",
    ]
    decoder, encoder, renderer = None, None, None
    decoder_log, encoder_log = None, None
    pending = deque()
    started = time.monotonic()

    def report(done):
        if done % 1000 == 0 or done == stop - first:
            print(f"Overlay: {done:,}/{stop - first:,} frames ({done / (time.monotonic() - started):.1f} frames/s)", flush=True)

    def write_next():
        future, slot, frame_id = pending.popleft()
        future.result()
        renderer.write(encoder.stdin, slot, preview if frame_id in (first, preview_frame) else None)
        report(frame_id - first + 1)

    try:
        encoder_log = tempfile.TemporaryFile()
        decoder_log = tempfile.TemporaryFile()
        decoder = subprocess.Popen(decoder_command, stdout=subprocess.PIPE, stderr=decoder_log)
        if workers > 1:
            renderer = _SharedRenderer(data, width, height, fps, workers, overlay_type)
        print(f"Overlay: {workers} drawing workers, encoder={video_encoder}", flush=True)
        encoder = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-filter_threads", "2", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{overlay.size[0]}x{overlay.size[1]}", "-r", fps_text, "-i", "pipe:0", "-an",
             *codec_args,
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(partial)], stdin=subprocess.PIPE, stderr=encoder_log,
        )
        for frame_id in range(first, stop):
            if renderer is not None:
                slot = (frame_id - first) % renderer.slots
                if len(pending) == renderer.slots:
                    write_next()
            if renderer is not None:
                count = renderer.read(decoder.stdout, slot)
            else:
                payload = decoder.stdout.read(width * height * 3)
                count = len(payload)
            # A clean EOF handles the Basler AVI's one-frame header overcount.
            if (not count and decoder.wait() == 0 and frame_id > first
                    and expected_count is None and stop == total and stop - frame_id == 1):
                stop = frame_id
                break
            if count != width * height * 3:
                if decoder.wait() != 0:
                    raise _ffmpeg_error(decoder, decoder_log, "AVI decoding")
                raise RuntimeError(f"AVI decode stopped at frame {frame_id}; expected {stop} frames.")
            if renderer is not None:
                future = renderer.pool.submit(_render_shared_frame, slot, frame_id, int(rows[frame_id]))
                pending.append((future, slot, frame_id))
            else:
                frame = Image.frombytes("RGB", (width, height), payload)
                canvas = overlay.draw(frame, frame_id, int(rows[frame_id]))
                encoder.stdin.write(canvas.tobytes())
                if frame_id in (first, preview_frame):
                    canvas.save(preview)
                report(frame_id - first + 1)
        while pending:
            write_next()
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise _ffmpeg_error(encoder, encoder_log, f"{video_encoder} encoding")
        if decoder is not None and decoder.wait() != 0:
            raise _ffmpeg_error(decoder, decoder_log, "AVI decoding")
        partial.replace(output_path)
    except BrokenPipeError as error:
        raise _ffmpeg_error(encoder, encoder_log, f"{video_encoder} encoding") from error
    finally:
        # ffmpeg may wait for raw input despite SIGTERM; close its input first.
        if encoder is not None and not encoder.stdin.closed:
            try:
                encoder.stdin.close()
            except BrokenPipeError:
                pass
        for process in (decoder, encoder):
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait()
        if decoder is not None:
            decoder.stdout.close()
        try:
            if renderer is not None:
                renderer.close()
        finally:
            for log in (decoder_log, encoder_log):
                if log is not None:
                    log.close()
            partial.unlink(missing_ok=True)
    selected = rows[first:stop]
    available = selected >= 0
    geometry_info = data.get("geometry_metadata", {})
    summary = dict(video=str(output_path), source_video=str(video_path),
                   preview=str(preview), arena_type=data.get("arena_type"),
                   session=str(data["session"]), phase=data["phase"],
                   provenance=data.get("provenance", {}),
                   **geometry_info,
                   ray_bearings_deg=RAY_DEG.tolist(),
                   frames=stop - first, fps=fps, container_frame_count=total,
                   drawing_workers=workers, video_encoder=video_encoder,
                   duration_s=(stop - first) / fps, source_first_frame=first, source_last_frame=stop - 1,
                   missing_pose_frames=int((~available).sum()),
                   outside_arena_frames=int((~overlay.valid[selected[available]]).sum()),
                   render_seconds=round(time.monotonic() - started, 2))
    if "overlay_info" in data:
        info = data["overlay_info"]
        summary.update(ray_bearings_deg=info.bearings_deg.tolist(),
                       missing_pose_frames=int((~available).sum() + (~info.pose_valid[selected[available]]).sum()),
                       outside_arena_frames=int(info.outside_boundary[selected[available]].sum()),
                       unsynchronized_frames=int((~available).sum() + (~info.synchronized[selected[available]]).sum()),
                       pose_rows_without_video=int((~within_video).sum()))
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def export_good_unit_videos(data, video_path, save_path, *, start_s=0., duration_s=None, gain=.35,
                            workers=8, audio_workers=4, video_encoder="auto",
                            audio_source="continuous", audio_band_hz=(300., 6000.), audio_gate_sigma=3.,
                            audio_expander_ratio=4., frame_rate=None, first_frame=None, stop_frame=None):
    """Save save_path/<unit_id>.mp4 for every good unit.

    Continuous audio is its main electrode's bandpassed voltage, including other
    units and background on that electrode. It is not an isolated sorted unit.
    Gain is the OE volume fraction, independent of clip/channel peak amplitude.
    The soft expander suppresses background and also attenuates weak spikes.
    """
    save_path = Path(save_path).expanduser()
    if workers < 1 or audio_workers < 1:
        raise ValueError("workers and audio_workers must be at least 1.")
    if not 0 < gain <= 1:
        raise ValueError("gain must be between 0 (exclusive) and 1.")
    if audio_source not in ("continuous", "clicks"):
        raise ValueError("audio_source must be 'continuous' or 'clicks'.")
    if not np.isfinite(audio_gate_sigma) or audio_gate_sigma < 0:
        raise ValueError("audio_gate_sigma must be nonnegative (0 disables the gate).")
    if not np.isfinite(audio_expander_ratio) or audio_expander_ratio < 1:
        raise ValueError("audio_expander_ratio must be at least 1 (1 disables expansion).")
    if audio_source == "continuous":
        # Resolve channels and raw input before doing the expensive video render.
        source = _open_ephys_source(data["session"], data["probe"])
        unit_channels = _good_unit_channels(data["source"]["kilosort_dir"])
        good_units = list(unit_channels)
    save_path.mkdir(parents=True, exist_ok=True)
    # Keep the shared picture on local scratch storage; do not read the same
    # video from a network recording directory once per unit.
    with tempfile.TemporaryDirectory(prefix="ebc_batch_") as directory:
        temporary = Path(directory)
        overlay = export_ebc_overlay(
            data, video_path, temporary / "overlay.mp4", start_s=start_s, duration_s=duration_s,
            workers=workers, video_encoder=video_encoder, frame_rate=frame_rate,
            first_frame=first_frame, stop_frame=stop_frame,
        )
        saved_video = save_path / "overlay.mp4"
        saved_preview = save_path / "overlay.png"
        shutil.copyfile(overlay["video"], saved_video)
        shutil.copyfile(overlay["preview"], saved_preview)
        saved_overlay = dict(overlay, video=str(saved_video), preview=str(saved_preview))
        (save_path / "overlay.json").write_text(json.dumps(saved_overlay, indent=2) + "\n")
        duration = overlay["frames"] / overlay["fps"]
        if audio_source == "continuous":
            # Invert the same exposure clock used for sorted spikes. Do not zero
            # the probe and ADC clocks independently: OE already synchronized them.
            clock = _video_audio_clock(data, overlay)
            channels = sorted(set(unit_channels.values()))
            print(f"Reading {len(channels)} electrode channels once for {len(good_units)} good units")
            tracks = _cache_continuous_audio(source, channels, clock, duration, temporary,
                                            band_hz=audio_band_hz)
            units_by_channel = {channel: [] for channel in channels}
            for unit, channel in unit_channels.items():
                units_by_channel[channel].append(unit)
        else:
            good_units, spike_seconds, _ = _load_audio_spikes(data, overlay)
        pool = ProcessPoolExecutor(max_workers=audio_workers, mp_context=get_context("spawn"))
        try:
            if audio_source == "continuous":
                futures = {pool.submit(
                    _export_channel_videos, overlay["video"], save_path,
                    units, data["probe"], tracks[channel], duration, gain, audio_gate_sigma,
                    audio_expander_ratio,
                ): len(units) for channel, units in units_by_channel.items()}
            else:
                futures = {pool.submit(
                    _mux_spike_audio, overlay["video"], save_path / f"{unit}.mp4",
                    int(unit), data["probe"], spike_seconds[int(unit)], duration, gain,
                ): 1 for unit in good_units}
            with tqdm(total=len(good_units), desc="Good-unit videos", unit="unit") as progress:
                for future in as_completed(futures):
                    future.result()
                    progress.update(futures[future])
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    videos = [save_path / f"{unit}.mp4" for unit in good_units]
    print(f"Saved {len(videos)} videos to {save_path.resolve()}")
    manifest = dict(saved_overlay, probe=data["probe"], units=[int(unit) for unit in good_units],
                    videos=[str(path) for path in videos], provenance=data.get("provenance", {}),
                    audio=dict(source=audio_source, gain=gain, band_hz=list(audio_band_hz),
                               gate_sigma=audio_gate_sigma, expander_ratio=audio_expander_ratio))
    (save_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _video_audio_clock(data, video_result):
    """Map AVI clip seconds to common OE seconds using exposure timestamps."""
    fps, first = video_result["fps"], video_result["source_first_frame"]
    exposures = np.asarray(data["exposure_times"])
    frames = first + np.arange(video_result["frames"] + 1)
    times = np.full(len(frames), np.nan)
    known = frames < len(exposures)
    times[known] = exposures[frames[known]]
    # The final recorded image lasts one output frame. Never extrapolate past
    # a missing final exposure or through an interior synchronization gap.
    if frames[-1] == len(exposures) and np.isfinite(exposures[-1]):
        times[-1] = exposures[-1] + 1. / fps
    return np.arange(len(frames)) / fps, times + data["adc_time_origin_s"]


def _open_ephys_source(session, probe):
    """Read the raw session stream, not Kilosort's concatenated/preprocessed input."""
    info = read_formatted_json(Path(session) / "data/session_info.json")["session_info"]
    structure_path = Path(info["session_name"])
    structure = json.loads(structure_path.read_text())
    folder = info[f"continuous_probe_{probe}_folder"]
    stream = next(item for item in structure["continuous"] if item["folder_name"].rstrip("/") == folder)
    directory = structure_path.parent / "continuous" / folder
    timestamps = directory / "timestamps.npy"
    times = np.load(timestamps, mmap_mode="r")
    binary = directory / "continuous.dat"
    channels = int(stream["num_channels"])
    if times.dtype.kind != "f":
        raise ValueError("Audio requires synchronized OE timestamps.npy in seconds (OE 0.6+).")
    if binary.stat().st_size != len(times) * channels * 2:
        raise ValueError(f"{binary}: int16 sample count does not match timestamps and channel count.")
    return dict(binary=binary, timestamps=timestamps, sample_rate=float(stream["sample_rate"]),
                num_channels=channels, bit_volts=np.array([ch["bit_volts"] for ch in stream["channels"]]))


def _good_unit_channels(kilosort):
    """Map each good cluster to the peak channel of its dominant, unwhitened template."""
    kilosort = Path(kilosort)
    labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t")
    units = np.sort(labels.loc[labels.KSLabel == "good", "cluster_id"].to_numpy(int))
    if not len(units):
        raise ValueError(f"No good units in {kilosort}.")
    templates = np.load(kilosort / "templates.npy", mmap_mode="r")
    channel_map = np.load(kilosort / "channel_map.npy").ravel()
    whitening_inv = np.load(kilosort / "whitening_mat_inv.npy")
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").ravel()
    template_ids = np.load(kilosort / "spike_templates.npy", mmap_mode="r").ravel()
    counts = np.zeros((len(units), len(templates)), dtype=np.int64)
    original_ids = True
    # One bounded pass over all spikes; cluster IDs can differ from template IDs
    # after curation. Avoid scanning millions of spikes separately for every unit.
    for start in range(0, len(clusters), 1_000_000):
        cluster = clusters[start:start + 1_000_000]
        template = template_ids[start:start + 1_000_000]
        original_ids &= np.array_equal(cluster, template)
        rows = np.searchsorted(units, cluster)
        selected = units[np.minimum(rows, len(units) - 1)] == cluster
        codes = rows[selected] * len(templates) + template[selected]
        counts += np.bincount(codes, minlength=counts.size).reshape(counts.shape)
    dominant = counts.argmax(axis=1)
    absent = ~counts.any(axis=1)
    if absent.any():
        # Session slices retain all original KS labels/templates, even if a unit
        # never fired in this session. Original cluster IDs are template indices.
        if not original_ids or np.any((units[absent] < 0) | (units[absent] >= len(templates))):
            raise ValueError(f"No template assignment for silent curated units {units[absent].tolist()}.")
        dominant[absent] = units[absent]
    template_indices = np.unique(dominant)
    selected = np.asarray(templates[template_indices])
    # One matrix product avoids starting a threaded BLAS operation per unit.
    waveforms = (selected.reshape(-1, selected.shape[-1]) @ whitening_inv).reshape(selected.shape)
    peak_channels = channel_map[np.ptp(waveforms, axis=1).argmax(axis=1)]
    peaks = dict(zip(template_indices, peak_channels))
    return {int(unit): int(peaks[template]) for unit, template in zip(units, dominant)}


def _clock_times(seconds, clock):
    """Piecewise linear video→OE clock, extrapolating the final exposure period."""
    video_times, oe_times = clock
    left = np.clip(np.searchsorted(video_times, seconds, side="right") - 1, 0, len(video_times) - 2)
    fraction = (seconds - video_times[left]) / (video_times[left + 1] - video_times[left])
    return oe_times[left] + fraction * (oe_times[left + 1] - oe_times[left])


def _continuous_audio_blocks(source, channels, clock, duration_s, *, band_hz=(300., 6000.),
                              sample_rate=48_000, block_s=1.):
    """One sequential raw read, causal bandpass, and clock-aligned audio resampling.

    Filter state survives block boundaries. The sample rate changes, not playback
    speed: each audio sample looks up the corresponding synchronized OE time.
    """
    channels = np.asarray(channels, dtype=int)
    times = np.load(source["timestamps"], mmap_mode="r")
    raw = np.memmap(source["binary"], dtype="<i2", mode="r",
                    shape=(len(times), source["num_channels"]))
    sos = butter(2, band_hz, btype="bandpass", fs=source["sample_rate"], output="sos")
    volts = np.asarray(source["bit_volts"])[channels]
    # Warm up the high-pass filter using the actual preceding voltage.
    first_time = _clock_times(np.array([0.]), clock)[0]
    cursor = int(np.searchsorted(times, first_time - .25)) if np.isfinite(first_time) else None
    state = None
    previous_times = np.empty(0)
    previous_voltage = np.empty((0, len(channels)))
    sample_count = int(round(duration_s * sample_rate))
    block_samples = max(1, int(round(block_s * sample_rate)))
    for start in range(0, sample_count, block_samples):
        seconds = np.arange(start, min(start + block_samples, sample_count)) / sample_rate
        targets = _clock_times(seconds, clock)
        finite = np.isfinite(targets)
        output = np.zeros((len(targets), len(channels)), dtype=np.float32)
        # Unknown synchronization stays silent; it must not advance the source
        # cursor or reset the filter before the next measured interval.
        if not finite.any():
            yield output
            continue
        measured_targets = targets[finite]
        if cursor is None:
            cursor = int(np.searchsorted(times, measured_targets[0] - .25))
        stop = min(len(times), int(np.searchsorted(times, measured_targets[-1], side="right")) + 1)
        if stop > cursor:
            voltage = raw[cursor:stop, channels].astype(np.float32) * volts
            if state is None:
                state = sosfilt_zi(sos)[:, :, None] * voltage[0]
            voltage, state = sosfilt(sos, voltage, axis=0, zi=state)
            block_times = np.r_[previous_times, times[cursor:stop]]
            voltage = np.vstack((previous_voltage, voltage))
            cursor = stop
        else:
            block_times, voltage = previous_times, previous_voltage
        if len(block_times) > 1:
            left = np.clip(np.searchsorted(block_times, measured_targets, side="right") - 1,
                           0, len(block_times) - 2)
            fraction = (measured_targets - block_times[left]) / (block_times[left + 1] - block_times[left])
            output[finite] = voltage[left] + fraction[:, None] * (voltage[left + 1] - voltage[left])
            # Keep unavailable acquisition time silent instead of stretching
            # the first/last measured value into the missing part of a clip.
            output[(targets < block_times[0]) | (targets > block_times[-1])] = 0
        # Keep both endpoints of the final interpolation interval: after
        # upsampling the next target can still fall before its right endpoint.
        previous_times, previous_voltage = block_times[-2:].copy(), voltage[-2:].copy()
        yield output


def _cache_continuous_audio(source, channels, clock, duration_s, directory, *,
                            band_hz=(300., 6000.), sample_rate=48_000):
    """Cache each distinct electrode once on scratch; share it across its units."""
    tracks = {int(ch): dict(path=Path(directory) / f"channel_{ch}.f32", channel=int(ch),
                           sample_rate=sample_rate, band_hz=band_hz) for ch in channels}
    noise_levels = []
    with ExitStack() as stack:
        files = [stack.enter_context(tracks[ch]["path"].open("wb")) for ch in channels]
        blocks = _continuous_audio_blocks(source, channels, clock, duration_s,
                                          band_hz=band_hz, sample_rate=sample_rate)
        for block in tqdm(blocks, total=int(np.ceil(duration_s)), desc="OE voltage audio", unit="s"):
            # Estimate background from sparse samples, robust to large spikes.
            sample = block[::16]
            noise_levels.append(np.median(abs(sample - np.median(sample, axis=0)), axis=0) / .67448975)
            for column, (ch, stream) in enumerate(zip(channels, files)):
                stream.write(block[:, column].astype("<f4").tobytes())
    noise_levels = np.asarray(noise_levels)
    for column, ch in enumerate(channels):
        # A short movement artifact must not set the gate for the entire clip.
        positive = noise_levels[:, column][noise_levels[:, column] > 0]
        tracks[ch]["noise_sigma"] = float(np.median(positive)) if len(positive) else 0.
    return tracks


@njit(cache=True, nogil=True)
def _expand_audio(voltage, threshold, ratio, envelope, level):
    """In-place downward expansion, retaining detector/gain state across blocks."""
    # OE 1.1 AudioNode.cpp uses these per-sample coefficients, not millisecond
    # time constants. Our threshold is in uV; OE's slider acts after volume gain.
    decay, smoothing = np.exp(-4.), np.exp(-1.)
    for i in range(len(voltage)):
        amplitude = abs(voltage[i]) + 1.e-29
        envelope = amplitude + decay * max(envelope - amplitude, 0.)
        target = min(envelope / threshold, 1.) ** (ratio - 1.)
        level = target + smoothing * (level - target)
        voltage[i] *= level
    return envelope, level


def _continuous_pcm(track, gain, gate_sigma=3., expander_ratio=4., *, block_s=1.):
    """Fixed voltage gain and soft expansion, without threshold-triggered windows.

    Gain/limiting follow OE AudioNode; the noise-relative threshold and stronger
    default ratio are ours. OE uses a ratio of 1.2 and an absolute slider value.
    Reference: https://github.com/open-ephys/plugin-GUI/blob/v1.1.0/Source/Processors/AudioNode/AudioNode.cpp
    """
    # OE normalized output = uV * volume_fraction / (32767 * .02).
    # Express the same conversion directly in int16 PCM units.
    scale = gain / .02
    threshold = gate_sigma * track["noise_sigma"]
    envelope, level = 0., 1.
    source = np.memmap(track["path"], dtype="<f4", mode="r")
    block_samples = max(1, round(block_s * track["sample_rate"]))
    for start in range(0, len(source), block_samples):
        block = np.clip(source[start:start + block_samples], -1000., 1000.)
        if threshold > 0 and expander_ratio > 1:
            envelope, level = _expand_audio(block, threshold, expander_ratio, envelope, level)
        # Reserve headroom and prevent int16 wraparound at high volume.
        yield np.rint(np.clip(block * scale, -.95 * 32767, .95 * 32767)).astype("<i2").tobytes()


def _mux_continuous_audio(video_path, output_path, unit_id, probe, track, duration, gain, gate_sigma=3.,
                          expander_ratio=4.):
    title = _continuous_audio_title(unit_id, probe, track, gain, gate_sigma, expander_ratio)
    _mux_pcm(video_path, output_path, title, duration,
             _continuous_pcm(track, gain, gate_sigma, expander_ratio), track["sample_rate"])


def _continuous_audio_title(unit_id, probe, track, gain, gate_sigma, expander_ratio):
    low, high = track["band_hz"]
    return (f"Probe {probe} unit {unit_id}; electrode {track['channel']} (zero-based); "
             f"continuous voltage {low:g}-{high:g} Hz; OE volume {gain:g}; "
             f"expander {expander_ratio:g}:1 below {gate_sigma:g} x noise sigma")


def _export_channel_videos(video_path, save_path, units, probe, track, duration, gain, gate_sigma,
                           expander_ratio):
    # Units sharing an electrode have identical audio. Encode it once on local
    # scratch, then copy both media streams while retaining each unit's label.
    audio_path = Path(track["path"]).with_suffix(".m4a")
    _mux_pcm(None, audio_path, f"Electrode {track['channel']}", duration,
             _continuous_pcm(track, gain, gate_sigma, expander_ratio), track["sample_rate"])
    for unit in units:
        title = _continuous_audio_title(unit, probe, track, gain, gate_sigma, expander_ratio)
        _mux_encoded_audio(video_path, audio_path, Path(save_path) / f"{unit}.mp4", title, duration)


def _mux_encoded_audio(video_path, audio_path, output_path, title, duration):
    partial = output_path.with_name(f".{output_path.stem}.partial.mp4")
    try:
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video_path), "-i", str(audio_path),
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-t", f"{duration:.9f}",
             "-movflags", "+faststart", "-metadata:s:a:0", f"title={title}",
             "-metadata:s:a:0", f"handler_name={title}", str(partial)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"{output_path.name} audio export failed (ffmpeg exit {result.returncode}):\n"
                               f"{result.stderr.strip()}")
        partial.replace(output_path)
    finally:
        partial.unlink(missing_ok=True)


def _load_audio_spikes(data, video_result):
    """Read spikes once for the batch, reusing the pose loader's exposure clock."""
    origin = data["adc_time_origin_s"]
    source = data["source"]
    kilosort = Path(source["kilosort_dir"])
    labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t")
    units = labels.loc[labels.KSLabel == "good", "cluster_id"].astype(int).tolist()
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r").ravel()
    times = np.load(source["spike_times_path"], mmap_mode="r").ravel()
    selected = np.isin(clusters, units)
    if source.get("selected_interval_s") is not None:
        start, end = source["selected_interval_s"]
        selected &= (times >= start + origin) & (times <= end + origin)
    cluster_ids = clusters[selected]
    clock = _video_audio_clock(data, video_result)
    seconds = _spike_times_on_video_clock(np.asarray(times[selected], dtype=float), clock)
    in_clip = (seconds >= 0) & (seconds < video_result["frames"] / video_result["fps"])
    cluster_ids, seconds = cluster_ids[in_clip], seconds[in_clip]
    order = np.argsort(cluster_ids, kind="stable")
    cluster_ids, seconds = cluster_ids[order], seconds[order]
    # Keep every good label, including units with no spikes in this clip.
    left = np.searchsorted(cluster_ids, units, side="left")
    right = np.searchsorted(cluster_ids, units, side="right")
    grouped = {int(unit): seconds[a:b] for unit, a, b in zip(units, left, right)}
    return units, grouped, data["camera_timing"]


def _spike_times_on_video_clock(spike_times, clock):
    """Invert only adjacent known exposures; sorted clicks stay silent at gaps."""
    video_times, oe_times = clock
    starts = np.flatnonzero(np.isfinite(oe_times[:-1]) & np.isfinite(oe_times[1:]))
    result = np.full(len(spike_times), np.nan)
    if not len(starts):
        return result
    interval = np.searchsorted(oe_times[starts], spike_times, side="right") - 1
    left = starts[np.maximum(interval, 0)]
    valid = (interval >= 0) & (spike_times < oe_times[left + 1])
    fraction = (spike_times[valid] - oe_times[left[valid]]) / (oe_times[left[valid] + 1] - oe_times[left[valid]])
    result[valid] = video_times[left[valid]] + fraction * (video_times[left[valid] + 1] - video_times[left[valid]])
    return result


def _click_blocks(samples, click, sample_count, block_samples):
    for start in range(0, sample_count, block_samples):
        stop = min(start + block_samples, sample_count)
        block = np.zeros(stop - start, dtype=np.float32)
        # Include the tail of a click that began before this audio block.
        first = np.searchsorted(samples, start - len(click) + 1)
        last = np.searchsorted(samples, stop)
        for sample in samples[first:last]:
            offset = int(sample) - start
            a, b = max(offset, 0), min(offset + len(click), len(block))
            block[a:b] += click[a - offset:b - offset]
        yield block


def _spike_pcm(spike_seconds, duration_s, sample_rate, gain):
    """Prepare click gain and stream PCM blocks without an intermediate WAV."""
    if not 0 < gain <= 1:
        raise ValueError("gain must be between 0 (exclusive) and 1.")
    sample_count = int(round(duration_s * sample_rate))
    spike_seconds = np.asarray(spike_seconds, dtype=float)
    selected = spike_seconds[(spike_seconds >= 0) & (spike_seconds < duration_s)]
    # A sub-sample spike at the end belongs to the final audio sample.
    samples = np.minimum(np.rint(selected * sample_rate).astype(np.int64), sample_count - 1)
    samples.sort()
    t = np.arange(int(.004 * sample_rate)) / sample_rate
    click = np.exp(-t / .0007) * np.cos(2 * np.pi * 1800 * t)
    click -= click.mean()
    click = (click / np.max(np.abs(click))).astype(np.float32)
    block_samples = sample_rate * 10
    # Two streaming passes give one global gain without clipping dense bursts.
    peak = max(float(np.max(np.abs(block)))
               for block in _click_blocks(samples, click, sample_count, block_samples))
    applied_gain = min(gain, .95 / peak) if peak else gain

    def blocks():
        for block in _click_blocks(samples, click, sample_count, block_samples):
            block *= applied_gain * 32767
            np.rint(block, out=block)
            yield block.astype("<i2").tobytes()

    info = dict(spike_count=len(samples), sample_rate_hz=sample_rate,
                requested_gain=gain, applied_gain=applied_gain,
                sound="4 ms exponentially decaying 1.8 kHz biphasic click")
    return info, blocks()


def write_spike_clicks(path, spike_seconds, duration_s, *, sample_rate=48_000, gain=.35):
    """Write short biphasic clicks with constant per-spike gain, in bounded RAM."""
    info, blocks = _spike_pcm(spike_seconds, duration_s, sample_rate, gain)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        for block in blocks:
            stream.writeframes(block)
    return info


def _mux_spike_audio(video_path, output_path, unit_id, probe, spike_seconds, duration, gain):
    audio_info, blocks = _spike_pcm(spike_seconds, duration, 48_000, gain)
    _mux_pcm(video_path, output_path, f"Probe {probe} unit {unit_id} spike clicks", duration, blocks, 48_000)
    return audio_info


def _mux_pcm(video_path, output_path, title, duration, blocks, sample_rate):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.stem}.partial{output_path.suffix}")
    video_input = [] if video_path is None else ["-i", str(video_path)]
    stream_map = [] if video_path is None else ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy"]
    # Pipe PCM directly to ffmpeg to keep memory bounded; an optional video
    # input is copied without decoding or re-encoding its frames.
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *video_input,
             "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
             *stream_map, "-c:a", "aac", "-b:a", "192k",
             "-threads", "1", "-t", f"{duration:.9f}", "-movflags", "+faststart",
             "-metadata:s:a:0", f"title={title}", str(partial)],
            stdin=subprocess.PIPE, stderr=log,
        )
        try:
            for block in blocks:
                process.stdin.write(block)
            process.stdin.close()
            if process.wait() != 0:
                raise _ffmpeg_error(process, log, f"{output_path.name} audio export")
            partial.replace(output_path)
        except BrokenPipeError as error:
            raise _ffmpeg_error(process, log, f"{output_path.name} audio export") from error
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            partial.unlink(missing_ok=True)
