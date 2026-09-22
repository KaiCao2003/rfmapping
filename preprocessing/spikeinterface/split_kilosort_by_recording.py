from pathlib import Path
import re
import shutil

import numpy as np


def split_kilosort_by_recording(data_base_folder: Path, remote_output: Path, date: str, sessions: list[int], probe_list: list[str], *,
                                num_channels: int = 384, dtype_bytes: int = 2, fs: int = 30000,
                                overwrite_outputs: bool = True, split_spike_length_npy_files: bool = True,
                                chunk_spikes: int = 1_000_000) -> None:
    # =====================================================
    # CONFIG - edit these values, then run this file directly
    # =====================================================

    # data_base_folder = Path("/mnt/ssd4.1")
    #
    # date = "260730"
    #
    # # Use integer session IDs, including multi-digit IDs when needed.
    # sessions = [1, 2, 10]
    #
    # # Set to ["A"], ["B"], or ["A", "B"]. Set to None to auto-detect probes
    # # from /mnt/ssd4.1/Data/<date>_<sessions>/Probe*/kilosort.
    # probes = ["A",]


    # =====================================================
    # HELPERS
    # =====================================================

    ONEBOX_RE = re.compile(r"OneBox-(?P<onebox_id>\d+)\.Probe(?P<probe>[A-Za-z0-9]+)$")

    def session_group_name() -> str:
        session_tag = (
            "".join(str(session_id) for session_id in sessions)
            if all(1 <= session_id <= 9 for session_id in sessions)
            else "sessions-" + "-".join(str(session_id) for session_id in sessions)
        )
        return f"{date}_{session_tag}"

    def raw_session_folder(session: int) -> Path:
        return data_base_folder / f"{date}_{session}"

    def find_continuous_dat(session: int, probe: str) -> Path:
        session_dir = raw_session_folder(session)
        matches = sorted(
            session_dir.glob(
                f"Record Node */experiment*/recording*/continuous/"
                f"OneBox-*.Probe{probe}/continuous.dat"
            )
        )

        if not matches:
            raise FileNotFoundError(
                f"No OneBox-*.Probe{probe}/continuous.dat found in {session_dir}"
            )

        if len(matches) > 1:
            match_list = "\n".join(f"  {path}" for path in matches)
            raise RuntimeError(
                f"Found more than one OneBox folder for session {session}, probe {probe}:\n"
                f"{match_list}"
            )

        return matches[0]

    def onebox_id_from_dat(dat_path: Path) -> str:
        match = ONEBOX_RE.match(dat_path.parent.name)
        if match is None:
            return "unknown"
        return match.group("onebox_id")

    def sample_count(path: Path) -> int:
        bytes_per_sample = num_channels * dtype_bytes
        size_bytes = path.stat().st_size
        if size_bytes % bytes_per_sample != 0:
            raise RuntimeError(
                f"{path} size is not divisible by {bytes_per_sample} bytes/sample"
            )
        return size_bytes // bytes_per_sample

    def concat_probe_folder(probe: str) -> Path:
        return data_base_folder / "Data" / session_group_name() / f"Probe{probe}"

    def split_ranges(lengths: list[int]) -> list[tuple[int, int]]:
        ranges = []
        start = 0
        for length in lengths:
            end = start + length
            ranges.append((start, end))
            start = end
        return ranges

    def validate_config() -> None:

        if len(set(sessions)) != len(sessions):
            raise ValueError(f"SESSIONS contains duplicates: {sessions}")

        if chunk_spikes <= 0:
            raise ValueError("CHUNK_SPIKES must be positive")

    def load_spike_times(ks_folder: Path) -> np.ndarray:
        spike_times = np.load(ks_folder / "spike_times.npy", mmap_mode="r").reshape(-1)
        if len(spike_times) == 0:
            raise RuntimeError(f"No spikes found in {ks_folder / 'spike_times.npy'}")
        return spike_times

    def spike_index_ranges(
            spike_times: np.ndarray, sample_ranges: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        ranges = []
        for start_sample, end_sample in sample_ranges:
            start_index = int(np.searchsorted(spike_times, start_sample, side="left"))
            end_index = int(np.searchsorted(spike_times, end_sample, side="left"))
            ranges.append((start_index, end_index))
        return ranges

    def npy_shape(path: Path) -> tuple[int, ...] | None:
        try:
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
        except ValueError:
            return None
        return tuple(arr.shape)

    def split_npy_file(
            source: Path,
            destination: Path,
            start_index: int,
            end_index: int,
            sample_offset: int,
    ) -> None:
        source_arr = np.load(source, mmap_mode="r", allow_pickle=False)
        out_shape = (end_index - start_index,) + tuple(source_arr.shape[1:])
        destination.parent.mkdir(parents=True, exist_ok=True)

        out = np.lib.format.open_memmap(
            destination,
            mode="w+",
            dtype=source_arr.dtype,
            shape=out_shape,
        )

        write_pos = 0
        for chunk_start in range(start_index, end_index, chunk_spikes):
            chunk_end = min(chunk_start + chunk_spikes, end_index)
            chunk = source_arr[chunk_start:chunk_end]

            if source.name == "spike_times.npy":
                out[write_pos: write_pos + len(chunk)] = chunk - sample_offset
            else:
                out[write_pos: write_pos + len(chunk)] = chunk

            write_pos += len(chunk)

        del out

    def copy_metadata_files(
            ks_folder: Path,
            out_folder: Path,
            spike_length_npy_files: set[str],
    ) -> None:
        out_folder.mkdir(parents=True, exist_ok=True)

        for item in ks_folder.iterdir():
            target = out_folder / item.name

            if item.is_dir():
                shutil.copytree(item, target)
                continue

            if item.suffix == ".npy" and item.name in spike_length_npy_files:
                continue

            shutil.copy2(item, target)

    def discover_spike_length_npy_files(ks_folder: Path, n_spikes: int) -> set[str]:
        spike_length_files = {"spike_times.npy", "spike_clusters.npy"}

        if not split_spike_length_npy_files:
            return spike_length_files

        for path in sorted(ks_folder.glob("*.npy")):
            shape = npy_shape(path)
            if shape and shape[0] == n_spikes:
                spike_length_files.add(path.name)

        return spike_length_files

    def prepare_output_folder(out_folder: Path) -> None:
        if out_folder.exists():
            if not overwrite_outputs:
                raise FileExistsError(
                    f"{out_folder} already exists. Set overwrite_outputs=True to replace it."
                )
            shutil.rmtree(out_folder)

    def split_probe(probe: str) -> None:
        probe_folder = concat_probe_folder(probe)
        ks_folder = probe_folder / "kilosort"
        concat_raw = probe_folder / "concat" / "traces_cached_seg0.raw"

        if not ks_folder.exists():
            raise FileNotFoundError(f"Missing Kilosort folder: {ks_folder}")

        if not concat_raw.exists():
            raise FileNotFoundError(f"Missing concat raw file: {concat_raw}")

        dat_paths = [find_continuous_dat(session, probe) for session in sessions]
        recording_lengths = [sample_count(path) for path in dat_paths]
        concat_samples = sample_count(concat_raw)
        sample_ranges = split_ranges(recording_lengths)

        print()
        print("=" * 72)
        print(f"Probe {probe}: {session_group_name()}")
        print("=" * 72)

        print("Recording lengths")
        for session, dat_path, length, (start, end) in zip(
                sessions, dat_paths, recording_lengths, sample_ranges
        ):
            print(
                f"  session {session}: "
                f"OneBox-{onebox_id_from_dat(dat_path)}, "
                f"samples {length:,}, "
                f"sec {length / fs:.3f}, "
                f"concat range [{start:,}, {end:,})"
            )
            print(f"    {dat_path}")

        print(f"  concat: samples {concat_samples:,}, sec {concat_samples / fs:.3f}")

        if sum(recording_lengths) != concat_samples:
            raise RuntimeError(
                f"ERROR: sum of recordings ({sum(recording_lengths):,}) "
                f"!= concat ({concat_samples:,})"
            )

        print()
        print("Loading Kilosort spike times")
        spike_times = load_spike_times(ks_folder)
        n_spikes = len(spike_times)
        valid_start = int(np.searchsorted(spike_times, 0, side="left"))
        valid_end = int(np.searchsorted(spike_times, concat_samples, side="left"))

        if valid_start != 0 or valid_end != n_spikes:
            print(
                f"  excluding {n_spikes - (valid_end - valid_start):,} spike(s) "
                f"outside valid sample range [0, {concat_samples:,})"
            )

        valid_spike_times = spike_times[valid_start:valid_end]
        index_ranges = spike_index_ranges(valid_spike_times, sample_ranges)

        if index_ranges[0][0] != 0 or index_ranges[-1][1] != len(valid_spike_times):
            raise RuntimeError(
                "Some spike_times are outside the concatenated recording range. "
                f"Covered spikes {index_ranges[0][0]:,}..{index_ranges[-1][1]:,}, "
                f"total spikes {n_spikes:,}."
            )

        print(f"  valid spikes: {len(valid_spike_times):,} / {n_spikes:,}")
        for session, (sample_start, sample_end), (idx_start, idx_end) in zip(
                sessions, sample_ranges, index_ranges
        ):
            print(
                f"  session {session}: "
                f"spikes {idx_end - idx_start:,}, "
                f"sample range [{sample_start:,}, {sample_end:,})"
            )

        spike_length_npy_files = discover_spike_length_npy_files(ks_folder, n_spikes)

        print()
        print("Spike-length .npy files to split")
        for name in sorted(spike_length_npy_files):
            print(f"  {name}")

        for session, sample_range, index_range in zip(sessions, sample_ranges, index_ranges):
            out_folder = probe_folder / f"kilosort_{session}"
            sample_start, _ = sample_range
            idx_start, idx_end = index_range

            print()
            print(f"Writing session {session} -> {out_folder}")

            prepare_output_folder(out_folder)
            copy_metadata_files(ks_folder, out_folder, spike_length_npy_files)

            for name in sorted(spike_length_npy_files):
                print(f"  splitting {name}")
                split_npy_file(
                    source=ks_folder / name,
                    destination=out_folder / name,
                    start_index=valid_start + idx_start,
                    end_index=valid_start + idx_end,
                    sample_offset=sample_start,
                )

            remote_folder = (
                remote_output
                / f"{date}_{session}"
                / "kilosort"
                / f"Probe{probe}"
                / f"kilosort_{session}"
            )
            print(f"Copying to remote -> {remote_folder}")
            shutil.copytree(out_folder, remote_folder, dirs_exist_ok=True)

            pipeline_remote_folder = (
                remote_output
                / session_group_name()
                / "pipeline"
                / session_group_name()
                / f"Probe{probe}"
                / f"kilosort_{session}"
            )
            print(f"Deferring pipeline archive copy -> {pipeline_remote_folder}")
            # Uploaded with the complete local pipeline folder after analysis.
            # shutil.copytree(out_folder, pipeline_remote_folder, dirs_exist_ok=True)

        print()
        print(f"Probe {probe} done.")

    validate_config()
    for probe in probe_list:
        split_probe(probe)

    print()
    print("=" * 72)
    print("DONE")
    print("=" * 72)
    print("Cluster IDs are preserved.")
    print("Spike times in each output are shifted back to that recording's time zero.")
