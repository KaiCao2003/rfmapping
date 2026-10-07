from __future__ import annotations

import csv
import json
import shutil
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from numpy.typing import NDArray

from Utils.si_utils import (
    compute_template_ptp_summary,
    export_waveform_data,
)


FloatArray = NDArray[np.floating[Any]]
IntArray = NDArray[np.integer[Any]]


def _session_group_name(date: str, sessions: list[int]) -> str:
    session_tag = (
        "".join(str(session_id) for session_id in sessions)
        if all(1 <= session_id <= 9 for session_id in sessions)
        else "sessions-" + "-".join(str(session_id) for session_id in sessions)
    )
    return f"{date}_{session_tag}"


@dataclass(frozen=True)
class UnitPositionStatistics:
    unit_ids: IntArray
    spike_counts: IntArray
    modal_shank_ids: IntArray
    modal_shank_spike_counts: IntArray
    modal_shank_fractions: FloatArray
    x_medians_um: FloatArray
    y_medians_um: FloatArray
    x_mads_um: FloatArray
    y_mads_um: FloatArray
    x_iqrs_um: FloatArray
    y_iqrs_um: FloatArray

    def index_by_unit_id(self) -> dict[int, int]:
        return {
            int(unit_id): unit_index
            for unit_index, unit_id in enumerate(self.unit_ids)
        }


@dataclass(frozen=True)
class CanonicalWaveforms:
    unit_ids: IntArray
    spike_counts: IntArray
    templates_uv: FloatArray
    pc_feature_norm_medians: FloatArray
    pc_feature_norm_mads: FloatArray
    pc_feature_norm_iqrs: FloatArray
    channel_ids: IntArray
    channel_positions_um: FloatArray
    channel_shank_ids: IntArray
    sampling_frequency_hz: float
    gain_to_uv: float
    nbefore: int
    time_ms: FloatArray

    def index_by_unit_id(self) -> dict[int, int]:
        return {
            int(unit_id): unit_index
            for unit_index, unit_id in enumerate(self.unit_ids)
        }


def read_good_unit_ids(kilosort_dir: str | Path) -> IntArray:
    kilosort_dir = Path(kilosort_dir)
    with (kilosort_dir / "cluster_group.tsv").open(newline="") as tsv_file:
        rows = list(csv.DictReader(tsv_file, delimiter="\t"))

    label_column = "KSLabel" if "KSLabel" in rows[0] else "group"
    return np.asarray(
        sorted(
            int(row["cluster_id"])
            for row in rows
            if row[label_column] == "good"
        ),
        dtype=np.int64,
    )


def summarize_kilosort_positions(
    kilosort_dir: str | Path,
    *,
    unit_ids: IntArray,
) -> UnitPositionStatistics:
    kilosort_dir = Path(kilosort_dir)
    spike_clusters = np.load(
        kilosort_dir / "spike_clusters.npy",
        mmap_mode="r",
    ).reshape(-1)
    spike_positions = np.load(
        kilosort_dir / "spike_positions.npy",
        mmap_mode="r",
    )

    unit_ids = np.asarray(unit_ids, dtype=np.int64)
    lookup = np.zeros(
        max(int(np.max(spike_clusters)), int(np.max(unit_ids))) + 1,
        dtype=bool,
    )
    lookup[unit_ids] = True
    selected_mask = lookup[spike_clusters]
    selected_clusters = np.asarray(spike_clusters[selected_mask])
    selected_positions = np.asarray(spike_positions[selected_mask])

    group_order = np.argsort(selected_clusters, kind="stable")
    grouped_clusters = selected_clusters[group_order]
    grouped_positions = selected_positions[group_order]
    observed_unit_ids, group_starts, spike_counts = np.unique(
        grouped_clusters,
        return_index=True,
        return_counts=True,
    )

    channel_positions = np.load(kilosort_dir / "channel_positions.npy")
    channel_shank_ids = np.load(
        kilosort_dir / "channel_shanks.npy"
    ).astype(np.int64)
    shank_ids = np.unique(channel_shank_ids)
    shank_x_centers = np.asarray([
        np.median(channel_positions[channel_shank_ids == shank_id, 0])
        for shank_id in shank_ids
    ])
    center_order = np.argsort(shank_x_centers)
    ordered_shank_ids = shank_ids[center_order]
    ordered_centers = shank_x_centers[center_order]
    shank_boundaries = (ordered_centers[:-1] + ordered_centers[1:]) / 2.0

    modal_shank_ids: list[int] = []
    modal_shank_spike_counts: list[int] = []
    modal_shank_fractions: list[float] = []
    x_medians_um: list[float] = []
    y_medians_um: list[float] = []
    x_mads_um: list[float] = []
    y_mads_um: list[float] = []
    x_iqrs_um: list[float] = []
    y_iqrs_um: list[float] = []

    for group_start, spike_count in zip(group_starts, spike_counts):
        group_stop = int(group_start + spike_count)
        unit_positions = grouped_positions[group_start:group_stop]
        assigned_shanks = ordered_shank_ids[
            np.searchsorted(shank_boundaries, unit_positions[:, 0])
        ]
        present_shanks, shank_counts = np.unique(
            assigned_shanks,
            return_counts=True,
        )
        modal_index = int(np.argmax(shank_counts))
        modal_shank_id = int(present_shanks[modal_index])
        modal_count = int(shank_counts[modal_index])
        modal_positions = unit_positions[assigned_shanks == modal_shank_id]

        median_xy = np.median(modal_positions, axis=0)
        mad_xy = np.median(np.abs(modal_positions - median_xy), axis=0)
        q25_xy, q75_xy = np.percentile(modal_positions, [25, 75], axis=0)
        iqr_xy = q75_xy - q25_xy

        modal_shank_ids.append(modal_shank_id)
        modal_shank_spike_counts.append(modal_count)
        modal_shank_fractions.append(modal_count / int(spike_count))
        x_medians_um.append(float(median_xy[0]))
        y_medians_um.append(float(median_xy[1]))
        x_mads_um.append(float(mad_xy[0]))
        y_mads_um.append(float(mad_xy[1]))
        x_iqrs_um.append(float(iqr_xy[0]))
        y_iqrs_um.append(float(iqr_xy[1]))

    return UnitPositionStatistics(
        unit_ids=np.asarray(observed_unit_ids, dtype=np.int64),
        spike_counts=np.asarray(spike_counts, dtype=np.int64),
        modal_shank_ids=np.asarray(modal_shank_ids, dtype=np.int64),
        modal_shank_spike_counts=np.asarray(
            modal_shank_spike_counts,
            dtype=np.int64,
        ),
        modal_shank_fractions=np.asarray(modal_shank_fractions, dtype=float),
        x_medians_um=np.asarray(x_medians_um, dtype=float),
        y_medians_um=np.asarray(y_medians_um, dtype=float),
        x_mads_um=np.asarray(x_mads_um, dtype=float),
        y_mads_um=np.asarray(y_mads_um, dtype=float),
        x_iqrs_um=np.asarray(x_iqrs_um, dtype=float),
        y_iqrs_um=np.asarray(y_iqrs_um, dtype=float),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _write_csv(
    path: Path,
    columns: list[str],
    rows: list[list[object]],
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)
        writer.writerows(rows)


def _update_position_root(
    data_dir: Path,
    *,
    probe_name: str,
    generated_at_utc: str,
    source_session: str,
    unit_scope: str,
    unit_count: int,
) -> None:
    position_root = data_dir / "spike_position"
    config_path = position_root / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        config = {
            "schema_name": "rfmapping-spike-position-config",
            "schema_version": 2,
            "probes": {},
        }
    probe_directory = f"Probe{probe_name}"
    config["schema_version"] = 2
    config["updated_at_utc"] = generated_at_utc
    config["probes"][probe_directory] = {
        "manifest": f"{probe_directory}/manifest.json",
        "run": {
            "source_session": source_session,
            "method": "concat_kilosort_spike_position_median",
        },
    }
    _write_json(config_path, config)
    with (position_root / "run.log").open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"{generated_at_utc} status=complete"
            f" session={source_session}"
            f" probe={probe_directory}"
            f" scope={unit_scope} units={unit_count}"
            f" method=concat_kilosort_spike_position_median\n"
        )


def _write_canonical_position_artifacts(
    data_dir: Path,
    *,
    probe_name: str,
    session_name: str,
    source_kilosort_dir: Path,
    canonical_stats: UnitPositionStatistics,
) -> None:
    probe_directory = f"Probe{probe_name}"
    position_dir = data_dir / "spike_position" / probe_directory
    if position_dir.exists():
        shutil.rmtree(position_dir)
    position_dir.mkdir(parents=True)

    position_rows = [
        [
            unit_index,
            int(unit_id),
            float(canonical_stats.x_medians_um[unit_index]),
            float(canonical_stats.y_medians_um[unit_index]),
        ]
        for unit_index, unit_id in enumerate(canonical_stats.unit_ids)
    ]
    position_columns = ["unit_index", "unit_id", "x_um", "y_um"]
    _write_csv(position_dir / "positions.csv", position_columns, position_rows)
    _write_csv(position_dir / "positions.probe", position_columns, position_rows)

    qc_rows = [
        [
            int(unit_id),
            int(canonical_stats.modal_shank_ids[unit_index]),
            int(canonical_stats.spike_counts[unit_index]),
            int(canonical_stats.modal_shank_spike_counts[unit_index]),
            float(canonical_stats.modal_shank_fractions[unit_index]),
            float(canonical_stats.x_mads_um[unit_index]),
            float(canonical_stats.y_mads_um[unit_index]),
            float(canonical_stats.x_iqrs_um[unit_index]),
            float(canonical_stats.y_iqrs_um[unit_index]),
        ]
        for unit_index, unit_id in enumerate(canonical_stats.unit_ids)
    ]
    _write_csv(
        position_dir / "canonical_qc.csv",
        [
            "unit_id",
            "shank_id",
            "concat_spike_count",
            "modal_shank_spike_count",
            "modal_shank_fraction",
            "x_mad_um",
            "y_mad_um",
            "x_iqr_um",
            "y_iqr_um",
        ],
        qc_rows,
    )

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    _write_json(
        position_dir / "manifest.json",
        {
            "schema_name": "rfmapping-spike-positions",
            "schema_version": 2,
            "generated_at_utc": generated_at_utc,
            "session": {
                "name": session_name,
                "probe": probe_name,
                "kilosort_dir": str(source_kilosort_dir),
            },
            "units": {
                "scope": "good",
                "count": len(position_rows),
            },
            "spike_position": {
                "method": "concat_kilosort_spike_position_median",
                "shank_assignment": "nearest_shank_x_center",
                "statistic": "median_within_modal_shank",
            },
            "files": {
                "positions": "positions.probe",
                "positions_legacy": "positions.csv",
                "canonical_qc": "canonical_qc.csv",
            },
        },
    )
    _update_position_root(
        data_dir,
        probe_name=probe_name,
        generated_at_utc=generated_at_utc,
        source_session=session_name,
        unit_scope="good",
        unit_count=len(position_rows),
    )


def _write_session_position_artifacts(
    data_dir: Path,
    *,
    probe_name: str,
    session_name: str,
    concat_session_name: str,
    source_kilosort_dir: Path,
    canonical_stats: UnitPositionStatistics,
    session_stats: UnitPositionStatistics,
    min_local_spikes: int,
) -> None:
    probe_directory = f"Probe{probe_name}"
    position_dir = data_dir / "spike_position" / probe_directory
    if position_dir.exists():
        shutil.rmtree(position_dir)
    position_dir.mkdir(parents=True)

    canonical_index = canonical_stats.index_by_unit_id()
    position_rows: list[list[object]] = []
    deviation_rows: list[list[object]] = []
    for unit_index, unit_id_value in enumerate(session_stats.unit_ids):
        unit_id = int(unit_id_value)
        canonical_unit_index = canonical_index[unit_id]
        canonical_x = float(canonical_stats.x_medians_um[canonical_unit_index])
        canonical_y = float(canonical_stats.y_medians_um[canonical_unit_index])
        local_x = float(session_stats.x_medians_um[unit_index])
        local_y = float(session_stats.y_medians_um[unit_index])
        spike_count = int(session_stats.spike_counts[unit_index])
        canonical_shank = int(
            canonical_stats.modal_shank_ids[canonical_unit_index]
        )
        local_shank = int(session_stats.modal_shank_ids[unit_index])
        if local_shank != canonical_shank:
            local_status = "ambiguous_shank"
        elif spike_count < min_local_spikes:
            local_status = "insufficient_spikes"
        else:
            local_status = "available"

        position_rows.append([
            unit_index,
            unit_id,
            canonical_x,
            canonical_y,
        ])
        deviation_rows.append([
            unit_id,
            session_name,
            spike_count,
            local_status,
            local_shank,
            int(session_stats.modal_shank_spike_counts[unit_index]),
            float(session_stats.modal_shank_fractions[unit_index]),
            local_x,
            local_y,
            local_x - canonical_x,
            local_y - canonical_y,
            float(session_stats.x_mads_um[unit_index]),
            float(session_stats.y_mads_um[unit_index]),
            float(session_stats.x_iqrs_um[unit_index]),
            float(session_stats.y_iqrs_um[unit_index]),
        ])

    position_columns = ["unit_index", "unit_id", "x_um", "y_um"]
    _write_csv(position_dir / "positions.csv", position_columns, position_rows)
    _write_csv(position_dir / "positions.probe", position_columns, position_rows)
    _write_csv(
        position_dir / "session_position_deviation.csv",
        [
            "unit_id",
            "session_name",
            "session_spike_count",
            "local_status",
            "local_shank_id",
            "session_modal_shank_spike_count",
            "session_modal_shank_fraction",
            "local_x_um",
            "local_y_um",
            "delta_x_um",
            "delta_y_um",
            "local_x_mad_um",
            "local_y_mad_um",
            "local_x_iqr_um",
            "local_y_iqr_um",
        ],
        deviation_rows,
    )

    generated_at_utc = datetime.now(timezone.utc).isoformat()
    _write_json(
        position_dir / "manifest.json",
        {
            "schema_name": "rfmapping-spike-positions",
            "schema_version": 2,
            "generated_at_utc": generated_at_utc,
            "session": {
                "name": session_name,
                "probe": probe_name,
                "kilosort_dir": str(source_kilosort_dir),
            },
            "canonical_source": {
                "session": concat_session_name,
                "method": "concat_kilosort_spike_position_median",
            },
            "units": {
                "scope": "present_good",
                "count": len(position_rows),
            },
            "local_position": {
                "role": "post_registration_residual_only",
                "min_spikes": min_local_spikes,
            },
            "files": {
                "positions": "positions.probe",
                "positions_legacy": "positions.csv",
                "session_deviation": "session_position_deviation.csv",
            },
        },
    )
    _update_position_root(
        data_dir,
        probe_name=probe_name,
        generated_at_utc=generated_at_utc,
        source_session=session_name,
        unit_scope="present_good",
        unit_count=len(position_rows),
    )


def _find_remote_raw_file(
    session_dir: Path,
    *,
    date: str,
    probe_name: str,
) -> Path:
    return next(
        (session_dir / date).glob(
            "*/experiment*/recording*/continuous/"
            f"OneBox-*.Probe{probe_name}/continuous.dat"
        )
    )


def _summarize_kilosort_pc_feature_norms(
    kilosort_dir: Path,
    *,
    unit_ids: IntArray,
) -> tuple[IntArray, FloatArray, FloatArray, FloatArray]:
    spike_clusters = np.load(
        kilosort_dir / "spike_clusters.npy",
        mmap_mode="r",
    ).reshape(-1)
    amplitudes = np.load(
        kilosort_dir / "amplitudes.npy",
        mmap_mode="r",
    ).reshape(-1)

    lookup = np.zeros(
        max(int(np.max(spike_clusters)), int(np.max(unit_ids))) + 1,
        dtype=bool,
    )
    lookup[unit_ids] = True
    selected_mask = lookup[spike_clusters]
    selected_clusters = np.asarray(spike_clusters[selected_mask])
    selected_amplitudes = np.asarray(amplitudes[selected_mask])
    group_order = np.argsort(selected_clusters, kind="stable")
    grouped_clusters = selected_clusters[group_order]
    grouped_amplitudes = selected_amplitudes[group_order]
    _, group_starts, spike_counts = np.unique(
        grouped_clusters,
        return_index=True,
        return_counts=True,
    )

    medians: list[float] = []
    mads: list[float] = []
    iqrs: list[float] = []
    for group_start, spike_count in zip(group_starts, spike_counts):
        group_stop = int(group_start + spike_count)
        unit_amplitudes = grouped_amplitudes[group_start:group_stop]
        median = float(np.median(unit_amplitudes))
        q25, q75 = np.percentile(unit_amplitudes, [25, 75])
        medians.append(median)
        mads.append(float(np.median(np.abs(unit_amplitudes - median))))
        iqrs.append(float(q75 - q25))

    return (
        np.asarray(spike_counts, dtype=np.int64),
        np.asarray(medians, dtype=float),
        np.asarray(mads, dtype=float),
        np.asarray(iqrs, dtype=float),
    )


def _load_canonical_waveforms(
    kilosort_dir: Path,
    *,
    unit_ids: IntArray,
    gain_to_uv: float,
) -> CanonicalWaveforms:
    spike_counts, norm_medians, norm_mads, norm_iqrs = (
        _summarize_kilosort_pc_feature_norms(
            kilosort_dir,
            unit_ids=unit_ids,
        )
    )
    templates = np.asarray(
        np.load(kilosort_dir / "templates.npy", mmap_mode="r")[unit_ids],
        dtype=np.float32,
    )
    whitening_mat_inv = np.asarray(
        np.load(kilosort_dir / "whitening_mat_inv.npy"),
        dtype=np.float32,
    )
    templates_uv = np.asarray(
        np.einsum("ij,utj->uti", whitening_mat_inv, templates)
        * gain_to_uv,
        dtype=np.float32,
    )
    ops = np.load(kilosort_dir / "ops.npy", allow_pickle=True).item()
    sampling_frequency_hz = float(ops["fs"])
    nbefore = int(ops["nt0min"])
    time_ms = (
        np.arange(templates_uv.shape[1]) - nbefore
    ) / sampling_frequency_hz * 1_000.0

    return CanonicalWaveforms(
        unit_ids=np.asarray(unit_ids, dtype=np.int64),
        spike_counts=spike_counts,
        templates_uv=templates_uv,
        pc_feature_norm_medians=norm_medians,
        pc_feature_norm_mads=norm_mads,
        pc_feature_norm_iqrs=norm_iqrs,
        channel_ids=np.asarray(
            np.load(kilosort_dir / "channel_map.npy")
        ).reshape(-1),
        channel_positions_um=np.asarray(
            np.load(kilosort_dir / "channel_positions.npy")
        ),
        channel_shank_ids=np.asarray(
            np.load(kilosort_dir / "channel_shanks.npy"),
            dtype=np.int64,
        ).reshape(-1),
        sampling_frequency_hz=sampling_frequency_hz,
        gain_to_uv=gain_to_uv,
        nbefore=nbefore,
        time_ms=time_ms,
    )


def _select_spikes(
    sorting: Any,
    *,
    max_spikes_per_unit: int,
    waveform_seed: int,
) -> NDArray[np.void]:
    selected_indices = si.random_spikes_selection(
        sorting,
        method="uniform",
        max_spikes_per_unit=max_spikes_per_unit,
        margin_size=None,
        seed=waveform_seed,
    )
    return sorting.to_spike_vector()[selected_indices]


def _write_canonical_waveform_qc(
    waveform_dir: Path,
    *,
    canonical_waveforms: CanonicalWaveforms,
    unit_ids: IntArray,
) -> None:
    canonical_index = canonical_waveforms.index_by_unit_id()
    rows = []
    for unit_id_value in unit_ids:
        unit_id = int(unit_id_value)
        unit_index = canonical_index[unit_id]
        rows.append([
            unit_id,
            int(canonical_waveforms.spike_counts[unit_index]),
            float(canonical_waveforms.pc_feature_norm_medians[unit_index]),
            float(canonical_waveforms.pc_feature_norm_mads[unit_index]),
            float(canonical_waveforms.pc_feature_norm_iqrs[unit_index]),
        ])
    _write_csv(
        waveform_dir / "canonical_waveform_qc.csv",
        [
            "unit_id",
            "concat_spike_count",
            "kilosort_pc_feature_norm_median",
            "kilosort_pc_feature_norm_mad",
            "kilosort_pc_feature_norm_iqr",
        ],
        rows,
    )


def _export_waveforms(
    data_dir: Path,
    *,
    probe_name: str,
    session_name: str,
    concat_session_name: str,
    sorting_kilosort_dir: Path,
    source_kilosort_dir: Path,
    canonical_kilosort_dir: Path,
    source_raw_file: Path,
    canonical_waveforms: CanonicalWaveforms,
    unit_scope: str,
    max_spikes_per_unit: int,
    waveform_seed: int,
) -> None:
    waveform_dir = data_dir / "waveform" / f"Probe{probe_name}"
    if waveform_dir.exists():
        shutil.rmtree(waveform_dir)

    sorting = se.read_kilosort(sorting_kilosort_dir, keep_good_only=True)
    unit_ids = np.asarray(sorting.unit_ids, dtype=np.int64)
    canonical_unit_index = canonical_waveforms.index_by_unit_id()
    template_indices = np.asarray([
        canonical_unit_index[int(unit_id)]
        for unit_id in unit_ids
    ])
    template_array = canonical_waveforms.templates_uv[template_indices]
    template_ptp_summary = compute_template_ptp_summary(template_array)
    selected_spikes = _select_spikes(
        sorting,
        max_spikes_per_unit=max_spikes_per_unit,
        waveform_seed=waveform_seed,
    )

    bytes_per_frame = (
        len(canonical_waveforms.channel_ids)
        * np.dtype("int16").itemsize
    )
    recording_num_frames = source_raw_file.stat().st_size // bytes_per_frame
    recording_duration_minutes = (
        recording_num_frames
        / canonical_waveforms.sampling_frequency_hz
        / 60.0
    )
    pre_spike_ms = (
        canonical_waveforms.nbefore
        / canonical_waveforms.sampling_frequency_hz
        * 1_000.0
    )
    post_spike_ms = (
        (template_array.shape[1] - canonical_waveforms.nbefore)
        / canonical_waveforms.sampling_frequency_hz
        * 1_000.0
    )
    export_waveform_data(
        data_dir,
        source_session=session_name,
        source_probe=probe_name,
        source_kilosort_dir=source_kilosort_dir,
        source_raw_file=source_raw_file,
        only_good_units=True,
        sorting=sorting,
        unit_ids=unit_ids,
        selected_spikes=selected_spikes,
        template_array=template_array,
        template_ptp_summary=template_ptp_summary,
        channel_ids=canonical_waveforms.channel_ids,
        channel_locations=canonical_waveforms.channel_positions_um,
        channel_shank_ids=canonical_waveforms.channel_shank_ids,
        sampling_frequency=canonical_waveforms.sampling_frequency_hz,
        recording_num_frames=recording_num_frames,
        recording_duration_minutes=recording_duration_minutes,
        time_ms=canonical_waveforms.time_ms,
        nbefore=canonical_waveforms.nbefore,
        pre_spike_ms=pre_spike_ms,
        post_spike_ms=post_spike_ms,
        max_spikes_per_unit=max_spikes_per_unit,
        waveform_seed=waveform_seed,
        run_config={
            "template_source": "concat",
            "template_session": concat_session_name,
            "spike_samples_source": session_name,
            "selected_spikes_source": session_name,
            "template_method": "kilosort_cluster_template_unwhitened",
            "template_units": "uV",
        },
    )
    _write_canonical_waveform_qc(
        waveform_dir,
        canonical_waveforms=canonical_waveforms,
        unit_ids=unit_ids,
    )
    manifest_path = waveform_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["units"]["scope"] = unit_scope
    manifest["waveform"].update({
        "template_method": "kilosort_cluster_template_unwhitened",
        "template_spike_scope": "all_concat_spikes_for_unit",
        "unwhitening_matrix": "whitening_mat_inv.npy",
        "template_units": "uV",
        "selection_role": "display_and_time_coverage_only",
        "gain_to_uv": canonical_waveforms.gain_to_uv,
    })
    manifest["canonical_source"] = {
        "session": concat_session_name,
        "kilosort_dir": str(canonical_kilosort_dir),
        "template_method": "kilosort_cluster_template_unwhitened",
        "pc_feature_norms_role": "qc_only_not_template_scaling",
    }
    if session_name == concat_session_name:
        manifest["session_spikes"] = {
            "all": "concat_sorting",
            "selected": "uniform_concat_selection",
        }
    else:
        manifest["session_spikes"] = {
            "all": "session_split_sorting",
            "selected": "uniform_session_local_selection",
        }
    manifest["files"]["canonical_waveform_qc"] = "canonical_waveform_qc.csv"
    _write_json(manifest_path, manifest)


def generate_canonical_unit_artifacts(
    *,
    data_base_folder: Path,
    remote_output: Path,
    date: str,
    sessions: list[int],
    probe_list: list[str],
    min_local_spikes: int = 100,
    max_spikes_per_unit: int = 2_000,
    waveform_seed: int = 0,
    gain_to_uv: float = 0.195,
) -> None:
    concat_session_name = _session_group_name(date, sessions)
    concat_data_dir = remote_output / concat_session_name / "data"

    for probe_name in probe_list:
        print(f"----- Canonical artifacts for Probe{probe_name} -----")
        probe_folder = (
            data_base_folder
            / "Data"
            / concat_session_name
            / f"Probe{probe_name}"
        )
        concat_kilosort_dir = probe_folder / "kilosort"
        good_unit_ids = read_good_unit_ids(concat_kilosort_dir)
        canonical_stats = summarize_kilosort_positions(
            concat_kilosort_dir,
            unit_ids=good_unit_ids,
        )
        _write_canonical_position_artifacts(
            concat_data_dir,
            probe_name=probe_name,
            session_name=concat_session_name,
            source_kilosort_dir=concat_kilosort_dir,
            canonical_stats=canonical_stats,
        )

        canonical_waveforms = _load_canonical_waveforms(
            concat_kilosort_dir,
            unit_ids=good_unit_ids,
            gain_to_uv=gain_to_uv,
        )
        concat_raw_file = probe_folder / "concat" / "traces_cached_seg0.raw"
        _export_waveforms(
            concat_data_dir,
            probe_name=probe_name,
            session_name=concat_session_name,
            concat_session_name=concat_session_name,
            sorting_kilosort_dir=concat_kilosort_dir,
            source_kilosort_dir=concat_kilosort_dir,
            canonical_kilosort_dir=concat_kilosort_dir,
            source_raw_file=concat_raw_file,
            canonical_waveforms=canonical_waveforms,
            unit_scope="good",
            max_spikes_per_unit=max_spikes_per_unit,
            waveform_seed=waveform_seed,
        )

        source_raw_files = [
            _find_remote_raw_file(
                remote_output / f"{date}_{session_id}",
                date=date,
                probe_name=probe_name,
            )
            for session_id in sessions
        ]

        for session_id, source_raw_file in zip(
            sessions,
            source_raw_files,
        ):
            session_name = f"{date}_{session_id}"
            local_kilosort_dir = probe_folder / f"kilosort_{session_id}"
            session_stats = summarize_kilosort_positions(
                local_kilosort_dir,
                unit_ids=good_unit_ids,
            )
            session_dir = remote_output / session_name
            data_dir = session_dir / "data"
            source_kilosort_dir = (
                session_dir
                / "kilosort"
                / f"Probe{probe_name}"
                / f"kilosort_{session_id}"
            )
            _write_session_position_artifacts(
                data_dir,
                probe_name=probe_name,
                session_name=session_name,
                concat_session_name=concat_session_name,
                source_kilosort_dir=source_kilosort_dir,
                canonical_stats=canonical_stats,
                session_stats=session_stats,
                min_local_spikes=min_local_spikes,
            )
            _export_waveforms(
                data_dir,
                probe_name=probe_name,
                session_name=session_name,
                concat_session_name=concat_session_name,
                sorting_kilosort_dir=local_kilosort_dir,
                source_kilosort_dir=source_kilosort_dir,
                canonical_kilosort_dir=concat_kilosort_dir,
                source_raw_file=source_raw_file,
                canonical_waveforms=canonical_waveforms,
                unit_scope="present_good",
                max_spikes_per_unit=max_spikes_per_unit,
                waveform_seed=waveform_seed,
            )
            print(
                f"Session {session_id}: exported "
                f"{len(session_stats.unit_ids)} present good units"
            )

    print("Canonical positions and waveforms exported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mouse-id", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--sessions", nargs="+", type=int, required=True)
    parser.add_argument("--local-raw-base", type=Path, default=Path("/mnt/ssd4.1"))
    parser.add_argument("--recording-root", type=Path,
                        default=Path("/mnt/senzailab/Kai/#Recording"))
    args = parser.parse_args()

    cli_sessions: list[int] = args.sessions
    cli_data_base_folder = args.local_raw_base
    cli_session_name = _session_group_name(args.date, cli_sessions)
    cli_session_folder = cli_data_base_folder / "Data" / cli_session_name
    cli_probe_list = sorted(
        probe_json.parents[1].name.removeprefix("Probe")
        for probe_json in cli_session_folder.glob("Probe*/concat/probe.json")
        if (probe_json.parents[1] / "kilosort/spike_positions.npy").is_file()
    )
    generate_canonical_unit_artifacts(
        data_base_folder=cli_data_base_folder,
        remote_output=(
            args.recording_root
            / args.mouse_id
            / args.date
        ),
        date=args.date,
        sessions=cli_sessions,
        probe_list=cli_probe_list,
    )
