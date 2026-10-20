"""Export original Hz ranges for the saved inhibitory comparison cohort.

This reads saved sources and writes a separate presentation manifest. It does
not change classifications, peaks, source curves, or their display scaling.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from RF.rflocate import RFMapList, load_rf, load_rfmap
from RF.rflocate.rates import resolve_presentation_counts
from RF.tc_generation import calculate_rf_projection
from Utils.direction_comparison import minmax_profiles, profile_table, tc_loader


SESSIONS = ("m14_260609", "m15_260630", "m19_260827", "m20_260921", "m21_261006", "m22_261008")


def file_record(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def hd_rates(data, prefix, bins):
    """Rebin saved raw spike counts and occupancy to the displayed native grid."""
    occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
    counts = np.asarray(data["spike_counts"], dtype=float)
    factor = len(occupancy) // bins
    occupancy = occupancy.reshape(bins, factor).sum(axis=1)
    counts = counts.reshape(len(data["unit_id"]), bins, factor).sum(axis=2)
    rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan), where=occupancy > 0)
    edges = np.asarray(data["angle_bin_edges_deg"])[::factor]
    return profile_table(rates, [f"{prefix}:{unit}" for unit in data["unit_id"]],
                         (edges[:-1] + edges[1:]) / 2)


def range_record(table):
    values = table.to_numpy()
    return dict(minimum_hz=float(np.nanmin(values)), maximum_hz=float(np.nanmax(values)),
                unit_count=len(table), angular_bins=len(table.columns))


def export_ranges(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    input_manifest = json.loads((source / "manifest.json").read_text())
    pairs = {item["key"]: item for item in input_manifest["input_pairs"]}
    sources = {}
    sessions = []
    all_unit_rows = []

    def remember(path):
        sources[str(path)] = file_record(path)

    remember(source / "manifest.json")
    for key in SESSIONS:
        directory = source / key
        summary_path = directory / "summary.json"
        peaks_path = directory / "paired_native_directions.csv"
        summary = json.loads(summary_path.read_text())
        selected = pd.read_csv(peaks_path, index_col=0).index
        remember(summary_path)
        remember(peaks_path)
        hd_parts, rf_parts, rows, probe_records = [], [], [], []
        for pair_key in summary["input_pairs"]:
            pair = pairs[pair_key]
            prefix = f"{pair['mouse']}:{pair['date']}:{pair['probe']}"
            units = selected[selected.str.startswith(prefix + ":")]
            hd_path, rf_path, detection_path = [Path(item["path"]) for item in pair["sources"]]
            hd_csv = source / "inputs" / pair_key / "hd_class3.csv"
            rf_csv = source / "inputs" / pair_key / "rf_inhibitory.csv"
            for path in (hd_path, rf_path, detection_path, hd_csv, rf_csv):
                remember(path)
            saved_hd = tc_loader(hd_csv).loc[units]
            saved_rf = tc_loader(rf_csv).loc[units]
            hd_data = json.loads(hd_path.read_text())
            hd = hd_rates(hd_data, prefix, len(saved_hd.columns)).loc[units]
            np.testing.assert_allclose(hd.to_numpy(), saved_hd.to_numpy(), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(hd.columns, saved_hd.columns)
            hd_classes = dict(zip(hd_data["unit_id"], hd_data["unit_data"]["hd_class"], strict=True))
            assert all(hd_classes[int(unit.rsplit(":", 1)[1])] == 3 for unit in units)

            maps = load_rfmap(rf_path)
            detected = load_rf(detection_path)
            presentations, provenance = resolve_presentation_counts(
                maps[0].metadata, rf_path, maps[0].x_positions, maps[0].y_positions)
            for name in ("trials_mat", "onsets_npy"):
                if name in provenance:
                    remember(Path(provenance[name]))
            by_id = {item.unit_id: item for item in maps}
            selected_maps = RFMapList([by_id[int(unit.rsplit(":", 1)[1])] for unit in units], rf_path)
            projected = calculate_rf_projection(maps, detected, time_range_s=(0.0, 0.2), rf_only=True)
            projected.index = pd.Index([f"{prefix}:{unit}" for unit in projected.index], name="unit_id")
            projected = projected.loc[units, saved_rf.columns]
            np.testing.assert_allclose(projected.to_numpy(), saved_rf.to_numpy(), rtol=0, atol=0, equal_nan=True)
            window = selected_maps[0].sum(0.0, 0.2)
            duration = float(np.diff(window.time_bin_edges_s)[0])
            mask_rows = {int(unit): mask for unit, mask in zip(detected.unit_ids, detected.mask_2d, strict=True)}
            rates, exposures, selected_rows = [], [], []
            for unit in units:
                selected_y = mask_rows[int(unit.rsplit(":", 1)[1])].any(axis=1)
                # The TC sums whole elevation rows, so its exposure must sum
                # those same rows. Display duration is not this denominator.
                exposure = presentations[selected_y].sum(axis=0) * duration
                count = saved_rf.loc[unit].to_numpy()
                rate = np.divide(count, exposure, out=np.full_like(count, np.nan), where=exposure > 0)
                rates.append(rate)
                exposures.append(exposure)
                selected_rows.append(int(selected_y.sum()))
            rf = pd.DataFrame(rates, index=units, columns=saved_rf.columns)
            exposure_array = np.asarray(exposures)
            constant_exposure = np.all(np.isclose(exposure_array, exposure_array[:, :1], rtol=1e-12, atol=1e-12), axis=1)
            normalized_equal = np.all(np.isclose(minmax_profiles(rf).to_numpy(),
                                                minmax_profiles(saved_rf).to_numpy(),
                                                rtol=1e-12, atol=1e-12, equal_nan=True), axis=1)
            for i, unit in enumerate(units):
                rows.append(dict(mouse=pair["mouse"], date=pair["date"], probe=pair["probe"], unit_id=unit,
                                 hd_minimum_hz=float(hd.loc[unit].min()), hd_maximum_hz=float(hd.loc[unit].max()),
                                 rf_minimum_hz=float(rf.loc[unit].min()), rf_maximum_hz=float(rf.loc[unit].max()),
                                 rf_selected_elevation_rows=selected_rows[i],
                                 rf_exposure_minimum_s=float(exposure_array[i].min()),
                                 rf_exposure_maximum_s=float(exposure_array[i].max()),
                                 rf_exposure_constant_across_angle=bool(constant_exposure[i]),
                                 rf_minmax_matches_existing_panel=bool(normalized_equal[i])))
            hd_parts.append(hd)
            rf_parts.append(rf)
            probe_records.append(dict(pair_key=pair_key, unit_count=len(units), window_duration_s=duration,
                                      presentation_count_source=provenance,
                                      presentation_count_minimum=float(presentations.min()),
                                      presentation_count_maximum=float(presentations.max()),
                                      exposure_constant_across_angle=bool(constant_exposure.all()),
                                      minmax_matches_existing_panel=bool(normalized_equal.all())))
        hd = pd.concat(hd_parts).loc[selected]
        rf = pd.concat(rf_parts).loc[selected]
        hd.to_csv(destination / f"{key}_hd_native_hz.csv")
        rf.to_csv(destination / f"{key}_rf_native_hz.csv")
        pd.DataFrame(rows).to_csv(destination / f"{key}_unit_ranges.csv", index=False)
        mouse, date = key.split("_")
        sessions.append(dict(mouse=mouse, date=date, paired_units=len(selected), hd=range_record(hd),
                             rf=range_record(rf), probes=probe_records,
                             per_unit_ranges=f"{key}_unit_ranges.csv"))
        all_unit_rows.extend(rows)
        print(key, "HD", range_record(hd), "RF", range_record(rf), flush=True)
    assert sources == {path: file_record(path) for path in sources}
    pd.DataFrame(all_unit_rows).to_csv(destination / "unit_ranges.csv", index=False)
    manifest = dict(sessions=sessions, sources=list(sources.values()), sources_unchanged=True,
                    script=file_record(__file__),
                    scope="Original native rates for the existing selected comparison units; no scientific inputs changed",
                    rf_rate_definition="Projected counts / (native window duration * presentations summed over the same selected whole elevation rows)",
                    hd_rate_definition="Raw spike counts / occupancy, rebinned to the same native30 angle bins",
                    annotation_definition="Overall finite native FR minimum and maximum across all displayed units and angles; same range for reordering/alignment views",
                    minmax_display_unchanged=all(row["rf_minmax_matches_existing_panel"] for row in all_unit_rows))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/session_comparisons_20261012/inhibitory"))
    parser.add_argument("--output", type=Path, default=Path("output/session_comparisons_20261013/inhibitory_fr_ranges"))
    args = parser.parse_args()
    export_ranges(args.source, args.output)
