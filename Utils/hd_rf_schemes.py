"""Extract native 2-D RF peaks and three explicit HD/RF selection schemes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.direction_comparison import hd_pick, load_tc
from Utils.hd_rf_prediction import wrap_deg
from Utils.json_tools import read_formatted_json
from Utils.rfmap import load_rf_maps


def load_peak_pairs(hd_path, rf_path, *, hd_is_clockwise=True,
                    mouse="m19", date=260827, probe="A"):
    """Keep every existing Class 3 unit and audit undefined HD/RF peaks.

    RF peak is the native full-map maximum of the presentation-exposure rate
    averaged over 0–200 ms. Saved detection masks affect membership only.
    Ties use the first native (y, x) position in row-major source order. For
    documented legacy counter-clockwise HD, negate only the extracted peak.
    """
    hd_path, rf_path = Path(hd_path), Path(rf_path)
    recording = dict(mouse=mouse, date=date, probe=probe)
    cohort = hd_pick(load_tc(hd_path, **recording, bins=30, smoothing_deg=0), hd_class=3)
    unit_ids = cohort.index.get_level_values("unit_id").to_numpy(dtype=int)
    hd_profiles = cohort.copy()
    hd_profiles.index = unit_ids
    heading_provenance = dict(
        hd_profile_method="Existing saved TC counts / occupancy; 30 bins; no smoothing",
        classification_source="Cohort selected by existing update_hd_classification on saved TC; no new classification or shuffles",
    )
    maps = load_rf_maps(rf_path, unit_firing_rate=True).sum(0., .2, show_progress=False)
    maps_by_id = {int(item.unit_id): item for item in maps}
    detection_path = rf_path.with_suffix(".npz")
    with np.load(detection_path, allow_pickle=False) as saved:
        masks = {int(unit): mask.astype(bool) for unit, mask in
                 zip(saved["unit_ids"], saved["mask_2d"], strict=True)}
    rows = []
    for unit in unit_ids:
        row = dict(mouse=mouse, date=date, probe=probe, unit_id=int(unit), hd_class=3,
                   hd_native_preferred_deg=np.nan, hd_preferred_deg=np.nan,
                   hd_peak_hz=np.nan, hd_peak_ties=0,
                   rf_map_present=unit in maps_by_id, rf_detection_record_present=unit in masks,
                   rf_saved_2d=unit in masks and bool(masks[unit].any()),
                   rf_ego_deg=np.nan, rf_peak_x=np.nan, rf_peak_y=np.nan,
                   rf_peak_hz=np.nan, rf_peak_ties=0, rf_peak_x_index=-1, rf_peak_y_index=-1,
                   rf_peak_in_saved_2d_mask=False, rf_finite_bins=0, rf_zero_bins=0,
                   hd_plus_rf_deg=np.nan, peak_valid=False)
        reasons = []
        hd = hd_profiles.loc[unit].to_numpy(dtype=float)
        if not np.isfinite(hd).any() or np.nanmax(hd) <= 0:
            reasons.append("no_positive_finite_hd_response")
        else:
            maximum = np.nanmax(hd)
            peak = int(np.flatnonzero(hd == maximum)[0])
            native_angle = float(hd_profiles.columns[peak] % 360.)
            row.update(hd_native_preferred_deg=native_angle,
                       hd_preferred_deg=native_angle if hd_is_clockwise else (-native_angle) % 360.,
                       hd_peak_hz=float(maximum), hd_peak_ties=int(np.sum(hd == maximum)))
        if unit not in maps_by_id:
            reasons.append("missing_rf_map")
        else:
            rf_map = maps_by_id[unit]
            matrix = rf_map.to_2d_array()
            valid = np.isfinite(matrix)
            row.update(rf_finite_bins=int(valid.sum()),
                       rf_zero_bins=int(np.sum(matrix[valid] == 0)))
            if not valid.any() or np.max(matrix[valid]) <= 0:
                reasons.append("no_positive_finite_rf_response")
            else:
                maximum = np.max(matrix[valid])
                peaks = np.argwhere(matrix == maximum)
                y, x = peaks[0]
                row.update(rf_ego_deg=float(wrap_deg(rf_map.x_positions[x])),
                           rf_peak_x=float(rf_map.x_positions[x]), rf_peak_y=float(rf_map.y_positions[y]),
                           rf_peak_hz=float(maximum), rf_peak_ties=len(peaks),
                           rf_peak_x_index=int(x), rf_peak_y_index=int(y),
                           rf_peak_in_saved_2d_mask=unit in masks and bool(masks[unit][y, x]))
        row["peak_valid"] = not reasons
        row["exclusion_reason"] = ";".join(reasons)
        if row["peak_valid"]:
            row["hd_plus_rf_deg"] = (row["hd_preferred_deg"] + row["rf_ego_deg"]) % 360.
        rows.append(row)
    pairs = pd.DataFrame(rows)
    provenance = dict(
        recording=recording, hd_source=str(hd_path), rf_source=str(rf_path),
        rf_detection_source=str(detection_path), selected_units=unit_ids.tolist(),
        hd_class=3, hd_bins=30, hd_smoothing_deg=0, rf_window_s=[0., .2],
        rf_peak_method="global maximum of native full 2-D response-window Hz map; no spatial projection, interpolation, smoothing or localization-mask restriction",
        rf_response_normalization="load_rf_maps(unit_firing_rate=True).sum(0, 0.2).to_2d_array() per unit",
        rf_detection_method="read saved excitatory mask_2d; any true bin marks a detected RF; no detection recomputation",
        peak_tie_rule="first native source-order HD bin; first row-major (y, x) RF bin; all tie counts reported",
        quality_filters="None: zero bin counts are audit columns only; exclude only missing RF maps or undefined positive finite HD/RF peaks",
        hd_is_clockwise=bool(hd_is_clockwise), hd_angle_sign=1 if hd_is_clockwise else -1,
        hd_angle_conversion="native peak modulo 360" if hd_is_clockwise else "(-native TC peak) modulo 360; profile/classification unchanged",
        hd_source_angle_note=read_formatted_json(hd_path).get("metadata", {}).get("angle_convention_note"),
        coordinate_note="Analysis HD and native RF horizontal angles positive clockwise; HD_plus_RF=wrap360(HD+RFego). Source HD peak and explicit sign conversion are retained.",
        hd_class3_units=len(unit_ids), finite_pairs=int(pairs.peak_valid.sum()),
        sources_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (hd_path, rf_path, detection_path)},
    )
    provenance.update(heading_provenance)
    return pairs, provenance


def select_schemes(all_pairs, *, n_bins=30, top_n_bins=3):
    """Select all valid Class 3 peaks, saved 2-D RFs, then top HD+RF bins.

    Top bins are ranked independently, not selected as a contiguous window.
    Equal counts are resolved by lower bin index, including at the cutoff.
    Scheme 3 selects on the observed outcome and is descriptive selection.
    """
    if not 1 <= top_n_bins <= n_bins:
        raise ValueError("top_n_bins must lie between one and n_bins")
    pairs = all_pairs.copy()
    edges = np.linspace(0., 360., n_bins + 1)
    valid = pairs.peak_valid.astype(bool)
    pairs["hd_plus_rf_bin"] = -1
    pairs.loc[valid, "hd_plus_rf_bin"] = np.searchsorted(
        edges, pairs.loc[valid, "hd_plus_rf_deg"].to_numpy() % 360., side="right",
    ) - 1
    first = pairs.loc[valid].copy()
    second = first.loc[first.rf_saved_2d.astype(bool)].copy()
    counts = np.bincount(second.hd_plus_rf_bin, minlength=n_bins)
    ranked = np.lexsort((np.arange(n_bins), -counts))
    chosen = ranked[:top_n_bins]
    cutoff_count = int(counts[chosen[-1]])
    tied = np.flatnonzero(counts == cutoff_count)
    third = second.loc[second.hd_plus_rf_bin.isin(chosen)].copy()
    schemes = {"scheme1": first, "scheme2": second, "scheme3": third}
    manifest = dict(
        definitions={"scheme1": "All existing HD Class 3 units with defined HD and native 2-D RF maximum",
                     "scheme2": "Scheme 1 AND any saved excitatory 2-D RF mask bin",
                     "scheme3": f"Scheme 2 AND HD+RF lies in one of the {top_n_bins} largest independent histogram bins"},
        n_bins=n_bins, bin_edges_deg=edges.tolist(), bin_counts=counts.tolist(),
        bin_interval_rule="[lower, upper); wrapped 360 degrees belongs to bin 0",
        top_n_bins=top_n_bins, ranked_bin_indices=ranked.tolist(), selected_bin_indices=chosen.tolist(),
        selected_bin_intervals_deg=[[float(edges[i]), float(edges[i + 1])] for i in chosen],
        bin_tie_rule="descending count, then ascending bin index; no adjacency constraint",
        cutoff_count=cutoff_count, cutoff_tied_bin_indices=tied.tolist(),
        cutoff_tie_crosses_selection=bool(np.any(np.isin(tied, chosen)) and np.any(~np.isin(tied, chosen))),
        cohort_count=len(pairs), counts={key: len(value) for key, value in schemes.items()},
        selected_units={key: value.unit_id.tolist() for key, value in schemes.items()},
        missing=pairs.loc[~valid, ["unit_id", "exclusion_reason"]].to_dict("records"),
        rf_detection_record_missing_units=pairs.loc[~pairs.rf_detection_record_present, "unit_id"].tolist(),
        interpretation="Scheme 3 is selected using observed HD+RF and cannot independently establish an anticorrelation.",
    )
    return schemes, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hd-file", type=Path, required=True)
    parser.add_argument("--rf-file", type=Path, required=True)
    parser.add_argument("--hd-counter-clockwise", action="store_true",
                        help="Negate extracted HD peaks from a documented CCW source, without changing profiles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    pairs, provenance = load_peak_pairs(
        args.hd_file, args.rf_file, hd_is_clockwise=not args.hd_counter_clockwise,
    )
    schemes, manifest = select_schemes(pairs)
    for key, table in schemes.items():
        pairs[key] = pairs.unit_id.isin(table.unit_id)
        table.to_csv(args.output / f"{key}.csv", index=False)
    pairs.to_csv(args.output / "all_pairs.csv", index=False)
    for name, data in (("selection_manifest", manifest), ("provenance", provenance)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
