"""Render saved ADN centers and full native tuning curves at physical depth."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from HD.profiles import load_hd_tc
from RF.profiles import load_rf_centers
from RF.rflocate import load_rf, load_rfmap, rf_result_path
from RF.rflocate.rates import resolve_presentation_counts
from Utils.plotting import LIGHT_PLOT_STYLE
from Utils.retinotopography_scatter import prepare_depth_angles


SOURCE = Path("output/retinotopography/adn_depth_scatter")
OUTPUT = Path("output/session_comparisons_20261013/retinotopography_30bins")
IDENTITY = ["mouse", "date", "probe", "unit_id"]
SCATTER_DEPTH_STEP_UM = 25
RF_WINDOW_S = (0.0, 0.2)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def refresh_canonical_rf_points(source_dir, sources_path, selection_path):
    """Retain the selected HD/depth cohort and load RF centers from its session sidecars."""
    verification = json.loads((source_dir / "verification.json").read_text())
    assert verification["normalized"] is False
    assert verification["depth_units"] == "um"
    assert sha(selection_path) == verification["selection_sha256"]
    for path, expected in verification["hd_source_sha256"].items():
        assert sha(path) == expected, f"Saved HD input changed: {path}"
    units = pd.read_csv(source_dir / "adn_units.csv", dtype={"date": str}, float_precision="round_trip")
    original_hd = pd.read_csv(source_dir / "hd_depth_points.csv", dtype={"date": str}, float_precision="round_trip")
    sources = pd.read_csv(sources_path, dtype={"date": str})
    sources = sources.merge(units[["mouse", "date", "probe"]].drop_duplicates(),
                            on=["mouse", "date", "probe"], validate="one_to_one")
    center_tables, records = [], []
    for source in sources.itertuples():
        path = rf_result_path(source.rf_file)
        result = load_rf(path)
        assert result.manifest["center_algorithm"] == "component-response-weighted-medoid-v2"
        assert result.manifest["detector_algorithm"] == "pooled-spatial-z-v3-zero-filled"
        assert result.manifest["parameters"]["exclude_zero_bins"] is False
        assert result.manifest["parameters"]["alternative"] == "greater"
        assert result.manifest["parameters"]["cluster_forming_z"] == 1.8
        assert result.manifest["parameters"]["drop_bins"] == 2
        centers = load_rf_centers(source.rf_file, probe=source.probe).reset_index()
        center_tables.append(centers.assign(mouse=source.mouse, date=source.date))
        selected = units.mouse.eq(source.mouse) & units.date.eq(source.date) & units.probe.eq(source.probe)
        detected_ids = result.unit_ids[result.mask_2d.any(axis=(1, 2))]
        units.loc[selected, "rf_detected_2d"] = units.loc[selected, "unit_id"].isin(detected_ids)
        records.append(dict(mouse=source.mouse, date=source.date, probe=source.probe,
                            rf_session=int(source.rf_session), rf_source=source.rf_file,
                            rf_source_sha256=sha(source.rf_file), detection=str(path),
                            detection_sha256=sha(path), detection_manifest=dict(result.manifest)))
    centers = pd.concat(center_tables, ignore_index=True)
    points = prepare_depth_angles(units, centers, kind="rf")
    pd.testing.assert_frame_equal(prepare_depth_angles(units, centers, kind="hd"), original_hd,
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    points.to_csv(source_dir / "rf_depth_points.csv", index=False)
    units.to_csv(source_dir / "adn_units.csv", index=False)
    verification.update(rf_centers=len(points), rf_cells=len(points[IDENTITY].drop_duplicates()),
                        rf_source_policy="current canonical session RF sidecars; historical center caches are not inputs",
                        center_files={record["detection"]: record["detection_sha256"] for record in records},
                        rf_sources=records, rf_preparation_script_sha256=sha(__file__),
                        hd_points_sha256=sha(source_dir / "hd_depth_points.csv"))
    (source_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return verification



def load_sources(source_dir):
    """Read the saved cohort without changing selections or canonical results."""
    verification = json.loads((source_dir / "verification.json").read_text())
    assert verification["normalized"] is False
    assert verification["depth_units"] == "um"
    checked = {}
    selection = Path("config/retinotopography_adn_units.json")
    assert sha(selection) == verification["selection_sha256"]
    checked[str(selection)] = sha(selection)
    for group in ("center_files", "hd_source_sha256"):
        for path, expected in verification[group].items():
            checked[path] = sha(path)
            assert checked[path] == expected, f"Saved input changed: {path}"
    for record in verification["rf_sources"]:
        path = record["rf_source"]
        checked[path] = sha(path)
        assert checked[path] == record["rf_source_sha256"], f"Saved RF input changed: {path}"
    units = pd.read_csv(source_dir / "adn_units.csv", dtype={"date": str}, float_precision="round_trip")
    np.testing.assert_allclose(units.groupby(["mouse", "date", "probe", "shank"]).depth_um.min(), 0)
    points = {}
    for kind in ("hd", "rf"):
        path = source_dir / f"{kind}_depth_points.csv"
        checked[str(path)] = sha(path)
        table = pd.read_csv(path, dtype={"date": str}, float_precision="round_trip")
        joined = table.merge(units[IDENTITY + ["depth_um"]], on=IDENTITY,
                             validate="many_to_one", suffixes=("", "_reference"))
        assert len(joined) == len(table)
        np.testing.assert_allclose(joined.depth_um, joined.depth_um_reference)
        table["angle_deg"] = table.angle_deg % 360
        points[kind] = table
    assert len(points["hd"]) == verification["hd_cells"]
    assert len(points["rf"]) == verification["rf_centers"]
    checked[str(source_dir / "adn_units.csv")] = sha(source_dir / "adn_units.csv")
    checked[str(source_dir / "verification.json")] = sha(source_dir / "verification.json")
    return verification, units, points, checked


def load_native_curves(verification, units, rf_points):
    """Read HD Hz and calculate RF Hz from raw counts and presentation exposure."""
    curves = {"hd": [], "rf": []}
    provenance = []
    rf_identities = rf_points[IDENTITY].drop_duplicates()
    for source in verification["rf_sources"]:
        identity = {name: source[name] for name in ("mouse", "date", "probe")}
        selected = units.loc[
            units.mouse.eq(source["mouse"]) & units.date.eq(source["date"])
            & units.probe.eq(source["probe"])
        ].copy()
        hd_path = next(path for path in verification["hd_source_sha256"]
                       if f"/{source['mouse']}/{source['date']}/" in path
                       and f"/Probe{source['probe']}/" in path)
        hd = load_hd_tc(hd_path)
        hd_ids = pd.Index(hd["unit_id"])
        rows = hd_ids.get_indexer(selected.unit_id)
        assert np.all(rows >= 0)
        assert np.all(np.asarray(hd["unit_data"]["hd_class"])[rows] == 3)
        hd_counts = np.asarray(hd["spike_counts"], dtype=float)[rows]
        occupancy = np.asarray(hd["occupancy_time_s"], dtype=float)
        hd_rates = np.asarray(hd["firing_rate_hz"], dtype=float)[rows]
        calculated = np.divide(hd_counts, occupancy, out=np.full_like(hd_counts, np.nan),
                               where=occupancy > 0)
        np.testing.assert_allclose(hd_rates, calculated, equal_nan=True)
        edges = np.asarray(hd["angle_bin_edges_deg"], dtype=float)
        hd_angles = (edges[:-1] + edges[1:]) / 2
        curves["hd"].append(dict(identity=identity, units=selected, angles=hd_angles,
                                  rates=hd_rates, counts=hd_counts, exposure=occupancy,
                                  native_edges=edges, source=hd_path))

        selected_rf = selected.merge(rf_identities, on=IDENTITY, validate="one_to_one")
        maps = load_rfmap(source["rf_source"])
        first = maps[0]
        assert first.metadata["responseUnits"] == "spike_count"
        presentations = first.presentation_counts
        if presentations is None:
            presentations, exposure_source = resolve_presentation_counts(
                first.metadata, first.source_path, first.x_positions, first.y_positions)
        else:
            exposure_source = {"method": "stimulusPresentationCounts"}
        map_indices = pd.Index(maps.unit_ids).get_indexer(selected_rf.unit_id)
        assert np.all(map_indices >= 0)
        pooled_counts = []
        for index in map_indices:
            window = maps[index].sum(*RF_WINDOW_S)
            duration = float(np.diff(window.time_bin_edges_s)[0])
            np.testing.assert_allclose(duration, RF_WINDOW_S[1] - RF_WINDOW_S[0])
            pooled_counts.append(window.to_2d_array().sum(axis=0))
        pooled_counts = np.asarray(pooled_counts, dtype=float)
        # A horizontal-angle rate pools spikes AND exposure across elevations.
        # Summing already-normalized Hz across elevations would inflate the FR.
        exposure = presentations.sum(axis=0) * duration
        rf_rates = np.divide(pooled_counts, exposure, out=np.full_like(pooled_counts, np.nan),
                             where=exposure > 0)
        curves["rf"].append(dict(identity=identity, units=selected_rf,
                                  angles=first.x_positions.copy(), rates=rf_rates,
                                  counts=pooled_counts, exposure=exposure,
                                  native_edges=None, source=source["rf_source"]))
        provenance.append(dict(**identity, hd_source=hd_path, rf_source=source["rf_source"],
                               hd_cells=len(selected), rf_cells=len(selected_rf),
                               hd_angle_bins=len(hd_angles), rf_angle_bins=first.n_x,
                               rf_time_range_s=list(RF_WINDOW_S),
                               rf_rate_formula="sum_y(spikes in window) / (sum_y(presentations) * window_seconds)",
                               rf_presentation_source=exposure_source,
                               rf_exposure_source_sha256={key: sha(exposure_source[key]) for key in ("trials_mat", "onsets_npy") if key in exposure_source},
                               rf_presentations=presentations.tolist(),
                               hd_rate_formula="saved spike counts / saved occupancy seconds"))
        print(f"{source['mouse']} Probe{source['probe']}: HD {len(selected)}, RF {len(selected_rf)}", flush=True)
    return curves, provenance


def save_native_curves(curves, output):
    """Save native per-neuron values before angular display or depth aggregation."""
    for kind, sources in curves.items():
        for source in sources:
            identity = source["identity"]
            stem = f"{identity['mouse']}_{identity['date']}_Probe{identity['probe']}_{kind}"
            units = source["units"].sort_values(["depth_um", "probe", "shank", "unit_id"], kind="stable")
            order = source["units"].index.get_indexer(units.index)
            units[IDENTITY + ["shank", "depth_um"]].to_csv(output / f"{stem}_units.csv", index=False)
            np.savez_compressed(output / f"{stem}_native_profiles.npz",
                                unit_ids=units.unit_id.to_numpy(), probe=units.probe.to_numpy(dtype=str),
                                shank=units.shank.to_numpy(), depth_um=units.depth_um.to_numpy(),
                                native_angles_deg=source["angles"],
                                firing_rate_hz=source["rates"][order],
                                spike_counts=source["counts"][order], exposure_s=source["exposure"])


def rebin_counts_exposure(counts, exposure, native_angles, angle_edges):
    """Pool discrete native angle samples into half-open bins before forming Hz."""
    target = np.searchsorted(angle_edges, native_angles % 360, side="right") - 1
    n_bins = len(angle_edges) - 1
    binned_counts = np.zeros((len(counts), n_bins))
    binned_exposure = np.zeros(n_bins)
    native_bin_count = np.zeros(n_bins, dtype=int)
    for index in range(n_bins):
        selected = target == index
        binned_counts[:, index] = counts[:, selected].sum(axis=1)
        binned_exposure[index] = exposure[selected].sum()
        native_bin_count[index] = selected.sum()
    np.testing.assert_allclose(binned_counts.sum(axis=1), counts.sum(axis=1), rtol=1e-12)
    np.testing.assert_allclose(binned_exposure.sum(), exposure.sum(), rtol=1e-12)
    return binned_counts, binned_exposure, native_bin_count


def unit_rate_heatmap(sources, mouse):
    """Rebin counts/exposure to 30 angles, keeping separate depth-ordered units."""
    selected = [source for source in sources if source["identity"]["mouse"] == mouse]
    angle_edges = np.linspace(0, 360, 31)
    counts, exposures, members = [], [], []
    for source in selected:
        binned_counts, binned_exposure, native_bin_count = rebin_counts_exposure(
            source["counts"], source["exposure"], source["angles"], angle_edges)
        counts.append(binned_counts)
        exposures.append(np.broadcast_to(binned_exposure, binned_counts.shape))
        members.append(np.broadcast_to(native_bin_count, binned_counts.shape))
    rows = pd.concat([source["units"][IDENTITY + ["shank", "depth_um"]]
                      for source in selected], ignore_index=True)
    ordered_rows = rows.sort_values(["depth_um", "probe", "shank", "unit_id"], kind="stable")
    binned_counts = np.concatenate(counts, axis=0)[ordered_rows.index]
    binned_exposure = np.concatenate(exposures, axis=0)[ordered_rows.index]
    native_bin_count = np.concatenate(members, axis=0)[ordered_rows.index]
    rates = np.divide(binned_counts, binned_exposure,
                      out=np.full_like(binned_counts, np.nan), where=binned_exposure > 0)
    ordered_rows = ordered_rows.reset_index(drop=True)
    ordered_rows.insert(0, "row", np.arange(1, len(rows) + 1))
    assert not ordered_rows.duplicated(IDENTITY).any()
    assert np.all(np.isfinite(rates).any(axis=1))
    return rates, ordered_rows, angle_edges, binned_counts, binned_exposure, native_bin_count


def normalize_unit_max(rates):
    """Scale display rows from zero to each unit's finite maximum, preserving gaps."""
    maximum = np.nanmax(rates, axis=1, keepdims=True)
    normalized = np.where(np.isfinite(rates), 0.0, np.nan)
    return np.divide(rates, maximum, out=normalized, where=maximum > 0)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    verification, units, points, checked = load_sources(SOURCE)
    curves, provenance = load_native_curves(verification, units, points["rf"])
    save_native_curves(curves, OUTPUT)
    units.to_csv(OUTPUT / "adn_units.csv", index=False)
    for kind, table in points.items():
        table.to_csv(OUTPUT / f"{kind}_depth_points.csv", index=False)
    plt.rcParams.update(LIGHT_PLOT_STYLE)
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16})
    max_depth = (np.floor(units.depth_um.max() / SCATTER_DEPTH_STEP_UM) + 1) * SCATTER_DEPTH_STEP_UM
    reports = []
    for mouse in verification["mice"]:
        dates = units.loc[units.mouse.eq(mouse), "date"].unique()
        assert len(dates) == 1
        record = dict(mouse=mouse, date=dates[0], plots={})
        for kind in ("hd", "rf"):
            table = points[kind].loc[points[kind].mouse.eq(mouse)]
            rates, rows, angle_edges, binned_counts, binned_exposure, native_bin_count = unit_rate_heatmap(curves[kind], mouse)
            normalized = normalize_unit_max(rates)
            n_units = len(table[IDENTITY].drop_duplicates())
            assert len(rows) == n_units
            record[kind] = dict(points=len(table), cells=n_units, heatmap_rows=len(rows),
                                angle_bins=len(angle_edges) - 1,
                                native_observed_angle_bins=len(next(source for source in curves[kind]
                                    if source["identity"]["mouse"] == mouse)["angles"]),
                                finite_unit_angle_bins=int(np.isfinite(rates).sum()),
                                fr_min_hz=float(np.nanmin(rates)), fr_max_hz=float(np.nanmax(rates)))
            rows.to_csv(OUTPUT / f"{mouse}_{kind}_unit_rows.csv", index=False)
            np.savez_compressed(OUTPUT / f"{mouse}_{kind}_unit_rates.npz",
                                firing_rate_hz=rates, normalized_response=normalized,
                                spike_counts=binned_counts, exposure_s=binned_exposure,
                                native_bin_count=native_bin_count,
                                row_max_hz=np.nanmax(rates, axis=1), unit_ids=rows.unit_id.to_numpy(),
                                probe=rows.probe.to_numpy(dtype=str), shank=rows.shank.to_numpy(),
                                depth_um=rows.depth_um.to_numpy(), angle_edges_deg=angle_edges)
            for mode in ("heatmap", "scatter"):
                fig, ax = plt.subplots(figsize=(7, 6.8), layout="constrained", facecolor="white")
                if mode == "heatmap":
                    cmap = plt.get_cmap("viridis").copy()
                    cmap.set_bad("white")
                    image = ax.pcolormesh(angle_edges, np.arange(len(rows) + 1), normalized, cmap=cmap,
                                          vmin=0, vmax=1, shading="flat")
                    fig.colorbar(image, ax=ax, label="Normalized FR", ticks=[0, .5, 1], pad=.03)
                    ticks = np.unique(np.linspace(0, len(rows) - 1, min(6, len(rows)), dtype=int))
                    ax.set(ylim=(len(rows), 0), yticks=ticks + .5,
                           yticklabels=[f"{rows.depth_um.iloc[index]:.1f}" for index in ticks],
                           ylabel="Depth (µm; ordered units)")
                else:
                    scatter = ax.scatter(table.angle_deg, table.depth_um, s=28,
                                         color="#2468a2", alpha=.8, linewidths=0, clip_on=False)
                    np.testing.assert_array_equal(scatter.get_offsets(), table[["angle_deg", "depth_um"]])
                    ax.grid(alpha=.22, linewidth=.6)
                    ax.set_axisbelow(True)
                    ax.set(ylim=(max_depth, 0), ylabel="Depth (µm)")
                ax.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360],
                       xlabel="HD (°)" if kind == "hd" else ("RF center (°)" if mode == "scatter" else "RF (°)"))
                ax.tick_params(axis="y", labelleft=True)
                ax.set_facecolor("white")
                filename = OUTPUT / f"{mouse}_{kind}_{mode}.png"
                fig.savefig(filename, dpi=160, facecolor="white", transparent=False)
                plt.close(fig)
                record["plots"][f"{kind}_{mode}"] = str(filename)
        reports.append(record)
    result = dict(status="passed", sessions=reports, hd_cells=len(points["hd"]),
                  rf_centers=len(points["rf"]), rf_cells=verification["rf_cells"],
                  normalized=True, raw_rates_normalized=False, depth_normalized=False,
                  display_normalization="FR / finite per-unit maximum; no minimum subtraction; positive constant rows 1, all-zero rows 0, missing angles NaN",
                  normalized_array="normalized_response", raw_rate_array="firing_rate_hz",
                  display_angle_range_deg=[0, 360],
                  angular_binning="30 half-open 12-degree bins [0,12), ..., [348,360); native centers modulo 360 assign discrete samples",
                  angle_bin_count=30, angle_edges_deg=np.linspace(0, 360, 31).tolist(),
                  angle_aggregation="sum native spike counts and exposure, then compute Hz; no averaging of Hz or fractional interpolation",
                  normalization_order="rebin counts and exposure, calculate Hz, then divide each row by its finite maximum",
                  depth_aggregation="none; one unit per uniformly spaced row, including equal-depth units",
                  row_sort=["depth_um", "probe", "shank", "unit_id"],
                  heatmap_y_axis="uniform unit rows; tick labels show actual depth in um, not metric spacing",
                  depth_origin="topmost selected HD cell per mouse/date/probe/shank",
                  positive_depth_direction="toward tip", color_units="fraction of per-unit maximum",
                  color_range=[0, 1], color_ticks=[0, .5, 1], raw_rate_units="Hz",
                  absence="white NaN angle bins indicate unobserved angles, not zero firing; there are no empty depth rows",
                  rf_native_unobserved_angles={"m14": [[150, 210]]},
                  rf_empty_display_bins={"m14": [[156, 168], [168, 180], [180, 192], [192, 204]]},
                  rf_partial_display_bins={"m14": {"144-156": [145], "204-216": [215]}},
                  rf_coverage_note="m14 display bins pool discrete measured centers; edge bins do not imply sampling throughout their full angular extent",
                  hd_cohort="saved selected ADN class-3 cells",
                  rf_cohort="saved selected ADN class-3 cells with canonical detected RF centers",
                  source_sha256=checked, source_provenance=provenance,
                  script_sha256=sha(__file__), source_verification=str(SOURCE / "verification.json"))
    (OUTPUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "hd_cells", "rf_centers", "rf_cells")}), flush=True)


if __name__ == "__main__":
    main()
