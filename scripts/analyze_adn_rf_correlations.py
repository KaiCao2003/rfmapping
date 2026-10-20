"""Analyze RF-region counts within the user-marked ADN cohort.

Run on the recording host with ``python -m scripts.analyze_adn_rf_correlations``.
Source masks stay fixed; corrected results are saved
under the output directory, leaving the original recording sidecars intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from RF.rflocate import RFMapList, RFResult, load_rf, load_rfmap, save_rf
from RF.rflocate._detector import _label_clusters
from RF.rflocate.detection import _center_only_mask


RECORDINGS = [("m14", "260609", 1, 3, "A"),
              ("m15", "260630", 1, 3, "A"),
              ("m15", "260630", 1, 3, "B"),
              ("m19", "260827", 9, 2, "A"),
              ("m20", "260921", 9, 2, "A"),
              ("m21", "261006", 1, 3, "A")]
IDENTITY = ["mouse", "date", "probe", "unit_id"]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(root: Path, selection: Path, output: Path, m21_hd_session: int = 1):
    selected = {(row["mouse"], row["date"], row["probe"]): set(row["unit_ids"])
                for row in json.loads(selection.read_text())["recordings"]}
    unit_rows, center_rows, sources, examples = [], [], [], []
    (output / "centers").mkdir(exist_ok=True)
    for mouse, date, hd_session, rf_session, probe in RECORDINGS:
        if mouse == "m21":
            hd_session = m21_hd_session
        hd_path = (root / mouse / date / f"{date}_{hd_session}" /
                   "data/tuning_curves" / f"Probe{probe}" / "tuning_curves.tc")
        rf_path = (root / mouse / date / f"{date}_{rf_session}" /
                   "data/rfmapping/good/-100_400_1ms" / f"Probe{probe}" /
                   f"regular_unitsSpikeCounts_{date}_{rf_session}.rfmap")
        old_path = rf_path.with_suffix(".npz")
        before = {str(path): sha256(path) for path in (hd_path, rf_path, old_path)}
        hd = json.loads(hd_path.read_text())
        edges = np.asarray(hd["angle_bin_edges_deg"])
        angles = (edges[:-1] + edges[1:]) / 2
        counts = np.asarray(hd["spike_counts"], dtype=float)
        occupancy = np.asarray(hd["occupancy_time_s"], dtype=float)
        rates = np.divide(counts, occupancy, out=np.full_like(counts, np.nan),
                          where=occupancy > 0)
        np.testing.assert_allclose(rates, hd["firing_rate_hz"], equal_nan=True)
        hd_by_id = {int(uid): (int(cls), float(angles[np.nanargmax(curve)]))
                    for uid, cls, curve in zip(hd["unit_id"], hd["unit_data"]["hd_class"],
                                               rates, strict=True)
                    if np.isfinite(curve).any() and np.nanmax(curve) > 0}
        old = load_rf(old_path)
        # The original sidecars used summed native responses over this window.
        maps = load_rfmap(rf_path).sum(*old.manifest["time_range_s"], show_progress=False)
        by_id = {item.unit_id: item for item in maps}
        maps = RFMapList([by_id[int(uid)] for uid in old.unit_ids], rf_path)
        pooled = maps.to_2d_array().astype(float)
        pooled[np.isnan(pooled)] = 0
        valid = np.isfinite(pooled)
        if old.manifest["parameters"].get("exclude_zero_bins", False):
            valid &= pooled != 0
        null_mean = np.nanmean(np.where(valid, pooled, np.nan), axis=(1, 2), keepdims=True)
        centers = _center_only_mask(
            dict(final_mask=old.mask_2d, response_map=pooled,
                 null_mean_map=np.broadcast_to(null_mean, pooled.shape)),
            alternative=old.manifest["parameters"]["alternative"],
            wrap_x=old.manifest["parameters"]["wrap_x"], show_progress=False)
        manifest = dict(old.manifest, center_algorithm="component-response-weighted-medoid-v2",
                        center_recomputed_from_saved_mask_sha256=before[str(old_path)])
        key = hashlib.sha256((old.cache_key + json.dumps(manifest, sort_keys=True)
                              + before[str(rf_path)]).encode()).hexdigest()
        fixed = RFResult(mask_2d=old.mask_2d, center_2d=centers, unit_ids=old.unit_ids,
                         manifest=manifest, cache_key=key)
        np.testing.assert_array_equal(fixed.mask_2d, old.mask_2d)
        np.testing.assert_array_equal(fixed.unit_ids, old.unit_ids)
        fixed_path = output / "centers" / f"{mouse}_{date}_Probe{probe}.npz"
        save_rf(fixed, fixed_path)
        np.testing.assert_array_equal(load_rf(fixed_path).center_2d, fixed.center_2d)
        rf_by_id = {int(uid): index for index, uid in enumerate(fixed.unit_ids)}
        audit = dict(mouse=mouse, date=date, probe=probe, source_hashes=before,
                     saved_hd_classification=hd["metadata"]["classification"],
                     saved_manifest=dict(old.manifest), corrected_manifest=dict(fixed.manifest),
                     corrected_result=str(fixed_path), native_hd_bins=len(angles),
                     missing_rf_ids=[], not_current_hd3_ids=[], selected_ids=sorted(selected[mouse, date, probe]))
        for uid in sorted(selected[mouse, date, probe]):
            cls, preferred = hd_by_id.get(uid, (None, np.nan))
            index = rf_by_id.get(uid)
            row = dict(mouse=mouse, date=date, probe=probe, unit_id=uid,
                       hd_class=cls, hd_preferred_deg=preferred, rf_measured=index is not None,
                       rf_bins=np.nan, n_centers=np.nan, hd_session=hd_session, rf_session=rf_session)
            if cls != 3:
                audit["not_current_hd3_ids"].append(uid)
            if index is None:
                audit["missing_rf_ids"].append(uid)
                unit_rows.append(row)
                continue
            mask, center = fixed.mask_2d[index], fixed.center_2d[index]
            labels, masses = _label_clusters(mask.astype(float), mask.astype(bool),
                                             wrap_x=old.manifest["parameters"]["wrap_x"])
            assert int(center.sum()) == len(masses)
            row.update(rf_bins=int(mask.sum()), n_centers=int(center.sum()))
            unit_rows.append(row)
            for component in range(1, len(masses) + 1):
                coordinates = np.argwhere(center.astype(bool) & (labels == component))
                assert len(coordinates) == 1
                iy, ix = coordinates[0]
                center_rows.append(dict(**row, component=component,
                                        center_x_deg=float(maps[0].x_positions[ix]),
                                        center_y_deg=float(maps[0].y_positions[iy]),
                                        component_bins=int((labels == component).sum()),
                                        neuron_weight=1 / int(center.sum())))
            if cls == 3 and center.sum() > 1:
                examples.append((row, mask, old.center_2d[index], center,
                                 maps[0].x_positions, maps[0].y_positions))
        for path, digest in before.items():
            assert sha256(Path(path)) == digest
        sources.append(audit)
        print(f"Verified masks and corrected centers: {mouse} Probe {probe}", flush=True)
    units, centers = pd.DataFrame(unit_rows), pd.DataFrame(center_rows)
    assert not units.duplicated(IDENTITY).any()
    units.to_csv(output / "adn_units.csv", index=False)
    centers.to_csv(output / "adn_centers.csv", index=False)
    return units, centers, sources, examples


def association(table: pd.DataFrame, column: str, permutations: int):
    """Weighted partial association; HD alone uses unit permutations.

    Center rows have 1/k weight for a neuron with k centers. Thus every neuron
    contributes total weight one. Center correlations are descriptive because
    the outcome is also the number of center rows belonging to that neuron.
    """
    table = table.reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(table[IDENTITY])
    unit_index, unique = pd.factorize(keys, sort=False)
    first = np.array([np.flatnonzero(unit_index == i)[0] for i in range(len(unique))])
    unit_table = table.iloc[first]
    values = unit_table.n_centers.to_numpy(float)
    weights = (table.neuron_weight.to_numpy(float) if "neuron_weight" in table
               else np.ones(len(table)))
    np.testing.assert_allclose(np.bincount(unit_index, weights=weights), 1)
    strata = unit_table.probe.astype(str).to_numpy()
    labels = np.unique(strata)
    intercepts = np.column_stack([strata[unit_index] == label for label in labels]).astype(float)
    angle = table[column].to_numpy(float)
    circular = column != "center_y_deg"
    predictors = (np.column_stack([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                  if circular else angle[:, None])
    sqrt_w = np.sqrt(weights)
    nuisance = intercepts * sqrt_w[:, None]
    angular = predictors * sqrt_w[:, None]
    angular -= nuisance @ np.linalg.lstsq(nuisance, angular, rcond=None)[0]
    value = values[unit_index] * sqrt_w
    value -= nuisance @ np.linalg.lstsq(nuisance, value, rcond=None)[0]
    total = value @ value
    q, singular, _ = np.linalg.svd(angular, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-10))
    groups = [np.flatnonzero(strata == label) for label in labels]
    base = dict(n_units=len(unique), n_centers=len(table) if "neuron_weight" in table else 0,
                movable_units=sum(len(group) for group in groups if len(group) > 1),
                stratum_sizes=";".join(str(len(group)) for group in groups),
                statistic="circular_linear_R" if circular else "partial_Pearson_r",
                inference="within-probe unit permutation" if column == "hd_preferred_deg" else "descriptive only")
    if rank != predictors.shape[1] or total <= 1e-12:
        return dict(base, coefficient=np.nan, permutation_p=np.nan, status="insufficient variation"), None
    q = q[:, :rank]
    r2 = float(np.clip(np.sum((value @ q) ** 2) / total, 0, 1))
    design = np.column_stack([intercepts, predictors])
    coefficients = np.linalg.lstsq(design * sqrt_w[:, None],
                                   values[unit_index] * sqrt_w, rcond=None)[0]
    coefficient = np.sqrt(r2) * (1 if circular else np.sign(coefficients[-1]))
    if column != "hd_preferred_deg":
        return dict(base, coefficient=coefficient, permutation_p=np.nan, status="ok"), (labels, coefficients)
    # Collapse weighted center rows to one vector per neuron before shuffling.
    unit_projection = np.zeros((len(unique), rank))
    np.add.at(unit_projection, unit_index, q * sqrt_w[:, None])
    rng = np.random.default_rng(0)
    exceed = 0
    for start in range(0, permutations, 256):
        size = min(256, permutations - start)
        order = np.broadcast_to(np.arange(len(unique)), (size, len(unique))).copy()
        for group in groups:
            order[:, group] = rng.permuted(np.broadcast_to(group, (size, len(group))), axis=1)
        shuffled = np.sum((values[order] @ unit_projection) ** 2, axis=1) / total
        exceed += int(np.count_nonzero(shuffled >= r2 - 1e-12))
    return dict(base, coefficient=coefficient, permutation_p=(exceed + 1) / (permutations + 1),
                status="ok"), (labels, coefficients)


def fdr(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    finite = table.permutation_p.notna()
    p = table.loc[finite, "permutation_p"].to_numpy()
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q = np.empty(len(p))
    q[order] = np.minimum(adjusted, 1)
    table.loc[finite, "fdr_q"] = q
    return table


def analyze(units, centers, permutations, output):
    hd = units.loc[units.hd_class.eq(3) & units.rf_measured & units.hd_preferred_deg.notna()]
    rf = centers.loc[centers.hd_class.eq(3)]
    columns = ["hd_preferred_deg", "center_x_deg", "center_y_deg"]
    primary, fits = [], {}
    for mouse in hd.mouse.unique():
        for column in columns:
            data = (hd if column == columns[0] else rf).loc[lambda d: d.mouse.eq(mouse)]
            result, fit = association(data, column, permutations)
            primary.append(dict(mouse=mouse, angle=column, **result))
            fits[mouse, column] = fit
    primary = fdr(pd.DataFrame(primary))
    primary.to_csv(output / "correlations.csv", index=False)
    cohort = hd.groupby(["mouse", "probe"], sort=False).agg(
        n_hd3=("unit_id", "size"), n_rf_positive=("rf_bins", lambda s: int(s.gt(0).sum())),
        n_multiple_centers=("n_centers", lambda s: int(s.gt(1).sum())),
        total_centers=("n_centers", "sum"))
    cohort.to_csv(output / "cohort_counts.csv")
    mice = hd.mouse.unique()
    fig, axes = plt.subplots(len(mice), 3, figsize=(15, 3.25 * len(mice)),
                             squeeze=False, layout="constrained")
    fig.suptitle("ADN | independent RF regions per neuron", fontsize=17)
    colors = {"A": "#2878b5", "B": "#d78018"}
    titles = ["HD preferred direction (zero RF included)", "RF center azimuth", "RF center elevation"]
    for row, mouse in enumerate(mice):
        for col, column in enumerate(columns):
            ax = axes[row, col]
            data = (hd if col == 0 else rf).loc[lambda d: d.mouse.eq(mouse)]
            stats = primary.loc[primary.mouse.eq(mouse) & primary.angle.eq(column)].iloc[0]
            for probe, group in data.groupby("probe", sort=False):
                coincident = group.groupby([column, "n_centers"]).size().reset_index(name="multiplicity")
                ax.scatter(coincident[column], coincident.n_centers, s=30*coincident.multiplicity,
                           alpha=.65, color=colors[probe],
                           edgecolors="white", linewidth=.4, label=f"Probe {probe}")
            fit = fits[mouse, column]
            if fit is not None:
                labels, coefficients = fit
                grid = np.linspace(0 if col == 0 else data[column].min(),
                                   360 if col == 0 else data[column].max(), 361)
                for index, probe in enumerate(labels):
                    fitted = (coefficients[index] + coefficients[-2] * np.cos(np.deg2rad(grid))
                              + coefficients[-1] * np.sin(np.deg2rad(grid)) if col < 2
                              else coefficients[index] + coefficients[-1] * grid)
                    ax.plot(grid, fitted, color=colors[probe], linewidth=1.3)
            source = hd.loc[hd.mouse.eq(mouse)].iloc[0]
            session_label = f"HD {int(source.hd_session)} / RF {int(source.rf_session)}"
            ax.set(title=f"{mouse} ({session_label})\n{titles[col]}", xlabel="Degrees",
                   ylabel="Independent RF regions per neuron" if col == 0 else "")
            if col < 2:
                ax.set_xlim((0, 360) if col == 0 else (-180, 180))
            ax.set_ylim(-.4, hd.loc[hd.mouse.eq(mouse), "n_centers"].max() + 1.3)
            ax.set_yticks(np.arange(int(hd.loc[hd.mouse.eq(mouse), "n_centers"].max()) + 1))
            count_label = f"{int(stats.n_units)} neurons"
            if col:
                count_label += f", {int(stats.n_centers)} centers"
            inference = (f"p = {stats.permutation_p:.3g}; q = {stats.fdr_q:.3g}" if col == 0
                         else "descriptive; equal neuron weights")
            ax.text(.03, .98, f"{count_label}\n{'R' if col < 2 else 'r'} = {stats.coefficient:.2f}; "
                    f"{inference}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=1))
            ax.grid(axis="y", color="#dddddd", linewidth=.6)
            if mouse == "m15" and col == 0:
                ax.legend(loc="upper right", facecolor="white", framealpha=1, fontsize=8)
    fig.savefig(output / "adn_correlations.png", dpi=170, facecolor="white", transparent=False)
    fig.savefig(output / "adn_correlations.pdf", facecolor="white", transparent=False)
    plt.close(fig)
    print(cohort.to_string())
    print(primary.to_string(index=False))
    return primary


def plot_examples(examples, output):
    examples = examples[:8]
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 1, figsize=(12, 2.4 * len(examples)), layout="constrained")
    for ax, (row, mask, old, center, x, y) in zip(np.atleast_1d(axes), examples, strict=True):
        dx, dy = np.diff(x).mean(), np.diff(y).mean()
        ax.imshow(mask, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1,
                  extent=(x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2), interpolation="nearest")
        iy, ix = np.nonzero(old)
        ax.scatter(x[ix], y[iy], marker="x", color="#d78018", s=100, label="Previous center")
        iy, ix = np.nonzero(center)
        ax.scatter(x[ix], y[iy], facecolors="none", edgecolors="#1677bd", s=120,
                   linewidths=1.8, label="One center per connected region")
        ax.set(title=f"{row['mouse']} Probe {row['probe']} unit {row['unit_id']}: "
               f"{row['n_centers']} regions / {row['n_centers']} centers", ylabel="Elevation (deg)")
        ax.legend(loc="upper right", fontsize=8, facecolor="white", framealpha=1)
    np.atleast_1d(axes)[-1].set_xlabel("Azimuth (deg)")
    fig.savefig(output / "center_fix_examples.png", dpi=150, facecolor="white", transparent=False)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-root", type=Path, default=Path("/mnt/senzailab/Kai/#Recording"))
    parser.add_argument("--selection", type=Path, default=Path("config/retinotopography_adn_units.json"))
    parser.add_argument("--output", type=Path, default=Path("output/adn_rf_correlations_20261009_with_m21_hd1"))
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--m21-hd-session", type=int, choices=(1, 9), default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False,
                         "text.color": "black", "axes.labelcolor": "black",
                         "axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black"})
    units, centers, sources, examples = collect(args.recording_root, args.selection, args.output,
                                               args.m21_hd_session)
    analyze(units, centers, args.permutations, args.output)
    plot_examples(examples, args.output)
    provenance = dict(python=sys.executable, selection_path=str(args.selection),
                      selection_sha256=sha256(args.selection), script_sha256=sha256(Path(__file__)),
                      classification_source="saved hd_class; source definitions retained per recording",
                      hd_class=3, permutations=args.permutations, seed=0,
                      rf_count="Number of independent connected RF regions per neuron = number of centers",
                      rf_bins="Preserved separately as number of true native (Y,X) mask bins; not the analyzed count",
                      centers="One response-weighted medoid per four-connected region, original wrap_x",
                      cohorts="HD: measured RF including zero; centers: RF-positive; both saved HD Class 3 ADN",
                      statistics="Circular-linear R for HD/azimuth; signed partial Pearson r for elevation; probe intercepts",
                      center_weight="1 / number of centers per neuron; total neuron weight one",
                      permutation="HD only: neuron region count shuffled within probe",
                      center_inference="Descriptive only: region count determines number of center rows, so shuffling counts over fixed centers is inappropriate",
                      fdr="BH across per-mouse HD tests",
                      scope="Exploratory within-recording associations; no across-animal inference",
                      native_grids="M14 10x10 degrees; other mice 12x12 degrees; mice not pooled",
                      source_files_unchanged=True, sources=sources)
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
