"""Render compact presentation panels from saved curves, peaks and statistics."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from tqdm import tqdm

from Utils.direction_comparison import align_profiles, plot_profiles, tc_loader, tcRange
from Utils.plotting import LIGHT_PLOT_STYLE, plot_direction_comparison
from Utils.statistic_utils import circular_distance_matrix_deg
from scripts.run_inhibitory_comparisons_20261012 import minmax_profiles


BASE = Path("output/session_comparisons_20261012")
OUT = BASE / "presentation_figures"
LOCAL_ROOT = Path("/Users/vxf1610/Developer/rfmapping")
SESSIONS = [
    ("m14", "260609", 3, 1), ("m15", "260630", 3, 1),
    ("m19", "260827", 2, 9), ("m20", "260921", 2, 9),
    ("m21", "261006", 3, 9), ("m22", "261008", 2, 9),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup():
    plt.rcParams.update(LIGHT_PLOT_STYLE)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 16,
                         "axes.labelsize": 19, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 14})


def save(fig, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=170, facecolor="white", transparent=False)
    fig.savefig(destination.with_suffix(".svg"), facecolor="white", transparent=False)
    plt.close(fig)
    return str(destination)


def panel(table, label, destination, *, aligned=False, grayscale=False, ticks=None,
          unit_minmax=False):
    labels = (-180, -90, 0, 90, 180) if aligned else ticks
    fig, ax = plot_profiles(table, table.index, label, axis={"ticklabels": labels},
                            aligned=aligned, show=False, figsize=(6.2, 5.7),
                            cmap="Greys" if grayscale else "viridis", vmin=0, vmax=1,
                            colorbar_label="Normalized FR" if unit_minmax else "Response")
    ax.set(title="", xlabel=f"{'Δ' if aligned else ''}{label} (°)", ylabel="Unit")
    ranks = np.unique(np.linspace(0, len(table) - 1, min(6, len(table))).astype(int))
    ax.set_yticks(ranks, labels=ranks + 1)
    ax.tick_params(axis="y", length=0)
    fig.axes[1].set_yticks([0, .5, 1])
    if unit_minmax:
        fig.axes[1].set_yticklabels(["min", "0.5", "max"])
    fig.tight_layout(pad=.4)
    save(fig, destination)
    return dict(path=str(destination), unit_ids=list(table.index),
                native_angles_deg=list(table.columns), aligned=aligned,
                colormap="Greys" if grayscale else "viridis",
                unit_minmax=unit_minmax)


def stats_panels(x, y, statistics, x_label, y_label, directory, *, x_ticks, y_ticks,
                 unique_pairs=False, fisher_lee=None, names=("scatter", "mantel")):
    figures = plot_direction_comparison(
        x, y, statistics, reference_label=x_label, matched_label=y_label,
        reference_ticklabels=x_ticks, matched_ticklabels=y_ticks,
        density_bin_count=15, show=False,
    )
    output = []
    for index, (fig, ax) in enumerate(figures):
        fig.set_size_inches(6.2, 5.7)
        ax.set_title("")
        prefix = "Δ" if index else ""
        ax.set_xlabel(f"{prefix}{x_label} (°)")
        ax.set_ylabel(f"{prefix}{y_label} (°)")
        if index == 0:
            stats = statistics["same_unit"]
            message = f"ρ = {stats['circular_correlation_rho']:.3f}\np = {stats['permutation_p']:.4g}"
            if fisher_lee is not None:
                message = (f"JS ρ = {stats['circular_correlation_rho']:.3f}, p = {stats['permutation_p']:.4g}\n"
                           f"FL ρ = {fisher_lee['rho']:.3f}, p = {fisher_lee['permutation_p']:.4g}")
        else:
            stats = statistics["pairwise"]
            message = f"Mantel ρ = {stats['spearman_mantel_rho']:.3f}\np = {stats['permutation_p']:.4g}"
            ax.get_legend().remove()
            if unique_pairs:
                # The overlap analysis displays unique off-diagonal pairs.
                ax.collections[0].colorbar.remove()
                for collection in list(ax.collections):
                    collection.remove()
                pairs = np.triu_indices(len(x), k=1)
                dx, dy = circular_distance_matrix_deg(x), circular_distance_matrix_deg(y)
                image = ax.hist2d(dx[pairs], dy[pairs], bins=np.linspace(0, 180, 16), cmap="viridis")[3]
                fig.colorbar(image, ax=ax, pad=.06, label="Pairs")
                ax.set(xlim=(0, 180), ylim=(0, 180))
            else:
                fig.axes[1].set_ylabel("Pairs")
        ax.texts[0].set_text(message)
        ax.texts[0].set_fontsize(14 if fisher_lee is not None and index == 0 else 16)
        fig.tight_layout(pad=.4)
        output.append(save(fig, directory / f"{names[index]}.png"))
    return output


def polar(x, y, pvalue, labels, destination, *, counts=None, edges=None):
    if counts is None:
        counts, edges = np.histogram(np.deg2rad((np.asarray(x) + y) % 360),
                                    bins=15, range=(0, 2 * np.pi))
    counts, edges = np.asarray(counts), np.asarray(edges)
    assert len(counts) == 15 and int(counts.sum()) == len(x)
    fig, ax = plt.subplots(figsize=(6.2, 5.7), subplot_kw={"projection": "polar"})
    ax.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges),
           color="#3475A8", edgecolor="white")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids((0, 90, 180, 270), labels=("0°", "90°", "180°", "−90°"))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.set_xlabel(f"{labels[0]} + {labels[1]} (°)\nRayleigh p = {pvalue:.4g}", labelpad=22)
    fig.tight_layout(pad=.8)
    return save(fig, destination)


def image(path, reference=False):
    return dict(path=str(LOCAL_ROOT / path), reference=reference)


def slide(title, footer, paths, metadata, *, reference=None):
    return dict(title=title, footer=footer,
                images=[image(path, i == reference) for i, path in enumerate(paths)],
                notes=json.dumps(metadata, indent=2))


def render_inhibitory(session, *, source_key=None):
    setup()
    mouse, date, rf, hd = session
    key = source_key or f"{mouse}_{date}"
    source = BASE / "inhibitory" / key
    destination = OUT / "inhibitory" / key
    metadata = json.loads((source / "summary.json").read_text())
    before = {str(path): sha(path) for path in source.glob("*.csv")}
    before[str(source / "summary.json")] = sha(source / "summary.json")
    probes = "+".join(key.rsplit("_", 1)[1] for key in metadata["input_pairs"])
    footer = f"{mouse} {date} · RF{rf}/HD{hd} · Probe {probes}"
    title = f"Inhibitory RF and HD · n = {metadata['paired_units']}"
    if metadata.get("sensitivity_only"):
        title += " · sensitivity"
    slides, panels = [], []
    for mode in ("native", "aligned"):
        for reference in ("hd", "rf"):
            paths = []
            for kind, label in (("hd", "HD"), ("rf", "RF")):
                table = tc_loader(source / f"{mode}_sorted_by_{reference}_{kind}.csv")
                path = destination / f"{mode}_sorted_by_{reference}_{kind}.png"
                panels.append(panel(minmax_profiles(table), label, path, aligned=mode != "native",
                                    grayscale=kind == "rf", ticks=tcRange(False) if kind == "hd" else tcRange(True),
                                    unit_minmax=True))
                paths.append(str(path))
            notes = dict(metadata, display_mode=mode, reference=reference,
                         display_normalization="Both HD and RF: (row - finite min) / (finite max - finite min); constant rows 0; NaNs preserved",
                         rf_colormap="Greys, higher response darker")
            slides.append(slide(title, footer, paths, notes, reference=0 if reference == "hd" else 1))
    peaks = pd.read_csv(source / "paired_native_directions.csv", index_col=0)
    x, y = peaks.reference_peak_deg.to_numpy(), peaks.matched_peak_deg.to_numpy()
    paths = stats_panels(x, y, metadata["statistics"], "HD", "RF", destination,
                         x_ticks=tcRange(False), y_ticks=tcRange(True))
    slides.append(slide(title, footer, paths, metadata))
    histogram = metadata["sum_histogram"]
    path = polar(x, y, histogram["rayleigh_p"], ("HD", "RF"), destination / "polar.png",
                 counts=histogram["counts"], edges=histogram["edges_rad"])
    slides.append(slide(title, footer, [path], metadata))
    assert before == {path: sha(path) for path in before}
    return dict(set="inhibitory", key=key, slides=slides, panels=panels,
                protected_inputs=before, statistics=metadata["statistics"], n=len(peaks))


def read_numeric(path):
    table = pd.read_csv(path, index_col=0, float_precision="round_trip")
    table.columns = table.columns.astype(float)
    return table


def render_overlap(session):
    setup()
    mouse, date, rf, hd = session
    key = f"{mouse}_{date}_rf{rf}_hd{hd}"
    source = BASE / "overlap" / key
    destination = OUT / "overlap" / key
    metadata = json.loads((source / "manifest.json").read_text())
    before = {str(path): sha(path) for path in source.glob("*.csv")}
    before[str(source / "manifest.json")] = sha(source / "manifest.json")
    peaks = pd.read_csv(source / "paired_peaks.csv", index_col="key")
    hd_tables, rf_tables = [], []
    for probe in metadata["probes"]:
        hd_tables.append(read_numeric(source / f"Probe{probe['probe']}_hd30_class3.csv"))
        rf_tables.append(read_numeric(source / f"Probe{probe['probe']}_overlap_rates_native.csv"))
    hd_table = pd.concat(hd_tables).loc[peaks.index]
    rf_table = pd.concat(rf_tables).loc[peaks.index]
    hd_table = hd_table.div(hd_table.max(axis=1), axis=0)
    rf_table = rf_table.div(rf_table.max(axis=1), axis=0)
    hd_table.attrs["range"] = [-180, 180]
    rf_table.attrs["range"] = [-150, 150] if mouse == "m14" else [-180, 180]
    footer = f"{mouse} {date} · RF{rf}/HD{hd} · Probe {'+'.join(p['probe'] for p in metadata['probes'])}"
    title = f"RF and HD · n = {len(peaks)}"
    slides, panels = [], []
    for mode, reference in (("native", "hd"), ("native", "rf"), ("aligned", "hd")):
        order = peaks.sort_values(f"{reference}_peak_deg", kind="stable").index
        paths = []
        for kind, label, table in (("hd", "HD", hd_table), ("rf", "RF", rf_table)):
            displayed = table.loc[order]
            if mode == "aligned":
                displayed = align_profiles(table, order, -peaks.loc[order, "hd_peak_deg"].to_numpy(),
                                           bins=60, fill_value=np.nan)
            path = destination / f"{mode}_sorted_by_{reference}_{kind}.png"
            panels.append(panel(displayed, label, path, aligned=mode == "aligned", ticks=tcRange(True)))
            paths.append(str(path))
        slides.append(slide(title, footer, paths, dict(metadata, display_mode=mode, reference=reference),
                            reference=0 if reference == "hd" else 1))
    x, y = peaks.hd_peak_deg.to_numpy(), peaks.rf_peak_deg.to_numpy()
    statistics = metadata["statistics"]
    paths = stats_panels(x, y, statistics["direction"], "HD", "RF", destination,
                         x_ticks=tcRange(True), y_ticks=tcRange(True), unique_pairs=True,
                         fisher_lee=statistics["fisher_lee"])
    slides.append(slide(title, footer, paths, metadata))
    path = polar(x, y, statistics["sum_rayleigh"]["p"], ("HD", "RF"), destination / "polar.png")
    slides.append(slide(title, footer, [path], metadata))
    assert before == {path: sha(path) for path in before}
    return dict(set="overlap", key=key, slides=slides, panels=panels,
                protected_inputs=before, statistics=statistics, n=len(peaks))


def render_ebc(metadata):
    setup()
    source = Path(metadata["figures"][0]["path"]).parent
    destination = OUT / "ebc" / source.parent.name / source.name
    before = {str(source / name): sha(source / name) for name in ("reference.csv", "matched.csv", "peaks.csv", "manifest.json")}
    assert metadata["ebc_source_metadata"]["angleConvention"] == "egocentric clockwise: 0 forward, 90 right"
    first = tc_loader(source / "reference.csv", range=tcRange(True))
    label = metadata["comparison"]
    second = tc_loader(source / "matched.csv", range=tcRange(False) if label == "HD" else tcRange(True))
    peaks = pd.read_csv(source / "peaks.csv", index_col=0)
    panels = []
    for index, (table, name, ticks) in enumerate(((first, "EBC", tcRange(True)),
                                                (second, label, tcRange(False) if label == "HD" else tcRange(True)))):
        panels.append(panel(table.loc[peaks.index], name, destination / f"comparison_{index:02d}.png", ticks=ticks))
    x, y = peaks.reference_peak_deg.to_numpy(), peaks.matched_peak_deg.to_numpy()
    stats_panels(x, y, metadata["direction"], "EBC", label, destination,
                 x_ticks=tcRange(True), y_ticks=tcRange(False) if label == "HD" else tcRange(True),
                 names=("comparison_13", "comparison_14"))
    polar(x, y, metadata["rayleigh_p"], ("EBC", label), destination / "comparison_12.png")
    assert before == {path: sha(path) for path in before}
    return dict(set="ebc", key=metadata["name"], panels=panels, protected_inputs=before,
                source_manifest=str(source / "manifest.json"), statistics=metadata["direction"],
                figures=[str(destination / f"comparison_{index:02d}.png") for index in (0, 1, 12, 13, 14)],
                reference_image_index=0, n=len(peaks))


def job(item):
    global OUT
    kind, data, OUT = item
    return {"inhibitory": render_inhibitory, "overlap": render_overlap, "ebc": render_ebc}[kind](data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sets", nargs="+", choices=("inhibitory", "overlap", "ebc"), default=["inhibitory", "overlap", "ebc"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--first-only", action="store_true")
    parser.add_argument("--output-root", type=Path, default=OUT)
    parser.add_argument("--ebc-root", type=Path, default=BASE / "spatial_ebc/clockwise")
    args = parser.parse_args()
    jobs = []
    for kind in args.sets:
        if kind == "ebc":
            entries = json.loads((args.ebc_root / "comparison_manifest.json").read_text())
            data = [entry for entry in entries if entry["mouse"] != "pooled" and entry["comparison"] in ("RF", "HD") and entry["n_shared"]]
        else:
            data = SESSIONS
        jobs.extend((kind, item, args.output_root) for item in (data[:1] if args.first_only else data))
    args.output_root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        results = list(tqdm(pool.map(job, jobs), total=len(jobs), desc="Presentation comparisons", unit="cohort"))
    for kind in args.sets:
        selected = [entry for entry in results if entry["set"] == kind]
        destination = args.output_root / kind
        destination.mkdir(exist_ok=True)
        (destination / "manifest.json").write_text(json.dumps(dict(script_sha256=sha(__file__), cohorts=selected,
            statistics_recomputed=False, scientific_inputs_unchanged=True, statistics_bins=15), indent=2) + "\n")
        if kind != "ebc":
            (destination / "slides.json").write_text(json.dumps([slide for group in selected for slide in group["slides"]], indent=2) + "\n")


if __name__ == "__main__":
    main()
