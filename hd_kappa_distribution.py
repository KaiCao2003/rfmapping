"""Inspect HD concentration for the recordings in tc_comparison_pairs.ipynb.

Run on hhw9l84 with ~/.virtualenvs/rfmapping/bin/python. Source tuning curves
and classifications are read only; no cutoff is applied or saved to them.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from Utils.direction_comparison import hd_pick, load_rf, load_tc, rf_pick
from Utils.plotting import LIGHT_PLOT_STYLE
from Utils.tuning_curve_utils import von_mises_kappa


RECORDINGS = (
    ("m14", 260609, 1, 3, "A"),
    ("m15", 260630, 1, 3, "AB"),
    ("m19", 260827, 9, 2, "A"),
    ("m20", 260921, 9, 2, "A"),
)


def collect_units(root):
    rows, sources = [], []
    for mouse, date, hd_session, rf_session, probes in RECORDINGS:
        for probe in probes:
            base = root / mouse / str(date)
            tc_path = base / f"{date}_{hd_session}/data/tuning_curves/Probe{probe}/tuning_curves.tc"
            rf_path = base / (
                f"{date}_{rf_session}/data/rfmapping/good/-100_400_1ms/Probe{probe}/"
                f"regular_unitsSpikeCounts_{date}_{rf_session}.rfmap"
            )
            content = tc_path.read_bytes()
            data = json.loads(content)
            rates = np.asarray(data["firing_rate_hz"], dtype=float)
            occupancy = np.asarray(data["occupancy_time_s"], dtype=float)
            edges = np.asarray(data["angle_bin_edges_deg"], dtype=float)
            # All directions must be measured; an unvisited bin is not zero firing.
            assert np.all(occupancy > 0), f"Unvisited HD bins: {tc_path}"
            assert len(occupancy) == 180 and np.allclose(np.diff(edges), 2)
            assert np.all(np.isfinite(rates)) and np.all(rates >= 0)
            np.testing.assert_allclose(rates, np.asarray(data["spike_counts"]) / occupancy)
            angles = np.deg2rad((edges[:-1] + edges[1:]) / 2)

            recording = dict(mouse=mouse, date=date, probe=probe)
            hd = load_tc(tc_path, **recording)
            # This cutoff audit includes both classes that pass the significance tests.
            hd_significant = hd_pick(hd, hd_class=(2, 3))
            rf = load_rf(rf_path, **recording, rf_type="2d")
            paired = hd_significant.index.intersection(rf_pick(rf, max_zero_bins=2).index)
            for index, (unit_id, curve) in enumerate(zip(data["unit_id"], rates, strict=True)):
                key = (mouse, str(date), probe, unit_id)
                info = data["unit_data"]
                rows.append({
                    "mouse": mouse, "date": date, "rf_session": rf_session,
                    "hd_session": hd_session, "probe": probe, "unit_id": unit_id,
                    "saved_hd_class": info["hd_class"][index],
                    "rayleigh_p": info["rayleigh_p"][index],
                    "shuffle_p": info["shuffle_p"][index],
                    "passes_rayleigh_and_shuffle": key in hd_significant.index,
                    "paired_rf_unit": key in paired,
                    "kappa": von_mises_kappa(curve, angles),
                })
            sources.append({
                "tuning_curve": str(tc_path), "sha256": hashlib.sha256(content).hexdigest(),
                "rf_map": str(rf_path), "rf_detection": str(rf_path.with_suffix(".npz")),
                "classification": data["metadata"]["classification"],
            })
            print(f"{mouse} Probe {probe}: {len(hd_significant)} Class 2 + 3, {len(paired)} paired", flush=True)
    return pd.DataFrame(rows), sources


def plot_distribution(units, output):
    hd = units.loc[units.passes_rayleigh_and_shuffle]
    assert np.isfinite(hd.kappa).all()
    upper = np.ceil(hd.kappa.max())
    edges = np.arange(0, upper + .25, .25)
    colors = ("#2874A6", "#148A80", "#B57929", "#8064AF")
    with plt.rc_context(LIGHT_PLOT_STYLE | {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "#222222", "axes.labelcolor": "#222222",
        "xtick.color": "#222222", "ytick.color": "#222222",
        "font.size": 10,
    }):
        fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.5), sharex=True)
        for ax, recording, color in zip(axes.flat, RECORDINGS, colors, strict=True):
            mouse, date, hd_session, rf_session, probes = recording
            group = hd.loc[hd.mouse == mouse]
            paired = group.loc[group.paired_rf_unit, "kappa"]
            ax.hist(group.kappa, bins=edges, color=color, edgecolor="white", linewidth=.5)
            ax.plot(paired, np.full(len(paired), .035), "|", color="#17212B",
                    markersize=9, markeredgewidth=1.1, transform=ax.get_xaxis_transform())
            ax.axvline(group.kappa.median(), color=color, linestyle="--", linewidth=1)
            probe_label = "+".join(probes)
            ax.set_title(f"{mouse} · {str(date)[2:]}_{rf_session} · Probe {probe_label}",
                         loc="left", fontweight="bold", pad=10)
            ax.text(.98, .94, f"Class 2 + 3: n = {len(group)}\nPaired RF: n = {len(paired)}\n"
                    f"Median κ = {group.kappa.median():.3f}",
                    transform=ax.transAxes, ha="right", va="top", color="#222222")
            ax.set(xlim=(0, upper), ylabel="HD cells", xticks=np.arange(0, upper + 1))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#E8EBEE", linewidth=.6)
        for ax in axes[-1]:
            ax.set_xlabel("von Mises concentration κ")
        fig.suptitle("HD concentration before choosing a cutoff", x=.07, ha="left",
                     fontsize=16, fontweight="bold", y=.98)
        fig.text(.07, .927, "Class 2 + 3 = Rayleigh + shuffle (both p ≤ 0.01)  ·  Bars: before κ cutoff  ·  Black ticks: paired RF subset",
                 fontsize=9, color="#404850")
        fig.text(.07, .015, "κ from normalized 180-bin HD rates; occupancy corrected, no smoothing or baseline subtraction.\n"
                 "HD source sessions: m14/m15 = 1; m19/m20 = 9. Dashed lines mark medians; histogram bin width = 0.25.",
                 fontsize=8.5, color="#404850")
        fig.subplots_adjust(left=.07, right=.98, top=.84, bottom=.15, hspace=.42, wspace=.2)
        for suffix in ("png", "svg"):
            fig.savefig(output / f"hd_kappa_distribution.{suffix}", dpi=180,
                        facecolor="white", transparent=False)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/senzailab/Kai/#Recording"))
    parser.add_argument("--output", type=Path, default=Path("output/hd_kappa_distribution"))
    args = parser.parse_args()
    units, sources = collect_units(args.root)
    args.output.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.output / "hd_kappa_units.csv", index=False)
    hd = units.loc[units.passes_rayleigh_and_shuffle]
    summary = hd.groupby("mouse", sort=False).agg(
        n=("kappa", "size"), paired_n=("paired_rf_unit", "sum"),
        minimum=("kappa", "min"), median=("kappa", "median"), maximum=("kappa", "max"),
    )
    summary.to_csv(args.output / "hd_kappa_summary.csv")
    metadata = {
        "method": "rate_weighted_von_mises_mle_180_bins",
        "equation": "I1(kappa)/I0(kappa) = abs(sum(rate * exp(1j * theta))) / sum(rate)",
        "smoothing": False, "baseline_subtracted": False, "cutoff": None,
        "zero_total_rate": "undefined; blank kappa in CSV",
        "single_direction_rate": "unbounded concentration; inf in CSV",
        "selection": "Both saved p values <= 0.01, via the paired notebook loader",
        "paired_rf_selection": "2d detection; at most 2 missing and 2 zero native x bins",
        "sources": sources,
    }
    (args.output / "hd_kappa_method.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_distribution(units, args.output)
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
