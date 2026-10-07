"""Save one HD-cell probe-position figure for a free-moving session."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def plot_hd_cell_distribution(session, probes):
    session = Path(session)
    data = session / "data"
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "text.color": "#1c2630", "axes.labelcolor": "#1c2630",
        "xtick.color": "#1c2630", "ytick.color": "#1c2630",
    })
    fig, axes = plt.subplots(1, len(probes), figsize=(5 * len(probes), 8.5), squeeze=False)
    fig.subplots_adjust(left=0.15 / len(probes), right=0.97, top=0.81, bottom=0.09, wspace=0.35)
    fig.suptitle(f"{session.name} · HD cell distribution", fontsize=15, y=0.98)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color="#d9572b", label="HD Class 3"),
        Line2D([], [], marker="o", linestyle="", color="#8a959f", label="Other good units"),
        Line2D([], [], marker="s", linestyle="", color="#d9dfe5", markersize=4, label="Contacts"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.94), frameon=False, ncol=3, fontsize=8)

    for probe, ax in zip(probes, axes.ravel(), strict=True):
        tc = json.loads((data / "tuning_curves" / f"Probe{probe}" / "tuning_curves.tc").read_text())
        positions = pd.read_csv(data / "spike_position" / f"Probe{probe}" / "positions.csv").set_index("unit_id")
        channels = pd.read_csv(data / "waveform" / f"Probe{probe}" / "channels.csv")
        hd_ids = np.asarray(tc["unit_id"])[np.asarray(tc["unit_data"]["hd_class"]) == 3]
        hd = positions.loc[hd_ids]
        other = positions.drop(hd_ids)
        cutoff = tc["metadata"]["classification"]["kappa_cutoff"]
        ax.scatter(channels.x_um, channels.y_um, marker="s", s=8, color="#d9dfe5", linewidths=0, zorder=1)
        ax.scatter(other.x_um, other.y_um, s=20, color="#8a959f", alpha=0.65, edgecolors="white", linewidths=0.4, zorder=2)
        ax.scatter(hd.x_um, hd.y_um, s=45, color="#d9572b", edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_title(f"Probe {probe} · {len(hd)} HD / {len(positions)} good units\nClass 3, κ ≥ {cutoff:g}", fontsize=11, pad=12)
        ax.set(xlabel="Probe x (µm)", ylabel="Probe y (µm)",
               xlim=(channels.x_um.min() - 55, channels.x_um.max() + 55),
               ylim=(channels.y_um.min() - 60, channels.y_um.max() + 80))
        for shank, contacts in channels.groupby("shank_id"):
            center = (contacts.x_um.min() + contacts.x_um.max()) / 2
            ax.text(center, channels.y_um.max() + 40, f"S{shank}", ha="center", fontsize=8)
        ax.grid(axis="y", color="#e9edf0", linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#acb5bd")
        ax.tick_params(labelsize=9)

    output = data / "tuning_curves" / "hd_cell_distribution.png"
    fig.savefig(output, dpi=200, facecolor="white", transparent=False)
    plt.close(fig)
    print(f"Saved HD cell distribution: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("probes", help="Probe letters, e.g. A or AB")
    args = parser.parse_args()
    plot_hd_cell_distribution(args.session, args.probes)
