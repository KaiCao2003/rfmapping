"""Plot saved ADN RF centers, retaining neuron identities and native coordinates."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


KEYS = ["mouse", "date", "probe", "unit_id"]
SAME = "#36779B"
CROSS = "#CF6639"
TRIPLE = "#8C569C"
DARK = "#26343F"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_data(source, selection):
    units = pd.read_csv(source / "adn_units.csv", dtype={"date": str})
    centers = pd.read_csv(source / "adn_centers.csv", dtype={"date": str})
    selected = {(row["mouse"], row["date"], row["probe"], uid)
                for row in json.loads(selection.read_text())["recordings"]
                for uid in row["unit_ids"]}
    assert set(units[KEYS].itertuples(index=False, name=None)) <= selected
    units = units.loc[units.hd_class.eq(3) & units.rf_measured].copy()
    centers = centers.merge(units[KEYS], on=KEYS, validate="many_to_one")
    assert centers.hd_class.eq(3).all()
    assert not units.duplicated(KEYS).any()
    grouped = centers.groupby(KEYS, sort=False)
    np.testing.assert_array_equal(grouped.size(), grouped.n_centers.first())
    pairs = []
    for identity, group in centers.loc[centers.n_centers.eq(2)].groupby(KEYS, sort=False):
        group = group.sort_values("center_x_deg")
        left, right = group.iloc[0], group.iloc[1]
        pairs.append(dict(zip(KEYS, identity), x1=left.center_x_deg, x2=right.center_x_deg,
                          y1=left.center_y_deg, y2=right.center_y_deg,
                          spans_zero=bool(left.center_x_deg < 0 < right.center_x_deg)))
    return units, centers, pd.DataFrame(pairs)


def style_axes(ax, mouse):
    ax.set_facecolor("white")
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.axvline(0, color="#87939D", linestyle="--", linewidth=1, zorder=0)
    if mouse == "m14":
        for bounds in [(-180, -150), (150, 180)]:
            ax.axvspan(*bounds, color="#F0F2F4", zorder=-1)
    for edge in ["top", "right"]:
        ax.spines[edge].set_visible(False)
    for edge in ["left", "bottom"]:
        ax.spines[edge].set_color("#ADB7BF")
    ax.tick_params(labelsize=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E6EBEF", linewidth=.65)


def save_figure(fig, output, name):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180,
                    facecolor="white", transparent=False)
    plt.close(fig)


def plot_pairs(pairs, output, mice):
    n_rows = (len(mice) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.5, 5.75*n_rows), squeeze=False)
    fig.subplots_adjust(left=.10, right=.98, bottom=.13, top=.82, wspace=.24, hspace=.34)
    crossing = int(pairs.spans_zero.sum())
    fig.suptitle("ADN only | two RF regions per neuron", fontsize=20, fontweight="bold", y=.97)
    fig.text(.5, .932, f"HD Class 3  ·  {len(pairs)} neurons  ·  {crossing} span X = 0°  ·  "
             f"{len(pairs)-crossing} stay on one side", ha="center", fontsize=12)
    fig.legend(handles=[
        Line2D([], [], color=CROSS, marker="o", label="Spans X = 0°"),
        Line2D([], [], color=SAME, marker="o", label="Same side of X = 0°"),
        Line2D([], [], color=DARK, linestyle="", marker="o", label="Probe A"),
        Line2D([], [], color=DARK, linestyle="", marker="s", label="Probe B"),
    ], loc="upper center", bbox_to_anchor=(.5, .909), ncol=4, frameon=False, fontsize=10)
    for ax, mouse in zip(axes.flat, mice):
        group = pairs.loc[pairs.mouse.eq(mouse)].sort_values(["probe", "x1", "x2", "unit_id"])
        labels = []
        for y, row in enumerate(group.itertuples(index=False)):
            color = CROSS if row.spans_zero else SAME
            ax.plot([row.x1, row.x2], [y, y], color=color, linewidth=2, alpha=.85)
            ax.scatter([row.x1, row.x2], [y, y], color=color, s=44,
                       marker="o" if row.probe == "A" else "s", edgecolors="white", linewidth=.65, zorder=3)
            star = "*" if mouse == "m14" and row.probe == "A" and row.unit_id == 671 else ""
            labels.append(f"{row.probe} {row.unit_id}{star}")
        style_axes(ax, mouse)
        ax.set_yticks(np.arange(len(group)), labels)
        ax.set_ylim(len(group)-.5, -.65)
        ax.set_xlabel("Native RF X (deg)", fontsize=11)
        ax.set_ylabel("Probe / unit ID", fontsize=10)
        ax.set_title(f"{mouse}  |  {int(group.spans_zero.sum())}/{len(group)} span zero",
                     loc="left", fontsize=13, fontweight="bold", pad=12)
    for ax in axes.flat[len(mice):]:
        ax.set_visible(False)
    fig.text(.10, .079, "Each row is one neuron. The line links its two centers; it does not show RF extent.",
             fontsize=10)
    fig.text(.10, .054, "X = 0° is the native stimulus reference, not independently verified head-forward.", fontsize=10)
    fig.text(.10, .029, "* m14 A671 splits into 3 regions without edge wrapping. Gray bands: unmeasured gap. Saved grouping shown.",
             fontsize=9.5, color="#5E6871")
    save_figure(fig, output, "adn_double_rf_pairs")


def plot_maps(units, centers, output, mice):
    fig, axes = plt.subplots(len(mice), 2, figsize=(13, 3.125*len(mice)), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.115, top=.84, wspace=.18, hspace=.6)
    n_single = int(units.n_centers.eq(1).sum())
    n_double = int(units.n_centers.eq(2).sum())
    n_triple = int(units.n_centers.eq(3).sum())
    fig.suptitle("ADN only | RF center locations", fontsize=20, fontweight="bold", y=.978)
    fig.text(.5, .942, f"HD Class 3  ·  {n_single} single-RF, {n_double} double-RF, "
             f"{n_triple} triple-RF neurons", ha="center", fontsize=12)
    fig.legend(handles=[
        Line2D([], [], color=SAME, marker="o", linestyle="", label="Single RF / same-side pair"),
        Line2D([], [], color=CROSS, marker="o", label="Two RFs spanning X = 0°"),
        Line2D([], [], color=TRIPLE, marker="D", linestyle="--", label="Three RFs"),
    ], loc="upper center", bbox_to_anchor=(.5, .916), ncol=3, frameon=False, fontsize=10)
    for row, mouse in enumerate(mice):
        for col in range(2):
            ax = axes[row, col]
            group = centers.loc[centers.mouse.eq(mouse) &
                                (centers.n_centers.eq(1) if col == 0 else centers.n_centers.gt(1))]
            style_axes(ax, mouse)
            if col == 0:
                for probe, subgroup in group.groupby("probe", sort=False):
                    points = subgroup.groupby(["center_x_deg", "center_y_deg"]).size().reset_index(name="n")
                    ax.scatter(points.center_x_deg, points.center_y_deg, s=34*points.n,
                               color=SAME, alpha=.75, edgecolors="white", linewidth=.6,
                               marker="o" if probe == "A" else "s", zorder=3)
                    for point in points.loc[points.n.gt(1)].itertuples(index=False):
                        ax.annotate(str(point.n), (point.center_x_deg, point.center_y_deg),
                                    xytext=(6, 5), textcoords="offset points", fontsize=8, color=DARK)
            else:
                for identity, pair in group.groupby(KEYS, sort=False):
                    pair = pair.sort_values("center_x_deg")
                    triple = len(pair) == 3
                    cross = pair.center_x_deg.min() < 0 < pair.center_x_deg.max()
                    color = TRIPLE if triple else CROSS if cross else SAME
                    ax.plot(pair.center_x_deg, pair.center_y_deg, color=color,
                            alpha=.6, linewidth=1.2, linestyle="--" if triple else "-")
                    ax.scatter(pair.center_x_deg, pair.center_y_deg, color=color, s=32,
                               marker="D" if triple else "o" if identity[2] == "A" else "s",
                               edgecolors="white", linewidth=.5, alpha=.9, zorder=3)
                    if mouse == "m14" and identity[2:] == ("A", 671):
                        ax.annotate("671*", (pair.center_x_deg.iloc[0], pair.center_y_deg.iloc[0]),
                                    xytext=(5, 7), textcoords="offset points", fontsize=8, color=DARK)
            n_units = group[KEYS].drop_duplicates().shape[0]
            category = "Single RF" if col == 0 else "Multiple RFs"
            ax.set_title(f"{mouse}  |  {category}  |  {n_units} neurons, {len(group)} centers",
                         loc="left", fontsize=11.5, fontweight="bold", pad=10)
            ax.set_ylim(-48, 48)
            ax.set_yticks([-40, -20, 0, 20, 40])
            ax.set_ylabel("Native RF Y (deg)", fontsize=10)
            if row == len(mice) - 1:
                ax.set_xlabel("Native RF X (deg)", fontsize=11)
    fig.text(.075, .069, "Lines link centers from the same neuron, not RF boundaries. "
             "Single-RF point labels count coincident neurons within each probe.", fontsize=9.5)
    fig.text(.075, .046, "Native stimulus coordinates; no binocular/monocular boundary is assumed. "
             "Circles: Probe A; squares: Probe B.", fontsize=9.5)
    fig.text(.075, .023, "* m14 A671 splits into 3 regions without edge wrapping; saved grouping shown. "
             f"{int(units.n_centers.eq(0).sum())} measured zero-RF neurons have no center to plot.",
             fontsize=9.5, color="#5E6871")
    save_figure(fig, output, "adn_rf_center_maps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/adn_rf_correlations_20261009_with_m21_hd1"))
    parser.add_argument("--selection", type=Path, default=Path("config/retinotopography_adn_units.json"))
    args = parser.parse_args()
    output = args.source / "center_locations"
    output.mkdir(exist_ok=True)
    sources = [args.source / name for name in ("adn_units.csv", "adn_centers.csv")] + [args.selection]
    before = {str(path): digest(path) for path in sources}
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False,
                         "text.color": DARK, "axes.labelcolor": DARK, "axes.edgecolor": DARK,
                         "xtick.color": DARK, "ytick.color": DARK, "font.family": "DejaVu Sans"})
    units, centers, pairs = load_data(args.source, args.selection)
    mice = units.mouse.unique().tolist()
    plot_pairs(pairs, output, mice)
    plot_maps(units, centers, output, mice)
    pairs.to_csv(output / "adn_double_rf_pairs.csv", index=False)
    counts = units.groupby(["mouse", "n_centers"]).size().unstack(fill_value=0).astype(int)
    counts.to_csv(output / "adn_region_counts.csv")
    assert before == {str(path): digest(path) for path in sources}
    verification = dict(source_hashes=before, script_sha256=digest(Path(__file__)),
                        selection="User-marked ADN, current HD Class 3, measured RF results",
                        measured_neurons=len(units), rf_positive_neurons=int(units.n_centers.gt(0).sum()),
                        centers=len(centers), double_rf_neurons=len(pairs),
                        double_rf_spanning_native_zero=int(pairs.spans_zero.sum()),
                        double_rf_same_side=int((~pairs.spans_zero).sum()),
                        source_files_unchanged=True,
                        coordinates="Native RF X/Y; physical head-forward zero not independently verified",
                        connectivity="Historical saved mask definitions; m14 A671 edge-connection sensitivity marked")
    (output / "plot_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(counts.to_string())
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
