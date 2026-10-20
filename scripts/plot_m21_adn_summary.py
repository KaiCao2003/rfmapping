"""Plot the m21 ADN cohort using session 1 HD and session 3 visual RF results."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


KEYS = ["mouse", "date", "probe", "unit_id"]
BLUE = "#267DB0"
ORANGE = "#DE7A30"
DARK = "#263744"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def style(ax):
    ax.set_facecolor("white")
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#A6B2BA")
    ax.tick_params(labelsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("output/adn_rf_correlations_20261009_with_m21_hd1"))
    args = parser.parse_args()
    sources = [args.source / name for name in ("adn_units.csv", "adn_centers.csv", "correlations.csv")]
    before = {str(path): sha256(path) for path in sources}
    units = pd.read_csv(sources[0], dtype={"date": str})
    centers = pd.read_csv(sources[1], dtype={"date": str})
    correlations = pd.read_csv(sources[2])
    selected = units.loc[units.mouse.eq("m21")].copy()
    measured = selected.loc[selected.hd_class.eq(3) & selected.rf_measured].copy()
    centers = centers.merge(measured[KEYS], on=KEYS, validate="many_to_one")
    assert not measured.duplicated(KEYS).any()
    assert measured.hd_session.eq(1).all() and measured.rf_session.eq(3).all()
    assert len(measured) == 17 and int(measured.n_centers.gt(0).sum()) == 15 and len(centers) == 16
    actual = centers.groupby(KEYS).size()
    expected = measured.loc[measured.n_centers.gt(0)].set_index(KEYS).n_centers.sort_index()
    np.testing.assert_array_equal(actual, expected)
    counts = measured.n_centers.astype(int).value_counts().reindex([0, 1, 2], fill_value=0)
    assert int(counts.sum()) == len(measured)
    stats = correlations.loc[correlations.mouse.eq("m21") &
                             correlations.angle.eq("hd_preferred_deg")].iloc[0]
    assert int(stats.n_units) == len(measured)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False,
                         "text.color": DARK, "axes.labelcolor": DARK,
                         "xtick.color": DARK, "ytick.color": DARK, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.5))
    fig.subplots_adjust(left=.055, right=.985, top=.75, bottom=.29, wspace=.29)
    fig.suptitle("m21 ADN | HD session 1 + visual RF session 3", fontsize=19,
                 fontweight="bold", y=.96)
    fig.text(.5, .884, f"{len(measured)} measured ADN HD Class 3 neurons · "
             "one center per independent connected RF region", ha="center", fontsize=11.5)

    ax = axes[0]
    bars = ax.bar(counts.index, counts, width=.59, color=["#ADB8C1", BLUE, ORANGE])
    ax.bar_label(bars, padding=4, fontsize=13, fontweight="bold")
    ax.set(title="1  How many RF regions?", xlabel="Regions per neuron", ylabel="Number of neurons",
           xticks=[0, 1, 2], ylim=(0, counts.max()*1.22))
    ax.set_yticks(np.arange(0, counts.max()+1, 2))

    ax = axes[1]
    for n, color in ((0, "#8A9AA6"), (1, BLUE), (2, ORANGE)):
        group = measured.loc[measured.n_centers.eq(n)].groupby("hd_preferred_deg").size()
        ax.scatter(group.index, np.full(len(group), n), s=52*group.values,
                   color=color, alpha=.9, edgecolors="white", linewidth=.7)
    ax.set(title="2  HD direction vs. region count", xlabel="HD preferred direction (deg)",
           ylabel="Regions per neuron", xlim=(0, 360), ylim=(-.3, 2.65),
           xticks=[0, 90, 180, 270, 360], yticks=[0, 1, 2])
    ax.text(.04, .94, f"Circular–linear R = {stats.coefficient:.3f}\np = {stats.permutation_p:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=10.5)

    ax = axes[2]
    singles = centers.loc[centers.n_centers.eq(1)]
    points = singles.groupby(["center_x_deg", "center_y_deg"]).size().reset_index(name="n")
    ax.scatter(points.center_x_deg, points.center_y_deg, s=64*points.n, color=BLUE,
               alpha=.82, edgecolors="white", linewidth=.7,
               label=f"Single RF ({len(singles)} neurons)")
    for point in points.loc[points.n.gt(1)].itertuples(index=False):
        ax.annotate(str(point.n), (point.center_x_deg, point.center_y_deg),
                    xytext=(6, 5), textcoords="offset points", fontsize=10, color=DARK)
    doubles = centers.loc[centers.n_centers.eq(2)]
    for identity, pair in doubles.groupby(KEYS):
        pair = pair.sort_values("center_x_deg")
        ax.plot(pair.center_x_deg, pair.center_y_deg, color=ORANGE, linewidth=1.7,
                marker="o", markersize=7, markeredgecolor="white",
                label=f"Double RF (unit {identity[-1]})")
        for row in pair.itertuples(index=False):
            below = row.center_y_deg == pair.center_y_deg.min()
            ax.annotate(f"({row.center_x_deg:g}, {row.center_y_deg:g})",
                        (row.center_x_deg, row.center_y_deg),
                        xytext=(6, -13) if below else (6, 8),
                        textcoords="offset points", va="top" if below else "bottom",
                        fontsize=9, color=ORANGE)
    ax.axvline(0, color="#ACB6BE", linestyle="--", linewidth=.8, zorder=0)
    ax.set(title="3  Where are the RF centers?", xlabel="Native RF X (deg)",
           ylabel="Native RF Y (deg)", xlim=(-180, 180), ylim=(-48, 50),
           xticks=[-180, -90, 0, 90, 180], yticks=[-40, -20, 0, 20, 40])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axes:
        style(ax)
    fig.text(.055, .18, "Zero RF means a measured map with no detected region; excluded or unavailable RF results are not counted as zero.",
             fontsize=10)
    fig.text(.055, .132, "RF centers come from session 3; session 1 contributes HD tuning only. Native coordinate zero is not physically calibrated.",
             fontsize=10)
    fig.text(.055, .084, "Larger blue circles and their labels show coincident single-RF neurons. The orange line joins two centers from the same neuron.",
             fontsize=10)
    fig.text(.055, .036, "HD p: 10,000 within-probe neuron permutations. RF positions and region counts alone do not identify separate visual inputs.",
             fontsize=9.5, color="#576875")
    path = args.source / "m21_hd1_rf3_summary.png"
    fig.savefig(path, dpi=180, facecolor="white", transparent=False)
    plt.close(fig)
    assert before == {str(path): sha256(path) for path in sources}
    verification = dict(source_hashes=before, sources_unchanged=True,
                        script_sha256=sha256(Path(__file__)), figure=str(path),
                        hd_session=1, rf_session=3, measured_neurons=len(measured),
                        rf_positive_neurons=int(measured.n_centers.gt(0).sum()), centers=len(centers),
                        region_counts={str(int(n)): int(value) for n, value in counts.items()},
                        excluded_or_unavailable_rf_ids=selected.loc[~selected.rf_measured, "unit_id"].astype(int).tolist(),
                        non_hd3_ids=selected.loc[~selected.hd_class.eq(3), "unit_id"].astype(int).tolist(),
                        hd_circular_linear_R=float(stats.coefficient), hd_permutation_p=float(stats.permutation_p),
                        coordinates="Native RF X/Y from session 3; physical zero uncalibrated")
    (args.source / "m21_hd1_rf3_summary_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
