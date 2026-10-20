"""Show measured RF response maps and curves for every saved ADN double RF."""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from RF.rflocate import RFMapList, load_rf, load_rfmap
from RF.rflocate._detector import _label_clusters


COLORS = ["#217BB5", "#E58128"]
DARK = "#253441"
KEYS = ["mouse", "date", "probe", "unit_id"]


@dataclass
class Cell:
    mouse: str
    date: str
    probe: str
    unit_id: int
    x: np.ndarray
    y: np.ndarray
    rate: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    labels: np.ndarray
    centers: np.ndarray
    presentations: np.ndarray
    counts: np.ndarray

    @property
    def name(self):
        return f"{self.mouse}_{self.date}_Probe{self.probe}_unit{self.unit_id}"

    @property
    def title(self):
        star = " *" if self.mouse == "m14" and self.unit_id == 671 else ""
        return f"{self.mouse} | Probe {self.probe} | unit {self.unit_id}{star}"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pooled_profile(counts, presentations, duration, axis):
    valid = np.isfinite(counts) & (presentations > 0)
    denominator = np.where(valid, presentations, 0).sum(axis=axis) * duration
    numerator = np.where(valid, counts, 0).sum(axis=axis)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def collect(source, root, selection):
    table = pd.read_csv(source / "adn_units.csv", dtype={"date": str})
    table = table.loc[table.hd_class.eq(3) & table.rf_measured & table.n_centers.eq(2)]
    members = {(r["mouse"], r["date"], r["probe"], uid)
               for r in json.loads(selection.read_text())["recordings"] for uid in r["unit_ids"]}
    assert set(table[KEYS].itertuples(index=False, name=None)) <= members
    cells, records = [], []
    for (mouse, date, probe), group in table.groupby(["mouse", "date", "probe"], sort=False):
        session = int(group.rf_session.iloc[0])
        path = (root / mouse / date / f"{date}_{session}" /
                f"data/rfmapping/good/-100_400_1ms/Probe{probe}" /
                f"regular_unitsSpikeCounts_{date}_{session}.rfmap")
        corrected_path = source / "centers" / f"{mouse}_{date}_Probe{probe}.npz"
        source_hash = sha256(path)
        result_hash = sha256(corrected_path)
        result = load_rf(corrected_path)
        raw = load_rfmap(path)
        ids = set(group.unit_id.astype(int))
        selected = RFMapList([m for m in raw if m.unit_id in ids], path)
        window = tuple(result.manifest["time_range_s"])
        duration = window[1] - window[0]
        rates = selected.to_firing_rate(reconstruct_presentations=True).mean_rate(*window, show_progress=False)
        summed = selected.sum(*window, show_progress=False)
        summed_by_id = {m.unit_id: m for m in summed}
        stored_by_id = {int(uid): i for i, uid in enumerate(result.unit_ids)}
        for rate_map in rates:
            uid = rate_map.unit_id
            response = rate_map.to_2d_array().astype(float)
            counts = summed_by_id[uid].to_2d_array().astype(float)
            presentations = np.asarray(rate_map.presentation_counts)
            expected = np.divide(counts, presentations * duration,
                                 out=np.full_like(counts, np.nan), where=presentations > 0)
            np.testing.assert_allclose(response, expected, equal_nan=True)
            index = stored_by_id[uid]
            mask = result.mask_2d[index].astype(bool)
            center = result.center_2d[index].astype(bool)
            labels, masses = _label_clusters(mask.astype(float), mask,
                                             wrap_x=result.manifest["parameters"]["wrap_x"])
            assert len(masses) == int(center.sum()) == 2
            positions = np.argwhere(center)
            positions = positions[np.argsort(rate_map.x_positions[positions[:, 1]])]
            relabeled = np.zeros_like(labels)
            for number, (iy, ix) in enumerate(positions, 1):
                relabeled[labels == labels[iy, ix]] = number
            horizontal = pooled_profile(counts, presentations, duration, axis=0)
            vertical = pooled_profile(counts, presentations, duration, axis=1)
            valid = np.isfinite(response) & (presentations > 0)
            weighted = np.where(valid, response * presentations, 0).sum(axis=0)
            exposure = np.where(valid, presentations, 0).sum(axis=0)
            np.testing.assert_allclose(horizontal, weighted / exposure, equal_nan=True)
            cells.append(Cell(mouse, date, probe, uid, rate_map.x_positions, rate_map.y_positions,
                              response, horizontal, vertical, relabeled, positions,
                              presentations, counts))
        assert sha256(path) == source_hash and sha256(corrected_path) == result_hash
        records.append(dict(mouse=mouse, date=date, probe=probe, rf_path=str(path), rf_sha256=source_hash,
                            corrected_result=str(corrected_path), corrected_result_sha256=result_hash,
                            window_s=window, n_cells=len(group),
                            presentation_count_source=rates[0].metadata["presentationCountSource"]))
        print(f"Loaded and verified {mouse} Probe {probe}: {len(group)} neurons", flush=True)
    cells.sort(key=lambda c: (c.mouse, c.probe, c.unit_id))
    assert len(cells) == len(table)
    return cells, records


def style(ax, *, grid=False):
    ax.set_facecolor("white")
    for edge in ("top", "right"):
        ax.spines[edge].set_visible(False)
    for edge in ("bottom", "left"):
        ax.spines[edge].set_color("#A9B3BA")
    ax.tick_params(labelsize=9)
    if grid:
        ax.grid(color="#E4E9ED", linewidth=.6)
        ax.set_axisbelow(True)


def draw_row(fig, axes, cell, *, row_title=True):
    x, y = cell.x, cell.y
    dx, dy = np.diff(x).mean(), np.diff(y).mean()
    extent = (x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2)
    heat, regions, curve = axes
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#DDE1E5")
    artist = heat.imshow(cell.rate, origin="lower", extent=extent, aspect="auto",
                         interpolation="nearest", cmap=cmap, vmin=0, vmax=np.nanmax(cell.rate))
    colorbar = fig.colorbar(artist, ax=heat, fraction=.045, pad=.025)
    colorbar.set_label("Hz", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    regions.imshow(cell.labels, origin="lower", extent=extent, aspect="auto",
                   interpolation="nearest", cmap=ListedColormap(["white", *COLORS]), vmin=0, vmax=2)
    for i, (iy, ix) in enumerate(cell.centers):
        regions.scatter(x[ix], y[iy], s=135, color="white", edgecolors=DARK, linewidth=.8, zorder=3)
        regions.text(x[ix], y[iy], str(i+1), color=DARK, ha="center", va="center", fontsize=9, zorder=4)
        curve.axvline(x[ix], color=COLORS[i], linestyle=":", linewidth=1.3, alpha=.8)
    curve.plot(x, cell.horizontal, "o-", color=DARK, markersize=3.5, linewidth=1.4)
    curve.set_ylim(bottom=0)
    curve.set_ylabel("Mean firing rate (Hz)", fontsize=10)
    for ax in axes:
        style(ax, grid=ax is curve)
        ax.set_xlim(-180, 180)
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_xlabel("Native X (deg)", fontsize=10)
        ax.axvline(0, color="#A8B0B8", linewidth=.7, linestyle="--", zorder=0)
        if cell.mouse == "m14":
            for lo, hi in [(-180, -150), (150, 180)]:
                ax.axvspan(lo, hi, color="#EFF1F3", zorder=-1)
    for ax in (heat, regions):
        ax.set_ylim(extent[2], extent[3])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_ylabel("Native Y (deg)", fontsize=10)
    prefix = cell.title + "\n" if row_title else ""
    heat.set_title(prefix + "1  Measured response", loc="left", fontsize=11, fontweight="bold", pad=10)
    regions.set_title("2  Detected regions", loc="left", fontsize=11, fontweight="bold", pad=10)
    curve.set_title("3  Horizontal response", loc="left", fontsize=11, fontweight="bold", pad=10)


def draw_detail(cell, page_number, page_count):
    fig = plt.figure(figsize=(13.5, 9))
    grid = fig.add_gridspec(2, 3, left=.07, right=.97, top=.83, bottom=.16,
                           wspace=.4, hspace=.62, height_ratios=[1.1, .8])
    top = [fig.add_subplot(grid[0, i]) for i in range(3)]
    draw_row(fig, top, cell, row_title=False)
    slices = fig.add_subplot(grid[1, :2])
    vertical = fig.add_subplot(grid[1, 2])
    center_y = cell.centers[:, 0]
    for i, iy in enumerate(center_y):
        if i and iy == center_y[0]:
            continue
        label = f"Region {i+1} center height: Y = {cell.y[iy]:g} deg"
        if center_y[0] == center_y[1]:
            label = f"Both centers at Y = {cell.y[iy]:g} deg"
        slices.plot(cell.x, cell.rate[iy], "o-", color=COLORS[i], linewidth=1.3,
                    markersize=3, label=label)
    slices.set(title="4  Responses at the RF center heights", xlabel="Native X (deg)",
               ylabel="Firing rate (Hz)", xlim=(-180, 180), ylim=(0, None))
    slices.set_xticks([-180, -90, 0, 90, 180])
    slices.legend(fontsize=8.5, frameon=False, loc="best")
    vertical.plot(cell.y, cell.vertical, "o-", color=DARK, linewidth=1.3, markersize=3)
    vertical.set(title="5  Vertical response", xlabel="Native Y (deg)",
                 ylabel="Mean firing rate (Hz)", ylim=(0, None))
    for ax in [slices, vertical]:
        style(ax, grid=True)
        ax.title.set_fontsize(11)
    fig.suptitle("ADN only | " + cell.title, fontsize=18, fontweight="bold", y=.97)
    fig.text(.5, .925, "HD Class 3 | two saved RF regions | response window 0-200 ms | native bins, no smoothing",
             ha="center", fontsize=11)
    fig.text(.07, .101, "1: Color shows measured firing rate.  2: Blue/orange show detected regions; numbered dots are centers.", fontsize=10)
    fig.text(.07, .073, "3: Mean across all measured heights. Dotted lines mark center X, not curve peaks. "
             "4: Individual height slices.  5: Mean across X.", fontsize=9.5)
    note = ("* m14 unit 671 splits into 3 regions without edge wrapping; historical saved grouping is shown."
            if cell.mouse == "m14" and cell.unit_id == 671 else
            "Region count is not a tested peak count. No fit or trial-reliability claim is made; adjacent 200-ms windows overlap.")
    fig.text(.07, .039, note, fontsize=9, color="#59656E")
    fig.text(.97, .039, f"{page_number}/{page_count}", ha="right", fontsize=9, color="#59656E")
    return fig


def draw_overview(cells, output):
    chosen = [("m14", "A", 516), ("m15", "B", 130), ("m19", "A", 263), ("m21", "A", 140)]
    available = {(cell.mouse, cell.probe, cell.unit_id) for cell in cells}
    chosen = [key for key in chosen if key in available]
    examples = [next(c for c in cells if (c.mouse, c.probe, c.unit_id) == key) for key in chosen]
    fig, axes = plt.subplots(len(examples), 3, figsize=(14.5, 3.5 * len(examples)), squeeze=False)
    fig.subplots_adjust(left=.07, right=.975, bottom=.115, top=.865, wspace=.37, hspace=.7)
    fig.suptitle("ADN RF examples | from response to detected regions", fontsize=19, fontweight="bold", y=.97)
    fig.text(.5, .929, "Read each row from left to right. All panels in one row show the same neuron.", ha="center", fontsize=11)
    for row, cell in enumerate(examples):
        draw_row(fig, axes[row], cell)
    fig.text(.07, .06, "Redder = stronger firing. Blue/orange = two detected spatial regions. "
             "Black curve = mean response across all heights.", fontsize=10)
    fig.text(.07, .034, "Native bins; no smoothing. Dotted lines locate region centers. "
             "These geometry examples are not a formal single-/double-peak classification.", fontsize=9.5, color="#59656E")
    fig.savefig(output / "adn_rf_shape_examples.png", dpi=180, facecolor="white", transparent=False)
    fig.savefig(output / "adn_rf_shape_examples.svg", facecolor="white", transparent=False)
    plt.close(fig)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output/adn_rf_correlations_20261009_with_m21_hd1"))
    parser.add_argument("--recording-root", type=Path, default=Path("/mnt/senzailab/Kai/#Recording"))
    parser.add_argument("--selection", type=Path, default=Path("config/retinotopography_adn_units.json"))
    parser.add_argument("--pdf", type=Path, default=Path("output/pdf/adn_double_rf_shapes_20261009_with_m21_hd1.pdf"))
    args = parser.parse_args()
    output = args.source / "response_shapes"
    (output / "units").mkdir(parents=True, exist_ok=True)
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "savefig.transparent": False,
                         "text.color": DARK, "axes.labelcolor": DARK, "axes.edgecolor": DARK,
                         "xtick.color": DARK, "ytick.color": DARK, "font.family": "DejaVu Sans"})
    cells, records = collect(args.source, args.recording_root, args.selection)
    index_rows = []
    with PdfPages(args.pdf) as pdf:
        for page, cell in enumerate(cells, 1):
            fig = draw_detail(cell, page, len(cells))
            png = output / "units" / f"{cell.name}.png"
            fig.savefig(png, dpi=160, facecolor="white", transparent=False)
            pdf.savefig(fig, facecolor="white", transparent=False)
            plt.close(fig)
            np.savez_compressed(output / "units" / f"{cell.name}.npz", x=cell.x, y=cell.y,
                                rate_hz=cell.rate, raw_response_counts=cell.counts,
                                presentation_counts=cell.presentations, horizontal_rate_hz=cell.horizontal,
                                vertical_rate_hz=cell.vertical, region_labels=cell.labels,
                                center_yx_indices=cell.centers)
            index_rows.append(dict(page=page, mouse=cell.mouse, date=cell.date,
                                   probe=cell.probe, unit_id=cell.unit_id, png=str(png),
                                   max_rate_hz=float(np.nanmax(cell.rate)),
                                   x1_deg=float(cell.x[cell.centers[0, 1]]), x2_deg=float(cell.x[cell.centers[1, 1]]),
                                   y1_deg=float(cell.y[cell.centers[0, 0]]), y2_deg=float(cell.y[cell.centers[1, 0]])))
    examples = draw_overview(cells, output)
    pd.DataFrame(index_rows).to_csv(output / "index.csv", index=False)
    verification = dict(n_cells=len(cells), pdf_pages=len(cells), pdf=str(args.pdf),
                        pdf_sha256=sha256(args.pdf), script_sha256=sha256(Path(__file__)),
                        source_records=records, selection_sha256=sha256(args.selection),
                        cohort="ADN selection, current HD Class 3, measured RF, exactly two saved regions",
                        response="Native counts / (presentation count * 0.2 seconds)",
                        horizontal_curve="Sum valid response counts over Y / sum valid presentation exposure over Y",
                        normalization_matches_public_rate_api=True, masks_unchanged=True,
                        smoothing=False, peak_fitting=False, trial_reliability_tested=False,
                        examples=examples, example_selection="Geometry illustrations, not a random or inferential sample",
                        notes=["RF region count need not equal horizontal peak count",
                               "Native axes are not independently physically calibrated",
                               "Adjacent stimulus response windows overlap",
                               "m14 A671 historical wrap_x grouping is explicitly marked"])
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Saved {len(cells)} unit figures and {args.pdf}", flush=True)


if __name__ == "__main__":
    main()
