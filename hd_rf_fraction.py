"""Print HD with 1D RF, 2D RF, or both / all HD, and plot a pie chart.

HD means class 3. Read saved RF detections; missing 1D results are calculated
in memory on the saved 2D QC unit pool. Run on hhw9l84 with the rfmapping venv.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from Utils.direction_comparison import load_tc
from Utils.plotting import LIGHT_PLOT_STYLE
from Utils.rfmap import RFMapList, load_rf_maps


ROOT = Path("/mnt/senzailab/Kai/#Recording")
# Mouse, date, HD session, RF session, probes; matches tc_comparison_pairs.ipynb.
RECORDINGS = [
    ("m14", 260609, 1, 3, ("A",)),
    ("m15", 260630, 1, 3, ("A", "B")),
    ("m19", 260827, 9, 2, ("A",)),
    ("m20", 260921, 9, 2, ("A",)),
]
HD_CLASS = 3
RF_WINDOW_S = (0.0, 0.2)
OUTPUT_FILE = Path("output/hd_rf_fraction.png")


def rf_unit_ids(rf_file):
    with np.load(rf_file.with_suffix(".npz"), allow_pickle=False) as result:
        qc_ids = result["unit_ids"]
        ids_2d = set(qc_ids[result["mask_2d"].any(axis=(1, 2))].tolist())

    rf_1d_file = rf_file.with_name(f"{rf_file.stem}_1d.npz")
    if rf_1d_file.exists():
        with np.load(rf_1d_file, allow_pickle=False) as result:
            ids_1d = set(result["unit_ids"][result["mask_2d"].any(axis=(1, 2))].tolist())
    else:
        summed = load_rf_maps(rf_file).sum(*RF_WINDOW_S, show_progress=False)
        qc_ids = set(qc_ids.tolist())
        summed = RFMapList([rf_map for rf_map in summed if rf_map.unit_id in qc_ids], rf_file)
        # Match the saved 1D detections: collapse x, mean + 1 SD, components >2 bins.
        mask = summed.rf_1d(
            is_shuffle=False, cluster_forming_z=1.0, drop_bins=2,
            wrap_x=True, show_progress=False,
        )
        ids_1d = set(np.asarray(summed.unit_ids)[mask.any(axis=1)].tolist())
    return ids_1d, ids_2d


def print_counts(label, hd, with_1d, with_2d):
    both = with_1d & with_2d
    for name, units in (("1D", with_1d), ("2D", with_2d), ("both", both)):
        percentage = f" ({len(units) / len(hd):.1%})" if hd else ""
        print(f"{label}: with {name} RF / all HD = {len(units)} / {len(hd)}{percentage}")


def main():
    all_hd = set()
    hd_with_1d = set()
    hd_with_2d = set()
    for mouse, date, hd_session, rf_session, probes in RECORDINGS:
        for probe in probes:
            recording = dict(mouse=mouse, date=date, probe=probe)
            base = ROOT / mouse / str(date)
            hd_file = base / f"{date}_{hd_session}/data/tuning_curves/Probe{probe}/tuning_curves.tc"
            rf_file = base / f"{date}_{rf_session}/data/rfmapping/good/-100_400_1ms/Probe{probe}/regular_unitsSpikeCounts_{date}_{rf_session}.rfmap"
            hd = set(load_tc(hd_file, **recording, hd_class=HD_CLASS).index)
            ids_1d, ids_2d = rf_unit_ids(rf_file)
            with_1d = {key for key in hd if key[-1] in ids_1d}
            with_2d = {key for key in hd if key[-1] in ids_2d}
            print_counts(f"{mouse} {date} Probe{probe}", hd, with_1d, with_2d)
            # Full unit keys keep equal numeric IDs on different probes distinct.
            all_hd.update(hd)
            hd_with_1d.update(with_1d)
            hd_with_2d.update(with_2d)

    print_counts("Total", all_hd, hd_with_1d, hd_with_2d)
    if not all_hd:
        return

    counts = [
        len(hd_with_1d - hd_with_2d), len(hd_with_2d - hd_with_1d),
        len(hd_with_1d & hd_with_2d), len(all_hd - (hd_with_1d | hd_with_2d)),
    ]
    labels = [f"{name}\nn = {count}" for name, count in zip(
        ("1D only", "2D only", "Both 1D + 2D", "No RF"), counts, strict=True,
    )]
    with plt.rc_context(LIGHT_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
        ax.pie(
            counts, labels=labels,
            colors=["#4c78a8", "#f2a541", "#72b7a1", "#d9dde3"],
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "black"}, wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title(f"RF detections in HD class {HD_CLASS} units (n = {len(all_hd)})")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_FILE, dpi=180, facecolor="white", transparent=False)
        plt.show()


if __name__ == "__main__":
    main()
