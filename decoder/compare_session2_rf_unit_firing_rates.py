"""Mean firing rates of session-2 RF units; run only on hhw9l84."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


root = Path("/mnt/senzailab/Kai/#Recording/m19/260827")
output = Path(__file__).resolve().parent / "session2_rf_unit_firing_rates"
output.mkdir(exist_ok=True)
selection_path = root / "260827_2/data/units_with_rf.npy"
selection = np.load(selection_path, allow_pickle=False)
assert selection.ndim == 2 and selection.shape[1] == 2
assert np.all(selection[:, 0] == "A")
units = selection[:, 1].astype(int)
assert len(units) == len(np.unique(units)) == 20
sessions = [2, 3, 5, 7, 10]
rates = np.empty((len(units), len(sessions)))
records = []

for column, session in enumerate(sessions):
    path = root / f"260827_{session}"
    edges = np.load(path / "data/on_list_times.npy", allow_pickle=False)
    times = np.load(path / "data/probeA/adc_spike_time.npy", mmap_mode="r", allow_pickle=False)
    clusters = np.load(path / f"kilosort/ProbeA/kilosort_{session}/spike_clusters.npy", mmap_mode="r", allow_pickle=False)
    assert times.ndim == clusters.ndim == 1 and times.shape == clusters.shape
    assert edges.ndim == 1 and len(edges) > 1 and np.all(np.diff(edges) > 0)
    assert np.isfinite(times).all() and np.isfinite(edges).all()
    assert np.issubdtype(clusters.dtype, np.integer) and np.all(clusters >= 0)
    in_epoch = (times >= edges[0]) & (times < edges[-1])
    counts = np.bincount(clusters[in_epoch], minlength=int(units.max()) + 1)[units]
    duration = float(edges[-1] - edges[0])
    rates[:, column] = counts / duration
    records.append({"session": session, "start_s": float(edges[0]), "end_s": float(edges[-1]),
        "duration_s": duration, "spike_counts": counts.tolist(), "mean_rate_hz": rates[:, column].tolist()})
    print(f"S{session}: {duration:.3f} s, rates calculated for {len(units)} units", flush=True)

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "text.color": "#111827", "axes.labelcolor": "#111827", "axes.edgecolor": "#374151",
    "xtick.color": "#111827", "ytick.color": "#111827",
    "legend.facecolor": "white", "legend.framealpha": 1.0,
})
fig, ax = plt.subplots(figsize=(8, 10), layout="constrained", facecolor="white")
im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=rates.max(), aspect="auto", interpolation="nearest")
for row, column in np.ndindex(rates.shape):
    value = rates[row, column]
    ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=10,
        color="white" if value > rates.max() * 0.6 else "#111827")
ax.set(xticks=np.arange(len(sessions)), xticklabels=[f"S{s}" for s in sessions],
    yticks=np.arange(len(units)), yticklabels=units, ylabel="Probe A unit",
    title="Firing rate (Hz) · units selected from session 2\nSpikes / full stimulus-epoch duration · all luminances")
ax.xaxis.tick_top()
fig.colorbar(im, ax=ax, shrink=0.7, label="Firing rate (Hz)")
fig.savefig(output / "firing_rates.png", dpi=160, facecolor="white", transparent=False)
plt.close(fig)

result = {"selection_file": str(selection_path), "unit_ids": units.tolist(), "probe": "A",
    "method": "Raw ADC spikes in [first onset, last trial end), divided by that interval's duration. All luminances and positions pooled. No RF response windows or occupancy normalization.",
    "sessions": records}
(output / "firing_rates.json").write_text(json.dumps(result, indent=2) + "\n")
lines = ["# Session 2 RF units 的平均 firing rate", "",
    "固定使用 session 2 保存的 20 个 Probe A RF units。每个 session 从第一个刺激开始到最后一个 trial 结束，FR = spike 数 / 总秒数；包含全部 luminance 和位置。单位：Hz。", "",
    "| Unit | S2 | S3 | S5 | S7 | S10 |", "|---|---:|---:|---:|---:|---:|"]
for unit, values in zip(units, rates):
    lines.append(f"| {unit} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
lines += ["", "![Firing rates](firing_rates.png)"]
(output / "README.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:27]), flush=True)
