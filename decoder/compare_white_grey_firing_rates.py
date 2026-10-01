"""Trial-luminance firing rates of session-2 RF units; run on hhw9l84."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


root = Path("/mnt/senzailab/Kai/#Recording/m19/260827")
output = Path(__file__).resolve().parent / "session2_rf_unit_firing_rates"
selection = np.load(root / "260827_2/data/units_with_rf.npy", allow_pickle=False)
assert selection.shape == (20, 2) and np.all(selection[:, 0] == "A")
units = selection[:, 1].astype(int)
assert len(np.unique(units)) == 20
previous = json.loads((output / "firing_rates.json").read_text())
assert units.tolist() == previous["unit_ids"]
expected_luminances = {2: [0, 1], 3: [1], 5: [1], 7: [1], 10: [0.5, 1]}
records = []

for session, expected in expected_luminances.items():
    path = root / f"260827_{session}"
    trials = loadmat(path / "260827.mat", simplify_cells=True)["trials"]
    luminance = np.array([trial["Square_Luminance"] for trial in trials])
    assert np.array_equal(np.unique(luminance), expected)
    edges = np.load(path / "data/on_list_times.npy", allow_pickle=False)
    times = np.load(path / "data/probeA/adc_spike_time.npy", mmap_mode="r", allow_pickle=False)
    clusters = np.load(path / f"kilosort/ProbeA/kilosort_{session}/spike_clusters.npy", mmap_mode="r", allow_pickle=False)
    assert times.ndim == clusters.ndim == 1 and times.shape == clusters.shape
    assert edges.ndim == 1 and len(edges) == len(luminance) + 1
    assert np.isfinite(times).all() and np.isfinite(edges).all() and np.all(np.diff(edges) > 0)
    counts = np.array([np.diff(np.searchsorted(np.sort(times[clusters == unit]), edges, side="left")) for unit in units])
    baseline = next(record for record in previous["sessions"] if record["session"] == session)
    np.testing.assert_array_equal(counts.sum(axis=1), baseline["spike_counts"])
    durations = np.diff(edges)
    for value in expected:
        selected = luminance == value
        spike_counts = counts[:, selected].sum(axis=1)
        duration = float(durations[selected].sum())
        records.append({"session": session, "luminance": value,
            "trial_count": int(selected.sum()), "duration_s": duration,
            "spike_counts": spike_counts.tolist(), "mean_rate_hz": (spike_counts / duration).tolist()})

white = next(record for record in records if record["session"] == 10 and record["luminance"] == 1)
grey = next(record for record in records if record["session"] == 10 and record["luminance"] == 0.5)
assert white["trial_count"] == grey["trial_count"] == 6000
rates = np.column_stack([white["mean_rate_hz"], grey["mean_rate_hz"]])
delta = rates[:, 0] - rates[:, 1]

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "text.color": "#111827", "axes.labelcolor": "#111827", "axes.edgecolor": "#374151",
    "xtick.color": "#111827", "ytick.color": "#111827",
    "legend.facecolor": "white", "legend.framealpha": 1.0,
})
fig, axes = plt.subplots(1, 2, figsize=(9, 10), width_ratios=[2, 1], sharey=True,
    layout="constrained", facecolor="white")
fig.suptitle("Session 10 · white / grey trial firing rates\nSession-2 RF units · 6000 trials per condition", fontsize=14)
for ax, data, labels, cmap, low, high in [
    (axes[0], rates, ["White (lum=1)", "Grey (lum=0.5)"], "Blues", 0, rates.max()),
    (axes[1], delta[:, None], ["White − grey"], "RdBu_r", -np.abs(delta).max(), np.abs(delta).max()),
]:
    im = ax.imshow(data, cmap=cmap, vmin=low, vmax=high, aspect="auto", interpolation="nearest")
    for row, column in np.ndindex(data.shape):
        value = data[row, column]
        label = f"{value:+.2f}" if ax is axes[1] else f"{value:.2f}"
        ax.text(column, row, label, ha="center", va="center", fontsize=10,
            color="white" if abs(value) > high * 0.6 else "#111827")
    ax.set(xticks=np.arange(data.shape[1]), xticklabels=labels,
        yticks=np.arange(len(units)), yticklabels=units)
    ax.xaxis.tick_top()
    fig.colorbar(im, ax=ax, shrink=0.7, label="ΔFR (Hz)" if ax is axes[1] else "FR (Hz)")
axes[0].set_ylabel("Probe A unit")
fig.savefig(output / "white_grey_s10.png", dpi=160, facecolor="white", transparent=False)
plt.close(fig)

result = {"unit_ids": units.tolist(), "probe": "A", "sessions": records,
    "method": "For each luminance: sum raw spikes in its half-open [trial onset, next onset) intervals, divided by the sum of those intervals' durations. All positions and consecutive same-luminance trials pooled; no latency shift or response-window selection.",
    "session10_white_minus_grey_hz": delta.tolist(),
    "condition_note": "S2: white (1) and black (0); S3/S5/S7: white only; S10: white (1) and grey (0.5). Black trials are not relabeled grey. Differences are descriptive means, not significance or decoding results."}
(output / "by_luminance.json").write_text(json.dumps(result, indent=2) + "\n")
lines = ["# 每个 unit 在 white / grey trial 的 FR", "",
    "这组 session 中只有 session 10 同时有 white（lum=1）和 grey（lum=0.5），各 6000 trials，累计时长分别为 "
    f"{white['duration_s']:.6f} s 和 {grey['duration_s']:.6f} s。Session 2 是 white/black，session 3/5/7 只有 white。", "",
    "FR = 该条件下的总 spike 数 / 该条件的总时长。每个 trial 使用 [onset, next onset)，合并全部位置和连续同 luminance 的 trials。以下均为 Hz；ΔFR = white − grey，是描述性均值差。", "",
    "| Unit | White FR | Grey FR | ΔFR |", "|---|---:|---:|---:|"]
for unit, (white_rate, grey_rate), difference in zip(units, rates, delta):
    lines.append(f"| {unit} | {white_rate:.2f} | {grey_rate:.2f} | {difference:+.2f} |")
lines += ["", "![White and grey firing rates](white_grey_s10.png)", "",
    "[全部 session 按实际 luminance 分组的计数、时长和 FR](by_luminance.json)"]
(output / "white_grey.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines[:29]), flush=True)
