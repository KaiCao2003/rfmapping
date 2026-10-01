"""Raw firing-rate and spike-timing comparison; run only on hhw9l84."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


root = Path("/mnt/senzailab/Kai/#Recording/m19/260827")
sessions = [3, 5, 7, 10]
unit_id = 243
output_dir = Path(__file__).resolve().parent / "unit243_session_comparison"
output_dir.mkdir(exist_ok=True)
colors = ["#2563eb", "#c2690c", "#198754", "#8b3fc4"]
plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "text.color": "#111827", "axes.labelcolor": "#111827",
    "axes.edgecolor": "#374151", "xtick.color": "#111827", "ytick.color": "#111827",
    "legend.facecolor": "white", "legend.framealpha": 1.0, "font.size": 11,
})

recordings, summary = [], []
for session in sessions:
    path = root / f"260827_{session}"
    kilosort = path / f"kilosort/ProbeA/kilosort_{session}"
    trials = loadmat(path / "260827.mat", simplify_cells=True)["trials"]
    luminance = np.array([trial["Square_Luminance"] for trial in trials])
    edges = np.load(path / "data/on_list_times.npy", allow_pickle=False)
    times = np.load(path / "data/probeA/adc_spike_time.npy", mmap_mode="r", allow_pickle=False)
    clusters = np.load(kilosort / "spike_clusters.npy", mmap_mode="r", allow_pickle=False)
    labels = pd.read_csv(kilosort / "cluster_KSLabel.tsv", sep="\t").set_index("cluster_id")
    assert labels.loc[unit_id, "KSLabel"] == "good"
    assert times.ndim == clusters.ndim == 1 and times.shape == clusters.shape
    assert len(edges) == len(luminance) + 1 and np.all(np.diff(edges) > 0)
    absolute_spikes = np.sort(times[clusters == unit_id])
    assert np.isfinite(absolute_spikes).all() and np.isfinite(edges).all()
    spikes = absolute_spikes[(absolute_spikes >= edges[0]) & (absolute_spikes < edges[-1])] - edges[0]
    duration = float(edges[-1] - edges[0])
    isi_ms = np.diff(spikes) * 1000
    assert len(isi_ms) > 0 and np.all(isi_ms > 0), "ISI plot requires distinct spike times."

    # Full stimulus epoch, with nonoverlapping 1-second bins and the final partial bin.
    bin_edges = np.r_[np.arange(0, duration, 1.0), duration]
    bin_duration = np.diff(bin_edges)
    counts = np.diff(np.searchsorted(spikes, bin_edges, side="left"))
    firing_rate = counts / bin_duration
    assert counts.sum() == len(spikes)
    trial_counts = np.diff(np.searchsorted(absolute_spikes, edges, side="left"))
    assert trial_counts.sum() == len(spikes)
    white = luminance == 1.0
    white_rate = trial_counts[white].sum() / np.diff(edges)[white].sum()
    complete = np.isclose(bin_duration, 1.0, rtol=0, atol=1e-9)
    summary.append({
        "session": session, "duration_s": duration, "spikes": len(spikes),
        "mean_rate_hz": len(spikes) / duration, "white_only_rate_hz": float(white_rate),
        "median_isi_ms": float(np.median(isi_ms)),
        "isi_under_10ms_percent": float(np.mean(isi_ms < 10) * 100),
        "empty_1s_bins_percent": float(np.mean(counts[complete] == 0) * 100),
        "max_1s_rate_hz": float(firing_rate[complete].max()),
        "luminance_trial_counts": {str(float(v)): int((luminance == v).sum()) for v in np.unique(luminance)},
    })
    recordings.append({"session": session, "duration": duration, "spikes": spikes,
        "isi_ms": isi_ms, "bin_edges": bin_edges, "firing_rate": firing_rate})

# Full recording: each panel has its actual time span and the same firing-rate scale.
fig, axes = plt.subplots(4, 1, figsize=(12, 8.5), sharey=True, layout="constrained", facecolor="white")
ymax = np.ceil(max(r["firing_rate"].max() for r in recordings) * 1.05 / 10) * 10
for ax, rec, row, color in zip(axes, recordings, summary, colors):
    ax.stairs(rec["firing_rate"], rec["bin_edges"], baseline=None, color=color, linewidth=1)
    ax.axhline(row["mean_rate_hz"], color="#64748b", linestyle="--", linewidth=0.8)
    ax.set(xlim=(0, rec["duration"]), ylim=(0, ymax), ylabel="Spikes/s",
        xlabel="Time since first stimulus (s)",
        title=f"Session {rec['session']}  |  mean {row['mean_rate_hz']:.2f} Hz  |  {row['spikes']:,} spikes")
fig.suptitle("Unit 243 · Probe A · full stimulus epochs · 1 s bins · common rate scale", fontsize=14)
fig.savefig(output_dir / "firing_rate.png", dpi=150, facecolor="white", transparent=False)
plt.close(fig)

# An a priori fixed window shows individual spikes; ISIs use each entire session.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained", facecolor="white")
for i, (rec, color) in enumerate(zip(recordings, colors)):
    example = rec["spikes"][rec["spikes"] < 30]
    axes[0].eventplot(example, lineoffsets=3-i, linelengths=0.65, linewidths=0.45, colors=color)
    intervals = np.sort(rec["isi_ms"])
    axes[1].step(intervals, 100 * np.arange(1, len(intervals)+1) / len(intervals),
        where="post", color=color, label=f"S{rec['session']}", linewidth=1.6)
axes[0].set(xlim=(0, 30), ylim=(-0.6, 3.6), yticks=[3, 2, 1, 0],
    yticklabels=["Session 3", "Session 5", "Session 7", "Session 10"],
    xlabel="Time since first stimulus (s)", title="Individual spikes · first 30 s in every session")
axes[1].set(xscale="log", ylim=(0, 102), xlabel="Interspike interval (ms, log scale)",
    ylabel="Intervals at or below this duration (%)", title="Spike timing · full-session ISI distributions")
axes[1].axvline(10, color="#94a3b8", linestyle=":", label="10 ms")
axes[1].legend()
fig.suptitle("Unit 243 · raw spike patterns", fontsize=14)
fig.savefig(output_dir / "spike_patterns.png", dpi=150, facecolor="white", transparent=False)
plt.close(fig)

result = {"unit_id": unit_id, "probe": "A", "date": "260827", "sessions": summary,
    "method": "Original ADC spike times; [first onset, last trial end); no smoothing or fitted model. Rate traces use 1 s bins and a duration-corrected final partial bin. Raster window is 0–30 s, fixed for all sessions. ISIs are computed within each full session. Short-ISI percentage is descriptive, not a burst classification.",
    "stimulus_note": "Sessions 3/5/7 contain only lum=1 trials; session 10 alternates lum=1 and lum=0.5. White-only rates are also reported."}
(output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
print(f"Saved figures to {output_dir}")
