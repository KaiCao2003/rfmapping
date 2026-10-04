"""Compare saved session-2 RF units across sessions 2/3/5/7/10 on hhw9l84."""
from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
from Utils.rflocate import load_rf, load_rfmap


root = Path("/mnt/senzailab/Kai/#Recording/m19/260827")
output = repo / "decoder/session2_rf_comparison"
output.mkdir(exist_ok=True)
selection_path = root / "260827_2/data/units_with_rf.npy"
selected = np.load(selection_path, allow_pickle=False)
assert selected.shape[1] == 2 and np.all(selected[:, 0] == "A")
units = selected[:, 1].astype(int).tolist()
sessions = [2, 3, 5, 7, 10]
files = {
    2: "regular_unitsSpikeCounts_260827_2.rfmap",
    3: "regular_unitsSpikeCounts_260827_3_vertical_bar_pooled_bin3deg.rfmap",
    5: "rotation_30_unitsSpikeCounts_260827_5.rfmap",
    7: "rotation_30_unitsSpikeCounts_260827_7.rfmap",
    10: "regular_unitsSpikeCounts_260827_10free_moving.rfmap",
}
titles = {2: "S2 · square / screen", 3: "S3 · bar / screen",
    5: "S5 · square / rotation", 7: "S7 · square / rotation", 10: "S10 · bar / ego"}
rf_dir = "data/rfmapping/good/-100_400_1ms/ProbeA"
cache_path = root / "260827_2" / rf_dir / "regular_unitsSpikeCounts_260827_2.npz"
detected_rf = load_rf(cache_path)
cache_units, centers = detected_rf.unit_ids, detected_rf.center_2d
assert np.array_equal(cache_units[np.any(centers, axis=(1, 2))], units)
center_index = {int(u): np.argwhere(c)[0] for u, c in zip(cache_units, centers) if int(u) in units}
selection_manifest = dict(detected_rf.manifest)

plt.style.use("default")
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "savefig.transparent": False,
    "text.color": "#111827", "axes.labelcolor": "#111827", "axes.edgecolor": "#374151",
    "xtick.color": "#111827", "ytick.color": "#111827", "font.size": 9,
    "legend.facecolor": "white", "legend.framealpha": 1.0,
})

maps, provenance = {}, []
for session in sessions:
    path = root / f"260827_{session}" / rf_dir / files[session]
    source = load_rfmap(path).to_firing_rate(reconstruct_presentations=True)
    missing = sorted(set(units) - set(source.unit_ids))
    assert missing == ([436] if session == 10 else []), (session, missing)
    if session == 10:
        clusters = np.load(root / "260827_10/kilosort/ProbeA/kilosort_10/spike_clusters.npy", mmap_mode="r")
        assert np.count_nonzero(clusters == 436) == 0
    provenance.append({"session": session, "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "missing_units": missing, "shape": list(source[0].shape),
        "x_positions": source[0].x_positions.tolist(), "y_positions": source[0].y_positions.tolist()})
    for unit in units:
        if unit in missing:
            maps[unit, session] = None
            continue
        rf = source.by_unit_id(unit)
        assert np.all(np.isfinite(rf.spike_counts))
        maps[unit, session] = {
            "x": rf.x_positions.copy(), "y": rf.y_positions.copy(),
            "response": rf.mean_rate(0.0, 0.2).to_2d_array().copy(),
            "contrast": (rf.mean_rate(0.08, 0.16) - rf.mean_rate(0.0, 0.08)).to_2d_array().copy(),
        }
    del source
    print(f"Loaded S{session}: {len(units) - len(missing)} selected RF maps", flush=True)


def scale_for(unit, kind):
    values = [maps[unit, s][kind] for s in sessions if maps[unit, s] is not None]
    upper = max(float(np.max(np.abs(v))) for v in values)
    assert upper > 0, (unit, kind)
    return Normalize(vmin=-upper if kind == "contrast" else 0, vmax=upper)


def draw(ax, unit, session, kind, norm):
    data = maps[unit, session]
    if data is None:
        ax.text(0.5, 0.5, "No recorded spikes\nUnit 436 omitted\nfrom RF source",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#374151")
        ax.set_axis_off()
        return
    x, y, values = data["x"], data["y"], data[kind]
    if len(y) == 1:
        # A vertical-bar map has no elevation resolution: display its 1D profile.
        ax.plot(x, values[0], color="#7c3aed", linewidth=1.2)
        ax.set_ylim(norm.vmin, norm.vmax)
        ax.set_ylabel("Response")
        ax.axhline(0, color="#9ca3af", linewidth=0.5)
    else:
        dx, dy = np.diff(x)[0], np.diff(y)[0]
        ax.imshow(values, origin="lower", aspect="auto", interpolation="nearest",
            extent=(x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2),
            norm=norm, cmap="RdBu_r" if kind == "contrast" else "viridis")
        ax.set_yticks([-39, -3, 33])
        ax.set_ylabel("Elevation (deg)")
        if session == 2:
            iy, ix = center_index[unit]
            ax.plot(x[ix], y[iy], "x", color="white", markeredgewidth=1.4, markersize=6)
    ax.set(xlim=(-180, 180), xticks=[-180, 0, 180], xlabel="Source azimuth (deg)")
    ax.tick_params(labelsize=8)


kind_labels = {"response": "0–200 ms response", "contrast": "80–160 ms minus 0–80 ms"}
for unit in units:
    fig, axes = plt.subplots(2, 5, figsize=(16, 5.7), layout="constrained", facecolor="white")
    for row, kind in enumerate(["response", "contrast"]):
        norm = scale_for(unit, kind)
        for col, session in enumerate(sessions):
            draw(axes[row, col], unit, session, kind, norm)
            axes[row, col].set_title(titles[session] + "\n" + kind_labels[kind], fontsize=9)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,
            cmap="RdBu_r" if kind == "contrast" else "viridis"), ax=axes[row, :], shrink=0.84)
        colorbar.set_label("Mean firing rate (Hz)", fontsize=8)
    fig.suptitle(f"Probe A · unit {unit} · selected by session 2's saved RF flag\n"
        "Within each row: same amplitude scale across sessions; native spatial grids; white x = saved S2 RF center", fontsize=11)
    fig.savefig(output / f"unit_{unit}.png", dpi=140, facecolor="white", transparent=False)
    plt.close(fig)

for kind in ["response", "contrast"]:
    for page, start in enumerate(range(0, len(units), 5), 1):
        page_units = units[start:start+5]
        fig, axes = plt.subplots(len(page_units), 5, figsize=(16, 2.4*len(page_units)),
            layout="constrained", squeeze=False, facecolor="white")
        for row, unit in enumerate(page_units):
            norm = scale_for(unit, kind)
            for col, session in enumerate(sessions):
                draw(axes[row, col], unit, session, kind, norm)
                axes[row, col].set_title(f"Unit {unit} · {titles[session]}", fontsize=9)
            colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm,
                cmap="RdBu_r" if kind == "contrast" else "viridis"), ax=axes[row, :], shrink=0.75)
            colorbar.ax.tick_params(labelsize=8)
        fig.suptitle(f"Session-2 RF units · {kind_labels[kind]} · page {page}/4\n"
            "Each unit uses one amplitude scale across sessions · units: Hz", fontsize=12)
        fig.savefig(output / f"{kind}_page_{page}.png", dpi=140, facecolor="white", transparent=False)
        plt.close(fig)

manifest = {"selected_units": units, "selection_file": str(selection_path),
    "selection_manifest": selection_manifest, "sources": provenance,
    "response_window_s": [0, 0.2], "contrast_windows_s": [[0.08, 0.16], [0, 0.08]],
    "normalization": "Mean firing rate in each named window: raw response counts divided by stimulus presentation count times response-window duration. Legacy presentation counts are explicitly reconstructed when absent. The contrast subtracts the two window means in Hz.",
    "comparison": "ON maps only. Native spatial grids and source coordinates are preserved. S2: screen-position squares; S3: full-height bars in 3-degree coverage bins; S5/S7: rotation-adjusted squares; S10: free-moving head-relative bars. No fitted spatial alignment, interpolation, smoothing, or new RF detection.",
    "missing": "Unit 436 has zero entries in S10 spike_clusters.npy and no map in the S10 RF source; it is explicitly annotated, not replaced with another map.",
    "interpretation": "The selected RF flags came from pooled-spatial-z detection with is_shuffle=false. The late-minus-early map is a timing contrast, not a significance test. Responses after 100 ms can overlap later stimuli; negative/early responses can include earlier stimuli."}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
lines = ["# Session 2 标记的 RF units：跨 session 对比", "",
    "参考 `260827_2/data/units_with_rf.npy` 中的 20 个 Probe A units；该列表与保存的 RF center cache 一致。", "",
    "主图为 0–200 ms，与保存 RF 标记的时间窗一致；另附 80–160 ms 减 0–80 ms，对应现有 notebook 的时间对比。同一个 unit、同一种图在全部 session 共用幅度范围。", "",
    "方块刺激显示二维 RF；竖条刺激显示一维水平曲线，不能判断垂直位置。S5/S7 使用 rotation 文件，S10 使用 free-moving 文件，保留各自坐标，不拟合位移。不同刺激几何和坐标处理限制直接的 RF 形状/幅度等同性判断。", "",
    "颜色与一维曲线均为窗口平均 firing rate（Hz）：窗口内 spike counts /（stimulus presentation count × 窗口长度）。未将每张图单独归一化到峰值。白色叉号为 S2 已保存的 RF center。", "",
    "Unit 436 在 session 10 原始 spike_clusters 中没有事件，图上明确标注。现有 RF 标记使用非 shuffle 的 pooled-spatial-z 筛选；本次只比较图，不重新判定统计显著性。", "",
    "## 图上观察", "",
    "- 135、136、267：S2 的局部响应在 S5/S7 仍出现在相近的正 azimuth、低 elevation 区域；S10 的晚减早响应较弱。",
    "- 243、261：S2 有清楚的局部 RF，S3 的正 azimuth 区域有响应峰，S5 仍可见局部响应，S7 变弱；S10 整体响应升高，但水平曲线较平，原来的局部峰不明显。",
    "- 263：S2/S3/S5 在负 azimuth 一侧可见局部响应，S7 较弱；S10 的晚减早曲线接近零。",
    "- 104、111、154：虽然在保存的 RF 列表中，这两种时间窗下的空间图较零散，缺少清楚、连续的局部响应。", "",
    "以上是对现有图的描述，不是新的 RF 检测结果，也不能据此认定 S10 没有视觉信息。", "",
    "## 全部 units", ""]
for page in range(1, 5):
    lines.append(f"- 第 {page} 页：[0–200 ms](response_page_{page}.png) · [80–160 减 0–80 ms](contrast_page_{page}.png)")
lines += ["", "## 单个 unit", "", "| Unit | S2 已保存 RF center (azimuth, elevation) | 对比图 |", "|---|---|---|"]
for unit in units:
    iy, ix = center_index[unit]
    data = maps[unit, 2]
    lines.append(f"| {unit} | ({data['x'][ix]:g}°, {data['y'][iy]:g}°) | [查看](unit_{unit}.png) |")
lines += ["", "分析脚本：[compare_session2_rf_units.py](../compare_session2_rf_units.py)", "",
    "时间窗内的 spikes 可包含相邻刺激的响应；短窗口图中不明显，不等于该 unit 在整个 session 没有 RF。"]
(output / "README.md").write_text("\n".join(lines) + "\n")
print(f"Saved 20 unit figures and 8 overview pages to {output}", flush=True)
