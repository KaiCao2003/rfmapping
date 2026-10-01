# 三仓库角度审计 · 2026-09-16

结论：**没有全部 align。当前二维 FM RF → minimal GLM 的刺激几何在代码和保存输入层面基本对齐；通用 bearing/EBC 与 RF 使用相反正方向，必须显式转换；HD 零点、旧 GLM 入口、部分 GUI 和导出仍不一致。另有一个已复现的 peak alignment 符号错误。RF 刺激脚本与 Motive 的物理对应，还不能仅凭这条二维链路认定已经完成标定。**

本次只调查和验证，没有修改分析、GUI、原始数据或既有结果。覆盖 `rfmapping`、`rfmapping_gui`、`glm` 当前有关角度的 Python/Swift/TypeScript 源码和 31 个 notebook 的源码；历史报告用于辨认旧入口，没有把其快照当成现行实现。MATLAB 以 `hhw9l84:/mnt/ssd4.1/Matlab` 为准。数值计算和 pytest 均在 `hhw9l84` 的 `~/.virtualenvs/rfmapping` 运行。

“对齐”需要分开判断：角度的物理对象、参考坐标系、零点、正方向、单位、采样时刻，以及最终画图方式。两张图都写 −180°～180°，不代表这些条件相同。

## 0. 你指定的两个源头

### RF：LED bar 实际生成脚本

已读取你指定的 [ReceptiveFieldMapping_bar_360degLED.m](/Volumes/SenzaiLab/Kai/MatlabApps/stimulus-delivery/StimGenStimuli/ReceptiveFieldMapping_bar_360degLED.m:235)，并确认它与远端 `/mnt/senzailab/Kai/MatlabApps/stimulus-delivery/StimGenStimuli/` 同名文件的 SHA-256 一致。

[monitorInformation.m](/Volumes/SenzaiLab/Kai/MatlabApps/stimulus-delivery/RigSpecificInfo/monitorInformation.m:29) 规定横向 960 pixels = 360°，所以每 pixel 为 0.375°。脚本的实际公式是：

```text
S = trials(trial).Square_PositionX       # degrees
bar_width_pixels = ceil(Square_Size / 0.375)
bar_center_column = (960 + 1)/2 + S/0.375
```

| 输入 S | 纹理中的中心列，MATLAB 1-based | 含义 |
|---|---|---|
| −90° | 240.5 | 中心左侧 |
| 0° | 480.5 | 整幅纹理正中 |
| +90° | 720.5 | 中心右侧 |

这证明的是 **S 增加 → 二维纹理向右移动**。`DrawTextures` 没有再传旋转角或做镜像；这个脚本本身不含 Motive、动物 HD 或零点标定。它也没有跨纹理边缘的 modulo wrap，超出纹理的部分不绘制；标准 30 个中心、12°宽 bar 刚好覆盖 960 列。

**这里的 `Square_PositionY` 虽然被读入，但不参与放置。** 第 260–261 行让 bar 覆盖 1…240 的全部行，因此这条刺激没有独立垂直位置。邻近的 [Fast square 脚本](/Volumes/SenzaiLab/Kai/MatlabApps/stimulus-delivery/StimGenStimuli/ReceptiveFieldMapping_Fast_360degLED.m:250) 则把正 `Square_PositionY` 加到图像行号，意味着纹理上向下，不能直接叫作“向上为正的 elevation”。

纹理向右，在实际围成一圈的 LED 上对应哪个 Motive 世界方向，还依赖 LED 的安装/映射和屏幕零点。当前分析用 `world_angle = z-S` 作这个连接；源码一致只能证明大家用了同一假设，不能独立测定安装方向。另一个版本边界：指定 bar 文件当前修改日期为 2026-09-11；本次原始数据样本是 2026-08-27，文件相同不能证明它就是当日运行版本。

### Motive：CSV 姿态输出与分析使用的 HD

session 10 的原始 CSV header 是 `Rotation Type=XYZ`、`Coordinate Space=Global`、120 Hz、148,147 frames。实测 `processed/head_direction.json` 的 HD 与 rigid body 的 **Rotation Z** 相同，最大 circular 差只有 **4.55×10⁻¹³°**。

因此当前三个仓库所用的 H，是一个 **rigid-body Euler 分量**。它不是从 `atan2(marker_y,marker_x)` 算出的头方向，也不是已经绕 LED 零点校准过的角度。本地上游 [compute_head_direction()](/Users/vxf1610/Developer/Analysis/analysis/core/kinematics.py:164) 也显示：Euler 输入按所选轴取一列、unwrap、最后 `%360`；没有 RF 屏幕校准。这里的源码用于解释行为，原始 CSV/JSON 的数值对应才是这份数据的直接证据。

Motive 官方允许配置 CSV 轴映射与 global/local 导出，Euler 使用右手系且旋转顺序可选。因此不能仅凭字段名字把 `Rotation Z` 无条件等同于物理 yaw；必须结合实际导出轴和 rigid-body local forward。[OptiTrack CSV 文档](https://docs.optitrack.com/motive/data-export/data-export-csv)

此外，如果“tracking output”指的是 sibling `tracking` 的实时控制台，而非这里的 CSV/JSON，它的 [Head angle 计算](/Users/vxf1610/Developer/tracking/VisualizationManager.m:125) 是 `v=R*[1,0,0]` 后 `atan2(v_y,v_x)`。这是投影后的 local +X 方向，与直接取 Euler Z 不是普遍相等的定义；本报告的主要数据对照针对 CSV/JSON。

## 1. 先建立一份角度字典

记 `wrap180(a) = (a + 180) % 360 - 180`，`wrap360(a) = a % 360`。以下公式的角度均为度；进入 sin/cos/atan2 时才换成弧度。

| 名称 | 当前含义 | 零点和正方向 | 主要位置 |
|---|---|---|---|
| 原始 HD，H | `processed/head_direction.json` 的 `head_direction_deg` | Motive 路径实际等于 CSV `Rotation Z`；legacy 几何把它当作世界 XY 的朝向使用 | `tuning_curves.ipynb`、`simple_hd_glm.ipynb` |
| 校准 HD，h | `wrap360(H - z)`，z=`calibration.screen_center` | 屏幕 RF=0 的参考方向；正方向保留 HD 的逆时针约定 | `matlab.ipynb`、`minimal_glm.ipynb`、FNN |
| trial HD 文件 | h × 960 / 360 | **像素偏移，不是度，也不是时间** | `data/hd_trials_times.npy` |
| 刺激屏幕角，S | MAT `Square_PositionX` | 生成脚本中纹理向右为正；当前 RF/FM 按顺时针解释，相应世界圆周角是 z−S | LED bar 源码；远端 `RFmapping_core_fm.m`；minimal GLM 的 `bearing()` |
| 世界 bearing，A | `atan2(target_y-eye_y, target_x-eye_x)` | XY +X 为 0、逆时针为正 | `Utils/bearing.py` |
| 通用 ego bearing，B | `wrap180(A-H)` | 前=0、左=+90、右=−90 | 两仓库的 `Utils/bearing.py`、EBC |
| RF / VS ego bearing，R | `wrap180(H-A)` | 前=0、右=+90、左=−90 | FM MATLAB、minimal GLM；**R=wrap180(−B)** |
| Basler HD | 图像坐标里的朝向 | 图像上方=0、左=90；图像 y 向下 | `spatial_cell_analysis.py`、`ebc_glm.py` |
| 角速度 AV | 展开后的 HD 变化量 / 时间 | 新分析为 deg/s，正值保留 HD 的左转约定 | minimal GLM、EBC、AV notebook |
| 角加速度 AA | HD 的二阶时间差分 | deg/s²，不是周期角度 | `simple_angular_velocity_glm.ipynb` |
| 球面 azimuth / elevation | 头中心方向 | 前=+Z、右=+X、上=+Y；azimuth 向右为正，elevation 向上为正 | 独立的 `rfmapping_fm_gui.py` |
| 椭圆 theta | 二维高斯/椭圆主轴旋转 | 内部 radians；是 RF 形状方向，不是头朝向 | legacy `vs.py` |

特别容易混淆的名字：MATLAB 的 `screenCenterDeg = screenDeg/2 = 180` 是有符号角转屏幕像素所需的半屏偏移；calibration 的 `screen_center` 是世界角度零点。两者不是同一个量。

另外，“Rotation Z 被代码用作 HD”不等于已证明它就是真正鼻尖前向。完整 XYZ 姿态、marker 连线、rigid body 的本地 forward 是不同定义，下面有实测差异。

## 2. 当前 rfmapping → MATLAB → minimal GLM 链路

### HD 和眼位置

[matlab.ipynb](/Users/vxf1610/Developer/rfmapping/matlab.ipynb:376) 先减 z，unwrap 后在 trial onset 插值，再保存 `h * 960/360`。因此 `hd_trials_times.npy` 读取时要除 `960/360`，不能把它直接当角度。

当前眼位置使用二维模型：

```text
h = wrap360(H - z)
eye_xy = body_xy + R2D(h) @ local_eye_offset_xy - screen_bottom_center_xy
```

它忽略 eye offset 的 z 分量和 body 的 X/Y rotation，且用校准后的 h 旋转 XY offset。这是当前明示的 legacy 约定；本次不替换它。

### 刺激转换

远端 `/mnt/ssd4.1/Matlab/Utils/RFmapping_core_fm.m:71–85` 与 [minimal_glm.ipynb](/Users/vxf1610/Developer/glm/minimal_glm.ipynb:253) 等价于：

```text
world_stimulus_angle = z - S
target_xy = radius * [cos(world_stimulus_angle), sin(world_stimulus_angle)]
A = atan2(target_xy.y - eye_xy.y, target_xy.x - eye_xy.x)
R = wrap180(h + z - A)
```

**动物/眼在圆心时，R = wrap180(S + h)。** 所以这里加 HD 是正确的：刺激角与 HD 本来就反向，不能凭直觉统一改成“刺激减 HD”。

MATLAB 通用 `Utils/bearing.m` 返回 A−H，FM 外层再取负；minimal GLM 内嵌 `bearing()` 直接返回 H−A。两者输出相同。通用 Python `Utils/bearing.py` 则返回 A−H，而且它的参数是两个世界圆弧端点；不能只因函数同名就直接替换 notebook 函数。

### 对齐的范围

| 路径 | 判断 |
|---|---|
| 当前 `matlab.ipynb` 保存的 trial HD/XY → FM MATLAB | HD 像素缩放、校准零点、XY 和 bearing 公式一致 |
| FM MATLAB → `minimal_glm.ipynb` | bearing 坐标系一致；空间编码不同 |
| `minimal_tracking.ipynb`、两仓库 `bearing_test.ipynb` | 使用同一二维眼位置和 RF bearing；世界图额外旋转 XY，使 RF 零点位于图上的 +X |
| `hd_av_vs_glm.ipynb` | bearing 公式同上，但编码整个 bar 覆盖范围 |
| `fnn/minimal_fnn.ipynb` | HD 标签使用同一个 H−z；不构建 VS bearing |

空间编码差异不能当成坐标 bug：FM MATLAB 变换 bar 左右边缘并计算覆盖像素；当前 minimal GLM 只把 **bar 中心射线**所在的角度 bin 置 1。前者还保留 `Square_PositionY` 行，后者水平模型合并这些行。离开圆心后，变换后的两个边缘角的中点，也不严格等于中心射线角。

实测 session 10 的 6,000 个白刺激中，这两种中心定义的绝对角差：中位数 **0.045°**，95 分位 **0.268°**，最大 **1.842°**。因此坐标一致不意味着两类模型或 RF 数值应该一样。

### 其他 RF 模式

远端 `Utils/RFmapping_core.m:138–153,179–242`：

- regular：直接使用 `Square_PositionX/Y`。`isUseRealCoordinate=false` 才会用网格大小重建坐标；当前配置为 true。
- rotation：`(S+180)*pxPerDeg + hdOffsetPix`，取屏幕宽度的模；在圆心假设下相当于 `wrap180(S+h)`，没有自由移动的眼位置校正。
- background moving：同样公式，但 offset 来自 `BackgroundRotation_XOffset_Pix`，当前 `rotationOffsetSign=+1`。这定义了计算符号；本次没有重放显示硬件来验证物理背景转向。
- coarse 像素图保存度数中心；fine moving 图可能保存 `0…959` 的像素位置。字段叫 `xPositions`，不保证永远是度。
- vertical bar 的 y=0 是单行占位，不提供垂直调谐。

实读 S2/S5/S10 的当前 RF 文件，x 都是 `−174,−162,…,174`，即 30 个 12°中心。S2/S5 的 y 为 `−39,−27,−15,−3,9,21,33`，S10 只有一行。**同样的 x 数字分别可能代表 screen、rotation 或 free-moving ego 坐标，不能只按轴标签判断物理等价。**

当前二维 FM 的 y 仍来自原始 `Square_PositionY`，没有用 pitch、眼高度或倾斜屏幕去重算 head-centric elevation。

## 3. HD、EBC 与比较图

### HD tuning 的零点没有自动统一

[tuning_curves.ipynb](/Users/vxf1610/Developer/rfmapping/tuning_curves.ipynb:85)、[run_tuning_curves.py](/Users/vxf1610/Developer/rfmapping/run_tuning_curves.py:49) 读取原始 H 后只做 `%360`，没有减 calibration.screen_center。

[tuning_curve_utils.py](/Users/vxf1610/Developer/rfmapping/Utils/tuning_curve_utils.py:502) 的 metadata 写的是输入“必须已校准为 0 up、CCW”；这是一条约定声明，**不是实际执行的校准**。GUI 也不会读 session calibration 再补偿。

结果：同一个 session 的 `.tc`/GUI HD 曲线与 minimal GLM/FNN 的 HD 标签，可能相差 z。session 10 的 z=**2.5°**；session 5/7 的 z=**2.561776°**。`simple_hd_glm.ipynb` 与 `simple_hd_latency.ipynb` 也使用原始 H；AV notebook、`hd_av_vs_glm.ipynb`、`vs_decoder.py` 在存在 calibration 时减 z，否则使用 0。

[direction_comparison.py:23](/Users/vxf1610/Developer/rfmapping/Utils/direction_comparison.py:23) 对 HD/EBC 都用 `wrap180(-angle)`，只统一了正方向，没有补入 z。如果比较原始 HD 与已校准 RF，校准 HD 的 RF 显示坐标应为 `wrap180(z-H)`。但 EBC 本身是相对角 B，转 RF 只需 `wrap180(-B)`，**不应该再给 EBC 加世界零点 z**。

### EBC 的两种世界坐标，目标相对角约定相同

[spatial_cell_analysis.py](/Users/vxf1610/Developer/rfmapping/spatial_cell_analysis.py:39)、[ebc_glm.py](/Users/vxf1610/Developer/glm/ebc_glm.py:204) 的 Basler 几何：

```text
图像 x 向右、y 向下
direction = [-sin(HD+B), -cos(HD+B)]
B=0 向前，B=90 向左，B=270 向右
```

[ebc_transform.py](/Users/vxf1610/Developer/glm/ebc_transform.py:22) 以同一图像坐标计算 world wall bearing，再减 HD；输出 animal-right/animal-forward 坐标。RF 仓库的 [circle_distances](/Users/vxf1610/Developer/rfmapping/Utils/ebc_analysis.py:94) 则用 Motive XY 的 `[cos(H+B), sin(H+B)]`。这两种写法在各自世界坐标里自洽；Basler 和 Motive 的原始 HD 数字不能未经世界坐标标定直接比较。

**最新 HD×Ego GLM 仍是左正。** [signed_bearing_display](/Users/vxf1610/Developer/glm/hd_ego_rate_models.py:298) 只做 `wrap180(B)`，不取负。因此：

| 图/函数 | +90°代表 |
|---|---|
| EBC、`simple_hd_ego_glm`、`hd_ego_joint_glm`、`hd_ego_rate_models` | 左 |
| `Utils/bearing.py` | 左 |
| minimal GLM 的 VS bearing、FM RF | 右 |
| HD/RF/EBC comparison 中已经转成 RF 坐标的 EBC | 右 |

另一个轴差异：EBC `.rfmap` 存 `x=distance(cm)`、`y=bearing(deg)`；普通 RF 存 `x=水平角`、`y=刺激垂直位置`。相同文件扩展名不代表相同轴语义。

EBC 的 6°网格在 0、6、12…°射线上计算距离，但输出 histogram 的中心标签是 3、9、15…°。读峰值时要允许这半个 bin 的采样/标签差异，不能把它解释成已测出的 3°物理校准误差。

### 已确认：peak alignment 符号错误

[direction_comparison.py:130](/Users/vxf1610/Developer/rfmapping/Utils/direction_comparison.py:130) 当前为：

```python
offsets = +peaks if align else np.zeros(len(order))
```

但 [align_profiles](/Users/vxf1610/Developer/rfmapping/Utils/direction_comparison.py:89) 的取样是 `profile(relative_angle - offset)`。要把原峰 p 移到 0，这里的 offset 应为 −p。

复现：输入峰 **+60°**，`align=True` 输出峰 **+120°**，而非 0°。原有两项测试失败。影响调用 `align=True` 的图；不影响只排序、不平移的默认图，也不改变未平移 profile 的峰值统计。

`peak_direction_sums()` 则明确计算峰值之和，`compare_direction_angles()` 的 offset 是 matched−reference；这是不同统计量，不应与“把峰移到 0”混为一谈。世界 HD 与相对 RF/EBC 即使坐标标签统一，也不要求不同类型神经元的 preferred directions 相等。

## 4. GUI 的角度和导出

| 实现/视图 | 当前约定 | 判断 |
|---|---|---|
| Python/Tk HD polar | 0上、90左、270右 | 与 Web 一致 |
| Web HD polar | 0上、90左、270右 | 与 Python 一致 |
| Python/Web HD line | 先用 `wrap180(-H)` 排布，刻度从左到右写 `180,90,0,270,180` | 正常的镜像展开；刻度仍显示原生 HD 值 |
| Python/Web/Swift 普通 RF polar | x 列从 −180 到 +180 顺时针排列，0在上 | 标准完整、居中 RF 网格下基本一致 |
| Swift HD polar | 0上、90右、270左 | **与 Python/Web 相反** |
| Swift HD line | 按原始 bin 顺序排布 | 与 Python/Web 镜像展开不同 |
| Python Figure Composer HD line | 原始 angle/rate 直接交给普通 line renderer | 与 live GUI 展开方式不同 |
| Python / Web 共享实现的 HD polar export | 默认曲线 90°在左，但刻度 90°在右 | **曲线与刻度矛盾** |
| RF 仓库 `plot_hd_tuning_curve()` | 默认 `clockwise=True`，0上、90右 | **默认与 Python/Web HD viewer 相反**；可显式传 false |

证据：

- [Python HD vector](/Users/vxf1610/Developer/rfmapping_gui/python/rfmapping_viewer/companions.py:739)：Canvas 坐标为 `(-sin(H), -cos(H))`。
- [Web HD vector](/Users/vxf1610/Developer/rfmapping_gui/web/frontend/src/hdMath.ts:146)：同一公式。
- [Swift HD drawing](/Users/vxf1610/Developer/rfmapping_gui/swift/Sources/RFMappingSwiftUI/Views/CompanionViews.swift:528)：使用 `angle=H−90°`，Canvas `(cos(angle),sin(angle))`，所以 90°在右。Swift 导出使用相同方向。
- [普通 RF polar](/Users/vxf1610/Developer/rfmapping_gui/python/rfmapping_gui.py:6587)：数学角 `90 + total/2 - total*column/columns`。这是显示布局；半径是 y 行排列，不是重新计算的球面 elevation。
- [HD 导出曲线与标签](/Users/vxf1610/Developer/rfmapping_gui/python/rfmapping_viewer/figure_export.py:1898)：默认曲线用 `−H−90°`，标签却用 `H−90°`。[Web 后端副本](/Users/vxf1610/Developer/rfmapping_gui/web/backend/rfmapping_web/shared_figure_export.py:1851) 同样如此。
- [分析侧 HD plot 默认方向](/Users/vxf1610/Developer/rfmapping/Utils/plotting.py:226)。

导出最小复现使用 400×400 画布，HD=90°单峰：峰位置 `(52,200)` 在左侧，但左侧写 `270°`，`90°`标签在右侧 `(365.76,200)`。这是数值错误标注，不只是两种审美布局。

### 独立的 FM 球面 viewer

[rfmapping_fm_gui.py](/Users/vxf1610/Developer/rfmapping_gui/python/rfmapping_fm_gui.py:192) 用：

```text
direction = [cos(el)*sin(az), sin(el), cos(el)*cos(az)]
az = atan2(x,z), el = asin(y)
```

azimuth 的前=0、右=+90，与 RF 的右正习惯相容。viewer 的 yaw/pitch 只是拖动视角，不是动物姿态，也不会改变输入数据的角度定义。投影/反投影的三个数值例子通过。

它读独立 HDF5 合约，要求 `unit,elevation,azimuth,time` 和 `rf-calib-1.0`、rigid-body-origin viewpoint：[fm_dataset.py](/Users/vxf1610/Developer/rfmapping_gui/python/rfmapping_viewer/fm_dataset.py:369)。session 10 当前 calibration 是 `rf-calib-2.0`、eye-midpoint 信息，当前 legacy FM writer 又是另一条二维路径。不能把“GUI 有三维球面”当成“当前 RF 已做完整三维头/眼/屏幕校正”的证据。

## 5. glm 内部存在的旧定义

### `free_moving_rf.py`

[旧入口](/Users/vxf1610/Developer/glm/free_moving_rf.py:93) 的 RF heading 来自 Marker002→Marker001 的 XY 连线，但同模型 HD 项来自原始 Euler Z。实测 session 10 两者的 circular 差值中位数 **42.14°**，5–95分位 **32.71°～50.06°**；不是当前 H−z 的那条输入。

[screen_angle](/Users/vxf1610/Developer/glm/free_moving_rf.py:131) 还写作 `screen_center + radians(S)`：把 `screen_center` 当弧度，且刺激 world 映射是 z+S，与当前 z−S 相反。例如把 2.5°当作 2.5 radians，会得到 143.24°。仓库 [README](/Users/vxf1610/Developer/glm/README.md:617) 已记载旧 radians 假设未重跑。

它还使用 full XYZ eye offset 和旧 camera-frame cache；当前 notebook 用二维眼位置和完整 low interval 中点。该脚本配置的 session 10 `260821.calib` 当前缺失。`compare_basis.py`、`population_rf.py`、`hd_delay.py` 依赖它，因此不能把这些历史结果作为当前 minimal GLM 的同坐标结果。

### `session_data.py`

[该 helper](/Users/vxf1610/Developer/glm/session_data.py:40) 已正确把 screen_center 当度、HD 也减同一零点；但刺激仍用 `world_angle=S+z`，输出 `A-H`，eye 使用 full XYZ。它与当前 `minimal_glm` 的 z−S / H−A / planar-eye 不一致，不能只改图的横轴范围就互换。

### VS-only 与 HD-only notebooks

- `minimal_glm_white_gray.ipynb` 的角度就是 MAT `Square_PositionX %360`，没有眼位置或 HD 校正；它是屏幕刺激位置，不是自由移动后的 ego bearing。
- `simple_hd_glm.ipynb`、`simple_hd_latency.ipynb` 保留原始 JSON HD。
- `simple_angular_velocity_glm.ipynb`、`hd_av_vs_glm.ipynb`、`vs_decoder.py` 有 calibration 时减 screen_center；没有时保留原零点。
- `simple_hd_ego_glm.ipynb`、`egocentric_boundary_glm.ipynb` 走 Basler/EBC 坐标，bearing 保持左正。
- 新 AV 定义为 `[unwrapHD(t)−unwrapHD(t−0.05)]/0.05`，deg/s；AA 用相应二阶差分。旧 `free_moving_rf.py` 则对 radians 求 gradient，原始单位是 rad/s，后续再标准化。比较原始 AV 值时必须转换。

## 6. 二维/三维姿态与其他角度

[motive_pose.py](/Users/vxf1610/Developer/rfmapping/Utils/motive_pose.py:14) 使用 intrinsic `XYZ`，即 `Rx @ Ry @ Rz`，通过 Slerp 插值完整姿态，再转 rigid-local eye offset。`session_data.py` / `free_moving_rf.py` 也转完整 XYZ，但在 frame 上转眼位置后插值 XY。当前 `matlab.ipynb` / minimal GLM 使用上述二维模型，未调用 `motive_pose.py`。

session 10 固定刺激公式、HD、时刻，只把二维眼位置换成 frame-wise full XYZ 后插值：

- 眼位置距离差中位数 **52.07 Motive units**，最大 **262.18**；按 calibration 的 3.74 units/mm，约 13.92 / 70.10 mm。
- 所有 trial 的 RF bearing 绝对变化：中位数 **1.55°**，90分位 **4.51°**，99分位 **10.10°**，最大 **20.23°**。

这说明眼位置模型差异足以改变角度 bin，不能当成纯粹常数零点偏移。该比较没有声称哪一个模型就是生物学真值。

当前 v2 calibration 还包含倾斜屏幕轴、screen-local 基向量、head forward/up 和 eye-midpoint 定义；当前 planar FM 路径主要读顶层 XY/radius/zero/offset，并未消费这些完整三维字段。确认物理鼻尖方向和真实 elevation，需要以对应标定模型独立验证。

更具体地说，session 10 的同一份 `260827.calib` 同时含有：

| 内容 | 文件中的值 / 直接推导 |
|---|---|
| legacy `screen_center` | 2.5°，当前二维链路使用 |
| v2 `screen.zero_radial_motive_unit` 的 XY 投影角 | 89.0285°，当前二维链路不使用 |
| v2 `screen.azimuth_direction_sign` | +1，v2 schema 定义为从屏幕 +Z 看向原点时逆时针 |
| v2 屏幕轴相对 Motive +Z 的倾角 | 31.1239° |
| v2 `head.forward_rigid_body_unit` 的 local XY 角 | 41.2624°，不是 local +X |

这些不是可直接互换的一套 scalar 零点。v2 的 [schema 说明](/Users/vxf1610/Developer/rf_calib_gui/README.md:205) 和 [屏幕零点标定代码](/Users/vxf1610/Developer/rf_calib_gui/src/rf_calib_gui/calibration.py:1167) 明确以实际纹理 X=0 的物理点建立三维基；当前二维代码没有应用这个基，也没有采用其方向符号。这里证明的是**两种模型尚未统一消费标定信息**，并不据此宣称某一版标定是真值、另一版有固定 86.5°误差。

FNN 使用 `(cos(h), sin(h))` 监督、`atan2(z2,z1)` 解码，角度与校准 HD 一致；[FNN notebook](/Users/vxf1610/Developer/rfmapping/fnn/minimal_fnn.ipynb) 的 projection plot 刻意用 **0在右、逆时针**，与 GUI **0在上**差一个显示旋转，不能靠图上象限直接比峰。

legacy `vs.py` 的 `theta` 是高斯/椭圆方向，参数单位为 rad，文字报告转 degree；它不是 HD 或 azimuth。`Utils/statistic_utils.py` 的 circular distance、相关性先将 degree 换 rad，offset 输出 matched−reference。`Utils/decoder_utils.py` 的 circular error 默认 period=360；这些数学辅助没有额外屏幕校准。

## 7. 实证、版本边界与建议顺序

详细数值与命令见 [verification.txt](/Users/vxf1610/Developer/rfmapping/docs/angle_audit_20260916/verification.txt)。

- session 10 原始 JSON HD 与 CSV Rotation Z 的最大 circular 差为 **4.55×10⁻¹³°**。
- 用原始 ADC 完整 low interval 中点重算的 12,000 个 trial HD，与保存文件的最大 circular 差为 **5.69×10⁻¹⁴°**。
- 当前二维模型重建的 12,000 个 eye XY，与保存值最大差 **0**。
- 10,000 个随机位置/角度中，通用 CCW bearing 取负后与 minimal GLM bearing 一致，最大 circular 差 **2.85×10⁻¹³°**。
- RF 的 geometry/direction/pose 测试 **28通过、2失败**；失败来自 peak alignment 外层符号。GLM bearing/coordinate/one-hot 定向检查 **43通过**，AV/AA 和 circular interpolation 检查另有 **6通过**；合计 **77通过、2失败**。
- Python GUI 当前本地源码中的相关纯函数提取后送到远端执行，复现了 export 标签错误。Swift/Web 方向核对以源码为依据，未构建或发布它们。

核对的 RF/GLM 核心模块与关键 notebook，本地和远端源码一致。GUI 远端 checkout 与本地不同，部分本地模块远端不存在，因此本报告 GUI 结论针对用户指定的**本地当前源码**，不是对已安装/已部署二进制的版本保证。

还有一条需要修正的流程文档：远端 `~/Developer/rfmapping/matlab.ipynb` 当前是 **0字节**，`~/Developer/sync/matlab.ipynb` 是旧版本，HD 只 `%360`、按 searchsorted 取帧、未减 screen_center。实际 session 10 保存的 HD/XY 却与**本地当前**二维 notebook 完全匹配。所以现有 AGENTS pipeline 箭头不能作为“当前数据由哪版 notebook 生成”的充分 provenance。本次未修改这些文件。

建议后续按以下顺序处理；本次仅列出，不擅自改科学约定：

1. 修复 `align=True` 的 offset 符号，以及 HD polar export 的曲线/刻度方向。
2. 明确一个 GUI HD 显示约定，让 Swift、普通 Matplotlib helper 和导出与 Python/Web 一致；HD line 导出复用 live view 的展开规则。
3. 给每种输入/输出记录 `coordinate_frame`、`angle_unit`、`positive_direction`、`zero_reference`、`heading_source`、`eye_model`、时间参考和代码版本。`.tc` 的自然语言提示不足以承担这些信息。
4. 保留 HD/EBC 左正、RF右正也可以，但所有跨类型比较必须显式做对应转换。对原始 HD 用 `wrap180(z-H)`；对相对 EBC 用 `wrap180(-B)`，两者不能共用“加同一个 offset”的逻辑。
5. 给旧 GLM 入口明确标注坐标版本，确认要采用哪个 eye/head/screen 模型后再迁移和重算。物理闭环应核对已知 LED 0°/+90°位置在 Motive 中的方向，以及动物朝向这些位置时的 Euler、marker forward 和 eye pose；不要把二维代码一致性当作这项验证。不可静默把二维分析替换为三维后继续混用旧结果。

最终判断：**现行 FM RF 与 minimal GLM 的核心刺激 bearing 是一套坐标；三个仓库整体不是一套完全统一的角度系统。最容易造成误判的是同名 bearing 的相反正方向、HD 是否减 screen_center、以及画图层的镜像/刻度问题。**
