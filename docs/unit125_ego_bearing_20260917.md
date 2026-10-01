# Unit 125：spatial 与 HD×Ego GLM 的 ego bearing 差异

核对对象：`260831_2 / Probe A / baseline / unit 125`。全部计算在 `hhw9l84` 的 `~/.virtualenvs/rfmapping` 环境运行。本轮没有修改分析代码、notebook 或原始分析结果；仅新增本报告和比较图。

## 统计解释更正与匹配对照

下文的“峰”均指指定数组的离散最大值，不自动代表可靠的生理 preferred bearing。此前将二维热图最大格、条件模型最大格和跨HD汇总曲线的峰混着解释，并据此直接断言差异的生理原因，证据不足。不同汇总方式允许最大值改变，只是数学上的解释机制；它本身不证明稳定的方向偏好发生了变化，也不证明整条分析管线正确。

新增以下针对性核对，结果均只写入临时审计目录：

1. 使用独立的射线与四条有限墙线段求交算法，重建GLM保存姿态的墙距，最大误差约1.39e-12 cm；HD和距离状态逐项一致。随后以逐格布尔选择直接重建Recorded图的spike总数及occupancy，与保存数组的最大差均为0。这支持保存结果的分组索引正确；不把它扩大解释成相机、物理标定或所有预处理均已正确。
2. 为HD-only补做匹配的对照：同样90339个时间bin、同样30个HD状态、同样10个outer folds、同样guard和3个inner folds。HD-only是带截距的Poisson log-link模型，仅含30个HD one-hot特征；正则强度在inner folds从1e-8至1e-1的8个候选值中选择，10折均选中1e-4。联合模型直接使用原来保存的OOF预测，未重拟合、未覆盖。

| 模型/比较 | 测试集结果（bits/spike） |
|---|---:|
| HD-only，相对相同训练均值常数baseline | 2.3927425 |
| HD×Ego联合模型，相对同一baseline | 2.3998104 |
| 联合模型减HD-only | 0.0070678 |
| 增量的时间块重采样95%区间 | [-0.0247880, 0.0393341] |

区间使用原来的50个时间块进行4000次成对有放回重采样，固定OOF预测，随机种子20260917；不是重拟合后的完整模型不确定性区间。该对照没有明确支持此联合模型相对HD-only的额外预测增益；这不等于证明该细胞没有ego效应，更不等于检验了其他模型或更高分辨率的HD控制。

3. 对图中最大格做同样的时间块重采样，所有射线共享块权重，保持时间依赖及射线之间的依赖。远距离Recorded最大格落在bearing0°的比例为63.55%；远距离OOF最大格落在90°、45°、0°的比例分别为42.63%、30.18%、23.78%。OOF预测固定，所以这些比例只反映时间块组成对图上最大格的敏感性，不是各角度为真实preferred bearing的概率，也不包含重拟合不确定性。因此不能直接把原图0°与90°的argmax差异解释成可靠的90°方向偏移。

已确认的事实是：图上离散最大值不同、部分计算口径不同、远距离覆盖范围不同。尚未建立的结论是：这些最大值对应稳定的独立ego方向偏好，或其中某张图的最亮角度在生理上更正确。

![相同 bearing 坐标下的比较](/Users/vxf1610/Developer/rfmapping/docs/unit125_ego_bearing_20260917.png)

## 用户截图实际亮点：Recorded / OOF

### 随后两张GLM截图的直接对照：OOF分组图与conditional refit图

第一张图下排用每个真实时刻的8条射线距离预测spike，再在指定HD、目标bearing和距离档内汇总。第二张图由 `conditional_surface()` 生成：固定HD和目标射线的距离档，其余7条射线全部换成全session各自距离档的占比特征，再计算模型响应。第二张使用全数据refit；第一张使用OOF模型。

为区分这两个因素，固定使用同一个已保存的全数据refit模型，另算一次真实姿态上的预测并按第一张图的规则汇总。对同一格（d≤8 cm、HD180–192°、bearing270°），结果为：

| 计算方式 | 该格发放率 |
|---|---:|
| 原OOF模型，真实姿态预测后汇总 | 124.2570 Hz |
| 同一refit模型，真实姿态预测后汇总 | 124.1265 Hz |
| 同一refit模型，其他射线设为平均特征 | 86.3522 Hz |

固定模型以后，两种显示定义仍有明确差异。三档距离的离散最亮bearing，真实姿态分组时分别是225°、225°、90°；平均参考特征下分别是315°、180°、0°。因此对于这两张GLM图，可以直接确认：改变其他射线的输入处理方式会改变亮点位置，不能将它们当成同一种统计量的两个估计。

这里的“平均特征”是三个距离档的占比，不是平均距离cm；多个方向合在一起也不一定对应真实可实现的墙几何。`conditional_surface`这个名称不等于在真实姿态的条件分布上对预测求平均。Poisson log-link模型先平均输入后预测，也不等于先预测再平均。本段只解释两种图的计算差异，不赋予单个最大格生理preferred bearing的含义。

用户随后贴出的截图是 `recorded_oof.png`，不是 `joint_surface.png`。此前用条件图最大值、或跨 HD 汇总后的峰来回答截图亮点的位置，比较对象不对应。角度坐标定义一致不等于亮点或调谐峰一致。

按截图实际使用的 occupancy ≥1秒掩膜，逐格检查每个距离面板的最大值：

| 距离段 | Spatial 曲线峰 | 截图上排 Recorded 最亮格 | 截图下排 OOF 最亮格 |
|---|---:|---|---|
| ≤8 cm | 339° | bearing225°，HD180–192°，132.71 Hz | bearing225°，HD180–192°，127.32 Hz |
| 8–16 cm | 357° | bearing225°，HD180–192°，132.63 Hz | bearing225°，HD168–180°，123.87 Hz |
| >16 cm | 63° | bearing0°，HD156–168°，127.30 Hz | bearing90°，HD180–192°，123.31 Hz |

因此 Spatial 与截图的最亮 bearing 确实不同；截图远距离栏的实测和 OOF 最大格也不同。

具体以≤8 cm为例：bearing225°在HD180–192°内达到132.71 Hz，但该bearing跨实际HD汇总后只有6.97 Hz；bearing0°跨HD汇总后是20.66 Hz。取二维图的最大格，与按真实停留时间加权汇总，不保证保留同一峰值。Spatial还增加了平滑和距离rate求和。这些数值描述不同统计量之间的关系，不能单凭它们证明不同的生理方向偏好。截图上排纯实测数据已经出现这一差异，因此近距离的差异并非必须由GLM拟合造成。

远距离栏的 Recorded 和 OOF 最亮格不同，则说明预测没有完全保留实测最大格；不能用汇总后峰值相同来声称两个二维面板的亮点相同。

## 条件图及跨 HD 汇总结果（与截图亮点区分）

两边的方向定义相同：0° 前、90° 左、270° 右；横向显示顺序是 180 → 90 → 0 → 270 → 180。也检查了 spatial 单元热图的像素取值，339° 对应原数组的 339°，没有上下翻转或标签错配。

| 距离段 | Spatial 保存曲线峰值 | GLM 条件图最大格的 bearing | 实际 spike 跨 HD 合并后的峰值 | GLM OOF 预测跨 HD 合并后的峰值 |
|---|---:|---:|---:|---:|
| ≤8 cm | 339° | 315° | 0° | 0° |
| 8–16 cm | 357° | 180° | 0° | 0° |
| >16 cm | 63° | 0° | 135° | 135° |

条件图最大格均位于 HD 156–168°，表中列出的角度是 ego bearing。跨 HD 合并使用各 bearing 的总 spike 数（或 OOF 预测数）除以总 occupancy，未对各 HD bin 的 rate 做等权平均。339°、357° 和 0° 在环形角度上相邻。

## 为什么不同

Spatial 曲线跨所有 HD 累积 spike 和 occupancy，再进行二维平滑，最后将选定距离 bin 的 rate 相加。当前参数是 6° 角度 bin、20 个距离 bin、Gaussian sigma=5 bins，即角度 sigma=30°、距离 sigma≈7.25 cm。保存的距离段是 `sum(rate)`，并非该距离段的 `sum(spikes) / sum(occupancy)`。

GLM 条件图按 HD、ray、distance 分开计算模型值。对某一格，将其对应射线设为指定距离状态，其他射线使用全局平均距离特征。它表达的是条件模型响应；这组参考特征也不一定对应一个实际可实现的动物姿态。因此条件图的最亮格，与跨实际姿态汇总得到的 bearing 曲线峰值可以不同。

Unit 125 的实测 HD rate 峰位于 180–192°（中心186°，114.48 Hz；总体平均9.99 Hz）。诊断性地仅使用同一批数据估计出的 HD rate，再按真实的 bearing/距离采样分布汇总，得到的三档 bearing 曲线与实测曲线相关系数分别为 0.9979、0.9891、0.9955。这支持：边际 ego 曲线很大程度受到 HD 调谐与采样分布关系的影响。这是同一数据上的诊断性分解，不是独立的交叉验证，也不据此判断该细胞完全没有 EBC 效应。

## 远距离范围确实不一致

Spatial 距离上限是 `41 / 2 * sqrt(2) = 28.99 cm`；GLM 是 `41 * sqrt(2) = 57.98 cm`。所以两边的“>16 cm”没有涵盖相同范围。

为单独检验这一点，固定 GLM 保存的全部有效时间点、坐标、HD 和 spike 数，在内存中应用 spatial 的算法。保留原距离 bin 宽度，扩大范围时由20 bins改为40 bins；不重拟合 GLM，不覆盖任何原结果。

| 相同输入上的计算方式 | ≤8 cm 峰 | 8–16 cm 峰 | >16 cm 峰 |
|---|---:|---:|---:|
| 原 spatial 方法：上限28.99 cm、sigma=5、sum(rate) | 339° | 357° | 63° |
| 仅把距离上限扩至57.98 cm | 339° | 357° | 153° |
| 再改为 pooled spikes / occupancy | 339° | 357° | 159° |
| 再取消平滑 | 357° | 21° | 159° |

仅修正距离覆盖范围就将远距离峰从63°移到153°；原来的远距离差异有明确的计算来源，不需要假设角度显示翻转。

## 验证与限制

- 使用当前 `spatial_cell_analysis.py` 从原数据重算 unit125，完整60×20矩阵与保存的 `.rfmap` 逐值一致，最大绝对差为0。
- Spatial 当前加载91443帧、36877个unit125 spikes；GLM保留90339个40 ms bins、36105个spikes。两者的时间处理与有效姿态筛选仍有差异。上述固定输入实验保持这些因素不变，仍重现了三个 spatial 峰值。
- Spatial 的角度射线实际在0、6、…、354°计算，保存为区间中心3、9、…、357°；存在3°的采样位置与显示中心差。这无法解释远距离或条件图的较大差异，也不是物理2.5°零点问题。
- GLM只有8条bearing射线，间隔45°；spatial有60条，间隔6°。不能把两者的离散最大值当成同等角度精度的估计。

要直接比较两种分析的 ego tuning，应统一距离范围、角度采样、平滑与汇总方式，并把 GLM 对真实姿态的 OOF 预测按 bearing/距离汇总；条件图继续用于查看随 HD 和距离变化的模型响应。

## 代码与数据位置

- [Spatial 几何、分箱和平滑](/Users/vxf1610/Developer/rfmapping/spatial_cell_analysis.py:253)
- [Spatial 距离段汇总](/Users/vxf1610/Developer/rfmapping/spatial_cell_analysis.py:301)
- [Spatial 绘图 notebook](/Users/vxf1610/Developer/rfmapping/spatial_cell_plotting.ipynb)
- [GLM 条件模型响应](/Users/vxf1610/Developer/glm/hd_ego_joint_glm.py:193)
- [GLM 实际姿态上的 occupancy / rate 汇总](/Users/vxf1610/Developer/glm/hd_ego_rate_models.py:284)
- [GLM bearing 显示顺序](/Users/vxf1610/Developer/glm/hd_ego_rate_models.py:299)

远程数据根目录：`/mnt/senzailab/Kai/#Recording/m19/260831/260831_2/data`。Spatial 文件位于 `spatial_cells/ProbeA/baseline/egocentric_rate_map*.rfmap`；GLM 文件位于 `glm/hd_ego_joint_30hd_8rays_d8_16/unit_125/{arrays.npz,joint_surface.png,recorded_oof.png}`。
