# R1#6 — 置信门控参数：不是拍定，是「源域网格 + 无标注常数 + 敏感性扫描」

生成：2026-11-18　对应审稿意见：R1#6「confidence mechanism 的参数是经验拍定还是实验验证」

## 0. 一句话结论

三个参数没有一个是拿目标域增益调出来的：β\* 由源域验证集网格选出并跨目标域冻结；c₀ 由**无标注**目标图的中位置信度给出；门控斜率 s=3.0 固定。c₀ 在 [0.15,0.45] 上重扫，**当前前向闭环**的四检测器平均增益只从 +0.0177 变到 +0.0211（带宽 0.0034），且每个 c₀ 下四个检测器全为正。

## 1. 参数与依据

| 参数 | 取值 | 依据 | 读目标标注 |
|---|---|---|---|
| β\* (校强度) | 源域验证集 AP50 网格 $\mathcal B=\{0,0.5,\dots,4\}$ 选一次，跨目标域冻结 | 源域有标注，目标域零标注 | 否 |
| c₀ (门中心) | 0.2974（前向闭环）/ 0.3003（条件轴探针） | 目标域**无标注**图像的逐图最大 proposal 置信度中位数，跨检测器合并 | 否 |
| s (门斜率) | 3.0 | 固定，未调；c₀ 扫描覆盖其有效区间 | 否 |
| 20 个源原型 / 3 负原型 | 固定 | 源域 crop 构造 | 否 |
| λ (负原型权重) | 0.5 | 固定 | 否 |

## 2. c₀ 敏感性（当前前向闭环配置，本次重跑）

- 脚本：`实验代码/r1_c0_sweep_closedloop.py`（由 `run_closed_loop_balanced.py` 生变体：只跑前向、内嵌 c₀ 网格）
- 产物：`实验数据/run_c0_sweep_closedloop_20261118.json`
- 配置：`prior_scale=3.2324`、`c0=0.2974`、`λ=0.5`、逐图 `max_conf`、四检测器 × 6 类（fwd DIOR→COCO）

| c₀ | GDINO | OWLv2 | YOLO | RT-DETR | 均值 |
|---|---|---|---|---|---|
| 0.15 | +0.0553 | +0.0118 | +0.0032 | +0.0003 | +0.0177 |
| 0.20 | +0.0561 | +0.0136 | +0.0031 | +0.0006 | +0.0183 |
| 0.25 | +0.0583 | +0.0148 | +0.0038 | +0.0007 | +0.0194 |
| **0.2974** | **+0.0607** | **+0.0152** | **+0.0046** | **+0.0008** | **+0.0203** |
| 0.30 | +0.0609 | +0.0152 | +0.0044 | +0.0008 | +0.0203 |
| 0.35 | +0.0613 | +0.0175 | +0.0043 | +0.0010 | +0.0210 |
| 0.40 | +0.0613 | +0.0176 | +0.0044 | +0.0012 | +0.0211 |
| 0.45 | +0.0613 | +0.0123 | +0.0045 | +0.0013 | +0.0199 |

**带宽 0.0034**（0.34 AP 点）；每个 c₀ 下四检测器全为正；增益随 c₀ 单调微增后回落，无临界点。

### 复现校验（关键）

c₀=0.2974 一行的四个数 **+0.0607 / +0.0152 / +0.0046 / +0.0008 与主稿 Table X（`tab:labelfree`）逐格完全一致**，说明管线与主表同源，不是另起一套口径。

## 3. 旧产物作废

`run_c0_sensitivity.json`（2026-10-13）跑在**条件轴 probe 缓存**上、`prior_scale=3.3706`、`c0=0.3003`，属上一代配置（其 GDINO 曲线 0.0419→0.0483，带宽 0.0018）。结论方向一致但仍应改引本表，避免两代配置混用。**引用时只用 `run_c0_sweep_closedloop_20261118.json`。**

## 4. 已踩过的口径坑（写给后来的人）

1. 闭环表 base = `unit(P_src) − λP_neg`（airplane 0.7818），confirm/R2 离线表 base = 原始检测器分（airplane 0.7484）。**两者不可混**：混用会得出「门控增益复现不出 +0.0607」的假结论。
2. 门控是**逐图** max_conf，不是逐框；`gamma = clip(PRIOR_SCALE·dc_mag·gate(max_conf,c0),0,4)` 作用在**原始 Δ** 上。写成逐框、或把 γ 直接乘 Δ（少/多一个 ‖Δ‖）都会把 c₀ 曲线算歪（实测能歪到带宽 0.039）。

## 5. 可直接用的英文（主稿 §III.4 替换句）

> The gate has three parameters and none of them is fitted to target-domain gains. The correction strength β\* is selected once by a grid search on a labelled source-validation split and then frozen for every target domain. The gate centre c₀ is the median of the per-image maximum proposal confidence over unlabeled target images, pooled across detectors, and therefore reads no target annotation. The slope is fixed at 3.0. Sweeping c₀ over [0.15, 0.45] on the forward closed loop moves the mean gain over four detectors only between +0.0177 and +0.0211 (band width 0.0034), with every detector positive at every value; the deployed c₀ = 0.2974 reproduces the per-detector gains of Table~\ref{tab:labelfree} exactly.

## 6. 补充材料表草稿（Table S21）

```
Parameter protocol of the confidence gate (no target annotation is read by any of them).
Parameter              Value                 Determined from                  Target labels
β* (strength)          grid {0,0.5,...,4}    source-validation AP50, once     no (source only)
c0 (gate centre)       0.2974 / 0.3003       median per-image max confidence  no
                                             over unlabeled target images
s (gate slope)         3.0                   fixed                            -
λ (negative weight)    0.5                   fixed                            -
c0 sensitivity         [0.15,0.45]           mean gain +0.0177..+0.0211,      -
                                             band width 0.0034, 4/4 positive
```
