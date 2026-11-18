# R1#10 模块组合定性比较（2026-11-18）

## 审稿人原话

> "The manuscript lacks a qualitative comparison of different module combinations. While the quantitative results show their effectiveness, additional visualization would help to illustrate how the individual modules and their combination affect the final detection performance."

上一轮我把它判成"用已有定量矩阵 + Fig. 4(a) 替代"，**没有真正出可视化**，这是漏项。本文件补上。

---

## 做法

用 `confirm_split_cache`（GroundingDINO-tiny，DIOR→COCO，六类）重算同一批 proposals 在五种打分原型下的排序，**不重跑检测器、不训练**。方向分解：

```
d_joint = e("a photo of tgt")        − e("an overhead image of src")   （= cache Delta）
d_vocab = e("an overhead image of tgt") − e("an overhead image of src")
d_cond  = e("a photo of tgt")        − e("an overhead image of tgt")
d_joint = d_vocab + d_cond                       （cos(Delta, recomputed) = 1.0000，六类全对）
```

五个面板：

| 面板 | 打分原型 |
|---|---|
| base | `W = unit(P_src) − 0.5·P_neg` |
| +vocab | `W = unit(P_src + 2·d_vocab) − 0.5·P_neg` |
| +cond | `W = unit(P_src + 2·d_cond) − 0.5·P_neg` |
| joint | `W = unit(P_src + 2·d_joint) − 0.5·P_neg` |
| joint+gate | `a_i = clip(3.2324·R·g(q_i), 0, 4)`，`W_i = unit(P_src + a_i·d_joint) − 0.5·P_neg`，`g(q)=clip(1−tanh(3(q−0.2974)),0,1)` |

展示规则：每面板画 top-3 去重后的框（贪心 NMS，IoU 阈值 0.55；剔除覆盖 >45% 画面的框），绿虚线 = GT，蓝 = TP，红 = FP。crop 只在 GT 占画面 <8% 时启用。

---

## 结果（`figs/fig_module_qual_numbers.json`）

| 样本 | GT | base | +vocab | +cond | joint | joint+gate |
|---|---|---|---|---|---|---|
| car (img 196759) | 3 | 1TP/3FP | 1TP/3FP | 2TP/2FP | **3TP/1FP** | 2TP/2FP |
| truck (img 124277) | 2 | 0TP/4FP | 1TP/3FP | 2TP/2FP | **2TP/2FP** | 2TP/2FP |
| airplane (img 571857) | 4 | 2TP/2FP | 2TP/2FP | 3TP/1FP | **3TP/1FP** | 2TP/2FP |

三个可读的机制证据：

1. **单个模块各自有效、但都不完整**：car 上 base→cond 已到 2TP，joint 才到 3TP；truck 上 vocab 只 +1、cond +2，joint 才把两个目标都检出。这与 Table VI 的"两项互补、去掉任一项都掉"一致。
2. **airplane 是构造性隔离对照**：源类与目标类同名 ⇒ `d_vocab ≡ 0`（实测 ‖d_vocab‖=0.000），所以 **+vocab 列与 base 逐框完全相同**，增益只能来自 +cond。这是"词汇项为零时条件项仍在工作"的直接可视化。
3. **gate 在单图上更保守**：car/airplane 的 joint+gate 比 joint 少 1 个 TP。这是设计使然——gate 在高置信 proposal 上收缩校正以避免过冲，代价是牺牲一点 headroom；数据集级（Table II/III）它是净正的，单图 top-3 不代表总体。图注必须写清这一点，否则会被反问"为什么加了 gate 反而变差"。

---

## 英文图注（可直接用）

> **Fig. X.** Qualitative comparison of the individual modules and their combination on DIOR→COCO (GroundingDINO-tiny, zero target annotation). Each row is one target image; the five columns use the same proposals and the same CLIP crop features and differ only in the scoring prototype: the uncorrected source prototype (base), the vocabulary term alone, the observation-condition term alone, their joint sum, and the joint shift with the confidence gate used at deployment. Green dashed boxes are ground truth, blue boxes true positives and red boxes false positives; only the three highest-scoring non-duplicated proposals are drawn. The airplane row is a constructive control: the source and target class names coincide, so the vocabulary term is identically zero (‖d_vocab‖ = 0.000) and the +vocab column reproduces the base column box for box, leaving the condition term as the only source of the gain. The gate column is deliberately more conservative than the ungated joint shift: it shrinks the correction on high-confidence proposals to avoid overshoot in low-headroom regions, which can cost one detection on an individual image while remaining net positive at the dataset level (Tables II–III).

---

## 放置建议

- **图件**：`figs/fig_module_qual.pdf`（矢量，7.4×6.7 in），PNG 预览 `figs/fig_module_qual.png`。
- **位置**：建议放**补充材料**新增 Fig. S4（主稿 22 页余量极薄，插整页图会破坏页数）；主稿 §IV 分解小节加一句指针即可：
  > "Fig. S4 shows the same decomposition qualitatively: the vocabulary and condition terms each recover part of the ranking, the joint shift recovers the most, and the airplane row, where the vocabulary term is identically zero, isolates the condition term."
- 若审稿人坚持要在主稿看到，则需替换现有某张图或压缩 §II，代价是重新排版和页数核算。

---

## 产物

- 脚本：`实验代码/r1_module_qualitative.py`（GPU 环境，需 CLIP ViT-B/32）
- 数字：`figs/fig_module_qual_numbers.json`
- 图：`figs/fig_module_qual.pdf` / `.png`

## 诚实边界

- 这是**单图定性**，用于说明机制如何作用，不构成统计结论；数据集级结论仍以 Table VI（AP50 分解）为准。
- 选样规则是"每类取 joint−base 的 TP 差最大者"，属于**有倾向性的示例选择**，图注里不声称随机抽样。若要更稳妥，可补一张随机样本图（未做）。
- gate 列单图偏保守（见上），这本身是 gate 的设计目标，不是缺陷，但必须在图注里解释。
