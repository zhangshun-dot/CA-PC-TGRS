# v2 冻结协议 运行指南（2026-08-26）

冻结参数：**β=1.0, λ=0.25**（DIOR→VOC 开发域选参，tge_dev_scan_voc.json）。
数据：COCO val2017 全量（即标准标注，已核实）；检测器 GDINO-tiny（主表）。
指标：AP50（11 点）+ AP@[0.5:0.95]（101 点）+ Recall@proposal；bootstrap R=1000×3 种子 + BH-FDR。

## 运行顺序（每步输出 JSON 到 `Desktop\实验数据\`）

| 步骤 | 命令 | 耗时 | 输出 |
|---|---|---|---|
| 1 | `python tge_v2_main.py` | ~25-35 分钟 | tge_v2_main.json（全量主表 + 统计） |
| 2 | `python tge_v2_detectors.py` | ~30-40 分钟 | tge_v2_detectors.json（GDINO-base/OWLv2 矩阵） |
| 3 | `python tge_v2_ablations.py` | ~20 分钟 | tge_v2_ablations.json（负原型 topk/分桶/gate） |
| 4 | `python tge_v2_visdrone.py` | ~10 分钟 | tge_v2_visdrone.json（需先设 VISDRONE_DIR） |

依赖：步骤 1 必须先跑（步骤 2 读它的均值行）；步骤 3/4 复用步骤 1 的检测缓存（tge_v2_dets_COCO-*.pkl）。

## 前置条件
- COCO 标准 val2017 标注已确认（`modeling\coco_data\annotations\instances_val2017.json`）。
- DIOR/NWPU/DOTA 数据在既有路径（unified_eval.py 常量）。
- **VisDrone**：下载后把 `tge_v2_visdrone.py` 顶部的 `VISDRONE_DIR` 改成实际路径（含 images/ 与 annotations/ 子目录）。
- 需要 GPU（检测与 CLIP 编码）。检测缓存首次生成 ~0.7-1s/图。

## 跑完回报我什么
- 三个 JSON 的路径即可（或把控制台表贴给我）。
- 我会：核对 (β,λ)=(1.0,0.25) 下的新主数字 → 更新论文（摘要/表1-3/结论）→ 补检测器矩阵章节 → 更新协议 §1 的"后果"数值。

## 注意事项
- 主表零样本在 `images[K:]`（K=10）评估；airplane N_pool=97 → N_test=87。
- OWLv2 行是 raw-logit 尺度 pre-filter 0.03（分数尺度陷阱，勿改 sigmoid）。
- 若某类检测缓存已存在会复用；换检测器矩阵是新缓存（GDINOB/OWLV2 前缀）。
