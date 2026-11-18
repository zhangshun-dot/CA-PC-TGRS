from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')

# Comprehensive Chinese-to-English mapping for table content
translation_map = {
    # Headers
    '方法': 'Method',
    '目标域数据': 'Target Data',
    '可训练参数': 'Trainable Params',
    '6类平均': '6-class Mean',
    '增益': 'Gain',
    '类别': 'Category',
    '基线': 'Baseline',
    '固定参数': 'Fixed Params',
    '自适应先验': 'Adaptive Prior',
    '固定增益': 'Fixed Gain',
    '自适应增益': 'Adaptive Gain',
    '层': 'Layer',
    '配置': 'Configuration',
    '平均增益': 'Mean Gain',
    '相对基线A的改进': 'Improvement vs Baseline A',
    '源域（原型池）': 'Source Domain (Prototype Pool)',
    '目标类映射': 'Target Class Mapping',
    '源域→目标域': 'Source → Target',
    '负原型方向': 'Negative Prototype Direction',
    '均值 AP': 'Mean AP',
    '与无负原型之差': 'Diff vs No Neg Proto',
    '检测器': 'Detector',
    '基线 AP50': 'Baseline AP50',
    '候选框 GT 覆盖率': 'Proposal GT Coverage',
    '增益（AP50）': 'Gain (AP50)',
    '增益（AP@[0.5:0.95]）': 'Gain (AP@[0.5:0.95])',
    '推理时目标域信息': 'Inference-time Target Info',
    '方法/实验': 'Method/Experiment',
    '目标域图像': 'Target Images',
    '目标域 GT': 'Target GT',
    '目标域统计量': 'Target Statistics',
    '目标域原型': 'Target Prototypes',
    '是否参与参数选择': 'Param Selection',
    'G(粒度失配)': 'G (Granularity Mismatch)',
    'S(尺度)': 'S (Scale)',
    'I(类内变异)': 'I (Intra-class Variance)',
    'C(类间混淆)': 'C (Inter-class Confusion)',
    '口径编号': 'Protocol ID',
    '口径名称': 'Protocol Name',
    'ρ值与含义': 'ρ Value and Meaning',
    '留出域': 'Holdout Domain',
    '说明': 'Notes',
    '6类平均AP50': '6-class Mean AP50',

    # Method names
    'B2 源域原型': 'B2 Source Prototype',
    'T4 纯文本原型': 'T4 Text-only Prototype',
    'CORAL 相关性对齐': 'CORAL Alignment',
    'DANN 域对抗 (200张图, 5种子)': 'DANN Adversarial (200 imgs, 5 seeds)',
    'DANN 域对抗（200张图，5种子）': 'DANN Adversarial (200 imgs, 5 seeds)',
    'A-CSPC 固定参数 (β=1.0,λ=0.25)': 'A-CSPC Fixed (β=1.0, λ=0.25)',
    'A-CSPC 固定参数（β=1.0,λ=0.25）': 'A-CSPC Fixed (β=1.0, λ=0.25)',
    'A-CSPC 物理先验自适应（自动化先验, α=1.5）': 'A-CSPC Adaptive Prior (Auto, α=1.5)',
    'A-CSPC 物理先验自适应（自动化先验，α=1.5）': 'A-CSPC Adaptive Prior (Auto, α=1.5)',
    'A-CSPC 物理先验自适应（人工先验对照, α=1.5）': 'A-CSPC Adaptive Prior (Manual, α=1.5)',
    'A-CSPC 物理先验自适应（人工先验对照，α=1.5）': 'A-CSPC Adaptive Prior (Manual, α=1.5)',
    'A-CSPC 主配置 (β=2.0,λ=0.5)': 'A-CSPC Main (β=2.0, λ=0.5)',
    'A-CSPC 主配置（β=2.0,λ=0.5）': 'A-CSPC Main (β=2.0, λ=0.5)',
    'GDINO 基线（不重打分）': 'GDINO Baseline (no re-ranking)',
    'PE80 提示集成 [1]': 'PE80 Prompt Ensemble [1]',
    '转导自训练 TST-style [18]（adapted）': 'Transductive Self-training TST-style [18] (adapted)',
    'A-CSPC (1.0, 0.25)（目标域无关）': 'A-CSPC (1.0, 0.25) (target-agnostic)',
    'A-CSPC (2.0, 0.5)（主配置）': 'A-CSPC (2.0, 0.5) (main)',
    '纯 TGE（λ=0，仅外推正原型）': 'TGE-only (λ=0, positive extrapolation only)',
    '完整 A-CSPC（λ=0.5）': 'Full A-CSPC (λ=0.5)',

    # Data/info values
    '无': 'None',
    '无标签候选框': 'Unlabeled Proposals',
    'GT调参': 'GT-tuned',
    '上界参考，不参与主对比': 'Upper-bound reference, not in main comparison',
    '20类平均': '20-class Mean',
    '人工映射（语义对应）': 'Manual Mapping (semantic correspondence)',
    '自动映射（语义无关）': 'Auto Mapping (semantic-agnostic)',
    '全部': 'All',
    '六类均值增益': '6-class Mean Gain',
    '六类均值': '6-class Mean',

    # Reverse transfer ablation
    'A：单模板文本外推（基线）': 'A: Single-template Text Extrapolation (baseline)',
    'B：多模板方向校准': 'B: Multi-template Direction Calibration',
    'C：目标域负原型替换': 'C: Target-domain Negative Prototype Replacement',
    'D：视觉质心收缩': 'D: Visual Centroid Contraction',
    '减少13.7%损害': '13.7% damage reduction',
    '减少27.3%损害': '27.3% damage reduction',
    '损害降至≈0': 'Damage reduced to ≈0',

    # Source domains
    'DIOR（20 类池）': 'DIOR (20-class pool)',
    'NWPU VHR-10（10 类池）': 'NWPU VHR-10 (10-class pool)',
    'DOTA v1.5（16 类池 + vehicle 合并）': 'DOTA v1.5 (16-class pool + vehicle merged)',
    'DIOR→COCO（主实验）': 'DIOR→COCO (main)',

    # Negative prototype directions
    'top-3 余弦近邻（现行）': 'top-3 cosine neighbors (current)',
    '全部其他类均值': 'Mean of all other classes',
    '含正类的全部 20 类均值': 'Mean of all 20 classes (incl. positive)',
    'top-3 去掉 P_src 分量': 'top-3 with P_src component removed',
    '源类张成空间内随机组合（5 seed）': 'Random combination in source span (5 seeds)',
    'R⁵¹² 纯随机单位向量（5 seed）': 'Random unit vectors in R^512 (5 seeds)',
    '不使用负原型（λ=0）': 'No negative prototype (λ=0)',

    # Information usage audit
    'A-CSPC(2.0, 0.5)（主配置）': 'A-CSPC(2.0, 0.5) (main)',
    'A-CSPC(1.0, 0.25)（目标域无关列）': 'A-CSPC(1.0, 0.25) (target-agnostic)',
    'g2 gate': 'g2 gate',
    'AUC 诊断': 'AUC Diagnosis',
    '框池召回审计': 'Proposal Recall Audit',
    'CoOp / CLIP-Adapter / NegPrompt': 'CoOp / CLIP-Adapter / NegPrompt',
    'CoOp few-shot（组 3 上限参照）': 'CoOp few-shot (Group 3 upper-bound ref)',
    '物理先验风险预测 R(c)': 'Physical Prior Risk Prediction R(c)',
    'realized γ 逐类控制（oracle上界分析）': 'realized γ per-class control (oracle upper-bound)',
    '仅推理时': 'Inference only',
    '调参用（事后消融，§5.5）': 'Tuning use (post-hoc ablation, §5.5)',
    '调参用（AUC 峰位）': 'Tuning use (AUC peak)',
    '是：β 由目标域 GT 的事后调参确定（§5.5 披露）；λ 为协议常数': 'Yes: β tuned post-hoc on target GT (disclosed §5.5); λ is protocol constant',
    '✗（开发域 DIOR→VOC 选参，目标域无关）': 'No (params selected on DIOR→VOC dev domain, target-agnostic)',
    '✗（零阈值、纯文本、测试前固定）': 'No (zero threshold, text-only, fixed before test)',
    '✓（仅评估）': 'Yes (evaluation only)',
    '不用于选择配置': 'Not used for config selection',
    '✓（仅统计覆盖）': 'Yes (coverage statistics only)',
    '✓（仅统计）': 'Yes (statistics only)',
    '训练用源域 DIOR K=16 标注': 'Trained on source DIOR K=16 annotations',
    '✓（K=10 校准段）': 'Yes (K=10 calibration split)',
    '✓（K=10 校准）': 'Yes (K=10 calibration)',
    '校准原型本身使用目标域 GT，仅作上限参照行，明确标注': 'Calibration prototypes use target GT, upper-bound reference row only, clearly labeled',
    '✗（人工专家先验，因子集合和权重在开发集DIOR→COCO确定，测试域先验值查看': 'No (expert prior; factor set and weights determined on DIOR→COCO dev set; test-domain prior values observed)',
    '✓（计算目标域baseline AP）': 'Yes (computes target baseline AP)',
    '是：文本相似度分段阈值（0.94/0.90）和输出值（1.5/0.5/0）为基于': 'Yes: text-similarity thresholds (0.94/0.90) and output values (1.5/0.5/0) based on',

    # Rho protocols
    '全数据': 'Full Data',
    '+0.699（37类/6域对全部数据，探索性证据，CI含0）': '+0.699 (37 classes/6 domain pairs, exploratory, CI contains 0)',
    '严格冻结留出': 'Strict Frozen Holdout',
    '+0.555（2:2 split，训练集定因子权重，测试集只执行一次）': '+0.555 (2:2 split, factor weights on train, single execution on test)',
    'LOSO阈值验证平均': 'LOSO Threshold Validation Mean',
    '+0.686（4个留出源域均正相关，VisDrone子集n=4较弱）': '+0.686 (all 4 holdout source domains positive, VisDrone subset n=4 weak)',
    '6类主实验基准': '6-class Main Experiment Baseline',
    '+0.354（主实验6类子集，用于先验扰动敏感性分析）': '+0.354 (6-class main subset, for prior perturbation sensitivity analysis)',

    # Holdout domain means
    '3 域均值': '3-domain Mean',
    '4 域均值': '4-domain Mean',

    # Misc
    '飞机': 'airplane',
    '船': 'boat',
    '汽车': 'car',
    '卡车': 'truck',
    '公交车': 'bus',
    '摩托车': 'motorcycle',
}

def translate_text(text):
    """Translate Chinese text using the mapping, with partial matching."""
    if not text.strip():
        return text

    # Exact match first
    if text.strip() in translation_map:
        return translation_map[text.strip()]

    # Try to replace known phrases within the text
    result = text
    for cn, en in sorted(translation_map.items(), key=lambda x: -len(x[0])):
        if cn in result:
            result = result.replace(cn, en)

    return result

# Process all tables
total_cells = 0
translated_cells = 0

for t_idx, table in enumerate(doc.tables):
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            total_cells += 1
            original = cell.text
            if re.search(r'[\u4e00-\u9fff]', original):
                translated = translate_text(original)
                if translated != original:
                    # Update cell text while preserving formatting
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.text = ''
                    if cell.paragraphs and cell.paragraphs[0].runs:
                        cell.paragraphs[0].runs[0].text = translated
                    else:
                        cell.text = translated
                    translated_cells += 1

print(f'Total cells: {total_cells}')
print(f'Translated cells: {translated_cells}')

# Verify remaining Chinese
remaining = 0
for t_idx, table in enumerate(doc.tables):
    for r_idx, row in enumerate(table.rows):
        for c_idx, cell in enumerate(row.cells):
            if re.search(r'[\u4e00-\u9fff]', cell.text):
                remaining += 1
                if remaining <= 10:
                    print(f'  Remaining CN: Table {t_idx}, Row {r_idx}, Col {c_idx}: {cell.text[:60]}')

print(f'Remaining Chinese cells: {remaining}')

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_english_translation_v2_final.docx')
print('Saved.')
