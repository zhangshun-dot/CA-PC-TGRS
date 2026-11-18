# -*- coding: utf-8 -*-
r"""fix_v21_figures.py — 为缺失的图片添加图题
图1（段落27）: 已有
图2（段落32）: 分层失效诊断
图3（段落75）: β崩溃点
图4（段落79）: 已有
图5（段落102）: 方向门g₂
图6（段落112）: 增益随检测器基线变化
图7（段落114）: 物理先验因子消融
图8（段落115）: Realized γ vs. Oracle γ
图9（段落118）: γ增益曲线
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'

doc = Document(PAPER_PATH)

# 图题定义：段落索引 -> 图题文本
figure_captions = {
    32: '图2　分层失效诊断：排序AUC与框池召回须分开报告（P1诊断区分排序质量与候选框覆盖；6类平均排序AUC 0.576，框池GT覆盖率0.743）',
    75: '图3　β崩溃点与源域自治往返残差拐点对齐（β=2安全、β=4爆炸；左：目标域AUC随β变化，右：源域自治往返残差随β变化，两者拐点均在β=2–4之间）',
    102: '图5　方向门g₂在三策略中的选择（19类，外推13类，决策准确率16/19；左：全外推/全B2/gate三策略均值对比，右：19类外推−B2增益，蓝=gate选对，红=误判）',
    112: '图6　增益随检测器基线与框池饱和度的变化：弱校准才有收益（GDINO-tiny蓝点均为正增益，OWLv2橙点均为负增益；横轴检测器基线AP50越左越弱，纵轴A-CSPC逐类增益）',
    114: '图7　物理先验因子消融：各因子对预测相关性的贡献（37类/6域对，Spearman ρ；Full=V+M+G+S+I+C六因子，w/o G下降最大Δ=−0.117，说明粒度失配G为核心因子）',
    115: '图8　Realized γ vs. Oracle γ vs. A-CSPC逐类增益对比（左：DIOR→COCO六类，右：DIOR→VOC五类；A-CSPC(γ=0)为基准，Oracle γ为逐类最优γ上界，Realized γ为文本相似度分段+域饱和系数的GT辅助控制）',
    118: '图9　细粒度源原型增益随γ变化曲线（DIOR→COCO；γ为细原型偏移强度，car/oracle γ=1.5、truck/oracle γ=0.25、bus/oracle γ=0.5、motorcycle/oracle γ=0.0；★为realized γ位置）',
}

def insert_caption_after(paragraph, caption_text):
    """在指定段落后插入图题段落"""
    new_p = paragraph._element.makeelement(qn('w:p'), {})
    paragraph._element.addnext(new_p)
    run = new_p.makeelement(qn('w:r'), {})
    new_p.append(run)
    t = run.makeelement(qn('w:t'), {})
    t.text = caption_text
    t.set(qn('xml:space'), 'preserve')
    run.append(t)
    # 设置字号
    rPr = run.makeelement(qn('w:rPr'), {})
    run.insert(0, rPr)
    sz = rPr.makeelement(qn('w:sz'), {qn('w:val'): '20'})  # 10pt
    rPr.append(sz)
    szCs = rPr.makeelement(qn('w:szCs'), {qn('w:val'): '20'})
    rPr.append(szCs)
    return new_p

# 按段落索引从大到小插入，避免索引偏移
for para_idx in sorted(figure_captions.keys(), reverse=True):
    caption_text = figure_captions[para_idx]
    if para_idx < len(doc.paragraphs):
        p = doc.paragraphs[para_idx]
        # 检查后一段是否已经有图题
        next_p = doc.paragraphs[para_idx + 1] if para_idx + 1 < len(doc.paragraphs) else None
        if next_p and next_p.text.strip().startswith('图'):
            print(f'段落{para_idx}后已有图题: {next_p.text[:50]}')
        else:
            insert_caption_after(p, caption_text)
            print(f'段落{para_idx}后插入图题: {caption_text[:50]}...')
    else:
        print(f'段落{para_idx}超出范围')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('图题添加完成')
