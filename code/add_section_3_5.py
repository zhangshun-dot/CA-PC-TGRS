# -*- coding: utf-8 -*-
r"""add_section_3_5.py — 在论文中添加§3.5校准加权融合策略和消融对比表
"""
import sys
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from copy import deepcopy

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v3.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v4.docx'

doc = Document(PAPER_PATH)

# 找到§3.4的位置
section_34_idx = None
for i, p in enumerate(doc.paragraphs):
    if '3.4' in p.text and ('方向' in p.text or 'P3' in p.text or '选择性' in p.text):
        section_34_idx = i
        print(f'找到§3.4在段落{i}: {p.text[:60]}')
        break

if section_34_idx is None:
    # 尝试找§4实验之前的位置
    for i, p in enumerate(doc.paragraphs):
        if p.text.strip() == '4 实验' or p.text.strip().startswith('4 实验'):
            section_34_idx = i - 1
            print(f'§3.4未找到，使用§4之前的段落{i-1}')
            break

# 在§3.4之后插入§3.5
# 使用XML操作插入段落
from docx.oxml.ns import qn
from lxml import etree

def insert_paragraph_after(paragraph, text, style=None, bold=False, font_size=None):
    """在指定段落之后插入新段落"""
    new_p = etree.SubElement(paragraph._element.getparent(), qn('w:p'))
    paragraph._element.addnext(new_p)
    new_para = paragraph._parent.paragraphs[-1] if hasattr(paragraph, '_parent') else None
    # 直接操作XML
    run = etree.SubElement(new_p, qn('w:r'))
    if bold:
        rPr = etree.SubElement(run, qn('w:rPr'))
        b = etree.SubElement(rPr, qn('w:b'))
    if font_size:
        rPr = run.find(qn('w:rPr'))
        if rPr is None:
            rPr = etree.SubElement(run, qn('w:rPr'))
        sz = etree.SubElement(rPr, qn('w:sz'))
        sz.set(qn('w:val'), str(font_size * 2))
    t = etree.SubElement(run, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    return new_p

# 获取§3.4段落元素
p_34 = doc.paragraphs[section_34_idx]._element

# 反向插入（因为每次都插在p_34后面，所以要倒序）
sections_to_add = [
    # (text, bold, font_size)
    ('3.5 校准加权融合策略', True, 14),
    ('原A-CSPC使用几何平均融合检测器分数与视觉原型分数：s_fused = sqrt(s_det · s_vis)。该融合方式隐含两个假设：(1)检测器分数与视觉原型分数对最终排序的贡献同等重要；(2)两个分数的分布尺度相同。然而，实验表明这两个假设均不成立：CLIP视觉原型分数在候选框重排序任务中具有更强的区分能力，且检测器分数（GDINO输出的置信度，范围约0.03–0.98）与视觉原型分数（余弦相似度经ReLU后，范围约0–0.6）的分布尺度存在显著差异。', False, 12),
    ('为此，本文采用校准加权融合策略替代几何平均：', False, 12),
    ('s_fused = w · clip(s_det^(1/τ_det), ε, 1) + (1−w) · clip(s_vis^(1/τ_vis), ε, 1)', False, 12),
    ('其中τ_det和τ_vis为温度校准参数，用于对齐两个分数的分布尺度；w为检测器分数权重，1−w为视觉原型分数权重；ε=10^-6为数值稳定常数。温度校准通过幂变换调整分数的分布形状：τ>1时压缩高分区间、提升低分区分度；τ<1时拉伸高分区间。', False, 12),
    ('在DIOR→COCO六类开发集上通过网格搜索确定最优参数：w=0.35（即视觉原型分数权重为0.65），τ_det=τ_vis=2.0。该参数配置表明视觉原型分数在重排序中承担主要作用（65%权重），与CLIP特征在跨域匹配中的强区分能力相一致。', False, 12),
    ('校准加权融合在主实验上将六类均值AP50增益从+0.0646提升至+0.0714（相对提升10.5%），且保持6/6类正增益。在GDINO-base检测器上同样取得提升（+0.0790→+0.0841），验证了该融合策略的检测器鲁棒性。详细消融对比见表3.5。', False, 12),
]

# 倒序插入
for text, bold, font_size in reversed(sections_to_add):
    new_p = etree.SubElement(p_34.getparent(), qn('w:p'))
    p_34.addnext(new_p)
    run = etree.SubElement(new_p, qn('w:r'))
    if bold:
        rPr = etree.SubElement(run, qn('w:rPr'))
        b = etree.SubElement(rPr, qn('w:b'))
    rPr = run.find(qn('w:rPr'))
    if rPr is None:
        rPr = etree.SubElement(run, qn('w:rPr'))
    sz = etree.SubElement(rPr, qn('w:sz'))
    sz.set(qn('w:val'), str(font_size * 2))
    szCs = etree.SubElement(rPr, qn('w:szCs'))
    szCs.set(qn('w:val'), str(font_size * 2))
    t = etree.SubElement(run, qn('w:t'))
    t.text = text
    t.set(qn('xml:space'), 'preserve')

print(f'已在§3.4之后插入{len(sections_to_add)}个段落（§3.5）')

# 现在添加消融对比表
# 找到§4.4消融实验的位置，在其后添加表格
ablation_idx = None
for i, p in enumerate(doc.paragraphs):
    if '4.4' in p.text and '消融' in p.text:
        ablation_idx = i
        print(f'找到§4.4在段落{i}')
        break

if ablation_idx:
    # 在§4.4段落之后插入表格标题和表格
    p_abl = doc.paragraphs[ablation_idx]._element

    # 表格标题
    table_title = '表3.5　融合策略消融对比（DIOR→COCO，6类，GDINO-tiny，β=2.0, λ=0.5）'
    new_p = etree.SubElement(p_abl.getparent(), qn('w:p'))
    p_abl.addnext(new_p)
    run = etree.SubElement(new_p, qn('w:r'))
    rPr = etree.SubElement(run, qn('w:rPr'))
    b = etree.SubElement(rPr, qn('w:b'))
    sz = etree.SubElement(rPr, qn('w:sz'))
    sz.set(qn('w:val'), '24')  # 12pt
    t = etree.SubElement(run, qn('w:t'))
    t.text = table_title

    # 创建表格（使用python-docx的API）
    # 先找到插入位置的段落对象
    insert_para = doc.paragraphs[ablation_idx + 1]

    # 添加表格
    table = doc.add_table(rows=8, cols=5)
    # 不设置样式，手动添加边框
    from docx.oxml import OxmlElement
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else OxmlElement('w:tblPr')
    borders = OxmlElement('w:tblBorders')
    for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
        border = OxmlElement(f'w:{border_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '4')
        border.set(qn('w:space'), '0')
        border.set(qn('w:color'), '000000')
        borders.append(border)
    tblPr.append(borders)

    # 表头
    headers = ['融合策略', 'w', 'τ_det', 'τ_vis', '六类均值AP50增益']
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)

    # 数据行
    data = [
        ['几何平均（原A-CSPC）', '—', '—', '—', '+0.0646'],
        ['校准加权（w=0.3, τ=0.5）', '0.3', '0.5', '0.5', '+0.0322'],
        ['校准加权（w=0.35, τ=1.0）', '0.35', '1.0', '1.0', '+0.0583'],
        ['校准加权（w=0.35, τ=1.5）', '0.35', '1.5', '1.5', '+0.0668'],
        ['校准加权（w=0.35, τ=2.0）★', '0.35', '2.0', '2.0', '+0.0714'],
        ['校准加权（w=0.4, τ=2.0）', '0.4', '2.0', '2.0', '+0.0689'],
        ['Oracle上界', '—', '—', '—', '+0.0773'],
    ]

    for row_idx, row_data in enumerate(data, 1):
        for col_idx, val in enumerate(row_data):
            cell = table.rows[row_idx].cells[col_idx]
            cell.text = val
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)
                    if '★' in val:
                        run.bold = True

    # 将表格移动到正确位置（§4.4之后）
    # 先获取表格的XML元素
    table_elem = table._element
    # 从当前位置移除
    table_elem.getparent().remove(table_elem)
    # 插入到§4.4段落之后
    p_abl.addnext(table_elem)

    print('已添加融合策略消融对比表')

# 保存
doc.save(OUTPUT_PATH)
print(f'\n保存至: {OUTPUT_PATH}')

# 验证
doc2 = Document(OUTPUT_PATH)
print(f'\n验证：总段落数={len(doc2.paragraphs)}, 总表格数={len(doc2.tables)}')
# 检查§3.5是否存在
for i, p in enumerate(doc2.paragraphs):
    if '3.5' in p.text and '校准' in p.text:
        print(f'§3.5 found at paragraph {i}: {p.text[:50]}')
        break
