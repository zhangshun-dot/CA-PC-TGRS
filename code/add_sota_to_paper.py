# -*- coding: utf-8 -*-
r"""add_sota_to_paper.py — 将SOTA对比实验结果写入论文
在§4.2主结果之后插入：
  1. SOTA对比段落（DANN/CORAL等域自适应方法对比）
  2. SOTA对比表格
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, Inches
from lxml import etree

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v13.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v14.docx'

doc = Document(PAPER_PATH)

# 找到§4.2主结果段落（段落45）
main_result_idx = None
for i, p in enumerate(doc.paragraphs):
    if '主结果（DIOR→COCO' in p.text and '+0.0646' in p.text:
        main_result_idx = i
        break

print(f'主结果段落: {main_result_idx}')

if main_result_idx is None:
    print('未找到主结果段落')
    exit(1)

# 在主结果段落之后插入SOTA对比内容
main_result_elem = doc.paragraphs[main_result_idx]._element

# SOTA对比段落
sota_text = (
    '与现有域自适应重排序方法的对比（相同候选框池、相同CLIP特征、公平协议）：'
    '在6类平均上，A-CSPC取得+0.0646增益，显著优于纯文本原型重排序（T4，+0.0357）、'
    'CORAL二阶统计量对齐（+0.0210）和源域原型直接迁移（B2，−0.0031）。'
    '值得注意的是，DANN域对抗特征对齐在该设置下反而损害性能（−0.0564），'
    '说明在CLIP特征空间中，通过Gradient Reversal Layer强制域不变会破坏类别判别信息；'
    '而A-CSPC的非对称原型校正（正原型沿文本方向外推、负原型保留源域空间）'
    '在保留类别判别性的同时适应域偏移，因此取得最优结果。'
    '逐类结果见表SOTA（6类逐类增益对比）。'
)

# 创建新段落
new_p = etree.SubElement(main_result_elem.getparent(), qn('w:p'))
main_result_elem.addnext(new_p)
run = etree.SubElement(new_p, qn('w:r'))
t = etree.SubElement(run, qn('w:t'))
t.text = sota_text
t.set(qn('xml:space'), 'preserve')
print('已插入SOTA对比段落')

# 创建SOTA对比表格
# 表格数据
sota_data = [
    ['方法', 'airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle', '6类平均', '平均增益'],
    ['Baseline (GDINO-tiny)', '0.7484', '0.1758', '0.1245', '0.2345', '0.5317', '0.4092', '0.3707', '—'],
    ['B2 源域原型', '0.7579', '0.1650', '0.1063', '0.2498', '0.4909', '0.4357', '0.3676', '−0.0031'],
    ['T4 纯文本原型', '0.7637', '0.1904', '0.1466', '0.2770', '0.5733', '0.4876', '0.4064', '+0.0357'],
    ['A-CSPC (本文)', '0.7601', '0.2011', '0.2054', '0.3119', '0.6052', '0.5281', '0.4353', '+0.0646'],
    ['DANN 域对抗', '0.7484', '0.1450', '0.1085', '0.1409', '0.3209', '0.4222', '0.3143', '−0.0564'],
    ['CORAL 相关性对齐', '0.7495', '0.1898', '0.1268', '0.2653', '0.5774', '0.4415', '0.3917', '+0.0210'],
]

# 创建表格XML
table = doc.add_table(rows=len(sota_data), cols=len(sota_data[0]))
# 手动设置表格边框
tbl = table._tbl
tblPr = tbl.tblPr if tbl.tblPr is not None else etree.SubElement(tbl, qn('w:tblPr'))
borders = etree.SubElement(tblPr, qn('w:tblBorders'))
for border_name in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
    border = etree.SubElement(borders, qn(f'w:{border_name}'))
    border.set(qn('w:val'), 'single')
    border.set(qn('w:sz'), '4')
    border.set(qn('w:space'), '0')
    border.set(qn('w:color'), '000000')

# 填充表格
for i, row_data in enumerate(sota_data):
    row = table.rows[i]
    for j, cell_text in enumerate(row_data):
        cell = row.cells[j]
        cell.text = cell_text
        # 设置字体
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(9)
                if i == 0:  # 表头加粗
                    run.font.bold = True

# 将表格移动到SOTA段落之后
table_elem = table._element
table_elem.getparent().remove(table_elem)
new_p.addnext(table_elem)
print('已插入SOTA对比表格')

# 在表格后插入表注
caption_p = etree.SubElement(table_elem.getparent(), qn('w:p'))
table_elem.addnext(caption_p)
caption_run = etree.SubElement(caption_p, qn('w:r'))
caption_t = etree.SubElement(caption_run, qn('w:t'))
caption_t.text = '表SOTA　与现有域自适应重排序方法的对比（相同候选框池、相同CLIP特征、公平协议；AP50，11点插值）'
caption_t.set(qn('xml:space'), 'preserve')
print('已插入表注')

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('SOTA对比实验结果已写入§4.2')
