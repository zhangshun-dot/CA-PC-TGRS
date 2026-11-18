# -*- coding: utf-8 -*-
r"""add_sota_notarget_to_paper.py — 将无目标域数据版本SOTA对比写入论文
在§4.2 SOTA对比段落之后补充无目标域数据版本的结果
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt
from lxml import etree

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v14.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v15.docx'

doc = Document(PAPER_PATH)

# 找到SOTA对比表注的位置
sota_caption_idx = None
for i, p in enumerate(doc.paragraphs):
    if '表SOTA' in p.text and '现有域自适应重排序方法' in p.text:
        sota_caption_idx = i
        break

print(f'SOTA表注在段落{sota_caption_idx}')

if sota_caption_idx is None:
    print('未找到SOTA表注')
    exit(1)

# 在SOTA表注之后插入无目标域数据版本的内容
sota_caption_elem = doc.paragraphs[sota_caption_idx]._element

# 无目标域数据版本段落
notarget_text = (
    '完全无目标域数据设置下的对比（信息预算与A-CSPC目标域无关配置一致，均不使用目标域任何数据）：'
    'CORAL源域白化（仅用源域协方差矩阵做特征标准化，不着色目标域）取得+0.0367增益，'
    'A-CSPC目标域无关配置（β=1.0, λ=0.25，DIOR→VOC开发域选参）取得+0.0167增益，'
    'B2源域原型直接迁移为−0.0031，DANN源域有监督特征提取灾难性失败（−0.1388）。'
    '值得注意的是，CORAL白化在无目标域数据设置下反而优于其使用目标域无标签数据的版本（+0.0367 vs +0.0210），'
    '说明在CLIP特征空间中，目标域无标签数据可能引入噪声，而源域特征标准化本身就是有效的预处理。'
    'DANN无目标域版的失败则表明，仅用源域类别标签训练的有监督特征提取器会过拟合于源域类别结构，无法迁移到目标域。'
    'A-CSPC目标域无关配置在完全无目标域数据的设置下仍取得正增益，验证了非对称原型校正的鲁棒性。'
    '逐类结果见表SOTA-NoTarget。'
)

# 创建新段落
new_p = etree.SubElement(sota_caption_elem.getparent(), qn('w:p'))
sota_caption_elem.addnext(new_p)
run = etree.SubElement(new_p, qn('w:r'))
t = etree.SubElement(run, qn('w:t'))
t.text = notarget_text
t.set(qn('xml:space'), 'preserve')
print('已插入无目标域数据版本段落')

# 创建无目标域数据版本表格
notarget_data = [
    ['方法', 'airplane', 'boat', 'car', 'truck', 'bus', 'motorcycle', '6类平均', '平均增益'],
    ['Baseline (GDINO-tiny)', '0.7484', '0.1758', '0.1245', '0.2345', '0.5317', '0.4092', '0.3707', '—'],
    ['B2 源域原型', '0.7579', '0.1650', '0.1063', '0.2498', '0.4909', '0.4357', '0.3676', '−0.0031'],
    ['A-CSPC 目标域无关 (β=1.0,λ=0.25)', '0.7598', '0.1659', '0.1270', '0.2614', '0.5326', '0.4775', '0.3873', '+0.0167'],
    ['DANN 无目标域 (源域有监督特征提取)', '0.2930', '0.1844', '0.0771', '0.1279', '0.4507', '0.2586', '0.2319', '−0.1388'],
    ['CORAL 无目标域 (源域白化)', '0.7607', '0.1899', '0.1492', '0.2808', '0.5747', '0.4888', '0.4074', '+0.0367'],
]

table = doc.add_table(rows=len(notarget_data), cols=len(notarget_data[0]))
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
for i, row_data in enumerate(notarget_data):
    row = table.rows[i]
    for j, cell_text in enumerate(row_data):
        cell = row.cells[j]
        cell.text = cell_text
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(9)
                if i == 0:
                    run.font.bold = True

# 将表格移动到无目标域段落之后
table_elem = table._element
table_elem.getparent().remove(table_elem)
new_p.addnext(table_elem)
print('已插入无目标域数据版本表格')

# 在表格后插入表注
caption_p = etree.SubElement(table_elem.getparent(), qn('w:p'))
table_elem.addnext(caption_p)
caption_run = etree.SubElement(caption_p, qn('w:r'))
caption_t = etree.SubElement(caption_run, qn('w:t'))
caption_t.text = '表SOTA-NoTarget　完全无目标域数据设置下的对比（所有方法均不使用目标域任何数据，包括无标签数据；AP50，11点插值）'
caption_t.set(qn('xml:space'), 'preserve')
print('已插入表注')

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('无目标域数据版本SOTA对比已写入§4.2')
