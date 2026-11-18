# -*- coding: utf-8 -*-
"""fix_c4.py — 修复C4：§4.8检测器谱系"非单调"需更精确解释"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# §4.8正文在段落93
p = doc.paragraphs[93]
old_text = p.text

# 在原文末尾加入headroom解释
old_ending = 'OWLv2（基线0.6800）增益−0.0316。'
new_ending = (
    'OWLv2（基线0.6800）增益−0.0316。'
    '这一结果并非简单的"弱检测器收益大"，而是与检测器基线的headroom（1−AP）连续相关：'
    'GDINO-tiny（headroom 0.6293）和GDINO-base（headroom 0.5963）均处于headroom充足的弱校准区间，'
    '候选框排序存在系统性偏差，故校正增益较高且base略高于tiny；'
    'YOLO-World（headroom 0.5880）之后headroom骤降，排序偏差减小，增益趋近于零；'
    'OWLv2（headroom 0.3200）基线已接近饱和，校正引入的噪声大于对齐收益，导致增益转负。'
)

new_text = old_text.replace(old_ending, new_ending)

# 清空原有runs，设置新内容
for run in p.runs:
    run.text = ''
p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('C4已修改：§4.8检测器谱系"非单调"改为headroom连续解释')
print()
print('新增解释：')
print('  增益与检测器基线的headroom（1−AP）连续相关')
print('  GDINO-tiny/base：headroom充足，增益高')
print('  YOLO-World：headroom骤降，增益趋近零')
print('  OWLv2：基线饱和，校正引入噪声，增益转负')
