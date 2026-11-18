# -*- coding: utf-8 -*-
"""fix_l2.py — 修复L2：§2.2相邻两句语义重复"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 段落20：删掉末尾重复的"本文的A-CSPC..."
p20 = doc.paragraphs[20]
old_text_20 = p20.text
new_text_20 = old_text_20.replace(
    '本文的A-CSPC在原型外推基础上引入非对称构造，并通过分层失效诊断明确外推的适用边界。',
    ''
)
for run in p20.runs:
    run.text = ''
p20.runs[0].text = new_text_20

# 段落21：删掉开头重复的"原型方法通过聚合源域标注框特征构建类别原型，在目标域候选框上进行重排序。"
p21 = doc.paragraphs[21]
old_text_21 = p21.text
new_text_21 = old_text_21.replace(
    '原型方法通过聚合源域标注框特征构建类别原型，在目标域候选框上进行重排序。',
    ''
)
for run in p21.runs:
    run.text = ''
p21.runs[0].text = new_text_21

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('L2已修改：§2.2删除了两句重复表述')
print()
print('修改后段落20末尾：')
print(new_text_20[-100:])
print()
print('修改后段落21：')
print(new_text_21)
