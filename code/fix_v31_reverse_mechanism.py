# -*- coding: utf-8 -*-
r"""fix_v31_reverse_mechanism.py — 修改反向实验表述为"机制验证"
"""
from docx import Document
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v30.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v31.docx'

doc = Document(PAPER_PATH)

print('=' * 60)
print('修改反向实验表述为"机制验证"')
print('=' * 60)

# 段落66：§4.5标题 - 已经写了"组件归因"，保持不变
p66 = doc.paragraphs[66]
print(f'\n段落66标题: {p66.text}')

# 段落67：反向实验开头 - 修改表述
p67 = doc.paragraphs[67]
old_p67 = p67.text
print(f'\n原文段落67:\n{old_p67[:400]}')

# 修改：把"反向实验呈现相反规律"改成更中性的"机制验证"表述
new_p67 = old_p67.replace(
    '反向实验呈现相反规律',
    '反向迁移实验用于验证A-CSPC外推组件的方向选择性（机制验证而非方法失败）：结果呈现相反规律'
)

for run in p67.runs:
    run.text = ''
if p67.runs:
    p67.runs[0].text = new_p67
else:
    p67.add_run(new_p67)
print(f'\n新文段落67:\n{new_p67[:400]}')

print('\n' + '=' * 60)
print('搜索§5.7反向组件归因实验')
print('=' * 60)

# 搜索§5.7或反向组件归因相关段落
for i in range(100, 150):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text
        if '5.7' in text or '组件归因' in text or '方向非对称' in text or '文本外推方向' in text:
            print(f'\n段落{i} [{doc.paragraphs[i].style.name}]:')
            print(f'  {text[:200]}')

print('\n' + '=' * 60)
print('搜索结论中的反向实验表述')
print('=' * 60)

for i in range(145, 160):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text
        if '反向' in text or 'photo→overhead' in text or '方向失配' in text:
            print(f'\n段落{i} [{doc.paragraphs[i].style.name}]:')
            print(f'  {text[:200]}')

doc.save(OUTPUT_PATH)
print(f'\n\n论文已保存: {OUTPUT_PATH}')
print('反向实验表述修改完成（第一部分）')
