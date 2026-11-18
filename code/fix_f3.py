# -*- coding: utf-8 -*-
"""fix_f3.py — 修复F3：§4.9标题不明确"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

p = doc.paragraphs[98]
old_text = p.text
new_text = old_text.replace('4.9 配置选择审计', '4.9 超参数来源与信息使用协议')

for run in p.runs:
    run.text = ''
p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('F3已完成：§4.9标题改为"超参数来源与信息使用协议"')
print(f'  原标题：{old_text}')
print(f'  新标题：{new_text}')
