# -*- coding: utf-8 -*-
"""fix_f1.py — 修复F1：表格命名不标准（表SOTA-a/b → 表2a/b）"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

count = 0
for i, p in enumerate(doc.paragraphs):
    old_text = p.text
    new_text = old_text
    if '表SOTA-a' in new_text:
        new_text = new_text.replace('表SOTA-a', '表2a')
        count += 1
    if '表SOTA-b' in new_text:
        new_text = new_text.replace('表SOTA-b', '表2b')
        count += 1
    if new_text != old_text:
        for run in p.runs:
            run.text = ''
        p.runs[0].text = new_text
        print(f'  [{i}] 替换: {old_text[:60]}... → {new_text[:60]}...')

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print(f'\nF1已完成：共替换{count}处表格命名')
print('  表SOTA-a → 表2a')
print('  表SOTA-b → 表2b')
