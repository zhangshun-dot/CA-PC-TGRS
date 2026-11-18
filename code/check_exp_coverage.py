# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v31.docx')

print('=== 检查AP@[.5:.95]指标 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if '0.5:0.95' in text or '0.5:.95' in text or 'AP@[' in text:
        print(f'\n段落{i}: {text[:200]}')

print('\n\n=== 检查表格中的AP指标 ===')
for idx, table in enumerate(doc.tables):
    for row_idx, row in enumerate(table.rows):
        for cell_idx, cell in enumerate(row.cells):
            if '0.5:0.95' in cell.text or '0.5:.95' in cell.text:
                print(f'\n表格{idx}, 行{row_idx}, 列{cell_idx}: {cell.text[:100]}')

print('\n\n=== 检查所有图题注 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if re.match(r'^图\s*\d+', text):
        print(f'段落{i}: {text[:120]}')

print('\n\n=== 检查所有表题注 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if re.match(r'^表\s*\d+', text) or re.match(r'^表SOTA', text) or re.match(r'^表VOC', text):
        print(f'段落{i}: {text[:120]}')

print('\n\n=== 检查实验数据目录 ===')
import os
exp_data_dir = r'C:\Users\张三\Desktop\实验数据'
if os.path.exists(exp_data_dir):
    files = os.listdir(exp_data_dir)
    json_files = [f for f in files if f.endswith('.json')]
    print(f'实验数据目录中有 {len(json_files)} 个JSON文件:')
    for f in sorted(json_files):
        print(f'  {f}')
