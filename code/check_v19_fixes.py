# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v19.docx')

print('=== 检查1: 「显著为负」/「整体损害性能」表述 ===')
for i, p in enumerate(doc.paragraphs):
    if '显著为负' in p.text or '整体损害性能' in p.text or '显著损害' in p.text:
        print(f'  段落{i}: {p.text[:200]}')

print('\n=== 检查2: DANN行数值 ===')
for i, table in enumerate(doc.tables):
    for row in table.rows:
        first_cell = row.cells[0].text.strip()
        if 'DANN' in first_cell:
            cells = [c.text.strip() for c in row.cells]
            print(f'  表格{i}, DANN行: {cells}')

print('\n=== 检查3: 表注T4身份说明 ===')
for i, p in enumerate(doc.paragraphs):
    if '表SOTA-a' in p.text:
        print(f'  段落{i}: {p.text[:300]}')

print('\n=== 检查4: 扩类别验证是否存在 ===')
for i, p in enumerate(doc.paragraphs):
    if 'VOC→COCO' in p.text or '扩类别' in p.text or '20类' in p.text:
        print(f'  段落{i}: {p.text[:120]}')
