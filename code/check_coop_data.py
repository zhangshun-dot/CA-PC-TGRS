# -*- coding: utf-8 -*-
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v29.docx')

print('=== 表格14（信息使用审计表）完整内容 ===')
table14 = doc.tables[13]
print(f'表格14: {len(table14.rows)}行 x {len(table14.columns)}列')
print()

for row_idx, row in enumerate(table14.rows):
    print(f'--- 行{row_idx} ---')
    for cell_idx, cell in enumerate(row.cells):
        text = cell.text.strip()
        print(f'  列{cell_idx}: {text[:80]}')
    print()

print('\n=== 搜索论文中CoOp的实验结果 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'CoOp' in text and ('结果' in text or '增益' in text or 'AP' in text or '实验' in text):
        print(f'段落{i}: {text[:150]}')

print('\n=== 搜索论文中few-shot相关内容 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'few-shot' in text.lower() or 'few shot' in text.lower():
        print(f'段落{i}: {text[:150]}')

print('\n=== 搜索表格中CoOp的出现 ===')
for idx, table in enumerate(doc.tables):
    for row_idx, row in enumerate(table.rows):
        for cell_idx, cell in enumerate(row.cells):
            if 'CoOp' in cell.text:
                print(f'表格{idx}, 行{row_idx}, 列{cell_idx}: {cell.text[:100]}')
