# -*- coding: utf-8 -*-
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v28.docx')

print('=== 表格13完整内容 ===')
table = doc.tables[12]  # 表格13（索引12）
print(f'表格13: {len(table.rows)}行 x {len(table.columns)}列')
print()

for row_idx, row in enumerate(table.rows):
    row_text = []
    for cell_idx, cell in enumerate(row.cells):
        text = cell.text.strip()[:30]
        row_text.append(f'[{cell_idx}]{text}')
    print(f'行{row_idx}: {" | ".join(row_text)}')

print('\n=== 检查CoOp相关参考文献 ===')
# 搜索参考文献中的CoOp
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'CoOp' in text or 'Learning to prompt' in text:
        print(f'段落{i}: {text[:120]}')

print('\n=== 检查CLIP-Adapter相关参考文献 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'CLIP-Adapter' in text or 'CLIP_Adapter' in text:
        print(f'段落{i}: {text[:120]}')

print('\n=== 检查Tip-Adapter相关参考文献 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'Tip-Adapter' in text or 'Tip_Adapter' in text:
        print(f'段落{i}: {text[:120]}')
