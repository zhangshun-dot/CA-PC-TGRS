# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v28.docx')

print('=== 搜索所有表格中的CDVM ===')
for idx, table in enumerate(doc.tables):
    for row_idx, row in enumerate(table.rows):
        for cell_idx, cell in enumerate(row.cells):
            if 'CDVM' in cell.text:
                print(f'表格{idx}, 行{row_idx}, 列{cell_idx}: {cell.text[:100]}')
                # 打印整个表格的内容
                print(f'\n表格{idx}完整内容:')
                for r_idx, r in enumerate(table.rows):
                    row_text = [c.text.strip()[:25] for c in r.cells]
                    print(f'  行{r_idx}: {" | ".join(row_text)}')

print('\n=== 搜索TGE相关内容 ===')
for i, p in enumerate(doc.paragraphs):
    if 'TGE' in p.text:
        print(f'段落{i}: {p.text[:150]}')

print('\n=== 检查段落23的引用编号 ===')
print(f'段落23: {doc.paragraphs[23].text[:200]}')

print('\n=== 检查参考文献[10][13][14][15][16][18][31] ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    match = re.match(r'\[(\d+)\]', text)
    if match:
        ref_num = int(match.group(1))
        if ref_num in [10, 13, 14, 15, 16, 18, 31]:
            print(f'[{ref_num}] {text[:100]}')
