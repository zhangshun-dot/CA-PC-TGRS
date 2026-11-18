# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v28.docx')

print('=== 搜索CDVM出现位置 ===')
count = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'CDVM' in text:
        count += 1
        print(f'\n段落{i}:')
        print(f'  内容: {text[:200]}')
        # 高亮CDVM位置
        idx = text.find('CDVM')
        print(f'  CDVM位置: 第{idx}个字符')
        print(f'  上下文: ...{text[max(0,idx-30):idx+50]}...')

print(f'\n共找到 {count} 处CDVM')

# 也搜索表格中的CDVM
print('\n=== 搜索表格中的CDVM ===')
for idx, table in enumerate(doc.tables):
    for row_idx, row in enumerate(table.rows):
        for cell_idx, cell in enumerate(row.cells):
            if 'CDVM' in cell.text:
                print(f'表格{idx}, 行{row_idx}, 列{cell_idx}: {cell.text[:100]}')
