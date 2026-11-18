# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v27.docx')

print('=== 检查表格顺序 ===')
print(f'总表格数: {len(doc.tables)}')

# 遍历所有段落，找到表题注和表格的位置
print('\n=== 文档中表题注和表格的出现顺序 ===')
table_idx = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    # 检查表题注
    if re.match(r'^表\s*\d+', text) or re.match(r'^表SOTA', text) or re.match(r'^表VOC', text):
        print(f'段落{i}: [表题注] {text[:80]}')
    
    # 检查表格（通过检查下一个元素是否是表格）
    # 这里我们只统计表题注

print('\n=== 表格内容预览（前5行）===')
for idx, table in enumerate(doc.tables):
    # 获取表格第一行第一列的内容
    try:
        first_cell = table.rows[0].cells[0].text[:30]
    except:
        first_cell = 'N/A'
    
    # 尝试找到表格前面的题注
    # 这里我们只打印表格索引和内容
    print(f'表格{idx}: 第一格="{first_cell}" ({len(table.rows)}行x{len(table.columns)}列)')

print('\n=== 检查§4.3和§4.4的位置 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if '4.3' in text or '4.4' in text or '4.5' in text:
        if p.style.name.startswith('Heading'):
            print(f'段落{i}: [{p.style.name}] {text}')
