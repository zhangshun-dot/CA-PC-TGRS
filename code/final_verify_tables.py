# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx')
body = doc.element.body

print('=== 最终验证：所有表题与表格对应关系 ===')
table_idx = 0
table_captions = []
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        # 获取表头
        header = []
        first_row = child.find('.//' + qn('w:tr'))
        if first_row is not None:
            for cell in first_row.findall('.//' + qn('w:tc')):
                cell_text = ''.join([t.text or '' for t in cell.findall('.//' + qn('w:t'))])
                header.append(cell_text[:10])
        # 找前面的题注
        caption = ''
        j = i - 1
        while j >= 0 and body[j].tag == qn('w:p'):
            text = ''.join([t.text or '' for t in body[j].findall('.//' + qn('w:t'))])
            if text.strip().startswith('表') or text.strip().startswith('补充表'):
                caption = text[:50]
                break
            elif text.strip() == '':
                j -= 1
                continue
            else:
                break
            j -= 1
        print(f'tbl[{table_idx}]: {caption}')
        print(f'  表头: {header[:5]}')
        table_captions.append(caption)
        table_idx += 1

print(f'\n共 {table_idx} 张表格')

# 检查撞车
print('\n=== 检查表号撞车 ===')
table_nums = []
for cap in table_captions:
    match = re.match(r'(?:补充)?表\s*(\w+)', cap)
    if match:
        num = match.group(1)
        table_nums.append(num)

from collections import Counter
counts = Counter(table_nums)
duplicates = {k: v for k, v in counts.items() if v > 1}
if duplicates:
    print(f'  发现撞车: {duplicates}')
else:
    print('  无撞车 ✓')

# 检查缺题注
print('\n=== 检查缺题注 ===')
missing = [i for i, cap in enumerate(table_captions) if not cap]
if missing:
    print(f'  缺题注的表格: {missing}')
else:
    print('  所有表格均有题注 ✓')
