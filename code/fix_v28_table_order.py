# -*- coding: utf-8 -*-
r"""fix_v28_table_order.py — 修复表格顺序
把表4（§4.3失效诊断）移到表5（§4.4消融）之前
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v27.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v28.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

print('第一步：定位关键元素')

# 找到§4.3标题、表4题注、表5题注的位置
sec43_elem = None
table4_caption_elem = None
table5_caption_elem = None
table4_elem = None  # 表4对应的表格元素

all_paragraphs = list(body.iterchildren(qn('w:p')))
all_tables = list(body.iterchildren(qn('w:tbl')))

for p_elem in all_paragraphs:
    text = ''.join([t.text or '' for t in p_elem.findall('.//' + qn('w:t'))])
    if '4.3' in text and '失效诊断' in text:
        sec43_elem = p_elem
        print(f'  找到§4.3标题: {text[:50]}')
    elif text.strip().startswith('表4'):
        table4_caption_elem = p_elem
        print(f'  找到表4题注: {text[:50]}')
    elif text.strip().startswith('表5'):
        table5_caption_elem = p_elem
        print(f'  找到表5题注: {text[:50]}')

# 找到表4对应的表格（第一格="层"）
for tbl_elem in all_tables:
    first_cell_text = ''
    first_row = tbl_elem.find('.//' + qn('w:tr'))
    if first_row is not None:
        first_cell = first_row.find('.//' + qn('w:tc'))
        if first_cell is not None:
            first_cell_text = ''.join([t.text or '' for t in first_cell.findall('.//' + qn('w:t'))])
    
    if first_cell_text.strip() == '层':
        table4_elem = tbl_elem
        print(f'  找到表4表格: 第一格="{first_cell_text}"')
        break

print('\n第二步：检查元素在body中的顺序')
# 获取body中所有子元素的顺序
body_children = list(body)
for i, child in enumerate(body_children):
    if child == sec43_elem:
        print(f'  §4.3标题在body子元素索引{i}')
    elif child == table4_caption_elem:
        print(f'  表4题注在body子元素索引{i}')
    elif child == table4_elem:
        print(f'  表4表格在body子元素索引{i}')
    elif child == table5_caption_elem:
        print(f'  表5题注在body子元素索引{i}')

print('\n第三步：移动表4题注和表格到§4.3标题之后')
# 先移除表4题注和表格
body.remove(table4_caption_elem)
body.remove(table4_elem)

# 插入到§4.3标题之后
# 注意：addnext会把元素插入到当前元素之后
# 我们需要先插入表格，再插入题注（因为题注在表格前面）
# 实际上，题注应该在表格前面，所以：
# 1. 先在sec43_elem后插入题注
# 2. 再在题注后插入表格

sec43_elem.addnext(table4_caption_elem)
table4_caption_elem.addnext(table4_elem)

print('  已移动表4题注和表格到§4.3标题之后')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('表格顺序修复完成')
