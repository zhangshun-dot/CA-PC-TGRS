# -*- coding: utf-8 -*-
r"""fix_v21_p2.py — 修复§6.1数值冲突 + 检查所有表格内容
"""
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx'

doc = Document(PAPER_PATH)

# ══════════════════════════ 修复§6.1数值冲突 ══════════════════════════
print('修复§6.1数值冲突')
p106 = doc.paragraphs[106]
old_text = p106.text
print(f'  旧: {old_text[:200]}')

# 修正数值：airplane+0.113 → +0.0117，motorcycle+0.038 → +0.1189
# 同时调整表述，保持"类别间差异显著"的论点
new_text = old_text.replace(
    'airplane增益+0.113，motorcycle仅+0.038',
    'airplane增益仅+0.0117，而motorcycle达+0.1189（后者是前者的10.2倍）'
)

for run in p106.runs:
    run.text = ''
if p106.runs:
    p106.runs[0].text = new_text
else:
    p106.add_run(new_text)
print(f'  新: {new_text[:200]}')

# ══════════════════════════ 检查所有表格内容 ══════════════════════════
print('\n=== 所有表格内容概览 ===')
for i, table in enumerate(doc.tables):
    # 获取第一行作为表头
    header = []
    if table.rows:
        for cell in table.rows[0].cells:
            header.append(cell.text.strip()[:20])
    # 获取行数和列数
    n_rows = len(table.rows)
    n_cols = len(table.columns) if table.columns else 0
    print(f'  表{i}: {n_rows}行×{n_cols}列, 表头: {header}')

# ══════════════════════════ 检查所有图引用 ══════════════════════════
print('\n=== 所有图引用 ===')
import re
for i, p in enumerate(doc.paragraphs):
    refs = re.findall(r'图\s*(\d+)', p.text)
    if refs:
        print(f'  段落{i}: 图{refs}, {p.text[:100]}')

# 检查内联图片
print(f'\n文档内联图片数量: {len(doc.inline_shapes)}')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
