# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx')
body = doc.element.body

print('=== 所有表题段落 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if text.startswith('表') and len(text) > 2:
        print(f'段落{i}: {text[:80]}')

print('\n=== 表格与前后段落 ===')
table_idx = 0
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        # 获取表头
        header = []
        first_row = child.find('.//' + qn('w:tr'))
        if first_row is not None:
            for cell in first_row.findall('.//' + qn('w:tc')):
                cell_text = ''.join([t.text or '' for t in cell.findall('.//' + qn('w:t'))])
                header.append(cell_text[:15])
        # 前后段落
        prev_text = ''
        next_text = ''
        if i > 0 and body[i-1].tag == qn('w:p'):
            prev_text = ''.join([t.text or '' for t in body[i-1].findall('.//' + qn('w:t'))])[:50]
        if i < len(body)-1 and body[i+1].tag == qn('w:p'):
            next_text = ''.join([t.text or '' for t in body[i+1].findall('.//' + qn('w:t'))])[:50]
        print(f'tbl[{table_idx}]: 表头={header}')
        print(f'  前段: {prev_text}')
        print(f'  后段: {next_text}')
        table_idx += 1

print('\n=== 段46内容（显著性检验引用）===')
if 46 < len(doc.paragraphs):
    print(doc.paragraphs[46].text[:300])

print('\n=== 段111内容（物理先验因子表）===')
if 111 < len(doc.paragraphs):
    print(doc.paragraphs[111].text[:200])
