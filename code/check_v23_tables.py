# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx')
body = doc.element.body

print('=== v23 表格与前后段落 ===')
table_idx = 0
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        header = []
        first_row = child.find('.//' + qn('w:tr'))
        if first_row is not None:
            for cell in first_row.findall('.//' + qn('w:tc')):
                cell_text = ''.join([t.text or '' for t in cell.findall('.//' + qn('w:t'))])
                header.append(cell_text[:12])
        prev_text = ''
        next_text = ''
        if i > 0 and body[i-1].tag == qn('w:p'):
            prev_text = ''.join([t.text or '' for t in body[i-1].findall('.//' + qn('w:t'))])[:60]
        if i < len(body)-1 and body[i+1].tag == qn('w:p'):
            next_text = ''.join([t.text or '' for t in body[i+1].findall('.//' + qn('w:t'))])[:60]
        print(f'tbl[{table_idx}]: {header}')
        print(f'  前: {prev_text}')
        print(f'  后: {next_text}')
        table_idx += 1
