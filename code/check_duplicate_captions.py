# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx')
body = doc.element.body

print('=== 检查重复题注 ===')
# 检查表格前面的题注段落
table_idx = 0
for i, child in enumerate(body):
    if child.tag == qn('w:tbl'):
        # 检查前面的段落
        captions = []
        j = i - 1
        while j >= 0 and body[j].tag == qn('w:p'):
            text = ''.join([t.text or '' for t in body[j].findall('.//' + qn('w:t'))])
            if text.strip().startswith('表'):
                captions.append((j, text[:60]))
            else:
                break
            j -= 1
        if len(captions) > 1:
            print(f'tbl[{table_idx}]: 发现{len(captions)}个题注')
            for idx, (para_idx, text) in enumerate(captions):
                print(f'  题注{idx}: 段落{para_idx} - {text}')
        table_idx += 1

print('\n=== 段落60-70内容 ===')
for i in range(60, 72):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text[:80]
        if text.strip():
            print(f'段落{i}: {text}')
