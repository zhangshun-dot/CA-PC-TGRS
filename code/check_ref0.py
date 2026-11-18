# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx')

print('=== 检查[0]引用 ===')
for i, p in enumerate(doc.paragraphs):
    if '[0]' in p.text:
        print(f'段落{i}: {p.text[:200]}')
        print()

print('=== 检查段落130附近 ===')
for i in range(128, 135):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text[:150]
        if text.strip():
            print(f'段落{i}: {text}')

print('\n=== 检查参考文献[50][51] ===')
ref_start = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text and p.style.name.startswith('Heading'):
        ref_start = i
        break

if ref_start:
    for i in range(ref_start+1, len(doc.paragraphs)):
        text = doc.paragraphs[i].text.strip()
        if text.startswith('[50]') or text.startswith('[51]'):
            print(f'段落{i}: {text[:150]}')
