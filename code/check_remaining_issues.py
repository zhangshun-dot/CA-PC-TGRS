# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx')

print('=== 1. 补充材料中的表题 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if text.startswith('表') and ('补充' in text or '完整消融' in text or '信息使用审计' in text or '实验协议' in text):
        print(f'  段落{i}: {text[:80]}')

print('\n=== 2. 正文中所有"表X"引用 ===')
table_refs = {}
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 匹配"表X"模式（X为数字或SOTA-a等）
    matches = re.findall(r'表\s*(\d+|SOTA-[ab]|VOC20)', text)
    for m in matches:
        if m not in table_refs:
            table_refs[m] = []
        table_refs[m].append(i)

for ref in sorted(table_refs.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0)):
    print(f'  表{ref}: 出现在段落 {table_refs[ref][:10]}')

print('\n=== 3. 表6题注内容 ===')
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('表6'):
        print(f'  段落{i}: {p.text[:80]}')
