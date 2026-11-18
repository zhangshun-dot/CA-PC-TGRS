# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v25.docx')

print('=== 检查参考文献部分结构 ===')
ref_start = None
for i, p in enumerate(doc.paragraphs):
    if '参考文献' in p.text and p.style.name.startswith('Heading'):
        ref_start = i
        print(f'参考文献标题在段落{i}: {p.text}')
        break

if ref_start:
    print(f'\n参考文献部分（段落{ref_start+1}到{ref_start+60}）:')
    for i in range(ref_start+1, min(ref_start+60, len(doc.paragraphs))):
        p = doc.paragraphs[i]
        text = p.text.strip()
        if text:
            style = p.style.name
            # 检查是否有编号
            match = re.match(r'\[(\d+)\]', text)
            ref_num = match.group(1) if match else '续行'
            print(f'  段落{i} [{ref_num}] (样式={style}): {text[:80]}')
