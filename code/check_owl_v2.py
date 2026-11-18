# -*- coding: utf-8 -*-
from docx import Document
import re

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v29.docx')

print('=== 搜索OWLv2相关内容 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'OWLv2' in text or '强检测器' in text or '弱校准' in text or '校正器边界' in text:
        print(f'\n段落{i} [{p.style.name}]:')
        print(f'  {text[:200]}')

print('\n\n=== 搜索检测器谱系相关内容 ===')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if '检测器谱系' in text or 'GDINO-tiny' in text or 'GDINO-base' in text or '检测器依赖' in text:
        print(f'\n段落{i} [{p.style.name}]:')
        print(f'  {text[:200]}')

print('\n\n=== 搜索摘要和引言 ===')
for i in range(min(30, len(doc.paragraphs))):
    text = doc.paragraphs[i].text
    if '无目标域' in text or 'zero-shot' in text.lower() or 'training-free' in text.lower() or '目标域GT' in text:
        print(f'\n段落{i} [{doc.paragraphs[i].style.name}]:')
        print(f'  {text[:200]}')
