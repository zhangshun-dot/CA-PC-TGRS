# -*- coding: utf-8 -*-
from docx import Document
doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v21.docx')

print('=== 表16引用 ===')
for i, p in enumerate(doc.paragraphs):
    if '表16' in p.text:
        print(f'段落{i}: {p.text[:300]}')
        print()

print('=== 表17引用 ===')
for i, p in enumerate(doc.paragraphs):
    if '表17' in p.text:
        print(f'段落{i}: {p.text[:300]}')
        print()

print('=== 所有内联图片位置 ===')
# 检查每个段落是否包含图片
for i, p in enumerate(doc.paragraphs):
    has_image = False
    for run in p.runs:
        if run._element.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip'):
            has_image = True
        if run._element.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing'):
            has_image = True
    if has_image:
        print(f'段落{i}: 包含图片, 文本: {p.text[:100]}')

print('\n=== 段落75-90的内容（图4附近）===')
for i in range(75, 95):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text[:120]
        if text.strip():
            print(f'段落{i}: {text}')
