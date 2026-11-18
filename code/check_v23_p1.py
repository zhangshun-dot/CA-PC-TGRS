# -*- coding: utf-8 -*-
from docx import Document
from docx.oxml.ns import qn

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx')

print('=== 段58-70内容（表4位置检查）===')
for i in range(58, 72):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text[:80]
        if text.strip():
            print(f'段落{i}: {text}')

print('\n=== 段68内容（交叉引用检查）===')
if 68 < len(doc.paragraphs):
    print(doc.paragraphs[68].text[:300])

print('\n=== 段205-225内容（补充材料检查）===')
for i in range(205, 226):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text[:100]
        if text.strip():
            print(f'段落{i}: {text}')
