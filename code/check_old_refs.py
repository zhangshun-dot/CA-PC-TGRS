# -*- coding: utf-8 -*-
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx')

print('=== 段落58内容（表1引用）===')
print(doc.paragraphs[58].text[:200])

print('\n=== 段落61内容（表2引用）===')
print(doc.paragraphs[61].text[:200])

print('\n=== 段落223内容（表7引用）===')
if 223 < len(doc.paragraphs):
    print(doc.paragraphs[223].text[:200])

print('\n=== 段落205-210内容（补充材料开头）===')
for i in range(205, 212):
    if i < len(doc.paragraphs):
        text = doc.paragraphs[i].text[:100]
        if text.strip():
            print(f'段落{i}: {text}')
