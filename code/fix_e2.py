# -*- coding: utf-8 -*-
"""fix_e2.py — 修复E2：摘要句子嵌套过深"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 摘要在段落2
p = doc.paragraphs[2]
old_text = p.text

# 把长句拆成两句
old_phrase = '在NWPU/DOTA/VisDrone三个替换源域上复现；目标域无关配置(1.0,0.25)增益为+0.0166且boat转负。'
new_phrase = '在NWPU/DOTA/VisDrone三个替换源域上复现。目标域无关配置(1.0,0.25)增益为+0.0166且boat转负。'

new_text = old_text.replace(old_phrase, new_phrase)

# 清空原有runs，设置新内容
for run in p.runs:
    run.text = ''
p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('E2已修改：摘要长句拆成两句')
print()
print('修改后摘要末尾：')
idx = new_text.find('在DIOR→COCO六类测试中')
print(new_text[idx:])
