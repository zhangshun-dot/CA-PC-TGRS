# -*- coding: utf-8 -*-
"""fix_c2.py — 修复C2：摘要CI与ρ同行"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 摘要在段落3
p = doc.paragraphs[3]
old_text = p.text

# 替换表述：把CI含0写在ρ同一句
old_phrase = '不仅验证事前逐类增益预测（37类/6域对ρ=+0.699）'
new_phrase = '不仅验证事前逐类增益预测（37类/6域对Spearman ρ=+0.699，但按域对聚类的bootstrap双侧95% CI包含0，属探索性证据）'

new_text = old_text.replace(old_phrase, new_phrase)

# 清空原有runs，设置新内容
for run in p.runs:
    run.text = ''
p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('C2已修改：摘要中ρ=+0.699与CI含0写在同一句')
print()
print('修改后摘要相关部分：')
idx = new_text.find('不仅验证事前逐类增益预测')
print(new_text[idx:idx+150])
