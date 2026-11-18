# -*- coding: utf-8 -*-
r"""fix_v32_add_coco_metrics.py — 把标准COCO指标(AP@[.5:.95])补充到论文
"""
from docx import Document
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v31.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v32.docx'

doc = Document(PAPER_PATH)

print('=' * 60)
print('补充标准COCO指标(AP@[.5:.95])到论文')
print('=' * 60)

# 段落46：主结果 - 补充AP@[.5:.95]指标
p46 = doc.paragraphs[46]
old_p46 = p46.text
print(f'\n原文段落46:\n{old_p46[:400]}')

# 在主结果中补充AP@[.5:.95]指标
new_p46 = old_p46.replace(
    '目标域调参配置A-CSPC(2.0,0.5)取得+0.0646的AP50增益（6/6类正增益）',
    '目标域调参配置A-CSPC(2.0,0.5)取得+0.0646的AP50增益（6/6类正增益），在标准COCO指标AP@[0.5:0.95]上取得+0.0463增益（6/6类正增益，逐类为airplane +0.0129、boat +0.0010、car +0.0460、truck +0.0732、bus +0.0670、motorcycle +0.0779）'
)

for run in p46.runs:
    run.text = ''
if p46.runs:
    p46.runs[0].text = new_p46
else:
    p46.add_run(new_p46)
print(f'\n新文段落46:\n{new_p46[:500]}')

print('\n' + '=' * 60)
print('检查摘要中是否需要补充AP@[.5:.95]')
print('=' * 60)

# 段落2：摘要 - 可以简要提及AP@[.5:.95]
p2 = doc.paragraphs[2]
old_p2 = p2.text
print(f'\n原文段落2:\n{old_p2[:300]}')

# 在摘要中简要提及标准COCO指标
new_p2 = old_p2.replace(
    '取得+0.0646的AP50增益（6/6类正增益）',
    '取得+0.0646的AP50增益和+0.0463的AP@[0.5:0.95]增益（均为6/6类正增益）'
)

for run in p2.runs:
    run.text = ''
if p2.runs:
    p2.runs[0].text = new_p2
else:
    p2.add_run(new_p2)
print(f'\n新文段落2:\n{new_p2[:400]}')

doc.save(OUTPUT_PATH)
print(f'\n\n论文已保存: {OUTPUT_PATH}')
print('标准COCO指标补充完成')
