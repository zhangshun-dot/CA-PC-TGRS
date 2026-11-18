# -*- coding: utf-8 -*-
"""fix_c1.py — 修复C1：§7与§4.5根因表述不一致"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# §7局限性在段落153
p = doc.paragraphs[153]
old_text = p.text

# 替换第三条的表述
old_phrase = '反向方向（photo→overhead）因文本外推方向失配而收益有限，双向自适应外推是值得探索的方向'
new_phrase = '反向方向（细→粗）因细类原型已高度聚集于粗类质心（cos>0.95）、外推无收益，双向自适应校正（如弹性锚点框架）是值得探索的方向'

new_text = old_text.replace(old_phrase, new_phrase)

# 清空原有runs，设置新内容
for run in p.runs:
    run.text = ''
p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('C1已修改：§7第三条从"文本外推方向失配"改为"细类原型已高度聚集于粗类质心（cos>0.95）"')
print()
print('修改后§7第三条：')
idx = new_text.find('(3)')
print(new_text[idx:])
