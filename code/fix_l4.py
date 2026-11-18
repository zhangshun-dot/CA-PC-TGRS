# -*- coding: utf-8 -*-
"""fix_l4.py — 修复L4："诊断—修复—边界"三支柱与实验章节无显式映射"""
from docx import Document
from docx.shared import Pt
from copy import deepcopy

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# §4实验在段落41，§4.1在段落42
# 在§4标题后插入一段指引
section4_title = doc.paragraphs[41]

# 创建新段落（在§4标题后、§4.1前）
new_para = section4_title._element.addnext(deepcopy(section4_title._element))
new_para_obj = doc.paragraphs[42]  # 插入后，原来的§4.1变成了段落43

# 设置新段落的样式和内容
new_para_obj.style = doc.styles['Body Text']
for run in new_para_obj.runs:
    run.text = ''

guide_text = (
    '实验章节按"诊断—修复—边界"三支柱组织：'
    '§4.3分层失效诊断对应P1诊断支柱，识别原型失效的真正维度；'
    '§4.4非对称构造消融对应P2修复支柱，验证正/负原型非对称构造的必要性；'
    '§4.5反向迁移验证、§4.7反向组件归因与§4.8检测器谱系对应P3边界支柱，系统界定方法的可修复边界与适用范围。'
)

run = new_para_obj.runs[0] if new_para_obj.runs else new_para_obj.add_run()
run.text = guide_text
run.font.name = '宋体'
run.font.size = Pt(11)
run.font.bold = False

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('L4已修改：§4开头加入三支柱映射指引')
print()
print('指引内容：')
print(guide_text)
