# -*- coding: utf-8 -*-
"""fix_l1.py — 修复L1：摘要与引言贡献表述大量重复"""
from docx import Document

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# 摘要在段落3
p = doc.paragraphs[3]
old_text = p.text

# 精简版摘要：只保留最核心的结果数字，删掉细节
new_text = (
    '本文定位为跨域原型失效诊断与可修复边界的分析性研究：'
    '分层失效诊断发现外推带来的排序改善量ΔAUC（而非源原型失效程度）具有预测力（ρ=+0.577）；'
    '提出类别级物理先验框架，验证事前逐类增益预测（37类/6域对Spearman ρ=+0.699，但按域对聚类的bootstrap双侧95% CI包含0，属探索性证据），'
    '并实现物理先验驱动的无目标域标注自适应参数控制，在DIOR→COCO六类上取得+0.0461增益（固定参数+0.0166）；'
    'A-CSPC非对称构造在DIOR→COCO六类目标域调参配置下取得+0.0646增益，'
    '并通过三源域跨源复现、反向组件归因和检测器谱系实验系统界定方法的可修复边界（弱校准检测器适用，反向细→粗方向因细类原型已高度聚集于粗类质心而无需校正）。'
)

# 清空原有runs，设置新内容
for run in p.runs:
    run.text = ''
p.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('L1已修改：摘要精简，删除与引言重复的细节')
print()
print(f'原摘要长度：{len(old_text)}字符')
print(f'新摘要长度：{len(new_text)}字符')
print(f'精简了：{len(old_text) - len(new_text)}字符')
print()
print('新摘要：')
print(new_text)
