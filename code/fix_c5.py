# -*- coding: utf-8 -*-
"""fix_c5.py — 修复C5：§6.3物理先验ρ四个口径速查表"""
from docx import Document
from docx.shared import Pt
from docx.enum.table import WD_TABLE_ALIGNMENT
from copy import deepcopy

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# §6.3标题在段落127，段落128是空段落
# 在段落128的位置插入口径速查表

# 1. 先在空段落128里写表格标题
p128 = doc.paragraphs[128]
for run in p128.runs:
    run.text = ''
run = p128.runs[0] if p128.runs else p128.add_run()
run.text = '表12　物理先验预测力ρ的四个口径速查'
run.font.name = '宋体'
run.font.size = Pt(10)
run.font.bold = True

# 2. 在文档末尾创建表格
table = doc.add_table(rows=5, cols=3)
table.style = 'Table'
table.alignment = WD_TABLE_ALIGNMENT.CENTER

# 表头
hdr_cells = table.rows[0].cells
hdr_cells[0].text = '口径编号'
hdr_cells[1].text = '口径名称'
hdr_cells[2].text = 'ρ值与含义'

# 数据行
data = [
    ('①', '全数据', '+0.699（37类/6域对全部数据，探索性证据，CI含0）'),
    ('②', '严格冻结留出', '+0.555（2:2 split，训练集定因子权重，测试集只执行一次）'),
    ('③', 'LOSO阈值验证平均', '+0.686（4个留出源域均正相关，VisDrone子集n=4较弱）'),
    ('④', '6类主实验基准', '+0.354（主实验6类子集，用于先验扰动敏感性分析）'),
]

for i, (num, name, meaning) in enumerate(data):
    row_cells = table.rows[i + 1].cells
    row_cells[0].text = num
    row_cells[1].text = name
    row_cells[2].text = meaning

# 设置表格字体
for row in table.rows:
    for cell in row.cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.name = '宋体'
                run.font.size = Pt(10)

# 3. 把表格从文档末尾移动到标题后面
table_element = table._element
table_element.getparent().remove(table_element)
p128._element.addnext(table_element)

# 4. 修改段落130，删掉文字版的口径速查（因为已经有表格了）
p130 = doc.paragraphs[130]
old_text = p130.text
old_phrase = '（口径速查：全数据ρ=+0.699 / 严格冻结留出ρ=+0.555 / LOSO阈值验证平均ρ=+0.686 / 6类主实验基准ρ=+0.354）'
new_text = old_text.replace(old_phrase, '（四个口径的定义与含义见表12）')

for run in p130.runs:
    run.text = ''
p130.runs[0].text = new_text

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('C5已修改：§6.3物理先验ρ四个口径改成表格（表12）')
print()
print('表格内容：')
print('  ① 全数据：+0.699')
print('  ② 严格冻结留出：+0.555')
print('  ③ LOSO阈值验证平均：+0.686')
print('  ④ 6类主实验基准：+0.354')
