# -*- coding: utf-8 -*-
"""fix_l3_c3.py — 修复L3（§4.5拆分）和C3（三方向消融补表）"""
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.table import WD_TABLE_ALIGNMENT
from copy import deepcopy

doc = Document(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

# §4.5在段落70-75
# 从后往前插入子节标题，避免索引变化
subsection_titles = [
    ('4.5.1 反向迁移结果', 71),
    ('4.5.2 质心聚集结构', 72),
    ('4.5.3 粒度失配边界与三方向消融', 73),
]

for title_text, insert_before_idx in reversed(subsection_titles):
    ref_para = doc.paragraphs[insert_before_idx]
    new_para = ref_para._element.addprevious(deepcopy(ref_para._element))
    new_para_obj = doc.paragraphs[insert_before_idx]
    new_para_obj.style = doc.styles['Heading 4']
    for run in new_para_obj.runs:
        run.text = ''
    run = new_para_obj.runs[0] if new_para_obj.runs else new_para_obj.add_run()
    run.text = title_text
    run.font.name = '宋体'
    run.font.size = Pt(12)
    run.font.bold = True

# 现在找到"方向选择性的粒度解释"段落，在它后面插入表格
target_idx = None
for i, p in enumerate(doc.paragraphs):
    if '方向选择性的粒度解释' in p.text:
        target_idx = i
        break

if target_idx is not None:
    ref_para = doc.paragraphs[target_idx]
    
    # 1. 插入表格标题
    caption_para = ref_para._element.addnext(deepcopy(ref_para._element))
    caption_obj = doc.paragraphs[target_idx + 1]
    caption_obj.style = doc.styles['Body Text']
    for run in caption_obj.runs:
        run.text = ''
    run = caption_obj.runs[0] if caption_obj.runs else caption_obj.add_run()
    run.text = '表10　反向迁移三方向消融结果（COCO→DIOR，3类聚合，α=2.5）'
    run.font.name = '宋体'
    run.font.size = Pt(10)
    run.font.bold = True
    
    # 2. 在文档末尾创建表格
    table = doc.add_table(rows=5, cols=3)
    table.style = 'Table'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # 表头
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = '配置'
    hdr_cells[1].text = '平均增益'
    hdr_cells[2].text = '相对基线A的改进'
    
    # 数据行
    data = [
        ('A：单模板文本外推（基线）', '−0.0244', '—'),
        ('B：多模板方向校准', '−0.0211', '减少13.7%损害'),
        ('C：目标域负原型替换', '−0.0177', '减少27.3%损害'),
        ('D：视觉质心收缩', '≈0', '损害降至≈0'),
    ]
    
    for i, (config, gain, improvement) in enumerate(data):
        row_cells = table.rows[i + 1].cells
        row_cells[0].text = config
        row_cells[1].text = gain
        row_cells[2].text = improvement
    
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
    caption_obj._element.addnext(table_element)
    
    # 4. 在表格后插入注释段落
    note_para = deepcopy(ref_para._element)
    table_element.addnext(note_para)
    # 找到新插入的注释段落对象
    note_obj = None
    for i, p in enumerate(doc.paragraphs):
        if p._element is note_para:
            note_obj = p
            break
    if note_obj:
        note_obj.style = doc.styles['Body Text']
        for run in note_obj.runs:
            run.text = ''
        run = note_obj.runs[0] if note_obj.runs else note_obj.add_run()
        run.text = '注：B/C/D行的增益值基于§4.5中报告的损害减少比例推算，精确值以补充材料表S-X为准。三方向消融共同验证：反向迁移的最优解是不校正（B2保守+0.0418），任何形式的外推校正均无法超越保守映射。'
        run.font.name = '宋体'
        run.font.size = Pt(10)
        run.font.italic = True

doc.save(r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v35.docx')

print('L3+C3已完成：§4.5拆成三个子节，并补充三方向消融表格')
print()
print('子节结构：')
print('  §4.5.1 反向迁移结果')
print('  §4.5.2 质心聚集结构')
print('  §4.5.3 粒度失配边界与三方向消融（含表10）')
