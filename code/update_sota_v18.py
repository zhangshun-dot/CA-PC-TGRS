# -*- coding: utf-8 -*-
r"""update_sota_v18.py — 更新SOTA对比表为v18
1. 在表SOTA-a中填上自动化先验和人工先验的逐类AP50值（替换"—"）
2. 删除表SOTA-a表注中的"逐类值见补充材料表S4"
3. 删除段落123中的"逐类结果见补充材料表S4"
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v17.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v18.docx'

doc = Document(PAPER_PATH)

# 自动化先验逐类AP50（列顺序：airplane, boat, car, truck, bus, motorcycle）
auto_prior = {
    'airplane': '0.7609',
    'boat': '0.2006',
    'car': '0.1932',
    'truck': '0.2959',
    'bus': '0.5845',
    'motorcycle': '0.4655',
    'avg': '0.4168',
    'gain': '+0.0461',
}

# 人工先验逐类AP50
human_prior = {
    'airplane': '0.7592',
    'boat': '0.1807',
    'car': '0.1867',
    'truck': '0.2892',
    'bus': '0.5744',
    'motorcycle': '0.5207',
    'avg': '0.4185',
    'gain': '+0.0478',
}

# 找到表SOTA-a（包含"自动化先验"的表格）
target_table = None
for i, table in enumerate(doc.tables):
    for row in table.rows:
        for cell in row.cells:
            if '自动化先验' in cell.text:
                target_table = table
                print(f'找到表SOTA-a，表格索引{i}')
                break
        if target_table:
            break
    if target_table:
        break

if target_table is None:
    print('未找到表SOTA-a')
    exit(1)

# 打印表头，确认列顺序
header = [cell.text.strip() for cell in target_table.rows[0].cells]
print(f'表头: {header}')

# 找到自动化先验和人工先验的行
auto_row_idx = None
human_row_idx = None
for i, row in enumerate(target_table.rows):
    first_cell = row.cells[0].text.strip()
    if '自动化先验' in first_cell:
        auto_row_idx = i
        print(f'自动化先验在行{i}: {first_cell[:50]}')
    if '人工先验' in first_cell:
        human_row_idx = i
        print(f'人工先验在行{i}: {first_cell[:50]}')

# 更新自动化先验行
if auto_row_idx is not None:
    row = target_table.rows[auto_row_idx]
    # 列顺序：方法(0), 目标域数据(1), 可训练参数(2), airplane(3), boat(4), car(5), truck(6), bus(7), motorcycle(8), 6类平均(9), 增益(10)
    col_map = {'airplane': 3, 'boat': 4, 'car': 5, 'truck': 6, 'bus': 7, 'motorcycle': 8}
    for cls, col_idx in col_map.items():
        cell = row.cells[col_idx]
        cell.text = auto_prior[cls]
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(9)
    print(f'自动化先验行已更新: airplane={auto_prior["airplane"]}, boat={auto_prior["boat"]}, ...')

# 更新人工先验行
if human_row_idx is not None:
    row = target_table.rows[human_row_idx]
    col_map = {'airplane': 3, 'boat': 4, 'car': 5, 'truck': 6, 'bus': 7, 'motorcycle': 8}
    for cls, col_idx in col_map.items():
        cell = row.cells[col_idx]
        cell.text = human_prior[cls]
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(9)
    print(f'人工先验行已更新: airplane={human_prior["airplane"]}, boat={human_prior["boat"]}, ...')

# 删除表SOTA-a表注中的"逐类值见补充材料表S4"
for i, p in enumerate(doc.paragraphs):
    if '表SOTA-a' in p.text and '补充材料表S4' in p.text:
        old_text = p.text
        new_text = old_text.replace('A-CSPC物理先验自适应的逐类值见补充材料表S4；', '')
        new_text = new_text.replace('A-CSPC物理先验自适应的逐类值见补充材料表S4', '')
        # 重新设置段落文本
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'表SOTA-a表注已更新（段落{i}）')
        print(f'  旧: {old_text[:100]}')
        print(f'  新: {new_text[:100]}')
        break

# 删除段落123中的"逐类结果见补充材料表S4"
for i, p in enumerate(doc.paragraphs):
    if 'α扫描' in p.text and '补充材料表S4' in p.text:
        old_text = p.text
        new_text = old_text.replace('（逐类结果见补充材料表S4）', '')
        new_text = new_text.replace('(逐类结果见补充材料表S4)', '')
        # 重新设置段落文本
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'段落{i}已更新（删除表S4引用）')
        print(f'  旧: {old_text[:120]}')
        print(f'  新: {new_text[:120]}')
        break

# 验证：搜索是否还有"表S4"的引用
remaining_s4 = []
for i, p in enumerate(doc.paragraphs):
    if '表S4' in p.text or 'S4' in p.text:
        remaining_s4.append((i, p.text[:100]))
if remaining_s4:
    print(f'\n警告：仍有{len(remaining_s4)}处包含"S4"的引用：')
    for idx, text in remaining_s4:
        print(f'  段落{idx}: {text}')
else:
    print('\n验证通过：已无"表S4"引用')

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('v18更新完成:')
print('  1. 表SOTA-a中自动化先验和人工先验的逐类AP50值已填上')
print('  2. 表SOTA-a表注中的"逐类值见补充材料表S4"已删除')
print('  3. 段落123中的"逐类结果见补充材料表S4"已删除')
