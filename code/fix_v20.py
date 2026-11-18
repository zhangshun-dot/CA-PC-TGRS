# -*- coding: utf-8 -*-
r"""fix_v20.py — 修复v19的三处问题，生成v20
1. 段落48：「显著为负」→严谨表述（CI跨0，不能说显著）
2. DANN行：0.316±0.109 → 0.3156±0.1090，逐类值也更新为4位小数
3. 表注：添加T4身份说明
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v19.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v20.docx'

doc = Document(PAPER_PATH)

# ══════════════════════════ 修复1：段落48「显著为负」 ══════════════════════════
print('修复1：段落48「显著为负」→严谨表述')
p48 = doc.paragraphs[48]
old_text = p48.text
new_text = old_text.replace(
    '多种子均值显著为负，说明域对抗训练在冻结CLIP特征+候选框级适配的设置下整体损害性能，而非训练数据不足导致的稻草人baseline。',
    '多种子均值为负（−0.0551），但标准差0.1090为均值绝对值的2.0倍，95%置信区间[−0.1904, +0.0802]跨越0，不能判定为显著损害性能；稳健结论是非但无稳定收益，且5个种子下均未超过CORAL(+0.0210)与A-CSPC自适应(+0.0461)，故并非稻草人baseline。A-CSPC零训练参数、输出确定性，DANN需训练约1.3M参数且种子间标准差达0.109，确定性本身在算力受限场景即实用优势。'
)

# 重新设置段落文本
for run in p48.runs:
    run.text = ''
if p48.runs:
    p48.runs[0].text = new_text
else:
    p48.add_run(new_text)
print(f'  旧文本包含「显著为负」: {"显著为负" in old_text}')
print(f'  新文本包含「显著为负」: {"显著为负" in new_text}')
print(f'  新文本包含「95%置信区间」: {"95%置信区间" in new_text}')

# ══════════════════════════ 修复2：DANN行数值 ══════════════════════════
print('\n修复2：DANN行数值更新为4位小数')
dann_new_values = {
    'airplane': '0.5647±0.2028',
    'boat': '0.1616±0.0304',
    'car': '0.1220±0.0051',
    'truck': '0.2207±0.0175',
    'bus': '0.4681±0.1341',
    'motorcycle': '0.3565±0.1047',
    'avg': '0.3156±0.1090',
    'gain': '−0.0551',
}

for table_idx, table in enumerate(doc.tables):
    for row_idx, row in enumerate(table.rows):
        first_cell = row.cells[0].text.strip()
        if 'DANN' in first_cell and '200张图' in first_cell:
            print(f'  找到DANN行：表格{table_idx}, 行{row_idx}')
            # 列顺序：方法(0), 目标域数据(1), 可训练参数(2), airplane(3), boat(4), car(5), truck(6), bus(7), motorcycle(8), 6类平均(9), 增益(10)
            col_map = {'airplane': 3, 'boat': 4, 'car': 5, 'truck': 6, 'bus': 7, 'motorcycle': 8}
            for cls, col_idx in col_map.items():
                cell = row.cells[col_idx]
                old_val = cell.text.strip()
                cell.text = dann_new_values[cls]
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
                print(f'    {cls}: {old_val} → {dann_new_values[cls]}')
            # 6类平均
            cell = row.cells[9]
            old_val = cell.text.strip()
            cell.text = dann_new_values['avg']
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
            print(f'    6类平均: {old_val} → {dann_new_values["avg"]}')
            # 增益
            cell = row.cells[10]
            old_val = cell.text.strip()
            cell.text = dann_new_values['gain']
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(9)
            print(f'    增益: {old_val} → {dann_new_values["gain"]}')
            break

# ══════════════════════════ 修复3：表注T4身份说明 ══════════════════════════
print('\n修复3：表注添加T4身份说明')
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('表SOTA-a') and '目标域无标注' in p.text:
        old_text = p.text
        # 在表注末尾添加T4身份说明
        if 'T4' not in old_text or '退化配置' not in old_text:
            new_text = old_text.rstrip('）') + '；T4纯文本原型为本文方法族退化配置（β=0,λ=0），供组件贡献拆解，非外部对比方法）'
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'  段落{i}表注已更新')
            print(f'  旧: {old_text[:100]}...')
            print(f'  新: {new_text[:100]}...')
        else:
            print(f'  段落{i}表注已包含T4身份说明，跳过')
        break

# ══════════════════════════ 验证 ══════════════════════════
print('\n=== 验证 ===')
# 验证1：显著为负
sig_count = 0
for i, p in enumerate(doc.paragraphs):
    if '显著为负' in p.text or '整体损害性能' in p.text:
        sig_count += 1
        print(f'  警告：段落{i}仍包含「显著为负」或「整体损害性能」')
print(f'  「显著为负」/「整体损害性能」出现次数: {sig_count}')

# 验证2：DANN行
for table in doc.tables:
    for row in table.rows:
        if 'DANN' in row.cells[0].text and '200张图' in row.cells[0].text:
            avg_val = row.cells[9].text.strip()
            gain_val = row.cells[10].text.strip()
            print(f'  DANN行6类平均: {avg_val}, 增益: {gain_val}')
            print(f'  验证通过: {"0.3156" in avg_val and "0.0551" in gain_val}')

# 验证3：T4身份说明
t4_found = False
for i, p in enumerate(doc.paragraphs):
    if '表SOTA-a' in p.text and '退化配置' in p.text:
        t4_found = True
        print(f'  T4身份说明已在段落{i}')
print(f'  T4身份说明存在: {t4_found}')

# 保存
doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('v20修复完成:')
print('  1. 段落48「显著为负」→严谨表述（CI跨0，不能说显著）')
print('  2. DANN行数值更新为4位小数（0.3156±0.1090 / −0.0551）')
print('  3. 表注添加T4身份说明（退化配置，非外部对比方法）')
