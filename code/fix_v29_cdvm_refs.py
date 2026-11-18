# -*- coding: utf-8 -*-
r"""fix_v29_cdvm_refs.py — 替换CDVM为CoOp，修复引用编号
1. 段落13: CDVM → CoOp[10]
2. 表格14: CDVM few-shot → CoOp few-shot
3. 段落23: 修复引用编号（CoOp[13]→[10], CoCoOp[14]→[21], CLIP-Adapter[15]→[31], Tip-Adapter[16]→[16], TTA[17-20]→[17-20]）
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v28.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v29.docx'

doc = Document(PAPER_PATH)

print('第一步：替换段落13中的CDVM为CoOp')
p13 = doc.paragraphs[13]
old_text = p13.text
print(f'  原文: {old_text[:150]}')

# 替换CDVM为CoOp[10]
new_text = old_text.replace(
    'CDVM（源域视觉原型+负原型抑制）仅在源域空间构造原型，未进行跨空间外推',
    'CoOp[10]（学习可优化文本提示向量）仅在文本空间进行原型学习，未进行视觉原型的跨空间外推'
)

# 更新段落文本
for run in p13.runs:
    run.text = ''
if p13.runs:
    p13.runs[0].text = new_text
else:
    p13.add_run(new_text)

print(f'  新文: {new_text[:150]}')

print('\n第二步：替换表格14中的CDVM few-shot为CoOp few-shot')
table14 = doc.tables[13]  # 表格14（索引13）
print(f'  表格14: {len(table14.rows)}行 x {len(table14.columns)}列')

for row_idx, row in enumerate(table14.rows):
    for cell_idx, cell in enumerate(row.cells):
        if 'CDVM' in cell.text:
            old_cell_text = cell.text
            new_cell_text = old_cell_text.replace('CDVM', 'CoOp')
            print(f'  行{row_idx}, 列{cell_idx}: "{old_cell_text[:50]}" → "{new_cell_text[:50]}"')
            
            # 更新单元格文本
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.text = ''
                if paragraph.runs:
                    paragraph.runs[0].text = new_cell_text
                    break
            else:
                if cell.paragraphs:
                    cell.paragraphs[0].add_run(new_cell_text)

print('\n第三步：修复段落23的引用编号')
p23 = doc.paragraphs[23]
old_text23 = p23.text
print(f'  原文: {old_text23[:200]}')

# 修复引用编号
# CoOp[13] → CoOp[10]
# CoCoOp[14] → CoCoOp[21]
# CLIP-Adapter[15] → CLIP-Adapter[31]
# Tip-Adapter[16] → Tip-Adapter[16] (这个是对的)
# TTA方法[17-20] → TTA方法[17-20] (这个是对的)

new_text23 = old_text23
new_text23 = new_text23.replace('CoOp[13]', 'CoOp[10]')
new_text23 = new_text23.replace('CoCoOp[14]', 'CoCoOp[21]')
new_text23 = new_text23.replace('CLIP-Adapter[15]', 'CLIP-Adapter[31]')
# Tip-Adapter[16]是对的，不需要改
# TTA[17-20]是对的，不需要改

# 更新段落文本
for run in p23.runs:
    run.text = ''
if p23.runs:
    p23.runs[0].text = new_text23
else:
    p23.add_run(new_text23)

print(f'  新文: {new_text23[:200]}')

print('\n第四步：检查其他段落是否有类似的引用编号问题')
# 搜索所有包含[数字]的段落，检查引用编号是否合理
problematic_refs = []
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 跳过参考文献部分
    if i > 150:
        break
    
    # 查找所有引用
    refs = re.findall(r'\[(\d{1,2}(?:[,\-]\d{1,2})*)\]', text)
    for ref in refs:
        try:
            if '-' in ref:
                start, end = map(int, ref.split('-'))
                if start > 51 or end > 51:
                    problematic_refs.append((i, ref, text[:80]))
            elif ',' in ref:
                for r in ref.split(','):
                    if int(r.strip()) > 51:
                        problematic_refs.append((i, ref, text[:80]))
            else:
                if int(ref) > 51:
                    problematic_refs.append((i, ref, text[:80]))
        except:
            pass

if problematic_refs:
    print(f'  发现 {len(problematic_refs)} 个可能有问题的引用:')
    for para_idx, ref, context in problematic_refs[:10]:
        print(f'    段落{para_idx}: [{ref}] - {context}')
else:
    print('  未发现超出范围的引用编号')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('CDVM替换和引用编号修复完成')
