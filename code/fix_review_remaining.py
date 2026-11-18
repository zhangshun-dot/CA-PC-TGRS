# -*- coding: utf-8 -*-
r"""fix_review_remaining.py — 修复审稿残留的3处问题
"""
import sys
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v7.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v8.docx'

doc = Document(PAPER_PATH)
updates = []

def replace_in_paragraph(idx, old, new, reason):
    p = doc.paragraphs[idx]
    if old in p.text:
        replaced = False
        for run in p.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                replaced = True
                break
        if not replaced:
            full_text = p.text
            new_full = full_text.replace(old, new)
            if p.runs:
                p.runs[0].text = new_full
                for run in p.runs[1:]:
                    run.text = ''
        updates.append(f'[{idx}] {reason}')
        return True
    else:
        print(f'  警告: [{idx}] 未找到 "{old[:50]}..."')
        return False

# ═══════════════════════════════════════════════════════════════
# 1. §8结论自动化赋值→更大规模验证（段落136）
# ═══════════════════════════════════════════════════════════════
print('[1] 修复§8结论自动化赋值矛盾...')
old_136 = '未来工作包括：物理先验的自动化赋值'
new_136 = '未来工作包括：自动化先验的更大规模验证'
replace_in_paragraph(136, old_136, new_136, '§8结论：自动化赋值→更大规模验证')

# ═══════════════════════════════════════════════════════════════
# 2. g₂术语统一：19类→19个域对（段落203，补充材料）
# ═══════════════════════════════════════════════════════════════
print('[2] 统一g₂术语...')
old_203 = '混合方向的 19 类验证'
new_203 = '混合方向的 19 个域对验证'
replace_in_paragraph(203, old_203, new_203, '补充材料：19类→19个域对')

# ═══════════════════════════════════════════════════════════════
# 3. 覆盖率口径：在检测器谱系段落（74）加口径说明
# ═══════════════════════════════════════════════════════════════
print('[3] 补充检测器谱系覆盖率口径说明...')
# 段落74是检测器谱系实验，在开头加口径说明
old_74 = '检测器谱系实验（N=100/类子集协议，与主表一致）验证A-CSPC的校正器边界'
new_74 = '检测器谱系实验（N=100/类子集协议，与主表一致；候选框GT覆盖率按各检测器原始输出统计，不同检测器候选框数量与置信度阈值不同，覆盖率范围存在差异）验证A-CSPC的校正器边界'
replace_in_paragraph(74, old_74, new_74, '检测器谱系：补充覆盖率口径说明')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\n残留修复完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
