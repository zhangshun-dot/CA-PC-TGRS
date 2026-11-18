# -*- coding: utf-8 -*-
r"""fix_review_p0.py — 修复审稿意见P0级4个硬伤
"""
import sys
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v2.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v5.docx'

doc = Document(PAPER_PATH)
updates = []

def replace_in_paragraph(idx, old, new, reason):
    """在指定段落中替换文本"""
    p = doc.paragraphs[idx]
    if old in p.text:
        # 尝试在单个run中替换
        replaced = False
        for run in p.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                replaced = True
                break
        # 如果跨run，整段替换
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
        print(f'  警告: [{idx}] 未找到 "{old[:40]}..."')
        return False

# ═══════════════════════════════════════════════════════════════
# P0-1: §4.1 类别列表修正（段落40）
# ═══════════════════════════════════════════════════════════════
print('[P0-1] 修正§4.1类别列表...')
old_40 = '主实验选取6个共享类别：airplane、ship、vehicle、bus、motorcycle、bicycle，其中vehicle为粗粒度源域类别，映射到COCO的bus/motorcycle等细粒度类别。'
new_40 = '主实验选取源域DIOR侧3个映射源类（airplane、ship、vehicle），映射到COCO侧6个目标类（airplane、boat、car、truck、bus、motorcycle）；其中vehicle为粗粒度源域类别，映射到COCO的car/truck/bus/motorcycle四个细粒度类别，ship映射到boat。'
replace_in_paragraph(40, old_40, new_40, '§4.1类别列表：6个共享类别→3源类映射6目标类')

# ═══════════════════════════════════════════════════════════════
# P0-2: 候选框覆盖率口径统一（段落52）
# ═══════════════════════════════════════════════════════════════
print('[P0-2] 统一候选框覆盖率口径...')
old_52 = '候选框池最大可达召回R_max在6类上平均为0.82（范围0.71–0.91）'
new_52 = '候选框池最大可达召回R_max（按类别GT框被候选框IoU≥0.5覆盖的比例，N=100预过滤口径）在6类上平均为0.82（范围0.71–0.91）'
replace_in_paragraph(52, old_52, new_52, '§4.3覆盖率：明确R_max定义和口径')

# ═══════════════════════════════════════════════════════════════
# P0-3: §6.5.1 α扫描先验来源明确（段落122）
# ═══════════════════════════════════════════════════════════════
print('[P0-3] 明确§6.5.1 α扫描先验来源...')
old_122 = '结果表明：α=1.5为最佳配置，六类均值增益从固定参数的+0.0166提升至+0.0461'
new_122 = '本节α扫描使用自动化赋值的物理先验（人工先验对比结果见下节）。结果表明：α=1.5为最佳配置，六类均值增益从固定参数的+0.0166提升至+0.0461'
replace_in_paragraph(122, old_122, new_122, '§6.5.1：明确α扫描使用自动化先验')

# ═══════════════════════════════════════════════════════════════
# P0-4: 结论"自动化赋值"矛盾修正（段落134）
# ═══════════════════════════════════════════════════════════════
print('[P0-4] 修正结论自动化赋值矛盾...')
old_134 = '自动化赋值和多标注者一致性验证留待后续工作'
new_134 = '自动化先验的更大规模验证与多标注者一致性留待后续工作'
replace_in_paragraph(134, old_134, new_134, '§7局限性：自动化赋值→更大规模验证')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\nP0修复完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
