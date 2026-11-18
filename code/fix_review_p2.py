# -*- coding: utf-8 -*-
r"""fix_review_p2.py — 修复审稿意见P2级建议问题
"""
import sys
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v6.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v7.docx'

doc = Document(PAPER_PATH)
updates = []

def replace_in_paragraph(idx, old, new, reason):
    """在指定段落中替换文本"""
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
# P2-2: §5.4观察3表述修正（段落91）
# ═══════════════════════════════════════════════════════════════
print('[P2-2] 修正观察3表述...')
old_91 = '观察3（非对称合理性）：正原型外推、负原型不外推的非对称构造在源域背景统计与目标域背景统计差异较小时最优。直觉：背景类跨域差异小，外推反而引入噪声。'
new_91 = '观察3（非对称合理性）：正原型外推、负原型不外推的非对称构造在源域背景统计与目标域背景统计差异较大时最优。直觉：背景类跨域差异大时，正原型需要外推以适应目标域分布，而负原型外推会引入更多域偏移噪声，非对称设计（正原型外推、负原型保守）的收益更明显。'
replace_in_paragraph(91, old_91, new_91, 'P2-2：观察3差异较小→差异较大，逻辑方向修正')

# ═══════════════════════════════════════════════════════════════
# P2-4: 摘要headline数字明确（段落3）
# ═══════════════════════════════════════════════════════════════
print('[P2-4] 明确摘要headline数字...')
old_3 = '在DIOR→COCO六类上最佳配置取得+0.0461增益（固定参数+0.0166，相对提升177%）'
new_3 = '在DIOR→COCO六类上最佳配置取得+0.0461增益（自动化先验；人工先验对照+0.0478，两者差距仅0.0017；固定参数+0.0166，相对提升177%）'
replace_in_paragraph(3, old_3, new_3, 'P2-4：摘要明确headline数字为自动化先验+0.0461')

# ═══════════════════════════════════════════════════════════════
# P2-5: 图1标题与正文一致（段落26）
# ═══════════════════════════════════════════════════════════════
print('[P2-5] 统一图1标题...')
old_26 = '图 1　A-CSPC方法pipeline（诊断—修复—边界三支柱：P1分层失效诊断区分排序质量与候选框覆盖；P2非对称原型构造沿文本方向外推正原型、保留源域负原型；P3方向门g₂在外推与保守模式间选择）'
new_26 = '图 1　面向跨域目标检测的非对称原型校正A-CSPC整体流程（诊断—修复—边界三支柱：P1分层失效诊断区分排序质量与候选框覆盖；P2非对称原型构造沿文本方向外推正原型、保留源域负原型；P3方向门g₂在外推与保守模式间选择）'
replace_in_paragraph(26, old_26, new_26, 'P2-5：图1标题与正文标题一致')

# ═══════════════════════════════════════════════════════════════
# P2-1: 标点符号统一（半角负号→全角负号，仅处理数值中的负号）
# ═══════════════════════════════════════════════════════════════
print('[P2-1] 统一数值中的负号（半角→全角）...')
# 这个操作比较危险，只处理明确的数值负号
# 搜索包含"-0."的段落，替换为"−0."
count = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 只替换数值中的负号，如"-0.01"→"−0.01"
    # 不替换公式中的减号、日期中的连字符
    import re
    new_text = re.sub(r'(?<![a-zA-Z])-(\d+\.\d+)', r'−\1', text)
    if new_text != text:
        # 整段替换
        if p.runs:
            p.runs[0].text = new_text
            for run in p.runs[1:]:
                run.text = ''
        count += 1
        updates.append(f'[{i}] P2-1：数值负号统一（半角→全角）')

print(f'  共处理{count}个段落')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\nP2修复完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
