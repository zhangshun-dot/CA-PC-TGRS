# -*- coding: utf-8 -*-
r"""fix_cdvm.py — 修复CDVM残次品问题
CDVM没有参考文献，且CDVM[1]引用错误（[1]是CLIP）。
CDVM本质是A-CSPC去除外推的消融版本（β=0），不是独立外部基线。
处理：改称为"源域视觉原型基线（β=0消融配置）"，删除错误引用。
"""
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v8.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v9.docx'

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
        print(f'  警告: [{idx}] 未找到 "{old[:60]}..."')
        return False

# ═══════════════════════════════════════════════════════════════
# 1. 段落8：旧CDVM零样本基线 → 源域视觉原型基线
# ═══════════════════════════════════════════════════════════════
print('[1] 修复段落8...')
old_8 = '旧CDVM零样本基线在六类平均下降0.0710'
new_8 = '源域视觉原型基线（去除外推的消融配置）在六类平均下降0.0710'
replace_in_paragraph(8, old_8, new_8, '段落8：旧CDVM零样本基线→源域视觉原型基线')

# ═══════════════════════════════════════════════════════════════
# 2. 段落19：删除CDVM的"需补充说明"整段
# ═══════════════════════════════════════════════════════════════
print('[2] 修复段落19...')
p19_text = doc.paragraphs[19].text
if '需补充说明' in p19_text and 'CDVM' in p19_text:
    # 整段替换为更简洁的表述
    old_19 = p19_text
    new_19 = '原型方法通过聚合源域标注框特征构建类别原型，在目标域候选框上进行重排序。本文提出的A-CSPC在源域原型基础上引入非对称跨空间校正：正原型沿CLIP文本方向外推至目标域，负原型保留在源域空间以抑制类间混淆。'
    # 整段替换
    if doc.paragraphs[19].runs:
        doc.paragraphs[19].runs[0].text = new_19
        for run in doc.paragraphs[19].runs[1:]:
            run.text = ''
    updates.append('[19] 段落19：删除CDVM需补充说明，改写为原型方法介绍')
    print('  已替换段落19')
else:
    print('  段落19不包含需补充说明+CDVM，跳过')

# ═══════════════════════════════════════════════════════════════
# 3. 段落43：CDVM[1]−0.071 → 源域视觉原型基线−0.071（删除错误引用[1]）
# ═══════════════════════════════════════════════════════════════
print('[3] 修复段落43...')
old_43 = 'CDVM[1]−0.071'
new_43 = '源域视觉原型基线（β=0消融）−0.071'
replace_in_paragraph(43, old_43, new_43, '段落43：CDVM[1]→源域视觉原型基线（β=0消融），删除错误引用')

# ═══════════════════════════════════════════════════════════════
# 4. 段落56：CDVM（N=100口径）−0.0451 → 源域视觉原型基线（N=100口径）−0.0451
# ═══════════════════════════════════════════════════════════════
print('[4] 修复段落56...')
old_56 = 'CDVM −0.0451'
new_56 = '源域视觉原型基线（β=0）−0.0451'
replace_in_paragraph(56, old_56, new_56, '段落56：CDVM→源域视觉原型基线（β=0）')

# ═══════════════════════════════════════════════════════════════
# 5. 检查是否还有CDVM残留
# ═══════════════════════════════════════════════════════════════
print('[5] 检查CDVM残留...')
remaining = []
for i, p in enumerate(doc.paragraphs):
    if 'CDVM' in p.text:
        remaining.append((i, p.text[:100]))
if remaining:
    print(f'  仍有{len(remaining)}处CDVM残留:')
    for idx, text in remaining:
        print(f'    [{idx}] {text}')
else:
    print('  无CDVM残留')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\nCDVM修复完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
