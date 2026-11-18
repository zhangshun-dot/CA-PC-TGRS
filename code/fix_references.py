# -*- coding: utf-8 -*-
r"""fix_references.py — 系统性修正参考文献引用编号错位
修正内容：
  §2.1: [9-11]→[20-22], [12-14]→[23], [1-3,15]→[29]
  §2.2: GLIP[17]→[24], OWLv2[19]→[3], 原型方法[1-3,15]→[29]
  §2.3: CoCoOp[21]→[50](新增), CLIP-Adapter[22]→[11], Tip-Adapter[23]→[51](新增), TTA[24-25]→[16-19]
  §2.4: 重排序[27-29]→[30,32]
  §4.1: OWLv2[19]→[3]
补充文献：[50]CoCoOp, [51]Tip-Adapter
"""
from docx import Document

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v9.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v10.docx'

doc = Document(PAPER_PATH)
updates = []

def replace_in_para(idx, old, new, reason):
    p = doc.paragraphs[idx]
    if old in p.text:
        for run in p.runs:
            if old in run.text:
                run.text = run.text.replace(old, new)
                updates.append(f'[{idx}] {reason}')
                return True
        # 跨run
        full = p.text.replace(old, new)
        p.runs[0].text = full
        for run in p.runs[1:]:
            run.text = ''
        updates.append(f'[{idx}] {reason} (跨run)')
        return True
    else:
        print(f'  警告: [{idx}] 未找到 "{old[:50]}"')
        return False

# ═══════════════════════════════════════════════════════════════
# §2.1 跨域目标检测（段落16）
# ═══════════════════════════════════════════════════════════════
print('[1/7] 修正§2.1引用...')
replace_in_para(16, '基于域对抗训练的方法[9-11]', '基于域对抗训练的方法[20-22]', '§2.1 域对抗训练[9-11]→[20-22]')
replace_in_para(16, '基于自训练的方法[12-14]', '基于自训练的方法[23]', '§2.1 自训练[12-14]→[23]')
replace_in_para(16, '基于原型匹配的方法[1-3,15]', '基于原型匹配的方法[29]', '§2.1 原型匹配[1-3,15]→[29]')

# ═══════════════════════════════════════════════════════════════
# §2.2 开放词汇检测与原型方法（段落18）
# ═══════════════════════════════════════════════════════════════
print('[2/7] 修正§2.2引用...')
replace_in_para(18, 'GLIP[17]', 'GLIP[24]', '§2.2 GLIP[17]→[24]')
replace_in_para(18, 'OWLv2[19]', 'OWLv2[3]', '§2.2 OWLv2[19]→[3]')
replace_in_para(18, '原型方法[1-3,15]', '原型方法[29]', '§2.2 原型方法[1-3,15]→[29]')

# ═══════════════════════════════════════════════════════════════
# §2.3 提示学习、特征适配与测试时适配（段落21）
# ═══════════════════════════════════════════════════════════════
print('[3/7] 修正§2.3引用...')
replace_in_para(21, 'CoCoOp[21]', 'CoCoOp[50]', '§2.3 CoCoOp[21]→[50](新增)')
replace_in_para(21, 'CLIP-Adapter[22]', 'CLIP-Adapter[11]', '§2.3 CLIP-Adapter[22]→[11]')
replace_in_para(21, 'Tip-Adapter[23]', 'Tip-Adapter[51]', '§2.3 Tip-Adapter[23]→[51](新增)')
replace_in_para(21, '测试时适配（TTA）方法[24-25]', '测试时适配（TTA）方法[16-19]', '§2.3 TTA[24-25]→[16-19]')

# ═══════════════════════════════════════════════════════════════
# §2.4 候选框重排序（段落23）
# ═══════════════════════════════════════════════════════════════
print('[4/7] 修正§2.4引用...')
replace_in_para(23, '候选框重排序方法[27-29]', '候选框重排序方法[30,32]', '§2.4 重排序[27-29]→[30,32]')

# ═══════════════════════════════════════════════════════════════
# §4.1 数据集（段落40）- 检查OWLv2引用
# ═══════════════════════════════════════════════════════════════
print('[5/7] 修正§4.1引用...')
p40_text = doc.paragraphs[40].text
if 'OWLv2[19]' in p40_text:
    replace_in_para(40, 'OWLv2[19]', 'OWLv2[3]', '§4.1 OWLv2[19]→[3]')
elif 'OWLv2' in p40_text and '[19]' in p40_text:
    print('  §4.1有OWLv2和[19]但不连续，需手动检查')
else:
    print('  §4.1无OWLv2[19]引用')

# ═══════════════════════════════════════════════════════════════
# 补充缺失文献：[50]CoCoOp, [51]Tip-Adapter
# ═══════════════════════════════════════════════════════════════
print('[6/7] 补充缺失文献...')
# 找到最后一篇文献[49]的位置
last_ref_idx = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('[49]'):
        last_ref_idx = i
        break

if last_ref_idx:
    print(f'  最后一篇文献[49]在段落{last_ref_idx}')
    # 在[49]之后插入[50]和[51]
    p49 = doc.paragraphs[last_ref_idx]._element
    from docx.oxml.ns import qn
    from lxml import etree

    # [50] CoCoOp
    new_p50 = etree.SubElement(p49.getparent(), qn('w:p'))
    p49.addnext(new_p50)
    run50 = etree.SubElement(new_p50, qn('w:r'))
    t50 = etree.SubElement(run50, qn('w:t'))
    t50.text = '[50] Zhou K, Yang J, Loy C C, et al. Conditional prompt learning for vision-language models[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2022: 16816-16825.'
    t50.set(qn('xml:space'), 'preserve')
    updates.append('[50] 补充CoCoOp文献')

    # [51] Tip-Adapter
    new_p51 = etree.SubElement(p49.getparent(), qn('w:p'))
    new_p50.addnext(new_p51)
    run51 = etree.SubElement(new_p51, qn('w:r'))
    t51 = etree.SubElement(run51, qn('w:t'))
    t51.text = '[51] Zhang R, Fang W, Jiang Z, et al. Tip-Adapter: Training-free CLIP-Adapter for better vision-language models[C]//European Conference on Computer Vision (ECCV). 2022: 445-462.'
    t51.set(qn('xml:space'), 'preserve')
    updates.append('[51] 补充Tip-Adapter文献')
else:
    print('  未找到[49]文献位置')

# ═══════════════════════════════════════════════════════════════
# 全文检查是否还有明显的引用错位
# ═══════════════════════════════════════════════════════════════
print('[7/7] 全文检查残留引用错位...')
# 检查GLIP[17]、OWLv2[19]、CLIP-Adapter[22]等是否还有残留
problem_patterns = [
    ('GLIP[17]', 'GLIP应为[24]'),
    ('OWLv2[19]', 'OWLv2应为[3]'),
    ('CLIP-Adapter[22]', 'CLIP-Adapter应为[11]'),
    ('Tip-Adapter[23]', 'Tip-Adapter应为[51]'),
    ('CoCoOp[21]', 'CoCoOp应为[50]'),
    ('域对抗训练的方法[9-11]', '域对抗训练应为[20-22]'),
    ('自训练的方法[12-14]', '自训练应为[23]'),
    ('原型匹配的方法[1-3,15]', '原型匹配应为[29]'),
    ('TTA）方法[24-25]', 'TTA应为[16-19]'),
    ('重排序方法[27-29]', '重排序应为[30,32]'),
]
for pattern, desc in problem_patterns:
    found = []
    for i, p in enumerate(doc.paragraphs):
        if pattern in p.text:
            found.append(i)
    if found:
        print(f'  残留: {desc} 在段落{found}')
    else:
        print(f'  OK: {desc} 无残留')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\n引用修正完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
