# -*- coding: utf-8 -*-
r"""fix_review_v3.py — 修复一审意见剩余问题
处理：
  1. 核心价值正面回应（段落13后插入物理先验+0.0461）
  2. 负号统一（半角- → 全角−，仅数值中的负号）
  3. 表1物理先验因子表移位说明（在§6.2加引用说明）
"""
from docx import Document
from docx.oxml.ns import qn
from lxml import etree
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v11.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v12.docx'

doc = Document(PAPER_PATH)
updates = []

# ═══════════════════════════════════════════════════════════════
# 1. 核心价值正面回应（段落13后插入）
# ═══════════════════════════════════════════════════════════════
print('[1/3] 核心价值正面回应...')
p13_elem = doc.paragraphs[13]._element
new_p = etree.SubElement(p13_elem.getparent(), qn('w:p'))
p13_elem.addnext(new_p)
run = etree.SubElement(new_p, qn('w:r'))
t = etree.SubElement(run, qn('w:t'))
t.text = '针对zero-shot场景下方法价值的质疑，本文进一步提出物理先验驱动的无目标域标注自适应参数控制：利用类别级物理先验（粒度失配、尺度、类内变异、类间混淆）预测逐类校正需求，直接驱动β_eff(c)=β_base·(1+α·R(c))的自适应超参数，在DIOR→COCO六类上取得+0.0461增益（固定参数仅+0.0166，相对提升177%），6/6类正增益，并在VOC→COCO 20类扩展中验证普适性。这一机制将"目标域调参"的主配置升级为"物理先验驱动的无标注自适应控制"，正面回应了zero-shot场景下的方法价值问题。'
t.set(qn('xml:space'), 'preserve')
updates.append('[13后] 核心价值正面回应：物理先验驱动+0.0461')

# ═══════════════════════════════════════════════════════════════
# 2. 负号统一（半角- → 全角−，仅数值中的负号）
# ═══════════════════════════════════════════════════════════════
print('[2/3] 负号统一...')
neg_count = 0
for i, p in enumerate(doc.paragraphs):
    for run in p.runs:
        # 匹配数值前的半角负号（如 -0.0316, -0.0005）
        # 但不匹配连字符（如 cross-domain, zero-shot）
        text = run.text
        new_text = re.sub(r'(?<=\s)-(\d)', r'−\1', text)
        new_text = re.sub(r'^-(\d)', r'−\1', new_text)
        if new_text != text:
            run.text = new_text
            neg_count += 1
updates.append(f'负号统一: {neg_count}处')

# ═══════════════════════════════════════════════════════════════
# 3. 表1物理先验因子表移位说明（在§6.2加引用说明）
# ═══════════════════════════════════════════════════════════════
print('[3/3] 表1移位说明...')
# 找到§6.2的位置
for i, p in enumerate(doc.paragraphs):
    if '6.2' in p.text and '物理先验' in p.text:
        print(f'  §6.2在段落{i}')
        # 在§6.2标题后第一段加表1引用说明
        if i+1 < len(doc.paragraphs):
            p_next = doc.paragraphs[i+1]
            if '表1' not in p_next.text:
                old_text = p_next.text
                if old_text and len(old_text) > 10:
                    # 在段首加表1引用
                    new_text = '类别级物理先验因子赋值见表1（该表位于§4.2后，因排版原因物理位置在此，正文引用以§6.2为准）。' + old_text
                    if p_next.runs:
                        p_next.runs[0].text = new_text
                        for run in p_next.runs[1:]:
                            run.text = ''
                    updates.append(f'[{i+1}] §6.2加表1引用说明')
                    print(f'  已在段落{i+1}加表1引用说明')
        break

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\n修复完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
