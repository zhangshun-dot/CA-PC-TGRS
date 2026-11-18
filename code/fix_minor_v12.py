# -*- coding: utf-8 -*-
r"""fix_minor_v12.py — 修复v12小修清单
处理：
  1. 表1真正移到§6.2（删除"排版原因"临时说明）
  2. 全文统一负号（半角- → 全角−，更全面的匹配）
  3. 检查中英文标点混用
"""
from docx import Document
from docx.oxml.ns import qn
from lxml import etree
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v12.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v13.docx'

doc = Document(PAPER_PATH)
updates = []

# ═══════════════════════════════════════════════════════════════
# 1. 表1真正移到§6.2
# ═══════════════════════════════════════════════════════════════
print('[1/3] 表1移到§6.2...')

# 找到表1题注的位置
table1_caption_idx = None
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('表1') and '物理先验因子' in p.text:
        table1_caption_idx = i
        break

print(f'  表1题注在段落{table1_caption_idx}')

# 找到§6.2的位置
section62_idx = None
for i, p in enumerate(doc.paragraphs):
    if '6.2' in p.text and '物理先验' in p.text:
        section62_idx = i
        break

print(f'  §6.2在段落{section62_idx}')

if table1_caption_idx is not None and section62_idx is not None:
    # 获取表1题注段落和其后的表格元素
    caption_elem = doc.paragraphs[table1_caption_idx]._element
    # 表格元素通常在题注之后
    table_elem = caption_elem.getnext()
    # 可能还有空段落
    elements_to_move = [caption_elem]
    if table_elem is not None:
        elements_to_move.append(table_elem)
        # 检查表格后是否有空段落
        next_elem = table_elem.getnext()
        if next_elem is not None and next_elem.tag == qn('w:p'):
            next_p_text = ''.join(t.text or '' for t in next_elem.iter(qn('w:t')))
            if not next_p_text.strip():
                elements_to_move.append(next_elem)

    print(f'  要移动的元素数: {len(elements_to_move)}')

    # 找到§6.2标题后的插入位置
    section62_elem = doc.paragraphs[section62_idx]._element
    # 在§6.2标题后第一段之前插入
    insert_after = section62_elem
    # 先找到§6.2后第一段（跳过可能的空段落）
    next_p = section62_elem.getnext()
    while next_p is not None and next_p.tag == qn('w:p'):
        text = ''.join(t.text or '' for t in next_p.iter(qn('w:t')))
        if text.strip():
            break
        next_p = next_p.getnext()

    if next_p is not None:
        insert_after = next_p.getprevious()
        # 在next_p之前插入
        for elem in elements_to_move:
            # 先从原位置移除
            elem.getparent().remove(elem)
            # 插入到目标位置
            next_p.addprevious(elem)
        updates.append('表1移到§6.2')
        print('  表1已移到§6.2')
    else:
        print('  未找到§6.2后第一段，跳过')

    # 删除§6.2段首的"排版原因"临时说明
    for i, p in enumerate(doc.paragraphs):
        if '因排版原因物理位置在此' in p.text:
            old_text = p.text
            new_text = old_text.replace('类别级物理先验因子赋值见表1（该表位于§4.2后，因排版原因物理位置在此，正文引用以§6.2为准）。', '')
            if p.runs:
                p.runs[0].text = new_text
                for run in p.runs[1:]:
                    run.text = ''
            updates.append('删除表1"排版原因"临时说明')
            print('  已删除"排版原因"临时说明')
            break
else:
    print('  未找到表1或§6.2位置')

# ═══════════════════════════════════════════════════════════════
# 2. 全文统一负号（更全面的匹配）
# ═══════════════════════════════════════════════════════════════
print('[2/3] 全文统一负号...')
neg_count = 0
for i, p in enumerate(doc.paragraphs):
    for run in p.runs:
        text = run.text
        if not text:
            continue
        # 匹配各种模式的半角负号：
        # - 数值前的负号（-0.0316, -0.0005）
        # - 括号中的负号（(-0.0316)）
        # - 逗号后的负号（, -0.0316）
        # - 冒号后的负号（: -0.0316）
        # - 句首的负号
        # 但不匹配：
        # - 连字符（cross-domain, zero-shot, text-to-image）
        # - 日期（2024-01-01）
        # - 范围（1-10, 5-10分钟）
        new_text = text
        # 模式1: 空格/括号/逗号/冒号/句首 + 负号 + 数字
        new_text = re.sub(r'(?<=[\s\(\[,，：:])-(\d)', r'−\1', new_text)
        # 模式2: 句首负号+数字
        new_text = re.sub(r'^-(\d)', r'−\1', new_text)
        # 模式3: 增益表述中的负号（如"增益-0.0316"）
        new_text = re.sub(r'(增益|下降|损害|转负|负增益|为)-(\d)', r'\1−\2', new_text)
        if new_text != text:
            run.text = new_text
            neg_count += 1

updates.append(f'负号统一: {neg_count}处')
print(f'  负号统一: {neg_count}处')

# ═══════════════════════════════════════════════════════════════
# 3. 检查中英文标点混用（报告，不自动修改）
# ═══════════════════════════════════════════════════════════════
print('[3/3] 检查中英文标点混用...')
cn_comma_in_en = 0
en_comma_in_cn = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 检查英文句子中的中文逗号（简单启发式：英文单词后接中文逗号）
    if re.search(r'[a-zA-Z]，', text):
        cn_comma_in_en += 1
    # 检查中文句子中的英文逗号（简单启发式：中文字符后接英文逗号）
    if re.search(r'[\u4e00-\u9fff],', text):
        en_comma_in_cn += 1

print(f'  英文后接中文逗号: {cn_comma_in_en}处（需人工确认）')
print(f'  中文后接英文逗号: {en_comma_in_cn}处（需人工确认）')
updates.append(f'标点检查: 英文后中文逗号{cn_comma_in_en}处, 中文后英文逗号{en_comma_in_cn}处（需人工确认）')

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
doc.save(OUTPUT_PATH)
print(f'\n小修完成！共{len(updates)}处更新：')
for u in updates:
    print(f'  {u}')
print(f'\n保存至: {OUTPUT_PATH}')
print(f'\n仍需手动处理: 图2/图3编号（需检查9张嵌入图片内容）')
