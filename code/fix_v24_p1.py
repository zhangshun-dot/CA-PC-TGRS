# -*- coding: utf-8 -*-
r"""fix_v24_p1.py — 修复v23的6项P1问题
1. 表4题注移到表5题注之前
2. 段68交叉引用修正（第4.7节→第4.3节）
3. 删除补充表S3/S4的外部文档引用
4. 补充表S3标题加"（完整版）"
5. 统一负号（- → −）
"""
from docx import Document
from docx.oxml.ns import qn
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v24.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

def get_paragraph_text(elem):
    return ''.join([t.text or '' for t in elem.findall('.//' + qn('w:t'))])

def set_paragraph_text(elem, new_text):
    """设置段落文本（保留第一个run的格式）"""
    runs = elem.findall('.//' + qn('w:r'))
    if runs:
        # 清空所有run的文本
        for r in runs:
            for t in r.findall('.//' + qn('w:t')):
                t.text = ''
        # 在第一个run中设置新文本
        first_t = runs[0].find('.//' + qn('w:t'))
        if first_t is not None:
            first_t.text = new_text
            first_t.set(qn('xml:space'), 'preserve')
        else:
            # 创建新的t元素
            t = runs[0].makeelement(qn('w:t'), {})
            t.text = new_text
            t.set(qn('xml:space'), 'preserve')
            runs[0].append(t)
    else:
        # 创建新的run
        r = elem.makeelement(qn('w:r'), {})
        t = r.makeelement(qn('w:t'), {})
        t.text = new_text
        t.set(qn('xml:space'), 'preserve')
        r.append(t)
        elem.append(r)

# ══════════════════════════ 修复1：表4题注移到表5之前 ══════════════════════════
print('修复1：表4题注移到表5题注之前')

# 找到表4题注和表5题注
table4_caption = None
table5_caption = None
for i, child in enumerate(body):
    if child.tag == qn('w:p'):
        text = get_paragraph_text(child)
        if text.strip().startswith('表4') and '分层失效诊断' in text:
            table4_caption = child
            table4_idx = i
            print(f'  找到表4题注在位置{i}')
        elif text.strip().startswith('表5') and 'top-k' in text:
            table5_caption = child
            table5_idx = i
            print(f'  找到表5题注在位置{i}')

if table4_caption is not None and table5_caption is not None:
    # 移除表4题注
    body.remove(table4_caption)
    # 插入到表5题注之前
    table5_caption.addprevious(table4_caption)
    print(f'  已将表4题注移到表5题注之前')

# ══════════════════════════ 修复2：段68交叉引用修正 ══════════════════════════
print('\n修复2：段68交叉引用修正（第4.7节→第4.3节）')
for i, p in enumerate(doc.paragraphs):
    if '第 4.7 节表 4' in p.text or '第4.7节表4' in p.text:
        old_text = p.text
        new_text = old_text.replace('第 4.7 节表 4', '第 4.3 节表 4')
        new_text = new_text.replace('第4.7节表4', '第4.3节表4')
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 第4.7节表4 → 第4.3节表4')
        break

# ══════════════════════════ 修复3：删除补充表S3/S4的外部文档引用 ══════════════════════════
print('\n修复3：删除补充表S3/S4的外部文档引用')
for i, p in enumerate(doc.paragraphs):
    text = p.text
    if 'cross_domain_paper_framework.md' in text:
        old_text = text
        # 删除外部引用部分
        new_text = re.sub(r'[（(]?来源[：:]?\s*cross_domain_paper_framework\.md\s*§?[\d.]+[）)]?', '', old_text)
        new_text = re.sub(r'[（(]?对应\s*cross_domain_paper_framework\.md\s*§?[\d.]+\s*补全[）)]?', '', new_text)
        new_text = new_text.strip()
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        print(f'  段落{i}: 已删除外部引用')
        print(f'    原文: {old_text[:80]}')
        print(f'    新文: {new_text[:80]}')

# ══════════════════════════ 修复4：补充表S3标题加"（完整版）" ══════════════════════════
print('\n修复4：补充表S3标题加"（完整版）"')
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('补充表S3') and '信息使用审计' in p.text:
        old_text = p.text
        if '完整版' not in old_text:
            new_text = old_text.replace('信息使用审计', '信息使用审计（完整版，含主表14之外的扩展列）')
            for run in p.runs:
                run.text = ''
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            print(f'  段落{i}: 已加"（完整版）"')
        break

# ══════════════════════════ 修复5：统一负号 ══════════════════════════
print('\n修复5：统一负号（- → −）')
count = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 匹配数字前的负号（如 -0.05, −0.05）
    # 只替换数字前的连字符为真正的负号
    new_text = re.sub(r'(?<=\s)-(?=\d)', '−', text)
    new_text = re.sub(r'(?<=[（(])-(?=\d)', '−', new_text)
    new_text = re.sub(r'(?<=[，,])-(?=\d)', '−', new_text)
    if new_text != text:
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = new_text
        else:
            p.add_run(new_text)
        count += 1
print(f'  共修改{count}个段落的负号')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('v23 P1问题修复完成')
