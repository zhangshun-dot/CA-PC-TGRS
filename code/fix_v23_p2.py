# -*- coding: utf-8 -*-
r"""fix_v23_p2.py — 修复表题位置和缺题注问题
1. tbl[4]题注"表8 λ扫描"→"表7 β扫描"
2. tbl[5]补题注"表8 λ扫描"
3. tbl[6]补题注"表4 分层失效诊断"
4. tbl[7]题注"表11 三源域复现"→"表10 跨源域类别映射"
5. tbl[8]补题注"表11 三源域跨源复现"
6. tbl[10]题注"表14 多目标域"→"表13 检测器谱系"
7. tbl[11]补题注"表14 多目标域结果对比"
8. tbl[15]题注"表15 LODO"→"表17 LODO"
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

def create_caption_paragraph(text):
    """创建题注段落"""
    p = body.makeelement(qn('w:p'), {})
    r = p.makeelement(qn('w:r'), {})
    p.append(r)
    rPr = r.makeelement(qn('w:rPr'), {})
    r.insert(0, rPr)
    sz = rPr.makeelement(qn('w:sz'), {qn('w:val'): '20'})
    rPr.append(sz)
    szCs = rPr.makeelement(qn('w:szCs'), {qn('w:val'): '20'})
    rPr.append(szCs)
    t = r.makeelement(qn('w:t'), {})
    t.text = text
    t.set(qn('xml:space'), 'preserve')
    r.append(t)
    return p

def get_table_index(body, table_elem):
    """获取表格在body中的索引"""
    for i, child in enumerate(body):
        if child is table_elem:
            return i
    return -1

# 收集所有表格元素
tables = []
for child in body:
    if child.tag == qn('w:tbl'):
        tables.append(child)

print(f'共找到 {len(tables)} 张表格')

# ══════════════════════════ 修复1：tbl[4]题注内容 ══════════════════════════
print('\n修复1：tbl[4]题注"表8 λ扫描"→"表7 β扫描"')
tbl4 = tables[4]
idx4 = get_table_index(body, tbl4)
prev_elem = body[idx4-1] if idx4 > 0 else None
if prev_elem is not None and prev_elem.tag == qn('w:p'):
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表8' in prev_text and 'λ扫描' in prev_text:
        # 修改文本
        for t in prev_elem.findall('.//' + qn('w:t')):
            if t.text:
                t.text = t.text.replace('表8', '表7').replace('λ扫描', 'β扫描').replace('N=500逐图AP50', 'N=100口径')
        print(f'  已修改: {prev_text[:40]} → 表7 β扫描')
    else:
        print(f'  题注内容不匹配: {prev_text[:40]}')

# ══════════════════════════ 修复2：tbl[5]补题注 ══════════════════════════
print('\n修复2：tbl[5]补题注"表8 λ扫描"')
tbl5 = tables[5]
idx5 = get_table_index(body, tbl5)
prev_elem = body[idx5-1] if idx5 > 0 else None
if prev_elem is not None:
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表8' not in prev_text:
        caption = create_caption_paragraph('表8　负原型权重λ扫描实验（N=500逐图AP50）')
        tbl5.addprevious(caption)
        print(f'  已插入题注: 表8 λ扫描')
    else:
        print(f'  已有题注: {prev_text[:40]}')

# ══════════════════════════ 修复3：tbl[6]补题注 ══════════════════════════
print('\n修复3：tbl[6]补题注"表4 分层失效诊断"')
tbl6 = tables[6]
idx6 = get_table_index(body, tbl6)
prev_elem = body[idx6-1] if idx6 > 0 else None
if prev_elem is not None:
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表4' not in prev_text and '分层' not in prev_text:
        caption = create_caption_paragraph('表4　分层失效诊断结果（排序AUC与框池召回分层报告；n为每层类别数，B2为β=0源域原型，T4为纯文本原型，oracle β为逐类最优外推强度）')
        tbl6.addprevious(caption)
        print(f'  已插入题注: 表4 分层失效诊断')
    else:
        print(f'  已有题注: {prev_text[:40]}')

# ══════════════════════════ 修复4：tbl[7]题注内容 ══════════════════════════
print('\n修复4：tbl[7]题注"表11 三源域复现"→"表10 跨源域类别映射"')
tbl7 = tables[7]
idx7 = get_table_index(body, tbl7)
prev_elem = body[idx7-1] if idx7 > 0 else None
if prev_elem is not None and prev_elem.tag == qn('w:p'):
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表11' in prev_text and '三源域' in prev_text:
        for t in prev_elem.findall('.//' + qn('w:t')):
            if t.text:
                t.text = t.text.replace('表11', '表10').replace('三源域跨源复现结果（逐类增益与均值）', '跨源域类别映射表（DIOR/NWPU/DOTA/VisDrone→COCO）')
        print(f'  已修改: 表11 三源域复现 → 表10 类别映射')
    else:
        print(f'  题注内容不匹配: {prev_text[:40]}')

# ══════════════════════════ 修复5：tbl[8]补题注 ══════════════════════════
print('\n修复5：tbl[8]补题注"表11 三源域跨源复现"')
tbl8 = tables[8]
idx8 = get_table_index(body, tbl8)
prev_elem = body[idx8-1] if idx8 > 0 else None
if prev_elem is not None:
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表11' not in prev_text:
        caption = create_caption_paragraph('表11　三源域跨源复现结果（DIOR/NWPU/DOTA→COCO，逐类增益与均值；VisDrone为4类不同粒度，单列于正文）')
        tbl8.addprevious(caption)
        print(f'  已插入题注: 表11 三源域复现')
    else:
        print(f'  已有题注: {prev_text[:40]}')

# ══════════════════════════ 修复6：tbl[10]题注内容 ══════════════════════════
print('\n修复6：tbl[10]题注"表14 多目标域"→"表13 检测器谱系"')
tbl10 = tables[10]
idx10 = get_table_index(body, tbl10)
prev_elem = body[idx10-1] if idx10 > 0 else None
if prev_elem is not None and prev_elem.tag == qn('w:p'):
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表14' in prev_text and '多目标域' in prev_text:
        for t in prev_elem.findall('.//' + qn('w:t')):
            if t.text:
                t.text = t.text.replace('表14', '表13').replace('多目标域结果对比（COCO/VOC/NWPU，21类）', '检测器谱系实验（校正器边界：弱→强检测器）')
        print(f'  已修改: 表14 多目标域 → 表13 检测器谱系')
    else:
        print(f'  题注内容不匹配: {prev_text[:40]}')

# ══════════════════════════ 修复7：tbl[11]补题注 ══════════════════════════
print('\n修复7：tbl[11]补题注"表14 多目标域结果对比"')
tbl11 = tables[11]
idx11 = get_table_index(body, tbl11)
prev_elem = body[idx11-1] if idx11 > 0 else None
if prev_elem is not None:
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表14' not in prev_text:
        caption = create_caption_paragraph('表14　多目标域结果对比（COCO/VOC/NWPU，21类；按目标域信息预算分组）')
        tbl11.addprevious(caption)
        print(f'  已插入题注: 表14 多目标域对比')
    else:
        print(f'  已有题注: {prev_text[:40]}')

# ══════════════════════════ 修复8：tbl[15]题注编号撞车 ══════════════════════════
print('\n修复8：tbl[15]题注"表15 LODO"→"表17 LODO"')
tbl15 = tables[15]
idx15 = get_table_index(body, tbl15)
prev_elem = body[idx15-1] if idx15 > 0 else None
if prev_elem is not None and prev_elem.tag == qn('w:p'):
    prev_text = ''.join([t.text or '' for t in prev_elem.findall('.//' + qn('w:t'))])
    if '表15' in prev_text and 'LODO' in prev_text:
        for t in prev_elem.findall('.//' + qn('w:t')):
            if t.text:
                t.text = t.text.replace('表15', '表17')
        print(f'  已修改: 表15 LODO → 表17 LODO')
    else:
        print(f'  题注内容不匹配: {prev_text[:40]}')

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('第二部分表题位置修复完成')
