# -*- coding: utf-8 -*-
r"""fix_v23_p1.py — 修复表编号体系问题（第一部分：文本修改）
1. 表12撞车：段87反向迁移"表12"→"表14"
2. 表13撞车：段102信息使用审计"表13"→"表15"
3. 物理先验因子表：段111"表1 V/M/S/I"→"表16 G/S/I/C"
4. 段46删"表3（显著性检验）"引用
5. 表8/9题注内容修正
6. 表3题注删除（内容不匹配）
"""
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v22.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)

def replace_paragraph_text(p, old_text, new_text):
    if old_text in p.text:
        full_text = p.text.replace(old_text, new_text)
        for run in p.runs:
            run.text = ''
        if p.runs:
            p.runs[0].text = full_text
        else:
            p.add_run(full_text)
        return True
    return False

# ══════════════════════════ 修复1：表12撞车 ══════════════════════════
print('修复1：表12撞车（反向迁移→表14）')
for i, p in enumerate(doc.paragraphs):
    if '表12' in p.text and '反向迁移' in p.text:
        old_text = p.text
        new_text = old_text.replace('表12（§4.5', '表14（§4.5')
        new_text = new_text.replace('表12　', '表14　')
        replace_paragraph_text(p, old_text, new_text)
        print(f'  段落{i}: 表12→表14')
        break

# ══════════════════════════ 修复2：表13撞车 ══════════════════════════
print('\n修复2：表13撞车（信息使用审计→表15）')
for i, p in enumerate(doc.paragraphs):
    if '表13' in p.text and '信息使用审计' in p.text:
        old_text = p.text
        new_text = old_text.replace('表13　', '表15　')
        replace_paragraph_text(p, old_text, new_text)
        print(f'  段落{i}: 表13→表15')
        break

# ══════════════════════════ 修复3：物理先验因子表 ══════════════════════════
print('\n修复3：物理先验因子表（表1 V/M/S/I→表16 G/S/I/C）')
for i, p in enumerate(doc.paragraphs):
    if '表1' in p.text and '物理先验因子' in p.text:
        old_text = p.text
        new_text = old_text.replace('表1　', '表16　')
        new_text = new_text.replace('V视角/M材质/S尺度/I变异', 'G粒度失配/S尺度/I类内变异/C类间混淆')
        replace_paragraph_text(p, old_text, new_text)
        print(f'  段落{i}: 表1→表16, V/M/S/I→G/S/I/C')
        break

# 同时修改物理先验因子表的列头
print('  修改物理先验因子表列头...')
for table in doc.tables:
    if table.rows and len(table.rows[0].cells) >= 5:
        first_row_text = [c.text.strip() for c in table.rows[0].cells]
        if 'V(视角)' in first_row_text or 'V（视角）' in first_row_text:
            # 修改列头
            new_headers = ['类别', 'G(粒度失配)', 'S(尺度)', 'I(类内变异)', 'C(类间混淆)']
            for j, cell in enumerate(table.rows[0].cells):
                if j < len(new_headers):
                    cell.text = new_headers[j]
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(9)
                            run.font.bold = True
            print(f'    列头已修改为: {new_headers}')
            break

# ══════════════════════════ 修复4：段46删"表3（显著性检验）"引用 ══════════════════════════
print('\n修复4：段46删"表3（显著性检验）"引用')
p46 = doc.paragraphs[46]
old_text = p46.text
if '表3' in old_text and '显著性' in old_text:
    new_text = old_text.replace('和表3（显著性检验）', '')
    new_text = new_text.replace('表2（检测器逐类）', '表SOTA-a（主结果逐类）')
    replace_paragraph_text(p46, old_text, new_text)
    print(f'  已删除显著性检验引用')
else:
    print(f'  段46未找到"表3（显著性检验）"，当前内容: {old_text[:100]}')

# ══════════════════════════ 修复5：表8/9题注内容修正 ══════════════════════════
print('\n修复5：表8/9题注内容修正')
for i, p in enumerate(doc.paragraphs):
    # 表8题注：λ扫描（内容正确，但位置需要移动到tbl[5]前面）
    if p.text.strip().startswith('表8') and 'λ扫描' in p.text:
        print(f'  段落{i} 表8题注内容正确，位置待移动')
    # 表9题注：78类扩展（但tbl[5]是λ表，不是78类，需要删除或修改）
    if p.text.strip().startswith('表9') and '78类' in p.text:
        old_text = p.text
        # 78类扩展表不存在，改为λ扫描的表注或删除
        # 实际上表9应该是78类扩展，但表格不存在，所以删除这个题注
        new_text = ''  # 清空
        replace_paragraph_text(p, old_text, new_text)
        print(f'  段落{i} 表9题注（78类扩展）已删除（无对应表格实体）')

# ══════════════════════════ 修复6：表3题注删除 ══════════════════════════
print('\n修复6：表3题注删除（内容不匹配）')
for i, p in enumerate(doc.paragraphs):
    if p.text.strip().startswith('表 3') and 'NWPU/DOTA' in p.text:
        old_text = p.text
        new_text = ''  # 清空，表3编号不再使用
        replace_paragraph_text(p, old_text, new_text)
        print(f'  段落{i} 表3题注已删除')
        break

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('第一部分文本修改完成')
