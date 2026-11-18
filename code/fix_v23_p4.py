# -*- coding: utf-8 -*-
r"""fix_v23_p4.py — 删除重复题注
1. tbl[4]前面删除错误的"表7 负原型权重β扫描实验"
2. tbl[7]前面删除一个重复的"表10 跨源域类别映射表"
3. tbl[10]前面删除一个重复的"表13 检测器谱系实验"
"""
from docx import Document
from docx.oxml.ns import qn

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'
OUTPUT_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v23.docx'

doc = Document(PAPER_PATH)
body = doc.element.body

def get_paragraph_text(elem):
    return ''.join([t.text or '' for t in elem.findall('.//' + qn('w:t'))])

# 收集所有表格元素
tables = []
for child in body:
    if child.tag == qn('w:tbl'):
        tables.append(child)

print(f'共找到 {len(tables)} 张表格')

# ══════════════════════════ 修复1：tbl[4]删除错误题注 ══════════════════════════
print('\n修复1：tbl[4]删除错误的"表7 负原型权重β扫描实验"')
tbl4 = tables[4]
idx4 = list(body).index(tbl4)
# 检查前面的段落
for j in range(idx4-1, max(idx4-5, 0), -1):
    if body[j].tag == qn('w:p'):
        text = get_paragraph_text(body[j])
        if '负原型权重β扫描' in text or ('表7' in text and '负原型' in text):
            print(f'  删除段落{j}: {text[:60]}')
            body.remove(body[j])
            break
        elif text.strip() == '':
            continue
        else:
            break

# ══════════════════════════ 修复2：tbl[7]删除重复题注 ══════════════════════════
print('\n修复2：tbl[7]删除一个重复的"表10 跨源域类别映射表"')
# 重新获取表格列表（因为删除了元素）
tables = []
for child in body:
    if child.tag == qn('w:tbl'):
        tables.append(child)
tbl7 = tables[7]
idx7 = list(body).index(tbl7)
found = False
for j in range(idx7-1, max(idx7-5, 0), -1):
    if body[j].tag == qn('w:p'):
        text = get_paragraph_text(body[j])
        if '表10' in text and '跨源域类别映射' in text:
            if not found:
                found = True
                print(f'  保留段落{j}: {text[:60]}')
            else:
                print(f'  删除段落{j}: {text[:60]}')
                body.remove(body[j])
                break
        elif text.strip() == '':
            continue
        else:
            break

# ══════════════════════════ 修复3：tbl[10]删除重复题注 ══════════════════════════
print('\n修复3：tbl[10]删除一个重复的"表13 检测器谱系实验"')
tables = []
for child in body:
    if child.tag == qn('w:tbl'):
        tables.append(child)
tbl10 = tables[10]
idx10 = list(body).index(tbl10)
found = False
for j in range(idx10-1, max(idx10-5, 0), -1):
    if body[j].tag == qn('w:p'):
        text = get_paragraph_text(body[j])
        if '表13' in text and '检测器谱系' in text:
            if not found:
                found = True
                print(f'  保留段落{j}: {text[:60]}')
            else:
                print(f'  删除段落{j}: {text[:60]}')
                body.remove(body[j])
                break
        elif text.strip() == '':
            continue
        else:
            break

doc.save(OUTPUT_PATH)
print(f'\n论文已保存: {OUTPUT_PATH}')
print('重复题注删除完成')
