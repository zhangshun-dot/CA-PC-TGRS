# -*- coding: utf-8 -*-
r"""self_check_v8.py — 论文v8全面自查
检查：数值一致性、术语一致性、逻辑矛盾、引用正确性、格式残留
"""
from docx import Document
import re

PAPER_PATH = r'C:\Users\张三\Desktop\论文初稿\acspc_输出_改稿v8.docx'
doc = Document(PAPER_PATH)

issues = []
warnings = []
info = []

def add_issue(level, location, desc):
    if level == 'ERROR':
        issues.append(f'🔴 [{location}] {desc}')
    elif level == 'WARN':
        warnings.append(f'🟡 [{location}] {desc}')
    else:
        info.append(f'🟢 [{location}] {desc}')

print('='*60)
print('A-CSPC论文 v8 全面自查报告')
print('='*60)

# ═══════════════════════════════════════════════════════════════
# 1. 数值一致性检查
# ═══════════════════════════════════════════════════════════════
print('\n【1/6】数值一致性检查')

# 主结果+0.0646
main_result_count = 0
main_result_locs = []
for i, p in enumerate(doc.paragraphs):
    if '+0.0646' in p.text:
        main_result_count += 1
        main_result_locs.append(i)
print(f'  主结果+0.0646出现{main_result_count}次: 段落{main_result_locs}')

# 检查是否有不一致的主结果（如+0.065、0.064等）
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 查找类似+0.06xx的数值，但排除+0.0646
    matches = re.findall(r'\+0\.06\d{2,3}', text)
    for m in matches:
        if m != '+0.0646' and m != '+0.0652' and m != '+0.0668' and m != '+0.0689':
            add_issue('WARN', i, f'发现疑似主结果不一致数值: {m}')

# 消融结果检查
ablation_values = {
    '+0.0652': '完整A-CSPC消融',
    '−0.0005': '去外推',
    '+0.0239': '纯外推',
    '+0.0413': '负原型条件增量',
    '0.4197': 'EC对称外推',
    '0.4377': '源域负原型',
}
for val, name in ablation_values.items():
    count = sum(1 for p in doc.paragraphs if val in p.text)
    print(f'  {name} {val}: {count}次')

# 多检测器结果
detector_values = {
    '+0.0790': 'GDINO-base',
    '+0.0063': 'YOLO-World',
    '−0.0316': 'OWLv2',
    '0.3707': 'GDINO-tiny基线',
    '0.4037': 'GDINO-base基线',
    '0.6800': 'OWLv2基线',
}
for val, name in detector_values.items():
    count = sum(1 for p in doc.paragraphs if val in p.text)
    print(f'  {name} {val}: {count}次')

# 物理先验结果
physics_values = {
    '+0.699': '全数据ρ',
    '+0.555': '留出ρ',
    '+0.686': 'LOSO ρ',
    '+0.0461': '自动化先验增益',
    '+0.0478': '人工先验增益',
    '+0.0166': '固定参数增益',
}
for val, name in physics_values.items():
    count = sum(1 for p in doc.paragraphs if val in p.text)
    print(f'  {name} {val}: {count}次')

# ═══════════════════════════════════════════════════════════════
# 2. 术语一致性检查
# ═══════════════════════════════════════════════════════════════
print('\n【2/6】术语一致性检查')

# 域对vs类
domain_pair_count = sum(1 for p in doc.paragraphs if '域对' in p.text)
class_count = sum(1 for p in doc.paragraphs if '19类' in p.text or '19 类' in p.text)
print(f'  "域对"出现{domain_pair_count}次')
print(f'  "19类"出现{class_count}次')
if class_count > 0:
    add_issue('ERROR', '全文', '仍有"19类"表述，应统一为"19个域对"')

# A-CSPC命名一致性
acspc_count = sum(1 for p in doc.paragraphs if 'A-CSPC' in p.text)
acs_count = sum(1 for p in doc.paragraphs if 'ACSPC' in p.text and 'A-CSPC' not in p.text)
print(f'  "A-CSPC"出现{acspc_count}次')
if acs_count > 0:
    add_issue('ERROR', '全文', f'发现{acs_count}处"ACSPC"（缺少连字符），应统一为"A-CSPC"')

# 自动化赋值vs更大规模验证
auto_assign_count = sum(1 for i, p in enumerate(doc.paragraphs) if '自动化赋值' in p.text and i != 122)
print(f'  "自动化赋值"（排除§6.5.1正确表述）出现{auto_assign_count}次')
if auto_assign_count > 0:
    for i, p in enumerate(doc.paragraphs):
        if '自动化赋值' in p.text and i != 122:
            add_issue('ERROR', i, '残留"自动化赋值"，应改为"更大规模验证"')

# ═══════════════════════════════════════════════════════════════
# 3. 逻辑矛盾检查
# ═══════════════════════════════════════════════════════════════
print('\n【3/6】逻辑矛盾检查')

# 检查是否同时说"已实现自动化"和"未来工作包括自动化"
has_automated = any('可以完全自动化' in p.text for p in doc.paragraphs)
has_future_auto = any('未来工作' in p.text and '自动化赋值' in p.text for p in doc.paragraphs)
print(f'  声称"可以完全自动化": {has_automated}')
print(f'  未来工作包括"自动化赋值": {has_future_auto}')
if has_future_auto:
    add_issue('ERROR', '结论', '逻辑矛盾：已声称可完全自动化，但未来工作仍包括自动化赋值')

# 检查主结果是否同时出现+0.0646和其他值
has_0646 = any('+0.0646' in p.text for p in doc.paragraphs)
has_0714 = any('+0.0714' in p.text for p in doc.paragraphs)
print(f'  主结果+0.0646: {has_0646}')
print(f'  C_fixed结果+0.0714: {has_0714}')
if has_0714:
    add_issue('WARN', '全文', '发现C_fixed结果+0.0714，确认是否应保留（已决定用原方法）')

# ═══════════════════════════════════════════════════════════════
# 4. 引用正确性检查
# ═══════════════════════════════════════════════════════════════
print('\n【4/6】引用正确性检查')

# 检查表引用
table_refs = re.findall(r'表\s?\d+', ' '.join(p.text for p in doc.paragraphs))
from collections import Counter
table_ref_counts = Counter(table_refs)
print(f'  表引用统计: {dict(table_ref_counts)}')

# 检查图引用
fig_refs = re.findall(r'图\s?\d+', ' '.join(p.text for p in doc.paragraphs))
fig_ref_counts = Counter(fig_refs)
print(f'  图引用统计: {dict(fig_ref_counts)}')

# 检查"见表1"是否正确（表1应该是物理先验因子表，不是主结果）
for i, p in enumerate(doc.paragraphs):
    if '见表1' in p.text or '见表 1' in p.text:
        if '主结果' in p.text or '逐类' in p.text:
            add_issue('ERROR', i, '主结果引用"见表1"错误，表1是物理先验因子表')

# ═══════════════════════════════════════════════════════════════
# 5. 格式残留检查
# ═══════════════════════════════════════════════════════════════
print('\n【5/6】格式残留检查')

# 检查半角负号（数值中）
half_minus_count = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    matches = re.findall(r'(?<![a-zA-Z])-(\d+\.\d+)', text)
    if matches:
        half_minus_count += 1
        add_issue('WARN', i, f'发现半角负号数值: {matches}')
print(f'  半角负号数值段落数: {half_minus_count}')

# 检查中英文标点混用
mixed_punct = 0
for i, p in enumerate(doc.paragraphs):
    text = p.text
    # 检查中文后面跟半角逗号或句号
    if re.search(r'[\u4e00-\u9fa5],', text) or re.search(r'[\u4e00-\u9fa5]\.', text):
        mixed_punct += 1
        if mixed_punct <= 3:
            add_issue('WARN', i, '中英文标点混用（中文后跟半角标点）')
print(f'  中英文标点混用段落数: {mixed_punct}')

# 检查残留的编辑标记
edit_markers = ['原稿', '修改前', 'TODO', 'FIXME', '待补充', '需说明']
for marker in edit_markers:
    count = sum(1 for p in doc.paragraphs if marker in p.text)
    if count > 0:
        add_issue('WARN', '全文', f'发现编辑标记"{marker}"{count}处')

# ═══════════════════════════════════════════════════════════════
# 6. 关键段落内容检查
# ═══════════════════════════════════════════════════════════════
print('\n【6/6】关键段落内容检查')

# §4.1类别列表
p40 = doc.paragraphs[40].text
if '3个映射源类' in p40 and '6个目标类' in p40:
    print('  ✓ §4.1类别列表：正确（3源类→6目标类）')
else:
    add_issue('ERROR', 40, '§4.1类别列表不正确')

# §4.3覆盖率
p52 = doc.paragraphs[52].text
if 'N=100预过滤口径' in p52:
    print('  ✓ §4.3覆盖率口径：已标注')
else:
    add_issue('ERROR', 52, '§4.3覆盖率未标注口径')

# §6.5.1先验归属
p122 = doc.paragraphs[122].text
if '自动化赋值的物理先验' in p122:
    print('  ✓ §6.5.1先验归属：已明确')
else:
    add_issue('ERROR', 122, '§6.5.1未明确先验归属')

# §7局限性
p134 = doc.paragraphs[134].text
if '更大规模验证' in p134:
    print('  ✓ §7局限性：已改为更大规模验证')
else:
    add_issue('ERROR', 134, '§7局限性仍为自动化赋值')

# §8结论
p136 = doc.paragraphs[136].text
if '更大规模验证' in p136 and '自动化赋值' not in p136:
    print('  ✓ §8结论：已改为更大规模验证')
else:
    add_issue('ERROR', 136, '§8结论仍为自动化赋值')

# ═══════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('自查汇总')
print('='*60)
print(f'🔴 错误（必须修）: {len(issues)}')
for issue in issues:
    print(f'  {issue}')
print(f'\n🟡 警告（建议修）: {len(warnings)}')
for warn in warnings[:10]:
    print(f'  {warn}')
if len(warnings) > 10:
    print(f'  ... 还有{len(warnings)-10}条')
print(f'\n🟢 信息: {len(info)}')
for inf in info[:5]:
    print(f'  {inf}')

print('\n' + '='*60)
if len(issues) == 0:
    print('结论：无硬伤，可以投稿')
else:
    print(f'结论：发现{len(issues)}个硬伤，需修复后再投稿')
print('='*60)
